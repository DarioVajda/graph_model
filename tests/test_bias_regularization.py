"""
Trainer/model-level tests for the TODO_reg regularization knobs:

  * ``GraphTrainerV2.create_optimizer``'s shape-based weight-decay split for
    graph-bias params (the ``bias_weight_decay`` knob): magnetic matrices decay,
    the SPD lookup table (``_no_weight_decay``) and all 1-D gains/offsets don't,
    and the backbone split stays byte-identical to HF's name-based rule.
  * The no-regression guarantee: the default ``bias_weight_decay=0.0``
    reproduces the historical behavior (no decay anywhere on the bias path).
  * Gradient parity of ``checkpoint_graph_bias=True`` vs ``False`` with the new
    bias dropouts active (``preserve_rng_state`` replays identical masks).

Run with:  pytest tests/test_bias_regularization.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))   # sibling test modules (helpers)

import types

import torch
from transformers import TrainingArguments

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import GraphAttentionBias, MagneticBias, SPDBias, LaplacianBias, RWSEBias
from src.utils import GraphTrainerV2
from src.utils.text_graph_trainer_v2 import RegOnsetCallback
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from test_modeling_gtlm_llama_v2 import _BASE, _make_items

_BIAS = dict(spd=True, max_spd=8, laplacian=True, rwse=True,
             magnetic=True, magnetic_dim=8)


def _tiny_model(**overrides):
    torch.manual_seed(0)
    cfg = GTLMLlamaConfig(k_hop=0, graph_attn_impl="eager", **_BIAS, **_BASE, **overrides)
    return GTLMLlamaForCausalLM(cfg)


def _param_groups(model, tmp_path, **trainer_kwargs):
    args = TrainingArguments(output_dir=str(tmp_path), weight_decay=0.1,
                             learning_rate=1e-4, report_to=[])
    trainer = GraphTrainerV2(model=model, args=args,
                             active_params=["graph_bias"], bias_lr=5e-3,
                             **trainer_kwargs)
    trainer.create_optimizer()
    by_param = {}
    for g in trainer.optimizer.param_groups:
        for p in g["params"]:
            by_param[id(p)] = (g["weight_decay"], g.get("is_bias"), g["lr"])
    return trainer, by_param


def _bias_module(model, cls):
    gb = model.model.layers[0].self_attn.graph_bias
    return next(m for m in gb.bias_modules if type(m) is cls)


def test_decay_grouping_with_bias_weight_decay(tmp_path):
    model = _tiny_model()
    trainer, by_param = _param_groups(model, tmp_path, bias_weight_decay=0.1)

    mag = _bias_module(model, MagneticBias)
    # Magnetic matrices — the suspected fingerprint pathway — get the decay.
    for t in (mag.lambda_lin.weight, mag.deep_set[0].weight,
              mag.proj[0].weight, mag.proj[2].weight):
        assert by_param[id(t)] == (0.1, True, 5e-3)
    # All magnetic .bias vectors are 1-D -> exempt.
    for t in (mag.lambda_lin.bias, mag.deep_set[0].bias,
              mag.proj[0].bias, mag.proj[2].bias):
        assert by_param[id(t)] == (0.0, True, 5e-3)

    # SPD table: 2-D by shape but tagged _no_weight_decay -> exempt.
    spd = _bias_module(model, SPDBias)
    assert spd.weights._no_weight_decay
    assert by_param[id(spd.weights)] == (0.0, True, 5e-3)

    # 1-D per-head gains (Laplacian / RWSE) -> exempt by the ndim rule.
    for cls in (LaplacianBias, RWSEBias):
        assert by_param[id(_bias_module(model, cls).weights)] == (0.0, True, 5e-3)

    # Backbone params: byte-identical to HF's name-based split, base lr.
    decay_names = set(trainer.get_decay_parameter_names(model))
    for n, p in model.named_parameters():
        if "graph_bias" in n:
            continue
        wd, is_bias, lr = by_param[id(p)]
        assert (is_bias, lr) == (False, 1e-4), n
        assert wd == (0.1 if n in decay_names else 0.0), n


def test_default_reproduces_historical_no_decay(tmp_path):
    """bias_weight_decay defaults to 0.0: every graph-bias param keeps the
    historical (accidental) zero decay, so existing runs stay comparable."""
    model = _tiny_model()
    _, by_param = _param_groups(model, tmp_path)   # default bias_weight_decay
    for n, p in model.named_parameters():
        if "graph_bias" in n:
            wd, is_bias, lr = by_param[id(p)]
            assert (wd, is_bias, lr) == (0.0, True, 5e-3), n


def test_checkpoint_dropout_grad_parity():
    """checkpoint_graph_bias recompute must replay the SAME dropout masks
    (preserve_rng_state), so grads match the uncheckpointed path exactly."""
    drop = dict(magnetic_eigvec_dropout=0.3, magnetic_mlp_dropout=0.3,
                bias_droppath=0.3)
    a = _tiny_model(checkpoint_graph_bias=False, **drop)
    b = _tiny_model(checkpoint_graph_bias=True, **drop)
    b.load_state_dict(a.state_dict())

    batch = GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in _make_items()])
    losses = {}
    for tag, m in (("plain", a), ("ckpt", b)):
        m.train()
        torch.manual_seed(42)
        loss = m(**batch).loss
        loss.backward()
        losses[tag] = loss.item()
    assert abs(losses["plain"] - losses["ckpt"]) < 1e-6

    grads_a = {n: p.grad for n, p in a.named_parameters() if p.grad is not None}
    grads_b = {n: p.grad for n, p in b.named_parameters() if p.grad is not None}
    assert set(grads_a) == set(grads_b)
    assert any("graph_bias" in n and g.abs().sum() > 0 for n, g in grads_a.items()), \
        "graph-bias params received no gradient — test would be vacuous"
    for n, g in grads_a.items():
        assert torch.allclose(g, grads_b[n], atol=1e-6), f"grad mismatch: {n}"


def _collect_keep_masks(model, batch):
    """Run one forward, returning the magnetic_keep kwarg each layer's
    MagneticBias received (via forward pre-hooks)."""
    seen = []
    handles = [m.register_forward_pre_hook(
                   lambda mod, args, kwargs: seen.append(kwargs.get("magnetic_keep")),
                   with_kwargs=True)
               for m in model.modules() if type(m) is MagneticBias]
    try:
        model(**batch)
    finally:
        for h in handles:
            h.remove()
    return seen


def test_shared_eigvec_mask_is_one_tensor_across_layers():
    model = _tiny_model(magnetic_eigvec_dropout=0.5, magnetic_eigvec_shared_mask=True)
    batch = GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in _make_items()])

    model.train()
    masks = _collect_keep_masks(model, batch)
    assert len(masks) == _BASE["num_hidden_layers"]
    assert all(m is masks[0] for m in masks), "all layers must share ONE mask object"
    assert masks[0] is not None and masks[0].dtype == torch.bool

    model.eval()
    masks = _collect_keep_masks(model, batch)
    assert all(m is None for m in masks), "no mask may be sampled in eval mode"


def test_per_layer_masks_when_shared_flag_off():
    model = _tiny_model(magnetic_eigvec_dropout=0.5)   # shared_mask defaults False
    batch = GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in _make_items()])
    model.train()
    masks = _collect_keep_masks(model, batch)
    assert all(m is None for m in masks), "without the flag each layer samples its own"


def test_reg_onset_callback_flips_rates_at_threshold():
    drop = dict(magnetic_eigvec_dropout=0.1, magnetic_mlp_dropout=0.2,
                bias_droppath=0.3, bias_dropout=0.4)
    model = _tiny_model(**drop)
    cb = RegOnsetCallback(model, onset_frac=0.5)
    state = types.SimpleNamespace(max_steps=100, global_step=0)

    mag = _bias_module(model, MagneticBias)
    gb = model.model.layers[0].self_attn.graph_bias

    cb.on_train_begin(None, state, None)
    assert (mag.eigvec_dropout, mag.mlp_dropout, gb.droppath, gb.bias_dropout) \
        == (0.0, 0.0, 0.0, 0.0), "pre-onset: every rate forced to 0"

    state.global_step = 49
    cb.on_step_begin(None, state, None)
    assert mag.eigvec_dropout == 0.0

    state.global_step = 50
    cb.on_step_begin(None, state, None)
    assert (mag.eigvec_dropout, mag.mlp_dropout, gb.droppath, gb.bias_dropout) \
        == (0.1, 0.2, 0.3, 0.4), "post-onset: configured rates restored"

    # Resume-safety: a callback built fresh at global_step past onset activates.
    model2 = _tiny_model(**drop)
    cb2 = RegOnsetCallback(model2, onset_frac=0.5)
    cb2.on_train_begin(None, types.SimpleNamespace(max_steps=100, global_step=70), None)
    assert _bias_module(model2, MagneticBias).eigvec_dropout == 0.1
