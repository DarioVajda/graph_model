"""
Trainer-level tests for the graph-bias weight-decay fix (TODO_reg Part 1):

  * ``GraphTrainerV2.create_optimizer``'s shape-based weight-decay split for
    graph-bias params (the ``bias_weight_decay`` knob): magnetic matrices decay,
    the SPD lookup table (``_no_weight_decay``) and all 1-D gains/offsets don't,
    and the backbone split stays byte-identical to HF's name-based rule.
  * The no-regression guarantee: the default ``bias_weight_decay=0.0``
    reproduces the historical behavior (no decay anywhere on the bias path).

Run with:  pytest tests/test_bias_regularization.py -v
"""


import torch
from transformers import TrainingArguments

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import MagneticBias, SPDBias, LaplacianBias, RWSEBias
from src.utils import GraphTrainerV2
from tests.helpers.tiny_model import BASE_CONFIG

_BIAS = dict(spd=True, max_spd=8, laplacian=True, rwse=True,
             magnetic=True, magnetic_dim=8)


def _tiny_model(**overrides):
    torch.manual_seed(0)
    cfg = GTLMLlamaConfig(k_hop=0, graph_attn_impl="eager", **_BIAS, **BASE_CONFIG, **overrides)
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
