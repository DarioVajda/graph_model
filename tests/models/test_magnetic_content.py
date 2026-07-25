"""Tests for the content-conditioned magnetic bias (``magnetic_content``).

``magnetic_content`` widens ``MagneticBias``'s final projection with a live,
per-node semantic summary sliced from the first token of every node's sequence
(captured by a forward pre-hook on each attention module). This suite pins:

  * ``_node_start_indices`` (the inverse of ``node_ids``, with the clamp-to-0
    sentinel for absent / padded node slots),
  * the forward pre-hook actually captures the hidden states,
  * zero-init inertness (the content term starts at exactly 0),
  * non-vacuousness (once grown in, it moves the loss),
  * end-to-end gradient flow back into the token embeddings (the whole point:
    the LM is incentivised to pack payload into each node's first token),
  * gradient-checkpointing parity (the pre-hook re-fires on recompute, so grads
    match the non-checkpointed run), and
  * coexistence with the plain ``magnetic`` bias.

Run with:  pytest tests/models/test_magnetic_content.py -v
"""

import pytest
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.causal_lm import _node_start_indices
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from tests.helpers.tiny_model import BASE_CONFIG

DEVICE = torch.device("cpu")

# magnetic_content on, plain magnetic off (isolates the content arm).
_MC = dict(magnetic=False, magnetic_content=True, magnetic_dim=16, magnetic_content_dim=16)
_NODE_COUNTS = (4, 9, 6)   # a spread → the collator pads, exercising sentinel slots


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _make_items(node_counts=_NODE_COUNTS, seed=0):
    """Items carrying full (m = N) magnetic eigenvectors and per-node token ids."""
    torch.manual_seed(seed)
    items = []
    for n in node_counts:
        item = {
            "num_nodes": n,
            "prompt_node": n - 1,
            "edges": [(i, (i + 1) % n) for i in range(n)],
            "input_ids": [torch.randint(1, 256, (3,)).tolist() for _ in range(n)],
            "magnetic_V": torch.randn(n, n, 2),
            "magnetic_lambdas": torch.randn(n),
        }
        item["labels"] = torch.tensor(item["input_ids"][item["prompt_node"]], dtype=torch.long)
        items.append(item)
    return items


def _batch(items=None):
    items = items if items is not None else _make_items()
    return GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in items])


def _model(**overrides):
    cfg = dict(k_hop=0, graph_attn_impl="eager")
    cfg.update(_MC)
    cfg.update(overrides)
    torch.manual_seed(0)
    return GTLMLlamaForCausalLM(GTLMLlamaConfig(**cfg, **BASE_CONFIG)).to(DEVICE, torch.float32)


def _mc_module(model, layer=0):
    """The MagneticContentBias instance (last active per-layer bias module)."""
    return model.model.layers[layer].self_attn.graph_bias.bias_modules[-1]


def _grow_in(model, std=0.05, seed=42):
    """Randomise the zero-init final projection so the content term is live."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in model.model.layers:
            mc = _mc_module_of(layer)
            torch.nn.init.normal_(mc.proj[2].weight, std=std)
            torch.nn.init.normal_(mc.proj[2].bias, std=std)


def _mc_module_of(layer):
    return layer.self_attn.graph_bias.bias_modules[-1]


# ─── _node_start_indices ──────────────────────────────────────────────────────

class TestNodeStartIndices:

    def test_first_occurrence_and_sentinel(self):
        # tokens → node id: node 2 (pos 0,1), node 0 (pos 2,3,4), node 1 (pos 5).
        # nodes 3 and 4 never appear → clamp-to-0 sentinel.
        node_ids = torch.tensor([[2, 2, 0, 0, 0, 1]])
        idx = _node_start_indices(node_ids, num_node_slots=5)
        assert idx.tolist() == [[2, 5, 0, 0, 0]]

    def test_trailing_padding_does_not_shadow_real_first_occurrence(self):
        # Right-pad tokens carry node id 0; the real node-0 token appears first,
        # and amin must keep the smallest position (0), not a trailing pad.
        node_ids = torch.tensor([[0, 1, 2, 0, 0]])   # last two are padding on node 0
        idx = _node_start_indices(node_ids, num_node_slots=3)
        assert idx.tolist() == [[0, 1, 2]]

    def test_batch_independent(self):
        node_ids = torch.tensor([[0, 1, 2], [2, 2, 0]])
        idx = _node_start_indices(node_ids, num_node_slots=3)
        assert idx.tolist() == [[0, 1, 2], [2, 0, 0]]  # row1: node1 absent → 0


# ─── Mechanism ────────────────────────────────────────────────────────────────

class TestMagneticContent:

    def test_forward_finite(self):
        model = _model()
        _grow_in(model)
        out = model(**_batch())
        assert torch.isfinite(out.logits).all()
        assert torch.isfinite(out.loss)

    def test_prehook_captures_hidden_states(self):
        model = _model()
        _ = model(**_batch())
        for layer in model.model.layers:
            hs = layer.self_attn._captured_hidden_states
            assert hs is not None
            assert hs.shape[-1] == BASE_CONFIG["hidden_size"]

    def test_no_prehook_when_disabled(self):
        model = _model(magnetic_content=False)
        for layer in model.model.layers:
            assert not hasattr(layer.self_attn, "_captured_hidden_states")

    def test_truncated_eigenvectors(self):
        # Regression: with truncated eigenvectors (magnetic_m < N) the node count
        # and the eigenvalue count differ, so node_start_indices MUST be sized to
        # the node dim (magnetic_V.shape[1]), not magnetic_lambdas.shape[1].
        # node_counts max = 9, magnetic_m = 3  →  spectral is (B, 9, 9, m), and a
        # 3-slot node_start_indices would fail the endpoint-summary concat.
        model = _model()
        _grow_in(model)
        batch = GraphCollatorV2(pad_token_id=0, k_hop=0, magnetic_m=3)(
            [dict(it) for it in _make_items()])
        assert batch["magnetic_V"].shape[1] != batch["magnetic_lambdas"].shape[1]
        out = model(**batch)
        assert torch.isfinite(out.logits).all()
        assert torch.isfinite(out.loss)

    def test_zero_init_is_inert(self):
        # At init proj[2] is zero → the content bias is exactly 0, so logits must
        # match a model with no soft bias at all (identical backbone weights).
        on = _model()
        off = _model(magnetic_content=False)
        off.load_state_dict(on.state_dict(), strict=False)   # copy backbone
        on.eval(); off.eval()
        batch = _batch()
        with torch.no_grad():
            lo = on(**batch).logits
            lf = off(**batch).logits
        assert torch.allclose(lo, lf, atol=1e-6), \
            "zero-init content bias must be inert (logits unchanged)"

    def test_content_bias_moves_loss(self):
        # Guard against a vacuous suite: once grown in, it must change the loss.
        model = _model()
        model.eval()
        batch = _batch()
        with torch.no_grad():
            loss_inert = model(**batch).loss.item()
        _grow_in(model)
        with torch.no_grad():
            loss_active = model(**batch).loss.item()
        assert abs(loss_active - loss_inert) > 1e-6

    def test_down_projection_receives_gradient(self):
        model = _model()
        _grow_in(model)
        model.train()
        model(**_batch()).loss.backward()
        down_w = _mc_module(model).down[0].weight
        assert down_w.grad is not None
        assert down_w.grad.abs().sum() > 0

    def test_gradient_flows_into_embeddings(self):
        # The defining property: the content pathway feeds gradient back into the
        # token embeddings (via the sliced hidden states). Turning the content
        # term on must change the embedding gradient.
        batch = _batch()

        def embed_grad(grow):
            model = _model()
            if grow:
                _grow_in(model)
            model.train()
            model.zero_grad()
            model(**batch).loss.backward()
            return model.model.embed_tokens.weight.grad.detach().clone()

        g_inert = embed_grad(grow=False)    # proj[2] = 0 → content path is dead
        g_active = embed_grad(grow=True)    # content path is live
        assert not torch.allclose(g_inert, g_active), \
            "content bias must contribute to the embedding gradient"

    def test_gradient_checkpointing_parity(self):
        # The pre-hook re-fires on recompute, so the down-MLP gradient under
        # gradient checkpointing must match the non-checkpointed run.
        batch = _batch()

        def down_grad(use_ckpt):
            model = _model()
            _grow_in(model)
            if use_ckpt:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            model.zero_grad()
            model(**batch).loss.backward()
            return _mc_module(model).down[0].weight.grad.detach().clone()

        g_plain = down_grad(use_ckpt=False)
        g_ckpt = down_grad(use_ckpt=True)
        assert torch.allclose(g_plain, g_ckpt, rtol=1e-3, atol=1e-5), \
            f"checkpoint grad mismatch (max {(g_plain - g_ckpt).abs().max().item():.2e})"

    def test_coexists_with_plain_magnetic(self):
        model = _model(magnetic=True)
        _grow_in(model)
        out = model(**_batch())
        assert torch.isfinite(out.logits).all()
        assert torch.isfinite(out.loss)
