"""
v0 vs v2 parity when magnetic eigenvector counts vary widely WITHIN a batch.

Datasets built with ``m=0`` store *all* eigenvectors, so ``magnetic_V`` is
``(N, N, 2)`` and the eigenvector count is the graph's own node count. Batching
such graphs makes ``GraphCollatorV2`` pad ``m`` up to the batch maximum and
zero-fill the rest, while v0's ragged ``GraphCollator`` never padded at all — so
a mis-masked pad slot would change the bias for the *smaller* graphs in a mixed
batch and silently move results rather than crash.

``test_modeling_gtlm_llama_v2`` already covers this mechanism, but only at a 3-vs-4
node spread, where a padding bug moves almost nothing. The graphqa experiment
batches whole-graph spreads of 6 vs 191 (incidence Levi graphs), so this pins the
same parity at a wide spread with graphqa's real bias config (spd + rrwp +
magnetic together).
"""

import pytest
import torch

from src.models.legacy.modeling_gtlm_llama_v0 import (
    GraphAttnBiasConfig, GraphLlamaConfig, GraphLlamaForCausalLM,
)
from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.utils.text_graph_collator import GraphCollator
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from tests.helpers.bias_params import iter_bias_param_pairs, transfer_bias_weights
from tests.helpers.tiny_model import BASE_CONFIG

DEVICE = torch.device("cpu")

# graphqa's bias configuration: all three features on at once.
_BIAS = dict(spd=True, max_spd=8, laplacian=False, rwse=False,
             rrwp=True, max_rw_steps=16, magnetic=True, magnetic_dim=32)

# A batch mixing tiny and large graphs: m gets padded 4 -> 40 for the small ones.
_NODE_COUNTS = (4, 40, 7)


def _make_items(node_counts=_NODE_COUNTS, seed=0):
    """Items carrying FULL (m = N) magnetic eigenvectors, as an m=0 cache stores."""
    torch.manual_seed(seed)
    items = []
    for n in node_counts:
        item = {
            "num_nodes": n,
            "prompt_node": n - 1,
            "edges": [(i, (i + 1) % n) for i in range(n)],
            "input_ids": [torch.randint(1, 256, (3,)).tolist() for _ in range(n)],
            "shortest_path_dists": torch.randint(0, 6, (n, n)),
            "rrwp": torch.randn(n, n, _BIAS["max_rw_steps"]),
            "magnetic_V": torch.randn(n, n, 2),        # m == N (full spectrum)
            "magnetic_lambdas": torch.randn(n),
        }
        item["labels"] = torch.tensor(item["input_ids"][item["prompt_node"]], dtype=torch.long)
        items.append(item)
    return items


def _build_pair():
    """(v0, v2-eager) with identical weights and a non-zero magnetic projection."""
    bias_cfg = GraphAttnBiasConfig(**_BIAS)
    torch.manual_seed(0)
    v0 = GraphLlamaForCausalLM(
        GraphLlamaConfig(graph_attn_bias=bias_cfg.to_dict(), **BASE_CONFIG)
    ).to(DEVICE, torch.float32)
    v2 = GTLMLlamaForCausalLM(
        GTLMLlamaConfig(k_hop=0, graph_attn_impl="eager", **_BIAS, **BASE_CONFIG)
    ).to(DEVICE, torch.float32)

    missing, _ = v2.load_state_dict(v0.state_dict(), strict=False)
    assert not [k for k in missing if "inv_freq" not in k and "graph_bias" not in k], missing

    # The magnetic output projection is zero-init; leave it and the magnetic path
    # cannot reach the logits, making this test pass vacuously.
    torch.manual_seed(42)
    with torch.no_grad():
        for layer in v0.model.layers:
            torch.nn.init.normal_(layer.self_attn.magnetic_bias_proj[2].weight, std=0.02)
            torch.nn.init.normal_(layer.self_attn.magnetic_bias_proj[2].bias, std=0.02)
    transfer_bias_weights(v0, v2, bias_cfg)
    return v0, v2, bias_cfg


def test_collator_pads_m_to_batch_max():
    """Guard the premise: the batch really does exercise ragged-m padding."""
    items = _make_items()
    batch = GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in items])
    assert batch["magnetic_V"].shape[1:3] == (max(_NODE_COUNTS), max(_NODE_COUNTS)), \
        "expected magnetic_V padded to the batch's max node count"


def test_magnetic_path_is_live():
    """Guard against a vacuous parity pass: magnetic must move the loss."""
    items = _make_items()
    torch.manual_seed(0)
    on = GTLMLlamaForCausalLM(
        GTLMLlamaConfig(k_hop=0, graph_attn_impl="eager", **_BIAS, **BASE_CONFIG)
    ).to(DEVICE, torch.float32)
    off = GTLMLlamaForCausalLM(
        GTLMLlamaConfig(k_hop=0, graph_attn_impl="eager", **dict(_BIAS, magnetic=False),
                        **BASE_CONFIG)
    ).to(DEVICE, torch.float32)
    off.load_state_dict(on.state_dict(), strict=False)

    torch.manual_seed(42)
    with torch.no_grad():
        for layer in on.model.layers:
            proj = layer.self_attn.graph_bias.bias_modules[-1].proj[2]
            torch.nn.init.normal_(proj.weight, std=0.02)
            torch.nn.init.normal_(proj.bias, std=0.02)

    batch = GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in items])
    assert abs(on(**batch).loss.item() - off(**batch).loss.item()) > 1e-6, \
        "magnetic bias does not affect the loss — parity tests would be vacuous"


def test_v2_matches_v0_with_ragged_m():
    """v2's zero-padded m must reproduce v0's ragged batch: loss + every bias grad."""
    items = _make_items()
    v0, v2, bias_cfg = _build_pair()
    v0.train(); v2.train()

    out0 = v0(input_graph_batch=GraphCollator()([dict(it) for it in items]),
              labels=[it["labels"] for it in items])
    out2 = v2(**GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in items]))

    assert abs(out0.loss.item() - out2.loss.item()) < 1e-4, \
        f"loss diff {abs(out0.loss.item() - out2.loss.item()):.3e}"

    out0.loss.backward()
    out2.loss.backward()
    for name, p0, p2 in iter_bias_param_pairs(v0, v2, bias_cfg):
        assert p0.grad is not None and p2.grad is not None, f"missing grad {name}"
        assert torch.allclose(p0.grad, p2.grad, rtol=1e-3, atol=1e-5), \
            f"grad mismatch {name} (max {(p0.grad - p2.grad).abs().max().item():.2e})"


@pytest.mark.parametrize("node_counts", [(4, 40), (5, 5), (3, 60, 3)])
def test_v2_matches_v0_across_spreads(node_counts):
    """Uniform-N (no padding) and extreme spreads must be equally exact."""
    items = _make_items(node_counts, seed=1)
    v0, v2, _ = _build_pair()
    v0.eval(); v2.eval()

    with torch.no_grad():
        out0 = v0(input_graph_batch=GraphCollator()([dict(it) for it in items]),
                  labels=[it["labels"] for it in items])
        out2 = v2(**GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in items]))
    assert abs(out0.loss.item() - out2.loss.item()) < 1e-4
