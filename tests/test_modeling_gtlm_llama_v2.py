"""
Parity tests for the v2 GTLM-Llama model.

These assert that the clean v2 rewrite preserves the exact behaviour of v1:

  * ``test_v2_eager_matches_v1`` — v2 (eager) logits equal v1 logits on the same
    graphs and weights, across bias combinations and K-hop values.
  * ``test_v2_sdpa_matches_eager`` — the sdpa backend matches eager.
  * ``test_v2_grad_checkpointing_parity`` — enabling gradient checkpointing does
    not change the loss (the context-on-module mechanism survives recompute).

v1 and v2 use identical sub-module names (both build ``GraphAttentionBias`` inside
a ``LlamaAttention`` subclass), so weights transfer via a plain ``load_state_dict``.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from src.models.legacy.modeling_gtlm_llama_v0 import GraphAttnBiasConfig
from src.models.legacy.modeling_gtlm_llama_v1 import KHopGraphLlamaForCausalLM, KHopLlamaConfig
from src.models.modeling_gtlm_llama_v2 import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.utils.text_graph_collator import GraphCollator
from src.utils.text_graph_collator_v2 import GraphCollatorV2, _single_k_hop_mask

DEVICE = torch.device("cpu")

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=256,
    pad_token_id=0, _attn_implementation="eager",
)

# A spread of bias configurations to exercise every code path.
_BIAS_CONFIGS = {
    "none": dict(),
    "spd": dict(spd=True, max_spd=8),
    "laplacian": dict(laplacian=True),
    "rwse": dict(rwse=True),
    "rrwp": dict(rrwp=True, max_rw_steps=4),
    "magnetic": dict(magnetic=True, magnetic_dim=8),
    "all": dict(spd=True, max_spd=8, laplacian=True, rwse=True,
                rrwp=True, max_rw_steps=4, magnetic=True, magnetic_dim=8),
}


def _bias_kwargs(name):
    base = dict(spd=False, laplacian=False, max_spd=32, rwse=False, rrwp=False,
                max_rw_steps=8, magnetic=False, magnetic_dim=32)
    base.update(_BIAS_CONFIGS[name])
    return base


def _make_items(seed=0):
    """A small batch of synthetic TextGraph items carrying every feature."""
    torch.manual_seed(seed)
    specs = [  # (node token lengths, prompt_node, edges)
        ([3, 2, 4, 2], 3, [(0, 1), (1, 2), (2, 3), (0, 3)]),
        ([2, 3, 2], 0, [(0, 1), (1, 2)]),
    ]
    spec_dim, rwse_dim, rw_steps = 4, 4, 4
    items = []
    for tok_lens, prompt, edges in specs:
        N = len(tok_lens)
        item = {
            "num_nodes": N,
            "prompt_node": prompt,
            "edges": edges,
            "input_ids": [torch.randint(1, 256, (l,)).tolist() for l in tok_lens],
            "laplacian_coordinates": torch.randn(N, spec_dim),
            "shortest_path_dists": torch.randint(0, 5, (N, N)),
            "rwse": torch.randn(N, rwse_dim),
            "rrwp": torch.randn(N, N, rw_steps),
            "magnetic_V": torch.randn(N, N, 2),
            "magnetic_lambdas": torch.randn(N),
        }
        item["labels"] = torch.tensor(item["input_ids"][prompt], dtype=torch.long)
        items.append(item)
    return items


def _build_pair(bias_name, k_hop, impl="eager"):
    bias = _bias_kwargs(bias_name)
    v1_cfg = KHopLlamaConfig(
        graph_attn_bias=GraphAttnBiasConfig(**bias).to_dict(), k_hop=k_hop, **_BASE
    )
    v1 = KHopGraphLlamaForCausalLM(v1_cfg).to(DEVICE).eval()

    v2_cfg = GTLMLlamaConfig(k_hop=k_hop, graph_attn_impl=impl, **bias, **_BASE)
    v2 = GTLMLlamaForCausalLM(v2_cfg).to(DEVICE).eval()

    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    # Only buffer-like keys (e.g. rotary inv_freq) may differ between builds.
    assert not [k for k in missing if "inv_freq" not in k], f"missing: {missing}"
    assert not [k for k in unexpected if "inv_freq" not in k], f"unexpected: {unexpected}"
    return v1, v2


@pytest.mark.parametrize("bias_name", list(_BIAS_CONFIGS.keys()))
@pytest.mark.parametrize("k_hop", [0, 2])
def test_v2_eager_matches_v1(bias_name, k_hop):
    items = _make_items()
    v1, v2 = _build_pair(bias_name, k_hop, impl="eager")

    old_batch = GraphCollator(k_hop=k_hop)([dict(it) for it in items])
    new_batch = GraphCollatorV2(pad_token_id=0, k_hop=k_hop)([dict(it) for it in items])

    labels_list = [it["labels"] for it in items]
    with torch.no_grad():
        out_v1 = v1(input_graph_batch=old_batch, labels=labels_list)
        out_v2 = v2(**new_batch)

    mask = new_batch["attention_mask"].bool()
    diff = (out_v1.logits[mask] - out_v2.logits[mask]).abs().max().item()
    assert diff < 1e-4, f"[{bias_name}, k={k_hop}] max logit diff {diff}"

    if out_v1.loss is not None and out_v2.loss is not None:
        assert abs(out_v1.loss.item() - out_v2.loss.item()) < 1e-4


@pytest.mark.parametrize("bias_name", ["none", "all"])
@pytest.mark.parametrize("k_hop", [0, 2])
def test_v2_sdpa_matches_eager(bias_name, k_hop):
    items = _make_items()
    _, v2_eager = _build_pair(bias_name, k_hop, impl="eager")
    _, v2_sdpa = _build_pair(bias_name, k_hop, impl="sdpa")
    v2_sdpa.load_state_dict(v2_eager.state_dict())

    batch = GraphCollatorV2(pad_token_id=0, k_hop=k_hop)([dict(it) for it in items])
    with torch.no_grad():
        oe = v2_eager(**batch)
        os_ = v2_sdpa(**batch)
    mask = batch["attention_mask"].bool()
    diff = (oe.logits[mask] - os_.logits[mask]).abs().max().item()
    assert diff < 1e-4, f"[{bias_name}, k={k_hop}] eager-vs-sdpa diff {diff}"


@pytest.mark.parametrize("bias_name", ["all"])
@pytest.mark.parametrize("k_hop", [0, 2])
def test_v2_grad_checkpointing_parity(bias_name, k_hop):
    items = _make_items()
    _, v2a = _build_pair(bias_name, k_hop, impl="eager")
    _, v2b = _build_pair(bias_name, k_hop, impl="eager")
    v2b.load_state_dict(v2a.state_dict())

    batch = GraphCollatorV2(pad_token_id=0, k_hop=k_hop)([dict(it) for it in items])

    v2a.train()
    v2b.gradient_checkpointing_enable()
    v2b.train()

    loss_a = v2a(**batch).loss
    loss_b = v2b(**batch).loss
    assert abs(loss_a.item() - loss_b.item()) < 1e-5, f"grad-ckpt loss diff {abs(loss_a.item()-loss_b.item())}"


# ── Directed K-hop gate ─────────────────────────────────────────────────────────

def test_directed_k_hop_mask_chain():
    """Directed out-edge traversal: i sees j iff a directed path i->j of length<=K."""
    edges = [(0, 1), (1, 2), (2, 3)]   # directed chain 0->1->2->3
    N, prompt, K = 4, 3, 2
    m = _single_k_hop_mask(edges, N, prompt, K, directed=True)

    assert bool(m[0, 1]) and bool(m[0, 2])      # 1-hop and 2-hop forward
    assert not bool(m[0, 3])                     # 3 hops > K
    assert not bool(m[1, 0])                      # against edge direction
    assert not bool(m[2, 0]) and not bool(m[3, 0])
    assert all(bool(m[i, i]) for i in range(N))   # diagonal always open

    # Undirected on the same graph is symmetric where directed is not.
    mu = _single_k_hop_mask(edges, N, prompt, K, directed=False)
    assert bool(mu[1, 0]) and bool(mu[0, 1])


def test_directed_flag_changes_attention():
    """End-to-end: feeding a directed vs undirected mask to the same weights
    produces different logits on a directed graph."""
    items = _make_items()
    _, model = _build_pair("all", k_hop=2, impl="eager")

    b_undir = GraphCollatorV2(pad_token_id=0, k_hop=2, k_hop_directed=False)([dict(it) for it in items])
    b_dir   = GraphCollatorV2(pad_token_id=0, k_hop=2, k_hop_directed=True)([dict(it) for it in items])

    assert not torch.equal(b_undir["k_hop_mask"], b_dir["k_hop_mask"])
    with torch.no_grad():
        lo_u = model(**b_undir).logits
        lo_d = model(**b_dir).logits
    mask = b_undir["attention_mask"].bool()
    assert (lo_u[mask] - lo_d[mask]).abs().max().item() > 1e-3


def test_k_hop_directed_config_roundtrip():
    """The directed flag must survive save/load so a downloaded checkpoint is
    self-describing."""
    cfg = GTLMLlamaConfig(k_hop=2, k_hop_directed=True, **_BASE)
    with tempfile.TemporaryDirectory() as d:
        GTLMLlamaForCausalLM(cfg).save_pretrained(d)
        rt = GTLMLlamaConfig.from_pretrained(d)
    assert rt.k_hop_directed is True
    assert rt.k_hop == 2
