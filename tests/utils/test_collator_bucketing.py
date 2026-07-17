"""
Tests for GraphCollatorV2 flex bucketing (L and N) and the bucketize helper.

Covers:
  * the default L (512-multiple + midpoint) and N (pow2 floored at 32) ladders;
  * ``bucketize`` accepting None / callable / sorted-list specs (+ overflow error);
  * ``pad_to_block`` padding both L and N to buckets, with real token/label
    positions byte-identical to the unpadded collation;
  * custom ``len_buckets`` / ``node_buckets`` overrides and the L-alignment guard;
  * padding is loss-neutral for the dense (eager) backend — the property the flex
    path relies on (padded tokens masked, padded nodes never gathered).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.models.flex_kernel import default_len_buckets, default_node_buckets, bucketize
from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)


def _make_items(seed=0):
    torch.manual_seed(seed)
    specs = [
        ([3, 2, 4, 2], 3, [(0, 1), (1, 2), (2, 3), (0, 3)]),
        ([2, 3, 2], 0, [(0, 1), (1, 2)]),
    ]
    spec_dim, rwse_dim, rw_steps = 4, 4, 4
    items = []
    for tok_lens, prompt, edges in specs:
        N = len(tok_lens)
        item = {
            "num_nodes": N, "prompt_node": prompt, "edges": edges,
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


# ── ladders & bucketize ──────────────────────────────────────────────────────

def test_default_len_ladder():
    assert [default_len_buckets(x) for x in (90, 512, 513, 1024, 1025, 1600, 2049)] \
        == [512, 512, 1024, 1024, 1536, 2048, 3072]


def test_default_node_ladder():
    assert [default_node_buckets(x) for x in (5, 32, 33, 64, 65, 200, 1025)] \
        == [32, 32, 64, 64, 128, 256, 2048]


def test_bucketize_forms():
    assert bucketize(300, None) == 300
    assert bucketize(300, default_len_buckets) == 512
    assert bucketize(300, [256, 512, 1024]) == 512
    assert bucketize(256, [256, 512]) == 256          # exact match is allowed
    with pytest.raises(ValueError):
        bucketize(2000, [256, 512])                    # nothing large enough


# ── collator padding ─────────────────────────────────────────────────────────

def test_pad_to_block_pads_L_and_N():
    items = _make_items()
    raw = GraphCollatorV2(pad_token_id=0, k_hop=2)([dict(it) for it in items])
    pad = GraphCollatorV2(pad_token_id=0, k_hop=2, pad_to_block=True)([dict(it) for it in items])

    L0 = raw["input_ids"].shape[1]
    N0 = raw["shortest_path_dists"].shape[1]
    assert pad["input_ids"].shape[1] == default_len_buckets(L0)
    assert pad["input_ids"].shape[1] % 128 == 0
    # every node-indexed feature + the k-hop mask share the bucketed N
    Np = default_node_buckets(N0)
    for key in ("shortest_path_dists", "laplacian_coordinates", "rwse", "rrwp",
                "magnetic_V", "k_hop_mask"):
        assert pad[key].shape[1] == Np, key
    assert pad["shortest_path_dists"].shape[2] == Np
    assert pad["k_hop_mask"].shape[2] == Np


def test_real_positions_preserved():
    items = _make_items()
    raw = GraphCollatorV2(pad_token_id=0, k_hop=2)([dict(it) for it in items])
    pad = GraphCollatorV2(pad_token_id=0, k_hop=2, pad_to_block=True)([dict(it) for it in items])
    L0, N0 = raw["input_ids"].shape[1], raw["shortest_path_dists"].shape[1]

    assert torch.equal(raw["input_ids"], pad["input_ids"][:, :L0])
    assert torch.equal(raw["attention_mask"], pad["attention_mask"][:, :L0])
    assert torch.equal(raw["labels"], pad["labels"][:, :L0])
    # padded token positions: masked, no loss, node -> prompt node
    assert (pad["attention_mask"][:, L0:] == 0).all()
    assert (pad["labels"][:, L0:] == -100).all()
    assert (pad["node_ids"][:, L0:] == pad["prompt_node"].view(-1, 1)).all()
    # each graph's genuinely-real node block is untouched (the don't-care fill
    # for intra-batch padding may differ — it's never gathered by real tokens)
    for i, n in enumerate(raw["num_nodes"].tolist()):
        assert torch.equal(raw["shortest_path_dists"][i, :n, :n],
                           pad["shortest_path_dists"][i, :n, :n])


def test_custom_bucket_specs():
    items = _make_items()
    c = GraphCollatorV2(pad_token_id=0, k_hop=0, pad_to_block=True,
                        len_buckets=[256, 1024], node_buckets=[16, 64])
    b = c([dict(it) for it in items])
    assert b["input_ids"].shape[1] == 256
    assert b["shortest_path_dists"].shape[1] == 16


def test_len_bucket_alignment_guard():
    items = _make_items()
    c = GraphCollatorV2(pad_token_id=0, pad_to_block=True,
                        len_buckets=lambda x: 300, block_size=128)
    with pytest.raises(ValueError, match="not a multiple of"):
        c([dict(it) for it in items])


# ── padding is loss-neutral for the dense backend ────────────────────────────

@pytest.mark.parametrize("k_hop", [0, 2])
def test_padding_loss_neutral_eager(k_hop):
    items = _make_items()
    cfg = GTLMLlamaConfig(
        k_hop=k_hop, graph_attn_impl="eager",
        spd=True, max_spd=8, magnetic=True, magnetic_dim=8, **_BASE,
    )
    model = GTLMLlamaForCausalLM(cfg).double().eval()   # fp64 for a tight check

    raw = GraphCollatorV2(pad_token_id=0, k_hop=k_hop)([dict(it) for it in items])
    pad = GraphCollatorV2(pad_token_id=0, k_hop=k_hop, pad_to_block=True)([dict(it) for it in items])
    for b in (raw, pad):
        for key in ("laplacian_coordinates", "rwse", "rrwp", "magnetic_V", "magnetic_lambdas"):
            if b.get(key) is not None:
                b[key] = b[key].double()

    with torch.no_grad():
        loss_raw = model(**raw).loss
        loss_pad = model(**pad).loss
    assert torch.allclose(loss_raw, loss_pad, atol=1e-9, rtol=1e-9), \
        f"k={k_hop}: padded loss {loss_pad.item()} != raw {loss_raw.item()}"
