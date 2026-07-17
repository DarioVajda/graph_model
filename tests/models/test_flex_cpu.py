"""
CPU-safe tests for the flex integration that don't need the CUDA kernel:

  * the K-hop block-size gate (#6);
  * graph_attention_dispatch refusing impl='flex' (flex is routed in the
    attention forward, not the dispatch);
  * the model's block-alignment guard firing (before any CUDA op) when a flex
    batch isn't length-aligned;
  * permutation invariance through the dataset's RCM on the *eager* backend —
    the same property the GPU test checks, but runnable in CI without a GPU and
    isolating the model+dataset from the flex kernel.
"""


import networkx as nx
import pytest
import torch

from src.models.dispatch import graph_attention_dispatch, flex_block_size
from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.utils.text_graph_dataset import TextGraphDataset

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)


def test_flex_block_size_gate():
    # default compile mode is autotune → the tight 64-wide K-hop blocks are kept
    assert flex_block_size(0) == 128         # dense mask: nothing to skip
    assert flex_block_size(2) == 64          # K-hop sparsity: tighter blocks
    assert flex_block_size(4) == 64

    # Non-autotune inductor can only lower 128-wide tiles, so a sub-128 block
    # would fail to compile; the recommendation rounds up to 128 (correctness is
    # unaffected — only sparsity). k_hop=0 is already 128 regardless.
    for mode in ("default", None):
        assert flex_block_size(2, mode) == 128
        assert flex_block_size(0, mode) == 128
    # autotune modes carry 64-tile configs → the tighter block survives
    assert flex_block_size(2, "max-autotune-no-cudagraphs") == 64


def test_dispatch_rejects_flex():
    # graph_attention_dispatch handles only the dense (eager) backend; the
    # flex path is owned by gtlm_flex / flex_attention_forward. Passing "flex"
    # here is a programming error and is rejected.
    with pytest.raises(ValueError, match="flex"):
        graph_attention_dispatch("flex", None, None, None, None, None, None,
                                 scaling=1.0, dropout=0.0)


def test_flex_requires_block_aligned_length():
    """A flex batch whose length isn't a block multiple must raise a clear error
    *before* touching the CUDA kernel (so this runs on CPU)."""
    cfg = GTLMLlamaConfig(k_hop=0, graph_attn_impl="flex", flex_block_size=128, **_BASE)
    model = GTLMLlamaForCausalLM(cfg).eval()
    items = _equiv_items(rcm=False)                      # short, unaligned L
    batch = GraphCollatorV2(pad_token_id=0, k_hop=0)([dict(it) for it in items])  # no pad_to_block
    assert batch["input_ids"].shape[1] % 128 != 0
    with pytest.raises(ValueError, match="multiple of"):
        model(**batch)


# ── permutation invariance via RCM, eager / CPU ──────────────────────────────

def _equiv_items(rcm: bool):
    graphs = []
    for i in range(2):
        g = nx.barabasi_albert_graph(6 + i, 2, seed=i)
        nx.set_node_attributes(g, {nd: "x" for nd in g.nodes()}, "text")
        g.graph["prompt_node"] = i % g.number_of_nodes()
        graphs.append(g)
    ds = TextGraphDataset(graphs, rcm_ordering=rcm)
    ds.compute_shortest_path_distances(use_gpu=False)

    def toks(o):
        return [1 + (o * 37 + 5) % 200, 1 + (o * 91 + 11) % 200]

    items = []
    for idx in range(len(ds)):
        it = ds[idx]
        n = it["num_nodes"]
        cur_to_orig = {cur: orig for orig, cur in it["original_ids"].items()}
        input_ids = [toks(cur_to_orig[i]) for i in range(n)]
        items.append({
            "num_nodes": n, "prompt_node": it["prompt_node"], "edges": it["edges"],
            "input_ids": input_ids,
            "shortest_path_dists": it["shortest_path_dists"],
            "labels": torch.tensor(input_ids[it["prompt_node"]], dtype=torch.long),
        })
    return items


@pytest.mark.parametrize("k_hop", [0, 2])
def test_eager_permutation_invariance_via_rcm(k_hop):
    """Reordering the prefix via the dataset's RCM leaves the prompt-span logits
    unchanged (eager backend, fp64 — tight). Catches both a non-invariant model
    and an inconsistent dataset reorder, with no GPU."""
    cfg = GTLMLlamaConfig(k_hop=k_hop, graph_attn_impl="eager", spd=True, max_spd=8, **_BASE)
    model = GTLMLlamaForCausalLM(cfg).double().eval()

    def collate(rcm):
        return GraphCollatorV2(pad_token_id=0, k_hop=k_hop)([dict(it) for it in _equiv_items(rcm)])

    b0, b1 = collate(False), collate(True)
    with torch.no_grad():
        o0, o1 = model(**b0), model(**b1)

    m0, m1 = b0["labels"] != -100, b1["labels"] != -100
    assert torch.equal(m0, m1)
    # fp64 accumulation floor is ~1e-7; a misaligned reorder would be ~1e-2+
    diff = (o0.logits[m0] - o1.logits[m1]).abs().max().item()
    assert diff < 1e-6, f"k={k_hop}: prompt-logit diff under RCM reorder {diff}"
