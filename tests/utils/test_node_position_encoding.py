"""
Pin the E3 SPD-depth position-encoding knob (2026-07-16, TODO.md).

Three properties, each with its own test:

1. ``node_position_mode="reset"`` is a byte-identical no-op against the
   pre-E3 collator code path — this is what licenses reusing
   `029_question_node_webqsp`'s isolated-arm numbers as this experiment's
   control instead of retraining it.
2. Backward compatibility: a single-node (prompt-only) graph collapses to
   plain ``arange(len)`` position_ids under EITHER mode — the core GTLM
   invariant ("single-node-graph collapses to normal LLM behaviour").
3. Permutation equivariance: relabeling which physical entity gets which
   node index must not change the model's output logits — the other core
   GTLM invariant, tested end-to-end (mask + bias + the new position
   mechanism together) on the actual v2 model stack, not just the collator
   in isolation, since that's the property that actually matters.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.utils.text_graph_collator_v2 import GraphCollatorV2

DEVICE = torch.device("cpu")
_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=4096,
    pad_token_id=0, _attn_implementation="eager",
)


# ── 1. "reset" is a no-op ────────────────────────────────────────────────────

def _make_multi_node_item(seed=0):
    torch.manual_seed(seed)
    tok_lens = [3, 2, 4, 2]     # last is the prompt node
    N, prompt = len(tok_lens), len(tok_lens) - 1
    return {
        "num_nodes": N, "prompt_node": prompt,
        "edges": [(0, 1), (1, 2), (0, prompt), (2, prompt)],
        "input_ids": [torch.randint(1, 256, (l,)).tolist() for l in tok_lens],
        "shortest_path_dists": torch.randint(0, 5, (N, N)),
    }


def test_reset_mode_is_a_noop():
    item = _make_multi_node_item()
    col_reset = GraphCollatorV2(node_position_mode="reset")
    col_default = GraphCollatorV2()   # node_position_mode omitted -> default "reset"
    out_reset = col_reset([item])
    out_default = col_default([item])
    assert torch.equal(out_reset["position_ids"], out_default["position_ids"])

    # And the historical formula, reproduced by hand: prompt packed last,
    # every node's positions are its own arange(len) with no offset.
    prompt_idx = item["prompt_node"]
    order = [j for j in range(item["num_nodes"]) if j != prompt_idx] + [prompt_idx]
    expected = torch.cat([torch.arange(len(item["input_ids"][j])) for j in order])
    L = out_reset["position_ids"].shape[1]
    assert torch.equal(out_reset["position_ids"][0, :expected.shape[0]], expected)
    assert (out_reset["position_ids"][0, expected.shape[0]:] == 0).all()  # padding


# ── 2. Backward compatibility (single-node graph) ───────────────────────────

def test_single_node_graph_is_plain_arange_under_both_modes():
    item = {
        "num_nodes": 1, "prompt_node": 0, "edges": [],
        "input_ids": [[10, 11, 12, 13, 14]],
        "shortest_path_dists": torch.zeros(1, 1, dtype=torch.long),
    }
    for mode in ("reset", "spd_depth"):
        col = GraphCollatorV2(node_position_mode=mode, max_spd=8)
        out = col([item])
        expected = torch.arange(5)
        assert torch.equal(out["position_ids"][0, :5], expected), mode


# ── 3. Permutation equivariance (end-to-end model forward) ──────────────────

def _spd_matrix(N, edges):
    """Real undirected BFS shortest-path distances (unreachable -> 32767,
    the same sentinel data prep uses)."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    M = torch.full((N, N), 32767, dtype=torch.long)
    for i, lengths in nx.all_pairs_shortest_path_length(G):
        for j, d in lengths.items():
            M[i, j] = d
    return M


def _build_star_item(perm, prompt_tokens, prefix_tokens, canonical_edges, prompt_idx):
    """One 5-node graph (4 prefix + 1 prompt) relabeled under ``perm``
    (a list: canonical prefix index -> new prefix index; prompt index is
    fixed). Returns a TextGraph-shaped item with real SPD features."""
    N = len(prefix_tokens) + 1
    full_perm = list(perm) + [prompt_idx]     # canonical index -> new index
    new_input_ids = [None] * N
    for canon, new in enumerate(range(len(prefix_tokens))):
        new_input_ids[full_perm[canon]] = prefix_tokens[canon]
    new_input_ids[prompt_idx] = prompt_tokens
    new_edges = [(full_perm[u], full_perm[v]) for u, v in canonical_edges]
    return {
        "num_nodes": N, "prompt_node": prompt_idx, "edges": new_edges,
        "input_ids": new_input_ids,
        "shortest_path_dists": _spd_matrix(N, new_edges),
    }


def test_permutation_equivariance_of_spd_depth_positions():
    """Relabeling the 4 prefix nodes of the same underlying graph must not
    change the model's logits at the prompt node's token positions — the
    end-to-end test of the actual invariant, not just the position values."""
    torch.manual_seed(0)
    prompt_idx = 4
    prompt_tokens = [50, 51, 52, 53, 54, 55]
    prefix_tokens = [[10, 11, 12], [20, 21], [30, 31, 32, 33], [40, 41]]
    canonical_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, prompt_idx), (2, prompt_idx)]

    identity = list(range(4))
    shuffled = [2, 0, 3, 1]   # canonical prefix index i now lives at index shuffled[i]

    item_a = _build_star_item(identity, prompt_tokens, prefix_tokens, canonical_edges, prompt_idx)
    item_b = _build_star_item(shuffled, prompt_tokens, prefix_tokens, canonical_edges, prompt_idx)

    config = GTLMLlamaConfig(spd=True, max_spd=8, magnetic=False, laplacian=False,
                             rwse=False, rrwp=False, k_hop=0, graph_attn_impl="eager", **_BASE)
    model = GTLMLlamaForCausalLM(config).to(DEVICE).double().eval()

    collator = GraphCollatorV2(node_position_mode="spd_depth", max_spd=8)

    def run(item):
        batch = collator([item])
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"], node_ids=batch["node_ids"],
                prompt_node=batch["prompt_node"], num_nodes=batch["num_nodes"],
                shortest_path_dists=batch["shortest_path_dists"],
            )
        prompt_len = len(prompt_tokens)
        return out.logits[0, -prompt_len:, :]

    logits_a = run(item_a)
    logits_b = run(item_b)
    # Double precision on CPU: floating-point summation-order noise from the
    # different packing order is negligible (~1e-8 observed) but not exactly
    # zero, so this checks "equivariant to float64 precision," not bit-exact
    # equality — still 4+ orders of magnitude tighter than any real break in
    # the invariant would produce.
    assert torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-5), \
        f"max abs diff: {(logits_a - logits_b).abs().max().item()}"


def test_permutation_changes_positions_isomorphically_not_arbitrarily():
    """Sanity check on the fixture itself: the two builds are NOT
    byte-identical (the shuffle really did relabel nodes) — otherwise the
    equivariance test above would be vacuous."""
    prompt_idx = 4
    prompt_tokens = [50, 51, 52, 53, 54, 55]
    prefix_tokens = [[10, 11, 12], [20, 21], [30, 31, 32, 33], [40, 41]]
    canonical_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, prompt_idx), (2, prompt_idx)]
    identity = list(range(4))
    shuffled = [2, 0, 3, 1]
    item_a = _build_star_item(identity, prompt_tokens, prefix_tokens, canonical_edges, prompt_idx)
    item_b = _build_star_item(shuffled, prompt_tokens, prefix_tokens, canonical_edges, prompt_idx)
    assert item_a["input_ids"] != item_b["input_ids"]
    assert not torch.equal(item_a["shortest_path_dists"], item_b["shortest_path_dists"])
