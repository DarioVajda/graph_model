"""
GPU parity tests for the FlexAttention backend of GTLM-Llama v2.

Flex requires CUDA + torch.compile, so the whole module is skipped without a GPU.
To stay fast we force ``flex_block_size=128`` and ``flex_compile_mode="default"``
(block size affects only sparsity, not correctness; default mode avoids the
~320s autotune) and pin a single small (L, N) bucket so the compiled kernels are
reused across tests.

Strategy: run the eager and flex backends on the *same* flex-padded batch and
compare on the real (unpadded) positions — padding is loss-neutral for eager
(see test_collator_bucketing), so this isolates the flex-vs-dense difference.

Covers:
  * forward logit parity (bias none/all, k=0/2);
  * backward bias-grad parity;
  * #7 bias-checkpointing parity (loss + grads, on vs off);
  * #9 decoder gradient-checkpointing parity under flex;
  * generation runs through the dense decode fallback (q_len < kv_len).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pytest
import torch

from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.utils.text_graph_dataset import TextGraphDataset
from src.models.modeling_gtlm_llama_v2 import GTLMLlamaConfig, GTLMLlamaForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="FlexAttention needs CUDA")

DEVICE = torch.device("cuda")
DTYPE = torch.float32   # tight parity; flex supports fp32 on CUDA

_BASE = dict(
    hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=256, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)
_BIAS = dict(spd=True, max_spd=8, magnetic=True, magnetic_dim=8)

# Fixed small buckets so every test shares one (L, N) shape.
_COLLATE = dict(pad_token_id=0, pad_to_block=True, block_size=128,
                len_buckets=[128], node_buckets=[16])

# Tolerances for fp32 flex (flash online softmax) vs eager (full softmax).
LOGIT_TOL = 3e-3
GRAD_RTOL, GRAD_ATOL = 3e-2, 1e-4


def _make_items(seed=0):
    torch.manual_seed(seed)
    specs = [
        ([3, 2, 4, 2], 3, [(0, 1), (1, 2), (2, 3), (0, 3)]),
        ([2, 3, 2], 0, [(0, 1), (1, 2)]),
    ]
    sd, rd, rw = 4, 4, 4
    items = []
    for tok_lens, prompt, edges in specs:
        N = len(tok_lens)
        item = {
            "num_nodes": N, "prompt_node": prompt, "edges": edges,
            "input_ids": [torch.randint(1, 256, (l,)).tolist() for l in tok_lens],
            "laplacian_coordinates": torch.randn(N, sd),
            "shortest_path_dists": torch.randint(0, 5, (N, N)),
            "rwse": torch.randn(N, rd),
            "rrwp": torch.randn(N, N, rw),
            "magnetic_V": torch.randn(N, N, 2),
            "magnetic_lambdas": torch.randn(N),
        }
        item["labels"] = torch.tensor(item["input_ids"][prompt], dtype=torch.long)
        items.append(item)
    return items


def _to_device(batch):
    out = {}
    for k, v in batch.items():
        if v is None:
            continue
        out[k] = v.to(device=DEVICE, dtype=DTYPE) if torch.is_floating_point(v) else v.to(DEVICE)
    return out


def _build(impl, k_hop, bias_name, checkpoint_bias=True):
    bias = _BIAS if bias_name == "all" else {}
    cfg = GTLMLlamaConfig(
        k_hop=k_hop, graph_attn_impl=impl, checkpoint_graph_bias=checkpoint_bias,
        flex_block_size=128, flex_compile_mode="default", **bias, **_BASE,
    )
    return GTLMLlamaForCausalLM(cfg).to(DEVICE).to(DTYPE)


def _pair(k_hop, bias_name, checkpoint_bias=True):
    """An (eager, flex) pair sharing identical weights, plus the padded batch."""
    eager = _build("eager", k_hop, bias_name, checkpoint_bias).eval()
    flex = _build("flex", k_hop, bias_name, checkpoint_bias).eval()
    flex.load_state_dict(eager.state_dict())
    batch = _to_device(GraphCollatorV2(k_hop=k_hop, **_COLLATE)(_make_items()))
    return eager, flex, batch


# ── forward parity ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("bias_name", ["none", "all"])
@pytest.mark.parametrize("k_hop", [0, 2])
def test_flex_matches_eager_forward(bias_name, k_hop):
    eager, flex, batch = _pair(k_hop, bias_name)
    with torch.no_grad():
        oe = eager(**batch)
        of = flex(**batch)
    mask = batch["attention_mask"].bool()
    diff = (oe.logits[mask] - of.logits[mask]).abs().max().item()
    assert diff < LOGIT_TOL, f"[{bias_name}, k={k_hop}] flex-vs-eager logit diff {diff}"


# ── backward bias-grad parity ────────────────────────────────────────────────

@pytest.mark.parametrize("k_hop", [0, 2])
def test_flex_bias_grad_parity(k_hop):
    # checkpoint off here to isolate flex-vs-eager (ckpt parity is its own test)
    eager, flex, batch = _pair(k_hop, "all", checkpoint_bias=False)
    eager.train(); flex.train()

    eager(**batch).loss.backward()
    flex(**batch).loss.backward()

    eg = dict(eager.named_parameters())
    worst = 0.0
    for name, p in flex.named_parameters():
        if "graph_bias" not in name or p.grad is None:
            continue
        ge, gf = eg[name].grad, p.grad
        assert torch.allclose(ge, gf, rtol=GRAD_RTOL, atol=GRAD_ATOL), \
            f"k={k_hop}: grad mismatch on {name} (max {(ge-gf).abs().max().item():.2e})"
        worst = max(worst, (ge - gf).abs().max().item())
    assert worst > 0.0  # sanity: we actually compared some grads


# ── #7 bias-checkpointing parity (flex) ──────────────────────────────────────

def test_flex_checkpoint_bias_parity():
    flex_on = _build("flex", 2, "all", checkpoint_bias=True).train()
    flex_off = _build("flex", 2, "all", checkpoint_bias=False).train()
    flex_off.load_state_dict(flex_on.state_dict())
    batch = _to_device(GraphCollatorV2(k_hop=2, **_COLLATE)(_make_items()))

    loss_on = flex_on(**batch).loss; loss_on.backward()
    loss_off = flex_off(**batch).loss; loss_off.backward()
    assert torch.allclose(loss_on, loss_off, rtol=1e-5, atol=1e-6)

    off = dict(flex_off.named_parameters())
    for name, p in flex_on.named_parameters():
        if "graph_bias" in name and p.grad is not None:
            assert torch.allclose(p.grad, off[name].grad, rtol=1e-4, atol=1e-6), name


# ── #9 decoder gradient-checkpointing parity (flex) ──────────────────────────

def test_flex_decoder_ckpt_parity():
    plain = _build("flex", 2, "all", checkpoint_bias=True).train()
    ckpt = _build("flex", 2, "all", checkpoint_bias=True)
    ckpt.load_state_dict(plain.state_dict())
    ckpt.config.use_cache = False
    ckpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ckpt.train()
    batch = _to_device(GraphCollatorV2(k_hop=2, **_COLLATE)(_make_items()))

    lp = plain(**batch).loss; lp.backward()
    lc = ckpt(**batch).loss; lc.backward()
    assert torch.allclose(lp, lc, rtol=1e-4, atol=1e-6), f"{lp.item()} vs {lc.item()}"

    cg = dict(ckpt.named_parameters())
    for name, p in plain.named_parameters():
        if "graph_bias" in name and p.grad is not None:
            assert torch.allclose(p.grad, cg[name].grad, rtol=1e-3, atol=1e-5), name


# ── permutation invariance through the dataset's RCM reorder ─────────────────

def _equivariance_items(rcm: bool):
    """Build collator-ready items from the same graphs, optionally RCM-reordered
    via the dataset. SPD is computed *after* reordering (so it exercises the
    dataset's feature/relabel consistency), and per-node tokens are keyed on the
    original node id so the two batches are exact permutations of each other.

    Only SPD + K-hop biases are used: they permute exactly (integer distances /
    booleans), unlike eigenvector features whose sign is basis-ambiguous on a
    recompute."""
    graphs = []
    for i in range(2):
        g = nx.barabasi_albert_graph(6 + i, 2, seed=i)
        nx.set_node_attributes(g, {nd: "x" for nd in g.nodes()}, "text")
        g.graph["prompt_node"] = i % g.number_of_nodes()
        graphs.append(g)

    ds = TextGraphDataset(graphs, rcm_ordering=rcm)
    ds.compute_shortest_path_distances(use_gpu=False)

    def toks(orig_id):                       # deterministic tokens per original node
        return [1 + (orig_id * 37 + 5) % 200, 1 + (orig_id * 91 + 11) % 200]

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
def test_flex_permutation_invariance_via_rcm(k_hop):
    """Reordering the prefix nodes (here via the dataset's RCM) must not change
    the model's output at the prompt-node tokens.

    The prompt node is always packed last, attends to the prefix bidirectionally
    via node-gathered bias with per-node position resets, so its representation —
    and thus its token logits — is invariant to prefix order. We assert exactly
    that: the logits at the prompt span are identical.

    We deliberately do NOT compare the cross-entropy loss. HF shifts labels by
    one, so the *first* prompt token would be predicted from the last prefix
    token (whichever node RCM places adjacent to the prompt) — an inherent
    teacher-forcing boundary effect, not a model difference. In practice the
    first prompt token is never a prediction target, so this never enters the
    real loss; here it would just be noise, so we compare logits, not loss."""
    cfg = GTLMLlamaConfig(
        k_hop=k_hop, graph_attn_impl="flex", checkpoint_graph_bias=False,
        flex_block_size=128, flex_compile_mode="default",
        spd=True, max_spd=8, **_BASE,
    )
    model = GTLMLlamaForCausalLM(cfg).to(DEVICE).to(DTYPE).eval()

    b_orig = _to_device(GraphCollatorV2(k_hop=k_hop, **_COLLATE)(_equivariance_items(False)))
    b_rcm = _to_device(GraphCollatorV2(k_hop=k_hop, **_COLLATE)(_equivariance_items(True)))

    with torch.no_grad():
        o_orig = model(**b_orig)
        o_rcm = model(**b_rcm)

    # prompt-span logits align position-wise (the prompt is packed last and its
    # span sits at the same absolute positions regardless of prefix order)
    m0 = b_orig["labels"] != -100
    m1 = b_rcm["labels"] != -100
    assert torch.equal(m0, m1)
    diff = (o_orig.logits[m0] - o_rcm.logits[m1]).abs().max().item()
    assert diff < LOGIT_TOL, f"k={k_hop}: prompt-logit diff under RCM reorder {diff}"


# ── generation: dense decode fallback ────────────────────────────────────────

def test_flex_generation_dense_fallback():
    flex = _build("flex", 2, "all").eval()
    batch = _to_device(GraphCollatorV2(k_hop=2, **_COLLATE)(_make_items()))
    gen_kwargs = {k: batch[k] for k in (
        "input_ids", "attention_mask", "node_ids", "prompt_node", "num_nodes",
        "shortest_path_dists", "laplacian_coordinates", "rwse", "rrwp",
        "magnetic_V", "magnetic_lambdas", "k_hop_mask") if k in batch}
    L = batch["input_ids"].shape[1]
    out = flex.generate(max_new_tokens=4, do_sample=False, **gen_kwargs)
    assert out.shape[1] == L + 4          # decode ran (q_len<kv_len fallback)
    assert torch.isfinite(flex.lm_head.weight).all()
