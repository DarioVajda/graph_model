"""
GPU parity tests for the FlexAttention backend on the Gemma-3 backbone.

The BLOOM probe wires eager only (its hand-written attention never dispatches to
the registered ``gtlm_flex``). Gemma-3 is Strategy B like Llama, so flex *is*
reachable — this module is what makes that claim testable rather than assumed, and
it is the only thing standing between ``GTLMGemma3ForCausalLM`` and being wired
with both backends in ``src/experiments/graphqa/config.py``.

Mirrors ``test_flex_attention.py`` (same harness, same tolerances, same
eager-vs-flex-on-one-padded-batch strategy); see that module for the rationale.
Gemma-3-specific: the padded bucket L=128 stays inside ``sliding_window``, so the
dropped window is not a confound here — this isolates flex-vs-dense.
"""

import pytest
import torch

from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.models import GTLMGemma3Config, GTLMGemma3ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="FlexAttention needs CUDA")

DEVICE = torch.device("cuda")
DTYPE = torch.float32

_BASE = dict(
    hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, intermediate_size=256, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, attention_dropout=0.0, _attn_implementation="eager",
    # pattern=2 over 2 layers => one sliding + one global layer, both bases live.
    sliding_window=512, sliding_window_pattern=2, query_pre_attn_scalar=32,
    rope_theta=1_000_000.0, rope_local_base_freq=10_000.0,
)
_BIAS = dict(spd=True, max_spd=8, magnetic=True, magnetic_dim=8)

_COLLATE = dict(pad_token_id=0, pad_to_block=True, block_size=128,
                len_buckets=[128], node_buckets=[16])

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
    cfg = GTLMGemma3Config(
        k_hop=k_hop, graph_attn_impl=impl, checkpoint_graph_bias=checkpoint_bias,
        flex_block_size=128, flex_compile_mode="default", **bias, **_BASE,
    )
    return GTLMGemma3ForCausalLM(cfg).to(DEVICE).to(DTYPE)


def _pair(k_hop, bias_name, checkpoint_bias=True):
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
    # guard against a silent fallback making this a trivial eager-vs-eager check
    ctx = flex.model.layers[0].self_attn._graph_ctx
    assert flex.config._attn_implementation == "gtlm_flex" and ctx.block_mask is not None, \
        "flex path did not run"

    mask = batch["attention_mask"].bool()
    diff = (oe.logits[mask] - of.logits[mask]).abs().max().item()
    assert diff < LOGIT_TOL, f"[{bias_name}, k={k_hop}] flex-vs-eager logit diff {diff}"


# ── backward bias-grad parity ────────────────────────────────────────────────

@pytest.mark.parametrize("k_hop", [0, 2])
def test_flex_bias_grad_parity(k_hop):
    eager, flex, batch = _pair(k_hop, "all", checkpoint_bias=False)
    eager.train(); flex.train()

    eager(**batch).loss.backward()
    flex(**batch).loss.backward()

    eg = dict(eager.named_parameters())
    compared = 0
    for name, p in flex.named_parameters():
        if "graph_bias" not in name or p.grad is None:
            continue
        ge, gf = eg[name].grad, p.grad
        assert ge is not None, f"eager has no grad for {name}"
        assert torch.allclose(ge, gf, rtol=GRAD_RTOL, atol=GRAD_ATOL), \
            f"[k={k_hop}] grad mismatch on {name}: max|diff|={(ge - gf).abs().max().item():.3e}"
        compared += 1
    assert compared, "no graph-bias gradients were compared"


# ── bias checkpointing is transparent under flex ─────────────────────────────

def test_flex_bias_checkpointing_parity():
    on = _build("flex", 2, "all", checkpoint_bias=True).train()
    off = _build("flex", 2, "all", checkpoint_bias=False).train()
    off.load_state_dict(on.state_dict())
    batch = _to_device(GraphCollatorV2(k_hop=2, **_COLLATE)(_make_items()))

    loss_on = on(**batch).loss
    loss_off = off(**batch).loss
    assert torch.allclose(loss_on, loss_off, rtol=1e-5, atol=1e-6)

    loss_on.backward(); loss_off.backward()
    named_off = dict(off.named_parameters())
    for name, p in on.named_parameters():
        if "graph_bias" in name and p.grad is not None:
            assert torch.allclose(p.grad, named_off[name].grad,
                                  rtol=GRAD_RTOL, atol=GRAD_ATOL), name


# ── generation falls back to the dense decode path ───────────────────────────

def test_flex_generation_runs_through_dense_decode():
    """Flex serves ``q_len == kv_len``; incremental decode drops to eager. On this
    backbone the fallback must also survive the adapter's DynamicCache hook."""
    from transformers.cache_utils import DynamicCache

    model = _build("flex", 0, "all").eval()
    batch = _to_device(GraphCollatorV2(k_hop=0, **_COLLATE)(_make_items()))
    batch.pop("labels", None)
    out = model.generate(**batch, max_new_tokens=4, do_sample=False)
    assert out.shape[1] == batch["input_ids"].shape[1] + 4

    seen = []
    model.model.register_forward_pre_hook(
        lambda m, a, kw: seen.append(type(kw.get("past_key_values"))), with_kwargs=True)
    model.generate(**batch, max_new_tokens=2, do_sample=False)
    assert seen and all(t is DynamicCache for t in seen), seen
