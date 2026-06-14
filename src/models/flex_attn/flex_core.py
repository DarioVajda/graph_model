"""
FlexAttention core for graph-biased Llama attention — benchmark-facing shim.

The kernel implementation has been promoted to the production module
``src.models.flex_kernel`` (so the shipped model does not depend on this
benchmark package). This file now **re-exports** that module's public API for
backward compatibility with the benchmark suite, and keeps the two
benchmark-only helpers that don't belong in the model: ``dense_reference``
(a parity oracle) and the ``__main__`` self-test.

See ``src/models/flex_kernel.py`` for the kernel itself and this package's
README for the full benchmark/optimization log.
"""

from __future__ import annotations

from typing import Optional

import torch

# Re-export the promoted kernel API. Existing imports such as
# ``from src.models.flex_attn import flex_core`` then ``flex_core.build_block_mask``
# continue to work unchanged.
from src.models.flex_kernel import (  # noqa: F401
    DEFAULT_COMPILE_MODE,
    flex_block_size,
    get_flex_attention,
    largest_block,
    align_len,
    bucket_len,
    pad_to_block,
    make_mask_mod,
    build_block_mask,
    make_score_mod,
    flex_attention_forward,
)


# ── Dense reference (for parity checks) ───────────────────────────────────────

@torch.no_grad()
def dense_reference(
    query, key, value,
    *,
    node_ids, prompt_node, pad_mask, k_hop_mask, k_hop, node_bias, scaling,
):
    """Eager dense attention matching the model's ``eager`` path, for parity.

    Builds the same dense structural mask + token-expanded soft bias the
    sdpa/eager backend uses, then does a plain softmax attention. Reuses the
    real model functions so the reference can't silently diverge.
    """
    from src.models.graph_attention_v2 import (
        build_dense_structural_mask,
        expand_node_to_token_bias,
    )
    from transformers.models.llama.modeling_llama import repeat_kv

    B, H, q_len, d = query.shape
    kv_len = key.shape[2]
    mask = build_dense_structural_mask(
        node_ids=node_ids, prompt_node=prompt_node, pad_mask=pad_mask,
        k_hop_mask=k_hop_mask, k_hop=k_hop, q_len=q_len, kv_len=kv_len,
        dtype=query.dtype, device=query.device,
    )                                                            # (B,1,q,kv)
    attn_mask = mask
    if node_bias is not None:
        soft = expand_node_to_token_bias(node_bias, node_ids, q_len, kv_len)
        attn_mask = attn_mask + soft                             # (B,H,q,kv)

    n_rep = H // key.shape[1]
    k = repeat_kv(key, n_rep)
    v = repeat_kv(value, n_rep)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling + attn_mask
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, v)


# ── Self-test: flex vs dense parity + a basic block-mask sanity check ─────────

if __name__ == "__main__":
    from src.models.flex_attn.inputs import GraphSpec, make_attention_inputs

    assert torch.cuda.is_available(), "flex_attention needs CUDA"
    dev = torch.device("cuda")
    dt = torch.bfloat16
    H, Hkv, d = 8, 2, 64

    for k_hop in (0, 2):
        spec = GraphSpec(n_nodes=24, tokens_per_node=6, prompt_tokens=32,
                         k_hop=k_hop, ordering="rcm", magnetic_m=16)
        ai, meta = make_attention_inputs(spec, 2, H, Hkv, d, dev, dtype=dt)
        q, key, val = ai["query"], ai["key"], ai["value"]
        scaling = d ** -0.5

        bm = build_block_mask(
            ai["node_ids"], ai["prompt_node"], ai["pad_mask"], ai["k_hop_mask"],
            k_hop, ai["q_len"], ai["kv_len"], block_size=128, device=dev,
        )
        smod = make_score_mod(ai["node_bias"], ai["node_ids"])
        out = flex_attention_forward(
            q, key, val, block_mask=bm, score_mod=smod, scaling=scaling,
        )
        ref = dense_reference(
            q.detach(), key.detach(), val.detach(),
            node_ids=ai["node_ids"], prompt_node=ai["prompt_node"],
            pad_mask=ai["pad_mask"], k_hop_mask=ai["k_hop_mask"], k_hop=k_hop,
            node_bias=ai["node_bias"], scaling=scaling,
        )
        diff = (out.detach().float() - ref.float()).abs()
        rel = (diff.max() / ref.float().abs().max()).item()
        sparsity = bm.sparsity()  # % of blocks skipped
        print(f"k_hop={k_hop}: L={meta['seq_len']:>4} max|Δ|={diff.max().item():.4e} "
              f"rel={rel:.2e} block_sparsity={sparsity:.1f}%  "
              f"{'OK' if diff.max().item() < 5e-2 else 'FAIL'}")
