"""
Attention plumbing for the v2 GTLM-Llama model.

This module isolates everything about *how graph structure enters attention* so
that the modeling file stays small and the model stack never reimplements any
HuggingFace internals.

Two orthogonal pieces of graph structure are handled here:

  1. Shared structural mask (layer-independent): causal masking, the
     bidirectional-prefix relaxation, the hard K-hop gate, and padding.  Built
     once per batch and reused by every layer.

  2. Per-layer soft bias (trainable, different per layer): produced by
     ``GraphAttentionBias`` as a node-level ``(B, H, N, N)`` tensor and expanded
     to token level here.

Backends:
  * ``eager`` / ``sdpa`` — fully implemented; we add the soft bias onto the dense
    structural mask and delegate the actual attention to HuggingFace's own
    ``eager_attention_forward`` / ``sdpa_attention_forward`` (no softmax/SDPA
    reimplementation).
  * ``flex`` — scaffolded.  The seams (``build_flex_block_mask`` for the hard
    structure, ``make_soft_score_mod`` for the additive bias, ``flex_attention_forward``
    for the call) are defined and raise ``NotImplementedError`` so the dispatch
    wiring is in place and the kernel can be dropped in later.
"""

import torch
from typing import Optional

from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    repeat_kv,  # noqa: F401  (re-exported for the future flex path)
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward


GRAPH_ATTN_IMPLS = ("eager", "sdpa", "flex")


# ── Shared structural mask ──────────────────────────────────────────────────────

def build_dense_structural_mask(
    node_ids:    torch.Tensor,            # (B, kv_len)  node index per kv token (pad → prompt node)
    prompt_node: torch.Tensor,            # (B,)
    pad_mask:    torch.Tensor,            # (B, kv_len)  1 for real tokens, 0 for padding
    k_hop_mask:  Optional[torch.Tensor],  # (B, N, N) bool, or None
    k_hop:       int,
    q_len:       int,
    kv_len:      int,
    dtype:       torch.dtype,
    device:      torch.device,
) -> torch.Tensor:
    """
    Build a ``(B, 1, q_len, kv_len)`` additive float mask (0 = allowed,
    ``finfo.min`` = blocked) combining:

      * causal masking, relaxed to **bidirectional** attention between any two
        prefix tokens (tokens whose node is not the prompt node);
      * a hard **K-hop gate** when ``k_hop > 0`` (blocks pairs whose nodes are
        more than K hops apart, using the precomputed ``k_hop_mask``);
      * **padding** (no token may attend to a padded key).

    Handles both the full-sequence (training/prefill, ``q_len == kv_len``) and the
    incremental-decoding (``q_len < kv_len``) cases via the standard
    last-``q_len`` / last-``kv_len`` slicing.  The diagonal (same absolute
    position) is always left open as a NaN guard for otherwise-empty rows.
    """
    B = node_ids.shape[0]
    node_q = node_ids[:, -q_len:]                                    # (B, q)
    node_k = node_ids[:, -kv_len:]                                   # (B, kv)
    prompt = prompt_node.view(B, 1).to(node_ids.device)

    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).view(q_len, 1)   # (q, 1)
    k_pos = torch.arange(kv_len, device=device).view(1, kv_len)                  # (1, kv)

    causal     = (k_pos <= q_pos)                                   # (q, kv)
    is_prefix_q = (node_q != prompt).unsqueeze(2)                  # (B, q, 1)
    is_prefix_k = (node_k != prompt).unsqueeze(1)                  # (B, 1, kv)
    both_prefix = is_prefix_q & is_prefix_k                        # (B, q, kv)

    allowed = causal.unsqueeze(0) | both_prefix                    # (B, q, kv)

    pad_k = pad_mask[:, -kv_len:].bool().unsqueeze(1)              # (B, 1, kv)
    allowed = allowed & pad_k

    if k_hop > 0 and k_hop_mask is not None:
        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        nq = node_q.view(B, q_len, 1)
        nk = node_k.view(B, 1, kv_len)
        kh = k_hop_mask[b_idx, nq, nk]                             # (B, q, kv) bool
        allowed = allowed & kh

    diag = (k_pos == q_pos).unsqueeze(0)                          # (1, q, kv) NaN guard
    allowed = allowed | diag

    min_val = torch.finfo(dtype).min
    mask = torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    )
    return mask.unsqueeze(1)                                       # (B, 1, q, kv)


# ── Node → token soft-bias expansion ────────────────────────────────────────────

def expand_node_to_token_bias(
    node_bias: torch.Tensor,   # (B, H, N, N)
    node_ids:  torch.Tensor,   # (B, full_seq_len)
    q_len:     int,
    kv_len:    int,
) -> torch.Tensor:             # (B, H, q_len, kv_len)
    """Lift a node-level bias to token level: token pair (q, k) inherits the bias
    of node pair ``(node[q], node[k])``."""
    B, H = node_bias.shape[0], node_bias.shape[1]
    device = node_ids.device

    node_ids_q  = node_ids[:, -q_len:]
    node_ids_kv = node_ids[:, -kv_len:]

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1)
    q_idx = node_ids_q.view(B, 1, q_len, 1)
    k_idx = node_ids_kv.view(B, 1, 1, kv_len)
    return node_bias[b_idx, h_idx, q_idx, k_idx]                  # (B, H, q, kv)


# ── Backend dispatch ────────────────────────────────────────────────────────────

def graph_attention_dispatch(
    impl:           str,
    module:         torch.nn.Module,
    query:          torch.Tensor,            # (B, H, q, d)
    key:            torch.Tensor,            # (B, Hkv, kv, d)
    value:          torch.Tensor,            # (B, Hkv, kv, d)
    structural_mask: torch.Tensor,           # (B, 1, q, kv) additive float
    token_soft_bias: Optional[torch.Tensor], # (B, H, q, kv) additive float, or None
    scaling:        float,
    dropout:        float,
):
    """
    Run attention for one layer with the given backend, applying the shared
    structural mask plus this layer's (already token-expanded) soft bias.

    Returns ``(attn_output, attn_weights)`` exactly like the HF backends.
    """
    if impl in ("eager", "sdpa"):
        attn_mask = structural_mask
        if token_soft_bias is not None:
            attn_mask = attn_mask + token_soft_bias              # (B, H, q, kv)
        if impl == "eager":
            return eager_attention_forward(
                module, query, key, value, attn_mask, scaling=scaling, dropout=dropout
            )
        return sdpa_attention_forward(
            module, query, key, value, attn_mask,
            dropout=dropout, scaling=scaling, is_causal=False,
        )

    if impl == "flex":
        return flex_attention_forward(
            module, query, key, value, structural_mask, token_soft_bias,
            scaling=scaling, dropout=dropout,
        )

    raise ValueError(f"Unknown graph attention implementation '{impl}'. Expected one of {GRAPH_ATTN_IMPLS}.")


# ── FlexAttention scaffold (to be implemented) ──────────────────────────────────

def build_flex_block_mask(*args, **kwargs):
    """
    TODO(flex): build a ``torch.nn.attention.flex_attention.BlockMask`` from a
    ``mask_mod(b, h, q_idx, kv_idx)`` closure that captures ``node_ids``,
    ``prompt_node`` and ``k_hop_mask`` to express causal + bidirectional-prefix +
    K-hop + padding as sparse block structure (the speedup over dense masks).
    """
    raise NotImplementedError("FlexAttention block-mask builder is not implemented yet.")


def make_soft_score_mod(*args, **kwargs):
    """
    TODO(flex): return a ``score_mod(score, b, h, q_idx, kv_idx)`` closure that
    captures this layer's node-level soft bias and adds the token-expanded value
    (indexing via ``node_ids``) to the attention score.
    """
    raise NotImplementedError("FlexAttention score-mod builder is not implemented yet.")


def flex_attention_forward(
    module, query, key, value, structural_mask, token_soft_bias, *, scaling, dropout,
):
    """
    TODO(flex): call ``torch.compile``'d ``flex_attention`` with the BlockMask
    from :func:`build_flex_block_mask` and the per-layer score_mod from
    :func:`make_soft_score_mod` (``enable_gqa=True`` for grouped-query attention),
    returning ``(attn_output, None)`` shaped like the other backends.
    """
    raise NotImplementedError(
        "graph_attn_impl='flex' is scaffolded but not yet implemented. "
        "Use 'eager' or 'sdpa' for now."
    )
