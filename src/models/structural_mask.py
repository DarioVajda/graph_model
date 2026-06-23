"""
Shared graph structural mask + node→token bias expansion.

These two functions are fully backbone-agnostic (pure tensor ops, no
``transformers`` import) and are reused by every backend. They were split out of
the old ``graph_attention_v2.py``, which mixed this structural concern with
backend dispatch.

  1. :func:`build_dense_structural_mask` — the layer-independent hard structure
     (causal + bidirectional-prefix relaxation + K-hop gate + padding), built
     once per batch and shared by every layer.
  2. :func:`expand_node_to_token_bias` — lift a per-layer node-level soft bias
     ``(B, H, N, N)`` to token level ``(B, H, q, kv)``.
"""

import torch
from typing import Optional


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
