"""
Mask coverage / density analysis.

The single most important reframe of this whole study (see the design notes):
flex's realized speedup tracks **block-level** density, not **element-level**
density, because the kernel can only skip a 128×128 block when it is *entirely*
masked.

  * element_density = (# allowed (q,k) pairs) / (# pairs)        ← theoretical ceiling
  * block_density   = (# non-fully-masked blocks) / (# blocks)   ← what flex delivers

Expected attention speedup from sparsity ≈ 1 / block_density (it runs that
fraction of blocks). A mask that is 75% element-sparse but whose allowed pairs
are scattered across blocks can still have block_density ≈ 1.0 → ~no speedup —
which is exactly the small-tokens-per-node regime we want to expose.

The mask rule mirrors :func:`build_dense_structural_mask` exactly (causal +
bidirectional-prefix + K-hop + padding + diagonal NaN-guard) but is evaluated in
chunked tiles so it never materializes the full ``L×L`` mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Density:
    element_density: float          # fraction of allowed (q, k) pairs
    block_density: float            # fraction of non-fully-masked blocks
    block_size: object              # int, or (Q_BLOCK, KV_BLOCK)
    expected_speedup: float         # 1 / block_density (attention-only, ideal)

    def as_dict(self) -> dict:
        bs = self.block_size
        return {
            "block_size": list(bs) if isinstance(bs, (tuple, list)) else bs,
            "element_density": self.element_density,         # fraction of allowed token pairs
            "element_sparsity": 1.0 - self.element_density,  # token-level sparsity
            "block_density": self.block_density,             # fraction of computed blocks
            "block_sparsity": 1.0 - self.block_density,      # block-level sparsity (what flex skips)
            "expected_attn_speedup": self.expected_speedup,  # ~1 / block_density
        }


def _allowed_tile(
    node_ids: torch.Tensor,      # (B, L)
    prompt_node: torch.Tensor,   # (B,)
    pad_mask: torch.Tensor,      # (B, L)
    k_hop_mask: Optional[torch.Tensor],
    k_hop: int,
    q0: int, q1: int,
    k0: int, k1: int,
) -> torch.Tensor:
    """Boolean ``(B, q1-q0, k1-k0)`` 'allowed' tile, matching the dense rule."""
    B = node_ids.shape[0]
    device = node_ids.device
    node_q = node_ids[:, q0:q1]                                   # (B, q)
    node_k = node_ids[:, k0:k1]                                   # (B, k)
    prompt = prompt_node.view(B, 1).to(device)

    q_pos = torch.arange(q0, q1, device=device).view(1, -1, 1)   # (1, q, 1)
    k_pos = torch.arange(k0, k1, device=device).view(1, 1, -1)   # (1, 1, k)
    causal = (k_pos <= q_pos)                                     # (1, q, k)

    is_prefix_q = (node_q != prompt).unsqueeze(2)                # (B, q, 1)
    is_prefix_k = (node_k != prompt).unsqueeze(1)                # (B, 1, k)
    allowed = causal | (is_prefix_q & is_prefix_k)               # (B, q, k)

    allowed = allowed & pad_mask[:, k0:k1].bool().unsqueeze(1)

    if k_hop > 0 and k_hop_mask is not None:
        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        nq = node_q.view(B, -1, 1)
        nk = node_k.view(B, 1, -1)
        allowed = allowed & k_hop_mask[b_idx, nq, nk]

    allowed = allowed | (k_pos == q_pos)                         # diagonal NaN-guard
    return allowed


@torch.no_grad()
def compute_density(
    node_ids: torch.Tensor,
    prompt_node: torch.Tensor,
    pad_mask: torch.Tensor,
    k_hop_mask: Optional[torch.Tensor],
    k_hop: int,
    block_size=128,
    chunk: int = 4096,
) -> Density:
    """Chunked element- and block-level density for a packed batch.

    ``block_size`` is an int or a ``(Q_BLOCK, KV_BLOCK)`` tuple. ``chunk``
    bounds peak memory at ``B * chunk^2`` bools; it is snapped to a multiple of
    the larger block dimension so block reductions tile cleanly.
    """
    B, L = node_ids.shape
    device = node_ids.device
    q_bs, kv_bs = (tuple(block_size) if isinstance(block_size, (tuple, list))
                   else (block_size, block_size))
    bs_max = max(q_bs, kv_bs)
    chunk = max(bs_max, (chunk // bs_max) * bs_max)

    n_qb = (L + q_bs - 1) // q_bs
    n_kb = (L + kv_bs - 1) // kv_bs

    allowed_total = 0
    pair_total = B * L * L
    active_blocks = torch.zeros(B, n_qb, n_kb, dtype=torch.bool, device=device)

    for q0 in range(0, L, chunk):
        q1 = min(q0 + chunk, L)
        for k0 in range(0, L, chunk):
            k1 = min(k0 + chunk, L)
            tile = _allowed_tile(node_ids, prompt_node, pad_mask, k_hop_mask, k_hop,
                                 q0, q1, k0, k1)                 # (B, q, k) bool
            allowed_total += int(tile.sum().item())

            # Reduce this tile into its constituent blocks via padded reshape.
            qh, kh = q1 - q0, k1 - k0
            qpad = (q_bs - qh % q_bs) % q_bs
            kpad = (kv_bs - kh % kv_bs) % kv_bs
            if qpad or kpad:
                tile = torch.nn.functional.pad(tile, (0, kpad, 0, qpad))
            tb = tile.view(B, (qh + qpad) // q_bs, q_bs,
                                (kh + kpad) // kv_bs, kv_bs)
            tile_active = tb.any(dim=4).any(dim=2)               # (B, qb, kb)
            qb0, kb0 = q0 // q_bs, k0 // kv_bs
            active_blocks[:, qb0:qb0 + tile_active.shape[1],
                             kb0:kb0 + tile_active.shape[2]] |= tile_active

    element_density = allowed_total / pair_total
    block_density = float(active_blocks.float().mean().item())
    speedup = (1.0 / block_density) if block_density > 0 else float("inf")
    return Density(
        element_density=element_density,
        block_density=block_density,
        block_size=block_size,
        expected_speedup=speedup,
    )


if __name__ == "__main__":
    import torch as _t
    from src.models.flex_attn.inputs import GraphSpec, make_attention_inputs

    dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
    print(f"{'spec':>22} | {'elem':>7} | {'block':>7} | {'~speedup':>9}")
    for k in (0, 1, 2, 4):
        spec = GraphSpec(n_nodes=128, tokens_per_node=8, k_hop=k, ordering="rcm")
        ai, meta = make_attention_inputs(spec, 1, 32, 8, 64, dev)
        d = compute_density(ai["node_ids"], ai["prompt_node"], ai["pad_mask"],
                            ai["k_hop_mask"], k, block_size=128)
        print(f"  n128 t8 rcm k={k:<2} L={meta['seq_len']:>5} | "
              f"{d.element_density:6.3f} | {d.block_density:6.3f} | {d.expected_speedup:8.2f}x")
