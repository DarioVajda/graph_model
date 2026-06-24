"""
FlexAttention kernel core for graph-biased Llama attention.

This is the production home of the FlexAttention implementation. It was
developed and validated under ``src/models/flex_attn/`` (see that package's
README for the full benchmark/optimization log); the kernel-level pieces live
here so the shipped model does not depend on the benchmark suite.
``flex_attn/flex_core.py`` re-exports this module for backward compatibility.

Two pieces, mirroring the dense path's split:

  * **Structural mask -> BlockMask** (``build_block_mask``): the layer-independent
    causal + bidirectional-prefix + K-hop + padding + diagonal-guard rule, built
    once per batch as a ``mask_mod`` and compiled into a sparse ``BlockMask``.
    This is what lets flex *skip* fully-masked blocks (the compute win).

  * **Per-layer soft bias -> score_mod** (``make_score_mod``): instead of
    materializing the ``(B,H,L,L)`` token-level bias, we capture the small
    ``(B,H,N,N)`` node-level bias and gather ``node_bias[b,h,node[q],node[k]]``
    inside the kernel. This is the memory win, and it is agnostic to which bias
    types summed into ``node_bias`` (spd, magnetic, ...) — they all come for free.

The actual attention is ``torch.compile``'d ``flex_attention`` with
``enable_gqa=True`` (no ``repeat_kv``).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

# Compiled-callable cache, keyed by (dynamic, mode). flex_attention must be
# compiled to get the kernel fusion / sparsity speedup; eager flex is a slow
# fallback.
_FLEX_CACHE: dict = {}

# torch.compile mode for the flex kernels (#4). Autotuning (H100, 512x32) bought
# fwd 35.9->7.6 ms (4.7x) at k=2 and 83->46 ms at k=0 for a net fwd+bwd
# 1.47x / 1.18x, at ~320 s one-time compile per distinct shape — amortized by
# length bucketing (``bucket_len``) and persisted across processes by inductor's
# on-disk cache. ``-no-cudagraphs`` because the per-batch BlockMask tensors and
# fresh grad leaves are not CUDA-graph-safe. Pass ``mode=None`` explicitly for
# inductor's fast-compiling heuristic configs.
DEFAULT_COMPILE_MODE: Optional[str] = "max-autotune-no-cudagraphs"


def flex_block_size(k_hop: int, compile_mode: Optional[str] = DEFAULT_COMPILE_MODE) -> int:
    """The recommended BlockMask block size for a K-hop setting (#6).

    64 when ``k_hop > 0`` (smaller blocks fit scattered K-hop neighbourhoods
    more tightly, raising block sparsity — 1.3-1.6x on fwd *and* bwd), 128 when
    ``k_hop == 0`` (dense mask, nothing to skip; 64-wide tiles are ~20% slower
    on dense work).

    Compile-mode gate: a non-autotune inductor compile only emits 128-wide tiles
    (``BLOCK_M=128``) and the mask block size must be a multiple of that tile, so
    a 64-wide block fails to lower (``Q and KV block size must be divisible by
    BLOCK_M``). Only the ``max-autotune`` modes carry 64-tile configs. So when the
    compile mode is not autotune we round the recommendation up to 128 — the
    block size only affects sparsity, not correctness, so this stays valid (just
    less tightly blocked) instead of erroring at the first forward.
    """
    bs = 64 if k_hop > 0 else 128
    if bs < 128 and not (compile_mode or "").startswith("max-autotune"):
        bs = 128
    return bs


def get_flex_attention(dynamic: bool = False,
                       mode: Optional[str] = DEFAULT_COMPILE_MODE) -> Callable:
    """Compiled ``flex_attention``, cached per ``(dynamic, mode)``.

    ``mode`` is the ``torch.compile`` mode; see ``DEFAULT_COMPILE_MODE`` for
    the measured trade-off. Keying the cache on the mode keeps variants
    comparable within one process.
    """
    key = ("flex", bool(dynamic), mode)
    if key not in _FLEX_CACHE:
        _FLEX_CACHE[key] = torch.compile(flex_attention, dynamic=dynamic, mode=mode)
    return _FLEX_CACHE[key]


# ── Sequence-length alignment (REQUIRED for performance) ──────────────────────

def largest_block(block_size) -> int:
    """The padding granularity for an int or ``(Q_BLOCK, KV_BLOCK)`` block size.

    With rectangular blocks the sequence is padded to the *larger* of the two
    (both are powers of two, so the smaller always divides it).
    """
    if isinstance(block_size, (tuple, list)):
        q_b, kv_b = block_size
        if max(q_b, kv_b) % min(q_b, kv_b) != 0:
            raise ValueError(f"incompatible block sizes {block_size!r}")
        return max(q_b, kv_b)
    return block_size


def align_len(length: int, block_size: int = 128) -> int:
    """Round a sequence length up to a multiple of ``block_size`` (int or
    ``(Q_BLOCK, KV_BLOCK)`` — the larger of the two)."""
    block_size = largest_block(block_size)
    return ((length + block_size - 1) // block_size) * block_size


def bucket_len(length: int, block_size: int = 128, midpoints: bool = True) -> int:
    """Round a length up to a coarse bucket ladder: power-of-2 multiples of
    ``block_size`` (128, 256, 512, …), with 1.5× midpoints by default
    (384, 768, 1536, …) so padding waste is bounded at ~33% instead of ~100%.

    Why this exists (and why ``align_len`` alone is not enough): the compiled
    flex kernel guards on the q/k/v length AND the captured ``node_bias`` /
    ``node_ids`` shapes, and the ``_compile=True`` BlockMask builder pays a
    ~5-40 s dynamo compile per *distinct* L — so a training run over
    heterogeneous graphs should bucket its padded L (and N, e.g.
    ``bucket_len(N, 128)``) to this ladder, keeping both compile caches small
    and hit-rates high. Validated by ``run_sweep --recompile-probe``.
    """
    if length <= block_size:
        return block_size
    n_blocks = (length + block_size - 1) // block_size
    pow2 = 1 << (n_blocks - 1).bit_length()         # next power of 2 ≥ n_blocks
    if midpoints:
        mid = 3 * (pow2 // 4)                       # 1.5× the previous power of 2
        if n_blocks <= mid:
            return mid * block_size
    return pow2 * block_size


def default_len_buckets(length: int) -> int:
    """Default sequence-length ladder for flex training: multiples of 512 with
    1.5× midpoints (512, 1024, 1536, 2048, 3072, …). Coarser than the raw
    block-size ladder — small L is cheap, so the extra padding there is free, and
    fewer distinct L keep the per-shape compile/autotune count low."""
    return bucket_len(length, 512)


def default_node_buckets(n: int) -> int:
    """Default node-count ladder for flex training: pure powers of two floored at
    32 (32, 64, 128, 256, 512, 1024, …). Coarse on purpose — N drives the
    ``(B,H,N,N)`` bias shape that the compiled flex kernel guards on, so few
    distinct N values are what keep recompiles bounded; small N is tiny in
    absolute memory, so the headroom there is cheap."""
    return bucket_len(n, 32, midpoints=False)


def bucketize(value: int, spec) -> int:
    """Round ``value`` up to an allowed bucket. ``spec`` is one of:

      * ``None`` — no bucketing (returns ``value`` unchanged);
      * a callable ``f(value) -> bucket`` (e.g. :func:`default_len_buckets`);
      * a sorted iterable of allowed sizes — the smallest entry ``>= value``.

    Raises if a list ``spec`` has no entry large enough, or if the result is
    smaller than ``value`` (a malformed callable/list).
    """
    if spec is None:
        return value
    if callable(spec):
        out = spec(value)
    else:
        spec = list(spec)
        out = next((b for b in spec if b >= value), None)
        if out is None:
            raise ValueError(
                f"value {value} exceeds the largest bucket {spec[-1] if spec else None}; "
                f"widen the bucket list.")
    if out < value:
        raise ValueError(f"bucketize produced {out} < value {value} (bad spec).")
    return out


def pad_to_block(q, k, v, node_ids, pad_mask, block_size=128):
    """Pad q/k/v/node_ids/pad_mask so the sequence length is a multiple of
    ``block_size`` (an int, or a ``(Q_BLOCK, KV_BLOCK)`` tuple — padding uses
    the larger). Returns ``(q, k, v, node_ids, pad_mask, orig_len, padded_len)``.

    This is not optional tuning — flex_attention's Triton kernel hits a severe
    config cliff at non-``block_size``-aligned lengths (measured ~14× slowdown at
    L%128≠0 on A100). Padded key positions get ``pad_mask=0`` (masked out); padded
    query rows produce garbage that the caller slices off. ``node_ids`` is padded
    with 0 (any valid node id; the rows are discarded anyway).
    """
    import torch.nn.functional as F
    L = q.shape[2]
    Lb = align_len(L, block_size)
    pad = Lb - L
    if pad == 0:
        return q, k, v, node_ids, pad_mask, L, Lb
    q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad)); v = F.pad(v, (0, 0, 0, pad))
    node_ids = F.pad(node_ids, (0, pad), value=0)
    pad_mask = F.pad(pad_mask, (0, pad), value=0)
    return q, k, v, node_ids, pad_mask, L, Lb


# ── Structural mask → BlockMask ───────────────────────────────────────────────

def make_mask_mod(
    node_ids: torch.Tensor,        # (B, kv_len) int
    prompt_node: torch.Tensor,     # (B,) int
    pad_mask: torch.Tensor,        # (B, kv_len) {0,1}
    k_hop_mask: Optional[torch.Tensor],  # (B, N, N) bool or None
    k_hop: int,
    q_offset: int,                 # = kv_len - q_len (absolute position of query 0)
) -> Callable:
    """Build the ``mask_mod(b, h, q_idx, kv_idx) -> bool`` closure.

    Matches the dense ``build_dense_structural_mask`` element-for-element:
    causal, relaxed to bidirectional between two prefix tokens; AND padding; AND
    the hard K-hop gate; OR the diagonal (NaN-guard so no query row is fully
    masked).
    """
    use_khop = k_hop > 0 and k_hop_mask is not None

    def mask_mod(b, h, q_idx, kv_idx):
        q_pos = q_idx + q_offset
        causal = kv_idx <= q_pos
        nq = node_ids[b, q_idx]
        nk = node_ids[b, kv_idx]
        is_prefix = (nq != prompt_node[b]) & (nk != prompt_node[b])
        allowed = causal | is_prefix
        allowed = allowed & (pad_mask[b, kv_idx] != 0)
        if use_khop:
            allowed = allowed & k_hop_mask[b, nq, nk]
        return allowed | (kv_idx == q_pos)

    return mask_mod


def build_block_mask(
    node_ids: torch.Tensor,
    prompt_node: torch.Tensor,
    pad_mask: torch.Tensor,
    k_hop_mask: Optional[torch.Tensor],
    k_hop: int,
    q_len: int,
    kv_len: int,
    *,
    block_size=128,
    device: Optional[torch.device] = None,
    compile_builder="auto",
    compile_threshold: int = 8192,
) -> BlockMask:
    """Build a (B, broadcast-over-H, q_len, kv_len) sparse ``BlockMask`` once per batch.

    ``block_size`` is an int or a ``(Q_BLOCK, KV_BLOCK)`` tuple (#6 — smaller KV
    blocks fit scattered K-hop neighbourhoods more tightly, raising block
    sparsity at small tokens-per-node, at some per-block kernel cost).

    ``compile_builder`` selects how ``create_block_mask`` evaluates ``mask_mod``:

      * ``False`` (eager): materializes O(L²) intermediates over the lattice —
        fast (tens of ms) and fine at small L, but the memory blows up
        quadratically (~48 GB at L≈70k, OOM with the extra K-hop gather).
      * ``True`` (``_compile=True``): evaluates ``mask_mod`` block-wise in a
        fused kernel — O(1) memory (~1 GB even at L≈70k) at the cost of a
        one-time ~6 s builder compile per distinct L. The resulting BlockMask
        and the attention kernel run at identical speed (verified).
      * ``"auto"`` (default): compile only when ``max(q_len,kv_len) >
        compile_threshold``, so small batches stay fast and large ones don't OOM.

    (Note: this uses the official ``_compile`` arg, *not* a ``torch.compile``
    wrapper around the builder — the latter pollutes the dynamo cache and slows
    the attention kernel; ``_compile`` does not.)
    """
    device = device or node_ids.device
    B = node_ids.shape[0]
    if compile_builder == "auto":
        compile_builder = max(q_len, kv_len) > compile_threshold
    mask_mod = make_mask_mod(
        node_ids, prompt_node, pad_mask, k_hop_mask, k_hop,
        q_offset=kv_len - q_len,
    )
    if isinstance(block_size, list):
        block_size = tuple(block_size)
    return create_block_mask(
        mask_mod, B, None, q_len, kv_len,
        device=device, BLOCK_SIZE=block_size,
        _compile=bool(compile_builder),
    )


# ── Per-layer soft bias → score_mod ───────────────────────────────────────────

def make_score_mod(
    node_bias: Optional[torch.Tensor],   # (B, H, N, N) float, or None
    node_ids: torch.Tensor,              # (B, kv_len) int
    q_offset: int = 0,
) -> Optional[Callable]:
    """Build the ``score_mod(score, b, h, q_idx, kv_idx)`` closure that gathers
    the token-expanded node bias on the fly. Returns None if there is no bias."""
    if node_bias is None:
        return None

    def score_mod(score, b, h, q_idx, kv_idx):
        nq = node_ids[b, q_idx + q_offset]
        nk = node_ids[b, kv_idx]
        return score + node_bias[b, h, nq, nk]

    return score_mod


# ── The attention call ────────────────────────────────────────────────────────

def flex_attention_forward(
    query: torch.Tensor,         # (B, H, q_len, d)
    key: torch.Tensor,           # (B, Hkv, kv_len, d)
    value: torch.Tensor,         # (B, Hkv, kv_len, d)
    *,
    block_mask: BlockMask,
    score_mod: Optional[Callable],
    scaling: float,
    enable_gqa: bool = True,
    dynamic: bool = False,
    compile_mode: Optional[str] = DEFAULT_COMPILE_MODE,
    return_lse: bool = False,
):
    """Run compiled ``flex_attention`` with the prebuilt BlockMask and score_mod.

    Returns ``attn_output`` (B, H, q_len, d) — or ``(attn_output, lse)`` if
    ``return_lse``.

    The Triton kernel's tile sizes must *divide* the BlockMask's block sizes
    (torch 2.6 lowering). Inductor's default mode generates a single 128-tile
    config and hard-errors on smaller mask blocks; the ``max-autotune`` modes
    carry 64-tile candidates and simply skip the incompatible ones — so
    sub-128 / rectangular blocks (#6) require an autotune compile mode.
    """
    q_bs, kv_bs = block_mask.BLOCK_SIZE
    if (q_bs < 128 or kv_bs < 128) and not (compile_mode or "").startswith("max-autotune"):
        raise ValueError(
            f"BlockMask block size {(q_bs, kv_bs)}: sub-128 blocks need a "
            f"max-autotune compile mode (got {compile_mode!r}) — inductor's "
            "default config only generates 128-tile kernels, which must divide "
            "the mask block size.")
    fa = get_flex_attention(dynamic=dynamic, mode=compile_mode)
    return fa(
        query, key, value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
    )
