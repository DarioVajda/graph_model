"""
Backend dispatch for GTLM attention.

This module owns *how the graph bias enters the attention score* for each
backend, and nothing about graph structure itself (that lives in
:mod:`src.models.structural_mask`).

Two ways the same logic is reached:

  * **Strategy A (legacy in-forward dispatch):** :func:`graph_attention_dispatch`
    and the flex seams are called directly from an overridden attention
    ``forward`` that received the structural mask / bias as arguments. This is
    what the v2 ``GTLMLlamaAttention.forward`` still uses.

  * **Strategy B (registered attention functions):** :func:`gtlm_eager` and
    :func:`gtlm_flex` match HuggingFace's attention-interface signature and are
    registered into ``ALL_ATTENTION_FUNCTIONS``. The backbone's
    *stock* attention ``forward`` does q/k/v projection + RoPE + cache, then calls
    the one named by ``config._attn_implementation``; our function reads the
    per-layer ``module.graph_bias`` and the per-batch ``module._graph_ctx`` and
    applies the structural mask + soft bias. This is what new backbone adapters
    use (no ``forward`` override).

Both paths share the same numerical core: the ``gtlm_eager`` dense function
computes the token-expanded bias and then delegates to
:func:`graph_attention_dispatch`, so there is a single source of truth for the
dense (eager) math.

Backends:
  * ``eager`` — add the soft bias onto the dense structural mask and delegate to
    HuggingFace's own ``eager_attention_forward`` (no softmax reimplementation).
    The fused SDPA kernels are intentionally not offered: a custom dense bias
    makes flash ineligible and the mem-efficient kernel both buggy in backward
    (GQA + per-head bias) and pointless (the dense bias is already materialized),
    so it would only ever reduce to eager. The decode (q_len < kv_len) fallback
    of ``flex`` therefore routes here too.
  * ``flex`` — implemented over :mod:`src.models.flex_kernel`; keeps the soft
    bias at node level ``(B,H,N,N)`` and gathers it inside the kernel (no
    token-level expansion). This is the only path that actually accelerates over
    eager (sparse block masks), at the cost of compiling a kernel per shape.
"""

import torch
from typing import Callable, Optional

from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    repeat_kv,  # noqa: F401  (re-exported for the flex path / external reference impls)
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# NB: explicit ``from .flex_kernel import ...`` (not ``from . import flex_kernel``)
# so HF's source-bundler picks up flex_kernel.py for trust_remote_code checkpoints
# — its regex only matches the ``from .<mod> import`` form (see REFACTOR.md §3a).
from .flex_kernel import (
    DEFAULT_COMPILE_MODE,
    flex_block_size as _flex_block_size,
    build_block_mask as _build_block_mask,
    make_score_mod as _make_score_mod,
    flex_attention_forward as _flex_attention_forward,
)
from .structural_mask import expand_node_to_token_bias


GRAPH_ATTN_IMPLS = ("eager", "flex")

# Registered attention-function names (Strategy B). One per backend; the matching
# `gtlm_*` name is what a backbone adapter pins on `config._attn_implementation`.
GTLM_ATTN_FN_NAMES = {
    "eager": "gtlm_eager",
    "flex":  "gtlm_flex",
}


# ── Backend dispatch (Strategy-A core; shared by the gtlm_* dense functions) ─────

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
    Run dense (eager) attention for one layer, applying the shared structural
    mask plus this layer's (already token-expanded) soft bias.

    Returns ``(attn_output, attn_weights)`` exactly like the HF backends, with
    ``attn_output`` shaped ``(B, q, H, d)``.
    """
    if impl == "eager":
        attn_mask = structural_mask
        if token_soft_bias is not None:
            attn_mask = attn_mask + token_soft_bias              # (B, H, q, kv)
        return eager_attention_forward(
            module, query, key, value, attn_mask, scaling=scaling, dropout=dropout
        )

    raise ValueError(
        f"graph_attention_dispatch got impl='{impl}'. Only 'eager' routes here; "
        f"'flex' is handled by gtlm_flex / flex_attention_forward. "
        f"Expected one of {GRAPH_ATTN_IMPLS}."
    )


# ── FlexAttention seams ──────────────────────────────────────────────────────────
#
# Thin wrappers over src.models.flex_kernel. Built once per batch
# (``build_flex_block_mask``) and per layer (``make_soft_score_mod`` +
# ``flex_attention_forward``).

def flex_block_size(k_hop: int, compile_mode=DEFAULT_COMPILE_MODE) -> int:
    """Recommended BlockMask block size for a K-hop setting: 64 when k>0 (#6),
    else 128 — rounded up to 128 for non-autotune compile modes (which can't
    lower sub-128 blocks). Re-exported from :mod:`flex_kernel`."""
    return _flex_block_size(k_hop, compile_mode)


def build_flex_block_mask(
    node_ids:    torch.Tensor,            # (B, kv_len) int
    prompt_node: torch.Tensor,            # (B,)
    pad_mask:    torch.Tensor,            # (B, kv_len) {0,1}
    k_hop_mask:  Optional[torch.Tensor],  # (B, N, N) bool, or None
    k_hop:       int,
    q_len:       int,
    kv_len:      int,
    *,
    block_size=128,
    device: Optional[torch.device] = None,
):
    """Build a sparse ``BlockMask`` expressing causal + bidirectional-prefix +
    K-hop + padding as block structure (the speedup over dense masks). One per
    batch, shared by every layer. Delegates to :func:`flex_kernel.build_block_mask`.
    """
    return _build_block_mask(
        node_ids, prompt_node, pad_mask, k_hop_mask, k_hop, q_len, kv_len,
        block_size=block_size, device=device,
    )


def make_soft_score_mod(
    node_bias: Optional[torch.Tensor],   # (B, H, N, N) float, or None
    node_ids:  torch.Tensor,             # (B, kv_len) int
    q_offset:  int = 0,
) -> Optional[Callable]:
    """Return a ``score_mod(score, b, h, q_idx, kv_idx)`` closure that gathers
    this layer's node-level soft bias (indexing via ``node_ids``) and adds it to
    the attention score — no token-level ``(B,H,L,L)`` expansion. ``None`` when
    there is no bias. Delegates to :func:`flex_kernel.make_score_mod`.
    """
    return _make_score_mod(node_bias, node_ids, q_offset=q_offset)


def flex_attention_forward(
    query, key, value,
    *,
    block_mask,
    score_mod,
    scaling: float,
    enable_gqa: bool = True,
    compile_mode: Optional[str] = DEFAULT_COMPILE_MODE,
):
    """Run compiled ``flex_attention`` with the prebuilt BlockMask and score_mod
    (``enable_gqa=True`` for grouped-query attention).

    Returns ``(attn_output, attn_weights)`` shaped like the other backends:
    the kernel yields ``(B, H, q, d)``; we transpose to ``(B, q, H, d)`` so the
    caller's shared ``reshape(B, q, -1)`` works for every backend, and
    ``attn_weights`` is ``None`` (flex never materializes the score matrix).
    """
    out = _flex_attention_forward(
        query, key, value,
        block_mask=block_mask, score_mod=score_mod, scaling=scaling,
        enable_gqa=enable_gqa, compile_mode=compile_mode,
    )                                                        # (B, H, q, d)
    out = out.transpose(1, 2).contiguous()                   # (B, q, H, d)
    return out, None


# ── Strategy B: registered attention functions ──────────────────────────────────
#
# HF's stock attention forward calls the function named by
# ``config._attn_implementation`` as
#   fn(module, query, key, value, attention_mask, dropout=..., scaling=..., **kwargs)
# *after* q/k/v projection + RoPE + cache update. These functions ignore the
# stock ``attention_mask`` (we substitute our own structural mask), read the
# per-batch ``module._graph_ctx`` and per-layer ``module.graph_bias``, and own
# only the score step.

def _require_ctx(module: torch.nn.Module) -> dict:
    ctx = getattr(module, "_graph_ctx", None)
    if ctx is None:
        raise RuntimeError(
            "A gtlm_* attention function ran without a graph context on the "
            "attention module. Call the model through its GTLM ForCausalLM "
            "forward, which installs `_graph_ctx` on every attention module."
        )
    return ctx


def compute_node_bias(
    module: torch.nn.Module,
    ctx: dict,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Compute this layer's node-level soft bias ``(B, H, N, N)`` (or ``None``).

    Optionally gradient-checkpointed: the bias compute is milliseconds, but
    autograd would otherwise keep every layer's ``(B, N, N, ·)`` intermediates
    alive. Safe to recompute — in training the bias cache is never read, and
    the bias dropouts (eigvec/MLP/droppath) rely on ``checkpoint``'s default
    ``preserve_rng_state=True`` to replay identical masks in the recompute.
    """
    feats = ctx.features

    def _bias():
        return module.graph_bias(
            dtype=dtype,
            device=device,
            num_nodes=ctx.num_nodes,
            spd=feats["spd"],
            laplacian=feats["laplacian"],
            rwse=feats["rwse"],
            rrwp=feats["rrwp"],
            magnetic=feats["magnetic"],
            magnetic_keep=feats.get("magnetic_keep"),
            k_hop_mask=None,
            cache_dict=ctx.cache,
        )

    if (getattr(module.config, "checkpoint_graph_bias", False)
            and module.training and torch.is_grad_enabled()):
        node_bias = torch.utils.checkpoint.checkpoint(_bias, use_reentrant=False)
    else:
        node_bias = _bias()

    # Shared bias (computed once per forward by the causal-LM mixin): added
    # OUTSIDE the checkpointed closure, so backward recompute never re-runs it —
    # each layer just reuses the one saved tensor.
    shared = ctx.shared_node_bias
    if shared is not None:
        shared = shared.to(dtype)
        node_bias = shared if node_bias is None else node_bias + shared
    return node_bias


def _gtlm_dense(
    impl: str,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float,
):
    """Shared body for the dense path (gtlm_eager): compute the bias, expand to
    token level, and delegate to :func:`graph_attention_dispatch`."""
    ctx = _require_ctx(module)
    q_len = query.shape[2]
    kv_len = key.shape[2]
    node_bias = compute_node_bias(module, ctx, query.dtype, query.device)
    token_soft_bias = (
        expand_node_to_token_bias(node_bias, ctx.node_ids, q_len, kv_len)
        if node_bias is not None else None
    )
    return graph_attention_dispatch(
        impl, module, query, key, value,
        ctx.structural_mask, token_soft_bias, scaling, dropout,
    )


def gtlm_eager(module, query, key, value, attention_mask=None, *, scaling, dropout=0.0, **kwargs):
    return _gtlm_dense("eager", module, query, key, value, scaling, dropout)


def gtlm_flex(module, query, key, value, attention_mask=None, *, scaling, dropout=0.0, **kwargs):
    ctx = _require_ctx(module)
    node_bias = compute_node_bias(module, ctx, query.dtype, query.device)
    score_mod = make_soft_score_mod(node_bias, ctx.node_ids_flex)
    return flex_attention_forward(
        query, key, value,
        block_mask=ctx.block_mask,
        score_mod=score_mod,
        scaling=scaling,
        enable_gqa=True,
        compile_mode=module.config.flex_compile_mode,
    )


def register_gtlm_attention_functions() -> None:
    """Register the gtlm_* functions into HF's ``ALL_ATTENTION_FUNCTIONS`` so a
    backbone can select one via ``config._attn_implementation``. Idempotent.

    NB: set ``config._attn_implementation`` to these names *directly* (as the
    config mixin does); do not route a custom name through the public
    ``attn_implementation=`` kwarg, which validates against the built-in set.
    """
    ALL_ATTENTION_FUNCTIONS["gtlm_eager"] = gtlm_eager
    ALL_ATTENTION_FUNCTIONS["gtlm_flex"] = gtlm_flex


# Register at import time. The package __init__ imports this module, so importing
# `src.models` is enough to make the gtlm_* implementations available.
register_gtlm_attention_functions()
