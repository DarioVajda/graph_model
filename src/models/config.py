"""
Backbone-agnostic graph config mixin.

A GTLM config for any backbone is ``class GTLMXConfig(GraphConfigMixin, XConfig)``
— this mixin contributes the flat graph-bias fields, the K-hop fields, and the
backend selector; the backbone config contributes everything else. The graph
fields are stored flat (not nested) so standard ``save_pretrained`` /
``from_pretrained`` round-trips them with no custom code, and so the config can be
handed directly to :class:`GraphAttentionBias` as its ``bias_config``.
"""

from typing import Optional


class GraphConfigMixin:
    def __init__(
        self,
        spd: bool = False,
        laplacian: bool = False,
        max_spd: int = 32,
        rwse: bool = False,
        rrwp: bool = False,
        max_rw_steps: int = 8,
        magnetic: bool = False,
        magnetic_shared: bool = False,
        magnetic_content: bool = False,
        magnetic_dim: int = 32,
        magnetic_content_dim: int = 128,
        magnetic_q: float = 0.25,
        k_hop: int = 0,
        k_hop_directed: bool = False,
        graph_attn_impl: str = "eager",
        checkpoint_graph_bias: bool = True,
        flex_compile_mode: str = "max-autotune-no-cudagraphs",
        flex_block_size: Optional[int] = None,
        flex_cache_size_limit: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spd = spd
        self.laplacian = laplacian
        self.max_spd = max_spd
        self.rwse = rwse
        self.rrwp = rrwp
        self.max_rw_steps = max_rw_steps
        self.magnetic = magnetic
        self.magnetic_shared = magnetic_shared
        # Content-conditioned magnetic bias (per-layer). Reuses the magnetic
        # spectral machinery and consumes the same magnetic_V / magnetic_lambdas
        # features, plus the live hidden states (see MagneticContentBias). Its
        # down-projection width is magnetic_content_dim (d_proj).
        self.magnetic_content = magnetic_content
        self.magnetic_dim = magnetic_dim
        self.magnetic_content_dim = magnetic_content_dim
        self.magnetic_q = magnetic_q
        self.k_hop = k_hop
        self.k_hop_directed = k_hop_directed
        self.graph_attn_impl = graph_attn_impl
        # Recompute the per-layer bias modules in backward instead of saving
        # their (B,N,N,·) intermediates — ~40 GB at N=2048 for ms of recompute.
        # Training-only (eval/generation paths are unaffected by the flag).
        self.checkpoint_graph_bias = checkpoint_graph_bias
        # FlexAttention knobs (only used when graph_attn_impl == "flex").
        #   flex_compile_mode: torch.compile mode for the flex kernel; the
        #     autotune default buys ~1.47x step for a ~320s one-time compile per
        #     shape. Pass "default" (or None) for inductor's fast heuristics.
        #   flex_block_size: BlockMask block size; None uses the K-hop gate
        #     (64 when k_hop>0, else 128), an int overrides it.
        #   flex_cache_size_limit: torch._dynamo cache_size_limit to raise to when
        #     the flex path runs. Each distinct (L, N) shape is a separate compiled
        #     kernel; dynamo's default limit of 8 would silently fall back to eager
        #     past 8 distinct shapes. Set this above the number of (L, N) buckets a
        #     run actually hits (collator len_buckets x node_buckets).
        self.flex_compile_mode = flex_compile_mode
        self.flex_block_size = flex_block_size
        self.flex_cache_size_limit = flex_cache_size_limit
