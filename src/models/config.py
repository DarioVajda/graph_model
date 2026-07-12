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
        magnetic_dim: int = 32,
        magnetic_q: float = 0.25,
        magnetic_eigvec_dropout: float = 0.0,
        magnetic_eigvec_shared_mask: bool = False,
        magnetic_mlp_dropout: float = 0.0,
        bias_droppath: float = 0.0,
        bias_dropout: float = 0.0,
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
        self.magnetic_dim = magnetic_dim
        self.magnetic_q = magnetic_q
        # Bias-path regularization (all training-only, 0.0/False = no-op):
        #   magnetic_eigvec_dropout — drop whole eigenvectors from MagneticBias;
        #   magnetic_eigvec_shared_mask — sample ONE eigvec keep-mask per forward
        #     (all layers see the same spectral truncation) instead of per-layer;
        #   magnetic_mlp_dropout    — dropout after MagneticBias' SiLUs;
        #   bias_droppath           — per-sample drop of a layer's summed bias;
        #   bias_dropout            — element-wise dropout on the summed bias.
        self.magnetic_eigvec_dropout = magnetic_eigvec_dropout
        self.magnetic_eigvec_shared_mask = magnetic_eigvec_shared_mask
        self.magnetic_mlp_dropout = magnetic_mlp_dropout
        self.bias_droppath = bias_droppath
        self.bias_dropout = bias_dropout
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
