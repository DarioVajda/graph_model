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
        magnetic_groups: int = 0,
        magnetic_content: bool = False,
        magnetic_linear: bool = False,
        magnetic_magnitude: bool = False,
        magnetic_hybrid: bool = False,
        magnetic_dim: int = 32,
        magnetic_content_dim: int = 128,
        magnetic_magnitude_dim: int = 64,
        magnetic_magnitude_repr_dim: int = 256,
        magnetic_q: float = 0.25,
        bias_self_node: bool = False,
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
        # Layer-grouped magnetic bias: G instances, layer l served by group
        # l*G // num_hidden_layers. One knob spanning the whole sharing spectrum —
        # G = num_hidden_layers is per-layer `magnetic`, G = 1 is `magnetic_shared`
        # (both verified numerically equivalent in tests/models/test_bias_sharing.py).
        # 0 disables. Mutually exclusive with `magnetic` / `magnetic_shared`, which
        # keep their own legacy code paths so existing checkpoints still load.
        self.magnetic_groups = magnetic_groups
        n_layers = getattr(self, "num_hidden_layers", None)
        if magnetic_groups:
            if sum(map(bool, (magnetic, magnetic_shared, magnetic_content,
                              magnetic_linear, magnetic_magnitude,
                              magnetic_hybrid))) > 0:
                raise ValueError(
                    "magnetic_groups is mutually exclusive with magnetic / "
                    "magnetic_shared / magnetic_content / magnetic_linear / "
                    "magnetic_magnitude / magnetic_hybrid; got "
                    f"magnetic_groups={magnetic_groups} alongside "
                    f"magnetic={magnetic}, magnetic_shared={magnetic_shared}, "
                    f"magnetic_content={magnetic_content}, "
                    f"magnetic_linear={magnetic_linear}, "
                    f"magnetic_magnitude={magnetic_magnitude}, "
                    f"magnetic_hybrid={magnetic_hybrid}.")
            if n_layers is not None and not 1 <= magnetic_groups <= n_layers:
                raise ValueError(
                    f"magnetic_groups={magnetic_groups} must be in [1, "
                    f"num_hidden_layers={n_layers}].")
        # Content-conditioned magnetic bias (per-layer). Reuses the magnetic
        # spectral machinery and consumes the same magnetic_V / magnetic_lambdas
        # features, plus the live hidden states (see MagneticContentBias). Its
        # down-projection width is magnetic_content_dim (d_proj).
        self.magnetic_content = magnetic_content
        # Linear-head magnetic bias (see LinearMagneticBias / LINEAR_BIAS.md).
        # Consumes exactly the same magnetic_V / magnetic_lambdas features as
        # `magnetic`, so every dataset/collator gate that keys on `magnetic` must
        # also accept this flag — a gate that misses it yields a run with NO bias
        # at all, which trains cleanly and looks like a clean negative result.
        self.magnetic_linear = magnetic_linear
        # Decoupled magnetic heads (see MIXED_BIAS.md): `magnetic_magnitude` is
        # the non-linear node-level magnitude channel alone; `magnetic_hybrid` is
        # that channel in tandem with the linear phase channel, i.e. the proposed
        # O(N) replacement. Both consume exactly the same magnetic_V /
        # magnetic_lambdas features as `magnetic`, so the gate warning above
        # applies to them verbatim — a dataset gate that misses one yields a run
        # with NO bias at all, which trains cleanly and reads as a clean negative.
        self.magnetic_magnitude = magnetic_magnitude
        self.magnetic_hybrid = magnetic_hybrid
        # Every placement of the magnetic term is a different HEAD on the same
        # features, so at most one may be enabled. Checked as one rule rather than
        # per-flag so a new placement cannot be added past a stale enumeration.
        placements = (("magnetic", magnetic), ("magnetic_shared", magnetic_shared),
                      ("magnetic_content", magnetic_content),
                      ("magnetic_linear", magnetic_linear),
                      ("magnetic_magnitude", magnetic_magnitude),
                      ("magnetic_hybrid", magnetic_hybrid),
                      ("magnetic_groups", bool(magnetic_groups)))
        for name in ("magnetic_linear", "magnetic_magnitude", "magnetic_hybrid"):
            if not dict(placements)[name]:
                continue
            clash = [n for n, v in placements if v and n != name]
            if clash:
                raise ValueError(
                    f"{name} replaces the magnetic head; enabling it "
                    f"alongside {clash} stacks two biases on the same features, "
                    "which is never the intended arm. Pass exactly one placement.")
        self.magnetic_dim = magnetic_dim
        self.magnetic_content_dim = magnetic_content_dim
        # Magnitude-channel widths. `magnetic_magnitude_repr_dim` is INTERNAL to
        # MLP_magnitude — evaluated once per node per forward and never seen by
        # attention, so it is free. `magnetic_magnitude_dim` is what Q/K append to
        # every head, and it is not (MIXED_BIAS.md §2.5).
        self.magnetic_magnitude_dim = magnetic_magnitude_dim
        self.magnetic_magnitude_repr_dim = magnetic_magnitude_repr_dim
        self.magnetic_q = magnetic_q
        # Keep the intra-node diagonal b_ii instead of zeroing it (see
        # MagneticBias._finalize / LINEAR_BIAS.md §7.3). Applies to the magnetic
        # family and RRWP. Default False = today's behaviour, bit-for-bit.
        #
        # SPD is deliberately NOT covered: SPDBias indexes `clamp(spd-1, ...)`, so
        # self-distance 0 collides with distance 1 and it has no table row of its
        # own. Adding one changes `weights`' shape and would break reload of every
        # existing bias_parameters.pt. Raise rather than let the flag mean
        # "self-node bias, except silently not for one of the three bias types".
        if bias_self_node and spd:
            raise ValueError(
                "bias_self_node does not cover SPDBias: its lookup has no row for "
                "self-distance 0 (idx = clamp(spd-1, ...)), so enabling the flag "
                "alongside spd would silently apply to only some of the active "
                "biases. Widening the table is a checkpoint-breaking change — do "
                "it deliberately, not as a side effect of this flag.")
        self.bias_self_node = bias_self_node
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
