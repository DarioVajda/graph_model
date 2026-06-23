"""
Backbone-agnostic attention mixin.

A GTLM attention class for any backbone is
``class GTLMXAttention(GraphAttentionMixin, XAttention)`` whose ``__init__`` calls
``super().__init__(...)`` then :meth:`GraphAttentionMixin.init_graph_bias`.

Under Strategy B (see REFACTOR.md §3a) there is **no ``forward`` override**: the
backbone's stock attention forward does q/k/v projection + RoPE + cache, then
dispatches to the registered ``gtlm_*`` function (selected by
``config._attn_implementation``), which reads this module's ``graph_bias`` and the
per-batch ``_graph_ctx``. The mixin therefore only *owns the parameters*; it adds
no attention math.
"""

from .bias import GraphAttentionBias


class GraphAttentionMixin:
    def init_graph_bias(self, config, layer_idx: int) -> None:
        """Attach the per-layer trainable graph bias. Call from the backbone
        attention's ``__init__`` after ``super().__init__``.

        K-hop is enforced by the shared structural mask, so the bias module runs
        in soft-only mode (``k_hop=0``) and just accumulates the trainable biases.
        ``head_dim`` is read from ``config.head_dim`` (not ``hidden_size //
        num_attention_heads``) so backbones where they differ (e.g. Gemma2) are
        correct.
        """
        self.graph_bias = GraphAttentionBias(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            layer_idx=layer_idx,
            bias_config=config,
            k_hop=0,
        )
        self._graph_ctx = None
