"""
GTLM-BLOOM — the BLOOM backbone adapter: GTLM on an **ALiBi** base model.

Purpose: demonstrate that the GTLM graph machinery (structural mask + trainable
soft graph bias) is not tied to RoPE. Everything graph-related is reused verbatim
from the shared modules; this file only bridges the two places where an ALiBi
backbone deviates from the RoPE backbones the shared code was written against.
**Nothing in the shared GTLM modules is modified for it** — every deviation is
absorbed here.

Scope: eager only, single-GPU, small graphs. This is a probe adapter (GraphQA
scale), not a second first-class backbone. `graph_attn_impl="flex"` is rejected at
construction rather than silently ignored.

## The two deviations

**1. Attention is Strategy A, not Strategy B.** BLOOM's attention forward is
hand-written (`alibi.baddbmm(q, k)` then `+ attention_mask` then softmax) and never
dispatches through HF's ``ALL_ATTENTION_FUNCTIONS``, so the registered
``gtlm_eager`` function is unreachable here. But the stock forward already adds a
4-D additive tensor to the scores before the softmax — exactly the slot GTLM needs.
So :class:`GTLMBloomAttention` overrides ``forward`` only to *substitute* that
argument with ``structural_mask + token_soft_bias`` and delegates the actual math
to ``super().forward()``. No attention math is reimplemented, and the shared
``compute_node_bias`` / ``expand_node_to_token_bias`` do the work.

**2. ALiBi has no ``position_ids``.** Stock BLOOM derives token positions from
``attention_mask.cumsum(-1)`` — i.e. from raw serialization order. Under GTLM that
would defeat the per-node position reset the collator emits: a node's ALiBi
penalty would grow with how early it was packed, so node *ordering* would leak into
attention as a recency prior (and the prompt node, packed last, would see distant
nodes through a large penalty). :meth:`GTLMBloomModel.build_alibi_tensor` therefore
rebuilds the bias from GTLM's ``position_ids`` instead, reusing HF's own pretrained
slopes. Consequences:

  * a single-node graph with ``arange`` positions reproduces stock ALiBi
    **bit-for-bit** (pinned by ``tests/models/test_modeling_gtlm_bloom.py``), so the
    pretrained model is untouched in the degenerate case;
  * with per-node reset, ALiBi becomes a *within-node* recency prior only, which is
    the ALiBi analogue of what the RoPE backbones already get;
  * ``node_position_mode="spd_depth"`` (GraphCollatorV2) flows through unchanged and
    turns ALiBi into a graph-distance-aware penalty.

``position_ids`` is not a BLOOM forward argument, so it is intercepted by a forward
pre-hook on the decoder stack (the same set-and-leave pattern the shared mixin uses
for ``magnetic_content``'s hidden-state capture) and stripped before the call.
"""

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import (
    BloomAttention,
    BloomBlock,
    BloomModel,
    BloomForCausalLM,
    build_alibi_tensor as hf_build_alibi_tensor,
)

from .config import GraphConfigMixin
from .attention import GraphAttentionMixin
from .causal_lm import GraphCausalLMMixin
from .dispatch import compute_node_bias
from .structural_mask import expand_node_to_token_bias
from .bias import build_group_bias_modules, build_shared_bias_modules

# ── trust_remote_code bundling manifest (see modeling_gtlm_llama.py) ────────────
from .context import GraphContext  # noqa: F401
from .flex_kernel import flex_block_size  # noqa: F401
from .io import save_bias_parameters, load_bias_parameters  # noqa: F401


# ── Config ──────────────────────────────────────────────────────────────────────

class GTLMBloomConfig(GraphConfigMixin, BloomConfig):
    """BloomConfig + the flat graph-bias fields (from :class:`GraphConfigMixin`)."""

    model_type = "gtlm_bloom"

    def __init__(self, **kwargs):
        # Derived, never an independent knob — drop any round-tripped value first.
        kwargs.pop("head_dim", None)
        super().__init__(**kwargs)
        # GraphAttentionBias reads `config.head_dim` (backbones like Gemma2 split it
        # from hidden_size/num_heads). BloomConfig has no such field and BLOOM never
        # splits, so derive it here rather than teaching the shared attention mixin
        # about configs that lack one.
        self.head_dim = self.hidden_size // self.n_head


# ── Attention: Strategy A, substituting the additive-bias argument ──────────────

class GTLMBloomAttention(GraphAttentionMixin, BloomAttention):
    """Stock BLOOM attention with GTLM's mask + graph bias in place of the causal
    mask. The graph bias is per layer, so it must enter here rather than at the
    model level where ALiBi and the causal mask are built once per forward."""

    def __init__(self, config: GTLMBloomConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # BloomAttention keeps no config reference, but the shared bias computation
        # reads `module.config` (checkpoint_graph_bias).
        self.config = config
        self.init_graph_bias(config, layer_idx)

    def forward(self, hidden_states, residual, alibi=None, attention_mask=None, **kwargs):
        ctx = self._graph_ctx
        if ctx is None:
            raise RuntimeError(
                "GTLMBloomAttention ran without a graph context. Call the model "
                "through GTLMBloomForCausalLM.forward, which installs `_graph_ctx` "
                "on every attention module."
            )

        q_len = hidden_states.shape[1]
        kv_len = ctx.node_ids.shape[1]          # node_ids always spans the full KV
        node_bias = compute_node_bias(self, ctx, hidden_states.dtype, hidden_states.device)

        # Stock BLOOM adds this tensor to the scores right after ALiBi and before the
        # softmax, and slices it to `[:, :, :, :kv_len]` — so a (B, 1, q, kv) mask and
        # a (B, H, q, kv) mask+bias are both accepted as-is.
        graph_mask = ctx.structural_mask
        if node_bias is not None:
            graph_mask = graph_mask + expand_node_to_token_bias(
                node_bias, ctx.node_ids, q_len, kv_len)

        return super().forward(
            hidden_states, residual, alibi=alibi, attention_mask=graph_mask, **kwargs)


class GTLMBloomBlock(BloomBlock):
    def __init__(self, config: GTLMBloomConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attention = GTLMBloomAttention(config, layer_idx)


# ── Decoder stack: ALiBi rebuilt from GTLM positions ────────────────────────────

def _graph_position_hook(module, args, kwargs):
    """Forward pre-hook: take ``position_ids`` off the call and stash the full-KV
    positions for :meth:`GTLMBloomModel.build_alibi_tensor`.

    BLOOM has no ``position_ids`` parameter, but the shared GTLM forward passes one
    (every other backbone needs it). ALiBi wants positions for the *whole* KV span
    while decoding passes only the new tokens, so prefill stores and later steps
    append. ``None`` anywhere disables the override and falls back to stock ALiBi.
    """
    positions = kwargs.pop("position_ids", None)
    past = kwargs.get("past_key_values")
    past_len = past.get_seq_length() if past is not None else 0

    if positions is None:
        module._alibi_positions = None
    elif past_len == 0:                                   # prefill: (B, kv_len)
        module._alibi_positions = positions
    else:                                                 # decode: (B, n_new)
        prev = module._alibi_positions
        # A mismatch means these positions don't continue the stored prefix (a
        # hand-rolled call, not `generate`); drop to stock ALiBi rather than
        # silently building a bias over the wrong positions.
        module._alibi_positions = (
            torch.cat([prev, positions], dim=1)
            if prev is not None and prev.shape[1] == past_len else None
        )
    return args, kwargs


class GTLMBloomModel(BloomModel):
    def __init__(self, config: GTLMBloomConfig):
        super().__init__(config)
        self.h = nn.ModuleList(
            [GTLMBloomBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        # Plain attributes, not buffers: they are per-batch scratch, must not enter
        # the state dict, and `_alibi_slopes` is re-derived per device on demand.
        self._alibi_positions = None
        self._alibi_slopes = None
        self.register_forward_pre_hook(_graph_position_hook, with_kwargs=True)
        self.post_init()

    def _slopes(self, num_heads: int, device: torch.device) -> torch.Tensor:
        """The pretrained per-head ALiBi slopes, ``(num_heads,)``.

        Read out of HF's own builder with a 2-token probe (whose bias is
        ``slope * [0, 1]``) instead of duplicating the slope formula here, so this
        adapter tracks upstream exactly. Cached per device.
        """
        cached = self._alibi_slopes
        if cached is None or cached.device != device or cached.shape[0] != num_heads:
            probe = torch.ones(1, 2, device=device)
            cached = hf_build_alibi_tensor(probe, num_heads, torch.float32)[:, 0, 1].clone()
            self._alibi_slopes = cached
        return cached

    def build_alibi_tensor(self, attention_mask, num_heads, dtype):
        """ALiBi over GTLM's per-node positions instead of raw sequence order.

        Same slopes and same ``(B*H, 1, kv_len)`` layout as stock BLOOM (the bias is
        a function of key position only), so the only change is *which* position
        each key is assigned — the whole point of the exercise.
        """
        positions = self._alibi_positions
        if positions is None:
            return super().build_alibi_tensor(attention_mask, num_heads, dtype)

        slopes = self._slopes(num_heads, positions.device)             # (H,)
        alibi = slopes.view(1, num_heads, 1) * positions[:, None, :].to(slopes.dtype)
        batch_size, seq_length = positions.shape
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    def _update_causal_mask(self, *args, **kwargs):
        """No-op: GTLM supplies its own structural mask per layer (the attention
        override substitutes it), so building a dense causal mask here would only
        allocate a (B, 1, q, kv) tensor for every forward and throw it away."""
        return None


# ── Causal LM ───────────────────────────────────────────────────────────────────

class GTLMBloomForCausalLM(GraphCausalLMMixin, BloomForCausalLM):
    """Graph-biased BLOOM causal LM. Eager backend only."""

    config_class = GTLMBloomConfig
    graph_model_cls = GTLMBloomModel

    def __init__(self, config: GTLMBloomConfig):
        # Deliberately NOT GraphCausalLMMixin.__init__: it installs the graph stack
        # at `self.model`, while BLOOM's pretrained weights are keyed under
        # `transformer.*`. The rest of its body is replicated verbatim and in order
        # — the hidden-state capture hook must be registered *after* the swap so it
        # lands on the graph attention modules.
        self._sanitize_attn_config(config)
        BloomForCausalLM.__init__(self, config)
        self.transformer = GTLMBloomModel(config)
        self.shared_graph_bias = build_shared_bias_modules(
            config.num_attention_heads, config.head_dim, config)
        self.group_graph_bias = build_group_bias_modules(
            config.num_attention_heads, config.head_dim, config, config.num_hidden_layers)
        self.config.architectures = [type(self).__name__]
        self._graph_bias_cache: dict = {}
        if getattr(config, "magnetic_content", False):
            self._register_hidden_state_capture()
        self.post_init()

    # ── Backbone-neutrality seams ───────────────────────────────────────────────

    @property
    def model(self):
        """Alias for the shared forward's ``self.model(...)`` call.

        A property, not an attribute: binding the same module under two names would
        duplicate it in ``state_dict`` and in everything that walks ``_modules``.
        """
        return self.transformer

    def _iter_attn_modules(self):
        for block in self.transformer.h:
            yield block.self_attention

    def _embed_dtype(self) -> torch.dtype:
        return self.transformer.word_embeddings.weight.dtype

    @staticmethod
    def _sanitize_attn_config(config) -> None:
        if getattr(config, "graph_attn_impl", None) == "flex":
            raise ValueError(
                "GTLM-BLOOM wires the eager backend only. The flex path needs the "
                "registered gtlm_flex attention function, which BLOOM's hand-written "
                "attention forward never dispatches to, and the ALiBi bias would have "
                "to be folded into the score_mod. Pass graph_attn_impl='eager'."
            )
        GraphCausalLMMixin._sanitize_attn_config(config)


# ── Auto-class registration ─────────────────────────────────────────────────────

GTLMBloomConfig.register_for_auto_class()
GTLMBloomForCausalLM.register_for_auto_class("AutoModelForCausalLM")
try:
    AutoConfig.register("gtlm_bloom", GTLMBloomConfig)
    AutoModelForCausalLM.register(GTLMBloomConfig, GTLMBloomForCausalLM)
except ValueError:
    pass
