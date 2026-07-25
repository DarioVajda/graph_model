"""
GTLM-Gemma3 — the Gemma-3 backbone adapter: GTLM on a **layer-heterogeneous RoPE**
base model.

Purpose: demonstrate that the GTLM graph machinery is not tied to a single global
positional encoding. Llama-3 gives one RoPE base for every layer; BLOOM
(``modeling_gtlm_bloom.py``) gives ALiBi and no rotation at all; Gemma-3 gives a
*third* regime — five of every six layers rotate at a **local** base
(``rope_local_base_freq``, 10k) while the sixth rotates at a **global** base
(``rope_theta``, 1M), so different layers read the same ``position_ids`` at
different frequencies. GTLM's per-node position reset flows into both.

Unlike the BLOOM probe this is **Strategy B, exactly like Llama** (see
``modeling_gtlm_llama.py``): ``Gemma3Attention.forward`` dispatches through HF's
``ALL_ATTENTION_FUNCTIONS``, so the registered ``gtlm_eager`` / ``gtlm_flex``
functions are reachable and there is **no attention forward override**. Both
backends are wired. **No shared GTLM module is modified for this backbone** — the
three deviations below are absorbed here.

## The three deviations

**1. The sliding window is dropped.** Gemma-3 alternates sliding-window layers
(``sliding_window=512`` on gemma-3-1b, 22 of its 26 layers) with full-attention
layers, and :class:`Gemma3DecoderLayer` applies the local band by editing the 4-D
``attention_mask``. GTLM passes ``attention_mask=None`` and the ``gtlm_*``
functions substitute their own structural mask, so the band never materializes.

For GraphQA this is **inert**: packed sequences there top out at ~90 tokens
(measured over all nine tasks' train splits — ``max_length=1024`` is a *per-node*
cap that never binds), far under any Gemma-3 window, so the adapter reproduces
stock ``Gemma3ForCausalLM`` bit-for-bit on every real batch (pinned by
``tests/models/test_modeling_gtlm_gemma3.py``).

Beyond that length dropping the band is a deliberate choice, not an oversight: a
512-token window over the *packed serialization order* would hide most of the
graph from five of every six layers and break the bidirectional prefix — the same
objection that made stock ALiBi wrong for BLOOM. Under GTLM locality is defined by
the graph, not by serialization distance.

Note the band is indexed by ``cache_position`` (raw packed order), **not** by
``position_ids``, so GTLM's per-node position reset does *not* keep it inactive —
it only keeps the RoPE rotation angles small. A dataset with larger graphs (kgqa's
triplet graphs, say) will cross the window, which is why
:meth:`GTLMGemma3Model._update_causal_mask` warns once when it does rather than
letting the semantics change silently.

**2. Gemma-3 would build a ``HybridCache``, which silently truncates the KV.**
``Gemma3TextModel.forward`` constructs one whenever ``use_cache`` is set, no cache
was passed and the model is in eval. Its sliding layers allocate only
``min(sliding_window, max_cache_len)`` slots and ``_sliding_update`` returns
``key_states[:, :, -max_cache_len:, :]`` — so at ``L > sliding_window`` those
layers would hand back the *last* 512 keys while GTLM's structural mask still
covers all ``L``, and HF's eager kernel would slice the mask to its first 512
columns. Wrong, and silent. A forward pre-hook on the decoder stack installs a
plain :class:`DynamicCache` instead, which is precisely what
``LlamaModel.forward`` does for the production backbone.

**3. Softcapping must not be silently dropped.** The shared causal-LM forward
calls ``self.lm_head`` directly (skipping ``Gemma3ForCausalLM``'s
``final_logit_softcapping`` block), and ``gtlm_eager`` swallows the ``softcap=``
kwarg the attention interface may carry. Gemma-3 sets both fields to ``None``, so
both omissions are exact no-ops — :meth:`GTLMGemma3ForCausalLM._sanitize_attn_config`
makes that load-bearing and *raises* if a checkpoint reintroduces either, rather
than training a model that is not the pretrained one. (This is why Gemma-2, which
ships ``attn_logit_softcapping=50.0`` and ``final_logit_softcapping=30.0``, is not
wired here.)

Note ``config.head_dim`` (256) differs from ``hidden_size // num_attention_heads``
on this family; the shared attention mixin already reads ``config.head_dim``
directly, so no config surgery is needed (contrast ``GTLMBloomConfig``).
"""

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.utils import logging
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3TextModel,
    Gemma3ForCausalLM,
)

from .config import GraphConfigMixin
from .attention import GraphAttentionMixin
from .causal_lm import GraphCausalLMMixin

# ── trust_remote_code bundling manifest (see modeling_gtlm_llama.py) ────────────
from .context import GraphContext  # noqa: F401
from .bias import GraphAttentionBias  # noqa: F401
from .dispatch import register_gtlm_attention_functions  # noqa: F401
from .structural_mask import build_dense_structural_mask  # noqa: F401
from .flex_kernel import flex_block_size  # noqa: F401
from .io import save_bias_parameters, load_bias_parameters  # noqa: F401

logger = logging.get_logger(__name__)


# ── Config ──────────────────────────────────────────────────────────────────────

class GTLMGemma3Config(GraphConfigMixin, Gemma3TextConfig):
    """Gemma3TextConfig + the flat graph-bias fields (from :class:`GraphConfigMixin`).

    Text-only: the multimodal ``gemma-3-4b`` and larger checkpoints nest their text
    config under ``text_config`` and will not load through this class directly.
    """

    model_type = "gtlm_gemma3"


# ── Attention / decoder layer (init-only swaps; no forward override) ─────────────

class GTLMGemma3Attention(GraphAttentionMixin, Gemma3Attention):
    """Stock Gemma-3 attention (QK-norm, local-or-global RoPE, GQA) that dispatches
    to the registered ``gtlm_*`` function instead of HF's own. ``Gemma3Attention``
    already sets ``self.config`` and exposes ``num_key_value_groups`` / ``head_dim``,
    so the shared bias computation needs nothing added here."""

    def __init__(self, config: GTLMGemma3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.init_graph_bias(config, layer_idx)


class GTLMGemma3DecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: GTLMGemma3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GTLMGemma3Attention(config, layer_idx)


# ── Decoder stack: no HybridCache, no dense causal mask ─────────────────────────

def _dynamic_cache_hook(module, args, kwargs):
    """Forward pre-hook: hand the stack a :class:`DynamicCache` before it can build
    a ``HybridCache`` of its own (deviation 2 in the module docstring).

    Mirrors the stock condition in ``Gemma3TextModel.forward`` exactly — including
    resolving a ``use_cache=None`` against the config the way the stock forward
    does — so the hook fires in precisely the cases that would otherwise allocate
    the hybrid cache, and never in training (where the stock code builds no cache
    either). Keyword-only by construction: the shared causal-LM forward calls the
    stack with keyword arguments.
    """
    use_cache = kwargs.get("use_cache")
    if use_cache is None:
        use_cache = module.config.use_cache
    if use_cache and kwargs.get("past_key_values") is None and not module.training:
        kwargs["past_key_values"] = DynamicCache()
    return args, kwargs


class GTLMGemma3Model(Gemma3TextModel):
    config_class = GTLMGemma3Config

    def __init__(self, config: GTLMGemma3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GTLMGemma3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.register_forward_pre_hook(_dynamic_cache_hook, with_kwargs=True)
        self.post_init()

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position,
                            past_key_values, output_attentions):
        """No-op: GTLM supplies its own structural mask via ``_graph_ctx``, so the
        stock builder would only allocate a ``(B, 1, q, max_cache_len)`` tensor per
        forward for the ``gtlm_*`` functions to ignore. Returning ``None`` also
        keeps :class:`Gemma3DecoderLayer`'s sliding-window branch from firing
        (deviation 1).

        This is the exact point where the window would otherwise be applied, so it
        is also where we say so: past ``sliding_window`` packed tokens the adapter
        stops matching the pretrained attention pattern on the sliding layers.
        GraphQA never gets there (~90 tokens), but a larger-graph dataset would,
        and the change would otherwise be invisible.
        """
        window = getattr(self.config, "sliding_window", None)
        if window is not None and cache_position is not None and cache_position[-1] >= window:
            logger.warning_once(
                f"GTLM-Gemma3: packed sequence length {int(cache_position[-1]) + 1} "
                f"exceeds sliding_window={window}. The adapter drops Gemma-3's "
                f"sliding-window band (it is defined over packed serialization order "
                f"and would hide most of the graph from the sliding layers), so those "
                f"layers now attend more widely than the pretrained model does. This "
                f"is intended, but it means logits no longer match stock Gemma-3."
            )
        return None


# ── Causal LM ───────────────────────────────────────────────────────────────────

class GTLMGemma3ForCausalLM(GraphCausalLMMixin, Gemma3ForCausalLM):
    """Graph-biased Gemma-3 causal LM. Both backends ('eager' and 'flex')."""

    config_class = GTLMGemma3Config
    graph_model_cls = GTLMGemma3Model

    @staticmethod
    def _sanitize_attn_config(config) -> None:
        # Deviation 3: the shared stack drops both softcapping sites. Gemma-3 sets
        # them to None so that is exact; refuse anything else rather than train a
        # backbone that quietly is not the pretrained one.
        for field in ("attn_logit_softcapping", "final_logit_softcapping"):
            if getattr(config, field, None) is not None:
                raise ValueError(
                    f"GTLM-Gemma3 requires config.{field} to be None, got "
                    f"{getattr(config, field)!r}. The shared GTLM stack applies "
                    f"neither softcapping site (the registered gtlm_* attention "
                    f"functions ignore the 'softcap' kwarg, and GraphCausalLMMixin."
                    f"forward calls lm_head directly), so a non-None value would be "
                    f"silently dropped and the model would no longer match its "
                    f"pretrained backbone. Gemma-2 checkpoints hit this by design."
                )
        # Keep a derived GenerationConfig from requesting the hybrid cache; the
        # _prepare_cache_for_generation override below is the belt to this braces.
        config.cache_implementation = None
        GraphCausalLMMixin._sanitize_attn_config(config)

    def _prepare_cache_for_generation(self, generation_config, *args, **kwargs):
        """Force ``generate`` onto :class:`DynamicCache`.

        Gemma-3 defaults ``cache_implementation="hybrid"``, whose sliding layers
        both truncate the KV span (deviation 2) and fix the cache length at the
        prefill size, which the shared decode path — it grows ``node_ids`` and
        ``position_ids`` per step off ``past_key_values.get_seq_length()`` — cannot
        work against."""
        generation_config.cache_implementation = None
        return super()._prepare_cache_for_generation(generation_config, *args, **kwargs)


# ── Auto-class registration ─────────────────────────────────────────────────────

GTLMGemma3Config.register_for_auto_class()
GTLMGemma3ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
try:
    AutoConfig.register("gtlm_gemma3", GTLMGemma3Config)
    AutoModelForCausalLM.register(GTLMGemma3Config, GTLMGemma3ForCausalLM)
except ValueError:
    pass
