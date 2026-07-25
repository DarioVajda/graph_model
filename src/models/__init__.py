"""
GTLM — graph-biased causal LMs, backbone-agnostic.

This package's public surface is the Llama adapter's classes. Importing the
package wires everything together as an import side-effect:

  * the ``gtlm_eager`` / ``gtlm_flex`` attention functions are
    registered into HF's ``ALL_ATTENTION_FUNCTIONS`` (via :mod:`dispatch`), and
  * ``GTLMLlamaConfig`` / ``GTLMLlamaForCausalLM`` are registered with
    ``AutoConfig`` / ``AutoModelForCausalLM`` (via the backbone adapter).

Adding a new backbone is a thin ``modeling_gtlm_<backbone>.py`` adapter in this
package (see REFACTOR.md §3c); re-export its classes here.
"""

from .modeling_gtlm_llama import (
    GTLMLlamaConfig,
    GTLMLlamaAttention,
    GTLMLlamaDecoderLayer,
    GTLMLlamaModel,
    GTLMLlamaForCausalLM,
)
# BLOOM: an ALiBi backbone, wired as a probe adapter (eager only). Self-contained —
# it overrides what it needs locally instead of generalising the shared modules.
from .modeling_gtlm_bloom import (
    GTLMBloomConfig,
    GTLMBloomAttention,
    GTLMBloomBlock,
    GTLMBloomModel,
    GTLMBloomForCausalLM,
)
# Gemma-3: a layer-heterogeneous RoPE backbone (local vs global base per layer),
# wired Strategy B like Llama — both backends, no attention forward override.
from .modeling_gtlm_gemma3 import (
    GTLMGemma3Config,
    GTLMGemma3Attention,
    GTLMGemma3DecoderLayer,
    GTLMGemma3Model,
    GTLMGemma3ForCausalLM,
)

__all__ = [
    "GTLMLlamaConfig",
    "GTLMLlamaAttention",
    "GTLMLlamaDecoderLayer",
    "GTLMLlamaModel",
    "GTLMLlamaForCausalLM",
    "GTLMBloomConfig",
    "GTLMBloomAttention",
    "GTLMBloomBlock",
    "GTLMBloomModel",
    "GTLMBloomForCausalLM",
    "GTLMGemma3Config",
    "GTLMGemma3Attention",
    "GTLMGemma3DecoderLayer",
    "GTLMGemma3Model",
    "GTLMGemma3ForCausalLM",
]
