"""
GTLM-Llama — the Llama backbone adapter for the graph-biased causal LM.

This is a *thin* adapter: all backbone-neutral logic lives in the shared GTLM
mixins (:mod:`config`, :mod:`attention`, :mod:`causal_lm`) and the registered
``gtlm_*`` attention functions (:mod:`dispatch`). This file only wires the Llama
base classes to those mixins — the same five-class pattern any new backbone
follows (see REFACTOR.md §3c).

Strategy B (see REFACTOR.md §3a): there is **no attention ``forward`` override**.
The stock ``LlamaAttention.forward`` does q/k/v projection + RoPE + cache, then
dispatches to the ``gtlm_*`` function named by ``config._attn_implementation``,
which the causal-LM forward pins per batch. Per-node position reset, padding
loss-neutrality, and backend agreement are unchanged (see the v2 test-suite).
"""

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

from .config import GraphConfigMixin
from .attention import GraphAttentionMixin
from .causal_lm import GraphCausalLMMixin

# ── trust_remote_code bundling manifest ──────────────────────────────────────────
# When a checkpoint is loaded from a *local* directory, HF copies only this entry
# file's *direct* relative imports into its module cache (it does not recurse —
# see `get_cached_module_file`). The graph mixins above pull in the rest of the
# package transitively, so without the explicit imports below those transitive
# modules (bias / io / context / dispatch / structural_mask / flex_kernel) would
# be missing on a local `from_pretrained(dir, trust_remote_code=True)`. Importing
# one symbol from each via the bundler-visible ``from .mod import name`` form pins
# the whole flat closure for both local *and* Hub-download loads. Keep these.
from .context import GraphContext  # noqa: F401
from .bias import GraphAttentionBias  # noqa: F401
from .dispatch import register_gtlm_attention_functions  # noqa: F401
from .structural_mask import build_dense_structural_mask  # noqa: F401
from .flex_kernel import flex_block_size  # noqa: F401
from .io import save_bias_parameters, load_bias_parameters  # noqa: F401


# ── Config ──────────────────────────────────────────────────────────────────────

class GTLMLlamaConfig(GraphConfigMixin, LlamaConfig):
    """LlamaConfig + the flat graph-bias fields (from :class:`GraphConfigMixin`)."""

    model_type = "gtlm_llama"


# ── Attention / decoder layer / model (init-only swaps; no forward override) ─────

class GTLMLlamaAttention(GraphAttentionMixin, LlamaAttention):
    def __init__(self, config: GTLMLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.init_graph_bias(config, layer_idx)


class GTLMLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GTLMLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GTLMLlamaAttention(config, layer_idx)


class GTLMLlamaModel(LlamaModel):
    def __init__(self, config: GTLMLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GTLMLlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


# ── Causal LM ───────────────────────────────────────────────────────────────────

class GTLMLlamaForCausalLM(GraphCausalLMMixin, LlamaForCausalLM):
    """Graph-biased Llama causal LM with a standard HF I/O contract."""

    config_class = GTLMLlamaConfig
    graph_model_cls = GTLMLlamaModel


# ── Auto-class registration ─────────────────────────────────────────────────────
# register_for_auto_class makes `save_pretrained` bundle this adapter and its
# (flat, same-directory) module closure into the checkpoint and set `auto_map`, so
# `AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)` works for
# Hub-shared checkpoints without the package installed. HF's source-bundler only
# follows single-dot, same-directory relative imports — which is why every GTLM
# module lives flat in this package (no subpackages, no `..` imports).
GTLMLlamaConfig.register_for_auto_class()
GTLMLlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
# In-process registration: resolves model_type "gtlm_llama" for
# AutoModel/AutoConfig once this package is imported (the path used when the
# package *is* installed).
try:
    AutoConfig.register("gtlm_llama", GTLMLlamaConfig)
    AutoModelForCausalLM.register(GTLMLlamaConfig, GTLMLlamaForCausalLM)
except ValueError:
    # Already registered (e.g. module re-imported) — safe to ignore.
    pass
