"""
GTLM-Llama v2 — a clean, self-contained graph-biased Llama causal LM.

How graph structure enters attention (see :mod:`graph_attention_v2`):
  * The layer-independent **structural mask** (causal + bidirectional-prefix +
    hard K-hop gate + padding) is built once per batch in ``forward`` and shared.
  * The per-layer trainable **soft bias** (SPD / Laplacian / RWSE / RRWP /
    Magnetic, from :class:`GraphAttentionBias`) is computed inside each attention
    layer and added on top.

The per-batch graph context is attached to each attention module as
``self_attn._graph_ctx`` before the stock model loop runs.  This deliberately
avoids threading custom kwargs through ``forward`` — HF 4.50 drops layer kwargs
under gradient checkpointing, whereas a module attribute survives recompute.
"""

import os
import json
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    logger,
)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from .graph_bias import GraphAttentionBias
from .graph_attention_v2 import (
    GRAPH_ATTN_IMPLS,
    build_dense_structural_mask,
    expand_node_to_token_bias,
    graph_attention_dispatch,
)
from .model_utils import load_bias_parameters


_FA2_WARNING = (
    "\n" + "=" * 78 + "\n"
    "  GTLM-Llama is INCOMPATIBLE with FlashAttention (it cannot express the\n"
    "  graph attention bias / K-hop / bidirectional-prefix mask). Falling back\n"
    "  to eager attention. Pass graph_attn_impl='sdpa' (or 'flex') instead.\n"
    + "=" * 78
)


# ── Config ──────────────────────────────────────────────────────────────────────

class GTLMLlamaConfig(LlamaConfig):
    """LlamaConfig + flat graph-bias fields, a K-hop field, and a backend selector.

    The graph-bias fields are stored flat (not nested) so that standard
    ``save_pretrained`` / ``from_pretrained`` round-trips them with no custom code,
    and so the config object can be handed directly to
    :class:`GraphAttentionBias` as its ``bias_config``.
    """

    model_type = "gtlm_llama"

    def __init__(
        self,
        spd: bool = False,
        laplacian: bool = False,
        max_spd: int = 32,
        rwse: bool = False,
        rrwp: bool = False,
        max_rw_steps: int = 8,
        magnetic: bool = False,
        magnetic_dim: int = 32,
        magnetic_q: float = 0.25,
        k_hop: int = 0,
        k_hop_directed: bool = False,
        graph_attn_impl: str = "eager",
        checkpoint_graph_bias: bool = True,
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
        self.magnetic_dim = magnetic_dim
        self.magnetic_q = magnetic_q
        self.k_hop = k_hop
        self.k_hop_directed = k_hop_directed
        self.graph_attn_impl = graph_attn_impl
        # Recompute the per-layer bias modules in backward instead of saving
        # their (B,N,N,·) intermediates — ~40 GB at N=2048 for ms of recompute.
        # Training-only (eval/generation paths are unaffected by the flag).
        self.checkpoint_graph_bias = checkpoint_graph_bias


# ── Attention (the one substantive override) ────────────────────────────────────

class GTLMLlamaAttention(LlamaAttention):
    """Llama attention + a per-layer :class:`GraphAttentionBias`.

    Reads the per-batch graph context from ``self._graph_ctx`` (set by
    :meth:`GTLMLlamaForCausalLM.forward`), ignores the ``attention_mask`` the
    stock model loop passes in, and delegates the actual attention math to the
    backend selected by ``ctx['impl']``.
    """

    def __init__(self, config: GTLMLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # K-hop is handled by the shared structural mask, so the bias module runs
        # in soft-only mode (k_hop=0) and just accumulates the trainable biases.
        self.graph_bias = GraphAttentionBias(
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            layer_idx=layer_idx,
            bias_config=config,
            k_hop=0,
        )
        self._graph_ctx = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,   # ignored; we use _graph_ctx
        past_key_value: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ctx = self._graph_ctx
        if ctx is None:
            raise RuntimeError(
                "GTLMLlamaAttention ran without a graph context. Call the model "
                "through GTLMLlamaForCausalLM.forward, which installs it."
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        q_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        # Per-layer trainable soft bias → node level → token level. Optionally
        # gradient-checkpointed: the bias compute is milliseconds, but autograd
        # would otherwise keep every layer's (B,N,N,·) intermediates alive.
        # Safe to recompute: in training the bias cache is never read, so the
        # recompute is deterministic.
        feats = ctx["features"]

        def _bias():
            return self.graph_bias(
                dtype=query_states.dtype,
                device=query_states.device,
                num_nodes=ctx["num_nodes"],
                spd=feats["spd"],
                laplacian=feats["laplacian"],
                rwse=feats["rwse"],
                rrwp=feats["rrwp"],
                magnetic=feats["magnetic"],
                k_hop_mask=None,
                cache_dict=ctx["cache"],
            )

        if (getattr(self.config, "checkpoint_graph_bias", False)
                and self.training and torch.is_grad_enabled()):
            node_bias = torch.utils.checkpoint.checkpoint(_bias, use_reentrant=False)
        else:
            node_bias = _bias()
        token_soft_bias = (
            expand_node_to_token_bias(node_bias, ctx["node_ids"], q_len, kv_len)
            if node_bias is not None else None
        )

        attn_output, attn_weights = graph_attention_dispatch(
            impl=ctx["impl"],
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            structural_mask=ctx["structural_mask"],
            token_soft_bias=token_soft_bias,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ── Decoder layer & model (init-only swaps; no forward override) ────────────────

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

class GTLMLlamaForCausalLM(LlamaForCausalLM):
    """Graph-biased Llama causal LM with a standard HF I/O contract."""

    config_class = GTLMLlamaConfig

    def __init__(self, config: GTLMLlamaConfig):
        self._sanitize_attn_config(config)
        super().__init__(config)
        self.model = GTLMLlamaModel(config)
        self.config.architectures = ["GTLMLlamaForCausalLM"]
        self._graph_bias_cache: dict = {}
        self.post_init()

    @staticmethod
    def _sanitize_attn_config(config: GTLMLlamaConfig) -> None:
        if getattr(config, "_attn_implementation", None) == "flash_attention_2":
            warnings.warn(_FA2_WARNING, stacklevel=2)
            logger.warning(_FA2_WARNING)
            config._attn_implementation = "eager"

        impl = getattr(config, "graph_attn_impl", None) or "eager"
        if impl not in GRAPH_ATTN_IMPLS:
            raise ValueError(
                f"config.graph_attn_impl='{impl}' is invalid; expected one of {GRAPH_ATTN_IMPLS}."
            )
        config.graph_attn_impl = impl
        # We never use HF's attention dispatch, so pin it to eager to avoid HF
        # building sdpa/flex masks (which we would ignore anyway).
        config._attn_implementation = "eager"

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,        # (B, L) padding mask
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,            # (B, L), -100 to ignore
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # ── graph features (standard columns) ───────────────────────────────────
        node_ids: Optional[torch.LongTensor] = None,
        prompt_node: Optional[torch.LongTensor] = None,
        num_nodes: Optional[torch.LongTensor] = None,
        shortest_path_dists: Optional[torch.Tensor] = None,
        laplacian_coordinates: Optional[torch.Tensor] = None,
        rwse: Optional[torch.Tensor] = None,
        rrwp: Optional[torch.Tensor] = None,
        magnetic_V: Optional[torch.Tensor] = None,
        magnetic_lambdas: Optional[torch.Tensor] = None,
        k_hop_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        if node_ids is None or prompt_node is None:
            raise ValueError("GTLMLlama requires 'node_ids' and 'prompt_node' (use GraphCollatorV2).")

        embed_dtype = self.model.embed_tokens.weight.dtype
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        q_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past_seen + q_len

        # Reset the per-generation bias cache at prefill (step 0).
        if past_seen == 0:
            self._graph_bias_cache = {}

        # Padding mask covering the full KV length (default: all-valid).
        if attention_mask is not None and attention_mask.dim() == 2:
            pad_mask = attention_mask
        else:
            pad_mask = torch.ones((node_ids.shape[0], kv_len), dtype=torch.long, device=device)

        structural_mask = build_dense_structural_mask(
            node_ids=node_ids,
            prompt_node=prompt_node,
            pad_mask=pad_mask,
            k_hop_mask=k_hop_mask,
            k_hop=self.config.k_hop,
            q_len=q_len,
            kv_len=kv_len,
            dtype=embed_dtype,
            device=device,
        )

        magnetic = (magnetic_V, magnetic_lambdas) if magnetic_V is not None else None
        ctx = {
            "impl": self.config.graph_attn_impl,
            "node_ids": node_ids,
            "num_nodes": num_nodes,
            "structural_mask": structural_mask,
            "cache": self._graph_bias_cache,
            "features": {
                "spd": shortest_path_dists,
                "laplacian": laplacian_coordinates,
                "rwse": rwse,
                "rrwp": rrwp,
                "magnetic": magnetic,
            },
        }
        # Install the context on every attention module. Not cleared afterwards so
        # it survives gradient-checkpointing recompute; overwritten next forward.
        for layer in self.model.layers:
            layer.self_attn._graph_ctx = ctx

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,                    # we supply the mask via _graph_ctx
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ── Generation ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self._graph_bias_cache = {}
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        node_ids = kwargs.get("node_ids", None)
        prompt_node = kwargs.get("prompt_node", None)
        position_ids = kwargs.get("position_ids", None)

        # Extend node_ids for newly generated tokens (they belong to the prompt node).
        if node_ids is not None and prompt_node is not None:
            cur_len = input_ids.shape[1]
            if cur_len > node_ids.shape[1]:
                pad = prompt_node.view(-1, 1).to(node_ids.device).expand(-1, cur_len - node_ids.shape[1])
                node_ids = torch.cat([node_ids, pad], dim=1)

        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            remove = past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            input_ids = input_ids[:, remove:]
            if position_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "node_ids": node_ids,                 # full KV length, sliced inside forward
            "prompt_node": prompt_node,
            "num_nodes": kwargs.get("num_nodes"),
            "k_hop_mask": kwargs.get("k_hop_mask"),
        }
        for key in ("shortest_path_dists", "laplacian_coordinates", "rwse", "rrwp",
                    "magnetic_V", "magnetic_lambdas"):
            model_inputs[key] = kwargs.get(key)
        return model_inputs

    # ── Loading ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Three loading scenarios (mirrors v1): LoRA+bias, bias-only, standard.

        ``graph_attn_impl`` may be passed to select the backend; a
        ``flash_attention_2`` request is downgraded to eager with a loud warning.
        """
        graph_attn_impl = kwargs.pop("graph_attn_impl", None)

        if kwargs.get("attn_implementation") == "flash_attention_2":
            warnings.warn(_FA2_WARNING, stacklevel=2)
            logger.warning(_FA2_WARNING)
            kwargs["attn_implementation"] = "eager"
        # We own attention; keep HF on the eager path regardless of request.
        kwargs["attn_implementation"] = "eager"

        is_directory = os.path.isdir(pretrained_model_name_or_path)
        is_lora = is_directory and os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))
        is_bias_only = is_directory and os.path.exists(os.path.join(pretrained_model_name_or_path, "graph_bias_config.json"))

        if is_lora:
            print("Loading GTLMLlama with LoRA adapters and custom graph biases...")
            with open(os.path.join(pretrained_model_name_or_path, "adapter_config.json")) as f:
                base_model_name = json.load(f).get("base_model_name_or_path")
            with open(os.path.join(pretrained_model_name_or_path, "config.json")) as f:
                config = GTLMLlamaConfig(**json.load(f))
            if graph_attn_impl is not None:
                config.graph_attn_impl = graph_attn_impl
            model = super().from_pretrained(base_model_name, config=config, *model_args, **kwargs)
            if PeftModel is None:
                raise ImportError("PEFT is required to load a LoRA checkpoint.")
            model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
            load_bias_parameters(model, pretrained_model_name_or_path)
            return model

        if is_bias_only:
            print("Loading GTLMLlama with custom graph biases (no LoRA adapters)...")
            with open(os.path.join(pretrained_model_name_or_path, "graph_bias_config.json")) as f:
                base_model_path = json.load(f).get("base_model_name_or_path")
            with open(os.path.join(pretrained_model_name_or_path, "config.json")) as f:
                config = GTLMLlamaConfig(**json.load(f))
            if graph_attn_impl is not None:
                config.graph_attn_impl = graph_attn_impl
            model = super().from_pretrained(base_model_path, config=config, *model_args, **kwargs)
            load_bias_parameters(model, pretrained_model_name_or_path)
            return model

        print("Loading GTLMLlama via standard HuggingFace loading...")
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if graph_attn_impl is not None:
            model.config.graph_attn_impl = graph_attn_impl
        if is_directory:
            load_bias_parameters(model, pretrained_model_name_or_path)
        return model


# ── Auto-class registration (enables AutoModelForCausalLM + Hub sharing) ────────

GTLMLlamaConfig.register_for_auto_class()
GTLMLlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
try:
    AutoConfig.register("gtlm_llama", GTLMLlamaConfig)
    AutoModelForCausalLM.register(GTLMLlamaConfig, GTLMLlamaForCausalLM)
except ValueError:
    # Already registered (e.g. module re-imported) — safe to ignore.
    pass
