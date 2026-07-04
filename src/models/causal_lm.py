"""
Backbone-agnostic causal-LM mixin.

A GTLM causal LM for any backbone is
``class GTLMXForCausalLM(GraphCausalLMMixin, XForCausalLM)`` that sets two class
attributes — ``config_class`` and ``graph_model_cls`` (the GTLM model swap) — and
nothing else (unless the backbone deviates from HF submodule conventions, in which
case it overrides the accessor seams below).

The mixin owns all the backbone-neutral orchestration: building the per-batch
:class:`GraphContext`, selecting the registered ``gtlm_*`` attention function for
the batch, the loss/logits slicing, generation cache reset, ``node_ids``
extension during decoding, and the 3-scenario loader.
"""

import os
import json
import warnings
from typing import Optional, Tuple, Union

import torch
import torch._dynamo

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from .structural_mask import build_dense_structural_mask
from .dispatch import GRAPH_ATTN_IMPLS, GTLM_ATTN_FN_NAMES, build_flex_block_mask, flex_block_size
from .bias import build_shared_bias_modules, compute_shared_node_bias
from .context import GraphContext
from .io import load_bias_parameters

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

logger = logging.get_logger(__name__)


_FA2_WARNING = (
    "\n" + "=" * 78 + "\n"
    "  GTLM is INCOMPATIBLE with FlashAttention (it cannot express the\n"
    "  graph attention bias / K-hop / bidirectional-prefix mask). Falling back\n"
    "  to eager attention. Pass graph_attn_impl='flex' for the fast path.\n"
    + "=" * 78
)

_SDPA_WARNING = (
    "graph_attn_impl='sdpa' is not a separate GTLM backend; running 'eager' "
    "instead. A custom dense attention bias makes the fused SDPA kernels either "
    "unusable (flash) or unhelpful (mem-efficient), so SDPA only reduces to eager. "
    "For the fast path use graph_attn_impl='flex' (sparse block-masked kernel; it "
    "compiles a kernel per (sequence-length, node-count) shape at start-up)."
)


def _to_dtype(t, dtype):
    """Cast a float feature tensor to ``dtype`` (no-op for None / non-float)."""
    if t is not None and torch.is_floating_point(t):
        return t.to(dtype)
    return t


def _normalize_graph_attn_impl(impl):
    """Map the accepted ``'sdpa'`` alias to ``'eager'`` (warning once). ``'sdpa'``
    is not a real GTLM backend — see :data:`_SDPA_WARNING`. Returns ``impl``
    unchanged otherwise (including ``None``)."""
    if impl == "sdpa":
        logger.warning_once(_SDPA_WARNING)
        return "eager"
    return impl


class GraphCausalLMMixin:
    """Graph-biased causal LM orchestration, backbone-neutral.

    Subclasses must set ``config_class`` and ``graph_model_cls`` (the backbone's
    GTLM model class used to swap in the graph attention layers).
    """

    graph_model_cls = None  # set by the backbone adapter

    def __init__(self, config):
        self._sanitize_attn_config(config)
        super().__init__(config)
        self.model = self.graph_model_cls(config)
        # Shared (once-per-forward) bias modules, e.g. magnetic_shared: one
        # instance for the whole model instead of one per layer. None when no
        # shared type is enabled. The attribute name contains "graph_bias" so
        # the standard active-params substring unfreezes it too.
        self.shared_graph_bias = build_shared_bias_modules(config.num_attention_heads, config.head_dim, config)
        self.config.architectures = [type(self).__name__]
        self._graph_bias_cache: dict = {}
        self.post_init()

    # ── Backbone-neutrality seams (#6) ───────────────────────────────────────────
    # Defaults assume the HF decoder-LM convention. A backbone that deviates
    # (e.g. GPT-2's transformer.h[i].attn) overrides only these.

    def _iter_attn_modules(self):
        for layer in self.model.layers:
            yield layer.self_attn

    def _embed_dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype

    @staticmethod
    def _sanitize_attn_config(config) -> None:
        if getattr(config, "_attn_implementation", None) == "flash_attention_2":
            warnings.warn(_FA2_WARNING, stacklevel=2)
            logger.warning(_FA2_WARNING)
            config._attn_implementation = "eager"

        impl = _normalize_graph_attn_impl(getattr(config, "graph_attn_impl", None) or "eager")
        if impl not in GRAPH_ATTN_IMPLS:
            raise ValueError(
                f"config.graph_attn_impl='{impl}' is invalid; expected one of "
                f"{GRAPH_ATTN_IMPLS} (or 'sdpa', an accepted alias for 'eager')."
            )
        config.graph_attn_impl = impl
        # Init on a builtin impl so HF's construction-time validation is happy; the
        # forward pins the actual gtlm_* function per batch (set directly — a custom
        # name would be rejected if routed through HF's attn_implementation kwarg).
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
            raise ValueError("GTLM requires 'node_ids' and 'prompt_node' (use GraphCollatorV2).")

        embed_dtype = self._embed_dtype()
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Align float graph features with the (possibly bf16) model weights so the
        # bias-module matmuls/einsums don't hit a dtype mismatch. Integer features
        # (SPD indices, k_hop mask) are left as-is — they're used as indices/masks.
        laplacian_coordinates = _to_dtype(laplacian_coordinates, embed_dtype)
        rwse = _to_dtype(rwse, embed_dtype)
        rrwp = _to_dtype(rrwp, embed_dtype)
        magnetic_V = _to_dtype(magnetic_V, embed_dtype)
        magnetic_lambdas = _to_dtype(magnetic_lambdas, embed_dtype)

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

        # Backend selection. Flex serves the full-sequence case (training /
        # prefill, q_len == kv_len): we build a sparse BlockMask once per batch
        # instead of the dense (B,1,L,L) mask. Incremental decode (q_len < kv_len,
        # typically q_len == 1) falls back to the dense eager path — flex gives no
        # benefit at q_len == 1 and this keeps generation untouched.
        impl = self.config.graph_attn_impl
        use_flex = (impl == "flex" and q_len == kv_len)

        if use_flex:
            # Each distinct (L, N) shape compiles its own flex kernel; raise
            # dynamo's recompile cap (default 8) so we don't silently fall back to
            # eager once a run touches more than 8 buckets. Only ever raise it, so
            # a user who set a higher limit manually keeps theirs.
            if torch._dynamo.config.cache_size_limit < self.config.flex_cache_size_limit:
                torch._dynamo.config.cache_size_limit = self.config.flex_cache_size_limit

        structural_mask = None
        block_mask = None
        node_ids_flex = None
        if use_flex:
            block_size = self.config.flex_block_size or flex_block_size(
                self.config.k_hop, self.config.flex_compile_mode)
            if kv_len % block_size != 0:
                raise ValueError(
                    f"flex requires the sequence length to be a multiple of the "
                    f"block size ({block_size}); got L={kv_len}. Pad the batch in "
                    f"the dataloader (GraphCollatorV2(pad_to_block=True)) — the "
                    f"flex kernel hits a ~14x slowdown at unaligned lengths."
                )
            node_ids_flex = node_ids.to(torch.int32)
            block_mask = build_flex_block_mask(
                node_ids_flex, prompt_node, pad_mask, k_hop_mask,
                self.config.k_hop, q_len, kv_len,
                block_size=block_size, device=device,
            )
            eff_impl = "flex"
        else:
            eff_impl = "eager" if impl == "flex" else impl
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
        features = {
            "spd": shortest_path_dists,
            "laplacian": laplacian_coordinates,
            "rwse": rwse,
            "rrwp": rrwp,
            "magnetic": magnetic,
        }

        # Shared bias: computed ONCE here — outside the (gradient-checkpointed)
        # decoder layers, so training saves its activations a single time and
        # every layer reuses the same (B, H, N, N) tensor. In eval/generation it
        # is cached like the per-layer biases (reset at prefill above).
        shared_node_bias = None
        if self.shared_graph_bias is not None:
            if not self.training and "shared_node_bias" in self._graph_bias_cache:
                shared_node_bias = self._graph_bias_cache["shared_node_bias"]
            else:
                shared_node_bias = compute_shared_node_bias(
                    self.shared_graph_bias, dtype=embed_dtype, device=device,
                    num_nodes=num_nodes, features=features)
                if shared_node_bias is not None and not self.training:
                    self._graph_bias_cache["shared_node_bias"] = shared_node_bias

        ctx = GraphContext(
            node_ids=node_ids,
            num_nodes=num_nodes,
            cache=self._graph_bias_cache,
            features=features,
            structural_mask=structural_mask,
            block_mask=block_mask,
            node_ids_flex=node_ids_flex,
            shared_node_bias=shared_node_bias,
        )
        ctx.install_on(self._iter_attn_modules())

        # Strategy B: pin the registered gtlm_* function for this batch. The config
        # object is shared by every attention layer, so this selects the function
        # HF's stock attention forward dispatches to. Set-and-leave (like
        # _graph_ctx): gradient-checkpointing recompute in backward must see the
        # same function; the next forward overwrites it.
        self.config._attn_implementation = GTLM_ATTN_FN_NAMES[eff_impl]

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

        # Newly generated tokens belong to the prompt node, so extend both columns
        # for them: node_ids gets the prompt node, and position_ids *continues the
        # prompt node's per-node local counter* (positions reset per node, and the
        # prompt node is packed last → its last local position is position_ids[:, -1];
        # token i after it is that + i). Without this, every decoded token would
        # reuse the last prompt position, breaking RoPE during generation.
        if node_ids is not None and prompt_node is not None:
            cur_len = input_ids.shape[1]
            n_new = cur_len - node_ids.shape[1]
            if n_new > 0:
                pad = prompt_node.view(-1, 1).to(node_ids.device).expand(-1, n_new)
                node_ids = torch.cat([node_ids, pad], dim=1)
                if position_ids is not None:
                    steps = torch.arange(1, n_new + 1, device=position_ids.device).view(1, -1)
                    new_pos = position_ids[:, -1:] + steps           # (B, n_new)
                    position_ids = torch.cat([position_ids, new_pos], dim=1)

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
        """Three loading scenarios: LoRA+bias, bias-only, standard.

        ``graph_attn_impl`` may be passed to select the backend ('eager' or
        'flex'; 'sdpa' is accepted as an alias for 'eager', warning once). A
        ``flash_attention_2`` request is downgraded to eager with a loud warning.
        """
        graph_attn_impl = _normalize_graph_attn_impl(kwargs.pop("graph_attn_impl", None))

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
            print("Loading GTLM with LoRA adapters and custom graph biases...")
            with open(os.path.join(pretrained_model_name_or_path, "adapter_config.json")) as f:
                base_model_name = json.load(f).get("base_model_name_or_path")
            with open(os.path.join(pretrained_model_name_or_path, "config.json")) as f:
                config = cls.config_class(**json.load(f))
            if graph_attn_impl is not None:
                config.graph_attn_impl = graph_attn_impl
            model = super().from_pretrained(base_model_name, config=config, *model_args, **kwargs)
            if PeftModel is None:
                raise ImportError("PEFT is required to load a LoRA checkpoint.")
            model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
            load_bias_parameters(model, pretrained_model_name_or_path)
            return model

        if is_bias_only:
            print("Loading GTLM with custom graph biases (no LoRA adapters)...")
            with open(os.path.join(pretrained_model_name_or_path, "graph_bias_config.json")) as f:
                base_model_path = json.load(f).get("base_model_name_or_path")
            with open(os.path.join(pretrained_model_name_or_path, "config.json")) as f:
                config = cls.config_class(**json.load(f))
            if graph_attn_impl is not None:
                config.graph_attn_impl = graph_attn_impl
            model = super().from_pretrained(base_model_path, config=config, *model_args, **kwargs)
            load_bias_parameters(model, pretrained_model_name_or_path)
            return model

        print("Loading GTLM via standard HuggingFace loading...")
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if graph_attn_impl is not None:
            model.config.graph_attn_impl = graph_attn_impl
        if is_directory:
            load_bias_parameters(model, pretrained_model_name_or_path)
        return model
