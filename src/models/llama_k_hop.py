"""
K-hop masked graph-Llama model.

Extends the existing GraphLlamaForCausalLM with a hard K-hop attention gate:
for every attention head, node pairs that are more than K hops apart in the
input graph receive -inf before softmax, preventing cross-hop attention.

K=0 disables the gate entirely, making this model behave identically to
GraphLlamaForCausalLM with the same bias configuration.

The K-hop mask is precomputed per batch by GraphCollator(k_hop=K) and stored
under the key 'k_hop_mask' in the batch dict.  Prompt-node semantics:
  - Prefix ↔ prefix reachability is computed on the graph with the prompt
    node's edges removed (the prompt cannot act as a bridge).
  - The prompt node's own row/col use the full graph so that its K-hop
    neighbourhood is determined by its actual edges.

New public classes
------------------
KHopLlamaConfig             - GraphLlamaConfig + k_hop field
LlamaAttentionWithKHop      - attention layer using GraphAttentionBias
LlamaDecoderLayerWithKHop   - decoder layer wiring custom args through
LlamaModelWithKHop          - model stack with KHop decoder layers
KHopGraphLlamaForCausalLM   - top-level causal LM (same API as GraphLlamaForCausalLM)
"""

import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Callable

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    logger,
)

from .llama_attn_bias import (
    GraphLlamaConfig,
    GraphLlamaForCausalLM,
    LlamaDecoderLayerWithBias,
    LlamaModelWithBias,
)
from .graph_bias import GraphAttentionBias
from ..utils.text_graph_dataset import TextGraph


# ── Config ────────────────────────────────────────────────────────────────────

class KHopLlamaConfig(GraphLlamaConfig):
    """GraphLlamaConfig extended with a k_hop field for the hard attention gate."""

    model_type = "k_hop_graph_llama"

    def __init__(self, k_hop: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.k_hop = k_hop

    def to_dict(self):
        output = super().to_dict()
        output['k_hop'] = self.k_hop
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        k_hop = kwargs.pop('k_hop', config_dict.pop('k_hop', 0))
        return cls(k_hop=k_hop, **config_dict)


# ── Node → token expansion (shared utility) ───────────────────────────────────

def expand_node_to_token_bias(
    node_bias: torch.Tensor,    # (B, H, N, N)
    node_ids:  torch.Tensor,    # (B, full_seq_len)
    q_len:     int,
    kv_len:    int,
) -> torch.Tensor:              # (B, H, q_len, kv_len)
    """
    Lift a node-level bias matrix to token level using node_ids.

    Each token inherits the bias of the node it belongs to, so the
    (q, k) token pair gets the bias of (node[q], node[k]).
    """
    B, H = node_bias.shape[0], node_bias.shape[1]
    device = node_ids.device

    node_ids_q  = node_ids[:, -q_len:]               # (B, q_len)
    node_ids_kv = node_ids[:, -kv_len:]              # (B, kv_len)

    b_idx = torch.arange(B,   device=device).view(B, 1, 1,     1)
    h_idx = torch.arange(H,   device=device).view(1, H, 1,     1)
    q_idx = node_ids_q.view(B,  1, q_len,  1)
    k_idx = node_ids_kv.view(B, 1, 1,     kv_len)

    return node_bias[b_idx, h_idx, q_idx, k_idx]     # (B, H, q_len, kv_len)


# ── Attention layer ───────────────────────────────────────────────────────────

class LlamaAttentionWithKHop(LlamaAttention):
    """
    Llama attention augmented with graph-structured biases and a K-hop gate.

    Holds a GraphAttentionBias module that owns all trainable graph parameters
    for this layer.  The node-level (B, H, N, N) bias is expanded to token
    level via node_ids before being added to the attention mask.
    """

    def __init__(self, config: KHopLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.graph_bias = GraphAttentionBias(
            num_heads   = config.num_attention_heads,
            head_dim    = config.hidden_size // config.num_attention_heads,
            layer_idx   = layer_idx,
            bias_config = config.graph_attn_bias,
            k_hop       = config.k_hop,
        )

    def forward(
        self,
        hidden_states:        torch.Tensor,
        position_embeddings:  Tuple[torch.Tensor, torch.Tensor],
        attention_mask:       Optional[torch.Tensor],
        past_key_value:       Optional[object]         = None,
        cache_position:       Optional[torch.LongTensor] = None,
        node_ids:             Optional[torch.LongTensor] = None,   # (B, seq_len)
        input_graph_batch:    Optional[TextGraph]        = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        input_shape  = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ── Graph bias ────────────────────────────────────────────────────────
        gb = self.graph_bias
        node_bias = gb(
            dtype      = query_states.dtype,
            device     = query_states.device,
            num_nodes  = input_graph_batch.get('num_nodes')              if input_graph_batch else None,
            spd        = input_graph_batch.get('shortest_path_dists')    if (input_graph_batch and gb.require_spd)      else None,
            laplacian  = input_graph_batch.get('laplacian_coordinates')  if (input_graph_batch and gb.require_laplacian) else None,
            rwse       = input_graph_batch.get('rwse')                   if (input_graph_batch and gb.require_rwse)      else None,
            rrwp       = input_graph_batch.get('rrwp')                   if (input_graph_batch and gb.require_rrwp)      else None,
            magnetic   = input_graph_batch.get('magnetic')               if (input_graph_batch and gb.require_magnetic)  else None,
            k_hop_mask = input_graph_batch.get('k_hop_mask')             if (input_graph_batch and gb.k_hop > 0)         else None,
            cache_dict = input_graph_batch,
        )

        if node_bias is not None and node_ids is not None:
            token_bias     = expand_node_to_token_bias(node_bias, node_ids, query_states.shape[2], key_states.shape[2])
            attention_mask = attention_mask + token_bias.to(dtype=query_states.dtype) if attention_mask is not None else token_bias.to(dtype=query_states.dtype)
        # ─────────────────────────────────────────────────────────────────────

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support "
                    "`output_attentions=True`. Falling back to eager attention. This warning "
                    'can be removed using `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout = 0.0 if not self.training else self.attention_dropout,
            scaling = self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ── Decoder layer ─────────────────────────────────────────────────────────────

class LlamaDecoderLayerWithKHop(LlamaDecoderLayerWithBias):
    """
    Decoder layer that swaps in LlamaAttentionWithKHop.

    Inherits forward() from LlamaDecoderLayerWithBias, which already threads
    node_ids and input_graph_batch through to self_attn.
    """

    def __init__(self, config: KHopLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttentionWithKHop(config, layer_idx)


# ── Model stack ───────────────────────────────────────────────────────────────

class LlamaModelWithKHop(LlamaModelWithBias):
    """
    Llama model stack using KHop decoder layers.

    Inherits forward() from LlamaModelWithBias, which handles:
    - Causal mask materialisation
    - Bidirectional prefix mask for non-prompt nodes
    - Propagation of node_ids and input_graph_batch to every decoder layer
    """

    def __init__(self, config: KHopLlamaConfig):
        # Initialise the base LlamaModel (embedding, norm, rotary) then replace layers.
        # We call LlamaModel.__init__ directly to avoid LlamaModelWithBias creating
        # LlamaDecoderLayerWithBias instances that we would immediately discard.
        LlamaModel.__init__(self, config)
        self.layers = nn.ModuleList([
            LlamaDecoderLayerWithKHop(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.post_init()


# ── Causal LM head ────────────────────────────────────────────────────────────

class KHopGraphLlamaForCausalLM(GraphLlamaForCausalLM):
    """
    Causal LM with graph-attention biases and a hard K-hop attention gate.

    API is identical to GraphLlamaForCausalLM.  The only behavioural difference
    is the addition of the K-hop gate when config.k_hop > 0.

    The collator must be instantiated with the matching K value:
        collator = GraphCollator(tokenizer=tok, k_hop=config.k_hop)
    """

    config_class = KHopLlamaConfig

    def __init__(self, config: KHopLlamaConfig):
        # Call LlamaForCausalLM.__init__ directly so that we can install the
        # right model type without GraphLlamaForCausalLM first creating a
        # LlamaModelWithBias that we would immediately replace.
        LlamaForCausalLM.__init__(self, config)
        self._init_requirements(config)
        self.config.architectures = ["KHopGraphLlamaForCausalLM"]
        self.model         = LlamaModelWithKHop(config)
        self.pad_token_id  = config.pad_token_id if config.pad_token_id is not None else config.eos_token_id
        self.post_init()
