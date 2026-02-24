"""
This is an implementation of a custom Llama model that incorporates graph-based attention biases.
There are two types of biases that can be added to the attention scores:
1. Shortest Path Distance (SPD) Bias
    - This bias is based on the shortest path distance between nodes in the graph.
    - It is calculated by this formula:
        b_ij = spd_weights[d_ij] if d_ij > 0 else 0
    - Trainable parameters: spd_weights (num_layers * num_heads * max_spd total parameters)
2. Spectral Bias
    - This bias is based on the spectral coordinates of the nodes in the graph
    - It is calculated by this formula:
        b_ij = w_k * d_ij
      where d_ij is the L2 distance between the spectral coordinates of nodes i and j
      (spectral coordinates are the eigenvectors of the graph Laplacian)
    - Trainable parameters: spectral_weights (num_layers * num_heads total parameters)
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.text_graph_dataset import TextGraphDataset, TextGraph, generate_text_graph_example, prepare_example_labels

from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
    FlashAttentionKwargs,
    CausalLMOutputWithPast,
    logger, 
    eager_attention_forward,
)
from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaAttentionWithBias(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, require_spd=False, require_spectral=False, max_spd=20):
        super().__init__(config, layer_idx)
        
        # initialise all additional parameters needed for the attention bias
        self.require_spd = require_spd
        self.require_spectral = require_spectral

        self.num_heads = config.num_attention_heads
        self.max_spd = max_spd

        # Set of lookup embeddings mapping distances {0, 1,... max_spd-1, ≥max_spd} to bias values for each attention head separately
        # Note that we use nn.Parameter, so that the initialisation doesn't get overridden by the HF's _init_weights method
        if self.require_spd:
            self.spd_weights = nn.Parameter(self._initial_spd_weights(max_spd, config.num_attention_heads, epsilon=0))  # This represents distances 1 to max_spd (inclusive)
        
        # One weight value per attention head to map the spectral coordinate distance d_ij to an attention bias value of b_ij = w_k * d_ij
        if self.require_spectral:
            self.spectral_weights = nn.Parameter(self._initial_spectral_weights(config.num_attention_heads, epsilon=0))

    def _initial_spd_weights(self, max_spd, num_heads, epsilon=1.0):
        """
            Initialize the shortest path distance (SPD) weights with a simple, yet logical starting point.
            The bias should be more negative as the distance increases, for example from 0 to -epsilon linearly.
            This initially encourages the model to pay more attention to closer nodes and less attention to farther nodes, which is a reasonable inductive bias to introduce at the start of training.
        """
        spd_weights = torch.zeros(max_spd, num_heads)
        for dist in range(max_spd):
            bias_value = -epsilon * (dist+1) / max_spd  # Linearly decreasing bias from -epsilon/max_spd to -epsilon
            spd_weights[dist] = bias_value
        return spd_weights

    def _initial_spectral_weights(self, num_heads, epsilon=0.02):
        """
            Initialize the spectral weights to small random values.
        """
        return torch.randn(num_heads) * epsilon

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[any] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # CUSTOM ARGUMENTS
        node_ids: Optional[torch.LongTensor] = None,
        shortest_path_distances: Optional[List[torch.Tensor]] = None,
        spectral_coordinates: Optional[List[torch.Tensor]] = None,
        
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # =====================================================================
        # CUSTOM LOGIC: Construct and add the graph bias
        # =====================================================================
        batch_size, q_len = query_states.shape[0], query_states.shape[2]
        kv_len = key_states.shape[2]

        # print("DEBUG: Inside LlamaAttentionWithBias.forward")
        # print("batch_size:", batch_size)
        # print("q_len:", q_len)
        # print("kv_len:", kv_len)

        if (self.require_spd or self.require_spectral) and node_ids is not None:
            device = query_states.device
            dtype = query_states.dtype
            # Initialize empty graph bias tensor: (batch_size, num_heads, q_len, kv_len)
            graph_bias = torch.zeros(batch_size, self.num_heads, q_len, kv_len, device=device, dtype=dtype)

            for batch_idx in range(batch_size):
                # Accommodate text generation where q_len is 1 but kv_len is the full cached history
                b_node_ids_q = node_ids[batch_idx, -q_len:]
                b_node_ids_kv = node_ids[batch_idx, -kv_len:]

                node_bias = 0
                
                # 1. Calculate SPD Bias
                if self.require_spd and shortest_path_distances is not None:
                    spd_mat = shortest_path_distances[batch_idx].to(device)
                    
                    # Create a mask for where distance is > 0 (False/0 on the diagonal; True/1 elsewhere)
                    non_zero_mask = (spd_mat > 0)
                    
                    # Clamp and Shift - Distances [1...max] become indices [0...max-1] (max_spd == self.spd_weights.shape[0])
                    spd_indices = torch.clamp(spd_mat - 1, min=0, max=self.spd_weights.shape[0] - 1)

                    # Perform lookup
                    spd_b = F.embedding(spd_indices, self.spd_weights).permute(2, 0, 1).to(dtype)
                    
                    # Zero out the entries where distance was originally 0 (non_zero_mask is (N, N), spd_b is (H, N, N))
                    spd_b = spd_b * non_zero_mask.unsqueeze(0)
                    
                    node_bias = node_bias + spd_b

                # 2. Calculate Spectral Bias
                if self.require_spectral and spectral_coordinates is not None:
                    spec_coords = spectral_coordinates[batch_idx].to(device)
                    # Compute pairwise L2 distance -> (num_nodes, num_nodes)
                    spec_dist = torch.cdist(spec_coords, spec_coords, p=2.0)
                    # Multiply by learned weights per head -> (num_heads, num_nodes, num_nodes)
                    spec_b = spec_dist.unsqueeze(0) * self.spectral_weights.view(-1, 1, 1).to(dtype)
                    node_bias = node_bias + spec_b

                # # modify the value of the graph_bias to easier inspect the steps that follow (for debugging)
                # for i in range(node_bias.shape[1]):
                #     for j in range(node_bias.shape[2]):
                #         node_bias[:, i, j] = i+j/10.0 # this will result in outputs i,j

                # 3. Expand Node-level bias to Token-level using advanced indexing
                if isinstance(node_bias, torch.Tensor):
                    idx_q = b_node_ids_q.unsqueeze(1)   # Shape: (q_len, 1)
                    idx_kv = b_node_ids_kv.unsqueeze(0) # Shape: (1, kv_len)
                    
                    # This broadcasts the N x N node matrix into the T x T token matrix
                    token_bias = node_bias[:, idx_q, idx_kv] # Shape: (num_heads, q_len, kv_len)

                    graph_bias[batch_idx] = token_bias

            # 4. Inject our custom graph bias into the existing attention mask
            if attention_mask is None:
                attention_mask = graph_bias
            else:
                attention_mask = attention_mask + graph_bias
        # =====================================================================

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayerWithBias(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, require_spd=False, require_spectral=False, max_spd=None):
        super().__init__(config, layer_idx)

        self.self_attn = LlamaAttentionWithBias(
            config=config, 
            layer_idx=layer_idx, 
            require_spd=require_spd, 
            require_spectral=require_spectral,
            max_spd=max_spd,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[any] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,

        # CUSTOM ARGUMENT propagated here
        node_ids: Optional[torch.LongTensor] = None,
        shortest_path_distances: Optional[List[torch.Tensor]] = None,
        spectral_coordinates: Optional[List[torch.Tensor]] = None,

        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass the custom argument into self_attn
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            
            # CUSTOM ARGUMENTS
            node_ids=node_ids,
            shortest_path_distances=shortest_path_distances,
            spectral_coordinates=spectral_coordinates,

            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaModelWithBias(LlamaModel):
    def __init__(self, config: LlamaConfig, require_spd=False, require_spectral=False, max_spd=None):
        super().__init__(config)
        # Re-initialize the layers using the LlamaDecoderLayerWithBias
        self.layers = nn.ModuleList([
            LlamaDecoderLayerWithBias(
                config, 
                layer_idx=layer_idx, 
                require_spd=require_spd, 
                require_spectral=require_spectral,
                max_spd=max_spd
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
        # Pass graph information as a custom argument to the model
        node_ids: Optional[torch.LongTensor] = None,
        shortest_path_distances: Optional[List[torch.Tensor]] = None,
        spectral_coordinates: Optional[List[torch.Tensor]] = None,

        **flash_attn_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        #region DEBUGGING PRINTS (inspecting the attention masks)
        # print("DEBUG: Inside LlamaModelWithBias.forward")
        # print("Causal Mask shape:", causal_mask.shape)
        # for batch_i in range(causal_mask.shape[0]):
        #     print(f"Causal Mask for element {batch_i}:")
        #     for i in range(causal_mask.shape[2]):
        #         for j in range(causal_mask.shape[3]):
        #             if causal_mask[batch_i, 0, i, j] == 0:
        #                 print("1", end="")
        #             elif causal_mask[batch_i, 0, i, j] == float('-inf'):
        #                 print("0", end="")
        #             else:
        #                 raise ValueError(f"Unexpected value in causal mask: {causal_mask[batch_i, 0, i, j]}")
        #         print()
        # exit()
        #endregion

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # LOOP THROUGH LAYERS
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,

                    # Graph data used for calculating the attention bias
                    node_ids,
                    shortest_path_distances,
                    spectral_coordinates,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,

                    # Graph data used for calculating the attention bias
                    node_ids=node_ids,
                    shortest_path_distances=shortest_path_distances,
                    spectral_coordinates=spectral_coordinates,

                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class GraphLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, bias_type="combined", max_spd=32):
        """
            Custom LlamaForCausalLM that integrates the LlamaModelWithBias to handle graph-based attention biases.

            Arguments:
                config      --> Standard LlamaConfig
                bias_type   --> "spd" (shortest path distance), "laplacian" (laplacian spectral coordinates), or "combined" (both)
        """
        super().__init__(config)

        if bias_type not in ["none", "spd", "laplacian", "combined"]:
            raise ValueError(f"Invalid bias_type: {bias_type}. Must be one of ['none', 'spd', 'laplacian', 'combined']")

        self.bias_type = bias_type
        self._init_requirements(bias_type)

        # Swap the internal model
        self.model = LlamaModelWithBias(
            config, 
            require_spd=self.require_spd, 
            require_spectral=self.require_spectral,
            max_spd=max_spd
        )

        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else config.eos_token_id
        print("Using pad_token_id:", self.pad_token_id)
        
        self.post_init()

    def _init_requirements(self, bias_type):
        """
        Set flags for what additional inputs are required based on the bias type.
        """
        if bias_type == "none":
            self.require_spd = False
            self.require_spectral = False
        elif bias_type == "spd":
            self.require_spd = True
            self.require_spectral = False
        elif bias_type == "laplacian":
            self.require_spd = False
            self.require_spectral = True
        else:
            self.require_spd = True
            self.require_spectral = True

    def _prepare_inputs(
        self, 
        input_ids, 
        prompt_node
    ) -> Dict[str, Any]:
        """
            Prepare input_ids and position_ids for the model based on the prompt node.

            Arguments:
                input_ids   --> List of lists of tensors with shape (seq_len,) for each node in each graph in the batch
                prompt_node --> Tensor of shape (batch_size,) containing the index of the prompt node for each graph in the batch
            Returns:
                Dictionary containing:
                - prepared_input_ids --> Tensor of shape (batch_size, max_total_seq_length) with input_ids properly arranged
                - position_ids       --> Tensor of shape (batch_size, max_total_seq_length) with position ids (eg. [0 1 2 3 0 1 2 0 1 2 3 4 ...])
                - node_ids           --> Tensor of shape (batch_size, max_total_seq_length) indicating which node each token belongs to (eg. [0 0 0 0 1 1 1 2 2 2 2 2 ...])
                - attention_mask     --> Tensor of shape (batch_size, 1, max_total_seq_length, max_total_seq_length) with bidirection attention mask on prefix nodes and causal mask on prompt node
                - prefix_lengths     --> List of length batch_size containing the prefix length for each graph (number of tokens in non-prompt nodes)
                - prompt_lengths     --> List of length batch_size containing the prompt length for each graph (number of tokens in the prompt node)
        """
        batch_size = len(input_ids)
        device = input_ids[0][0].device  # A

        # Create a list of lists of tensors with values [0, 1, 2, ... seq_len-1] for each node's input_ids
        graph_position_ids = []
        for i in range(batch_size):
            graph_position_ids.append([])
            for node_input_ids in input_ids[i]:  # node_input_ids is of shape (seq_len,)
                seq_len = node_input_ids.shape[0]
                graph_position_ids[i].append(torch.arange(seq_len, device=device))

        # Find the maximum total sequence length across all graphs in the batch
        max_total_seq_len = max([sum([ids.shape[0] for ids in graph_input_ids]) for graph_input_ids in input_ids])

        token_ids_list = []
        position_ids_list = []
        node_ids_list = []
        prefix_lengths = []
        prompt_lengths = []
        attention_mask = torch.full((batch_size, 1, max_total_seq_len, max_total_seq_len), float('-inf'), device=device)
        for i in range(batch_size):
            graph_input_ids = input_ids[i] # list of num_nodes tensors of shape (seq_len_j,)
            prompt_idx = prompt_node[i].item()

            # append node ids in the order of non-prompt nodes first, then prompt node at the end
            node_ids_list.append(torch.cat(
                [
                    torch.full((ids.shape[0],), j, dtype=torch.long, device=device) 
                    for j, ids in enumerate(graph_input_ids) if j != prompt_idx
                ] + [
                    torch.full((graph_input_ids[prompt_idx].shape[0],), prompt_idx, dtype=torch.long, device=device)
                ]
            ))

            # append token ids and position ids in the order of non-prompt nodes first, then prompt node at the end
            token_ids_list.append(torch.cat(
                [graph_input_ids[j] for j in range(len(graph_input_ids)) if j != prompt_idx] + [graph_input_ids[prompt_idx]]
            ))
            position_ids_list.append(torch.cat(
                [graph_position_ids[i][j] for j in range(len(graph_input_ids)) if j != prompt_idx] + [graph_position_ids[i][prompt_idx]]
            ))

            prefix_length = sum([graph_input_ids[j].shape[0] for j in range(len(graph_input_ids)) if j != prompt_idx])
            prompt_length = graph_input_ids[prompt_idx].shape[0]
            prefix_lengths.append(prefix_length)
            prompt_lengths.append(prompt_length)

            # Update attention mask for this graph
            attention_mask[i, 0, :prefix_length, :prefix_length] = 0  # Prefix nodes can attend to each other
            attention_mask[i, 0, prefix_length:prefix_length+prompt_length, :prefix_length] = 0  # Prompt node can attend to prefix nodes
            attention_mask[i, 0, prefix_length:prefix_length+prompt_length, prefix_length:prefix_length+prompt_length] = torch.full((prompt_length, prompt_length), float('-inf')).triu(diagonal=1)  # Prompt node has causal attention to itself
            attention_mask[i, 0, prefix_length+prompt_length:, prefix_length+prompt_length:] = torch.full((max_total_seq_len - prefix_length - prompt_length, max_total_seq_len - prefix_length - prompt_length), float('-inf')).fill_diagonal_(0.0)  # Padding tokens can attend only to themselves to avoid NaNs in attention weights

        prepared_token_ids = torch.full((batch_size, max_total_seq_len), self.pad_token_id, dtype=torch.long, device=device)
        prepared_position_ids = torch.zeros((batch_size, max_total_seq_len), dtype=torch.long, device=device)
        prepared_node_ids = torch.full((batch_size, max_total_seq_len), 0, dtype=torch.long, device=device)
        for i in range(batch_size):
            seq_len = token_ids_list[i].shape[0]
            prepared_token_ids[i, :seq_len] = token_ids_list[i]
            prepared_position_ids[i, :seq_len] = position_ids_list[i]

            prepared_node_ids[i, :] = prompt_node[i].item()     # Set all tokens to the prompt node id by default
            prepared_node_ids[i, :seq_len] = node_ids_list[i]   # Then overwrite with the correct node ids for the actual tokens

        #region DEBUGGING PRINTS
        # print("MAX TOTAL SEQ LEN:", max_total_seq_len)
        # print('-' * 50)
        # for i in range(batch_size):
        #     print(f"Graph {i}:")
        #     print(f"  Prompt Node Index: {prompt_node[i].item()}")
        #     print(f"  Prefix Length: {prefix_lengths[i]}")
        #     print(f"  Prompt Length: {prompt_lengths[i]}")
        #     print(f"  Input IDs:")
        #     for j, node_input_ids in enumerate(input_ids[i]):
        #         print(f"    Node {j}: {node_input_ids.tolist()}")
        #     print(f"  Position IDs:")
        #     for j, node_position_ids in enumerate(graph_position_ids[i]):
        #         print(f"    Node {j}: {node_position_ids.tolist()}")
        #     print(f"  Prepared Input IDs:\n {prepared_token_ids[i].tolist()}")
        #     print(f"  Prepared Position IDs:\n {prepared_position_ids[i].tolist()}")
        #     print(f"  Node IDs:\n {prepared_node_ids[i].tolist()}")
        #     print(f"  Prepared Attention Mask Shape: {attention_mask[i].shape}")
        #     for j in range(max_total_seq_len):
        #         print("{:4d}: ".format(j), end="")
        #         for k in range(max_total_seq_len):
        #             if attention_mask[i, 0, j, k] == 0:
        #                 print("1", end="")
        #             elif attention_mask[i, 0, j, k] == float('-inf'):
        #                 print(f"0", end="")
        #             else:
        #                 raise ValueError(f"Unexpected value in attention mask: {attention_mask[i, 0, j, k]}")
        #         print()
        #     print("-" * 50)
        # exit()
        #endregion

        return {
            'input_ids': prepared_token_ids,
            'position_ids': prepared_position_ids,
            'node_ids': prepared_node_ids,
            'attention_mask': attention_mask,
            'prefix_lengths': prefix_lengths,
            'prompt_lengths': prompt_lengths,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,

        # graph-specific inputs
        input_graph_batch: Optional[TextGraph] = None,

        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            Modified forward method to accept the text attributed graph as input.

            Arguments:
                input_ids          --> None (will be overridden)
                attention_mask     --> None (will be overridden)
                position_ids       --> None (will be overridden)
                labels             --> Labels for calculating the loss on the prompt node tokens only. Shoudl be a list of batch_size elements, each being a tensor of shape (prompt_node_seq_len,) containing the labels for the prompt node tokens in each graph.
                input_graph_batch  --> A dictionary containing the batched graph information for the current batch, expected to have keys like:
                    'text'                  - the raw text input (not tokenized)        ---> List of batch_size elements, each being a string of the raw text for the graph
                    'num_nodes'             - number of nodes in the graph              ---> Tensor of shape (batch_size,) containing the number of nodes for each graph in the batch
                    'prompt_node'           - index of the prompt node in the graph     ---> Tensor of shape (batch_size,) containing the index of the prompt node for each graph in the batch
                    'input_ids'             - tokenized input ids for the text          ---> List of batch_size elements, each being a list of shape num_nodes elements, each being a tensor of shape (seq_len_i_j) representing the tokenized input for each node
                    'edges'                 - list of edges in the graph                ---> List of batch_size elements, each being a tensor of shape (num_edges, 2) representing the edges between nodes
                    'spectral_coords'       - precomputed spectral coordinates          ---> List of batch_size elements, each being a tensor of shape (num_nodes, spectral_dim)
                    'shortest_path_dists'   - precomputed shortest path distances       ---> List of batch_size elements, each being a tensor of shape (num_nodes, num_nodes) representing the pairwise shortest path distances between nodes
                ...(other arguments are the same as the original forward method)
        """

        prompt_node = input_graph_batch.get('prompt_node', None)
        if prompt_node is None:
            raise ValueError("input_graph_batch must contain 'prompt_node' key with the index of the prompt node for each graph in the batch.")

        input_ids = input_graph_batch.get('input_ids', None)
        if input_ids is None:
            raise ValueError("input_graph_batch must contain 'input_ids' key with the tokenized input ids for each node in each graph in the batch.")

        # Validate required inputs based on bias type (set to None if not required)
        shortest_path_distances = input_graph_batch.get('shortest_path_dists', None)
        if self.require_spd:
            if shortest_path_distances is None:
                raise ValueError("Bias type requires shortest path distances, but 'shortest_path_dists' is missing in input_graph_batch.")
        else:
            shortest_path_distances = None

        spectral_coordinates = input_graph_batch.get('spectral_coords', None)
        if self.require_spectral:
            if spectral_coordinates is None:
                raise ValueError("Bias type requires spectral coordinates, but 'spectral_coords' is missing in input_graph_batch.")
        else:
            spectral_coordinates = None

        # Prepare the inputs for the forward pass
        prepared_inputs = self._prepare_inputs(input_ids, prompt_node)
        prepared_input_ids = prepared_inputs['input_ids']
        prepared_position_ids = prepared_inputs['position_ids']
        node_ids = prepared_inputs['node_ids']
        attention_mask = prepared_inputs['attention_mask']
        prefix_lengths = prepared_inputs['prefix_lengths']
        prompt_lengths = prepared_inputs['prompt_lengths']
        
        # Pass the custom argument to the model
        outputs = self.model(
            input_ids=prepared_input_ids,               # <-- concatenated tokens of all nodes in the graph
            attention_mask=attention_mask,              # <-- custom attention mask based on graph structure
            position_ids=prepared_position_ids,         # <-- position ids that reset for each node
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            
            # CUSTOM ARGUMENTS
            node_ids=node_ids,                               # <-- node ids indicating which node each token belongs to (used for calculating attention bias)
            shortest_path_distances=shortest_path_distances, # <-- pass shortest path distances if required
            spectral_coordinates=spectral_coordinates,       # <-- pass spectral coordinates if required

            **kwargs,
        )

        hidden_states = outputs[0]
        # Standard LM Head Logic
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            if not isinstance(labels, list) or len(labels) != prepared_input_ids.shape[0]:
                raise ValueError(f"Labels should be a list of length {prepared_input_ids.shape[0]}, where each element is a tensor of shape (prompt_node_seq_len,) containing the labels for the prompt node tokens in each graph.")

            batch_size, max_total_seq_len = prepared_input_ids.shape[0], prepared_input_ids.shape[1]
            full_labels = torch.full((batch_size, max_total_seq_len), -100, dtype=torch.long, device=prepared_input_ids.device)

            for i in range(batch_size):
                prompt_len, prefix_len = prompt_lengths[i], prefix_lengths[i]

                if prompt_len != labels[i].shape[0]:
                    raise ValueError(f"Length of labels for graph {i} does not match the prompt length. Expected {prompt_len}, got {labels[i].shape[0]}.")
                
                full_labels[i, prefix_len:prefix_len+prompt_len] = labels[i]

            loss = self.loss_function(logits=logits, labels=full_labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    graph_dataset_sample = generate_text_graph_example(
        dataset_size=3, 
        base_num_nodes=5, 
        calc_attributes=True, 
        tokenizer=tokenizer, 
        spec_emb_dim=4
    )
    example_labels = prepare_example_labels(graph_dataset_sample)

    from ..utils.text_graph_collator import GraphCollator
    collator = GraphCollator(tokenizer=tokenizer)
    input_graph_batch = collator([ graph_dataset_sample[i] for i in range(len(graph_dataset_sample)) ])

    model = GraphLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", bias_type="combined")
    print('=' * 70)
    print('=' * 70)
    outputs = model(input_graph_batch=input_graph_batch, labels=example_labels)

    # run the model to generate 5 tokens autoregressively
    # TODO: test once generation is implemented in the model