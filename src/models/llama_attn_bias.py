"""
    This is an implementation of a custom Llama model that incorporates graph-based attention biases.
    There are four types of biases that can be added to the attention scores:

    1. Shortest Path Distance (SPD) Bias
        - This bias is based on the shortest path distance between nodes in the graph.
        - It is calculated by this formula:
            b_ij = spd_weights[d_ij] if d_ij > 0 else 0
        - Trainable parameters: spd_weights (num_layers * num_heads * max_spd total parameters)

    2. Laplacian Bias
        - This bias is based on the Laplacian spectral coordinates of the nodes in the graph.
        - It is calculated by this formula:
            b_ij = w_k * d_ij
        where d_ij is the L2 distance between the Laplacian spectral coordinates of nodes i and j
        (Laplacian spectral coordinates are the eigenvectors of the graph Laplacian).
        - Trainable parameters: laplacian_weights (num_layers * num_heads total parameters)

    3. Random Walk Structural Encoding (RWSE) Bias
        - This bias is based on the random walk structural features of the nodes.
        - It is calculated by this formula:
            b_ij = w_k * d_ij
        where d_ij is the L2 distance between the RWSE feature vectors of nodes i and j.
        - Trainable parameters: rwse_weights (num_layers * num_heads total parameters)

    4. Relative Random Walk Probability (RRWP) Bias
        - This bias is based on the multi-hop probability of a random walk starting at node i and landing on node j.
        - It maps the vector of transition probabilities (from 1 to max_rw_steps) into a bias value for each attention head.
        - It is calculated by this formula:
            b_ij = MLP(RRWP_ij) if i != j else 0
        where the MLP is a 2-layer network (Linear -> SiLU -> Linear) mapping the max_rw_steps vector directly to num_heads.
        - Trainable parameters: rrwp_proj (num_layers * (layer_1_weights + layer_2_weights) total parameters)

    5. Magnetic Laplacian Bias
        - This bias is based on the magnetic Laplacian spectral coordinates of the nodes in the graph, which capture directed structural information.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.text_graph_dataset import TextGraphDataset, TextGraph, generate_text_graph_example, prepare_example_labels
from .llama_utils import save_bias_parameters, load_bias_parameters
from peft import PeftModel

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

import warnings

from transformers import LlamaConfig

import json
from transformers import LlamaConfig
from accelerate.utils import send_to_device

#region Graph Llama Config
class GraphAttnBiasConfig:
    def __init__(self, spd=False, laplacian=False, max_spd=32, rwse=False, rrwp=False, max_rw_steps=8, magnetic=False, magnetic_dim=32, magnetic_q=0.25):
        self.spd = spd
        self.laplacian = laplacian
        self.max_spd = max_spd
        self.rwse = rwse
        self.rrwp = rrwp
        self.max_rw_steps = max_rw_steps
        self.magnetic = magnetic
        self.magnetic_dim = magnetic_dim
        self.magnetic_q = 0.25

    def to_dict(self):
        return {
            "spd": self.spd,
            "laplacian": self.laplacian,
            "max_spd": self.max_spd,
            "rwse": self.rwse,
            "rrwp": self.rrwp,
            "max_rw_steps": self.max_rw_steps,
            "magnetic": self.magnetic,
            "magnetic_dim": self.magnetic_dim,
            "magnetic_q": self.magnetic_q
        }

    def __str__(self):
        return json.dumps(self.to_dict())

class GraphLlamaConfig(LlamaConfig):
    model_type = "graph_llama"

    def __init__(
        self, 
        spd=False, 
        laplacian=False, 
        max_spd=32, 
        rwse=False, 
        rrwp=False,
        max_rw_steps=8,
        graph_attn_bias=None, 
        magnetic=False,
        magnetic_dim=32,
        **kwargs
    ):
        # 1. Initialize standard Llama parameters
        super().__init__(**kwargs)

        # 2. Handle reconstruction from JSON
        # When loaded from JSON, the data comes in as a dict named 'graph_attn_bias'
        if isinstance(graph_attn_bias, dict):
            self.graph_attn_bias = GraphAttnBiasConfig(**graph_attn_bias)
        else:
            # When initialized manually (e.g. from_pretrained(..., spd=True))
            # or when 'graph_attn_bias' is missing from JSON
            self.graph_attn_bias = GraphAttnBiasConfig(
                spd=spd, 
                laplacian=laplacian, 
                max_spd=max_spd, 
                rwse=rwse,
                rrwp=rrwp,
                max_rw_steps=max_rw_steps,
                magnetic=magnetic,
                magnetic_dim=magnetic_dim
            )

    def to_dict(self):
        """Ensures the nested object is serialized as a dict, not a class instance."""
        output = super().to_dict()
        if hasattr(self, "graph_attn_bias"):
            output["graph_attn_bias"] = self.graph_attn_bias.to_dict()
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 1. Get the dictionary from the local path or Hub
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        
        # 2. Check if this is one of YOUR models (it will have the graph_attn_bias key)
        # If it's a standard Llama model, this key won't exist.
        graph_saved_data = config_dict.get("graph_attn_bias", {})

        # 3. Priority logic: 
        #    - First: Use values explicitly passed in kwargs (user overrides)
        #    - Second: Use values saved in the config_dict (saved model settings)
        #    - Third: Use the default values
        spd = kwargs.pop("spd", graph_saved_data.get("spd", None))
        laplacian = kwargs.pop("laplacian", graph_saved_data.get("laplacian", None))
        max_spd = kwargs.pop("max_spd", graph_saved_data.get("max_spd", None))
        rwse = kwargs.pop("rwse", graph_saved_data.get("rwse", None))
        rrwp = kwargs.pop("rrwp", graph_saved_data.get("rrwp", None))
        max_rw_steps = kwargs.pop("max_rw_steps", graph_saved_data.get("max_rw_steps", None))
        magnetic = kwargs.pop("magnetic", graph_saved_data.get("magnetic", None))
        magnetic_dim = kwargs.pop("magnetic_dim", graph_saved_data.get("magnetic_dim", 32))
        magnetic_q = kwargs.pop("magnetic_q", graph_saved_data.get("magnetic_q", 0.25))

        # 4. Return the instance
        return cls(
            spd=spd, 
            max_spd=max_spd, 
            laplacian=laplacian, 
            rwse=rwse, 
            rrwp=rrwp,
            max_rw_steps=max_rw_steps,
            magnetic=magnetic,
            magnetic_dim=magnetic_dim,
            magnetic_q=magnetic_q,
            **config_dict
        )
#endregion

class LlamaAttentionWithBias(LlamaAttention):
    def __init__(self, config: GraphLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # initialise all additional parameters needed for the attention bias
        graph_attn_bias_config = getattr(config, "graph_attn_bias", None)  # Get the graph_attn_bias config, or use defaults if not present

        if graph_attn_bias_config is None:
            raise ValueError("GraphLlamaConfig must contain a 'graph_attn_bias' attribute with the bias configuration.")

        self.require_spd = getattr(graph_attn_bias_config, 'spd', False)
        self.max_spd = getattr(graph_attn_bias_config, 'max_spd', 32)
        self.require_laplacian = getattr(graph_attn_bias_config, 'laplacian', False)
        self.require_rwse = getattr(graph_attn_bias_config, 'rwse', False)
        self.require_rrwp = getattr(graph_attn_bias_config, 'rrwp', False)
        self.max_rw_steps = getattr(graph_attn_bias_config, 'max_rw_steps', 8)
        self.require_magnetic = getattr(graph_attn_bias_config, 'magnetic', False)
        self.magnetic_dim = getattr(graph_attn_bias_config, 'magnetic_dim', 32)

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        # Set of lookup embeddings mapping distances {0, 1,... max_spd-1, ≥max_spd} to bias values for each attention head separately
        # Note that we use nn.Parameter, so that the initialisation doesn't get overridden by the HF's _init_weights method
        if self.require_spd:
            self.spd_weights = nn.Parameter(self._initial_spd_weights(self.max_spd, config.num_attention_heads, epsilon=0))  # This represents distances 1 to max_spd (inclusive)
        
        # One weight value per attention head to map the spectral coordinate distance d_ij to an attention bias value of b_ij = w_k * d_ij
        if self.require_laplacian:
            self.laplacian_weights = nn.Parameter(self._initial_laplacian_weights(config.num_attention_heads, epsilon=0))

        # One weight value per attention head to map the Random Walk Structural Encoding distance to an attention bias value of b_ij = w_k * t_ij
        if self.require_rwse:
            # use the same initialization as the laplacian weights since they serve a similar purpose (mapping a distance to a bias value)
            self.rwse_weights = nn.Parameter(self._initial_laplacian_weights(config.num_attention_heads, epsilon=0))

        # Linear transformation mapping RRWP_ij to Linear(RRWP_ij) (shared between all heads of each layer) + another Linear mapping to b_ij for each head (unique weights for each head) 
        if self.require_rrwp:
            hidden_rrwp_dim = 4 * self.max_rw_steps
            
            # Use an nn.Sequential block to combine the projections and the activation
            self.rrwp_proj = nn.Sequential(
                nn.Linear(self.max_rw_steps, hidden_rrwp_dim, bias=True),
                nn.SiLU(),
                # initialise the second layer weights to 0 so that the initial bias is 0 and doesn't interfere with training at the start
                nn.Linear(hidden_rrwp_dim, config.num_attention_heads, bias=True)
            )
            # Initialize the second layer's parameters to 0
            second_linear = self.rrwp_proj[2]
            nn.init.zeros_(second_linear.weight)
            nn.init.zeros_(second_linear.bias)
            second_linear._is_hf_initialized = True

        # MLP( V @ diag(phi(lambdas)) @ conj(V.T) )
        if self.require_magnetic:
            # Creating the eigenvalue-based features
            self.magnetic_lambda_lin = nn.Linear(1, self.head_dim, bias=True) # R --> R^m_dim: lambda_i --> h_lambda_i

            # The deep set architecture to process the eigenvalues in a permutation-equivariant way
            self.magnetic_lambda_deep_set = nn.Sequential(
                nn.Linear(self.head_dim*2, self.magnetic_dim, bias=True),
                nn.SiLU(),
            ) # R^(*2m_dim) --> R^m_dim:  concat(h_lambda_i, avg(h_lambda)) --> phi(lambda_i)

            # The final MLP to map the magnetic features to bias values for each head
            self.magnetic_bias_proj = nn.Sequential(
                nn.Linear(self.magnetic_dim * 2, self.magnetic_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.magnetic_dim, self.num_heads, bias=True)
            ) # R^m_dim*2 --> R^head_dim:   b_ij = MLP( V @ diag(phi(lambdas)) @ conj(V.T) )
            # (multiply by 2 to process both the real and imaginary parts of the magnetic features together)

            # initialise the final linear layer to 0 so that the initial bias is 0 and doesn't interfere with training at the start
            final_linear = self.magnetic_bias_proj[2]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
            final_linear._is_hf_initialized = True


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

    def _initial_laplacian_weights(self, num_heads, epsilon=0.02):
        """
            Initialize the spectral weights to small random values.
        """
        return torch.randn(num_heads) * epsilon

    def _validate_input_graph_batch(self, input_graph_batch):
        # Validate that input_graph_batch contains the necessary information based on the required bias types
        if input_graph_batch is None:
            raise ValueError("input_graph_batch is required but is None.")

        # Check for num_nodes field
        if 'num_nodes' not in input_graph_batch:
            raise ValueError("input_graph_batch must contain 'num_nodes' field indicating the number of nodes in each graph in the batch.")
        
        # Check bias-type-specific requirements
        if self.require_spd and (input_graph_batch is None or 'shortest_path_dists' not in input_graph_batch):
            raise ValueError("Shortest Path Distance bias is required but 'shortest_path_dists' is missing from input_graph_batch.")
        if self.require_laplacian and (input_graph_batch is None or 'laplacian_coordinates' not in input_graph_batch):
            raise ValueError("Laplacian bias is required but 'laplacian_coordinates' is missing from input_graph_batch.")
        if self.require_rwse and (input_graph_batch is None or 'rwse' not in input_graph_batch):
            raise ValueError("RWSE bias is required but 'rwse' is missing from input_graph_batch.")
        if self.require_rrwp and (input_graph_batch is None or 'rrwp' not in input_graph_batch):
            raise ValueError("RRWP bias is required but 'rrwp' is missing from input_graph_batch.")
        if self.require_magnetic and (input_graph_batch is None or 'magnetic' not in input_graph_batch):
            raise ValueError("Magnetic bias is required but 'magnetic' is missing from input_graph_batch. It should be a tuple with the magnetic eigenvectors (shape BxNxNx2) and eigenvalues (shape N).")

    def compute_bias(
        self,
        query_states: torch.Tensor,                             # (batch_size, num_heads, q_len, head_dim)
        key_states: torch.Tensor,                               # (batch_size, num_heads, kv_len, head_dim)
        node_ids: Optional[torch.LongTensor],                   # (batch_size, seq_len)
        num_nodes: Optional[torch.Tensor],                      # (batch_size,)
        shortest_path_distances: Optional[torch.Tensor],        # (batch_size, max_num_nodes, max_num_nodes)
        laplacian_coordinates: Optional[torch.Tensor],          # (batch_size, max_num_nodes, spec_dim)
        rwse_emb: Optional[torch.Tensor],                       # (batch_size, max_num_nodes, rwse_dim)
        rrwp_emb: Optional[torch.Tensor],                       # (batch_size, max_num_nodes, max_num_nodes, max_rw_steps)
        magnetic: Optional[Tuple[torch.Tensor, torch.Tensor]],  # (batch_size, max_num_nodes, max_num_nodes, 2), (batch_size, max_num_nodes)
        input_graph_batch: Optional[TextGraph] = None,          # the full graph batch, used for caching the computed bias to avoid redundant O(N^2) calculations during generation
    ):
        magnetic_V = magnetic[0] if magnetic is not None else None          # (batch_size, max_num_nodes, max_num_nodes, 2)
        magnetic_lambdas = magnetic[1] if magnetic is not None else None    # (batch_size, max_num_nodes)

        using_graph_bias = self.require_spd or self.require_laplacian or self.require_rwse or self.require_rrwp or self.require_magnetic
        if not using_graph_bias:
            return None  # No bias needed, return None to indicate this
        if node_ids is None:
            raise ValueError("Graph-based bias is required but 'node_ids' is not provided.")

        dtype = query_states.dtype
        device = query_states.device
        
        cache_key = f'cached_node_bias_{self.layer_idx}'
        
        # Bypass O(N^2) node calculations during generation if already cached
        if input_graph_batch is not None and cache_key in input_graph_batch:
            node_bias = input_graph_batch[cache_key]
        else:
            node_bias = 0

            if self.require_spd:
                if shortest_path_distances is None: raise ValueError("Shortest Path Distance bias is required but 'shortest_path_distances' is not provided.")
                non_zero_mask = (shortest_path_distances > 0)
                spd_indices = torch.clamp(shortest_path_distances - 1, min=0, max=self.spd_weights.shape[0] - 1) # mapping D -> D-1 for indexing in spd_weights
                
                spd_bias = F.embedding(spd_indices, self.spd_weights).permute(0, 3, 1, 2).to(dtype=dtype) # (batch_size, num_heads, max_num_nodes, max_num_nodes)
                spd_bias = spd_bias * non_zero_mask.unsqueeze(1)  # Zero out entries where distance was originally 0
                node_bias = node_bias + spd_bias

            if self.require_laplacian:
                if laplacian_coordinates is None: raise ValueError("Laplacian bias is required but 'laplacian_coordinates' is not provided.")
                lap_dist = torch.cdist(laplacian_coordinates, laplacian_coordinates, p=2.0) # (batch_size, max_num_nodes, max_num_nodes)
                lap_bias = lap_dist.unsqueeze(1) * self.laplacian_weights.view(-1, 1, 1).to(dtype) # (batch_size, num_heads, max_num_nodes, max_num_nodes)
                node_bias = node_bias + lap_bias

            if self.require_rwse:
                if rwse_emb is None: raise ValueError("RWSE bias is required but 'rwse_emb' is not provided.")
                rwse_dist = torch.cdist(rwse_emb, rwse_emb, p=2.0) # (batch_size, max_num_nodes, max_num_nodes)
                rwse_bias = rwse_dist.unsqueeze(1) * self.rwse_weights.view(-1, 1, 1).to(dtype) # (batch_size, num_heads, max_num_nodes, max_num_nodes)
                node_bias = node_bias + rwse_bias

            if self.require_rrwp:
                if rrwp_emb is None: raise ValueError("RRWP bias is required but 'rrwp_emb' is not provided.")
                rrwp_bias = self.rrwp_proj(rrwp_emb).permute(0, 3, 1, 2).contiguous() # <--- Added .contiguous()
                diag_mask = torch.eye(rrwp_bias.shape[-1], device=device, dtype=torch.bool)
                rrwp_bias = rrwp_bias.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), 0.0) # <--- Removed the underscore
                node_bias = node_bias + rrwp_bias

            if self.require_magnetic:
                if magnetic_V is None or magnetic_lambdas is None: raise ValueError("Magnetic bias is required but 'magnetic' data is not provided.")
                # compute the eigenvalue-based features
                h_i = self.magnetic_lambda_lin(magnetic_lambdas.unsqueeze(-1)) # (batch_size, max_num_nodes, head_dim)

                # create a boolean mask of valid nodes: shape (batch_size, max_num_nodes)
                valid_mask = torch.arange(magnetic_lambdas.shape[1], device=h_i.device).expand(num_nodes.shape[0], -1) < num_nodes.unsqueeze(1) # (batch_size, max_num_nodes) bool

                # mask the invalid nodes out of the sum
                h_i_masked = h_i * valid_mask.unsqueeze(-1) 

                # compute the average of the h_i features (only across the valid nodes)
                h_avg = h_i_masked.sum(dim=1, keepdim=True) / num_nodes.view(-1, 1, 1).to(h_i.dtype) # (batch_size, 1, head_dim) / (batch_size, 1, 1) -> (batch_size, 1, head_dim)
                h_avg_expanded = h_avg.expand(-1, magnetic_lambdas.shape[1], -1) # (batch_size, max_num_nodes, head_dim)

                # compute the magnetic features using a deep set architecture to ensure permutation equivariance
                deep_set_input = torch.cat([h_i, h_avg_expanded], dim=-1) # (batch_size, max_num_nodes, head_dim*2)
                phi_lambdas = self.magnetic_lambda_deep_set(deep_set_input) # (batch_size, max_num_nodes, magnetic_dim)

                # compute the edge-level embeddings using the magnetic eigenvectors and the phi_lambdas features
                V_real = magnetic_V[..., 0] # (batch_size, max_num_nodes, max_num_nodes)
                V_imag = magnetic_V[..., 1] # (batch_size, max_num_nodes, max_num_nodes)

                # Compute V @ diag(phi) @ conj(V.T) efficiently using Einstein summation
                # Real Part: (V_R * Phi * V_R.T) + (V_I * Phi * V_I.T)
                real_part = torch.einsum('bil, bjl, blk -> bijk', V_real, V_real, phi_lambdas) + torch.einsum('bil, bjl, blk -> bijk', V_imag, V_imag, phi_lambdas)
                
                # Imaginary Part: (V_I * Phi * V_R.T) - (V_R * Phi * V_I.T)
                imag_part = torch.einsum('bil, bjl, blk -> bijk', V_imag, V_real, phi_lambdas) - torch.einsum('bil, bjl, blk -> bijk', V_real, V_imag, phi_lambdas)

                # Concatenate real and imaginary parts to form the stable 2m-dim edge features
                magnetic_edge_features = torch.cat([real_part, imag_part], dim=-1) # (batch_size, max_num_nodes, max_num_nodes, magnetic_dim * 2)

                # Project the edge features to scalar bias values for each attention head
                magnetic_bias = self.magnetic_bias_proj(magnetic_edge_features) # (batch_size, max_num_nodes, max_num_nodes, num_heads)

                # Permute to match standard Transformer attention mask shape: (B, H, N, N)
                magnetic_bias = magnetic_bias.permute(0, 3, 1, 2).contiguous() # <--- Added .contiguous()

                # mask the diagonal entries where i=j to 0, since we don't want self-loops to contribute to the bias
                diag_mask = torch.eye(magnetic_bias.shape[-1], device=device, dtype=torch.bool)
                magnetic_bias = magnetic_bias.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), 0.0) # <--- Removed the underscore

                # Add the magnetic bias to the total node_bias
                node_bias = node_bias + magnetic_bias

            # Save to graph batch cache so we don't recompute on token generation steps > 0
            if input_graph_batch is not None:
                input_graph_batch[cache_key] = node_bias

        if type(node_bias) == int and node_bias == 0:
            return None  # No bias was actually computed, return None to indicate this

        # Expanding the node-level biases to token-level biases using advanced indexing
        batch_size, q_len = query_states.shape[0], query_states.shape[2]
        kv_len = key_states.shape[2]

        # Slice node_ids to match the current query and key lengths
        node_ids_q = node_ids[:, -q_len:]
        node_ids_kv = node_ids[:, -kv_len:]

        B = batch_size
        _, H, _, _ = node_bias.shape
        device = node_ids.device

        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1) # Shape: (B, 1, 1, 1)
        head_idx = torch.arange(H, device=device).view(1, H, 1, 1)  # Shape: (1, H, 1, 1)
        q_idx = node_ids_q.view(B, 1, q_len, 1)                     # Shape: (B, 1, q_len, 1)
        k_idx = node_ids_kv.view(B, 1, 1, kv_len)                   # Shape: (B, 1, 1, kv_len)

        token_bias = node_bias[batch_idx, head_idx, q_idx, k_idx]   # Shape: (B, H, q_len, kv_len)

        return token_bias.to(dtype=query_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[any] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # CUSTOM ARGUMENTS
        node_ids: Optional[torch.LongTensor] = None,
        input_graph_batch: Optional[TextGraph] = None,
        
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # Validate required inputs based on bias type (set to None if not required)
        self._validate_input_graph_batch(input_graph_batch)

        num_nodes = input_graph_batch.get('num_nodes', None) if self.require_magnetic else None

        shortest_path_distances = input_graph_batch.get('shortest_path_dists', None) if self.require_spd else None
        laplacian_coordinates = input_graph_batch.get('laplacian_coordinates', None) if self.require_laplacian else None
        rwse_emb = input_graph_batch.get('rwse', None) if self.require_rwse else None
        rrwp_emb = input_graph_batch.get('rrwp', None) if self.require_rrwp else None
        magnetic = input_graph_batch.get('magnetic', None) if self.require_magnetic else (None, None)

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
        custom_bias = self.compute_bias(
            query_states=query_states,
            key_states=key_states,
            node_ids=node_ids,
            num_nodes=num_nodes,
            shortest_path_distances=shortest_path_distances,
            laplacian_coordinates=laplacian_coordinates,
            rwse_emb=rwse_emb,
            rrwp_emb=rrwp_emb,
            magnetic=magnetic,
            input_graph_batch=input_graph_batch,
        ) # (batch_size, num_heads, seq_len, seq_len)
        if custom_bias is not None and attention_mask is not None:
            attention_mask = attention_mask + custom_bias
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
    def __init__(self, config: GraphLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = LlamaAttentionWithBias(
            config=config, 
            layer_idx=layer_idx, 
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
        input_graph_batch: Optional[TextGraph] = None,

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
            input_graph_batch=input_graph_batch,

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
    def __init__(self, config: GraphLlamaConfig):
        super().__init__(config)
        # Re-initialize the layers using the LlamaDecoderLayerWithBias
        self.layers = nn.ModuleList([
            LlamaDecoderLayerWithBias(
                config, 
                layer_idx=layer_idx, 
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
        input_graph_batch: Optional[TextGraph] = None,

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

        # APPLY BIDIRECTIONAL PREFIX MASK ON TOP OF CAUSAL MASK
        # HF creates a strict causal lower-triangle mask. We need to unmask the prefix-to-prefix region.
        if causal_mask is not None and node_ids is not None and input_graph_batch is not None:
            prompt_node = input_graph_batch.get("prompt_node")
            if prompt_node is not None:
                B, _, q_len, kv_len = causal_mask.shape
                # Get the node ids for the current Q and KV tokens
                node_ids_q = node_ids[:, -q_len:].unsqueeze(2) # (B, q_len, 1)
                node_ids_kv = node_ids[:, -kv_len:].unsqueeze(1) # (B, 1, kv_len)
                prompt_node_expanded = prompt_node.view(B, 1, 1).to(node_ids.device)
                
                # Mask where both Q and K are from prefix nodes
                is_prefix_q = node_ids_q != prompt_node_expanded # (B, q_len, kv_len)
                is_prefix_k = node_ids_kv != prompt_node_expanded # (B, q_len, kv_len)
                both_prefix = is_prefix_q & is_prefix_k # (B, q_len, kv_len)
                
                # Unmask (set to 0.0) where both are prefix nodes so they can attend bidirectionally
                causal_mask = causal_mask.masked_fill(both_prefix.unsqueeze(1), 0.0) # (B, 1, q_len, kv_len)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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

                    node_ids,
                    input_graph_batch,
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
                    input_graph_batch=input_graph_batch,

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
    def __init__(
        self, 
        config: GraphLlamaConfig,
        bias_type: Optional[str] = None,
        max_spd: Optional[int] = None,
    ):
        """
            Custom LlamaForCausalLM that integrates the LlamaModelWithBias to handle graph-based attention biases.

            Arguments:
                config      --> GraphLlamaConfig object containing the model configuration (from Llama config) and additional parameters related to the attention bias computation
                bias_type   --> Optional string to specify the type of bias to use. If provided, it will override the corresponding values in the config. Must be one of 'spd', 'laplacian', 'combined' (using those two), 'none', or simply don't specify it and leave as None to use the config values.
                max_spd     --> Optional int to specify the maximum shortest path distance to consider for the SPD bias. If provided, it will override the max_spd value in the config. This is only relevant if bias_type is 'spd' or 'combined'.
        """
        # keep everything backward compatible with earlier versions by allowing the use of bias_type and max_spd arguments and saving the parameter values in the config
        if bias_type is not None:
            # print a warning because the config values will be overridden by the bias_type argument
            warnings.warn(f"bias_type argument is provided ({bias_type}), which will override the corresponding values in the config. If you want to use the config values instead, simply don't specify bias_type and the parameters will be read from the config as intended. This warning can be removed by removing the bias_type argument and relying solely on the config values.")
            if bias_type=='spd':
                self.require_spd = True
                self.require_laplacian = False
                config.spd = True
                config.laplacian = False
            elif bias_type=='laplacian':
                self.require_laplacian = True
                self.require_spd = False
                config.laplacian = True
                config.spd = False
            elif bias_type=='combined':
                self.require_spd = True
                self.require_laplacian = True
                config.spd = True
                config.laplacian = True
            elif bias_type=='none':
                self.require_spd = False
                self.require_laplacian = False
                config.spd = False
                config.laplacian = False
            else:
                raise ValueError(f"Invalid bias_type: {bias_type}. Must be one of 'spd', 'laplacian', 'combined', 'none', or simply don't specify it and leave as None to use the config values.")
        if max_spd is not None:
            warnings.warn(f"max_spd argument is provided ({max_spd}), which will override the max_spd value in the config. If you want to use the config value instead, simply don't specify max_spd and the parameter will be read from the config as intended. This warning can be removed by removing the max_spd argument and relying solely on the config value.")
            self.max_spd = max_spd
            config.max_spd = max_spd

        super().__init__(config)
        self._init_requirements(config)

        # Set the architectures attribute to GraphLlamaForCausalLM instead of LlamaForCauseLM
        self.config.architectures = ["GraphLlamaForCausalLM"]

        # Swap the internal model
        self.model = LlamaModelWithBias(config)

        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else config.eos_token_id
        
        self.post_init()

    def _init_requirements(self, config):
        """
        Set flags for what additional inputs are required based on the bias type.
        """
        # check if config has the following attributes, if not set them to False by default
        self.require_spd = getattr(config, 'spd', False)
        self.require_laplacian = getattr(config, 'laplacian', False)

    def _prepare_inputs(
        self,
        input_ids,
        prompt_node,
        padding_side="right"
    ) -> Dict[str, Any]:
        """
            Prepare input_ids and position_ids for the model based on the prompt node.

            Arguments:
                input_ids   --> List of lists of tensors with shape (seq_len,) for each node in each graph in the batch
                prompt_node --> Tensor of shape (batch_size,) containing the index of the prompt node for each graph in the batch
                padding_side--> "left" for inference, "right" for training
            Returns:
                Dictionary containing:
                - prepared_input_ids --> Tensor of shape (batch_size, max_total_seq_length) with input_ids properly arranged
                - position_ids       --> Tensor of shape (batch_size, max_total_seq_length) with position ids (eg. [0 1 2 3 0 1 2 0 1 2 3 4 ...])
                - node_ids           --> Tensor of shape (batch_size, max_total_seq_length) indicating which node each token belongs to (eg. [0 0 0 0 1 1 1 2 2 2 2 2 ...])
                - attention_mask     --> Tensor of shape (batch_size, max_total_seq_length) (2D standard HF pad mask)
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
        
        # attention_mask = torch.full((batch_size, 1, max_total_seq_len, max_total_seq_len), float('-inf'), device=device)
        attention_mask = torch.zeros((batch_size, max_total_seq_len), dtype=torch.long, device=device)
        
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

        prepared_token_ids = torch.full((batch_size, max_total_seq_len), self.pad_token_id, dtype=torch.long, device=device)
        prepared_position_ids = torch.zeros((batch_size, max_total_seq_len), dtype=torch.long, device=device)
        prepared_node_ids = torch.zeros((batch_size, max_total_seq_len), dtype=torch.long, device=device)
        
        for i in range(batch_size):
            seq_len = token_ids_list[i].shape[0]

            if padding_side == "right":
                prepared_token_ids[i, :seq_len] = token_ids_list[i]
                prepared_position_ids[i, :seq_len] = position_ids_list[i]
                prepared_node_ids[i, :] = prompt_node[i].item()
                prepared_node_ids[i, :seq_len] = node_ids_list[i]
                attention_mask[i, :seq_len] = 1
            else: # left padding for batched inference/generation
                pad_len = max_total_seq_len - seq_len
                prepared_token_ids[i, pad_len:] = token_ids_list[i]
                prepared_position_ids[i, pad_len:] = position_ids_list[i]
                prepared_node_ids[i, :] = prompt_node[i].item()
                prepared_node_ids[i, pad_len:] = node_ids_list[i]
                attention_mask[i, pad_len:] = 1

        return {
            'input_ids': prepared_token_ids,
            'position_ids': prepared_position_ids,
            'node_ids': prepared_node_ids,
            'attention_mask': attention_mask,
            'prefix_lengths': prefix_lengths,
            'prompt_lengths': prompt_lengths,
        }

    @torch.no_grad()
    def generate(self, input_graph_batch=None, **kwargs):
        """Override standard HF generate to intercept graph kwargs and structure inputs safely."""
        if input_graph_batch is not None:
            device = next(self.parameters()).device
            input_graph_batch = send_to_device(input_graph_batch, device)
            prepared_inputs = self._prepare_inputs(input_graph_batch['input_ids'], input_graph_batch['prompt_node'], padding_side="left")
            kwargs['input_ids'] = prepared_inputs['input_ids']
            kwargs['attention_mask'] = prepared_inputs['attention_mask']
            kwargs['position_ids'] = prepared_inputs['position_ids']
            kwargs['node_ids'] = prepared_inputs['node_ids']
            kwargs['input_graph_batch'] = input_graph_batch
        return super().generate(**kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """Ensures that the new token generated at step t respects the graph's node structure and KV length."""
        position_ids = kwargs.get("position_ids", None)
        original_node_ids = kwargs.get("node_ids", None)
        input_graph_batch = kwargs.get("input_graph_batch", None)

        # 1. Reconstruct the full node_ids tensor up to the current KV sequence length
        node_ids = original_node_ids
        if original_node_ids is not None and input_graph_batch is not None:
            current_seq_len = input_ids.shape[1]
            orig_seq_len = original_node_ids.shape[1]
            if current_seq_len > orig_seq_len:
                # Pad the node_ids with the prompt_node ID for all newly generated tokens
                prompt_node = input_graph_batch['prompt_node'].view(-1, 1).to(original_node_ids.device)
                num_new_tokens = current_seq_len - orig_seq_len
                padding = prompt_node.expand(-1, num_new_tokens)
                node_ids = torch.cat([original_node_ids, padding], dim=1)

        # 2. Standard KV cache slicing for input_ids and position_ids
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

            if position_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "node_ids": node_ids, # Pass the FULL length node_ids so compute_bias can slice it correctly
            "input_graph_batch": input_graph_batch,
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

        # kwargs needed for generation
        node_ids: Optional[torch.LongTensor] = None,

        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            Modified forward method to accept the text attributed graph as input.
            ...
        """
        # get the device from the input_ids in the input_graph_batch
        device = input_graph_batch['input_ids'][0][0].device if input_graph_batch is not None else next(self.parameters()).device

        if input_graph_batch is not None:
            input_graph_batch = send_to_device(input_graph_batch, device)

        prompt_node = input_graph_batch.get('prompt_node', None) if input_graph_batch is not None else None
        
        if node_ids is None: # Meaning we are at generation step 0 or training
            input_ids_from_graph = input_graph_batch.get('input_ids', None) if input_graph_batch is not None else None
            if input_ids_from_graph is None and input_ids is None:
                raise ValueError("input_graph_batch must contain 'input_ids' or input_ids must be provided.")
            
            prepared_inputs = self._prepare_inputs(input_ids_from_graph, prompt_node, padding_side="right")
            prepared_input_ids = prepared_inputs['input_ids']
            prepared_position_ids = prepared_inputs['position_ids']
            node_ids = prepared_inputs['node_ids']
            attention_mask = prepared_inputs['attention_mask']
            prefix_lengths = prepared_inputs['prefix_lengths']
            prompt_lengths = prepared_inputs['prompt_lengths']
        else: # Autoregressive generation step > 0 (inputs already prepared via prepare_inputs_for_generation)
            prepared_input_ids = input_ids
            prepared_position_ids = position_ids
            prefix_lengths = [0] * prepared_input_ids.shape[0] # Not needed during generation loss bypass
            prompt_lengths = [0] * prepared_input_ids.shape[0] # Not needed during generation loss bypass
                
        # Pass the custom argument to the model
        outputs = self.model(
            input_ids=prepared_input_ids,               # concatenated tokens of all nodes in the graph
            attention_mask=attention_mask,              # custom attention mask based on graph structure
            position_ids=prepared_position_ids,         # position ids that reset for each node
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            
            # CUSTOM ARGUMENTS
            node_ids=node_ids,                               # node ids indicating which node each token belongs to (used for calculating attention bias)
            input_graph_batch=input_graph_batch,             # pass the whole graph batch in case it's needed for more complex bias calculations

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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        is_directory = os.path.isdir(pretrained_model_name_or_path)
        is_lora = is_directory and os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))
        is_bias_only = is_directory and os.path.exists(os.path.join(pretrained_model_name_or_path, "graph_bias_config.json"))

        # Scenario: LoRA adapter + Custom Biases
        if is_lora:
            print("Loading model with LoRA adapters and custom graph biases...")
            with open(os.path.join(pretrained_model_name_or_path, "adapter_config.json")) as f:
                base_model_name_or_path = json.load(f).get("base_model_name_or_path")
            
            # 1. Load the underlying base model
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            config = GraphLlamaConfig(**config)

            model = super().from_pretrained(base_model_name_or_path, config=config, *model_args, **kwargs)
            # 2. Wrap it with the LoRA adapters
            if PeftModel is None:
                raise ImportError("PEFT is required to load a LoRA checkpoint.")
            model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
            # 3. Inject our custom graph biases
            load_bias_parameters(model, pretrained_model_name_or_path)
            return model

        # Scenario: Lightweight custom format (No LoRA, just base model + Custom Biases)
        elif is_bias_only:
            print("Loading model with custom graph biases (no LoRA adapters)...")
            with open(os.path.join(pretrained_model_name_or_path, "graph_bias_config.json"), "r") as f:
                base_model_path = json.load(f).get("base_model_name_or_path")
            
            # Load the GraphLlama config
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            config = GraphLlamaConfig(**config)

            # 1. Load the underlying base model
            model = super().from_pretrained(base_model_path, config=config, *model_args, **kwargs)
            # 2. Inject our custom graph biases
            load_bias_parameters(model, pretrained_model_name_or_path)
            return model

        # Scenario: Standard Hugging Face loading
        else:
            print("Loading model using standard Hugging Face loading (no custom graph biases)...")
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            # If it's a local directory that just happens to have our biases alongside the full weights
            if is_directory:
                load_bias_parameters(model, pretrained_model_name_or_path)
            return model

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    graph_dataset_sample = generate_text_graph_example(
        dataset_size=3, 
        base_num_nodes=5, 
        calc_attributes=True, 
        tokenizer=tokenizer, 
        spec_emb_dim=4,
        max_rwse_steps=6,
        max_rrwp_steps=4,
    )
    example_labels = prepare_example_labels(graph_dataset_sample)

    from ..utils.text_graph_collator import GraphCollator
    collator = GraphCollator(tokenizer=tokenizer)
    input_graph_batch = collator([ graph_dataset_sample[i] for i in range(len(graph_dataset_sample)) ])

    # model = GraphLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", bias_type="combined")
    # config = GraphLlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B", spd=True, laplacian=True, max_spd=8, rwse=True)
    config = GraphLlamaConfig.from_pretrained("./src/models/graph_llama1b_config.json")
    print(config.graph_attn_bias)
    # model = GraphLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", config=config)
    model = GraphLlamaForCausalLM(config=config).to("cuda")
    
    print('=' * 70)
    outputs = model(input_graph_batch=input_graph_batch, labels=example_labels)
    print("Loss:", outputs.loss)
    print("Logits shape:", outputs.logits.shape)


    # run the model to generate 5 tokens autoregressively
    generated_ids = model.generate(input_graph_batch=input_graph_batch, max_new_tokens=5)
    print("Generated IDs:", generated_ids)