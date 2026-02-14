import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    AutoConfig,
    AutoModelForCausalLM
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig

#region Copied imports from transformers/models/llama/modeling_llama.py file:
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask
#endregion


# =============================================================================
# 1. THE CUSTOM ROTARY EMBEDDING
# =============================================================================

import json

class LlamaGraphTextEmbedding(LlamaRotaryEmbedding):
    """
    My custom implementation of Rotary Positional Embeddings.
    """
    def __init__(self, config: LlamaConfig, device=None, spectral_dims: int = 16):
        super().__init__(
            config=config,
            device=device,
        )

        # Initialise frequencies for graph spectral dimensions
        self.spectral_dims = spectral_dims
        head_dim = config.hidden_size // config.num_attention_heads # = d = d_model / num_heads
        freq_count = head_dim // 2 # = d/2
        self.spectral_freqs = nn.Parameter(torch.randn(freq_count, spectral_dims) * 0.05) # [d/2, spectral_dims]
        
        print("LlamaGraphTextEmbedding initialized.")
        # print(json.dumps(config.to_dict(), indent=4))


        print("RoPE TYPE: ", self.rope_type) # "llama3"
        
        # Internal state for graph features (injected before forward)
        self.current_node_ids = None
        self.current_node_spectral_features = None

    def update_graph_info(self, node_ids, node_spectral_features):
        """
        Updates the internal state with the graph information for the current forward pass.
        This allows us to use the standard LlamaModel forward pass without modification.
        """
        self.current_node_ids = node_ids
        self.current_node_spectral_features = node_spectral_features

        
    def get_graph_position_ids(self, node_ids):
        """
        Custom method to compute position IDs based on positions within each node.
        Example: [..., [0, 0, 0, 1, 1, 1, 1, 2, 2], ...] --> [..., [0, 1, 2, 0, 1, 2, 3, 0, 1], ...]

        Args:
            node_ids: [batch_size, seq_length]  <-- ints (node IDs 0,1,2,...)
        Returns:
            position_ids: [batch_size, seq_length]  <-- ints (positions within each node)
        """
        seq_length = node_ids.size(1)
        indices = torch.arange(seq_length, device=node_ids.device, dtype=torch.int32)
        node_id_changed = node_ids[:, 1:] != node_ids[:, :-1]
        group_starts = torch.zeros_like(node_ids, dtype=torch.int32)
        group_starts[:, 1:] = indices[1:] * node_id_changed.int()
        group_starts = group_starts.cummax(dim=1).values
        return (indices - group_starts).long() # ensuring return type is LongTensor

    def forward(self, x, position_ids=None, **kwargs):
        """
        Compute the rotary embeddings (cosine and sine) for the given inputs.
        
        We support the standard signature: forward(x, position_ids, **kwargs)
        But we secretly use self.current_node_ids and self.current_node_spectral_features
        which must be injected before calling this!
        """
        # Retrieve injected state
        node_ids = self.current_node_ids
        node_spectral_features = self.current_node_spectral_features
        
        # Safety check
        if node_ids is None or node_spectral_features is None:
            raise ValueError("Graph information (node_ids, node_spectral_features) must be injected via `update_graph_info` before forward pass.")

        # if position_ids is None:
        #     position_ids = self._get_position_ids(node_ids)
        
        # print("FORWARD of LlamaGraphTextEmbedding called!")
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # [batch_size, d_head//2, 1]
        # print("inv_freq_expanded.shape:", inv_freq_expanded.shape)

        position_ids_expanded = position_ids[:, None, :].float()
        # position_ids_expanded.shape: [batch_size, 1, seq_length]

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Standard RoPE frequencies
            rope_thetas = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            # [batch_size, d_head//2, 1] @ [batch_size, 1, seq_length] --> [batch_size, seq_length, d_head//2]

            # Graph spectral frequencies
            token_spectral_features = node_spectral_features.gather(
                dim=1,
                index=node_ids.unsqueeze(-1).expand(-1, -1, self.spectral_dims)
            ) # [batch_size, seq_length, spectral_dims]
            graph_freqs = token_spectral_features @ self.spectral_freqs.T
            # [batch_size, seq_length, spectral_dims] @ [spectral_dims, d_head//2] --> [batch_size, seq_length, d_head//2]

            thetas = rope_thetas + graph_freqs

            emb = torch.cat((thetas, thetas), dim=-1)
            # emb.shape:    [batch_size, seq_length, d_head]
            cos = emb.cos()
            # cos.shape:    [batch_size, seq_length, d_head]
            sin = emb.sin()
            # sin.shape:    [batch_size, seq_length, d_head]

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# =============================================================================
# 2. THE CUSTOM MODEL (Swapping the RoPE)
# =============================================================================

class GraphLlamaModel(LlamaModel):
    """
    A subclass of LlamaModel that replaces the default LlamaRotaryEmbedding
    with LlamaGraphTextEmbedding.
    """
    def __init__(self, config: LlamaConfig, spectral_dims: int = 16):
        super().__init__(config)
        
        # OVERRIDE: We replace the rotary_emb initialized by the parent class
        # with our custom version.
        self.rotary_emb = LlamaGraphTextEmbedding(config=config, spectral_dims=spectral_dims)
        
        # Ensure weights are initialized
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        node_ids: torch.LongTensor = None,
        node_spectral_features: torch.FloatTensor = None,
        position_ids: torch.LongTensor = None, # Catch explicit position_ids
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Inject state
        self.rotary_emb.update_graph_info(node_ids, node_spectral_features)
        
        if position_ids is None and node_ids is not None:
            position_ids = self.rotary_emb.get_graph_position_ids(node_ids)

        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs
        )



# =============================================================================
# 3. THE CUSTOM CAUSAL LM (The Wrapper & Vocab Extension)
# =============================================================================

class GraphLlamaForCausalLM(LlamaForCausalLM):
    """
    The top-level class. It uses my custom base model and handles vocab extension.
    """
    def __init__(self, config: LlamaConfig, spectral_dims: int = 16, **kwargs):
        super().__init__(config)
        
        self.model = GraphLlamaModel(config, spectral_dims=spectral_dims)
        
        self.post_init()

    def extend_vocabulary(self, new_vocab_size: int):
        """
        Cleanly handles vocabulary extension.
        """
        # 1. Update the underlying configuration
        self.config.vocab_size = new_vocab_size
        
        # 2. Resize the embeddings (input and output head)
        # This method (provided by the parent class) handles the resizing of 
        # self.model.embed_tokens and self.lm_head automatically.
        self.resize_token_embeddings(new_vocab_size)
        
        # TODO: Add specific logic here if you want to initialize the new 
        # tokens with something specific (e.g., average of other embeddings)
        # rather than the random initialization provided by default.
        print(f"Vocabulary extended. New size: {new_vocab_size}")



# Register so it works with AutoModel (optional)
# AutoConfig.register("my-custom-llama", LlamaConfig)
# AutoModelForCausalLM.register(LlamaConfig, GraphLlamaForCausalLM)
