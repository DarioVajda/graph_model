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
# 1. THE CUSTOM ROTARY EMBEDDING (My Invention)
# =============================================================================

import json

class LlamaGraphTextEmbedding(LlamaRotaryEmbedding):
    """
    My custom implementation of Rotary Positional Embeddings.
    """
    def __init__(self, config: LlamaConfig, device=None, spectral_dims: int = 16):
        # Note: The signature for __init__ might vary slightly depending on 
        # transformers version, but usually takes config or dim/max_pos.
        # We generally prefer to initialize using the config to be safe.
        super().__init__(
            config=config,
            device=device,
        )

        # Initialise frequencies for graph spectral dimensions
        self.spectral_dims = spectral_dims
        head_dim = config.hidden_size // config.num_attention_heads # = d = d_model / num_heads
        freq_count = head_dim // 2 # = d/2
        self.spectral_freqs = nn.Parameter(torch.randn(freq_count, spectral_dims)) # [d/2, spectral_dims]

        print("LlamaGraphTextEmbedding initialized.")
        # print(json.dumps(config.to_dict(), indent=4))

        print("RoPE TYPE: ", self.rope_type) # "llama3"


    def _get_position_ids(self, node_ids):
        """
        Custom method to compute position IDs based on positions within each node.
        Example: [..., [0, 0, 0, 1, 1, 1, 1, 2, 2], ...] --> [..., [0, 1, 2, 0, 1, 2, 3, 0, 1], ...]

        Args:
            node_ids: [batch_size, seq_length]  <-- ints (node IDs 0,1,2,...)
        Returns:
            position_ids: [batch_size, seq_length]  <-- ints (positions within each node)
        """
        seq_length = node_ids.size(1)
        
        # OPTIMIZATION: Use int32 to save 50% memory bandwidth. 
        # int64 is overkill unless seq_len > 2 billion.
        indices = torch.arange(seq_length, device=node_ids.device, dtype=torch.int32)
        
        # 1. Boolean mask of changes (Result is bool aka uint8)
        # We slice [:, 1:] to avoid shape mismatch errors
        node_id_changed = node_ids[:, 1:] != node_ids[:, :-1]
        
        # 2. Prepare container (int32)
        group_starts = torch.zeros_like(node_ids, dtype=torch.int32)
        
        # 3. Fill boundaries
        # We cast node_id_changed to int32 for the multiplication
        group_starts[:, 1:] = indices[1:] * node_id_changed.int()
        
        # 4. Scan (cummax is highly optimized on GPU)
        group_starts = group_starts.cummax(dim=1).values
        
        return indices - group_starts

    def forward(self, x, position_ids, node_ids, node_spectral_features):
        """
        Compute the rotary embeddings (cosine and sine) for the given inputs. Note that the rotations will not be performed for pairs [i, i+1], but instead for pairs [i, i+head_dim/2].

        Args:
            x:                        [batch_size, seq_length, emb_dim]       <-- floats
            position_ids (optional):  [batch_size, seq_length]                <-- ints (eg. 0,1,2,0,1,2,3,0,1,...)
            node_ids:                 [batch_size, seq_length]                <-- ints (eg. 0,0,0,1,1,1,1,2,2,...)
            node_spectral_features:   [batch_size, num_nodes, spectral_dims]  <-- floats (precomputed spectral features for each from Laplacian)
        Returns:
            cos:                      [batch_size, seq_length, emb_dim]       <-- floats
            sin:                      [batch_size, seq_length, emb_dim]       <-- floats
        """

        if not position_ids:
            position_ids = self._get_position_ids(node_ids)
        
        print("FORWARD of LlamaGraphTextEmbedding called!")
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # [batch_size, d_head//2, 1]
        print("inv_freq_expanded.shape:", inv_freq_expanded.shape)

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

class MyCustomLlamaModel(LlamaModel):
    """
    A subclass of LlamaModel that replaces the default LlamaRotaryEmbedding
    with LlamaGraphTextEmbedding.
    """
    def __init__(self, config: LlamaConfig, spectral_dims: int = 16):
        super().__init__(config)
        
        # OVERRIDE: We replace the rotary_emb initialized by the parent class
        # with our custom version.
        self.rotary_emb = LlamaGraphTextEmbedding(config=config, spectral_dims=spectral_dims)
        
        # Note: We do not need to modify LlamaDecoderLayer or LlamaAttention
        # because this model calculates the embeddings here and passes the
        # computed (cos, sin) down to the layers in the forward pass.
        
        # Ensure weights are initialized (standard practice)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        node_ids: torch.LongTensor = None,
        node_spectral_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        This  forward method was almost entirely copied from the parent class, with the only difference being that we pass node_ids and node_spectral_features to the rotary embedding computation.
        """
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids=None, node_ids=node_ids, node_spectral_features=node_spectral_features)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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



# =============================================================================
# 3. THE CUSTOM CAUSAL LM (The Wrapper & Vocab Extension)
# =============================================================================

class MyCustomLlamaForCausalLM(LlamaForCausalLM):
    """
    The top-level class. It uses my custom base model and handles vocab extension.
    """
    def __init__(self, config: LlamaConfig, spectral_dims: int = 16, **kwargs):
        # Call the super init, but it will initialize the standard LlamaModel which is slightly inefficient but ensures all attributes are set correctly.
        super().__init__(config)
        
        # OVERRIDE: Replace the internal 'model' with our custom class
        self.model = MyCustomLlamaModel(config, spectral_dims=spectral_dims)
        
        # Re-tie weights if necessary (standard HF logic)
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


# =============================================================================
# REGISTRATION & EXAMPLE
# =============================================================================

# Register so it works with AutoModel (optional)
# AutoConfig.register("my-custom-llama", LlamaConfig)
# AutoModelForCausalLM.register(LlamaConfig, MyCustomLlamaForCausalLM)

def _test_graph_llama(model_name = "meta-llama/Llama-3.2-1B"):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test the implementation
    print("Initializing Custom Llama Model...")

    # 1. Setup a dummy config
    config = LlamaConfig.from_pretrained(model_name)

    # 2.1 Load the model weights into the custom class
    custom_model = MyCustomLlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        spectral_dims=32,
        strict=False # ignore missing keys due to custom RoPE
    )

    # 2.2 Load the model weights into the default class
    default_model = LlamaForCausalLM.from_pretrained(model_name, config=config)

    print("Weights loaded successfully!")
    
    # exit(0)

    verify_architecture = False
    if verify_architecture:
        # 3.1 Verify the Custom Model Architecture
        print("\n--- Custom Model Architecture ---")
        print(custom_model)

        # 3.2 Verify the Default Model Architecture
        print("\n--- Default Model Architecture ---")
        print(default_model)

    # 4. Dummy Forward Pass
    dummy_input = torch.randint(0, 1050, (2, 10))
    dummy_node_ids = torch.zeros((2, 10), dtype=torch.long)  # All tokens belong to node 0
    dummy_spectral_features = torch.randn(2, 3, 32)  # [batch_size, num_nodes, spectral_dims]

    # 4.1 Forward pass through custom model
    print("\nPerforming forward pass through custom model...")
    custom_output = custom_model(dummy_input, node_ids=dummy_node_ids, node_spectral_features=dummy_spectral_features)
    print(f"Custom forward pass logits shape: {custom_output.logits.shape}")

    # 4.2 Forward pass through default model
    print("\nPerforming forward pass through default model...")
    default_output = default_model(dummy_input)
    print(f"Default forward pass logits shape: {default_output.logits.shape}")

    # 5.1 Output from custom model
    print("--- Custom Model Output ---")
    predicted_tokens = torch.argmax(custom_output.logits, dim=-1)
    print(predicted_tokens)

    # 5.2 Output from default model
    print("--- Default Model Output ---")
    predicted_tokens_default = torch.argmax(default_output.logits, dim=-1)
    print(predicted_tokens_default)
    

if __name__ == "__main__":
    _test_graph_llama(model_name="meta-llama/Llama-3.2-3B")
