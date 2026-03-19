import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# ==============================================================================
# 1. GLOBAL COMPILATION
# ==============================================================================
# We compile flex_attention at the module level. This ensures it is only 
# compiled ONCE when the Python process starts, rather than every time the 
# forward pass runs.
# 
# NOTE: If you are not padding your sequences to a fixed length, 
# change this to `dynamic=True` and remove `mode="max-autotune-no-cudagraphs"`.
# compiled_flex_attention = torch.compile(
#     flex_attention, 
#     dynamic=False, 
#     mode="max-autotune-no-cudagraphs"
# )
compiled_flex_attention = flex_attention # because compilation is being done at a higher level

# ==============================================================================
# 2. THE FUNCTIONAL INTERFACE
# ==============================================================================
def graph_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    node_bias: torch.Tensor,
    node_ids: torch.Tensor,
    valid_start: torch.Tensor,
    valid_end: torch.Tensor,
    prefix_end: torch.Tensor,
):
    """
    Computes FlashAttention with custom graph-aware node biases.
    
    Args:
        query, key, value:  (batch_size, num_heads, seq_len, head_dim)      -> Standard attention tensors.
        node_bias:          (batch_size, num_heads, num_nodes, num_nodes)   -> Precomputed bias tensor of shape (B, H, num_nodes, num_nodes)
        node_ids:           (batch_size, seq_len)                           -> Tensor mapping each token to its corresponding node, values are in [0, num_nodes-1]
        valid_start:        (batch_size,)                                   -> Start index of valid tokens for each sequence (for padding)
        valid_end:          (batch_size,)                                   -> End index of valid tokens for each sequence (for padding)
        prefix_end:         (batch_size,)                                   -> End index of the prefix for each sequence (for padding)
    """
    device = query.device
    
    # --------------------------------------------------------------------------
    # A. Define the Math (The Score Mod)
    # --------------------------------------------------------------------------
    # This closure automatically captures `node_bias` and `node_ids` 
    # from the outer scope. PyTorch handles passing these to the GPU kernel safely.
    def graph_bias_mod(score, b, h, q_idx, kv_idx):
        q_node = node_ids[b, q_idx]
        kv_node = node_ids[b, kv_idx]

        return score + node_bias[b, h, q_node, kv_node]

    # --------------------------------------------------------------------------
    # B. Define the Topology (The Mask Mod for Speed!)
    # --------------------------------------------------------------------------
    def graph_prefix_mask(b, h, q_idx, kv_idx):
        # Determine if the current Q and KV tokens are "real" data
        q_is_valid = (q_idx >= valid_start[b]) & (q_idx < valid_end[b])
        kv_is_valid = (kv_idx >= valid_start[b]) & (kv_idx < valid_end[b])
        
        # Topology rules (Only applied to valid tokens)
        prefix_mask = kv_idx < prefix_end[b]
        causal_mask = q_idx >= kv_idx
        topology_mask = prefix_mask | causal_mask
        
        # Combine: Both tokens must be valid, AND satisfy the topology
        valid_attention = q_is_valid & kv_is_valid & topology_mask
        
        # ANTI-NaN SAFETY NET - If the Q token is padding, force it to only look at itself.
        is_padding_self_attention = (~q_is_valid) & (q_idx == kv_idx)
        
        return valid_attention | is_padding_self_attention


    # --------------------------------------------------------------------------
    # C. Execute the Kernel
    # --------------------------------------------------------------------------
    # Generate the block mask (PyTorch calculates which 64x64 blocks to skip)
    # If max_graph_distance is heavily restricting, this step saves immense time.
    block_mask = create_block_mask(
        graph_prefix_mask, 
        B=query.shape[0], 
        H=query.shape[1], 
        Q_LEN=query.shape[2], 
        KV_LEN=key.shape[2], 
        device=device
    )

    # Run the ultra-fast compiled kernel
    attn_output = compiled_flex_attention(
        query, 
        key, 
        value, 
        score_mod=graph_bias_mod, 
        block_mask=block_mask
    )
    
    return attn_output


if __name__ == "__main__":
    # Example usage and sanity check
    B, H, S, D = 4, 32, 1024, 64
    node_count = 4
    q = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    k = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    v = torch.randn(B, H, S, D, device="cuda", requires_grad=True)
    
    node_bias = torch.rand(B, H, node_count, node_count, device="cuda", requires_grad=True)
    token_node_ids = torch.zeros((B, S), dtype=torch.int32, device="cuda")
    token_node_ids[:, :S//4] = 0
    token_node_ids[:, S//4:S//2] = 1
    token_node_ids[:, S//2:3*S//4] = 2
    token_node_ids[:, 3*S//4:] = 3

    valid_start = torch.zeros(B, dtype=torch.int32, device="cuda")
    valid_end = torch.full((B,), S-1, dtype=torch.int32, device="cuda")
    prefix_end = torch.full((B,), 3*S//4, dtype=torch.int32, device="cuda")

    grad_out = torch.randn_like(q)

    flex_out = graph_flex_attention(q, k, v, node_bias, token_node_ids, valid_start, valid_end, prefix_end)
    flex_out.backward(grad_out)
    print("Flex attention executed successfully with custom graph bias and topology!")