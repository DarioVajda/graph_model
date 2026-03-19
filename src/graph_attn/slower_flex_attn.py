import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import math
from typing import Dict, Any, List

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
compiled_flex_attention = flex_attention  # For testing without compilation. Replace with the above line for actual benchmarking.



# ==============================================================================
#region 2. DATA PREPARATION AND PADDING
# ==============================================================================
def prepare_inputs(
    input_ids: List[List[torch.Tensor]],
    prompt_node: torch.Tensor,
    padding_lengths: List[int],
    padding_side: str = "right",
    chunk_size: int = 64,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Prepare input_ids and position_ids for the model based on the prompt node.
    Prompt node is placed at the end of the sequence.
    The final sequence length is snapped to the smallest value in `padding_lengths`
    that fits the batch, ensuring consistent kernel sizes for torch.compile.

    Arguments:
        input_ids       --> List of lists of 1D tensors with shape (seq_len,) for each node
        prompt_node     --> Tensor of shape (batch_size,) containing the index of the prompt node
        padding_lengths --> List of allowed sequence lengths (e.g., [128, 256, 512, 1024]). Must be multiples of chunk_size.
        padding_side    --> "left" for inference, "right" for training
        chunk_size      --> Each node's tokens will be padded to a multiple of this chunk size.
        pad_token_id    --> The token ID to use for padding.
    Returns:
        Dictionary containing:
        - padded_input_ids --> (batch_size, max_padded_seq_len) # The graph is flattened into a single sequence with node-level padding.
        - position_ids     --> (batch_size, max_padded_seq_len) # Relative position IDs (within each node), with padding positions set to 0.
        - chunk_node_ids   --> (batch_size, max_chunk_count)    # Maps each chunk to its original node ID, with padding chunks set to 0.
        - chunk_tokens     --> (batch_size, max_chunk_count)    # Number of valid tokens in each chunk, used for masking (0 for padding chunks).
        - prefix_end       --> (batch_size,)                    # The index in the chunk sequence where the prefix ends (this is where the prompt node starts)
    """
    batch_size = len(input_ids)
    device = input_ids[0][0].device if batch_size > 0 and len(input_ids[0]) > 0 else torch.device("cpu")
    
    # 1. Validate and sort padding lengths
    valid_lengths = sorted(padding_lengths)
    for length in valid_lengths:
        if length % chunk_size != 0:
            raise ValueError(f"Padding length {length} is not divisible by chunk_size {chunk_size}.")
    
    batch_padded_seqs = []
    batch_position_ids = []
    batch_chunk_node_ids = []
    batch_chunk_tokens = []
    batch_prefix_ends = []  # <--- Added to track the unpadded prefix_end
    
    max_seq_len = 0
    
    # --------------------------------------------------------------------------
    # 2. Process Each Graph in the Batch (Node-level padding)
    # --------------------------------------------------------------------------
    for b in range(batch_size):
        p_idx = prompt_node[b].item()
        num_nodes = len(input_ids[b])
        
        # Order: All context nodes first, then the prompt node
        ordered_nodes = [i for i in range(num_nodes) if i != p_idx] + [p_idx]
        
        seq_parts = []
        pos_parts = []
        chunk_node_list = []
        chunk_tokens_list = []
        
        prefix_end_idx = 0  # <--- Initialize for this graph
        
        for n_idx in ordered_nodes:
            # When we hit the prompt node, the current number of chunks 
            # is exactly the index where the prompt begins.
            if n_idx == p_idx:
                prefix_end_idx = len(chunk_node_list)
                
            tokens = input_ids[b][n_idx]
            seq_len = tokens.size(0)
            
            # Ensure at least 1 chunk even if a node is empty
            num_chunks = max(1, math.ceil(seq_len / chunk_size))
            target_len = num_chunks * chunk_size
            pad_len = target_len - seq_len
            
            # A. Pad the tokens for this specific node
            if pad_len > 0:
                padded_tokens = torch.nn.functional.pad(tokens, (0, pad_len), value=pad_token_id)
            else:
                padded_tokens = tokens
            seq_parts.append(padded_tokens)
            
            # B. Generate Position IDs (Relative to the node)
            valid_positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
            pad_positions = torch.zeros(pad_len, dtype=torch.long, device=device)
            pos_parts.append(torch.cat([valid_positions, pad_positions]))
            
            # C. Track Chunk Metadata
            for c in range(num_chunks):
                chunk_node_list.append(n_idx)
                valid_in_chunk = max(0, min(chunk_size, seq_len - c * chunk_size))
                chunk_tokens_list.append(valid_in_chunk)
                
        # Flatten the graph into 1D tensors
        graph_seq = torch.cat(seq_parts)
        graph_pos = torch.cat(pos_parts)
        
        batch_padded_seqs.append(graph_seq)
        batch_position_ids.append(graph_pos)
        batch_chunk_node_ids.append(torch.tensor(chunk_node_list, dtype=torch.long, device=device))
        batch_chunk_tokens.append(torch.tensor(chunk_tokens_list, dtype=torch.long, device=device))
        batch_prefix_ends.append(prefix_end_idx) # <--- Store the unpadded prefix start index
        
        max_seq_len = max(max_seq_len, graph_seq.size(0))

    # --------------------------------------------------------------------------
    # 3. Determine Target Batch Length (Bucketing)
    # --------------------------------------------------------------------------
    target_seq_len = None
    for length in valid_lengths:
        if length >= max_seq_len:
            target_seq_len = length
            break
            
    if target_seq_len is None:
        target_seq_len = math.ceil(max_seq_len / chunk_size) * chunk_size
        
    target_chunks = target_seq_len // chunk_size

    # --------------------------------------------------------------------------
    # 4. Global Batch Padding
    # --------------------------------------------------------------------------
    out_input_ids = torch.full((batch_size, target_seq_len), pad_token_id, dtype=torch.long, device=device)
    out_position_ids = torch.zeros((batch_size, target_seq_len), dtype=torch.long, device=device)
    out_chunk_node_ids = torch.full((batch_size, target_chunks), 0, dtype=torch.long, device=device)
    out_chunk_tokens = torch.zeros((batch_size, target_chunks), dtype=torch.long, device=device)
    out_prefix_end = torch.zeros((batch_size,), dtype=torch.long, device=device) # <--- Pre-allocate prefix tensor
    
    for b in range(batch_size):
        s_len = batch_padded_seqs[b].size(0)
        c_len = batch_chunk_node_ids[b].size(0)
        
        if padding_side == "right":
            out_input_ids[b, :s_len] = batch_padded_seqs[b]
            out_position_ids[b, :s_len] = batch_position_ids[b]
            out_chunk_node_ids[b, :c_len] = batch_chunk_node_ids[b]
            out_chunk_tokens[b, :c_len] = batch_chunk_tokens[b]
            out_prefix_end[b] = batch_prefix_ends[b] # <--- Right padding does not shift the prefix
        else: # padding_side == "left"
            out_input_ids[b, -s_len:] = batch_padded_seqs[b]
            out_position_ids[b, -s_len:] = batch_position_ids[b]
            out_chunk_node_ids[b, -c_len:] = batch_chunk_node_ids[b]
            out_chunk_tokens[b, -c_len:] = batch_chunk_tokens[b]
            # <--- Left padding pushes the sequence to the right, so we add the padding offset
            out_prefix_end[b] = batch_prefix_ends[b] + (target_chunks - c_len) 

    return {
        'padded_input_ids': out_input_ids,
        'position_ids': out_position_ids,
        'chunk_node_ids': out_chunk_node_ids,
        'chunk_tokens': out_chunk_tokens,
        'prefix_end': out_prefix_end      # <--- Now returned
    }

def test_prepare_inputs():
    # Batch size of 2
    # Graph 0: 3 nodes. Node 2 is the prompt.
    # Graph 1: 2 nodes. Node 0 is the prompt.
    input_ids = [
        # Graph 0
        [
            torch.tensor([10, 11]),               # Node 0: len 2 -> needs 2 pads to reach chunk 4
            torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28]),   # Node 1: len 5 -> needs 3 pads to reach chunk 8 (2 chunks)
            torch.tensor([30, 31, 32, 33, 34]),           # Node 2: len 3 -> needs 1 pad to reach chunk 4
        ],
        # Graph 1
        [
            torch.tensor([88]),                   # Node 0: len 1 -> needs 3 pads to reach chunk 4
            torch.tensor([99, 98]),               # Node 1: len 2 -> needs 2 pads to reach chunk 4
        ]
    ]
    
    prompt_node = torch.tensor([2, 0]) # Graph 0 prompt is Node 2. Graph 1 prompt is Node 0.
    
    # Run with a tiny chunk size of 4 for visual inspection
    out = prepare_inputs(
        input_ids=input_ids, 
        prompt_node=prompt_node, 
        padding_lengths=[16, 32, 64],
        padding_side="right", 
        chunk_size=4, 
        pad_token_id=0
    )
    
    print("=== TEST RESULTS ===")
    print("\n1. PADDED INPUT IDS")
    print("Notice how the prompt node is moved to the end of the valid blocks,")
    print("and global batch padding (0) fills the rest of Graph 1.")
    print(out['padded_input_ids'])
    
    print("\n2. POSITION IDS")
    print("Notice how position IDs increment for valid tokens but stay 0 for node-padding.")
    print(out['position_ids'])
    
    print("\n3. CHUNK NODE IDS")
    print("Maps each 4-token block back to its original node ID. -1 is a global batch pad chunk.")
    print(out['chunk_node_ids'])
    
    print("\n4. CHUNK TOKENS")
    print("Shows how many *real* tokens are in each 4-token chunk. 0 for batch pad chunks.")
    print(out['chunk_tokens'])

    print("\n5. PREFIX END")
    print("The index in the chunk sequence where the prompt node starts. Used for masking.")
    print(out['prefix_end'])

def create_big_example(B=4, chunk_size=64, num_nodes=64, H=32, D=64, requires_grad=False, max_tok_per_node=64):
    max_id = 99
    # Create 64 nodes per batch, each with 10-49 tokens
    input_ids = [ [torch.randint(1, max_id, (torch.randint(1, max_tok_per_node, (1,)).item(),)) for _ in range(num_nodes)] for _ in range(B) ]
    prompt_node = torch.tensor([ torch.randint(0, len(input_ids[b]), (1,)).item() for b in range(B) ])

    prepared_inputs = prepare_inputs(
        input_ids=input_ids,
        prompt_node=prompt_node,
        padding_lengths=[2048, 4096, 8192, 16384],
        padding_side="right",
        chunk_size=chunk_size,
        pad_token_id=0
    )
    
    # Move outputs to CUDA for the kernel
    padded_input_ids = prepared_inputs['padded_input_ids'].cuda()  # (B, S)
    position_ids = prepared_inputs['position_ids'].cuda()          # (B, S)
    chunk_node_ids = prepared_inputs['chunk_node_ids'].cuda()      # (B, C)
    chunk_tokens = prepared_inputs['chunk_tokens'].cuda()          # (B, C)
    prefix_end = prepared_inputs['prefix_end'].cuda()              # (B,)

    q_lookup = torch.randn(max_id, H, D, device="cuda", requires_grad=requires_grad)
    k_lookup = torch.randn(max_id, H, D, device="cuda", requires_grad=requires_grad)
    v_lookup = torch.randn(max_id, H, D, device="cuda", requires_grad=requires_grad)

    # Transpose to (B, H, S, D) for flex_attention
    q = q_lookup[padded_input_ids].transpose(1, 2)
    k = k_lookup[padded_input_ids].transpose(1, 2)
    v = v_lookup[padded_input_ids].transpose(1, 2)

    node_bias_list = [ torch.randn(H, len(input_ids[b]), len(input_ids[b]), device="cuda", requires_grad=requires_grad) for b in range(B) ]
    node_bias = torch.zeros(B, H, num_nodes, num_nodes, device="cuda")
    for b in range(B):
        n = len(input_ids[b])
        node_bias[b, :, :n, :n] = node_bias_list[b]
    
    # Create broadcastable index grids
    b_idx = torch.arange(B, device="cuda")[:, None, None, None]  # (B, 1, 1, 1)
    h_idx = torch.arange(H, device="cuda")[None, :, None, None]  # (1, H, 1, 1)
    row_idx = chunk_node_ids[:, None, :, None]              # (B, 1, C, 1)
    col_idx = chunk_node_ids[:, None, None, :]              # (B, 1, 1, C)
    
    # Extract the chunk bias perfectly mapping (B, H, C, C)
    chunk_bias = node_bias[b_idx, h_idx, row_idx, col_idx]

    return q, k, v, position_ids, chunk_bias, chunk_tokens, prefix_end

#endregion
# ==============================================================================


# ==============================================================================
#region 3. THE FUNCTIONAL INTERFACE
# ==============================================================================
def graph_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int,
    chunk_bias: torch.Tensor,
    chunk_tokens: torch.Tensor,
    prefix_end: torch.Tensor
):
    """
    Computes FlashAttention with custom graph-aware node biases.
    
    Args:
        query, key, value:  (batch_size, num_heads, seq_len, head_dim)      -> Standard attention tensors.
        chunk_size:         int                                             -> Each node is being padded to a multiple of this chunk size.
        chunk_bias:         (batch_size, num_heads, num_chunks, num_chunks) -> Precomputed bias tensor of shape (B, H, C, C) where C is the total number of chunks.
        chunk_tokens:       (batch_size, num_chunks)                        -> Number of valid tokens in each chunk (for masking)
        prefix_end:         (batch_size,)                                   -> The index in the chunk sequence where the prefix ends (for prefix vs causal masking)
    """
    device = query.device

    # require that chunk_size is a power of 2 for efficient division using bit shifts
    if not(chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0):
        raise ValueError("chunk_size must be a power of 2 for efficient GPU indexing.")
    chunk_shift = int(math.log2(chunk_size))  # Number of bits to shift for chunk calculation
    
    # --------------------------------------------------------------------------
    # A. Define the Math (The Score Mod)
    # --------------------------------------------------------------------------
    # This closure automatically captures `node_bias` and `node_ids` 
    # from the outer scope. PyTorch handles passing these to the GPU kernel safely.
    def graph_bias_mod(score, b, h, q_idx, kv_idx):
        q_chunk = q_idx >> chunk_shift
        kv_chunk = kv_idx >> chunk_shift

        # Add the pre-calculated bias (L2 cache will coalesce this perfectly)
        return score + chunk_bias[b, h, q_chunk, kv_chunk]

    # --------------------------------------------------------------------------
    # B. Define the Topology (The Mask Mod for Speed!)
    # --------------------------------------------------------------------------
    def graph_prefix_mask(b, h, q_idx, kv_idx):
        # Calculate which chunk the Q and KV tokens belong to
        q_chunk = q_idx >> chunk_shift
        kv_chunk = kv_idx >> chunk_shift

        # Determine if the current Q and KV tokens are "real" data
        q_is_valid = q_idx < chunk_size * q_chunk + chunk_tokens[b, q_chunk]
        kv_is_valid = kv_idx < chunk_size * kv_chunk + chunk_tokens[b, kv_chunk]
        
        # Topology rules (Only applied to valid tokens)
        prefix_mask = kv_idx < prefix_end[b] * chunk_size  # Prefix can attend to all previous tokens
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

def test_graph_flex_attention():
    # Example usage and sanity check
    q, k, v, position_ids, chunk_bias, chunk_tokens, prefix_end = create_big_example(requires_grad=True)

    grad_out = torch.randn_like(q)

    flex_out = graph_flex_attention(q, k, v, chunk_size=64, chunk_bias=chunk_bias, chunk_tokens=chunk_tokens, prefix_end=prefix_end)
    flex_out.backward(grad_out)
    print("Flex attention executed successfully with custom graph bias and topology!")


#endregion
# ==============================================================================





if __name__ == "__main__":
    # test_prepare_inputs()
    test_graph_flex_attention()