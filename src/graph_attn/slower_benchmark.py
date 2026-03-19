import torch
import torch.nn.functional as F
from triton.testing import do_bench
import math

# Import strictly from your custom module
from .flex_attn import (
    create_big_example,
    graph_flex_attention
)

# ==============================================================================
# 1. DECODER FOR FAIR BASELINES
# ==============================================================================
def get_baselines_data(q, k, v, chunk_bias, chunk_tokens, prefix_end, chunk_size=64):
    """
    Decodes the chunk-padded FlexAttention data into tightly packed sequences 
    and generates the fully materialized dense mask for the Eager PyTorch benchmark.
    Gracefully handles OOM errors during memory-heavy operations.
    """
    B, H, S_padded, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # Calculate the exact number of valid tokens per graph in the batch
    actual_lengths = chunk_tokens.sum(dim=-1) # (B,)
    S_unpadded = actual_lengths.max().item()
    
    q_unp, k_unp, v_unp, dense_mask = None, None, None, None
    
    # --- PHASE A: Allocate and Pack Sequences (Needed for Causal FA2 & Eager) ---
    try:
        q_unp = torch.zeros(B, H, S_unpadded, D, device=device, dtype=dtype)
        k_unp = torch.zeros_like(q_unp)
        v_unp = torch.zeros_like(q_unp)
        
        for b in range(B):
            L = actual_lengths[b].item()
            c_tokens = chunk_tokens[b]
            valid_indices = []
            for c_idx, num_tok in enumerate(c_tokens):
                if num_tok > 0:
                    start = c_idx * chunk_size
                    valid_indices.append(torch.arange(start, start + num_tok, device=device))
            
            if not valid_indices:
                continue
                
            valid_indices = torch.cat(valid_indices) # (L,)
            
            # Pack the sequences tightly
            q_unp[b, :, :L, :] = q[b, :, valid_indices, :]
            k_unp[b, :, :L, :] = k[b, :, valid_indices, :]
            v_unp[b, :, :L, :] = v[b, :, valid_indices, :]
            
        q_unp = q_unp.detach().requires_grad_(True)
        k_unp = k_unp.detach().requires_grad_(True)
        v_unp = v_unp.detach().requires_grad_(True)
        
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM Warning] Failed to allocate packed sequences for S={S_unpadded}. Causal & Eager baselines will be skipped.")
        torch.cuda.empty_cache()
        return None, None, None, None, S_unpadded

    # --- PHASE B: Allocate Dense Mask (Needed ONLY for Eager) ---
    try:
        dense_mask_temp = torch.full((B, H, S_unpadded, S_unpadded), float('-inf'), device=device, dtype=dtype)
        for b in range(B):
            L = actual_lengths[b].item()
            c_tokens = chunk_tokens[b]
            valid_indices = []
            for c_idx, num_tok in enumerate(c_tokens):
                if num_tok > 0:
                    start = c_idx * chunk_size
                    valid_indices.append(torch.arange(start, start + num_tok, device=device))
            
            if not valid_indices:
                continue
            valid_indices = torch.cat(valid_indices)
            
            q_idx = valid_indices.unsqueeze(1).expand(L, L)
            kv_idx = valid_indices.unsqueeze(0).expand(L, L)
            
            q_chunk = q_idx // chunk_size
            kv_chunk = kv_idx // chunk_size
            
            prefix_boundary = prefix_end[b] * chunk_size
            topology = (kv_idx < prefix_boundary) | (q_idx >= kv_idx)
            
            gathered_bias = chunk_bias[b, :, q_chunk, kv_chunk]
            valid_bias = torch.where(topology.unsqueeze(0), gathered_bias, float('-inf'))
            dense_mask_temp[b, :, :L, :L] = valid_bias
            
        dense_mask = dense_mask_temp.detach().requires_grad_(True)
        
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM Warning] Failed to allocate {S_unpadded}x{S_unpadded} dense mask. Eager baseline will be skipped.")
        torch.cuda.empty_cache()

    return q_unp, k_unp, v_unp, dense_mask, S_unpadded

# ==============================================================================
# 2. BENCHMARK RUNNER
# ==============================================================================
def run_benchmark():
    B, H, D = 4, 16, 64  
    chunk_size = 64
    target_seq_lengths = [2048, 4096, 8192]
    dtype = torch.float16 

    print("\nStarting Benchmark...")
    print("NOTE: FlexAttention's `max-autotune` will trigger a 1-3 minute compilation ")
    print("phase at the beginning of each sequence length. Please be patient!\n")
    
    all_results = []

    # Compile the FlexAttention wrapper ONCE globally
    graph_flex_attention_compiled = torch.compile(
        graph_flex_attention, 
        dynamic=False, 
        mode="max-autotune-no-cudagraphs"
    )

    for S_target in target_seq_lengths:
        step_results = []
        num_nodes = max(1, S_target // 32)
        
        S_padded = "N/A"
        S_real = "N/A"
        
        # 1. Generate the padded flex-attention formatting
        try:
            q, k, v, pos_ids, chunk_bias, chunk_tokens, prefix_end = create_big_example(
                B=B, chunk_size=chunk_size, num_nodes=num_nodes, H=H, D=D, 
                requires_grad=True, max_tok_per_node=64
            )
            S_padded = q.shape[2]
            
            q = q.to(dtype).detach().requires_grad_(True)
            k = k.to(dtype).detach().requires_grad_(True)
            v = v.to(dtype).detach().requires_grad_(True)
            chunk_bias = chunk_bias.to(dtype).detach().requires_grad_(True)
            
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] GPU maxed out during data generation for target len {S_target}. Skipping length.")
            torch.cuda.empty_cache()
            for name in ["Causal FA2 (No Graph)", "Eager Matrix", "FlexGraph (Yours)"]:
                res_str = f"{S_target:<10} | {name:<22} | {'OOM (Data Gen)':<15} | {'OOM (Data Gen)':<15}"
                step_results.append(res_str)
                all_results.append(res_str)
                
            # Print skipped step results to console
            print(f"--- Testing Size: Target ~{S_target} ---")
            for res in step_results: print(res)
            print("\n")
            continue
            
        # 2. Extract baselines data (gracefully handles OOM inside)
        q_unp, k_unp, v_unp, dense_mask, S_real = get_baselines_data(
            q, k, v, chunk_bias, chunk_tokens, prefix_end, chunk_size
        )
        
        # Create gradients to catch the backward pass
        grad_out_flex = torch.randn_like(q)
        grad_out_unp = torch.randn_like(q_unp) if q_unp is not None else None

        # ----------------------------------------------------------------------
        # 3. Define the Execution Wrappers
        # ----------------------------------------------------------------------
        def run_causal():
            out = F.scaled_dot_product_attention(q_unp, k_unp, v_unp, is_causal=True)
            out.backward(grad_out_unp)
            q_unp.grad, k_unp.grad, v_unp.grad = None, None, None

        def run_eager():
            out = F.scaled_dot_product_attention(q_unp, k_unp, v_unp, attn_mask=dense_mask)
            out.backward(grad_out_unp)
            q_unp.grad, k_unp.grad, v_unp.grad = None, None, None

        def run_flex():
            out = graph_flex_attention_compiled(q, k, v, chunk_size, chunk_bias, chunk_tokens, prefix_end)
            out.backward(grad_out_flex)
            q.grad, k.grad, v.grad, chunk_bias.grad = None, None, None, None

        # ----------------------------------------------------------------------
        # 4. Memory Safety Checks for SDPA Eager
        # ----------------------------------------------------------------------
        eager_prep_success = True
        if dense_mask is not None:
            try:
                with torch.no_grad():
                    _ = F.scaled_dot_product_attention(q_unp, k_unp, v_unp, attn_mask=dense_mask)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "not supported" in str(e).lower():
                    eager_prep_success = False
                    torch.cuda.empty_cache()
                else:
                    raise e
        else:
            eager_prep_success = False

        methods = {
            "Causal FA2 (No Graph)": run_causal if q_unp is not None else None,
            "Eager Matrix": run_eager if eager_prep_success else None,
            "FlexGraph (Yours)": run_flex
        }

        # ----------------------------------------------------------------------
        # 5. Execute using Triton's do_bench
        # ----------------------------------------------------------------------
        print(f"--- Testing Size: Target ~{S_target} | Flex Padded = {S_padded} | Baseline Packed = {S_real} ---")
        print(f"{'Target Len':<10} | {'Method':<22} | {'FW+BW Time (ms)':<15} | {'Peak Memory (MB)':<15}")
        print("-" * 67)

        for name, func in methods.items():
            if func is None:
                # Differentiate between which part OOM'd based on the baseline
                reason = "OOM (Dense Mask)" if name == "Eager Matrix" and q_unp is not None else "OOM (Sequence)"
                res_str = f"{S_target:<10} | {name:<22} | {reason:<15} | {reason:<15}"
                step_results.append(res_str)
                all_results.append(res_str)
                continue

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                ms = do_bench(func)
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                res_str = f"{S_target:<10} | {name:<22} | {ms:<15.2f} | {peak_mem_mb:<15.2f}"
                step_results.append(res_str)
                all_results.append(res_str)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    res_str = f"{S_target:<10} | {name:<22} | {'OOM (Kernel)':<15} | {'OOM (Kernel)':<15}"
                    step_results.append(res_str)
                    all_results.append(res_str)
                    torch.cuda.empty_cache()
                else:
                    raise e

        for res in step_results:
            print(res)
        print("\n")

    # ==============================================================================
    # 6. Print Final Aggregation Table
    # ==============================================================================
    print("=" * 67)
    print("FINAL COMBINED BENCHMARK RESULTS")
    print("=" * 67)
    print(f"{'Target Len':<10} | {'Method':<22} | {'FW+BW Time (ms)':<15} | {'Peak Memory (MB)':<15}")
    print("-" * 67)
    for i, res in enumerate(all_results):
        print(res)
        if i % len(methods) == len(methods) - 1:
            print("-" * 67)

if __name__ == "__main__":
    run_benchmark()