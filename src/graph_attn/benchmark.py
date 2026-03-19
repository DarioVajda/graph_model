import torch
import torch.nn.functional as F
from triton.testing import do_bench

from .flex_attn import graph_flex_attention

def run_benchmark():
    # --- Benchmark Parameters ---
    B, H, D = 4, 16, 64  # Batch size, Heads, Head Dimension
    seq_lengths = [ 1024, 2048, 4096, 8192, 16384 ] # Sweep across sequence lengths
    # node_count = 256
    device = "cuda"
    dtype = torch.float16

    print("\nStarting Benchmark... (Note: First step may take a few minutes to compile)")
    
    # List to hold ALL results for the final printout
    all_results = []

    for S in seq_lengths:
        node_count = S // 32
        # List to hold results just for this specific sequence length step
        step_results = []
        
        # 1. Setup Data for this Sequence Length
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        grad_out = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # Graph node biases
        node_bias = torch.rand(B, H, node_count, node_count, device=device, dtype=dtype, requires_grad=True)

        # Create sorted token_node_ids of variable lengths
        token_node_ids = torch.zeros((B, S), dtype=torch.int32, device=device)
        for b in range(B):
            cutoffs = torch.randint(1, S, (node_count - 1,), device=device).sort()[0]
            cutoffs = torch.cat([torch.tensor([0], device=device), cutoffs, torch.tensor([S], device=device)])
            for n in range(node_count):
                token_node_ids[b, cutoffs[n]:cutoffs[n+1]] = n

        # Boundary variables for Left/Right padding and Prefix
        valid_start = torch.zeros(B, dtype=torch.int32, device=device) # Right padded simulation
        valid_end = torch.randint(S // 2, S, (B,), dtype=torch.int32, device=device)
        prefix_end = torch.randint(S // 4, S // 2, (B,), dtype=torch.int32, device=device)

        # =====================================================================
        # 2. Build the FAIR Dense Mask for Eager PyTorch (Helper Function)
        # =====================================================================
        def get_dense_mask(node_bias_tensor):
            b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
            h_idx = torch.arange(H, device=device).view(1, H, 1, 1)
            q_node = token_node_ids.view(B, 1, S, 1)
            kv_node = token_node_ids.view(B, 1, 1, S)
            dense_bias = node_bias_tensor[b_idx, h_idx, q_node, kv_node]

            q_idx_tensor = torch.arange(S, device=device).view(1, 1, S, 1)
            kv_idx_tensor = torch.arange(S, device=device).view(1, 1, 1, S)
            
            vs = valid_start.view(B, 1, 1, 1)
            ve = valid_end.view(B, 1, 1, 1)
            pe = prefix_end.view(B, 1, 1, 1)

            q_is_valid = (q_idx_tensor >= vs) & (q_idx_tensor < ve)
            kv_is_valid = (kv_idx_tensor >= vs) & (kv_idx_tensor < ve)
            topology_mask = (kv_idx_tensor < pe) | (q_idx_tensor >= kv_idx_tensor)
            
            valid_attention = q_is_valid & kv_is_valid & topology_mask
            is_padding_self_attention = (~q_is_valid) & (q_idx_tensor == kv_idx_tensor)
            
            final_bool_mask = valid_attention | is_padding_self_attention

            return torch.where(
                final_bool_mask, 
                dense_bias, 
                torch.tensor(-float('inf'), device=device, dtype=dtype)
            )

        # Check for OOM just to be safe before benchmarking
        try:
            with torch.no_grad(): # Don't need graph just for an OOM check
                 _ = get_dense_mask(node_bias)
            eager_prep_success = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                eager_prep_success = False
                torch.cuda.empty_cache()
            else:
                raise e

        # =====================================================================
        # 3. Define the Execution Wrappers
        # =====================================================================
        # graph_flex_attention_compiled = torch.compile(graph_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
        graph_flex_attention_compiled = torch.compile(graph_flex_attention)

        def run_causal():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            # out.backward(grad_out)

        def run_eager():
            # Build the mask inside the loop so the graph is fresh!
            current_dense_mask = get_dense_mask(node_bias)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=current_dense_mask)
            # out.backward(grad_out)

        def run_flex():
            out = graph_flex_attention_compiled(
                q, k, v, 
                node_bias, token_node_ids, 
                valid_start, valid_end, prefix_end
            )
            # out.backward(grad_out)

        methods = {
            "Causal FA2": run_causal,
            "Eager PyTorch": run_eager if eager_prep_success else None,
            "FlexGraph": run_flex
        }

        # =====================================================================
        # 4. Execute and Benchmark
        # =====================================================================
        for name, func in methods.items():
            if func is None:
                res_str = f"{S:<10} | {name:<20} | {'OOM (Prep)':<15} | {'OOM (Prep)':<15}"
                step_results.append(res_str)
                all_results.append(res_str)
                continue

            # Reset tracking states to ensure fair isolation
            q.grad, k.grad, v.grad = None, None, None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                # do_bench natively handles CUDA warmups and syncs
                ms = do_bench(func)
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                res_str = f"{S:<10} | {name:<20} | {ms:<15.2f} | {peak_mem_mb:<15.2f}"
                step_results.append(res_str)
                all_results.append(res_str)
                
            except RuntimeError as e:
                # Safely catch Out of Memory errors without crashing the loop
                if "out of memory" in str(e).lower():
                    res_str = f"{S:<10} | {name:<20} | {'OOM':<15} | {'OOM':<15}"
                    step_results.append(res_str)
                    all_results.append(res_str)
                    torch.cuda.empty_cache()
                else:
                    raise e
                    
        # --- Print the Step Results safely ---
        print(f"\n--- Results for Sequence Length: {S} ---")
        print(f"{'Seq Len':<10} | {'Method':<20} | {'FW+BW Time (ms)':<15} | {'Peak Memory (MB)':<15}")
        print("-" * 65)
        for res in step_results:
            print(res)

    # =====================================================================
    # 5. Print the Final Aggregated Table
    # =====================================================================
    print("\n" + "=" * 65)
    print("FINAL COMBINED BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Seq Len':<10} | {'Method':<20} | {'FW+BW Time (ms)':<15} | {'Peak Memory (MB)':<15}")
    print("-" * 65)
    for i, res in enumerate(all_results):
        print(res)
        if i % (len(methods)) == len(methods) - 1:  # After every set of methods for a sequence length
            print("-" * 65)
    print("-" * 65)

if __name__ == "__main__":
    run_benchmark()


"""
DEFAULT BIAS RESULTS:
=================================================================
FINAL COMBINED BENCHMARK RESULTS
=================================================================
Seq Len    | Method               | FW+BW Time (ms) | Peak Memory (MB)
-----------------------------------------------------------------
1024       | Causal FA2           | 0.43            | 505.02         
1024       | Eager PyTorch        | 36.05           | 3795.52        
1024       | FlexGraph            | 1.81            | 474.59         
-----------------------------------------------------------------
2048       | Causal FA2           | 1.26            | 1009.53        
2048       | Eager PyTorch        | 145.67          | 14277.81       
2048       | FlexGraph            | 7.32            | 947.30         
-----------------------------------------------------------------
4096       | Causal FA2           | 4.10            | 2786.57        
4096       | Eager PyTorch        | 572.22          | 55877.86       
4096       | FlexGraph            | 41.13           | 2661.10        
-----------------------------------------------------------------
8192       | Causal FA2           | 14.55           | 3268.63        
8192       | Eager PyTorch        | OOM (Prep)      | OOM (Prep)     
8192       | FlexGraph            | 134.94          | 3018.19        
-----------------------------------------------------------------
16384      | Causal FA2           | 54.41           | 4232.75        
16384      | Eager PyTorch        | OOM (Prep)      | OOM (Prep)     
16384      | FlexGraph            | 800.84          | 3738.41        
-----------------------------------------------------------------
32768      | Causal FA2           | 210.90          | 6161.00        
32768      | Eager PyTorch        | OOM (Prep)      | OOM (Prep)     
32768      | FlexGraph            | 5987.54         | 5202.81        
-----------------------------------------------------------------

FLATTENED BIAS RESULTS:
=================================================================
FINAL COMBINED BENCHMARK RESULTS
=================================================================
Seq Len    | Method               | FW+BW Time (ms) | Peak Memory (MB)
-----------------------------------------------------------------
1024       | Causal FA2           | 0.43            | 505.02         
1024       | Eager PyTorch        | 35.97           | 3795.52        
1024       | FlexGraph            | 1.57            | 474.59         
-----------------------------------------------------------------
2048       | Causal FA2           | 1.26            | 1009.53        
2048       | Eager PyTorch        | 145.14          | 14277.81       
2048       | FlexGraph            | 8.48            | 947.30         
-----------------------------------------------------------------
4096       | Causal FA2           | 4.10            | 2786.57        
4096       | Eager PyTorch        | 577.23          | 55877.86       
4096       | FlexGraph            | 44.38           | 2661.10        
-----------------------------------------------------------------
8192       | Causal FA2           | 14.52           | 3268.63        
8192       | Eager PyTorch        | OOM (Prep)      | OOM (Prep)     
8192       | FlexGraph            | 179.93          | 3018.19        
-----------------------------------------------------------------
16384      | Causal FA2           | 54.10           | 4232.75        
16384      | Eager PyTorch        | OOM (Prep)      | OOM (Prep)     
16384      | FlexGraph            | 958.32          | 3738.41        
-----------------------------------------------------------------
32768      | Causal FA2           | 209.24          | 6161.00        
32768      | Eager PyTorch        | OOM (Prep)      | OOM (Prep)     
32768      | FlexGraph            | 5696.64         | 5202.81        
-----------------------------------------------------------------
"""