"""
This module implements the utilitiy function used for computing the Relative Random Walk Probabilities (RRWP) for a given graph.
"""

import networkx as nx
import numpy as np
import torch

def compute_rrwp(graphs, max_distance: int = 8):
    """
    Highly optimized batched RRWP computation.
    Args:
        graphs: A single nx.Graph or a list of nx.Graph objects.
        max_distance: Number of random walk steps.
    Returns:
        If single graph: dict mapping (i, j) to list of floats.
        If list of graphs: list of flattened numpy arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle single graph input for backward compatibility
    is_single = isinstance(graphs, nx.Graph)
    graph_list = [graphs] if is_single else graphs
    
    num_graphs = len(graph_list)
    node_counts = [g.number_of_nodes() for g in graph_list]
    max_n = max(node_counts)

    # 1. Build Padded Adjacency Tensor [B, N, N]
    A = torch.zeros((num_graphs, max_n, max_n), device=device, dtype=torch.float32)
    for i, g in enumerate(graph_list):
        adj = nx.to_numpy_array(g)
        A[i, :node_counts[i], :node_counts[i]] = torch.from_numpy(adj)

    # 2. Handle Sink Nodes & Compute Transition Matrix M
    # Sum along rows to get out-degrees
    out_degrees = A.sum(dim=2)
    sink_mask = (out_degrees == 0)
    
    # Add self-loops to sinks to avoid division by zero
    if sink_mask.any():
        # Get indices of diagonal elements for sink nodes across the batch
        batch_idx, sink_idx = torch.where(sink_mask)
        # Only add self-loops within the valid node range for each graph
        valid_sink = sink_idx < torch.tensor(node_counts, device=device)[batch_idx]
        if valid_sink.any():
            A[batch_idx[valid_sink], sink_idx[valid_sink], sink_idx[valid_sink]] = 1.0
            out_degrees = A.sum(dim=2)

    # Clone out_degrees and set any remaining 0s (the padded nodes) to 1.0. 
    # Since A is 0 for these nodes, 0 / 1.0 safely evaluates to 0.0, preventing NaNs.
    safe_out_degrees = out_degrees.clone()
    safe_out_degrees[safe_out_degrees == 0] = 1.0

    # Transition Matrix M = D^-1 * A
    M = A / safe_out_degrees.unsqueeze(2)

    # 3. Iterative Power Computation [B, N, N, Steps]
    RRWP = torch.zeros((num_graphs, max_n, max_n, max_distance), device=device)
    current_power = torch.eye(max_n, device=device).unsqueeze(0).repeat(num_graphs, 1, 1)

    for d in range(max_distance):
        if d == 0:
            RRWP[:, :, :, d] = current_power
        else:
            # Batch Matrix Multiplication: [B, N, N] @ [B, N, N]
            current_power = torch.bmm(current_power, M)
            RRWP[:, :, :, d] = current_power

    # 4. Format Output
    RRWP_cpu = RRWP.cpu().numpy()

    if is_single:
        # Maintain old dictionary format for single graph calls
        n = node_counts[0]
        res_dict = {}
        for i in range(n):
            for j in range(n):
                res_dict[(i, j)] = RRWP_cpu[0, i, j, :].tolist()
        return res_dict
    else:
        # Return list of flattened arrays for the Dataset map function
        results = []
        for b in range(num_graphs):
            n = node_counts[b]
            # Slice out the padding and flatten: (n, n, dist) -> (n*n*dist,)
            results.append(RRWP_cpu[b, :n, :n, :].flatten())
        return results

if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')])
    rrwp_dict = compute_rrwp(G, max_distance=5)
    for key in rrwp_dict:
        print(f"{key}: {', '.join([f'{val:.2f}' for val in rrwp_dict[key]])}")