"""
This module implements the utilitiy function used for computing the Relative Random Walk Probabilities (RRWP) for a given graph.
"""

import networkx as nx
import numpy as np

def compute_rrwp(graph: nx.Graph, max_distance: int = 4):
    """
    Computes the Relative Random Walk Probabilities (RRWP) for a given graph up to a specified maximum distance.
    
    Args:
        graph: A NetworkX graph object representing the input graph.
        max_distance: The maximum shortest path distance to consider for computing RRWP. Default is 4.
    Returns:
        A dictionary where keys are node pairs (i, j) and values are the computed RRWP values for those pairs.
    """
    # get adjacency matrix of the graph 
    A = nx.to_numpy_array(graph)

    # compute the out-degrees
    out_degrees = A.sum(axis=1)

    # Add self-loops to sink nodes (those with zero out-degree)
    sink_nodes = np.where(out_degrees == 0)[0]
    if len(sink_nodes) > 0:
        A[sink_nodes, sink_nodes] = 1   # Add self-loops to sink nodes
        out_degrees = A.sum(axis=1)     # Recompute out-degrees after adding self-loops

    # compute the inverse out-degree matrix
    D_inv = np.diag(1 / out_degrees)  # Add small value to avoid division by zero

    # compute the M matrix (transition probabilities)
    M = D_inv @ A

    # initialise the RRWP 3D array to store the probabilities for each distance
    num_nodes = A.shape[0]
    RRWP = np.zeros((num_nodes, num_nodes, max_distance))

    # compute RRWP for each distance
    for d in range(max_distance):
        if d == 0:
            RRWP[:, :, d] = np.eye(num_nodes)
        else:
            RRWP[:, :, d] = M @ RRWP[:, :, d-1]

    # convert the RRWP 3D array into a dictionary for easier access
    rrwp_dict = {}
    for i, node_i in enumerate(graph.nodes()):
        for j, node_j in enumerate(graph.nodes()):
            rrwp_dict[(node_i, node_j)] = RRWP[i, j, :].tolist()  # Store the RRWP values for all distances as a list
    
    return rrwp_dict

if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')])
    rrwp_dict = compute_rrwp(G, max_distance=5)
    for key in rrwp_dict:
        print(f"{key}: {', '.join([f'{val:.2f}' for val in rrwp_dict[key]])}")