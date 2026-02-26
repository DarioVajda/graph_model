"""
This module implements the utility function for computing the Random Walk Structural Encodings (RWSE) for a given graph.
"""

import networkx as nx
import numpy as np

from .rrwp import compute_rrwp

def compute_rwse(graph: nx.Graph, max_distance: int = 4):
    """
    Computes the Random Walk Structural Encodings (RWSE) for a given graph up to a specified maximum distance.
    
    Args:
        graph: A NetworkX graph object representing the input graph.
        max_distance: The maximum shortest path distance to consider for computing RWSE. Default is 4.
    Returns:
        A dictionary where keys are nodes i and the values are the lists from RRWP for (i, i) pairs, which represent the RWSE for each node.
    """
    # Compute the RRWP for the graph
    rrwp_dict = compute_rrwp(graph, max_distance)

    # Extract the RWSE for each node (which corresponds to the RRWP values for (i, i) pairs)
    rwse_dict = {}
    for node in graph.nodes():
        rwse_dict[node] = rrwp_dict[(node, node)]  # Get the RRWP values for the (i, i) pair which represents the RWSE for node i
    
    return rwse_dict

if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')])
    rwse = compute_rwse(G, max_distance=5)
    for key in rwse:
        print(f"{key}: {', '.join([f'{val:.2f}' for val in rwse[key]])}")