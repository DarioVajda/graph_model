import json
import os
import sys
import torch
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, subgraph
from tqdm import tqdm
import random
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path

from ...utils import TextGraphDataset


def get_neighborhood(graph, edge, hops=2, max_nodes=30):
    """
    Extracts a subgraph around an edge, prioritizing nodes with the lowest 
    cumulative distance to the source and destination. Also returns the 
    shortest-path distances to both target nodes.
    """
    node_a, node_b = edge
    edge_index = graph.edge_index
    
    # 1. Extract the initial boundary subgraph
    target_nodes = torch.tensor([node_a, node_b], dtype=torch.long)
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=target_nodes, 
        num_hops=hops, 
        edge_index=edge_index, 
        relabel_nodes=True
    )
    
    num_nodes = subset.size(0)
    mapped_a, mapped_b = mapping[0].item(), mapping[1].item()
    
    # 2. Calculate shortest paths (since we need to return them)
    adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes=num_nodes)
    dist_matrix = shortest_path(
        adj, 
        directed=False, 
        unweighted=True, 
        indices=[mapped_a, mapped_b]
    )
    
    # 3. Handle cases where no downsampling is needed
    if num_nodes <= max_nodes:
        # Note: Unreachable nodes get 'inf', float32 handles this natively
        dist_to_a = torch.tensor(dist_matrix[0], dtype=torch.float32)
        dist_to_b = torch.tensor(dist_matrix[1], dtype=torch.float32)
        return subset, sub_edge_index, dist_to_a, dist_to_b

    # 4. Downsampling logic (if budget is exceeded)
    dist_sum = dist_matrix[0] + dist_matrix[1]
    
    # Add random noise for tie-breaking
    noise = np.random.rand(num_nodes) * 0.5
    noise[mapped_a] -= 10.0 # Anchor target A
    noise[mapped_b] -= 10.0 # Anchor target B
    
    randomized_dist = dist_sum + noise
    top_indices = np.argsort(randomized_dist)[:max_nodes]
    
    # 5. Build final constraints and slice distance arrays
    final_subset = subset[top_indices]
    
    final_edge_index, _ = subgraph(
        subset=final_subset, 
        edge_index=edge_index, 
        relabel_nodes=True
    )
    
    # Extract distances ONLY for the nodes we kept
    dist_to_a = torch.tensor(dist_matrix[0][top_indices], dtype=torch.float32)
    dist_to_b = torch.tensor(dist_matrix[1][top_indices], dtype=torch.float32)
    
    return final_subset, final_edge_index, dist_to_a, dist_to_b

def prepare_example(nodes, edges, dist_to_a, dist_to_b, label):
    """
    Prepares a single example for link prediction to be used in the TextGraphDataset.
    """
    # TODO: IMPLEMENT THIS
    return nodes

def save_link_prediction_data(data, split_data, max_nodes=30, samples_per_edge=1):
    graphs = []

    for i, item in enumerate(split_data):
        for _ in range(samples_per_edge):
            nodes, edges, dist_to_a, dist_to_b = get_neighborhood(data, (item['source'], item['target']), hops=2, max_nodes=max_nodes)
            processed_example = prepare_example(nodes, edges, dist_to_a, dist_to_b, item['label'])
            graphs.append(processed_example)

        if len(graphs) >= 10000:
            # save the graphs to disk in batches to avoid memory issues
            graphs = []
            pass
    
    if len(graphs) > 0:
        # save any remaining graphs
        pass

    print("Done processing link prediction data.")
    print(f"There were {len(split_data)} edges in the split, and we processed {i+1} of them.")
    return None

def process_data(dataset, split, max_nodes=30, samples_per_edge=1):
    split_path = f'./src/experiments/benchmarks/raw_data/{dataset}/edge_sampled_2_10_only_{split}.jsonl'
    split_data = []
    with open(split_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            split_data.append({
                'source': obj['id'][0],
                'target': obj['id'][1],
                'label': obj['conversations'][1]['value'],
            })

    data = torch.load(f'./src/experiments/benchmarks/raw_data/{dataset}/processed_data.pt', weights_only=False)
    
    save_link_prediction_data(data, split_data, max_nodes=max_nodes, samples_per_edge=samples_per_edge)


if __name__ == "__main__":
    process_data(dataset='cora', split='train', max_nodes=60, samples_per_edge=1)