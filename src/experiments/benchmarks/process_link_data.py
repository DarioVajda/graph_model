import json
import os
import sys
import torch
from torch_geometric.utils import coalesce, is_undirected, contains_self_loops, k_hop_subgraph
from tqdm import tqdm
import random
import numpy as np
import networkx as nx

from ...utils import TextGraphDataset


# ...WORK IN PROGRESS (WILL FINISH LATER)...
def save_link_prediction_data():
    pass

def process_data(dataset, split):
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

    for i in range(10):
        print(f"{i}: {json.dumps(split_data[i], indent=4)}")
        print('-'*20)

    data = torch.load(f'./src/experiments/benchmarks/raw_data/{dataset}/processed_data.pt', weights_only=False)
    

    save_link_prediction_data()

if __name__ == "__main__":
    process_data(dataset = 'cora', split = 'train')