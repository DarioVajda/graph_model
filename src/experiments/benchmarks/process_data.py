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

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add current directory for train modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_seed(seed):
    # set environment variables to ensure CUDA determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_text_graph(target_node, nodes, edges, label):
    graph = nx.Graph()

    # add nodes to the graph
    for node_id, text in nodes.items():
        if node_id == target_node:
            text = text + label
        graph.add_node(node_id, text=text)

    # add edges to the graph
    for source, target in edges:
        graph.add_edge(source, target)
    
    # set the prompt_node attribute to the target node
    graph.graph['prompt_node'] = target_node

    return graph


def get_neighborhood(data, node, subset, edge_index, distances, max_nodes=25, mapping=None, instruction=None):
    """
    Get the neighborhood of a node in a graph up to a certain number of hops, with a maximum number of nodes, which will be randomly sampled (where closer nodes have priority) up to the maximum number of nodes.

    Args:
        data: PyG data object containing the graph.
        node: The target node for which to extract the neighborhood.
        hops: The number of hops to consider for the neighborhood.
        max_nodes: The maximum number of nodes to include in the neighborhood.
        mapping: A function that takes the data object and a node index, and returns a string of text to be associated with the given node.
        instruction: A string containing the instruction to be associated with the given node.
    Returns:
        Tuple(List[Dict], List[List[int]], String): A tuple containing a list of node feature dictionaries and a list of edges (as tuples of node indices).
        - nodes:
            - 'id': The original node ID in the graph.
            - ... (other node features from data
        - edges:
            - (source_node_id, target_node_id)
        - label: The label of the target node (if available in data.y)
    """
    if mapping is None:
        raise ValueError("Mapping function must be provided to convert node indices to text features.")
    if instruction is None:
        raise ValueError("Instruction string must be provided to be associated with the target node.")

    if subset.shape[0] > max_nodes:

        # add random noise in range (0, 0.5) to the distances to break ties randomly and select the first max_nodes closest nodes
        noisy_distances = {n: dist + random.uniform(0, 0.5) for n, dist in distances.items()}
        sorted_nodes = sorted(subset.tolist(), key=lambda x: noisy_distances.get(x, float('inf')))
        filtered_nodes = sorted(sorted_nodes[:max_nodes])
        subset = torch.tensor(filtered_nodes, dtype=torch.long)
        
        # filter edge_index to only include edges between the selected nodes
        edge_index = edge_index[:, (torch.isin(edge_index[0], subset) & torch.isin(edge_index[1], subset))]

    nodes = {
        idx.item(): mapping(data, idx, distances[idx.item()]) + (f"\n{instruction}" if idx.item() == node else "")
        for idx in subset
    }
    edges = edge_index.t().tolist() # convert edge_index to list of [source, target]
    label = data.label_texts[data.y[node]]

    text_graph = to_text_graph(node, nodes, edges, label)
    return text_graph

def save_text_graph_dataset(graphs, path, params=None, per_graph_versions=1):
    dataset = TextGraphDataset(graphs, per_graph_versions=per_graph_versions)
    dataset.tokenize(params['tokenizer'], max_length=params['max_length'], add_eos=True)
    dataset.compute_shortest_path_distances()
    dataset.compute_rrwp(max_rrwp_steps=params['max_rrwp_steps'])
    dataset.compute_magnetic_lap(q=params['magnetic_q'])
    dataset.compute_labels(params['get_graph_labels'])
    dataset.save(path)


def construct_subgraphs(
    data, 
    hops=2, 
    max_nodes=25, 
    samples=4, 
    mapping=None, 
    instruction=None, 
    params=None, 
    save_path=None, 
    only_split=None, 
    per_graph_versions=1, 
    max_train_samples=5000,
    max_val_samples=2000,
):
    graphs = {
        "train": [],
        "val": [],
        "test": []
    }
    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    train_samples_seen = 0
    val_samples_seen = 0
    process_every_k_train_samples = 1 if max_train_samples is None or max_train_samples >= data.train_mask.sum().item() else data.train_mask.sum().item() // max_train_samples
    process_every_k_val_samples = 1 if max_val_samples is None or max_val_samples >= data.val_mask.sum().item() else data.val_mask.sum().item() // max_val_samples

    print(f"Processing only the {only_split} split." if only_split is not None else "Processing all splits.")

    for node in range(data.num_nodes):
        if data.train_mask[node]:  
            split = "train"
            train_samples_seen += 1
            if train_samples_seen % process_every_k_train_samples != 0:
                continue
        elif data.val_mask[node]:  
            split = "val"
            val_samples_seen += 1
            if val_samples_seen % process_every_k_val_samples != 0:
                continue
        elif data.test_mask[node]: 
            split = "test"

        if node % 500 == 0:
            print(f"Processing {node}/{data.num_nodes}." + (f"{only_split} split graphs: {len(graphs[only_split])}" if only_split is not None else ""))

        if only_split is not None and split != only_split:
            continue

        # get the k-hop neighborhood of the node
        subset, edge_index, _, _ = k_hop_subgraph(node, hops, data.edge_index)

        # compute distances from the target node to all other nodes in the subset
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        distances = nx.single_source_shortest_path_length(G, node)

        for _ in range((samples if split == "train" else 1)):
            graph = get_neighborhood(data, node, subset, edge_index, distances=distances, max_nodes=max_nodes, mapping=mapping, instruction=instruction)
            graphs[split].append(graph)

       
        
        if len(graphs[split]) % 10000 == 0:
            print('-'*25 + f" Processed {split_counts[split]+len(graphs[split])} graphs for {split} split. Saving to disk... " + '-'*25)
            first, last = split_counts[split], split_counts[split] + len(graphs[split])
            subset_save_path = save_path + f"/{split}/{first}-{last}"
            os.makedirs(os.path.dirname(subset_save_path), exist_ok=True)
            save_text_graph_dataset(graphs[split], subset_save_path, params=params, per_graph_versions=(per_graph_versions if split == "train" else 1))
            split_counts[split] += len(graphs[split])
            graphs[split] = []
        
    for split in ["train", "val", "test"]:
        if len(graphs[split]) > 0:
            print('-'*25 + f" Saving remaining {len(graphs[split])} graphs for {split} split to disk... " + '-'*25)
            first, last = split_counts[split], split_counts[split] + len(graphs[split])
            subset_save_path = save_path + f"/{split}/{first}-{last}"
            os.makedirs(os.path.dirname(subset_save_path), exist_ok=True)
            save_text_graph_dataset(graphs[split], subset_save_path, params=params, per_graph_versions=(per_graph_versions if split == "train" else 1))
            split_counts[split] += len(graphs[split])
            graphs[split] = []



#region Text Mappings
def get_titles(data, idx, target_dist):
    return f"{data.title[idx]}"
def get_abstracts(data, idx, target_dist):
    return f"{data.title[idx]}\n{data.abs[idx]}"
def get_titles_and_target_abstract(data, idx, target_dist):
    if target_dist == 0:
        return get_abstracts(data, idx, target_dist)
    else:
        return get_titles(data, idx, target_dist)
def get_titles_and_neighbor_abstracts(data, idx, target_dist):
    if target_dist <= 1:
        return get_abstracts(data, idx, target_dist)
    else:
        return get_titles(data, idx, target_dist)
class GetRandomAbstracts:
    def __init__(self, p):
        self.p = p
    def __call__(self, data, idx, target_dist):
        if target_dist == 0 or random.random() < self.p ** target_dist:
            return get_abstracts(data, idx, target_dist)
        else:
            return get_titles(data, idx, target_dist)
def get_reddit_text(data, idx, target_dist):
    return f"{data.raw_texts[idx]}"
class GetTruncatedRedditText:
    def __init__(self, max_length):
        self.max_length = max_length
    def __call__(self, data, idx, target_dist):
        full_text = get_reddit_text(data, idx, target_dist)
        return full_text[:self.max_length] + ("..." if len(full_text) > self.max_length else "")
def get_more_target_text(data, idx, target_dist):
    lengths = [ 196, 128, 64 ]
    full_text = get_reddit_text(data, idx, target_dist)
    if target_dist == 0:
        return full_text[:lengths[0]] + ("..." if len(full_text) > lengths[0] else "")
    elif target_dist == 1:
        return full_text[:lengths[1]] + ("..." if len(full_text) > lengths[1] else "")
    else:
        return full_text[:lengths[2]] + ("..." if len(full_text) > lengths[2] else "")

def get_mapping_name(mapping):
    if mapping == get_titles:
        return "all_titles"
    elif mapping == get_abstracts:
        return "all_abstracts"
    elif mapping == get_titles_and_target_abstract:
        return "target_abstract"
    elif mapping == get_titles_and_neighbor_abstracts:
        return "neighbor_abstracts"
    elif mapping.__class__ == GetRandomAbstracts:
        return f"random_abstracts_p{mapping.p}"
    elif mapping == get_reddit_text:
        return "full_text"
    elif mapping.__class__ == GetTruncatedRedditText:
        return f"truncated_text_{mapping.max_length}"
    elif mapping == get_more_target_text:
        return "more_target_text"
    else:
        raise ValueError("Unknown mapping function")
#endregion

class GetGraphLabels:
    """
    This is a callable class responsible for finding the question end in the prompt node and masking all tokens to -100 except for the answer (which follows the question end).
    """
    def __init__(self, question_end):
        if question_end is None:
            raise ValueError("question_end parameter cannot be None. It should be a list of token IDs that indicate the end of the question in the prompt node's text.")
        self.question_end = question_end

    def __call__(self, example):
        prompt_node = example.get('prompt_node', None)
        labels = example['input_ids'][prompt_node].copy()
        prompt_input_ids = example['input_ids'][prompt_node]

        # find question end in the prompt node's input_ids
        question_end_index = None
        for i in range(len(prompt_input_ids) - len(self.question_end) + 1):
            if prompt_input_ids[i:i+len(self.question_end)] == self.question_end:
                question_end_index = i + len(self.question_end) - 1
        if question_end_index is None:
            raise ValueError(f"Could not find question end token sequence {self.question_end} in the prompt node's input_ids: {prompt_input_ids}")

        # Mask all tokens before and including the question end index to -100
        for i in range(question_end_index + 1):
            labels[i] = -100
        return labels


if __name__ == "__main__":
    setup_seed(52)
    datasets = [ 
        # ('cora', get_titles_and_target_abstract, 60),
        # ('cora', get_titles_and_neighbor_abstracts, 15),
        # ('cora', GetRandomAbstracts(p=0.2), 20),
        # ('ogbn-arxiv', get_titles_and_target_abstract, 60), 
        # ('ogbn-arxiv', GetRandomAbstracts(p=0.2), 20),
        # ('pubmed', get_titles_and_target_abstract, 60), 
        # ('reddit', GetTruncatedRedditText(max_length=64), 60),
        ('reddit', get_more_target_text, 60),
    ]
    instructions = {
        'cora':         'Q: Given this paper citation graph, classify this paper into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory. Please tell me which class does this paper belong to?\nA: ',
        'ogbn-arxiv':   'Q: Given this paper citation graph, classify this paper into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics). Please tell me which class does this paper belong to?\nA: ',
        'pubmed':       'Q: Given this paper citation graph, classify this paper into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2. Please tell me which class does this paper belong to?\nA: ',
        'reddit':       'Q: Given this user reddit user post interaction graph, classify this reddit user into 2 classes: Normal Users and Popular Users. Please tell me which class does this reddit user belong to?\nA: ',
    }
    NUM_SAMPLES = 8
    MAX_VAL_SAMPLES = 2000e10
    MAX_TRAIN_SAMPLES = 5000e10

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    params = {
        # tokenizing
        'tokenizer': tokenizer,
        'max_length': 32_768,

        # no spd params

        # rrwp params
        'max_rrwp_steps': 16,

        # magnetic laplacian params
        'magnetic_q': 0.25,

        # label computation function
        'get_graph_labels': GetGraphLabels(question_end=[ 32, 25 ]), # this represents "A:"
    }
    ONLY_SPLIT = "test"
    ONLY_SPLIT = None

    modes = ['train', 'val', 'test']
    hops = 2
    for dataset, mapping, subgraph_size in datasets:
        
        data = torch.load(f'./src/experiments/benchmarks/raw_data/{dataset}/processed_data.pt', weights_only=False)

        print("=" * 100)
        print(f"Dataset: {dataset} (num_nodes={data.x.shape[0]}, samples_per_node={NUM_SAMPLES}) --> TOTAL SAMPLES: {data.x.shape[0] * NUM_SAMPLES}")
        print("=" * 100)
        print(data)

        if ONLY_SPLIT is None:
            print("Num train nodes:", data.train_mask.sum().item())
            print("Num val nodes:", data.val_mask.sum().item())
            print("Num test nodes:", data.test_mask.sum().item())
        elif ONLY_SPLIT == "train":
            print("Using only train split:", data.train_mask.sum().item(), "nodes")
        elif ONLY_SPLIT == "val":
            print("Using only val split:", data.val_mask.sum().item(), "nodes")
        elif ONLY_SPLIT == "test":
            print("Using only test split:", data.test_mask.sum().item(), "nodes")

        print("=" * 100)

        # is_undirected_graph = is_undirected(data.edge_index)
        # has_self_loops = contains_self_loops(data.edge_index)
        # edge_index_no_duplicates, _ = coalesce(data.edge_index, None, data.x.shape[0])
        # has_duplicate_edges = data.edge_index.shape[1] != edge_index_no_duplicates.shape[1]

        # print(f"is_undirected_graph: {is_undirected_graph}")
        # print(f"has_self_loops: {has_self_loops}")
        # print(f"has_duplicate_edges: {has_duplicate_edges}")
        # print("label_texts:", data.label_texts)
        # print(data)
        # print()

        mapping_name = get_mapping_name(mapping)
        save_path = f'./src/experiments/benchmarks/processed_data/{dataset}_hops{hops}_neighbors{subgraph_size}_{mapping_name}'

        construct_subgraphs(
            data, 
            hops=hops, 
            max_nodes=subgraph_size, 
            samples=NUM_SAMPLES, 
            mapping=mapping, 
            instruction=instructions[dataset], 
            params=params, 
            save_path=save_path, 
            only_split=ONLY_SPLIT,
            per_graph_versions=NUM_SAMPLES,
            max_train_samples=MAX_TRAIN_SAMPLES,
            max_val_samples=MAX_VAL_SAMPLES,
        )
