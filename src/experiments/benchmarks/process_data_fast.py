import json
import os
import sys
import torch
import multiprocessing as mp
from torch_geometric.utils import coalesce, is_undirected, contains_self_loops, k_hop_subgraph
from tqdm import tqdm
import random
import numpy as np
import networkx as nx

from ...utils import TextGraphDataset

# Disable HuggingFace Tokenizer Rust parallelism to prevent deadlocks in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add current directory for train modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_seed(seed):
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
    for node_id, text in nodes.items():
        if node_id == target_node:
            text = text + label
        graph.add_node(node_id, text=text)
    for source, target in edges:
        graph.add_edge(source, target)
    graph.graph['prompt_node'] = target_node
    return graph

def get_neighborhood(data, node, subset, edge_index, distances, max_nodes=25, mapping=None, instruction=None):
    if mapping is None: raise ValueError("Mapping function must be provided.")
    if instruction is None: raise ValueError("Instruction string must be provided.")

    if subset.shape[0] > max_nodes:
        noisy_distances = {n: dist + random.uniform(0, 0.5) for n, dist in distances.items()}
        sorted_nodes = sorted(subset.tolist(), key=lambda x: noisy_distances.get(x, float('inf')))
        filtered_nodes = sorted(sorted_nodes[:max_nodes])
        subset = torch.tensor(filtered_nodes, dtype=torch.long)
        edge_index = edge_index[:, (torch.isin(edge_index[0], subset) & torch.isin(edge_index[1], subset))]

    nodes = {
        idx.item(): mapping(data, idx, distances[idx.item()]) + (f"\n{instruction}" if idx.item() == node else "")
        for idx in subset
    }
    edges = edge_index.t().tolist()
    label = data.label_texts[data.y[node]]

    return to_text_graph(node, nodes, edges, label)

def save_text_graph_dataset(graphs, path, params=None, per_graph_versions=1):
    dataset = TextGraphDataset(graphs, per_graph_versions=per_graph_versions)
    dataset.tokenize(params['tokenizer'], max_length=params['max_length'], add_eos=True)
    dataset.compute_shortest_path_distances()
    dataset.compute_rrwp(max_rrwp_steps=params['max_rrwp_steps'])
    dataset.compute_magnetic_lap(q=params['magnetic_q'])
    dataset.compute_labels(params['get_graph_labels'])
    dataset.save(path)


# --- MULTIPROCESSING WORKER SETUP ---

# Globals to hold read-only memory inside workers
_WORKER_CACHE = {}

def init_worker(data, params, mapping):
    """Initializes large objects once per worker to avoid huge IPC/Pickle overheads."""
    global _WORKER_CACHE
    _WORKER_CACHE['data'] = data
    _WORKER_CACHE['params'] = params
    _WORKER_CACHE['mapping'] = mapping

def process_chunk(args):
    """Worker function to process a batch of nodes and save directly to disk."""
    split, first_idx, last_idx, chunk_nodes, hops, max_nodes, samples, instruction, save_path, per_graph_versions = args
    
    data = _WORKER_CACHE['data']
    params = _WORKER_CACHE['params']
    mapping = _WORKER_CACHE['mapping']
    
    graphs = []
    for node in chunk_nodes:
        subset, edge_index, _, _ = k_hop_subgraph(node, hops, data.edge_index)
        
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        distances = nx.single_source_shortest_path_length(G, node)

        for _ in range(samples):
            graph = get_neighborhood(data, node, subset, edge_index, distances=distances, max_nodes=max_nodes, mapping=mapping, instruction=instruction)
            graphs.append(graph)
            
    if graphs:
        subset_save_path = os.path.join(save_path, split, f"{first_idx}-{last_idx}")
        os.makedirs(os.path.dirname(subset_save_path), exist_ok=True)
        save_text_graph_dataset(graphs, subset_save_path, params=params, per_graph_versions=per_graph_versions)
        
    return len(graphs)

def construct_subgraphs_parallel(data, hops=2, max_nodes=25, samples=4, mapping=None, instruction=None, params=None, save_path=None, only_split=None, per_graph_versions=1):
    print(f"Processing only the {only_split} split." if only_split is not None else "Processing all splits.")

    # 1. Gather target nodes by split
    target_nodes = {"train": [], "val": [], "test": []}
    for node in range(data.num_nodes):
        if data.train_mask[node]:  split = "train" 
        elif data.val_mask[node]:  split = "val"
        elif data.test_mask[node]: split = "test"
        else: continue

        if only_split is not None and split != only_split: continue
        target_nodes[split].append(node)

    # 2. Group into chunks (aiming for ~10,000 graphs per save file like the original code)
    nodes_per_chunk = 10000 // samples if samples > 0 else 2500
    chunks = []
    
    for split, nodes in target_nodes.items():
        start_idx = 0
        for i in range(0, len(nodes), nodes_per_chunk):
            chunk = nodes[i:i + nodes_per_chunk]
            expected_graphs = len(chunk) * samples
            chunks.append((
                split, 
                start_idx, 
                start_idx + expected_graphs, 
                chunk, 
                hops, 
                max_nodes, 
                samples, 
                instruction, 
                save_path, 
                per_graph_versions
            ))
            start_idx += expected_graphs

    if not chunks:
        print("No nodes to process.")
        return

    # 3. Process via Pool
    num_workers = max(1, mp.cpu_count() - 2) # Leave 2 cores free to keep OS responsive
    print(f"Starting parallel processing across {num_workers} workers...")
    
    total_processed = 0
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(data, params, mapping)) as pool:
        with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
            for count in pool.imap_unordered(process_chunk, chunks):
                total_processed += count
                pbar.update(1)

    print(f"Successfully processed and saved {total_processed} total graphs.")


#region Text Mappings
def get_titles(data, idx, target_dist):
    return f"{data.title[idx]}"
def get_abstracts(data, idx, target_dist):
    return f"{data.title[idx]}\n{data.abs[idx]}"
def get_titles_and_target_abstract(data, idx, target_dist):
    if target_dist == 0: return get_abstracts(data, idx, target_dist)
    else: return get_titles(data, idx, target_dist)
def get_titles_and_neighbor_abstracts(data, idx, target_dist):
    if target_dist <= 1: return get_abstracts(data, idx, target_dist)
    else: return get_titles(data, idx, target_dist)
def get_reddit_text(data, idx, target_dist):
    return f"{data.raw_texts[idx]}"
#endregion

class GetGraphLabels:
    def __init__(self, question_end):
        if question_end is None: raise ValueError("question_end parameter cannot be None.")
        self.question_end = question_end

    def __call__(self, example):
        prompt_node = example.get('prompt_node', None)
        labels = example['input_ids'][prompt_node].copy()
        prompt_input_ids = example['input_ids'][prompt_node]

        question_end_index = None
        for i in range(len(prompt_input_ids) - len(self.question_end) + 1):
            if prompt_input_ids[i:i+len(self.question_end)] == self.question_end:
                question_end_index = i + len(self.question_end) - 1
        if question_end_index is None:
            raise ValueError(f"Could not find question end token sequence {self.question_end} in {prompt_input_ids}")

        for i in range(question_end_index + 1):
            labels[i] = -100
        return labels


if __name__ == "__main__":
    setup_seed(0)
    datasets = [ 
        # ('cora', get_titles_and_target_abstract, 60),
        # ('ogbn-arxiv', get_titles_and_target_abstract, 60), 
        ('pubmed', get_titles_and_target_abstract, 60), 
        # ('reddit', get_reddit_text, 20),
    ]
    instructions = {
        'cora':         'Q: Given this paper citation graph, classify this paper into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory. Please tell me which class does this paper belong to?\nA: ',
        'ogbn-arxiv':   'Q: Given this paper citation graph, classify this paper into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics). Please tell me which class does this paper belong to?\nA: ',
        'pubmed':       'Q: Given this paper citation graph, classify this paper into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2. Please tell me which class does this paper belong to?\nA: ',
        'reddit':       'Q: Given this user reddit user post interaction graph, classify this reddit user into 2 classes: Normal Users and Popular Users. Please tell me which class does this reddit user belong to?\nA: ',
    }

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    params = {
        'tokenizer': tokenizer,
        'max_length': 32_768,
        'max_rrwp_steps': 16,
        'magnetic_q': 0.25,
        'get_graph_labels': GetGraphLabels(question_end=[ 32, 25 ]),
    }
    NUM_SAMPLES = 4
    ONLY_SPLIT = None
    hops = 2
    
    for dataset, mapping, subgraph_size in datasets:
        
        data = torch.load(f'./src/experiments/benchmarks/raw_data/{dataset}/processed_data.pt', weights_only=False)

        print("=" * 100)
        print(f"Dataset: {dataset} (num_nodes={data.x.shape[0]}, samples_per_node={NUM_SAMPLES}) --> TOTAL SAMPLES: {data.x.shape[0] * NUM_SAMPLES}")
        print("=" * 100)

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

        save_path = f'./src/experiments/benchmarks/processed_data/{dataset}_hops{hops}_neighbors{subgraph_size}'

        construct_subgraphs_parallel(
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
        )