import networkx as nx
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ...utils.text_graph_dataset import TextGraphDataset
from ...utils.spectral_coordinates import get_spectral_coordinates

LETTERS = [
    ' ' + letter for letter in
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
]

def generate_graph(size=None, min_size=5, max_size=15):
    """
        Generate a graph with nodes {0, 1, ..., N-1} where {a_1, a_2,... a_M} are a fully connected clique A and the rest a fully connected clique B.

        Arguments:
        - size: the number of nodes in the graph. If None, a random size between min_size and max_size will be chosen.
        - min_size: the minimum number of nodes in the graph (inclusive).
        - max_size: the maximum number of nodes in the graph (inclusive).

        Returns:
        - G: the generated graph (networkx.Graph)
        - x: query node 1 (int)
        - y: query node 2 (int)
        - label: 1 if x and y are in the same clique, 0 otherwise 
    """
    if size is None:
        size = random.randint(min_size, max_size)
    
    labels = list(range(size))

    size_A = random.randint(2, size-2)

    A = random.sample(labels, size_A)
    B = [label for label in labels if label not in A]

    G = nx.Graph()
    G.add_nodes_from(labels)
    G.add_edges_from([(i, j) for i in A for j in A if i != j])
    G.add_edges_from([(i, j) for i in B for j in B if i != j])

    # generate a random label and choose the query nodes accordingly
    label = random.choice([0, 1])

    if label == 0:
        x = random.choice(A)
        y = random.choice(B)
    else:
        if random.choice([True, False]): # choose A with 50% probability
            x, y = random.sample(A, 2)
        else:
            x, y = random.sample(B, 2)

    if random.choice([True, False]): # swap x and y with 50% probability
        x, y = y, x

    return G, x, y, label

def generate_graph_dataset(num_examples, min_size=5, max_size=15):
    dataset = []
    for _ in range(num_examples):
        G, x, y, label = generate_graph(min_size=min_size, max_size=max_size)
        dataset.append((G, x, y, label))
    return dataset

def get_prompt_node_labels(example):
    """
    Function used to compute the labels which will provide the training signal on the prompt node.
    For this experiment, the only token used for training is the last token of the prompt node's text, which will be "Yes" or "No".
    """
    prompt_node = example['prompt_node']
    labels = example['input_ids'][prompt_node].copy()
    labels[:-1] = [-100] * (len(labels) - 1)
    return labels


def prepare_dataset(num_examples, min_size=5, max_size=15, spectral_dims=8, tokenizer_name=None):
    dataset = generate_graph_dataset(num_examples, min_size=min_size, max_size=max_size)

    # FOR EACH EXAMPLE:
    # 1. compute a random subset of the LETTERS and use them as the "text" fields of the nodes
    # 2. add the prompt node with the "text" field containing the question "Are the nodes X and Y connected? {'Yes' if label == 1 else 'No'}"
    # 3. set the new node's id as the graph-level attribute "prompt_node"
    
    graphs = []
    for G, x, y, label in tqdm(dataset, desc="Preparing dataset"):
        # 1. compute a random subset of the LETTERS and use them as the "text" fields of the nodes
        node_texts = random.sample(LETTERS, len(G.nodes))
        for node, text in zip(G.nodes, node_texts):
            G.nodes[node]['text'] = text

        # 2. add the prompt node with the "text" field containing the question "Are the nodes X and Y connected? [Yes/No]"
        prompt_node_id = max(G.nodes) + 1
        G.add_node(prompt_node_id, text=f"Are the nodes{G.nodes[x]['text']} and{G.nodes[y]['text']} connected? {'Yes' if label == 1 else 'No'}")
        G.add_edge(prompt_node_id, x)
        G.add_edge(prompt_node_id, y)

        # 3. set the new node's id as the graph-level attribute "prompt_node"
        G.graph['prompt_node'] = prompt_node_id
        graphs.append(G)

    ds = TextGraphDataset(graphs)
    ds.compute_spectral_coordinates(embedding_dim=spectral_dims)
    ds.compute_shortest_path_distances()

    if tokenizer_name is None:
        raise ValueError("Tokenizer must be provided to prepare_dataset function.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds.tokenize(tokenizer)

    ds.compute_labels(get_prompt_node_labels)

    return ds

def round_size_str(size):
    if size >= 1_000_000:
        return f"{size // 1_000_000}M", size // 1_000_000, 1_000_000
    elif size >= 1_000:
        return f"{size // 1_000}k", size // 1_000, 1_000
    else:
        return str(size), size, 1

def dataset_path_and_size(dataset_size):
    size_str, rounded_size, scale = round_size_str(dataset_size)
    dataset_path = f"./src/experiments/expressiveness/new_{size_str}_dataset.gtds"
    return dataset_path, rounded_size * scale

def create_and_save_dataset(dataset_size, min_nodes, max_nodes, spectral_dims, model_name):
    dataset_path, final_dataset_size = dataset_path_and_size(dataset_size)

    dataset = prepare_dataset(final_dataset_size, min_size=min_nodes, max_size=max_nodes, spectral_dims=spectral_dims, tokenizer_name=model_name)

    # Save the dataset to disk
    dataset.save(dataset_path)

    return dataset, dataset_path

if __name__ == "__main__":
    MIN_NODES = 10
    MAX_NODES = 20
    SPECTRAL_DIMS = 16
    DATASET_SIZE = 1000
    model_name = "meta-llama/Llama-3.2-1B"

    print(f"Creating dataset with {DATASET_SIZE // 1000}k examples, node sizes between {MIN_NODES} and {MAX_NODES}, spectral dimensions {SPECTRAL_DIMS}, and tokenizer {model_name}...")

    _, dataset_path = create_and_save_dataset(DATASET_SIZE, MIN_NODES, MAX_NODES, SPECTRAL_DIMS, model_name)
    print(f"Dataset created and saved at {dataset_path}")

    # laod the dataset to verify it works
    print(f"Loading dataset from {dataset_path} to verify...")
    loaded_ds = TextGraphDataset.load(dataset_path)
    print(f"Dataset loaded successfully with {len(loaded_ds)} examples.")
    
    print("Example graph:")
    for key in loaded_ds[0].keys():
        print('--------------------------------')
        print(f"{key}:\n{loaded_ds[0][key]}")