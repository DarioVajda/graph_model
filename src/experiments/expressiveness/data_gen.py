import networkx as nx
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ...utils.text_graph_dataset import TextGraphDataset

LETTERS = [
    ' ' + letter for letter in
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
]

import random
import networkx as nx
import itertools

def generate_easy_graph(size=None, min_size=5, max_size=15, balanced=True):
    """
        Generate a graph with nodes {0, 1, ..., N-1} where {a_1, a_2,... a_M} are a fully connected clique A and the rest a fully connected clique B.
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

    if balanced:
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
    else:
        x, y = random.sample(labels, 2)
        label = 1 if (x in A and y in A) or (x in B and y in B) else 0

    if random.choice([True, False]): # swap x and y with 50% probability
        x, y = y, x

    return G, x, y, label


def _generate_random_connected_component(nodes):
    """Helper function to generate a connected subgraph for a given set of nodes."""
    N = len(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    if N < 2:
        return G
        
    # 1. Randomly choose number of edges E
    min_edges = N - 1
    max_edges = N * (N - 1) // 2
    E = random.randint(min_edges, max_edges)
    
    # 2. Create a random tree to ensure connectivity
    complete = nx.complete_graph(N)
    for (u, v) in complete.edges():
        complete.edges[u, v]['weight'] = random.random()
    tree = nx.minimum_spanning_tree(complete)
    # Map the tree's default nodes (0 to N-1) to our actual node labels
    mapping = {i: nodes[i] for i in range(N)}
    tree = nx.relabel_nodes(tree, mapping)
    G.add_edges_from(tree.edges())
    
    # 3. Add the remaining E - (N - 1) edges randomly
    if E > min_edges:
        all_possible_edges = set(itertools.combinations(nodes, 2))
        existing_edges = set((min(u, v), max(u, v)) for u, v in G.edges())
        all_possible_edges = set((min(u, v), max(u, v)) for u, v in all_possible_edges)
        
        available_edges = list(all_possible_edges - existing_edges)
        new_edges = random.sample(available_edges, E - min_edges)
        G.add_edges_from(new_edges)
        
    return G

def generate_hard_graph(size=None, min_size=5, max_size=15, balanced=True):
    """
        Generate a graph with K connected components, generated dynamically.
    """
    if size is None:
        size = random.randint(min_size, max_size)
        
    labels = list(range(size))
    
    # Determine number of components K (between 2 and size // 2)
    max_components = max(2, size // 2)
    K = random.randint(2, max_components)
    
    # Distribute nodes to components (guarantee at least 2 nodes per component)
    random.shuffle(labels)
    components = [labels[i*2:(i+1)*2] for i in range(K)]
    
    remaining_nodes = labels[K*2:]
    for node in remaining_nodes:
        random.choice(components).append(node)
        
    # Build the full graph by generating each component
    G = nx.Graph()
    G.add_nodes_from(labels)
    
    for comp_nodes in components:
        comp_graph = _generate_random_connected_component(comp_nodes)
        G.add_edges_from(comp_graph.edges())
    
    # Sample the query nodes
    if balanced:
        label = random.choice([0, 1])
        if label == 1:
            # Positive example: Sample from the same component
            target_comp = random.choice(components)
            x, y = random.sample(target_comp, 2)
        else:
            # Negative example: Sample from two different components
            comp1, comp2 = random.sample(components, 2)
            x = random.choice(comp1)
            y = random.choice(comp2)
    else:
        # Natural distribution: Completely random sample
        x, y = random.sample(labels, 2)
        label = 0
        for comp in components:
            if x in comp and y in comp:
                label = 1
                break

    if random.choice([True, False]): # swap x and y with 50% probability
        x, y = y, x

    return G, x, y, label


def generate_graph_dataset(num_examples, min_size=5, max_size=15, easy=True):
    dataset = []
    for _ in range(num_examples):
        if easy:
            G, x, y, label = generate_easy_graph(min_size=min_size, max_size=max_size)
        else:
            G, x, y, label = generate_hard_graph(min_size=min_size, max_size=max_size)
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


def prepare_dataset(num_examples, min_size=5, max_size=15, spectral_dims=8, tokenizer_name=None, max_rwse_steps=16, max_rrwp_steps=16, easy=True):
    dataset = generate_graph_dataset(num_examples, min_size=min_size, max_size=max_size, easy=easy)

    # FOR EACH EXAMPLE:
    # 1. compute a random subset of the LETTERS and use them as the "text" fields of the nodes
    # 2. add the prompt node with the "text" field containing the question "Are the nodes X and Y connected? {'Yes' if label == 1 else 'No'}"
    # 3. set the new node's id as the graph-level attribute "prompt_node"
    
    graphs = []
    for G, x, y, label in tqdm(dataset, desc="Preparing dataset"):
        # if the dataset is hard, make the graph directed (preserve connectivity in both directions)
        if not easy:
            G = G.to_directed()

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
    ds.compute_laplacian_coordinates(embedding_dim=spectral_dims)
    ds.compute_shortest_path_distances()
    ds.compute_rwse(max_rwse_steps=max_rwse_steps)
    ds.compute_rrwp(max_rrwp_steps=max_rrwp_steps)

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

def dataset_path_and_size(dataset_size, easy=True):
    size_str, rounded_size, scale = round_size_str(dataset_size)
    dataset_path = f"./src/experiments/expressiveness/{size_str}_{'easy' if easy else 'hard'}_dataset.gtds"
    return dataset_path, rounded_size * scale

def create_and_save_dataset(dataset_size, min_nodes, max_nodes, spectral_dims, model_name, max_rrwp_steps=16, max_rwse_steps=16, easy=True):
    dataset_path, final_dataset_size = dataset_path_and_size(dataset_size, easy=easy)

    dataset = prepare_dataset(
        final_dataset_size, 
        min_size=min_nodes, 
        max_size=max_nodes, 
        spectral_dims=spectral_dims, 
        tokenizer_name=model_name, 
        max_rrwp_steps=max_rrwp_steps, 
        max_rwse_steps=max_rwse_steps,
        easy=easy
    )

    # Save the dataset to disk
    dataset.save(dataset_path)

    return dataset, dataset_path

if __name__ == "__main__":
    MIN_NODES = 10
    MAX_NODES = 20
    SPECTRAL_DIMS = 16
    DATASET_SIZE = 1
    model_name = "meta-llama/Llama-3.2-1B"
    EASY = False
    max_rwse_steps = 8
    max_rrwp_steps = 16

    print(f"Creating dataset with {DATASET_SIZE // 1000}k examples, node sizes between {MIN_NODES} and {MAX_NODES}, spectral dimensions {SPECTRAL_DIMS}, and tokenizer {model_name}...")

    _, dataset_path = create_and_save_dataset(
        dataset_size=DATASET_SIZE, 
        min_nodes=MIN_NODES, 
        max_nodes=MAX_NODES, 
        spectral_dims=SPECTRAL_DIMS,
        model_name=model_name,
        easy=EASY,
        max_rwse_steps=max_rwse_steps,
        max_rrwp_steps=max_rrwp_steps
    )
    print(f"Dataset created and saved at {dataset_path}")

    # laod the dataset to verify it works
    print(f"Loading dataset from {dataset_path} to verify...")
    loaded_ds = TextGraphDataset.load(dataset_path)
    print(f"Dataset loaded successfully with {len(loaded_ds)} examples.")
    
    print("Example graph:")
    for key in loaded_ds[0].keys():
        print('--------------------------------')
        print(f"{key}:\n{loaded_ds[0][key]}")