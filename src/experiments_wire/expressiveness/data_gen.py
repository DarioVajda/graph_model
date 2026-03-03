import networkx as nx
import random
import torch
from tqdm import tqdm

from ...utils.laplacian import get_laplacian_coordinates

def generate_example(size=None, min_size=5, max_size=15):
    if size is None:
        size = random.randint(min_size, max_size)
    
    # choose node labels as number in range [1, 50]
    node_labels = random.sample(range(1, 51), size)

    # choose random size of the first connected component from range [2, size-2]
    M = random.randint(2, size - 2)

    # create the two connected components nodes sets
    A = node_labels[:M]
    B = node_labels[M:]

    # choose whether or not the label will be 1 or 0
    label = random.choice([0, 1])

    if label == 1:
        # choose 2 random nodes from A or B with equal probability
        if random.random() < 0.5:
            x, y = random.sample(A, 2)
        else:
            x, y = random.sample(B, 2)
    else:
        # choose 1 random node from A and 1 from B
        x = random.choice(A)
        y = random.choice(B)

    # create the graph
    G = nx.Graph()
    G.add_nodes_from(node_labels)
    # add edges to make A a connected component
    for i in range(M - 1):
        for j in range(i + 1, M):
            G.add_edge(A[i], A[j])
    # add edges to make B a connected component
    for i in range(size - M - 1):
        for j in range(i + 1, size - M):
            G.add_edge(B[i], B[j])
    
    return G, x, y, label

def generate_graph_dataset(num_examples, min_size=5, max_size=15):
    dataset = []
    for _ in range(num_examples):
        G, x, y, label = generate_example(min_size=min_size, max_size=max_size)
        dataset.append((G, x, y, label))
    return dataset

def get_disconnected_spectral_coordinates(G):
    """
    Assume the graph has two fully connected components, each node in the first gets [1, 0] and each node in the second gets [0, 1].
    """
    components = list(nx.connected_components(G))
    spectral_coords = {}
    for component in components:
        for node in component:
            spectral_coords[node] = [1.0, 0.0] if component == components[0] else [0.0, 1.0]
    return spectral_coords


def tokenize_example(G, node_label_to_token_id, tokenized_prompt, graph_model=False, spectral_dims=8, max_nodes=50, spectral_type="laplacian"):
    """
    Tokenize the graph node labels and prepare the input sequence.
    If graph_model is True, also prepare node ids and spectral features.
    """
    graph_node_list = list(G.nodes)
    random.shuffle(graph_node_list)

    num_nodes = len(graph_node_list)
    
    node_sequence = torch.tensor([ node_label_to_token_id[str(node)] for node in graph_node_list ], dtype=torch.long)
    input_sequence = torch.cat([ node_sequence, tokenized_prompt ], dim=0)
    # print("node_label_to_token_id: ", node_label_to_token_id)
    # print("node sequence: ", node_sequence)
    # print("input sequence: ", input_sequence)
    # separated_input_sequence = torch.cat([ torch.tensor([input_sequence[i].item(), 62], dtype=torch.long) for i in range(0, len(input_sequence)) ], dim=0)
    # print("separated input sequence: ", separated_input_sequence)
    # print("separated input sequence decoded: ", tokenizer.decode(separated_input_sequence)) 
    # exit()

    # target sequence is a vector or -100s, except for the token predicting the "Yes"/"No" answer
    target_sequence = torch.full_like(input_sequence, fill_value=-100)
    target_sequence[-1] = input_sequence[-1]  # the last token is the answer token and we are predicting it from the second last token

    if not graph_model:
        return { "input_ids": input_sequence, "labels": target_sequence }

    # Prepare node ids mapping
    node_ids = torch.full_like(input_sequence, fill_value=num_nodes) # default node id for prompt tokens
    for i in range(num_nodes):
        node_ids[i] = i  # assign node ids for graph tokens (0 to num_nodes-1)

    # Compute spectral coordinates
    G_with_prompt = G.copy()
    G_with_prompt.add_node('prompt')  # add node for the prompt
    if spectral_type == "laplacian":
        spectral_coords = get_laplacian_coordinates(G_with_prompt, m=spectral_dims)
    elif spectral_type == "disconnected":
        spectral_coords = get_disconnected_spectral_coordinates(G)
        spectral_coords['prompt'] = [0.0, 0.0]

    spectral_features = torch.zeros((max_nodes + 1, spectral_dims), dtype=torch.float)  # +1 for the prompt node
    for i, node in enumerate(graph_node_list):
        spectral_features[i] = torch.tensor(spectral_coords[node], dtype=torch.float)           # graph node spectral features
    spectral_features[num_nodes] = torch.tensor(spectral_coords['prompt'], dtype=torch.float)  # prompt node spectral features

    # create attention mask, where graph tokens can attend to each other bidirectionally, and the prompt tokens can attend to all graph tokens but have a causal attention mask among themselves
    # the attention mask should be 0 for allowed attention and -inf for disallowed attention (after applying the mask to the attention scores, the disallowed attention will be effectively zeroed out after the softmax)
    attention_mask = torch.full((len(input_sequence), len(input_sequence)), fill_value=float('-inf'), dtype=torch.float)
    # allow full attention among graph tokens
    attention_mask[:num_nodes, :num_nodes] = 0.0
    # allow prompt tokens to attend to all graph tokens
    attention_mask[num_nodes:, :num_nodes] = 0.0
    # allow causal attention among prompt tokens
    for i in range(num_nodes, len(input_sequence)):
        attention_mask[i, num_nodes:i+1] = 0.0

    return {
        "input_ids": input_sequence,
        "labels": target_sequence,
        "node_ids": node_ids,
        "node_spectral_features": spectral_features,
        "attention_mask": attention_mask
    }


def prepare_prompt(x, y, label):
    prompt = f"Are nodes {x} and {y} in the same connected component of the graph (Yes/No)? Answer: {'Yes' if label == 1 else 'No'}"
    return prompt

def prepare_dataset(dataset, tokenizer, save_path=None, graph_model=False, max_nodes=15, min_nodes=5, spectral_dims=8, spectral_type="laplacian"):
    """
    Prepares the dataset for training/testing by converting each graph into the required format, depending on the model type.
    If save_path is provided, saves the dataset to the specified path.
    """

    # Step 0: Get key-value pairs of possible node labels and their token ids
    node_label_to_token_id = {}
    for i in range(1, 51):
        token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
        node_label_to_token_id[str(i)] = token_id

    # Step 1: Tokenize each node label in the graph with the tokenizer
    tokenized_dataset = []
    for G, x, y, label in tqdm(dataset, desc="Preparing dataset"):
        tokenized_prompt = tokenizer(prepare_prompt(x, y, label), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        tokenized_example = tokenize_example(G, node_label_to_token_id, tokenized_prompt, graph_model=graph_model, spectral_dims=spectral_dims, max_nodes=max_nodes, spectral_type=spectral_type)
        tokenized_dataset.append({
            "graph": G, 
            "x": x,
            "y": y,
            "label": label,
            **tokenized_example
        })
    
    if save_path is not None:
        torch.save({
            "hyperparameters": {
                "min_nodes": min_nodes,
                "max_nodes": max_nodes,
                "spectral_dims": spectral_dims,
                "dataset_size": len(dataset),
                "graph_model": graph_model
            },
            "dataset": tokenized_dataset
        }, save_path)
        print(f"Dataset saved to {save_path}")

    return tokenized_dataset

if __name__ == "__main__":
    # random.seed(42)
    MIN_NODES = 10
    MAX_NODES = 20
    SPECTRAL_DIMS = 8
    DATASET_SIZE = 2500
    SPECTRAL_TYPE = "disconnected"
    if SPECTRAL_TYPE == "disconnected":
        SPECTRAL_DIMS = 2

    dataset = generate_graph_dataset(DATASET_SIZE, min_size=MIN_NODES, max_size=MAX_NODES)
    print(f"Generated {len(dataset)} examples.")

    model_name = "meta-llama/Llama-3.2-1B"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("pad token id: ", tokenizer.pad_token_id)
    print("eos token id: ", tokenizer.eos_token_id)
    print()

    # # tempory exit to avoid overwriting existing dataset
    # exit()

    processed_dataset = prepare_dataset(
        dataset, 
        tokenizer, 
        graph_model=True, 
        max_nodes=MAX_NODES, 
        spectral_dims=SPECTRAL_DIMS, 
        save_path="src/experiments/expressiveness/processed_easy_dataset.pt",
        spectral_type=SPECTRAL_TYPE
    )
    print(f"Processed dataset with {len(processed_dataset)} examples.")

    # Example: print the 114th example
    example = processed_dataset[114]
    print("Example 114:")
    print(" Graph nodes:", example["graph"].nodes)
    print(" Graph edges:", example["graph"].edges)
    print(" Graph components:", list(nx.connected_components(example["graph"])))
    print(" x:", example["x"])
    print(" y:", example["y"])
    print(" label:", example["label"])
    print(" input_ids:", example["input_ids"])
    print(" decoded input_ids:", tokenizer.decode(torch.cat([ torch.tensor([example["input_ids"][i].item(), 62], dtype=torch.long) for i in range(0, len(example["input_ids"])) ], dim=0)))
    print(" labels:", example["labels"])
    print(" node_ids:", example["node_ids"])
    print(" node_spectral_features:", example["node_spectral_features"])
    