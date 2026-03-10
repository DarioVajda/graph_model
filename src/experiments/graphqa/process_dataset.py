import os
import json
import networkx as nx
import re
from tqdm import tqdm

from ...utils import TextGraphDataset

def load_json_dataset(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def extract_graph_data(text):
    nodes = []
    edges = []
    
    # 1. Extract the nodes
    # Looks for the text after "nodes " up to the next period.
    nodes_match = re.search(r'nodes\s+(.*?)\.', text)
    if nodes_match:
        nodes_str = nodes_match.group(1)
        # Find all individual digits/numbers in that specific substring
        nodes = [int(n) for n in re.findall(r'\d+', nodes_str)]
        
    # 2. Extract the edges
    # Looks for the text after "The edges in G are: " up to the next period.
    edges_match = re.search(r'The edges in G are:\s+(.*?)\.', text)
    if edges_match:
        edges_str = edges_match.group(1)
        # Find all pairs of numbers wrapped in parentheses
        edges_raw = re.findall(r'\((\d+),\s*(\d+)\)', edges_str)
        edges = [(int(u), int(v)) for u, v in edges_raw]
        
    return nodes, edges

def extract_node_preferences(example):
    """
    Extracts known node preferences from the question text 
    and returns them as a dictionary mapping node IDs to their preference.
    """
    text = example.get("question", "")
    matches = re.findall(r"Node (\d+) likes (\w+)\.", text)
    
    preferences = {}
    for node_str, preference in matches:
        preferences[int(node_str)] = preference
        
    return preferences

#region Extracting prompt connections
def extract_prompt_edges_connected_nodes(example):
    text = example.get("task_description", "")
    match = re.search(r"connected to (\d+)", text)
    if match:
        return [int(match.group(1))]
    raise ValueError(f"Could not extract prompt edge for connected_nodes problem. Task description: {text}")

def extract_prompt_edges_disconnected_nodes(example):
    text = example.get("task_description", "")
    match = re.search(r"not connected to (\d+)", text)
    if match:
        return [int(match.group(1))]
    raise ValueError(f"Could not extract prompt edge for disconnected_nodes problem. Task description: {text}")

def extract_prompt_edges_edge_existence(example):
    text = example.get("task_description", "")
    match = re.search(r"Is node (\d+) connected to node (\d+)", text)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    raise ValueError(f"Could not extract prompt edge for edge_existence problem. Task description: {text}")

def extract_prompt_edges_node_classification(example):
    text = example.get("task_description", "")    
    match = re.search(r"Does node (\d+) like", text)
    if match:
        return [int(match.group(1))]
    raise ValueError(f"Could not extract prompt edge for node_classification problem. Task description: {text}")

def extract_prompt_edges_node_degree(example):
    text = example.get("task_description", "")
    match = re.search(r"degree of node (\d+)", text)
    if match:
        return [int(match.group(1))]
    raise ValueError(f"Could not extract prompt edge for node_degree problem. Task description: {text}")

def extract_prompt_edges_reachability(example):
    text = example.get("task_description", "")
    match = re.search(r"path from node (\d+) to node (\d+)", text)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    raise ValueError(f"Could not extract prompt edge for reachability problem. Task description: {text}")

def extract_prompt_edges_shortest_path(example):
    text = example.get("task_description", "")
    match = re.search(r"shortest path from node (\d+) to node (\d+)", text)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    raise ValueError(f"Could not extract prompt edge for shortest_path problem. Task description: {text}")

def extract_prompt_edges(example, nodes, edges, problem_type):
    """
    Extracts a list of nodes that the prompt node shuuld be connected to with a directed edge.
    This is based on the problem type and the question content.
    If there are no clear connections (so we are dealing with a graph-level problem), we return all nodes and/or edges
    """
    if problem_type == "connected_nodes":
        return extract_prompt_edges_connected_nodes(example)
    elif problem_type == "cycle_check":
        return nodes
    elif problem_type == "disconnected_nodes":
        return extract_prompt_edges_disconnected_nodes(example)
    elif problem_type == "edge_count":
        return edges if edges is not None else nodes
    elif problem_type == "edge_existence":
        return extract_prompt_edges_edge_existence(example)
    elif problem_type == "maximum_flow":
        raise ValueError("Cannot extract prompt edges for maximum_flow problem because the edge capacities are missing from the dataset.")
    elif problem_type == "node_classification":
        return extract_prompt_edges_node_classification(example)
    elif problem_type == "node_count":
        return nodes
    elif problem_type == "node_degree":
        return extract_prompt_edges_node_degree(example)
    elif problem_type == "reachability":
        return extract_prompt_edges_reachability(example)
    elif problem_type == "shortest_path":
        return extract_prompt_edges_shortest_path(example)
    elif problem_type == "triangle_counting":
        return nodes + edges if edges is not None else nodes
    else:
        raise NotImplementedError(f"Prompt edge extraction not implemented for problem type: {problem_type}")
#endregion

def create_incidence_graph(G):
    """
    Transforms a standard graph G into its bipartite incidence (Levi) graph.
    """
    I = nx.Graph()
    
    # 1. Add original vertices (V) as nodes in the new graph
    # We assign a bipartite=0 attribute to easily identify them later
    I.add_nodes_from(G.nodes(data=True))
    
    # 2. Iterate through original edges (E) to create the new nodes and connections
    for u, v, edge_data in G.edges(data=True):
        # Represent the edge as a tuple to act as its unique node ID
        edge_node = (u, v) 
        
        # Add the edge as a new node (bipartite set 1), bringing along its attributes
        I.add_node(edge_node, **edge_data)
        
        # Connect the original vertices to this new edge-node
        I.add_edges_from([(u, edge_node), (v, edge_node)])
        
    return I

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
                if question_end_index is not None:
                    raise ValueError(f"Multiple occurrences of question end token sequence {self.question_end} found in the prompt node's input_ids: {prompt_input_ids}")
                question_end_index = i + len(self.question_end) - 1
        if question_end_index is None:
            raise ValueError(f"Could not find question end token sequence {self.question_end} in the prompt node's input_ids: {prompt_input_ids}")

        # Mask all tokens before and including the question end index to -100
        for i in range(question_end_index + 1):
            labels[i] = -100
        return labels


def example_to_graph(example, graph_type="standard", problem_type=None):
    raw_question = example['question']
    num_nodes = int(example['nnodes'])
    num_edges = int(example['nedges'])
    question = example['task_description']
    answer = example['answer']

    nodes, edges = extract_graph_data(raw_question)

    if len(nodes) != num_nodes:
        raise ValueError(f"Number of extracted nodes ({len(nodes)}) does not match expected nnodes ({num_nodes}).")
    if len(edges) != num_edges:
        raise ValueError(f"Number of extracted edges ({len(edges)}) does not match expected nedges ({num_edges}).")

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    if graph_type == "incidence":
        graph = create_incidence_graph(graph)

    # add text attributes to each node
    for node in graph.nodes():
        if type(node) == int:  # Original vertex nodes
            graph.nodes[node]['text'] = f"{node}"
        else:  # Edge nodes in the incidence graph
            graph.nodes[node]['text'] = f"{node[0]},{node[1]}"

    if problem_type == "node_classification":
        preferences = extract_node_preferences(example)
        for node_id, preference in preferences.items():
            if node_id in graph.nodes:
                graph.nodes[node_id]['text'] = f"{node_id} likes {preference}"

    graph = graph.to_directed()

    # create a new node for the question and the answer
    prompt_node = num_nodes
    graph.graph['prompt_node'] = prompt_node
    graph.add_node(prompt_node, text=f"{question}{answer}")

    prompt_connections = extract_prompt_edges(example, nodes, edges if graph_type=="incidence" else None, problem_type)
    for target_node in prompt_connections:
        graph.add_edge(prompt_node, target_node)

    return graph


def process_dataset(dataset_dir, output_dir, graph_type="standard", problem_type=None, tokenizer=None, spectral_params=None):
    """
    This function saves the processed dataset.

    Arguments:
        dataset_dir: Directory containing the input JSON files for training and testing.
        output_dir: Directory where the processed graph datasets will be saved (if needed).
        graph_type: Type of graph to create ("standard" or "incidence")
        problem_type: Type of problem to solve
        tokenizer: Tokenizer for processing text data
        spectral_params: Parameters for computing spectral features
    """
    print('=' * 100)
    print(f"Processing dataset for problem type: {problem_type}, graph type: {graph_type}")
    print('=' * 100)

    train_data = load_json_dataset(os.path.join(dataset_dir, problem_type, f"{problem_type}_zero_shot_train.json"))
    test_data = load_json_dataset(os.path.join(dataset_dir, problem_type, f"{problem_type}_zero_shot_test.json"))

    train_graphs = []
    for example in tqdm(train_data, desc=f"Processing training examples (graph_type={graph_type}, problem={problem_type})"): 
        train_graphs.append(example_to_graph(example, graph_type=graph_type, problem_type=problem_type))    
    test_graphs = []
    for example in tqdm(test_data, desc=f"Processing testing examples (graph_type={graph_type}, problem={problem_type})"): 
        test_graphs.append(example_to_graph(example, graph_type=graph_type, problem_type=problem_type))

    # Initialize TextGraphDataset instances for training and testing datasets
    train_ds = TextGraphDataset(train_graphs)
    test_ds = TextGraphDataset(test_graphs)

    # This represents --> "A:"
    question_end = [ 32, 25 ] 
    
    def dataset_post_processing(ds, question_end, params):
        # Tokenize all text data in each graph in the dataset 
        ds.tokenize(tokenizer, max_length=1024, add_eos=True)

        # Compute the labels for each graph in the dataset
        get_graph_labels = GetGraphLabels(question_end=question_end)
        ds.compute_labels(get_graph_labels)

        # Compute other spectral features for each graph in the dataset
        ds.compute_laplacian_coordinates(params.get("laplacian_embedding_dim", 16))
        ds.compute_shortest_path_distances()
        ds.compute_rwse(params.get("max_rwse_steps", 16))
        ds.compute_rrwp(params.get("max_rrwp_steps", 16))
        ds.compute_magnetic_lap(params.get("magnetic_laplacian_q", 0.25))

    dataset_post_processing(train_ds, question_end, spectral_params)
    dataset_post_processing(test_ds, question_end, spectral_params)

    # create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, graph_type), exist_ok=True)
        os.makedirs(os.path.join(output_dir, graph_type, problem_type), exist_ok=True)

        train_ds.save(os.path.join(output_dir, graph_type, problem_type, "train"))
        test_ds.save(os.path.join(output_dir, graph_type, problem_type, "test"))

    print(f"Finished processing dataset for problem type: {problem_type}, graph type: {graph_type}, saved to {output_dir}/{graph_type}/{problem_type}.\n\n")


if __name__ == "__main__":
    base_llama_model = "meta-llama/Llama-3.2-1B"

    dataset_dir = "./src/experiments/graphqa/hf_dataset"
    output_dir = "./src/experiments/graphqa/processed_datasets"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_llama_model)

    spectral_params = {
        "laplacian_embedding_dim": 16,
        "max_rwse_steps": 16,
        "max_rrwp_steps": 16,
        "magnetic_laplacian_q": 0.25,
    }
    
    # Problem type options:
    problem_types = [ "connected_nodes", "disconnected_nodes", "cycle_check", "edge_count", "edge_existence", "node_classification", "node_count", "node_degree", "reachability", "shortest_path", "triangle_counting" ] # (maximum_flow is excluded because the dataset does not contain edge capacities)


    # Process the dataset and generate both "standard" and "incidence" graph versions for each problem type
    for problem_type in problem_types:
        for graph_type in ["standard", "incidence"]:
            process_dataset(dataset_dir, output_dir, graph_type=graph_type, problem_type=problem_type, tokenizer=tokenizer, spectral_params=spectral_params)




