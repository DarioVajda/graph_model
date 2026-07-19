"""
Turning raw GraphQA json into text-attributed graphs.

Each raw example carries a question ("The edges in G are: (0, 1), ...") and an answer;
this module parses the edge list back out of that text, rebuilds the graph (optionally
as its bipartite Levi/incidence form), attaches the question+answer as a prompt node
wired to the nodes the question mentions, and computes the graph features the model's
biases consume. `data.py` drives it; `RunConfig` supplies the knobs.
"""

import os
import json
import networkx as nx
import re
from tqdm import tqdm

from ...utils import TextGraphDataset
from .config import RAW_DIR

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


def example_to_graph(example, graph_type="standard", problem_type=None,
                     question_node="off"):
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

    # Attach the prompt node (and, when question_node != "off", a QUESTION node).
    #
    # "off": one node carries "{question}{answer}" — the historical layout, byte-
    #   identical to pre-feature builds.
    # "isolated": the question body moves into its own QUESTION prefix node (no
    #   edges — the bidirectional prefix mask alone makes graph tokens question-
    #   aware, mirroring the kgqa experiment's best arm). The prompt node keeps
    #   the "A:" anchor onward, so the supervised span / generation anchor are
    #   byte-identical to "off"; `question` ends in "...\nA: ", so splitting on
    #   ANSWER_PREFIX cleanly separates the two.
    if question_node == "isolated":
        head, sep, tail = question.partition(ANSWER_PREFIX)
        if not sep:
            raise ValueError(
                f"question_node='isolated' needs the {ANSWER_PREFIX!r} anchor in the "
                f"task_description to split on, but none was found: {question!r}")
        q_node = num_nodes
        graph.add_node(q_node, text=head)
        graph.graph['question_node'] = q_node
        prompt_node = num_nodes + 1
        graph.add_node(prompt_node, text=f"{sep}{tail}{answer}")
    else:
        prompt_node = num_nodes
        graph.add_node(prompt_node, text=f"{question}{answer}")
    graph.graph['prompt_node'] = prompt_node

    # Prompt edges wire to the prompt node only; an "isolated" QUESTION node stays
    # edge-free by construction.
    prompt_connections = extract_prompt_edges(example, nodes, edges if graph_type=="incidence" else None, problem_type)
    for target_node in prompt_connections:
        graph.add_edge(prompt_node, target_node)

    return graph


# Every `task_description` ends with "...\nA: " and the answer follows it, so the
# supervised span starts just past this marker (see GetGraphLabels). Derived from the
# tokenizer rather than hardcoded as token ids: the ids are model-specific, and a
# silently wrong literal would mask the wrong span rather than fail.
ANSWER_PREFIX = "A:"


def raw_split_file(task, split):
    """Path to a raw GraphQA json ('train' | 'validation' | 'test')."""
    return os.path.join(RAW_DIR, task, f"{task}_zero_shot_{split}.json")


def has_raw_split(task, split):
    return os.path.exists(raw_split_file(task, split))


def build_split(cfg, split, tokenizer):
    """Build one split into a fully-featured TextGraphDataset (not saved).

    Computes exactly the features the collator can consume: shortest-path distances,
    RRWP and the magnetic Laplacian. All three are always computed regardless of which
    bias arm will *use* them, so every arm shares one built dataset — the bias flags
    are deliberately not part of the cache identity (see RunConfig.dataset_dir).
    """
    examples = load_json_dataset(raw_split_file(cfg.task, split))
    graphs = [
        example_to_graph(example, graph_type=cfg.graph_type, problem_type=cfg.task,
                         question_node=cfg.question_node)
        for example in tqdm(examples, desc=f"{cfg.graph_type}/{cfg.task}/{split}: building graphs")
    ]
    ds = TextGraphDataset(graphs, dataset_label=f"{cfg.graph_type}/{cfg.task}")

    ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=True)
    question_end = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    ds.compute_labels(GetGraphLabels(question_end=question_end))

    ds.compute_shortest_path_distances(use_gpu=cfg.use_gpu)
    ds.compute_rrwp(cfg.max_rw_steps, use_gpu=cfg.use_gpu)
    ds.compute_magnetic_lap(q=cfg.magnetic_q, use_gpu=cfg.use_gpu, m=cfg.magnetic_m)
    return ds
