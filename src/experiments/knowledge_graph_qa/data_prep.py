from .data_gen import generate_dataset
from ...utils.text_graph_dataset import TextGraphDataset

import os
import networkx as nx
import random
from tqdm import tqdm

def create_incidence_graph(graph):
    V = graph.nodes()
    E = graph.edges()

    incidence_graph = nx.DiGraph()
    for v in V:
        incidence_graph.add_node(v, text=v)
    for u, v, data in graph.edges(data=True):
        relation = graph.edges[u, v]['relation']
        text = relation.replace('_', ' ').lower()
        edge_key = f"{u}_{relation}_{v}"
        incidence_graph.add_node(edge_key, text=text)

        # add edge from u to edge_key and from edge_key to v
        incidence_graph.add_edge(u, edge_key)
        incidence_graph.add_edge(edge_key, v)

    # copy graph-level attributes
    for key, value in graph.graph.items():
        incidence_graph.graph[key] = value

    return incidence_graph

def prepare_graph(graph):
    graph = create_incidence_graph(graph)

    graphs = []
    for func_name, (q, pointers, a) in graph.graph['questions'].items():
        # create a new copy of the graph for each question-answer pair
        G = graph.copy()
        # add the question + answer as the "prompt" node to the graph
        prompt_node_id = f"prompt_{func_name}"
        G.add_node(prompt_node_id, text=f"Q: {q}\nA: {a}")

        # add edges from the prompt node to the pointer nodes
        for pointer in pointers:
            G.add_edge(prompt_node_id, pointer)

        G.graph['prompt_node'] = prompt_node_id
        graphs.append(G)
    
    return graphs


def prepare_dataset(raw_graphs):
    graphs = []
    for graph in tqdm(raw_graphs):
        graphs.extend(prepare_graph(graph))
    return graphs


def print_text_graph(graph):
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"  {data['type']}: '{data['text']}'")
    print("\nEdges:")
    for u, v, data in graph.edges(data=True):
        print(f"  {graph.nodes[u]['text']} --{data['relation']}--> {graph.nodes[v]['text']}")
    print(f"\nPrompt Node: {graph.graph['prompt_node']}")

#region Label Preparation
class GetGraphLabels:
    """
    This is a callable class responsible for finding the question end in the prompt node and masking all tokens to -100 except for the answer (which follows the question end).
    """
    def __init__(self, question_end, tokenizer):
        self.tokenizer = tokenizer
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
            for token_id in prompt_input_ids:
                print(f"{token_id}: {self.tokenizer.decode([token_id]).replace(' ', '_')}")
            raise ValueError(f"Could not find question end token sequence {self.question_end} in the prompt node's input_ids: {prompt_input_ids}")

        # Mask all tokens before and including the question end index to -100
        for i in range(question_end_index + 1):
            labels[i] = -100
        return labels
#endregion

def save_text_graph_dataset(graphs, path, params=None, per_graph_versions=1):
    # check if a dataset already exists at the path, and if so, load and return it instead of creating a new one
    # if os.path.exists(path+'.gtds'):
    #     print(f"Loading existing dataset from {path}...")
    #     dataset = TextGraphDataset.load(path)
        # return dataset

    dataset = TextGraphDataset(graphs, per_graph_versions=per_graph_versions)
    dataset.tokenize(params['tokenizer'], max_length=params['max_length'], add_eos=False)
    dataset.compute_labels(params['get_graph_labels'], num_proc=12)
    dataset.save(path)
    print('='*50)

    dataset.compute_shortest_path_distances()
    dataset.save(path)
    print('='*50)

    dataset.compute_rrwp(max_rrwp_steps=params['max_rrwp_steps'])
    dataset.save(path)
    print('='*50)

    dataset.compute_magnetic_lap(q=params['magnetic_q'])
    dataset.save(path)
    print('='*50)
    return dataset


if __name__ == "__main__":
    VAL_COUNT = 150
    TEST_COUNT = 250
    TRAIN_COUNT = 1200

    MIN_NODES = 50
    MAX_NODES = 100
    raw_train_graphs, raw_val_graphs, raw_test_graphs = generate_dataset(train_count=TRAIN_COUNT, val_count=VAL_COUNT, test_count=TEST_COUNT, min_nodes=MIN_NODES, max_nodes=MAX_NODES)
    print(f"Generated {len(raw_train_graphs)} training graphs, {len(raw_val_graphs)} validation graphs, and {len(raw_test_graphs)} test graphs with node counts between {MIN_NODES} and {MAX_NODES}.")

    val_dataset = prepare_dataset(raw_val_graphs)
    test_dataset = prepare_dataset(raw_test_graphs)
    train_dataset = prepare_dataset(raw_train_graphs)

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
        'get_graph_labels': GetGraphLabels(question_end=[ 32, 25 ], tokenizer=tokenizer), # this represents "A:"
    }

    step_size = 2000
    raw_datasets = {
        f'train_{i*step_size}-{(i+1)*step_size}': train_dataset[i*step_size:(i+1)*step_size] for i in range((len(train_dataset) + step_size - 1) // step_size)
    }
    raw_datasets = {
        **raw_datasets,
        'val': val_dataset,
        'test': test_dataset,
    }

    datasets = {}
    save_path = f"./src/experiments/knowledge_graph_qa/graph_datasets/dataset_{MIN_NODES}-{MAX_NODES}"
    for split, dataset in raw_datasets.items():
        if 'train' not in split: continue
        print(f"\n{split.upper()} DATASET:")
        datasets[split] = save_text_graph_dataset(dataset, os.path.join(save_path, split), params=params, per_graph_versions=1)

    total_tokens, total_graphs = 0, 0
    for split, dataset in datasets.items():
        print(split)
        for i in range(0, len(dataset), 12):
            graph = dataset[i]
            print(f"  graph {i} has {len(graph['input_ids'])} nodes")
            total_graphs += 1
            for j, input_ids in enumerate(graph['input_ids']):
                num_tokens = len(input_ids)
                total_tokens += num_tokens
        break
    print(f"\nAverage number of tokens in prompt node: {total_tokens / total_graphs:.2f}")