from .kgqa_gen import generate_dataset
from .prompt import node_texts, GetGraphLabels, label_masker
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

def prepare_graph(graph, question_node="off"):
    graph = create_incidence_graph(graph)

    graphs = []
    for func_name, (q, pointers, a) in graph.graph['questions'].items():
        # create a new copy of the graph for each question-answer pair
        G = graph.copy()
        # Split question/answer across nodes per `question_node` (see prompt.py). With
        # "off" this is exactly the historical single-node layout; with "isolated" the
        # question gets its own edge-free node and the prompt node keeps the "A:" anchor.
        prompt_node_id = f"prompt_{func_name}"
        question_text, prompt_text = node_texts(q, a, question_node)

        if question_text is not None:
            question_node_id = f"question_{func_name}"
            G.add_node(question_node_id, text=question_text)
            G.graph['question_node'] = question_node_id

        G.add_node(prompt_node_id, text=prompt_text)

        # add edges from the prompt node to the pointer nodes (the question node stays
        # edge-free by construction — only the attention mask connects it)
        for pointer in pointers:
            G.add_edge(prompt_node_id, pointer)

        G.graph['prompt_node'] = prompt_node_id
        graphs.append(G)

    return graphs


def prepare_dataset(raw_graphs, question_node="off"):
    graphs = []
    for graph in tqdm(raw_graphs):
        graphs.extend(prepare_graph(graph, question_node=question_node))
    return graphs


def print_text_graph(graph):
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"  {data['type']}: '{data['text']}'")
    print("\nEdges:")
    for u, v, data in graph.edges(data=True):
        print(f"  {graph.nodes[u]['text']} --{data['relation']}--> {graph.nodes[v]['text']}")
    print(f"\nPrompt Node: {graph.graph['prompt_node']}")

# Label preparation now lives in prompt.py (GetGraphLabels / label_masker), shared with
# family_tree_prep.py so the question_node feature is implemented once.

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
    VAL_COUNT = 30
    TEST_COUNT = 150
    TRAIN_COUNT = 500

    MIN_NODES = 30
    MAX_NODES = 50
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

    step_size = 5000
    raw_datasets = {
        f'train_{i*step_size}-{(i+1)*step_size}': train_dataset[i*step_size:(i+1)*step_size] for i in range((len(train_dataset) + step_size - 1) // step_size)
    }
    raw_datasets = {
        **raw_datasets,
        'val': val_dataset,
        'test': test_dataset,
    }
    for split, dataset in raw_datasets.items():
        print(f"{split.upper()} dataset has {len(dataset)} examples.")

    datasets = {}
    save_path = f"./src/experiments/our_tests/graph_datasets/dataset_{MIN_NODES}-{MAX_NODES}_v2"
    for split, dataset in raw_datasets.items():
        print(f"\n{split.upper()} DATASET:")
        datasets[split] = save_text_graph_dataset(dataset, os.path.join(save_path, split), params=params, per_graph_versions=1)

    total_tokens, total_graphs = 0, 0
    for split, dataset in datasets.items():
        print(split)
        for i in range(0, len(dataset), 7):
            graph = dataset[i]
            print(f"  graph {i} has {len(graph['input_ids'])} nodes")
            total_graphs += 1
            for j, input_ids in enumerate(graph['input_ids']):
                num_tokens = len(input_ids)
                total_tokens += num_tokens
        break
    print(f"\nAverage number of tokens: {total_tokens / total_graphs:.2f}")
    # 30-50:  Average number of tokens: ~315 tokens
    # 40-60:  Average number of tokens: ~524 tokens
    # 50-100: Average number of tokens: ~850-900 tokens