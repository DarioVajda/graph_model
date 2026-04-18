from .family_tree_gen import generate_dataset

import os
import networkx as nx
import random
import json
from tqdm import tqdm

import torch
from torch_geometric.data import Data
print("Imported all modules")

# ------------------------------------------------------------------------------
#region Graph Dataset Preparation (for GraphLLM)
# ------------------------------------------------------------------------------
def create_incidence_graph(graph):
    V = graph.nodes()
    E = graph.edges()

    incidence_graph = nx.DiGraph()
    for v in V:
        incidence_graph.add_node(v, text=graph.nodes[v]['text'])

    for u, v, data in graph.edges(data=True):
        relation = graph.edges[u, v]['relation']
        edge_key = f"{u}_{relation}_{v}"

        # handle spouse differently since it's a symmetric relation
        if relation == "SPOUSE":
            if u > v:
                continue
            else:
                text = "spouse"
                incidence_graph.add_node(edge_key, text=text)
                # add edges from u and v to edge_key and from edge_key to u and v
                incidence_graph.add_edge(u, edge_key)
                incidence_graph.add_edge(v, edge_key)
                incidence_graph.add_edge(edge_key, u)
                incidence_graph.add_edge(edge_key, v)
        elif relation == "CHILD":
            text = "child"
            incidence_graph.add_node(edge_key, text=text)
            # add edge from u to edge_key and from edge_key to v
            incidence_graph.add_edge(u, edge_key)
            incidence_graph.add_edge(edge_key, v)
        else:
            raise ValueError(f"Unknown relation: {relation}")

    # copy graph-level attributes
    for key, value in graph.graph.items():
        incidence_graph.graph[key] = value

    return incidence_graph

def prepare_graph(graph, person_id, question, answer):
    # add texts to nodes
    for node, data in graph.nodes(data=True):
        graph.nodes[node]['text'] = f"{data['first_name']} {data['last_name']}; {'male' if data['gender'] == 'M' else 'female'}, {data['birth_year']}, {data['fav_color']}, {data['fav_food']}, {data['fav_city']}"

    incidence_graph = create_incidence_graph(graph)

    # add the question + answer as the "prompt" node to the graph
    prompt_node_id = f"prompt_{person_id}"
    incidence_graph.add_node(prompt_node_id, text=f"Q: {question}\nA: {answer}")

    # add edge from the prompt node to the person node about whom the question is being asked
    incidence_graph.add_edge(prompt_node_id, person_id)

    incidence_graph.graph['prompt_node'] = prompt_node_id

    return incidence_graph

def prepare_graph_dataset(raw_dataset):
    prepared_dataset = {}

    for split, examples in raw_dataset.items():
        prepared_dataset[split] = []
        for example in examples:
            graph = example["graph"]
            person_id = example["person_id"]
            question = example["question"]
            answer = example["answer"]

            prepared_example = prepare_graph(graph, person_id, question, answer)
            prepared_dataset[split].append(prepared_example)

    return prepared_dataset

def save_graph_dataset(graph_datasets, output_dir, params):
    from ...utils.text_graph_dataset import TextGraphDataset
    os.makedirs(output_dir, exist_ok=True)

    for split, graphs in graph_datasets.items():
        print(f"Processing {split} split with {len(graphs)} graphs...")
        dataset = TextGraphDataset(graphs, per_graph_versions=1)
        path = os.path.join(output_dir, split)

        dataset.tokenize(params['tokenizer'], max_length=params['max_length'], add_eos=False)
        dataset.compute_labels(params['get_graph_labels'], num_proc=12)
        dataset.compute_shortest_path_distances()
        dataset.compute_rrwp(max_rrwp_steps=params['max_rrwp_steps'])
        dataset.compute_magnetic_lap(q=params['magnetic_q'])

        dataset.save(path)
        print(f"Saved {split} dataset to {path}")
        print('='*50)

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

#endregion
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#region Text Dataset Preparation (for standard LLM baseline)
# ------------------------------------------------------------------------------
def prepare_text(graph, person_id, question, answer, tokenizer, get_graph_labels):
    text = "This graph represents a family tree. Each node corresponds to a person and contains the following information about them in this format: (full name; gender, birth year, favorite color, favorite food, favorite city). The edges represent relationships between people and can be of two types: 'SPOUSE' or 'CHILD'.\n\n"
    text += "People (nodes):\n"
    for node, data in graph.nodes(data=True):
        text += f"({data['first_name']} {data['last_name']}; {'male' if data['gender'] == 'M' else 'female'}, {data['birth_year']}, {data['fav_color']}, {data['fav_food']}, {data['fav_city']})\n"
    text += "\nRelationships (edges):\n"
    for u, v, data in graph.edges(data=True):
        relation = graph.edges[u, v]['relation']
        text += f"({graph.nodes[u]['first_name']} {graph.nodes[u]['last_name']}) -[{relation}]-> ({graph.nodes[v]['first_name']} {graph.nodes[v]['last_name']})\n"

    text += f"Q: {question}\nA: {answer}"

    # tokenize the text
    input_ids = tokenizer.encode(text, truncation=True, max_length=32_768)

    # get the labels for the answer part
    labels = get_graph_labels({"input_ids": [input_ids], "prompt_node": 0})
    return {
        'text': text,
        'input_ids': input_ids,
        'labels': labels,
    }

def prepare_text_dataset(raw_dataset, tokenizer, get_graph_labels):
    prepared_dataset = {
        'train': [],
        'val': [],
        'test': []
    }

    for split, examples in raw_dataset.items():
        for example in examples:
            graph = example["graph"]
            person_id = example["person_id"]
            question = example["question"]
            answer = example["answer"]

            prepared_example = prepare_text(graph, person_id, question, answer, tokenizer, get_graph_labels)
            prepared_dataset[split].append(prepared_example)

    return prepared_dataset

def save_text_dataset(text_dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for split, examples in text_dataset.items():
        path = os.path.join(output_dir, f"{split}.jsonl")
        with open(path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
#endregion
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# region LLaGA Format Dataset Preparation (for RGLM)
# ------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

def prepare_llaga_dataset(raw_dataset, encoder_model_name='all-MiniLM-L6-v2', device='cuda'):
    """
    Converts raw graphs to LLaGA format and pre-computes node embeddings.
    """
    print(f"Loading SentenceTransformer '{encoder_model_name}' on {device}...")
    encoder = SentenceTransformer(encoder_model_name, device=device)
    
    prepared_dataset = { 'train': [], 'val': [], 'test': [] }
    global_graph_id = 0
    
    for split, examples in raw_dataset.items():
        print(f"Embedding nodes for {split} split...")
        for example in tqdm(examples, desc=f"Preparing LLaGA {split} split", total=len(examples)):
            nx_graph = example["graph"]
            question = example["question"]
            answer = example["answer"]
            
            # 1. Create the ShareGPT-style Conversation
            conversation = [
                {
                    "from": "human",
                    "value": f"<graph>\nQuestion: {question}"
                },
                {
                    "from": "gpt",
                    "value": str(answer)
                }
            ]
            
            llaga_json = {
                "id": f"family_tree_{split}_{global_graph_id}",
                "graph_id": f"{split}_{global_graph_id}",
                "conversations": conversation
            }
            
            # 2. Convert NetworkX graph to PyTorch Geometric format
            node_mapping = {n: i for i, n in enumerate(nx_graph.nodes())}
            
            edge_list = []
            for u, v in nx_graph.edges():
                edge_list.append([node_mapping[u], node_mapping[v]])
                
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            # 3. Extract and Embed Node Text Features
            node_texts = []
            for n in nx_graph.nodes():
                data = nx_graph.nodes[n]
                text_desc = f"{data['first_name']} {data['last_name']}; {'male' if data['gender'] == 'M' else 'female'}, {data['birth_year']}, {data['fav_color']}, {data['fav_food']}, {data['fav_city']}"
                node_texts.append(text_desc)
            
            # Compute embeddings on GPU in one batch per graph
            with torch.no_grad():
                x = encoder.encode(node_texts, convert_to_tensor=True, device=device)
                
            pyg_data = Data(x=x.cpu(), edge_index=edge_index)
            
            prepared_dataset[split].append({
                "json_data": llaga_json,
                "pyg_data": pyg_data,
                "graph_id": llaga_json["graph_id"]
            })
            
            global_graph_id += 1
            
    return prepared_dataset

def save_llaga_dataset(llaga_datasets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for split, examples in llaga_datasets.items():
        print(f"Saving LLaGA {split} split...")
        
        json_data_list = [ex["json_data"] for ex in examples]
        json_path = os.path.join(output_dir, f"{split}_instructions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data_list, f, indent=4)
            
        graphs_dir = os.path.join(output_dir, f"{split}_graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        for ex in tqdm(examples, desc=f"Saving {split} graphs", total=len(examples)):
            pt_path = os.path.join(graphs_dir, f"{ex['graph_id']}.pt")
            torch.save(ex["pyg_data"], pt_path)
            
        print(f"Saved {len(examples)} instructions to {json_path}")
        print(f"Saved {len(examples)} graph .pt files to {graphs_dir}/")
        print('-'*50)

# endregion
if __name__ == "__main__":
    print('-' * 50)
    print("Preparing family tree question-answering dataset!")
    print('-' * 50)
    raw_datasets = generate_dataset(n_train=3500, n_val=200, n_test=1000, return_dict=True)
    print("Finished generating raw datasets.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    get_graph_labels = GetGraphLabels(question_end=[ 32, 25 ], tokenizer=tokenizer) # this represents "A:"

    # # --------- Save Graph Dataset ----------
    # graph_datasets = prepare_graph_dataset(raw_datasets)

    # params = {
    #     'tokenizer': tokenizer,
    #     'max_length': 32_768,
    #     'max_rrwp_steps': 16,
    #     'magnetic_q': 0.25,
    #     'get_graph_labels': get_graph_labels,
    # }
    # output_dir = "./src/experiments/knowledge_graph_qa/family_tree_graph_dataset"
    # save_graph_dataset(graph_datasets, output_dir, params)

    
    # # --------- Save Text Dataset ----------
    # text_datasets = prepare_text_dataset(raw_datasets, tokenizer, get_graph_labels)
    # output_dir = "./src/experiments/knowledge_graph_qa/family_tree_text_dataset"
    # save_text_dataset(text_datasets, output_dir)

    # --------- Save LLaGA Dataset ----------
    llaga_datasets = prepare_llaga_dataset(raw_datasets)
    output_dir_llaga = "./src/experiments/knowledge_graph_qa/family_tree_llaga_dataset"
    save_llaga_dataset(llaga_datasets, output_dir_llaga)