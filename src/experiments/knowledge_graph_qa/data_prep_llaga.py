from .data_gen import generate_dataset

import os
import json
import torch
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

# Disable tokenizer parallelism to make it safe for multiprocessing/SentenceTransformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------------------------
# region LLaGA Format Dataset Preparation (for RGLM)
# ------------------------------------------------------------------------------

def prepare_kg_llaga_dataset(raw_datasets, encoder_model_name='all-MiniLM-L6-v2', device='cuda'):
    print(f"Loading SentenceTransformer '{encoder_model_name}' on {device}...")
    encoder = SentenceTransformer(encoder_model_name, device=device)
    
    prepared_dataset = { 
        'train': {'instructions': [], 'graphs': []}, 
        'val': {'instructions': [], 'graphs': []}, 
        'test': {'instructions': [], 'graphs': []},
    }
    
    for split, graphs in raw_datasets.items():
        for graph_idx, nx_graph in enumerate(tqdm(graphs, desc=f"Preparing LLaGA {split} split")):
            graph_id = f"{split}_{graph_idx}"
            
            # --- INCIDENCE GRAPH CONVERSION ---
            incidence_graph = nx.DiGraph()
            
            # 1. Add standard nodes
            for v in nx_graph.nodes():
                incidence_graph.add_node(v, text=str(v))
                
            # 2. Convert edges to nodes
            for u, v, data in nx_graph.edges(data=True):
                relation_text = data.get('relation', 'connected').lower()
                edge_node_id = f"{u}_{relation_text}_{v}"
                
                # Add the relationship as a node with text
                incidence_graph.add_node(edge_node_id, text=relation_text.lower().replace('_', ' '))
                
                # Connect: Source -> Relation -> Target
                incidence_graph.add_edge(u, edge_node_id)
                incidence_graph.add_edge(edge_node_id, v)
            # ----------------------------------

            # 3. Create PyG mapping and edge index from the INCIDENCE graph
            node_mapping = {n: i for i, n in enumerate(incidence_graph.nodes())}
            
            edge_list = [[node_mapping[u], node_mapping[v]] for u, v in incidence_graph.edges()]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
                
            # 4. Extract and Embed Texts (Now includes both entities AND relations)
            node_texts = [incidence_graph.nodes[n]['text'] for n in incidence_graph.nodes()]
            
            with torch.no_grad():
                x = encoder.encode(node_texts, convert_to_tensor=True, device=device)
                
            pyg_data = Data(x=x.cpu(), edge_index=edge_index)
            
            prepared_dataset[split]['graphs'].append({
                "graph_id": graph_id,
                "pyg_data": pyg_data
            })
            
            # 5. Create Conversations
            for func_name, (question, pointers, answer) in nx_graph.graph['questions'].items():
                instruction_id = f"kg_qa_{graph_id}_{func_name}"
                conversation = [
                    {"from": "human", "value": f"<graph>\nQuestion: {question}"},
                    {"from": "gpt", "value": str(answer)}
                ]
                prepared_dataset[split]['instructions'].append({
                    "id": instruction_id,
                    "graph_id": graph_id,
                    "conversations": conversation
                })
            
    return prepared_dataset

def save_llaga_dataset(llaga_datasets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for split, data in llaga_datasets.items():
        print(f"Saving LLaGA {split} split...")
        
        # Save JSON Instructions
        json_path = os.path.join(output_dir, f"{split}_instructions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data['instructions'], f, indent=4)
            
        # Save PyG Graph Tensors
        graphs_dir = os.path.join(output_dir, f"{split}_graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        for graph_data in tqdm(data['graphs'], desc=f"Saving {split} graphs"):
            pt_path = os.path.join(graphs_dir, f"{graph_data['graph_id']}.pt")
            torch.save(graph_data["pyg_data"], pt_path)
            
        print(f"Saved {len(data['instructions'])} instructions to {json_path}")
        print(f"Saved {len(data['graphs'])} graph .pt files to {graphs_dir}/")
        print('-'*50)

# endregion
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    VAL_COUNT = 30
    TEST_COUNT = 150
    TRAIN_COUNT = 500

    MIN_NODES = 30
    MAX_NODES = 50
    raw_train_graphs, raw_val_graphs, raw_test_graphs = generate_dataset(train_count=TRAIN_COUNT, val_count=VAL_COUNT, test_count=TEST_COUNT, min_nodes=MIN_NODES, max_nodes=MAX_NODES)
    print(f"Generated {len(raw_train_graphs)} training graphs, {len(raw_val_graphs)} validation graphs, and {len(raw_test_graphs)} test graphs with node counts between {MIN_NODES} and {MAX_NODES}.")

    
    print(f"Generated {len(raw_train_graphs)} training graphs, {len(raw_val_graphs)} validation graphs, and {len(raw_test_graphs)} test graphs.")

    # Organize into a dictionary mapped by splits
    raw_datasets = {
        'train': raw_train_graphs,
        'val': raw_val_graphs,
        'test': raw_test_graphs,
    }

    # --------- Prepare and Save LLaGA Dataset ----------
    # Determine device based on availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    llaga_datasets = prepare_kg_llaga_dataset(raw_datasets, device=device)
    output_dir_llaga = f"./src/experiments/knowledge_graph_qa/llaga_datasets/dataset_{MIN_NODES}-{MAX_NODES}"
    
    save_llaga_dataset(llaga_datasets, output_dir_llaga)
    print("Finished preparing and saving RGLM dataset.")