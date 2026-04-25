print("src/experiments/backward_compatibility/__main__.py: Running backward compatibility test for graph model...")
import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

import networkx as nx

from ...models.llama_attn_bias import GraphLlamaForCausalLM, GraphLlamaConfig
from ...utils.text_graph_dataset import TextGraphDataset, prepare_example_labels
from ...utils.text_graph_collator import GraphCollator
print("src/experiments/backward_compatibility/__main__.py: Imports completed successfully.")


def _load_graph_model(model_name):
    config = GraphLlamaConfig.from_pretrained("./src/models/graph_llama1b_config.json")
    config.use_cache = False
    model = GraphLlamaForCausalLM.from_pretrained(model_name, config=config)
    return model

ADJ_MATRIX = torch.tensor([
    [ 0, 1, 0, 1, 0, 0, 1],
    [ 1, 0, 1, 0, 1, 0, 0],
    [ 0, 1, 0, 1, 0, 1, 0],
    [ 1, 0, 1, 0, 1, 0, 1],
    [ 0, 1, 0, 1, 0, 1, 0],
    [ 0, 0, 1, 0, 1, 0, 1],
    [ 1, 0, 0, 1, 0, 1, 0]
])
N_NODES = ADJ_MATRIX.shape[0]

def generate_example(model_name, tokenizer):
    import random, string

    # create random permutation of the prefix nodes (0,... N-2)
    prefix_nodes = list(range(N_NODES - 1))  # nodes 0 to N-2
    random.shuffle(prefix_nodes)

    prompt_node = N_NODES - 1  # last node is the prompt node

    permuted_nodes = prefix_nodes + [prompt_node]

    G = nx.Graph()
    for i, node_id in enumerate(permuted_nodes):
        G.add_node(i, text=f"Node {node_id} text")
    for i, node1_id in enumerate(permuted_nodes):
        for j, node2_id in enumerate(permuted_nodes):
            if ADJ_MATRIX[node1_id, node2_id] == 1:
                G.add_edge(i, j)
    
    G.graph["prompt_node"] = prompt_node

    ds = TextGraphDataset([G])

    ds.compute_shortest_path_distances(use_gpu=False)
    ds.compute_rrwp(max_rrwp_steps=4, use_gpu=False)
    ds.compute_magnetic_lap(use_gpu=False)

    ds.tokenize(tokenizer)
    return ds[0], permuted_nodes

if __name__ == "__main__":
    print("Starting backward compatibility test for graph model...")
    model_name = "meta-llama/Llama-3.2-1B"
    
    graph_model = _load_graph_model(model_name)
    print("Models loaded successfully.")

    print("We are using", graph_model.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    outputs = []
    permutations = []
    node_lengths_list = [] # <-- NEW
    
    for i in range(2):
        print(f"\n--- Test Run {i+1} ---")
        example, permutation = generate_example(model_name, tokenizer)
        permutations.append(permutation)
        node_lengths_list.append([len(ids) for ids in example["input_ids"]]) # <-- NEW

        collator = GraphCollator(tokenizer)
        input_graph_batch = collator([example])
        labels = prepare_example_labels([example])
        print("Input graph batch prepared successfully.")

        # Forward pass through the graph model (passing all graph-related inputs)
        graph_outputs = graph_model(
            input_ids=None,
            input_graph_batch=input_graph_batch,
            labels=labels
        )
        outputs.append(graph_outputs)
        print(f"Graph model forward pass successful. Logits shape: {graph_outputs.logits.shape}")
        print("Loss from graph model: ", graph_outputs.loss.item())

    # Compare the logits from both models
    # max_diff = torch.max(torch.abs(outputs[0].logits - outputs[1].logits)) # <-- DELETE
    
    canonical_logits_list = [] # <-- NEW
    for k in range(2): # <-- NEW
        logits = outputs[k].logits[0]  # Shape: (seq_len, vocab_size) # <-- NEW
        perm = permutations[k] # <-- NEW
        lengths = node_lengths_list[k] # <-- NEW
        
        chunks = {} # <-- NEW
        current_idx = 0 # <-- NEW
        
        # 1. Extract prefix nodes (graph indices 0 to N-2) # <-- NEW
        for j in range(len(perm) - 1): # <-- NEW
            length = lengths[j] # <-- NEW
            orig_node_id = perm[j] # <-- NEW
            chunks[orig_node_id] = logits[current_idx : current_idx + length] # <-- NEW
            current_idx += length # <-- NEW
            
        # 2. Extract prompt node (graph index N-1) # <-- NEW
        prompt_length = lengths[-1] # <-- NEW
        orig_prompt_node_id = perm[-1] # <-- NEW
        chunks[orig_prompt_node_id] = logits[current_idx : current_idx + prompt_length] # <-- NEW
        
        # 3. Recombine in canonical node order (0 to N-1) # <-- NEW
        canonical_logits = torch.cat([chunks[i] for i in range(len(perm))], dim=0) # <-- NEW
        canonical_logits_list.append(canonical_logits) # <-- NEW

    max_diff = torch.max(torch.abs(canonical_logits_list[0] - canonical_logits_list[1])) # <-- NEW
    print(f"Maximum absolute difference in logits: {max_diff.item()}")

"""
OUTPUTS:
Maximum absolute difference in logits: 3.075599670410156e-05
Maximum absolute difference in logits: 2.47955322265625e-05
Maximum absolute difference in logits: 2.5153160095214844e-05
Maximum absolute difference in logits: 2.7298927307128906e-05
Maximum absolute difference in logits: 3.063678741455078e-05
"""