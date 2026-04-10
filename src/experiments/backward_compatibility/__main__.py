print("src/experiments/backward_compatibility/__main__.py: Running backward compatibility test for graph model...")
import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

import networkx as nx

from ...models.llama_attn_bias import GraphLlamaForCausalLM, GraphLlamaConfig
from ...utils.text_graph_dataset import TextGraphDataset, prepare_example_labels
from ...utils.text_graph_collator import GraphCollator
print("src/experiments/backward_compatibility/__main__.py: Imports completed successfully.")

def _load_default_model(model_name):
    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, config=config)
    return model

def _load_graph_model(model_name):
    config = GraphLlamaConfig.from_pretrained("./src/models/graph_llama1b_config.json")
    model = GraphLlamaForCausalLM.from_pretrained(model_name, config=config)
    return model

def generate_example(model_name, tokenizer):
    G = nx.Graph()
    G.add_node(0, text="Machine Learning is the art of drawing unforeseen insights from ordinary data.")
    G.graph["prompt_node"] = 0

    ds = TextGraphDataset([G])

    ds.compute_laplacian_coordinates()
    ds.compute_shortest_path_distances()
    ds.compute_rwse()
    ds.compute_rrwp(max_rrwp_steps=4)
    ds.compute_magnetic_lap()

    ds.tokenize(tokenizer)
    return ds[0]

if __name__ == "__main__":
    print("Starting backward compatibility test for graph model...")
    model_name = "meta-llama/Llama-3.2-1B"
    
    default_model = _load_default_model(model_name)
    graph_model = _load_graph_model(model_name)
    print("Models loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    example = generate_example(model_name, tokenizer)
    collator = GraphCollator(tokenizer)
    input_graph_batch = collator([example])
    labels = prepare_example_labels([example])
    print("Input graph batch prepared successfully.")

    # Forward pass through the default model (passing only input_ids and labels
    default_outputs = default_model(input_ids=input_graph_batch["input_ids"][0][0].unsqueeze(0), labels=labels[0].unsqueeze(0))
    print(f"Default model forward pass successful. Logits shape: {default_outputs.logits.shape}")
    print("Loss from default model: ", default_outputs.loss.item())

    # Forward pass through the graph model (passing all graph-related inputs)
    graph_outputs = graph_model(
        input_ids=None,
        input_graph_batch=input_graph_batch,
        labels=labels
    )
    print(f"Graph model forward pass successful. Logits shape: {graph_outputs.logits.shape}")
    print("Loss from graph model: ", graph_outputs.loss.item())

    # Compare the logits from both models
    max_diff = torch.max(torch.abs(default_outputs.logits - graph_outputs.logits))
    print(f"Maximum absolute difference in logits: {max_diff.item():.8f}")