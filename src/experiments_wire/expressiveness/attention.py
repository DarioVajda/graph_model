import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence

from ...models.llama_wire import GraphLlamaForCausalLM
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer

def create_batches(dataset, batch_size, pad_token_id=128001, bidirectional_prefix=False):
    """
    Create batches from the dataset.

    Args:
        dataset (list): List of dataset examples, each one is a dict with "graph", "x", "y", "label", "input_ids, "labels", "node_ids", and "node_spectral_features".
        batch_size (int): Size of each batch.
        pad_token_id (int): Token ID used for padding sequences.
    """
    batches_lists = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batches_lists.append(batch)
    
    # convert lists of dicts to dict of padded tensors
    batches = []
    for batch in batches_lists:
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        num_nodes = batch[0]["node_spectral_features"].size(0)
        node_ids = pad_sequence([item["node_ids"] for item in batch], batch_first=True, padding_value=num_nodes-1)
        node_spectral_features = torch.stack([item["node_spectral_features"] for item in batch], dim=0) # these should already be the same size [ max_num_nodes, spectral_dims ]
        
        if bidirectional_prefix:
            max_seq_len = input_ids.size(1)
            attention_mask = torch.full((input_ids.size(0), max_seq_len, max_seq_len), float('-inf'))
            for j, item in enumerate(batch):
                attention_mask[j][:len(item["input_ids"]), :len(item["input_ids"])] = item["attention_mask"]
            attention_mask = attention_mask.unsqueeze(1)
        else:
            attention_mask = (input_ids != pad_token_id).long()

        batches.append({
            "input_ids": input_ids,
            "labels": labels,
            "node_ids": node_ids,
            "node_spectral_features": node_spectral_features,
            "attention_mask": attention_mask,
        })
        
    return batches

def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print("Processed dataset not found. Generating new dataset...")
        MIN_NODES = 10
        MAX_NODES = 20
        SPECTRAL_DIMS = 8
        DATASET_SIZE = 2500

        dataset = generate_graph_dataset(DATASET_SIZE, min_size=MIN_NODES, max_size=MAX_NODES)
        print(f"Generated {len(dataset)} examples.")

        model_name = "meta-llama/Llama-3.2-1B"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        prepare_dataset(dataset, tokenizer, graph_model=True, max_nodes=MAX_NODES, spectral_dims=SPECTRAL_DIMS, save_path=dataset_path)
        print(f"Processed dataset with {DATASET_SIZE} examples.")

    print(f"Loading processed dataset from {dataset_path}...")
    data = torch.load(dataset_path, weights_only=False)
    print(f"Loaded dataset with {len(data['dataset'])} examples.")
    return data['hyperparameters'], data['dataset']

def split_dataset(dataset, val_ratio):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    
    train_set = dataset[:total_size - val_size]
    val_set = dataset[total_size - val_size:]

    return train_set, val_set


def visualize_layer_attention(
    model, 
    tokenizer, 
    sample_input, 
    layer_k, 
    output_path, 
    figsize=(16, 8),
    bidirectional_prefix=False
):
    """
    Runs a forward pass on a single sample, extracts attention from the k-th layer,
    and saves visualizations to the output path.
    
    Args:
        model: Your GraphLlamaForCausalLM instance.
        tokenizer: The tokenizer.
        sample_input: A dictionary containing 'input_ids', 'node_ids', 
                      'node_spectral_features', 'attention_mask'. 
                      (Should be a batch of size 1).
        layer_k: Integer index of the layer to inspect (0 to num_layers-1).
        output_path: File path to save the image (e.g., "debug_plots/attn_layer_10.png").
    """
    model.eval()
    
    # 1. Prepare Inputs
    device = next(model.parameters()).device
    
    # Ensure batch size is 1 for clear visualization
    inputs = create_batches([sample_input], 1, bidirectional_prefix=bidirectional_prefix)
    inputs = inputs[0]

    for k in inputs.keys():
        print(k, inputs[k])
        inputs[k] = inputs[k].to(device)

    print(inputs["input_ids"].device)

    # 2. Forward Pass with Attention Capture
    # We must force output_attentions=True. 
    # Note: If you are using Flash Attention, you might need to load the model 
    # with attn_implementation="eager" for this to work, as FlashAttn 
    # often doesn't return weights.
    print(f" probing Layer {layer_k}...")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            node_ids=inputs["node_ids"],
            node_spectral_features=inputs["node_spectral_features"],
            labels=inputs["labels"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            use_cache=False
        )
    
    # 3. Extract Attention Weights
    # outputs.attentions is a tuple of tensors: (batch, heads, seq_len, seq_len)
    # We grab the k-th layer.
    if outputs.attentions is None:
        raise ValueError("Model did not return attentions. Ensure attn_implementation='eager' is used if Flash Attention is interfering.")
        
    attn_matrix = outputs.attentions[layer_k] # [1, num_heads, seq_len, seq_len]
    
    # Squeeze batch dim
    attn_matrix = attn_matrix[0] # [num_heads, seq_len, seq_len]
    
    # 4. Process for Visualization
    # We assume 'Average over Heads' is the most useful quick diagnostic.
    # You can also change this to max() or look at specific heads.
    avg_attn = attn_matrix.mean(dim=0).float().cpu().numpy() # [seq_len, seq_len]

    # Get tokens for axis labels
    seq_len = avg_attn.shape[0]
    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = [tokenizer.decode([tid]).replace(" ", "") for tid in input_ids]
    
    # Truncate labels if too long (visual noise)
    tokens = [t if len(t) < 10 else "..." for t in tokens]

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- PLOT 1: The Full Heatmap (Context Structure) ---
    sns.heatmap(
        avg_attn, 
        cmap="viridis", 
        xticklabels=tokens, 
        yticklabels=tokens,
        ax=ax1,
        vmin=0, vmax=0.2 # Cap max value to see low-level structure clearly
    )
    ax1.set_title(f"Layer {layer_k} Average Attention Map\n(Look for block diagonals!)")
    ax1.tick_params(axis='x', rotation=90)
    ax1.tick_params(axis='y', rotation=0)

    # --- PLOT 2: The 'Prompt' Probe (Retrieval) ---
    # We look at the very last token (the Prompt) and see what it attended to.
    last_token_attn = avg_attn[-1, :]
    
    # Highlight the top 3 tokens it focused on
    top_indices = last_token_attn.argsort()[-3:][::-1]
    top_tokens = [f"{tokens[i]}({i})" for i in top_indices]
    
    ax2.bar(range(seq_len), last_token_attn, color='skyblue')
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(tokens, rotation=90)
    ax2.set_title(f"Prompt Token Attention Profile\nTop focuses: {', '.join(top_tokens)}")
    ax2.set_xlabel("Input Sequence Tokens")
    ax2.set_ylabel("Attention Weight")
    
    # Add a visual marker for where the prompt is
    ax2.axvline(x=seq_len-1, color='red', linestyle='--', alpha=0.5, label="Prompt Pos")

    plt.tight_layout()
    
    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Attention map saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # check if there is a processed dataset
    dataset_path = f"src/experiments/expressiveness/processed_easy_dataset.pt"
    parameters, dataset = load_dataset(dataset_path)

    # split into train and val
    train_set, val_set = split_dataset(dataset, val_ratio=0.2)
    train_set = train_set[:80]
    val_set = val_set[:20]
    print(f"Dataset split into {len(train_set)} training and {len(val_set)} validation examples.")

    model_name = "meta-llama/Llama-3.2-1B"
    model_path = "easy_graph_llama_model.pth"

    config = LlamaConfig.from_pretrained(model_name)
    print("Model config loaded.")

    model = GraphLlamaForCausalLM.from_pretrained(model_name, config=config, spectral_dims=2, strict=False)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # print(model.model.rotary_emb.spectral_freqs)
    # exit()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded tokenizer for", model_name)

    visualize_layer_attention(
        model=model,
        tokenizer=tokenizer,
        sample_input=val_set[0], 
        layer_k=0,
        output_path=f"src/experiments/expressiveness/bidirectional_layer_{0}_plot.png",
        bidirectional_prefix=True
    )

    visualize_layer_attention(
        model=model,
        tokenizer=tokenizer,
        sample_input=val_set[0], 
        layer_k=1,
        output_path=f"src/experiments/expressiveness/bidirectional_layer_{1}_plot.png",
        bidirectional_prefix=True
    )

    visualize_layer_attention(
        model=model,
        tokenizer=tokenizer,
        sample_input=val_set[0], 
        layer_k=15,
        output_path=f"src/experiments/expressiveness/bidirectional_layer_{15}_plot.png",
        bidirectional_prefix=True
    )

