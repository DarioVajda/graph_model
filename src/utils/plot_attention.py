import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from accelerate.utils import send_to_device

from .plot_text_graph import visualize_text_graph

def plot_graph_attention(
    model, 
    graph_example, 
    tokenizer, 
    collator, 
    output_path: str, 
    plot_means: bool = True, 
    plot_heads: bool = True,
    device: torch.device = torch.device("cpu"),
    plot_graph=True
):
    """
    Runs a forward pass to extract attention weights and plots heatmaps according to specifications and potentially also visualizes the graph structure itself.
    
    Args:
        model: The custom GraphLlamaForCausalLM model.
        graph_example: A single entry of type TextGraph.
        tokenizer: The tokenizer used for the model.
        collator: The GraphCollator instance to prepare the inputs.
        output_path: Base directory to save the plots.
        plot_means: If True, plots the mean attention across all heads.
        plot_heads: If True, plots individual head attentions.
        device: Torch device to run the inference on.
        plot_graph: If True, also visualizes the graph structure.
    """
    if plot_graph:
        print("Visualizing the graph structure...")
        visualize_text_graph(
            graph_data=graph_example,
            output_path=os.path.join(output_path, "graph_structure.png"),
            max_line_length=30,
            use_spectral_layout=False,
            figsize=(8, 8)
        )
        visualize_text_graph(
            graph_data=graph_example,
            output_path=os.path.join(output_path, "graph_structure_spectral.png"),
            max_line_length=30,
            use_spectral_layout=True,
            figsize=(8, 8)
        )

    if not plot_means and not plot_heads:
        print("Both plot_means and plot_heads are False. Nothing to plot.")
        return

    model.eval()
    model.to(device)

    # 1. Reconstruct the exact token sequence order (non-prompt nodes first, prompt node last)
    # Handle tensor or int for prompt_node
    prompt_idx = graph_example['prompt_node'].item() if torch.is_tensor(graph_example['prompt_node']) else graph_example['prompt_node']
    raw_input_ids = graph_example['input_ids']
    
    ordered_ids = []
    for j in range(len(raw_input_ids)):
        if j != prompt_idx:
            ordered_ids.append(raw_input_ids[j])
    ordered_ids.append(raw_input_ids[prompt_idx])
    
    # Flatten the sequence robustly
    if isinstance(ordered_ids[0], torch.Tensor):
        flat_ids = torch.cat(ordered_ids).tolist()
    else:
        flat_ids = [token_id for node_ids in ordered_ids for token_id in node_ids]
    
    # Decode tokens and format spaces
    # Note: Llama uses " " (U+2581) for spaces in SentencePiece. We replace it and standard spaces.
    raw_tokens = tokenizer.convert_ids_to_tokens(flat_ids)
    clean_tokens = [t.replace('Ġ', '_').replace(' ', '_').replace(' ', '_') for t in raw_tokens]

    # 2. Prepare the batched input for the model
    batched_inputs = collator([graph_example])
    batched_inputs = send_to_device(batched_inputs, device)
    
    # Remove labels if present as we only need attentions, not loss
    batched_inputs.pop("labels", None)

    # 3. Forward pass to get attentions
    with torch.no_grad():
        outputs = model(
            input_ids=None, 
            input_graph_batch=batched_inputs, 
            output_attentions=True,
            use_cache=False # Disable cache to get full attention matrix cleanly
        )

    # outputs.attentions is a tuple of (batch_size, num_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions

    # 4. Plotting Logic
    seq_len = len(clean_tokens)
    # Dynamically adjust figure size based on sequence length, but cap it so it doesn't crash
    fig_size = max(8, min(seq_len * 0.3, 30)) 

    for layer_idx, layer_attn in enumerate(tqdm(attentions, desc="Plotting Layers")):
        # Extract the single batch instance: shape (num_heads, seq_len, seq_len)
        attn_matrix = layer_attn[0].cpu().numpy()

        # Determine directory structure based on flags
        if plot_means and plot_heads:
            layer_dir = os.path.join(output_path, f"layer_{layer_idx}")
            heads_dir = os.path.join(layer_dir, "heads")
            os.makedirs(heads_dir, exist_ok=True)
        elif plot_means:
            os.makedirs(output_path, exist_ok=True)
        elif plot_heads:
            layer_dir = os.path.join(output_path, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)

        # Plot Mean Attention
        if plot_means:
            mean_attn = attn_matrix.mean(axis=0)
            
            plt.figure(figsize=(fig_size, fig_size))
            ax = sns.heatmap(mean_attn, xticklabels=clean_tokens, yticklabels=clean_tokens, 
                             square=True, cmap="viridis", cbar_kws={"shrink": .8})
            ax.xaxis.tick_top() # Put X-axis on top
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.title(f"Layer {layer_idx} - Mean Attention", pad=20, fontsize=14, fontweight='bold')
            
            if plot_heads:
                save_path = os.path.join(layer_dir, "mean.png")
            else:
                save_path = os.path.join(output_path, f"layer_{layer_idx}.png")
                
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

        # Plot Individual Heads
        if plot_heads:
            for head_idx in range(attn_matrix.shape[0]):
                head_attn = attn_matrix[head_idx]
                
                plt.figure(figsize=(fig_size, fig_size))
                ax = sns.heatmap(head_attn, xticklabels=clean_tokens, yticklabels=clean_tokens, 
                                 square=True, cmap="viridis", cbar_kws={"shrink": .8})
                ax.xaxis.tick_top() # Put X-axis on top
                plt.xticks(rotation=90, fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.title(f"Layer {layer_idx} - Head {head_idx}", pad=20, fontsize=14, fontweight='bold')
                
                if plot_means:
                    save_path = os.path.join(heads_dir, f"head_{head_idx}.png")
                else:
                    save_path = os.path.join(layer_dir, f"head_{head_idx}.png")
                    
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()


# ==============================================================================
# TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    from . import TextGraphDataset, GraphCollator
    from ..models.llama_attn_bias import GraphLlamaForCausalLM
    from ..experiments.expressiveness.data_gen import create_and_save_dataset, dataset_path_and_size
    from transformers import AutoTokenizer

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEVICE = get_device()

    # 1. Setup Data
    TEST_DATASET_SIZE = 10  # Smaller for plotting test
    test_dataset_path, _ = dataset_path_and_size(TEST_DATASET_SIZE)
    if not os.path.exists(test_dataset_path):
        print(f"Test dataset not found at {test_dataset_path}. Creating new dataset...")
        create_and_save_dataset(dataset_size=TEST_DATASET_SIZE, min_nodes=10, max_nodes=20, spectral_dims=16, model_name="meta-llama/Llama-3.2-1B")
    test_dataset = TextGraphDataset.load(test_dataset_path)

    # 2. Setup Model & Tokenizer
    trained_model_path = "./checkpoints/bias_only_combined/checkpoint-3120"
    model = GraphLlamaForCausalLM.from_pretrained(trained_model_path, bias_type="combined", max_spd=4, attn_implementation="eager")
    print(f"Loaded model from {trained_model_path}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    collator = GraphCollator(tokenizer=tokenizer)

    # 3. Pick one example from the dataset
    graph_example = test_dataset[0]

    # 4. Test Plotting Functions

    # Scenario A: Plot EVERYTHING (Means + Heads)
    # print('-'*80)
    # print("Plotting Everything (Means and Heads)...")
    # print('-'*80)
    # plot_graph_attention(
    #     model=model,
    #     graph_example=graph_example,
    #     tokenizer=tokenizer,
    #     collator=collator,
    #     output_path="./plots/attn_plots_all",
    #     plot_means=True,
    #     plot_heads=True,
    #     device=DEVICE
    # )

    # Scenario B: Plot MEANS ONLY
    print('-'*80)
    print("Plotting Means Only...")
    print('-'*80)
    plot_graph_attention(
        model=model,
        graph_example=graph_example,
        tokenizer=tokenizer,
        collator=collator,
        output_path="./plots/attn_plots_means",
        plot_means=True,
        plot_heads=False,
        device=DEVICE
    )

    # Scenario C: Plot HEADS ONLY
    # print('-'*80)
    # print("Plotting Heads Only...")
    # print('-'*80)
    # plot_graph_attention(
    #     model=model,
    #     graph_example=graph_example,
    #     tokenizer=tokenizer,
    #     collator=collator,
    #     output_path="./plots/attn_plots_heads",
    #     plot_means=False,
    #     plot_heads=True,
    #     device=DEVICE
    # )

    print("Finished plotting! Check the output directories.")