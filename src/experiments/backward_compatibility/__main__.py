import torch
from transformers import LlamaForCausalLM, LlamaConfig
from ...models.llama import GraphLlamaForCausalLM


def _test_graph_llama(model_name = "meta-llama/Llama-3.2-1B"):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test the implementation
    print("Initializing Custom Llama Model...")

    # 1. Setup a dummy config
    config = LlamaConfig.from_pretrained(model_name)

    # 2.1 Load the model weights into the custom class
    custom_model = GraphLlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        spectral_dims=32,
        strict=False # ignore missing keys due to custom RoPE
    )

    # 2.2 Load the model weights into the default class
    default_model = LlamaForCausalLM.from_pretrained(model_name, config=config)

    print("Weights loaded successfully!")
    
    # exit(0)

    verify_architecture = False
    if verify_architecture:
        # 3.1 Verify the Custom Model Architecture
        print("\n--- Custom Model Architecture ---")
        print(custom_model)

        # 3.2 Verify the Default Model Architecture
        print("\n--- Default Model Architecture ---")
        print(default_model)

    # 4. Dummy Forward Pass
    dummy_input = torch.randint(0, 1050, (2, 10))
    dummy_node_ids = torch.zeros((2, 10), dtype=torch.long)  # All tokens belong to node 0
    dummy_spectral_features = torch.randn(2, 3, 32)  # [batch_size, num_nodes, spectral_dims]

    # shift input to get labels and set -100 for padding
    dummy_labels = dummy_input[:, 1:].clone()
    dummy_labels = torch.cat([dummy_labels, torch.tensor([[ -100], [ -100]])], dim=1)

    print("dummy_input: ", dummy_input)
    print("dummy_labels: ", dummy_labels)

    # 4.1 Forward pass through custom model
    print("\nPerforming forward pass through custom model...")
    custom_output = custom_model(dummy_input, node_ids=dummy_node_ids, node_spectral_features=dummy_spectral_features, labels=dummy_labels)
    print(f"Custom forward pass logits shape: {custom_output.logits.shape}")

    # 4.2 Forward pass through default model
    print("\nPerforming forward pass through default model...")
    default_output = default_model(dummy_input, labels=dummy_labels)
    print(f"Default forward pass logits shape: {default_output.logits.shape}")
    print('\n')

    # 5.1 Output from custom model
    print("--- Custom Model Output ---")
    predicted_tokens = torch.argmax(custom_output.logits, dim=-1)
    print("Predicted tokens:")
    print(predicted_tokens)
    print("Custom Model Loss:", custom_output.loss)

    # 5.2 Output from default model
    print("--- Default Model Output ---")
    predicted_tokens_default = torch.argmax(default_output.logits, dim=-1)
    print("Predicted tokens:")
    print(predicted_tokens_default)
    print("Default Model Loss:", default_output.loss)
    

if __name__ == "__main__":
    _test_graph_llama(model_name="meta-llama/Llama-3.2-1B")
