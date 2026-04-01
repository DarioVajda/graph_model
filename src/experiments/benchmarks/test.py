# eval_test.py
import argparse
import os
import json
from transformers import TrainingArguments, AutoTokenizer

from ...utils import GraphTrainer, GraphCollator
from ...models.llama_attn_bias_slow import GraphLlamaForCausalLM
from .load_data import load_dataset
from .train_utils import get_device, compute_exact_match

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned GraphLLaMA model on the test dataset.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--output_file", type=str, default="./src/experiments/benchmarks/test_results.json", help="JSON file to save the test results.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    # Relying on your custom from_pretrained logic to handle LoRA/active params
    model = GraphLlamaForCausalLM.from_pretrained(args.checkpoint_path)
    tokenizer = GraphLlamaForCausalLM.from_pretrained(args.checkpoint_path)
    model.to(device)
    model.eval() # Ensure dropout and batchnorm are in eval mode

    # Load only the test dataset
    dataset_dir = f"./src/experiments/benchmarks/processed_data/{args.dataset_name}"
    datasets = load_dataset(dataset_dir, train=False, val=False, test=True)
    test_dataset = datasets["test"]
    collator = GraphCollator()

    print(f"Test dataset size: {len(test_dataset)}")

    # Setup TrainingArguments specifically for evaluation (WandB disabled)
    eval_args = TrainingArguments(
        output_dir="./eval_temp",                 # Dummy directory for Trainer internals
        per_device_eval_batch_size=args.batch_size,
        report_to="none",                         # Completely disable WandB and other loggers
        do_train=False,
        do_eval=True,
    )

    # Initialize the custom trainer
    trainer = GraphTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=collator,
        compute_metrics=compute_exact_match,
    )

    print("Starting exact match evaluation on the test set...")
    metrics = trainer.evaluate()
    
    print("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("--------------------------\n")

    # Load previous results from the JSON file
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Create a clean key for the JSON file based on the dataset and checkpoint folder
    checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    run_key = f"{args.dataset_name}_{checkpoint_name}"
    if run_key in all_results:
        version = 2
        while f"{run_key}_v{version}" in all_results:
            version += 1
        run_key = f"{run_key}_v{version}"

    all_results = {run_key: metrics, **all_results}  # Add new results to the top of the JSON file and keep existing results below

    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Metrics successfully appended to {args.output_file} under the key '{run_key}'.")

