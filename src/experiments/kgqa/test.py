"""
Evaluate a specific model checkpoint on the template test dataset.

Usage:
    python -m src.experiments.template.test \\
        --checkpoint_path ./checkpoints/template/<run_name> \\
        [--include_f1]

Results are appended to the experiment's results.json.
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from ...train.eval import evaluate_checkpoint
from .load_data import load_data

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the template test dataset."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint directory, e.g. ./checkpoints/template/<run_name>")
    parser.add_argument("--include_f1", action="store_true",
                        help="Also compute and log macro F1 score.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reconstruct tokenizer from the checkpoint's saved base model name
    graph_bias_config_path = os.path.join(args.checkpoint_path, "graph_bias_config.json")
    with open(graph_bias_config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading test dataset...")
    _, _, test_dataset = load_data(tokenizer)

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        test_dataset=test_dataset,
        experiment_dir=EXPERIMENT_DIR,
        include_f1=args.include_f1,
    )
