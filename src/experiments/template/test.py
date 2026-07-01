"""
Re-evaluate a saved checkpoint on the template test split.

Usage:
    python -m src.experiments.template.test \\
        --checkpoint-path ./checkpoints/template/<run_name> \\
        [--include-f1]

The test split is rebuilt from a default ``RunConfig``; if you trained with
non-default data/bias flags (e.g. --num-samples, --data-seed, --no-magnetic),
pass the matching flags here too so the features line up with the checkpoint.
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from ...train.eval import evaluate_checkpoint
from .config import RunConfig
from .data import load_data

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    d = RunConfig()
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the template test split.")
    p.add_argument("--checkpoint-path", required=True,
                   help="Checkpoint directory, e.g. ./checkpoints/template/<run_name>")
    p.add_argument("--include-f1", action="store_true", help="Also compute and log macro F1.")
    # data/bias flags that must match the checkpoint for the features to line up
    p.add_argument("--num-samples", type=int, default=d.num_samples)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--data-seed", type=int, default=d.data_seed)
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reconstruct the tokenizer from the checkpoint's saved base model name.
    with open(os.path.join(args.checkpoint_path, "graph_bias_config.json")) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    cfg = RunConfig(
        model_name=base_model_name,
        num_samples=args.num_samples, max_length=args.max_length, data_seed=args.data_seed,
        spd=args.spd, rrwp=args.rrwp, magnetic=args.magnetic,
    ).validate()

    print("Loading test dataset...")
    _, _, test_dataset = load_data(cfg, tokenizer)

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        test_dataset=test_dataset,
        experiment_dir=EXPERIMENT_DIR,
        include_f1=args.include_f1,
    )
