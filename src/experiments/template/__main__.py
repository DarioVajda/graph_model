import argparse
import os

from ...utils import GraphCollator
from ...train import (
    init_model, select_active_params, print_trainable_parameters,
    training_run, save_run_metadata, get_device,
)
from .load_data import load_data

EXPERIMENT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "template"

# Graph-attention bias configuration — edit for your experiment.
BIAS_PARAMS = {
    "spd":          True,
    "max_spd":      8,
    "laplacian":    False,
    "rwse":         False,
    "rrwp":         True,
    "max_rw_steps": 16,
    "magnetic":     True,
    "magnetic_dim": 32,
    "magnetic_q":   0.25,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Template experiment: train GraphLlama on a synthetic graph dataset."
    )
    parser.add_argument("--model_name",       type=str,   default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--num_epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",       type=int,   default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate",    type=float, default=3e-4)
    parser.add_argument("--bias_learning_rate", type=float, default=5e-3)
    parser.add_argument("--eval_every",       type=int,   default=20)
    parser.add_argument("--lora_r",           type=int,   default=16,
                        help="LoRA rank. Set to 0 to disable LoRA.")
    parser.add_argument("--k_hop",            type=int,   default=2,
                        help="K-hop attention gate. Set to 0 to disable.")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--include_f1",       action="store_true")
    parser.add_argument("--wandb_project",    type=str,   default="GraphLLM",
                        help="WandB project name. Pass 'none' to disable logging.")
    parser.add_argument("--run_name",         type=str,   default=None)
    # The KHop model stores all graph bias params under graph_bias.bias_modules.*
    # Use "graph_bias" to activate all of them, or override with specific substrings.
    parser.add_argument("--active_params",    nargs="+",
                        default=["graph_bias"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    wandb_project = None if args.wandb_project.lower() == "none" else args.wandb_project

    lora_config = {
        "r":              args.lora_r,
        "lora_alpha":     args.lora_r * 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout":   0.05,
        "bias":           "none",
    } if args.lora_r > 0 else None

    run_name = args.run_name or f"template_lora{args.lora_r}_khop{args.k_hop}"
    run_name = save_run_metadata(
        run_name=run_name,
        experiment_dir=EXPERIMENT_DIR,
        base_model=args.model_name,
        active_params=args.active_params,
        lr=args.learning_rate,
        bias_lr=args.bias_learning_rate,
        lora_config=lora_config,
        num_epochs=args.num_epochs,
        # experiment-specific extras:
        bias_params=BIAS_PARAMS,
        k_hop=args.k_hop,
    )

    device = get_device()
    model, tokenizer = init_model(
        model_name=args.model_name,
        device=device,
        bias_params=BIAS_PARAMS,
        k_hop=args.k_hop,
    )

    print("Loading data...")
    train_dataset, eval_dataset, test_dataset = load_data(tokenizer)
    collator = GraphCollator(k_hop=args.k_hop)

    print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}  Test: {len(test_dataset)}")

    model = select_active_params(model, active_params=args.active_params, lora=lora_config)
    print_trainable_parameters(model)

    training_run(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        collator=collator,
        run_name=run_name,
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=EXPERIMENT_DIR,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bias_learning_rate=args.bias_learning_rate,
        accumulation_steps=args.accumulation_steps,
        active_params=args.active_params,
        eval_every=args.eval_every,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        include_f1=args.include_f1,
        wandb_project=wandb_project,
        seed=args.seed,
    )
