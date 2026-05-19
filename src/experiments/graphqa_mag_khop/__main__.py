import argparse
import os

from ...utils import GraphCollator
from ...train import (
    init_model, select_active_params, print_trainable_parameters,
    training_run, save_run_metadata, get_device,
)
from ..graphqa.load_dataset import load_graphqa_datasets

EXPERIMENT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "graphqa_mag_khop"

BIAS_PARAMS = {
    "spd":          False,
    "max_spd":      8,
    "laplacian":    False,
    "rwse":         False,
    "rrwp":         False,
    "max_rw_steps": 16,
    "magnetic":     True,
    "magnetic_dim": 32,
    "magnetic_q":   0.25,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphQA with magnetic Laplacian bias only; sweeps K-hop mask depth and eigenvector count M."
    )
    parser.add_argument("--model_name",         type=str,   default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--task",               type=str,   default="shortest_path",
                        help="GraphQA task: node_count, edge_count, cycle_check, triangle_counting, "
                             "node_degree, connected_nodes, reachability, edge_existence, shortest_path")
    parser.add_argument("--graph_type",         type=str,   default="standard",
                        help="Graph representation: 'standard' or 'incidence'")
    parser.add_argument("--k_hop",              type=int,   default=0,
                        help="K-hop attention gate. 0 disables the mask.")
    parser.add_argument("--magnetic_m",         type=int,   default=0,
                        help="Max eigenvectors to keep per graph. 0 keeps all.")
    parser.add_argument("--num_epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--accumulation_steps", type=int,   default=8)
    parser.add_argument("--learning_rate",      type=float, default=3e-5)
    parser.add_argument("--bias_learning_rate", type=float, default=5e-3)
    parser.add_argument("--lora_r",             type=int,   default=16,
                        help="LoRA rank. 0 disables LoRA.")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--wandb_project",      type=str,   default="GraphLLM",
                        help="WandB project name. Pass 'None' to disable.")
    parser.add_argument("--run_name",           type=str,   default=None)
    parser.add_argument("--active_params",      nargs="+",  default=["graph_bias"])
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

    incidence_suffix = "_incidence" if args.graph_type == "incidence" else ""
    run_name = args.run_name or f"GraphQA_{args.task}_K={args.k_hop}_M={args.magnetic_m}{incidence_suffix}"
    run_name = save_run_metadata(
        run_name=run_name,
        experiment_dir=EXPERIMENT_DIR,
        base_model=args.model_name,
        active_params=args.active_params,
        lr=args.learning_rate,
        bias_lr=args.bias_learning_rate,
        lora_config=lora_config,
        num_epochs=args.num_epochs,
        bias_params=BIAS_PARAMS,
        k_hop=args.k_hop,
        magnetic_m=args.magnetic_m,
        task=args.task,
        graph_type=args.graph_type,
    )

    device = get_device()
    model, tokenizer = init_model(
        model_name=args.model_name,
        device=device,
        bias_params=BIAS_PARAMS,
        k_hop=args.k_hop,
    )

    datasets_dir = "./src/experiments/graphqa/processed_datasets"
    train_dataset, test_dataset = load_graphqa_datasets(
        datasets_dir, [args.task], [args.task], graph_type=args.graph_type,
    )
    train_ds_size = len(train_dataset)
    eval_dataset  = train_dataset[int(0.85 * train_ds_size):]
    train_dataset = train_dataset[:int(0.85 * train_ds_size)]

    collator = GraphCollator(k_hop=args.k_hop, magnetic_m=args.magnetic_m)

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
        wandb_project=wandb_project,
        seed=args.seed,
    )
