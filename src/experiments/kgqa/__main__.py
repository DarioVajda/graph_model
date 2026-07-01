"""
KGQA (SR-WebQSP) training entrypoint — new GTLM stack.

Trains GTLMLlamaForCausalLM (LoRA + graph bias) to generate the answer *set*
for a question given its retrieved KG subgraph, and selects the best checkpoint
on generative macro-F1 (see ``evaluate.KGQAGraphTrainer``).

Build the data once first:
    python -m src.experiments.kgqa.process_dataset
Then train:
    python -m src.experiments.kgqa --lora_r 16 --k_hop 2 --graph_attn_impl flex
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, TrainingArguments

from ...models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from ...utils import GraphCollatorV2
from ...utils.text_graph_trainer_v2 import make_compute_metrics, shift_logits_for_metrics
from ...train import select_active_params, print_trainable_parameters, save_run_metadata, get_device
from .load_data import load_data
from .evaluate import KGQAGraphTrainer

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "kgqa"

# Graph-attention bias configuration — must match what process_dataset built.
BIAS_PARAMS = {
    "spd":          True,
    "max_spd":      64,
    "magnetic":     True,
    "magnetic_dim": 128,
    "magnetic_q":   0.25,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train GTLM on SR-WebQSP (KGQA).")
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--bias_learning_rate", type=float, default=5e-3)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank (0 disables LoRA).")
    p.add_argument("--k_hop", type=int, default=2, help="K-hop attention gate (0 disables).")
    p.add_argument("--k_hop_directed", action="store_true")
    p.add_argument("--graph_attn_impl", default="flex", choices=["flex", "eager"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--gen_max_samples", type=int, default=None,
                   help="Cap generative-eval questions (None = full dev).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--wandb_project", default="GraphLLM", help="'none' to disable.")
    p.add_argument("--run_name", default=None)
    p.add_argument("--active_params", nargs="+", default=["graph_bias"])
    return p.parse_args()


def main():
    args = parse_args()
    wandb_project = None if args.wandb_project.lower() == "none" else args.wandb_project
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    } if args.lora_r > 0 else None

    run_name = args.run_name or f"kgqa_lora{args.lora_r}_khop{args.k_hop}_{args.graph_attn_impl}"
    run_name = save_run_metadata(
        run_name=run_name, experiment_dir=EXPERIMENT_DIR, base_model=args.model_name,
        active_params=args.active_params, lr=args.learning_rate, bias_lr=args.bias_learning_rate,
        lora_config=lora_config, num_epochs=args.num_epochs,
        bias_params=BIAS_PARAMS, k_hop=args.k_hop,
    )

    device = get_device()

    # ── Model (frozen backbone + graph bias), new GTLM stack ──────────────────
    config = GTLMLlamaConfig.from_pretrained(
        args.model_name, **BIAS_PARAMS,
        k_hop=args.k_hop, k_hop_directed=args.k_hop_directed, graph_attn_impl=args.graph_attn_impl,
    )
    model = GTLMLlamaForCausalLM.from_pretrained(
        args.model_name, config=config, graph_attn_impl=args.graph_attn_impl, torch_dtype=dtype)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    question_end = tokenizer("Answer:", add_special_tokens=False)["input_ids"]

    print("Loading data...")
    train_dataset, eval_dataset, test_dataset = load_data()
    print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}  Test: {len(test_dataset)}")

    magnetic_m = BIAS_PARAMS["magnetic_dim"] if BIAS_PARAMS.get("magnetic") else 0
    train_collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=args.k_hop, k_hop_directed=args.k_hop_directed,
        magnetic_m=magnetic_m, pad_to_block=(args.graph_attn_impl == "flex"))
    # Generation runs the dense decode path; keep its collator unbucketed.
    gen_collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=args.k_hop, k_hop_directed=args.k_hop_directed,
        magnetic_m=magnetic_m, pad_to_block=False)

    model = select_active_params(model, active_params=args.active_params, lora=lora_config)
    print_trainable_parameters(model)

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.accumulation_steps))
    total_steps = steps_per_epoch * args.num_epochs
    gc = not args.no_gradient_checkpointing

    training_args = TrainingArguments(
        num_train_epochs=args.num_epochs,
        output_dir=f"./checkpoints/{EXPERIMENT_NAME}/{run_name}",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gc else None,

        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.eval_every,
        metric_for_best_model="eval_f1",       # generative macro-F1 (see KGQAGraphTrainer)
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True,

        report_to="wandb" if wandb_project is not None else "none",
        run_name=run_name,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": args.learning_rate / 10},
        warmup_steps=total_steps // 10,
        weight_decay=0.1,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = KGQAGraphTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=train_collator,
        compute_metrics=make_compute_metrics(include_f1=False),       # cheap teacher-forced EM (secondary)
        preprocess_logits_for_metrics=shift_logits_for_metrics,       # bound eval memory
        active_params=args.active_params, bias_lr=args.bias_learning_rate,
        # generative (primary) eval:
        gen_tokenizer=tokenizer, gen_collator=gen_collator, question_end=question_end,
        gen_max_new_tokens=args.gen_max_new_tokens, gen_max_samples=args.gen_max_samples,
    )

    if wandb_project is not None:
        from ...utils import set_wandb_project
        set_wandb_project(wandb_project)

    trainer.train()

    print("\n" + "=" * 50 + "\nBest model — test set (generative):\n" + "=" * 50)
    results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    for k, v in results.items():
        if any(t in k for t in ("f1", "hits1", "hit_star", "loss", "accuracy")):
            print(f"  {k}: {v:.4f}")

    if wandb_project is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
