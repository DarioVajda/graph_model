"""
Expressiveness experiment — refactor-validation harness (entry point).

Runs the HARD connectivity task across the v0 and v2 GTLM implementations on a
single shared `.gtds` dataset (the `TextGraphDataset` is implementation-agnostic;
only the collator / forward contract / trainer differ), to confirm:

  (a) the refactored v2 model still learns the task — accuracy parity with v0;
  (b) it is faster / leaner at scale — throughput + peak memory + sparsity.

Selectable implementations (``--impls``):
    v0-eager   GraphLlamaForCausalLM (legacy)  + GraphCollator    + GraphTrainer
    v2-eager   GTLMLlamaForCausalLM            + GraphCollatorV2   + GraphTrainerV2
    v2-flex        "             (pad_to_block) +     "            +     "

Modes (``--mode``):
    train  small graphs, train to convergence  -> accuracy / parity
           (each run also records live ms/step + peak CUDA memory, and token /
            block sparsity is measured standalone on a sampled subset)
    bench  large graphs, few-step throughput + peak CUDA memory + sparsity

Every knob is an argparse flag; see ``--help``. This file is a thin dispatcher —
the implementation lives in the sibling modules (config / dispatch / evaluation /
datasets / instrumentation / train / bench).
"""

import argparse

from transformers import AutoTokenizer

from ...utils import set_wandb_project
from .config import RunConfig
from .train import run_train_mode
from .bench import run_bench_mode


def _int_list(raw):
    """Parse a comma-separated list of ints (argparse type)."""
    return [int(s) for s in raw.split(",") if s.strip()]


def _str_list(raw):
    """Parse a comma-separated list of strings (argparse type)."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_parser():
    defaults = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.expressiveness",
        description="Expressiveness / refactor-validation harness for the GTLM graph bias.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── What to run ──────────────────────────────────────────────────────────
    p.add_argument("--mode", choices=("train", "bench"), default=defaults.mode,
                   help="train (small graphs -> accuracy/parity) | bench (large graphs -> throughput).")
    p.add_argument("--impls", type=_str_list, default=list(defaults.impls),
                   help="Comma-separated implementations: v0-eager,v2-eager,v2-flex.")
    p.add_argument("--k-hops", type=_int_list, default=list(defaults.k_hops),
                   help="Comma-separated k-hop radii to sweep (v2 only; v0 is dense, run once).")
    p.add_argument("--k-hop-directed", action="store_true", default=defaults.k_hop_directed,
                   help="Respect edge direction in the k-hop mask (HARD graphs are bidirectional).")
    p.add_argument("--difficulty", choices=("HARD", "EASY"), default=defaults.difficulty,
                   help="Task variant.")
    p.add_argument("--report-to", dest="wandb_project", default=defaults.wandb_project,
                   help="wandb project to report to. Omit for no experiment tracking.")
    p.add_argument("--flex-compile-mode", default=defaults.flex_compile_mode,
                   help="torch.compile mode for the flex kernel ('default' for quick iteration).")
    p.add_argument("--model-name", default=defaults.model_name, help="Backbone model.")
    p.add_argument("--magnetic-m", type=int, default=defaults.magnetic_m,
                   help="Magnetic-Laplacian eigenvectors kept (0 -> all N).")

    # ── graph size ───────────────────────────────────────────────────────────
    p.add_argument("--num-nodes", type=int, default=defaults.num_nodes,
                   help="Graph size. In train mode it sets the node range (0.8x-1.2x); "
                        "in bench mode it is the single fixed size.")
    p.add_argument("--min-nodes", type=int, default=None,
                   help="Override the lower node bound (defaults to round(0.8*num_nodes)).")
    p.add_argument("--max-nodes", type=int, default=None,
                   help="Override the upper node bound (defaults to round(1.2*num_nodes)).")

    # ── train mode ───────────────────────────────────────────────────────────
    p.add_argument("--train-dataset-size", type=int, default=defaults.train_dataset_size)
    p.add_argument("--eval-dataset-size", type=int, default=defaults.eval_dataset_size)
    p.add_argument("--ordering", choices=("rcm", "original"), default=defaults.ordering,
                   help="Node ordering: rcm (block-locality) | original (baseline).")
    p.add_argument("--len-buckets", type=_int_list, default=defaults.len_buckets,
                   help="Comma-separated flex L padding ladder (each a multiple of the block size).")
    p.add_argument("--node-buckets", type=_int_list, default=defaults.node_buckets,
                   help="Comma-separated flex N padding ladder.")
    p.add_argument("--lr", type=float, default=defaults.lr, help="Backbone LR (backbone is frozen).")
    p.add_argument("--bias-lr", type=float, default=defaults.bias_lr,
                   help="LR for the trainable graph-bias params (the one that matters).")
    p.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=defaults.accumulation_steps)
    p.add_argument("--eval-steps", type=int, default=defaults.eval_steps)
    p.add_argument("--seeds", type=_int_list, default=list(defaults.seeds),
                   help="Comma-separated seeds to sweep.")
    p.add_argument("--max-steps", type=int, default=defaults.max_steps,
                   help=">0 caps optimizer steps (quick smoke tests).")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    p.add_argument("--lora", action="store_true", default=defaults.lora,
                   help="Train LoRA adapters on the backbone alongside the graph bias. "
                        "Adapters train at --lr (set it ~1e-4); bias stays at --bias-lr.")
    p.add_argument("--lora-r", type=int, default=defaults.lora_r, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha, help="LoRA alpha.")
    p.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout, help="LoRA dropout.")

    # ── density measurement ──────────────────────────────────────────────────
    p.add_argument("--no-measure-density", dest="measure_density", action="store_false",
                   default=defaults.measure_density, help="Skip the standalone sparsity probe.")
    p.add_argument("--density-sample-graphs", type=int, default=defaults.density_sample_graphs)
    p.add_argument("--density-sample-batches", type=int, default=defaults.density_sample_batches)

    # ── bench mode ───────────────────────────────────────────────────────────
    p.add_argument("--bench-batch-size", type=int, default=defaults.bench_batch_size)
    p.add_argument("--bench-num-warmup", type=int, default=defaults.bench_num_warmup)
    p.add_argument("--bench-num-iters", type=int, default=defaults.bench_num_iters)
    p.add_argument("--bench-num-examples", type=int, default=defaults.bench_num_examples)
    return p


def config_from_args(args):
    return RunConfig(
        mode=args.mode,
        impls=tuple(args.impls),
        k_hops=tuple(args.k_hops),
        k_hop_directed=args.k_hop_directed,
        difficulty=args.difficulty,
        wandb_project=args.wandb_project,
        flex_compile_mode=args.flex_compile_mode,
        model_name=args.model_name,
        magnetic_m=args.magnetic_m,
        num_nodes=args.num_nodes,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        train_dataset_size=args.train_dataset_size,
        eval_dataset_size=args.eval_dataset_size,
        ordering=args.ordering,
        len_buckets=tuple(args.len_buckets) if args.len_buckets is not None else None,
        node_buckets=tuple(args.node_buckets) if args.node_buckets is not None else None,
        lr=args.lr,
        bias_lr=args.bias_lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps,
        seeds=tuple(args.seeds),
        max_steps=args.max_steps,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        measure_density=args.measure_density,
        density_sample_graphs=args.density_sample_graphs,
        density_sample_batches=args.density_sample_batches,
        bench_batch_size=args.bench_batch_size,
        bench_num_warmup=args.bench_num_warmup,
        bench_num_iters=args.bench_num_iters,
        bench_num_examples=args.bench_num_examples,
    )


if __name__ == "__main__":
    cfg = config_from_args(build_parser().parse_args())

    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if cfg.mode == "train":
        run_train_mode(cfg, tokenizer, pad_token_id)
    elif cfg.mode == "bench":
        run_bench_mode(cfg)
    else:
        raise ValueError(f"Unknown mode='{cfg.mode}' (expected 'train' or 'bench').")
