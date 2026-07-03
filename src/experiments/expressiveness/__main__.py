"""
Expressiveness experiment — standalone single-run entry point.

This is a self-contained argparse program: given parameters for **one**
configuration it runs that configuration. It knows nothing about sweeps or job
submission — the generic ``sweep`` runner drives those and invokes this program
once per resolved config (rendering each config to the flags below):

    python3 -m sweep src.experiments.expressiveness configs/my_sweep.jsonc

Run it directly for a single config / quick iteration:

    python3 -m src.experiments.expressiveness --impl v2-flex --k-hop 0 --seed 0 --num-nodes 500
    python3 -m src.experiments.expressiveness --mode data_prep --num-nodes 500 --magnetic-m 128
    python3 -m src.experiments.expressiveness --init my_sweep      # write a sweep template

``--mode`` routes within the experiment: ``train`` (default) trains one config;
``data_prep`` just builds that config's datasets. Benchmarking is a separate
program: ``python3 -m src.experiments.expressiveness.bench``.
"""

import argparse
import os
import sys

from transformers import AutoTokenizer

from ...utils import set_wandb_project
from .config import RunConfig
from .training.train import run_train_mode
from .data.prep import run_data_prep_mode

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // Expressiveness sweep config (JSONC: // comments + trailing commas allowed).
  // Run with:  python3 -m sweep src.experiments.expressiveness <this file>
  //
  // Expansion rules (how a value becomes runs):
  //   scalar            -> fixed in every run
  //   [a, b, c]         -> a sweep AXIS (one run per value; cartesian with others)
  //   [[..], [..]]      -> a list-valued axis (e.g. several bucket ladders)
  //   [ {..}, {..} ]    -> a BUNDLE: params that vary TOGETHER; each object's keys
  //                        flatten into the run, and the bundle's label disappears.
  // A key may be defined in exactly one place (top-level OR one bundle).
  // Keys map 1:1 to the experiment's CLI flags (singular: impl / k_hop / seed).
  // ─────────────────────────────────────────────────────────────────────────

  "name": "my_sweep",                 // results land in <results_dir>/<name>/
  "results_dir": "src/experiments/expressiveness/results",

  "execution": {
    "mode": "local",                  // "local" | "sbatch"
    "sbatch": {                       // only used when mode == "sbatch"
      "granularity": "per_config",    // "single" (one job, in sequence) | "per_config"
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B200:1",               // --gres=gpu:<gpus>
      "cpus": 16,
      "mem": "64G",
      "time": "24:00:00",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh",
      // "nodelist": "ixb3",          // optional -w
      // "dry_run": true,             // write sbatch_commands.sh but don't submit
    }
  },

  // ── sweep axes (lists => swept) ───────────────────────────────────────────
  "mode": "train",                    // "train" | "data_prep" (run once in data_prep to build datasets)
  "impl":  ["v2-flex"],               // v0-eager | v2-eager | v2-flex
  "k_hop": [0, 1],
  "seed":  [0, 1, 2],

  // ── a size bundle (num_nodes + its buckets + micro-batch vary together) ───
  "size_profile": [
    { "num_nodes": 500,  "len_buckets": [640, 768],          "node_buckets": [512, 640],  "batch_size": 4, "accumulation_steps": 8 },
    { "num_nodes": 1000, "len_buckets": [1408, 1664, 1792],  "node_buckets": [1024, 1280], "batch_size": 4, "accumulation_steps": 8 }
  ],

  // ── fixed scalars ─────────────────────────────────────────────────────────
  "difficulty": "HARD",
  "ordering": "rcm",
  "magnetic_m": 128,
  "train_dataset_size": 2000,
  "eval_dataset_size": 200,
  "num_epochs": 20,
  "eval_steps": 75,
  "bias_lr": 1e-3,
  "lr": 1e-5,
  "lora": true,
  "lora_r": 8,
  "flex_compile_mode": "max-autotune-no-cudagraphs",   // "default" for quick iteration
  "wandb_project": null,              // e.g. "GraphLLM"; null = no tracking

  // ── graph-bias features (laplacian/rwse/rrwp are rejected by this experiment) ─
  "spd": true,
  "magnetic": true
}
"""


def _int_list(raw):
    return [int(s) for s in raw.split(",") if str(s).strip()]


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.expressiveness",
        description="Run ONE expressiveness configuration (the sweep runner invokes this per config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="template", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    p.add_argument("--mode", choices=("train", "data_prep"), default="train",
                   help="train one config | build that config's datasets.")

    # ── sweep axes (singular) ────────────────────────────────────────────────
    p.add_argument("--impl", default=d.impls[0], help="v0-eager | v2-eager | v2-flex.")
    p.add_argument("--k-hop", type=int, default=d.k_hops[0])
    p.add_argument("--seed", type=int, default=d.seeds[0])
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)
    p.add_argument("--difficulty", choices=("HARD", "EASY"), default=d.difficulty)

    # ── graph size ───────────────────────────────────────────────────────────
    p.add_argument("--num-nodes", type=int, default=d.num_nodes)
    # Default to None so RunConfig.__post_init__ derives the range from --num-nodes
    # (0.8x-1.2x). Using d.min_nodes/d.max_nodes here would bake in the 400-600 range
    # of the default num_nodes=500 and suppress the derivation for every other size.
    p.add_argument("--min-nodes", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=None)

    # ── data / spectral ──────────────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)
    p.add_argument("--train-dataset-size", type=int, default=d.train_dataset_size)
    p.add_argument("--eval-dataset-size", type=int, default=d.eval_dataset_size)
    p.add_argument("--ordering", choices=("rcm", "original"), default=d.ordering)
    p.add_argument("--len-buckets", type=_int_list, default=d.len_buckets)
    p.add_argument("--node-buckets", type=_int_list, default=d.node_buckets)

    # ── bias features ────────────────────────────────────────────────────────
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--laplacian", action=B, default=d.laplacian)
    p.add_argument("--rwse", action=B, default=d.rwse)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)

    # ── training schedule ────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--num-workers", type=int, default=d.num_workers,
                   help="DataLoader worker processes (0 = synchronous; >0 overlaps feature build with compute).")
    p.add_argument("--flex-compile-mode", default=d.flex_compile_mode)
    p.add_argument("--wandb-project", default=d.wandb_project,
                   help="wandb project to report to (omit for no tracking).")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    p.add_argument("--lora", action=B, default=d.lora)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)

    # ── density measurement ──────────────────────────────────────────────────
    p.add_argument("--measure-density", action=B, default=d.measure_density)
    p.add_argument("--density-sample-graphs", type=int, default=d.density_sample_graphs)
    p.add_argument("--density-sample-batches", type=int, default=d.density_sample_batches)

    # ── sweep-runner bookkeeping (where to log this run) ─────────────────────
    p.add_argument("--runs-jsonl", default=None, help="(runner) JSONL to append this run's record to.")
    p.add_argument("--run-name", default=None, help="(runner) this run's name within the sweep.")
    p.add_argument("--sweep-id", default=None, help="(runner) the sweep this run belongs to.")
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args (singular -> 1-tuples)."""
    return RunConfig(
        mode=args.mode,
        impls=(args.impl,), k_hops=(args.k_hop,), seeds=(args.seed,),
        k_hop_directed=args.k_hop_directed, difficulty=args.difficulty,
        wandb_project=args.wandb_project, flex_compile_mode=args.flex_compile_mode,
        model_name=args.model_name, magnetic_m=args.magnetic_m,
        num_nodes=args.num_nodes, min_nodes=args.min_nodes, max_nodes=args.max_nodes,
        train_dataset_size=args.train_dataset_size, eval_dataset_size=args.eval_dataset_size,
        ordering=args.ordering,
        len_buckets=tuple(args.len_buckets) if args.len_buckets else None,
        node_buckets=tuple(args.node_buckets) if args.node_buckets else None,
        spd=args.spd, magnetic=args.magnetic, laplacian=args.laplacian,
        rwse=args.rwse, rrwp=args.rrwp,
        max_spd=args.max_spd, magnetic_dim=args.magnetic_dim, magnetic_q=args.magnetic_q,
        lr=args.lr, bias_lr=args.bias_lr, num_epochs=args.num_epochs,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps, max_steps=args.max_steps,
        num_workers=args.num_workers,
        lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        measure_density=args.measure_density,
        density_sample_graphs=args.density_sample_graphs,
        density_sample_batches=args.density_sample_batches,
    ).validate()


def _do_init(name):
    if not (name.endswith(".json") or name.endswith(".jsonc")):
        name += ".jsonc"
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    path = os.path.join(CONFIGS_DIR, name)
    with open(path, "w") as f:
        f.write(TEMPLATE)
    print(f"Wrote sweep template to {path}\n"
          f"Edit it, then run:  python3 -m sweep src.experiments.expressiveness {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    if args.mode == "data_prep":
        run_data_prep_mode(cfg)
        return 0

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    run_train_mode(cfg, tokenizer, pad_token_id,
                   runs_jsonl=args.runs_jsonl, run_name=args.run_name, sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
