"""
Probe suite — standalone single-run entry point.

A self-contained argparse program: given parameters for **one** (task, bias
arm, k_hop, seed) configuration it runs that configuration. The generic
``sweep`` runner drives the full suite and invokes this program once per
resolved config:

    python3 -m sweep src.experiments.probes src/experiments/probes/configs/suite.jsonc

Run it directly for a single config / quick iteration:

    python3 -m src.experiments.probes --task direction --bias spd --k-hop 0 --seed 0
    python3 -m src.experiments.probes --task substructure --max-steps 4 --num-epochs 1   # smoke test
    python3 -m src.experiments.probes --mode data_prep --task text_path                  # build datasets only
    python3 -m src.experiments.probes --init my_sweep                                    # write a sweep template
"""

import argparse
import os

from .config import RunConfig, TASKS
from .data import run_data_prep_mode
from .train import run_train_mode

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // Probe-suite sweep config (JSONC: // comments + trailing commas allowed).
  // Run with:  python3 -m sweep src.experiments.probes <this file>
  //
  //   scalar            -> fixed in every run
  //   [a, b, c]         -> a sweep AXIS (one run per value; cartesian with others)
  //   [ {..}, {..} ]    -> a BUNDLE: params that vary TOGETHER
  // ─────────────────────────────────────────────────────────────────────────

  "name": "suite",
  "results_dir": "src/experiments/probes/results",

  "execution": {
    "mode": "sbatch",
    "sbatch": {
      "granularity": "per_config",
      "max_concurrent": 8,
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B200:1",
      "cpus": 16,
      "mem": "64G",
      "time": "24:00:00",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh",
      // "dry_run": true,
    }
  },

  // ── the suite's axes (4 tasks x 6 arms x 2 k x 3 seeds = 144 runs) ───────
  "task": ["direction", "substructure", "local_hop", "text_path"],
  "bias": ["none", "spd", "magnetic", "spd+magnetic", "magnetic_shared", "spd+magnetic_shared"],
  "k_hop": [0, 3],
  "seed":  [0, 1, 2],

  // ── fixed recipe (README suite-wide conventions) ──────────────────────────
  "impl": "v2-flex",
  "num_epochs": 25,
  "eval_steps": 100,
  "lr": 1e-5,
  "bias_lr": 1e-3,
  "lora": true,
  "lora_r": 8,
  "wandb_project": null
}
"""


def _str_list(raw):
    """Comma-splitting type for list-valued flags (rule 3 of the sweep contract)."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _int_list(raw):
    return [int(s) for s in raw.split(",") if str(s).strip()]


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.probes",
        description="Run ONE probe configuration (the sweep runner invokes this per config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="probes", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    p.add_argument("--mode", choices=("train", "data_prep"), default=d.mode,
                   help="train one config | build that config's dataset.")

    # ── the suite's axes ───────────────────────────────────────────────────────
    p.add_argument("--task", choices=TASKS, default=d.task)
    p.add_argument("--bias", default=d.bias,
                   help="'+'-joined bias arm ('none', 'spd', 'spd+magnetic_shared', ...).")
    p.add_argument("--k-hop", type=int, default=d.k_hop, help="K-hop attention mask (0 disables).")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)

    # ── model / bias architecture ──────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--impl", choices=("v2-flex", "v2-eager"), default=d.impl)
    p.add_argument("--flex-compile-mode", default=d.flex_compile_mode)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)

    # ── dataset ────────────────────────────────────────────────────────────────
    p.add_argument("--train-size", type=int, default=d.train_size)
    p.add_argument("--val-size", type=int, default=d.val_size)
    p.add_argument("--test-size", type=int, default=d.test_size)
    p.add_argument("--data-seed", type=int, default=d.data_seed)
    p.add_argument("--min-nodes", type=int, default=None,
                   help="None -> the task's README node range.")
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--ordering", choices=("rcm", "original"), default=d.ordering)
    p.add_argument("--len-buckets", type=_int_list, default=d.len_buckets)
    p.add_argument("--node-buckets", type=_int_list, default=d.node_buckets)

    # ── task-specific generator knobs ──────────────────────────────────────────
    p.add_argument("--p-bidirectional", type=float, default=d.p_bidirectional)
    p.add_argument("--num-colors", type=int, default=d.num_colors)
    p.add_argument("--hop-radius", type=int, default=d.hop_radius)

    # ── LoRA ───────────────────────────────────────────────────────────────────
    p.add_argument("--lora", action=B, default=d.lora)
    p.add_argument("--lora-r", type=int, default=d.lora_r, help="alpha is always 2*r (derived).")
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)

    # ── training schedule ──────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps,
                   help=">0 caps optimizer steps (quick tests).")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--gradient-checkpointing", action=B, default=d.gradient_checkpointing)

    # ── density measurement ────────────────────────────────────────────────────
    p.add_argument("--measure-density", action=B, default=d.measure_density)
    p.add_argument("--density-sample-graphs", type=int, default=d.density_sample_graphs)
    p.add_argument("--density-sample-batches", type=int, default=d.density_sample_batches)

    # ── tracking ───────────────────────────────────────────────────────────────
    p.add_argument("--wandb-project", default=d.wandb_project,
                   help="wandb project to report to (omit for no tracking).")

    # ── sweep-runner bookkeeping (where to log this run) ───────────────────────
    p.add_argument("--runs-jsonl", default=None, help="(runner) JSONL to append this run's record to.")
    p.add_argument("--run-name", default=None, help="(runner) this run's name within the sweep.")
    p.add_argument("--sweep-id", default=None, help="(runner) the sweep this run belongs to.")
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args."""
    return RunConfig(
        mode=args.mode,
        task=args.task, bias=args.bias, k_hop=args.k_hop, seed=args.seed,
        k_hop_directed=args.k_hop_directed,
        model_name=args.model_name, impl=args.impl,
        flex_compile_mode=args.flex_compile_mode,
        max_spd=args.max_spd, magnetic_dim=args.magnetic_dim,
        magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
        data_seed=args.data_seed, min_nodes=args.min_nodes, max_nodes=args.max_nodes,
        ordering=args.ordering,
        len_buckets=tuple(args.len_buckets) if args.len_buckets else None,
        node_buckets=tuple(args.node_buckets) if args.node_buckets else None,
        p_bidirectional=args.p_bidirectional, num_colors=args.num_colors,
        hop_radius=args.hop_radius,
        lora=args.lora, lora_r=args.lora_r, lora_dropout=args.lora_dropout,
        lr=args.lr, bias_lr=args.bias_lr, num_epochs=args.num_epochs,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps, max_steps=args.max_steps,
        num_workers=args.num_workers, gradient_checkpointing=args.gradient_checkpointing,
        measure_density=args.measure_density,
        density_sample_graphs=args.density_sample_graphs,
        density_sample_batches=args.density_sample_batches,
        wandb_project=args.wandb_project,
    ).validate()


def _do_init(name):
    if not (name.endswith(".json") or name.endswith(".jsonc")):
        name += ".jsonc"
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    path = os.path.join(CONFIGS_DIR, name)
    with open(path, "w") as f:
        f.write(TEMPLATE)
    print(f"Wrote sweep template to {path}\n"
          f"Edit it, then run:  python3 -m sweep src.experiments.probes {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)
    if cfg.mode == "data_prep":
        run_data_prep_mode(cfg)
        return 0

    from transformers import AutoTokenizer  # deferred: keep --help/--init fast
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id)
    run_train_mode(cfg, tokenizer, pad_token_id,
                   runs_jsonl=args.runs_jsonl, run_name=args.run_name, sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
