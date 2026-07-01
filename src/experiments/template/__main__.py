"""
Template experiment — standalone single-run entry point.

A self-contained argparse program: given parameters for **one** configuration it
runs that configuration. It knows nothing about sweeps or job submission — the
generic ``sweep`` runner drives those and invokes this program once per resolved
config (rendering each config key to the matching flag below):

    python3 -m sweep src.experiments.template src/experiments/template/configs/my_sweep.jsonc

Run it directly for a single config / quick iteration:

    python3 -m src.experiments.template --k-hop 2 --seed 0
    python3 -m src.experiments.template --max-steps 4 --num-epochs 1   # smoke test
    python3 -m src.experiments.template --init my_sweep                # write a sweep template

Copy this whole directory to start a new experiment: edit ``config.py`` (the knobs),
``data.py`` (the dataset), and the ``TEMPLATE`` below (the sweep config), then keep
this file's structure — that structure IS the sweep contract.
"""

import argparse
import os

from .config import RunConfig
from .train import run_train_mode

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // Template sweep config (JSONC: // comments + trailing commas allowed).
  // Run with:  python3 -m sweep src.experiments.template <this file>
  //
  // Expansion rules (how a value becomes runs):
  //   scalar            -> fixed in every run
  //   [a, b, c]         -> a sweep AXIS (one run per value; cartesian with others)
  //   [ {..}, {..} ]    -> a BUNDLE: params that vary TOGETHER; each object's keys
  //                        flatten into the run, and the bundle's label disappears.
  // A key may be defined in exactly one place (top-level OR one bundle).
  // Keys map 1:1 to this experiment's CLI flags (some_key -> --some-key).
  // ─────────────────────────────────────────────────────────────────────────

  "name": "my_sweep",                 // results land in <results_dir>/<name>/
  "results_dir": "src/experiments/template/results",

  "execution": {
    "mode": "local",                  // "local" (sequential, this process) | "sbatch"
    "sbatch": {                       // only used when mode == "sbatch"
      "granularity": "per_config",    // "single" (one job, in sequence) | "per_config"
      "max_concurrent": 4,            // cap concurrent per_config jobs (Slurm array %N); omit for no cap
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B200:1",               // --gres=gpu:<gpus>
      "cpus": 16,
      "mem": "64G",
      "time": "12:00:00",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh",
      // "nodelist": "ixb3",          // optional -w
      // "dry_run": true,             // write sbatch_commands.sh but don't submit
    }
  },

  // ── sweep axes (lists => swept) ───────────────────────────────────────────
  "k_hop": [0, 2],
  "seed":  [0, 1, 2],

  // ── fixed scalars ─────────────────────────────────────────────────────────
  "model_name": "meta-llama/Llama-3.2-1B",
  "num_samples": 80,
  "num_epochs": 5,
  "batch_size": 2,
  "accumulation_steps": 4,
  "lr": 3e-4,
  "bias_lr": 5e-3,
  "eval_steps": 20,
  "lora": true,
  "lora_r": 16,
  "wandb_project": null,              // e.g. "GraphLLM"; null = no tracking

  // ── graph-bias features (laplacian/rwse are rejected by this experiment) ──
  "spd": true,
  "rrwp": true,
  "magnetic": true
}
"""


def _str_list(raw):
    """Comma-splitting type for list-valued flags (rule 3 of the sweep contract)."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.template",
        description="Run ONE template configuration (the sweep runner invokes this per config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="template", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    p.add_argument("--mode", choices=("train",), default=d.mode)

    # ── model + LoRA ───────────────────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--lora", action=B, default=d.lora)
    p.add_argument("--lora-r", type=int, default=d.lora_r, help="LoRA rank (--no-lora disables LoRA).")
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    p.add_argument("--active-params", type=_str_list, default=list(d.active_params),
                   help="Comma-separated parameter-name substrings to unfreeze.")

    # ── graph-bias features ────────────────────────────────────────────────────
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--max-rw-steps", type=int, default=d.max_rw_steps)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)
    p.add_argument("--laplacian", action=B, default=d.laplacian)
    p.add_argument("--rwse", action=B, default=d.rwse)

    # ── k-hop gate ─────────────────────────────────────────────────────────────
    p.add_argument("--k-hop", type=int, default=d.k_hop, help="K-hop attention gate (0 disables).")

    # ── dataset ────────────────────────────────────────────────────────────────
    p.add_argument("--num-samples", type=int, default=d.num_samples)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--data-seed", type=int, default=d.data_seed)

    # ── training schedule ──────────────────────────────────────────────────────
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps, help=">0 caps optimizer steps (quick tests).")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--gradient-checkpointing", action=B, default=d.gradient_checkpointing)
    p.add_argument("--include-f1", action=B, default=d.include_f1)
    p.add_argument("--wandb-project", default=d.wandb_project,
                   help="wandb project to report to (omit for no tracking).")

    # ── sweep-runner bookkeeping (where to log this run) ─────────────────────
    p.add_argument("--runs-jsonl", default=None, help="(runner) JSONL to append this run's record to.")
    p.add_argument("--run-name", default=None, help="(runner) this run's name within the sweep.")
    p.add_argument("--sweep-id", default=None, help="(runner) the sweep this run belongs to.")
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args."""
    return RunConfig(
        mode=args.mode,
        model_name=args.model_name,
        lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        active_params=tuple(args.active_params),
        spd=args.spd, max_spd=args.max_spd,
        rrwp=args.rrwp, max_rw_steps=args.max_rw_steps,
        magnetic=args.magnetic, magnetic_dim=args.magnetic_dim,
        magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        laplacian=args.laplacian, rwse=args.rwse,
        k_hop=args.k_hop,
        num_samples=args.num_samples, max_length=args.max_length, data_seed=args.data_seed,
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr, bias_lr=args.bias_lr, eval_steps=args.eval_steps,
        max_steps=args.max_steps, seed=args.seed, num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing, include_f1=args.include_f1,
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
          f"Edit it, then run:  python3 -m sweep src.experiments.template {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)
    run_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name, sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
