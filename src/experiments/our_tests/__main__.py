"""
our_tests experiment (Family Tree + Knowledge-Graph QA) — standalone single-run entry point.

A self-contained argparse program: given parameters for **one** configuration it runs
that configuration. It knows nothing about sweeps or job submission — the generic
``sweep`` runner drives those and invokes this program once per resolved config
(rendering each config key to the matching flag below):

    python3 -m sweep src.experiments.our_tests src/experiments/our_tests/configs/003_question_node.jsonc

Run it directly for a single config / quick iteration:

    python3 -m src.experiments.our_tests --task family
    python3 -m src.experiments.our_tests --task family --question-node isolated
    python3 -m src.experiments.our_tests --no-magnetic --seed 43     # an ablation arm
    python3 -m src.experiments.our_tests --max-steps 4 --num-epochs 1  # smoke test
    python3 -m src.experiments.our_tests --mode data_prep --task kg_qa
    python3 -m src.experiments.our_tests --init my_sweep             # write a sweep template

``--mode`` routes within the experiment: ``train`` (default) trains one config and logs
one record; ``data_prep`` builds that config's dataset splits.

NOTE: data prep here generates thousands of graphs and eigendecomposes each one — it is
not a login-node job. Submit it through the sweep runner's sbatch mode.
"""

import argparse
import os

from .config import RunConfig, TASKS, IMPLS, DTYPES, QUESTION_NODES

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // our_tests sweep config (JSONC: // comments + trailing commas allowed).
  // Run with:  python3 -m sweep src.experiments.our_tests <this file>
  //
  // Expansion rules (how a value becomes runs):
  //   scalar            -> fixed in every run
  //   [a, b, c]         -> a sweep AXIS (one run per value; cartesian with others)
  //   [ {..}, {..} ]    -> a BUNDLE: params that vary TOGETHER; each object's keys
  //                        flatten into the run, and the bundle's label disappears.
  // A key may be defined in exactly one place (top-level OR one bundle).
  // Keys map 1:1 to this experiment's CLI flags (some_key -> --some-key).
  //
  // TWO-STEP WORKFLOW: run once with "mode": "data_prep" to build the datasets for
  // every (task, question_node) this file references, then again with "mode": "train".
  // Data prep is EXPENSIVE here (generation + an eigendecomposition per graph) and
  // wants a GPU node — always sbatch it.
  // ─────────────────────────────────────────────────────────────────────────

  "name": "my_sweep",                 // results land in <results_dir>/<name>/
  "results_dir": "src/experiments/our_tests/results",

  "execution": {
    "mode": "local",                  // "local" (sequential, this process) | "sbatch"
    "sbatch": {                       // only used when mode == "sbatch"
      "granularity": "per_config",    // "single" (one job, in sequence) | "per_config"
      "max_concurrent": 4,            // cap concurrent jobs (Slurm array %N); omit for no cap
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B300:1",               // --gres=gpu:<gpus>
      "cpus": 8,
      "mem": "64G",
      "time": "12:00:00",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh",
      // "dry_run": true,             // write sbatch_commands.sh but don't submit
    }
  },

  "mode": "train",                    // "train" | "data_prep"

  // ── sweep axes (lists => swept) ───────────────────────────────────────────
  "task": ["family", "kg_qa"],
  "seed": [42, 43, 44],

  // ── the encoding probe / ablation arms ────────────────────────────────────
  "question_node": "off",             // "off" | "isolated"
  "spd": true,
  "rrwp": true,
  "magnetic": true,

  // ── fixed scalars (the paper recipe) ──────────────────────────────────────
  "model_name": "meta-llama/Llama-3.2-1B",
  "impl": "v2-eager",                 // "v2-eager" | "v2-flex"
  "dtype": "fp32",                    // "bf16" is a numerical change, not a free speedup
  "k_hop": 0,                         // 0 disables the k-hop attention gate
  "lora": true,
  "lora_r": 32,
  "num_epochs": 10,
  "batch_size": 4,
  "accumulation_steps": 4,
  "lr": 5e-5,
  "bias_lr": 1e-2,
  "eval_steps": 40,
  "max_steps": -1,                    // >0 caps optimizer steps (quick smoke tests)
  "max_spd": 8,
  "max_rw_steps": 16,
  "magnetic_dim": 32,
  "magnetic_q": 0.25,
  "magnetic_m": 0,                    // # magnetic eigenvectors (0 = all N)

  "wandb_project": null               // e.g. "GraphLLM"; null = no tracking
}
"""


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.our_tests",
        description="Run ONE our_tests configuration (the sweep runner invokes this per config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="template", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    p.add_argument("--mode", choices=("train", "data_prep"), default=d.mode,
                   help="train one config | build that config's dataset splits.")

    # ── what to run ────────────────────────────────────────────────────────────
    p.add_argument("--task", choices=TASKS, default=d.task,
                   help="'family' (family-tree QA) | 'kg_qa' (knowledge-graph QA).")
    p.add_argument("--question-node", choices=QUESTION_NODES, default=d.question_node,
                   help="'off' (question in the prompt node) | 'isolated' (question in its "
                        "own edge-free prefix node, so graph tokens attend to it).")

    # ── model ──────────────────────────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--impl", choices=IMPLS, default=d.impl,
                   help="v2-eager is the implementation pinned equivalent to the legacy "
                        "v0 path these datasets were trained with.")
    p.add_argument("--flex-compile-mode", default=d.flex_compile_mode)
    p.add_argument("--dtype", choices=DTYPES, default=d.dtype,
                   help="fp32 is the dtype the v0<->v2 parity is proven at.")

    # ── k-hop gate ─────────────────────────────────────────────────────────────
    p.add_argument("--k-hop", type=int, default=d.k_hop, help="K-hop attention gate (0 disables).")
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)

    # ── graph-bias features (the ablation axes) ────────────────────────────────
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--max-rw-steps", type=int, default=d.max_rw_steps)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim,
                   help="model bias-MLP hidden width.")
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q,
                   help="magnetic-Laplacian charge (a data-prep knob: part of the cache key).")
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m,
                   help="# magnetic eigenvectors kept (0 = all N).")
    p.add_argument("--laplacian", action=B, default=d.laplacian)
    p.add_argument("--rwse", action=B, default=d.rwse)

    # ── dataset ────────────────────────────────────────────────────────────────
    p.add_argument("--max-length", type=int, default=d.max_length,
                   help="per-node token cap (a data-prep knob).")
    p.add_argument("--use-gpu", action=B, default=d.use_gpu,
                   help="build SPD/RRWP/magnetic features on GPU (data prep).")

    # ── data generation (data_prep mode only) ──────────────────────────────────
    p.add_argument("--gen-seed", type=int, default=d.gen_seed,
                   help="seed for dataset generation (distinct from the training seed).")
    p.add_argument("--family-train", type=int, default=d.family_train)
    p.add_argument("--family-val", type=int, default=d.family_val)
    p.add_argument("--family-test", type=int, default=d.family_test)
    p.add_argument("--family-generations", type=int, default=d.family_generations)
    p.add_argument("--family-marriage-prob", type=float, default=d.family_marriage_prob)
    p.add_argument("--family-child-prob", type=float, default=d.family_child_prob)
    p.add_argument("--kgqa-train", type=int, default=d.kgqa_train)
    p.add_argument("--kgqa-val", type=int, default=d.kgqa_val)
    p.add_argument("--kgqa-test", type=int, default=d.kgqa_test)
    p.add_argument("--kgqa-min-nodes", type=int, default=d.kgqa_min_nodes)
    p.add_argument("--kgqa-max-nodes", type=int, default=d.kgqa_max_nodes)

    # ── LoRA ───────────────────────────────────────────────────────────────────
    p.add_argument("--lora", action=B, default=d.lora)
    p.add_argument("--lora-r", type=int, default=d.lora_r,
                   help="LoRA rank; alpha is always 2*r (--no-lora disables LoRA).")
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)

    # ── training schedule ──────────────────────────────────────────────────────
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps,
                   help=">0 caps optimizer steps (quick tests).")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--gradient-checkpointing", action=B, default=d.gradient_checkpointing)
    p.add_argument("--include-f1", action=B, default=d.include_f1,
                   help="also log macro-F1 alongside exact match.")

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
        mode=args.mode, task=args.task, question_node=args.question_node,
        model_name=args.model_name, impl=args.impl,
        flex_compile_mode=args.flex_compile_mode, dtype=args.dtype,
        k_hop=args.k_hop, k_hop_directed=args.k_hop_directed,
        spd=args.spd, max_spd=args.max_spd,
        rrwp=args.rrwp, max_rw_steps=args.max_rw_steps,
        magnetic=args.magnetic, magnetic_dim=args.magnetic_dim,
        magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        laplacian=args.laplacian, rwse=args.rwse,
        max_length=args.max_length, use_gpu=args.use_gpu,
        gen_seed=args.gen_seed,
        family_train=args.family_train, family_val=args.family_val,
        family_test=args.family_test, family_generations=args.family_generations,
        family_marriage_prob=args.family_marriage_prob,
        family_child_prob=args.family_child_prob,
        kgqa_train=args.kgqa_train, kgqa_val=args.kgqa_val, kgqa_test=args.kgqa_test,
        kgqa_min_nodes=args.kgqa_min_nodes, kgqa_max_nodes=args.kgqa_max_nodes,
        lora=args.lora, lora_r=args.lora_r, lora_dropout=args.lora_dropout,
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
          f"Edit it, then run:  python3 -m sweep src.experiments.our_tests {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)

    if cfg.mode == "data_prep":
        from .data import run_data_prep_mode
        run_data_prep_mode(cfg)
        return 0

    from .train import run_train_mode
    run_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                   sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
