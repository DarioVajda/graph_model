"""
Molecules experiment — standalone single-run entry point.

One resolved configuration per invocation; the generic ``sweep`` runner drives
many. The structure of this file IS the sweep contract (see
``src/experiments/template/README.md``): every RunConfig field has exactly one
flag, booleans use ``BooleanOptionalAction``, lists comma-split, and the runner's
``--runs-jsonl`` / ``--run-name`` / ``--sweep-id`` are always accepted.

    python3 -m src.experiments.molecules --task ring_membership --arm graph
    python3 -m src.experiments.molecules --arm flat --bias none --max-steps 4
    python3 -m sweep src.experiments.molecules src/experiments/molecules/configs/000_smoke.jsonc
"""

import argparse
import os

from .config import RunConfig
from .data import ENCODINGS, QUESTION_NODE_MODES
from .dataset import ALL_TASKS, ARMS, run_data_prep_mode
from .train import run_train_mode

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // Molecules sweep config (JSONC). Run with:
  //   python3 -m sweep src.experiments.molecules <this file>
  //
  //   scalar        -> fixed in every run
  //   [a, b]        -> a sweep AXIS
  //   [{..}, {..}]  -> a BUNDLE of keys that vary together
  //
  // NOTE: the flat arm requires "bias": "none" (a single-node graph has no
  // structure for a bias to read), so arm and bias must move together as a
  // bundle rather than as two independent axes.

  "name": "my_sweep",
  "results_dir": "src/experiments/molecules/results",

  "execution": {
    "mode": "sbatch",
    "sbatch": {
      "granularity": "per_config",
      "max_concurrent": 4,
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B200:1",
      "cpus": 16,
      "mem": "64G",
      "time": "04:00:00",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
    }
  },

  "arm_bias": [
    { "arm": "graph", "bias": "spd+magnetic" },
    { "arm": "graph", "bias": "none" },
    { "arm": "flat",  "bias": "none" }
  ],

  "task": ["ring_membership"],
  "seed": [0, 1, 2],

  "encoding": "rich_levi",
  "model_name": "meta-llama/Llama-3.2-1B",
  "train_size": 4000,
  "val_size": 500,
  "test_size": 1000,
  "num_epochs": 20,
  "batch_size": 4,
  "accumulation_steps": 8,
  "lr": 1e-5,
  "bias_lr": 1e-3,
  "eval_steps": 100,
  "lora": true,
  "lora_r": 8
}
"""


def _str_list(raw):
    return [s.strip() for s in raw.split(",") if s.strip()]


def _int_list(raw):
    return [int(s) for s in raw.split(",") if str(s).strip()]


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.molecules",
        description="Run ONE molecules configuration (the sweep runner invokes this per config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="molecules", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    p.add_argument("--mode", choices=("train", "data_prep", "eval"), default=d.mode)
    p.add_argument("--checkpoint", default=d.checkpoint,
                   help="mode 'eval': a trained checkpoint dir to re-score and analyse.")
    p.add_argument("--expect-accuracy", type=float, default=None,
                   help="mode 'eval': the accuracy the training run recorded. The "
                        "reload is checked against it (project-load-best-model-bias-bug).")

    # ── the experiment's axes ──────────────────────────────────────────────────
    p.add_argument("--task", choices=ALL_TASKS, default=d.task,
                   help="a Tier-A generator name or a Tier-B corpus name.")
    p.add_argument("--arm", choices=ARMS, default=d.arm,
                   help="graph = atoms + Levi bonds; flat = single-node SMILES (the control).")
    p.add_argument("--encoding", choices=ENCODINGS, default=d.encoding,
                   help="graph arm only; 'terse_atom_only' is rejected by construction.")
    p.add_argument("--stereo-tags", action=B, default=d.stereo_tags,
                   help="parity tag in atom text. Never the CIP label — see PLAN.md §1.")
    p.add_argument("--bias", default=d.bias,
                   help="'+'-joined bias arm ('none', 'spd', 'spd+magnetic', ...).")
    p.add_argument("--k-hop", type=int, default=d.k_hop)
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)
    p.add_argument("--question-node", choices=QUESTION_NODE_MODES, default=d.question_node,
                   help="'on' (default, and settled) = own edge-free prefix node, so the graph "
                        "attends to the question; 'off' = inside the prompt node, which leaves "
                        "the prefix query-blind. Do not move without a concrete reason.")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--held-out-eval", action=B, default=d.held_out_eval,
                   help="build a permanently held-out task for EVALUATION only (PLAN.md §4.1).")

    # ── model / bias architecture ──────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--impl", choices=("v2-flex", "v2-eager"), default=d.impl)
    p.add_argument("--flex-compile-mode", default=d.flex_compile_mode)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)

    # ── dataset ────────────────────────────────────────────────────────────────
    p.add_argument("--pool", type=_str_list, default=list(d.pool),
                   help="comma-separated Tier-B corpora supplying molecules.")
    p.add_argument("--train-size", type=int, default=d.train_size)
    p.add_argument("--val-size", type=int, default=d.val_size)
    p.add_argument("--test-size", type=int, default=d.test_size)
    p.add_argument("--max-train-examples", type=int, default=d.max_train_examples,
                   help="Tier B only: 0 = the whole scaffold split.")
    p.add_argument("--max-eval-examples", type=int, default=d.max_eval_examples)
    p.add_argument("--data-seed", type=int, default=d.data_seed)
    p.add_argument("--ordering", choices=("rcm", "original"), default=d.ordering)
    p.add_argument("--len-buckets", type=_int_list, default=d.len_buckets)
    p.add_argument("--node-buckets", type=_int_list, default=d.node_buckets)

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
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--gradient-checkpointing", action=B, default=d.gradient_checkpointing)

    # ── measurement ────────────────────────────────────────────────────────────
    p.add_argument("--measure-density", action=B, default=d.measure_density)
    p.add_argument("--density-sample-graphs", type=int, default=d.density_sample_graphs)
    p.add_argument("--density-sample-batches", type=int, default=d.density_sample_batches)

    # ── tracking ───────────────────────────────────────────────────────────────
    p.add_argument("--wandb-project", default=d.wandb_project)

    # ── sweep-runner bookkeeping ───────────────────────────────────────────────
    p.add_argument("--runs-jsonl", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--sweep-id", default=None)
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args."""
    return RunConfig(
        mode=args.mode, checkpoint=args.checkpoint,
        task=args.task, arm=args.arm, encoding=args.encoding,
        stereo_tags=args.stereo_tags, bias=args.bias,
        k_hop=args.k_hop, k_hop_directed=args.k_hop_directed, seed=args.seed,
        question_node=args.question_node,
        held_out_eval=args.held_out_eval,
        model_name=args.model_name, impl=args.impl,
        flex_compile_mode=args.flex_compile_mode,
        max_spd=args.max_spd, magnetic_dim=args.magnetic_dim,
        magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        pool=tuple(args.pool),
        train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        data_seed=args.data_seed, ordering=args.ordering,
        len_buckets=tuple(args.len_buckets) if args.len_buckets else None,
        node_buckets=tuple(args.node_buckets) if args.node_buckets else None,
        lora=args.lora, lora_r=args.lora_r, lora_dropout=args.lora_dropout,
        lr=args.lr, bias_lr=args.bias_lr, num_epochs=args.num_epochs,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
        eval_steps=args.eval_steps, max_steps=args.max_steps,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
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
          f"Edit it, then run:  python3 -m sweep src.experiments.molecules {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)
    if cfg.mode == "data_prep":
        run_data_prep_mode(cfg)
        return 0

    if cfg.mode == "eval":
        from .evaluate_checkpoint import run_eval_mode   # deferred: heavy imports
        run_eval_mode(cfg, run_name=args.run_name, runs_jsonl=args.runs_jsonl,
                      sweep_meta={"sweep_id": args.sweep_id} if args.sweep_id else None,
                      expect_accuracy=args.expect_accuracy)
        return 0

    from transformers import AutoTokenizer   # deferred: keep --help/--init fast
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id)
    run_train_mode(cfg, tokenizer, pad_token_id,
                   runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                   sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
