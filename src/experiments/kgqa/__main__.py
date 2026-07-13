"""
KGQA (SR-WebQSP) experiment — standalone single-run entry point.

A self-contained argparse program: given parameters for **one** configuration it
runs that configuration. It knows nothing about sweeps or job submission — the
generic ``sweep`` runner drives those and invokes this program once per resolved
config (rendering each config to the flags below):

    python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/my_sweep.jsonc

Run it directly for a single config / quick iteration:

    python3 -m src.experiments.kgqa --lora-r 16 --k-hop 2 --graph-attn-impl flex
    python3 -m src.experiments.kgqa --mode data_prep            # build this config's datasets
    python3 -m src.experiments.kgqa --init my_sweep             # write a sweep template

``--mode`` routes within the experiment: ``train`` (default) trains one config and
logs one record; ``data_prep`` just builds that config's ``.gtds`` datasets.
"""

import argparse
import os

from .config import RunConfig, DATASETS, REL_MODES, GRAPH_ATTN_IMPLS, DTYPES, PROMPT_STYLES

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // KGQA (SR-WebQSP) sweep config (JSONC: // comments + trailing commas allowed).
  // Run with:  python3 -m sweep src.experiments.kgqa <this file>
  //
  // Expansion rules (how a value becomes runs):
  //   scalar            -> fixed in every run
  //   [a, b, c]         -> a sweep AXIS (one run per value; cartesian with others)
  //   [[..], [..]]      -> a list-valued axis
  //   [ {..}, {..} ]    -> a BUNDLE: params that vary TOGETHER; each object's keys
  //                        flatten into the run, and the bundle's label disappears.
  // A key may be defined in exactly one place (top-level OR one bundle).
  // Keys map 1:1 to the experiment's CLI flags (singular: lora_r / k_hop / seed).
  //
  // TWO-STEP WORKFLOW: run this file once with "mode": "data_prep" to build the
  // .gtds cache for every data config it references, then again with "mode":
  // "train". A data_prep sweep is cheap (no GPU training) and idempotent.
  // ─────────────────────────────────────────────────────────────────────────

  "name": "my_sweep",                 // results land in <results_dir>/<name>/
  "results_dir": "src/experiments/kgqa/results",

  "execution": {
    "mode": "local",                  // "local" | "sbatch"
    "sbatch": {                       // only used when mode == "sbatch"
      "granularity": "per_config",    // "single" (one job, in sequence) | "per_config"
      // Cap how many configs run at once (submits one throttled Slurm job array,
      // --array=0-(K-1)%N). KGQA runs are heavy (1B model + LoRA), so bound the
      // concurrent load. Omit for one independent job per config (all queued).
      "max_concurrent": 4,
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B200:1",               // --gres=gpu:<gpus>
      "cpus": 8,                      // cf. old train.sbatch: 8 CPUs
      "mem": "64G",                   // cf. old train.sbatch: 64G
      "time": "12:00:00",             // cf. old train.sbatch: 12h
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh",
      // "nodelist": "ixb3",          // optional -w
      // "dry_run": true,             // write sbatch_commands.sh but don't submit
      // HF + wandb creds are read from $HOME (~/.cache/huggingface/token, ~/.netrc);
      // torch bundles its CUDA runtime, so no `module load` is needed.
    }
  },

  // ── sweep axes (lists => swept) ───────────────────────────────────────────
  "mode":   "train",                  // "train" | "data_prep" (run once in data_prep first)
  "lora_r": [8, 16],                  // 0 disables LoRA
  "k_hop":  [0, 2],                   // 0 disables the k-hop attention gate
  "seed":   [0, 1, 2],

  // ── a data-config bundle (fields that change the .gtds cache dir, coupled) ─
  "data_profile": [
    { "rel_mode": "last_1", "max_nodes": 512 },
    { "rel_mode": "last_2", "max_nodes": 512 }
  ],

  // ── dataset roles (multi-benchmark; see TODO_cwq E2) ──────────────────────
  // One dataset: a plain string. Both: the LIST-OF-LISTS form (a plain list of
  // strings would be a sweep AXIS — one run per dataset — per the rules above).
  "train_datasets": "webqsp",         // or [["webqsp", "cwq"]] = concat training
  "eval_datasets": "webqsp",          // each eval dataset scored separately: eval_{ds}_f1, ...
  "selection_dataset": null,          // null = select on the MEAN of per-dataset dev F1s
  // max_nodes / n_max / versions accept a scalar or a per-dataset string:
  // "max_nodes": "webqsp=512,cwq=1024"

  // ── fixed scalars ─────────────────────────────────────────────────────────
  "model_name": "meta-llama/Llama-3.2-1B",
  "graph_attn_impl": "flex",          // "flex" | "eager"
  "dtype": "bf16",                    // "bf16" | "fp32"
  "k_hop_directed": false,
  "gradient_checkpointing": true,
  "num_epochs": 5,
  "batch_size": 2,
  "accumulation_steps": 4,
  "lr": 3e-4,
  "bias_lr": 5e-3,
  "eval_steps": 100,
  "max_steps": -1,                    // >0 caps optimizer steps (quick smoke tests)
  "num_workers": 4,
  "n_max": 20,
  "versions": 8,                      // per-graph answer-order augmentations (train only)
  "data_seed": 42,                    // augmentation seed (baked into the data cache key)

  // ── graph-bias features (spd + magnetic; both used by data prep and model) ─
  "spd": true,
  "magnetic": true,
  "max_spd": 64,
  "magnetic_dim": 128,                // model bias-MLP width
  "magnetic_m": 128,                  // # magnetic eigenvectors (data + collator; 0 = all N)
  "magnetic_q": 0.25,

  // ── generative eval ───────────────────────────────────────────────────────
  "gen_max_new_tokens": 128,
  "gen_max_samples": null,            // FINAL dev/test scoring cap; null = full split
  "gen_eval_samples": 128,            // in-training eval cap; null = gen_max_samples

  "wandb_project": null               // e.g. "GraphLLM"; null = no tracking
}
"""


def _str_list(raw):
    """Comma-splitting type for list flags (rendered comma-joined by the runner)."""
    return [x.strip() for x in raw.split(",") if x.strip()]


def _per_dataset_int(raw):
    """Type for dataset-resolvable int flags: "512" -> 512 (all datasets),
    "webqsp=512,cwq=1024" -> {"webqsp": 512, "cwq": 1024} (per dataset;
    the runner renders dict values back to exactly this form)."""
    if "=" not in raw:
        return int(raw)
    out = {}
    for part in _str_list(raw):
        key, _, value = part.partition("=")
        out[key.strip()] = int(value)
    return out


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.kgqa",
        description="Run ONE KGQA (SR-WebQSP) configuration (the sweep runner invokes this per config).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--init", nargs="?", const="template", default=None, metavar="NAME",
                   help="Write a sweep-config template to configs/<NAME>.jsonc and exit.")
    p.add_argument("--mode", choices=("train", "data_prep", "flat_train", "flat_data_prep"),
                   default=d.mode,
                   help="train one config | build that config's datasets. The flat_* modes "
                        "are the D2 text-serialization control arm (same subgraph as triple "
                        "text, plain HF causal LM, no graph biases).")

    # ── dataset roles (E2.1) ─────────────────────────────────────────────────
    p.add_argument("--dataset", choices=DATASETS, default=None,
                   help="single-benchmark alias: sets BOTH --train-datasets and "
                        "--eval-datasets to this one dataset (error if combined "
                        "with either explicit flag).")
    p.add_argument("--train-datasets", type=_str_list, default=list(d.train_datasets),
                   help="comma-separated benchmarks to train on (mixed = plain "
                        "concat of the built train splits).")
    p.add_argument("--eval-datasets", type=_str_list, default=list(d.eval_datasets),
                   help="comma-separated benchmarks to evaluate on — each scored "
                        "separately under its own metric namespace (eval_{ds}_f1, ...).")
    p.add_argument("--selection-dataset", choices=DATASETS, default=d.selection_dataset,
                   help="pin checkpoint selection to ONE eval dataset's dev F1 "
                        "(omit = mean of the per-dataset dev F1s, eval_sel_f1).")

    # ── data-prep keys (determine the .gtds cache directory) ─────────────────
    p.add_argument("--rel-mode", choices=REL_MODES, default=d.rel_mode)
    p.add_argument("--max-nodes", type=_per_dataset_int, default=d.max_nodes,
                   help="Levi node cap; scalar or per-dataset ('webqsp=512,cwq=1024').")
    p.add_argument("--n-max", type=_per_dataset_int, default=d.n_max,
                   help="max answers in the training target; scalar or per-dataset.")
    p.add_argument("--versions", type=_per_dataset_int, default=d.versions,
                   help="per-graph answer-order augmentations (train split only); "
                        "scalar or per-dataset (CWQ uses 1: near-singleton answer sets).")
    p.add_argument("--max-length", type=int, default=d.max_length, help="per-node token cap (kept non-binding)")
    p.add_argument("--rcm", action=B, default=d.rcm, help="reverse-Cuthill-McKee node ordering.")
    p.add_argument("--data-seed", type=int, default=d.data_seed, help="augmentation RNG seed (in the cache key).")
    p.add_argument("--data-format-version", type=int, default=d.data_format_version,
                   help="pipeline-semantics version (cache key + answer separator); "
                        "pin an older value to run a legacy-format control arm.")
    p.add_argument("--naming-version", type=int, default=d.naming_version,
                   help="mid->name dict: 1 = GNN-RAG's 560k seed (entities_names.v1.json), "
                        "2 = +Freebase aliases (entities_names.json). Pin 1 for a true dfv2 control.")
    p.add_argument("--prompt-style", choices=PROMPT_STYLES, default=d.prompt_style,
                   help="prompt-node format: plain '{q}\\nAnswer:' | chat-templated turns; "
                        "omit for the auto default (chat iff an Instruct model). Explicit "
                        "'plain' on an Instruct model = the weights-vs-formatting control arm.")
    p.add_argument("--cvt-collapse", action=B, default=d.cvt_collapse,
                   help="single-parent CVT mediator collapse; omit for the arm default "
                        "(graph: collapse, flat: raw triples). Explicit values run the "
                        "2x2 collapse ablation (--no-cvt-collapse graph arm / "
                        "--cvt-collapse flat arm).")
    p.add_argument("--use-gpu", action=B, default=d.use_gpu, help="build SPD/magnetic features on GPU (data prep).")
    p.add_argument("--analyse-dataset", action=B, default=d.analyse_dataset,
                   help="during data prep, also run the answer-coverage ceiling analysis "
                        "(the README table) and save coverage_analysis.json next to the splits.")

    # ── shared model/bias keys ───────────────────────────────────────────────
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim, help="model bias-MLP hidden width.")
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m,
                   help="# magnetic eigenvectors (data prep + collator; 0 = all N).")

    # ── train keys ───────────────────────────────────────────────────────────
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps, help=">0 caps optimizer steps (quick tests).")
    p.add_argument("--seed", type=int, default=d.seed, help="training seed (decoupled from --data-seed).")
    p.add_argument("--lora-r", type=int, default=d.lora_r, help="LoRA rank (0 disables LoRA).")
    p.add_argument("--k-hop", type=int, default=d.k_hop, help="k-hop attention gate (0 disables).")
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)
    p.add_argument("--graph-attn-impl", choices=GRAPH_ATTN_IMPLS, default=d.graph_attn_impl)
    p.add_argument("--dtype", choices=DTYPES, default=d.dtype)
    p.add_argument("--gradient-checkpointing", action=B, default=d.gradient_checkpointing)
    p.add_argument("--active-params", type=_str_list, default=list(d.active_params),
                   help="comma-separated trainable param groups (besides LoRA).")
    p.add_argument("--num-workers", type=int, default=d.num_workers,
                   help="DataLoader workers (0 = synchronous feature build; >0 overlaps with compute).")
    p.add_argument("--seq-len", type=int, default=d.seq_len,
                   help="flat arm only: max prompt+target tokens (outliers drop trailing triples).")
    p.add_argument("--boundary-loss-weight", type=float, default=d.boundary_loss_weight,
                   help="D4: loss weight on the '\\n' answer-boundary token (1.0 = off; "
                        "weights renormalized so total loss scale is unchanged).")
    # ── regularization (survivors of the TODO_reg probe campaign) ────────────
    p.add_argument("--bias-weight-decay", type=float, default=d.bias_weight_decay,
                   help="AdamW weight decay on graph-bias MATRICES (0.0 = the historical "
                        "accidental no-decay; SPD table + 1-D gains stay exempt).")
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout,
                   help="LoRA adapter-input dropout (honored in both arms).")

    # ── generative eval ──────────────────────────────────────────────────────
    p.add_argument("--gen-max-new-tokens", type=int, default=d.gen_max_new_tokens)
    p.add_argument("--gen-max-samples", type=int, default=d.gen_max_samples,
                   help="cap FINAL dev/test generative scoring (omit = full split).")
    p.add_argument("--gen-eval-samples", type=int, default=d.gen_eval_samples,
                   help="cap the in-training generative eval (checkpoint selection) "
                        "only; omit = fall back to --gen-max-samples. Capped splits "
                        "use a fixed seeded subsample (stable across checkpoints).")
    p.add_argument("--log-strict-metrics", action=B, default=d.log_strict_metrics,
                   help="also log the _strict exact-set metric variants "
                        "(off by default; the GNN-RAG-verbatim metrics always log).")

    # ── tracking ─────────────────────────────────────────────────────────────
    p.add_argument("--wandb-project", default=d.wandb_project,
                   help="wandb project to report to (omit for no tracking).")

    # ── sweep-runner bookkeeping (where to log this run) ─────────────────────
    p.add_argument("--runs-jsonl", default=None, help="(runner) JSONL to append this run's record to.")
    p.add_argument("--run-name", default=None, help="(runner) this run's name within the sweep.")
    p.add_argument("--sweep-id", default=None, help="(runner) the sweep this run belongs to.")
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args."""
    train_datasets, eval_datasets = args.train_datasets, args.eval_datasets
    if args.dataset is not None:
        d = RunConfig()
        if (train_datasets != list(d.train_datasets)
                or eval_datasets != list(d.eval_datasets)):
            raise SystemExit("--dataset is an alias for --train-datasets/--eval-datasets; "
                             "pass either the alias or the explicit flags, not both.")
        train_datasets = eval_datasets = [args.dataset]
    return RunConfig(
        mode=args.mode,
        train_datasets=tuple(train_datasets), eval_datasets=tuple(eval_datasets),
        selection_dataset=args.selection_dataset,
        rel_mode=args.rel_mode, max_nodes=args.max_nodes, n_max=args.n_max,
        versions=args.versions, max_length=args.max_length, rcm=args.rcm,
        data_seed=args.data_seed, data_format_version=args.data_format_version,
        naming_version=args.naming_version, prompt_style=args.prompt_style,
        cvt_collapse=args.cvt_collapse,
        use_gpu=args.use_gpu, analyse_dataset=args.analyse_dataset,
        model_name=args.model_name,
        spd=args.spd, max_spd=args.max_spd, magnetic=args.magnetic,
        magnetic_dim=args.magnetic_dim, magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps, lr=args.lr, bias_lr=args.bias_lr,
        eval_steps=args.eval_steps, max_steps=args.max_steps, seed=args.seed,
        lora_r=args.lora_r, k_hop=args.k_hop, k_hop_directed=args.k_hop_directed,
        graph_attn_impl=args.graph_attn_impl, dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        active_params=tuple(args.active_params), num_workers=args.num_workers,
        seq_len=args.seq_len, boundary_loss_weight=args.boundary_loss_weight,
        bias_weight_decay=args.bias_weight_decay, lora_dropout=args.lora_dropout,
        gen_max_new_tokens=args.gen_max_new_tokens, gen_max_samples=args.gen_max_samples,
        gen_eval_samples=args.gen_eval_samples,
        log_strict_metrics=args.log_strict_metrics,
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
          f"Edit it, then run:  python3 -m sweep src.experiments.kgqa {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init is not None:
        _do_init(args.init)
        return 0

    cfg = config_from_args(args)

    if args.mode == "data_prep":
        from .process_dataset import run_data_prep_mode
        run_data_prep_mode(cfg)
        return 0

    if args.mode == "flat_data_prep":
        from .flat_data import run_flat_data_prep_mode
        run_flat_data_prep_mode(cfg)
        return 0

    if args.mode == "flat_train":
        from .flat_train import run_flat_train_mode
        run_flat_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                            sweep_id=args.sweep_id)
        return 0

    from .train import run_train_mode
    run_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name, sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
