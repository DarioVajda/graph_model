"""RelBench x GTLM experiment -- standalone single-run entry point.

A self-contained argparse program: given parameters for **one** configuration it runs that
configuration. It knows nothing about sweeps or job submission -- the generic ``sweep``
runner drives those and invokes this program once per resolved config, rendering each config
key to the matching flag below.

    python3 -m src.experiments.relbench --mode data_prep --dataset rel-f1 --task driver-dnf
    python3 -m src.experiments.relbench --dataset rel-f1 --task driver-dnf --max-steps 8
    python3 -m src.experiments.relbench --arm-name flat --no-spd --no-magnetic
    python3 -m src.experiments.relbench --mode dump --task driver-dnf     # read the documents
    python3 -m src.experiments.relbench --init my_sweep

``--mode`` routes within the experiment: ``train`` trains one config and logs one record,
``data_prep`` builds that config's splits, ``dump`` prints assembled documents for reading.

Flags are generated from ``RunConfig``'s fields, so a knob added to the dataclass is
immediately settable from the CLI and from a sweep config with no second edit -- the two
lists cannot drift apart.
"""

import argparse
import dataclasses
import os

from .config import (
    ANONYMIZE, DTYPES, IMPLS, MODES, PROMPT_NODES, QUESTION_NODES, READOUTS, RunConfig,
    SAMPLING, TEXT_MODES, TIME_ENCODINGS,
)

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

# Choice sets, by field name. Anything not listed is a free scalar.
_CHOICES = {
    "mode": MODES, "impl": IMPLS, "dtype": DTYPES, "neighbor_sampling": SAMPLING,
    "text_mode": TEXT_MODES, "time_encoding": TIME_ENCODINGS, "anonymize": ANONYMIZE,
    "question_node": QUESTION_NODES, "prompt_node": PROMPT_NODES, "readout": READOUTS,
    "arm_name": ("graph", "flat"),
}

# Fields whose default is None but which take a real type when set.
_NONE_TYPES = {
    "relation_cap": int, "max_node_chars": int, "max_train_samples": int,
    "max_val_samples": int,
    "val_subsample": int, "test_subsample": int, "wandb_project": str,
}

_HELP = {
    "dataset": "relbench dataset id, e.g. rel-f1 or rel-trial.",
    "task": "relbench task id, e.g. driver-dnf or study-outcome.",
    "max_nodes": "budget in CONTENT rows; pure join rows ride along free.",
    "neighbor_sampling": "'recent' = the k most recent eligible rows; 'uniform' = RDL's policy.",
    "collapse_links": "contract contentless join rows into direct edges.",
    "sibling_fanout": ">0 pulls co-rows sharing a parent (include_siblings).",
    "max_value_chars": "per-field character cap; the dominant knob on text-heavy databases.",
    "arm_name": "'graph' = the full stack; 'flat' = the matched one-node control.",
    "readout": "'logit_margin' for binary (fp32 logit(yes)-logit(no)); 'numeric_text' for regression.",
    "k_hop": "K-hop attention gate (0 disables).",
    "max_steps": ">0 caps optimizer steps (smoke tests).",
}

TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // RelBench sweep config (JSONC: // comments + trailing commas allowed).
  // Run with:  python3 -m sweep src.experiments.relbench <this file>
  //
  // Expansion rules (how a value becomes runs):
  //   scalar            -> fixed in every run
  //   [a, b, c]         -> a sweep AXIS (one run per value; cartesian with others)
  //   [ {..}, {..} ]    -> a BUNDLE: params that vary TOGETHER
  // Keys map 1:1 to this experiment's CLI flags (some_key -> --some-key).
  //
  // TWO-STEP WORKFLOW: run once with "mode": "data_prep" to build every cache this
  // file references, then again with "mode": "train". Data prep is CPU-only.
  // The flat control needs its OWN data_prep run (arm_name: flat).
  // ─────────────────────────────────────────────────────────────────────────

  "name": "my_sweep",
  "results_dir": "src/experiments/relbench/results",

  "execution": {
    "mode": "local",                  // "local" | "sbatch"
    "sbatch": {
      "granularity": "per_config",
      "max_concurrent": 8,
      "partition": "frida",
      "account": "povejmo",
      "gpus": ["B200:1", "B300:1"],
      "cpus": 8,
      "mem": "64G",
      "time": "24:00:00",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
    }
  },

  "mode": "train",
  "dataset": "rel-f1",
  "task": "driver-dnf",
  "max_nodes": 64,

  "arm_name": ["graph", "flat"],
  "seed": [42, 43, 44],

  "model_name": "meta-llama/Llama-3.2-1B",
  "impl": "v2-flex",
  "dtype": "bf16",
  "lora": true,
  "lora_r": 32,
  "lr": 3e-4,
  "num_epochs": 4,
  "batch_size": 1,
  "accumulation_steps": 32,
  "eval_steps": 100,

  "wandb_project": null
}
"""


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(prog="python3 -m src.experiments.relbench",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", nargs="?", const="template", default=None, metavar="NAME",
                   help="write a sweep config template to configs/NAME.jsonc and exit.")

    B = argparse.BooleanOptionalAction
    for f in dataclasses.fields(RunConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(d, f.name)
        kwargs = {"default": default, "help": _HELP.get(f.name)}

        if f.name in _CHOICES:
            p.add_argument(flag, choices=_CHOICES[f.name], **kwargs)
        elif f.type is bool or isinstance(default, bool):
            p.add_argument(flag, action=B, **kwargs)
        elif f.name in _NONE_TYPES:
            p.add_argument(flag, type=_NONE_TYPES[f.name], **kwargs)
        elif isinstance(default, int):
            p.add_argument(flag, type=int, **kwargs)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, **kwargs)
        else:
            p.add_argument(flag, **kwargs)

    # -- dump-mode only -----------------------------------------------------
    p.add_argument("--dump-n", type=int, default=3,
                   help="(dump) how many documents to print.")

    # -- sweep-runner bookkeeping -------------------------------------------
    p.add_argument("--runs-jsonl", default=None,
                   help="(runner) JSONL to append this run's record to.")
    p.add_argument("--run-name", default=None, help="(runner) this run's name in the sweep.")
    p.add_argument("--sweep-id", default=None, help="(runner) the sweep this run belongs to.")
    return p


def config_from_args(args):
    """Build (and validate) a RunConfig from parsed args."""
    names = {f.name for f in dataclasses.fields(RunConfig)}
    return RunConfig(**{k: v for k, v in vars(args).items() if k in names}).validate()


def _do_init(name):
    if not (name.endswith(".json") or name.endswith(".jsonc")):
        name += ".jsonc"
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    path = os.path.join(CONFIGS_DIR, name)
    with open(path, "w") as f:
        f.write(TEMPLATE)
    print(f"Wrote sweep template to {path}\n"
          f"Edit it, then run:  python3 -m sweep src.experiments.relbench {path}")


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

    if cfg.mode == "dump":
        from .dump_documents import dump
        dump(cfg, n=args.dump_n)
        return 0

    from .train import run_train_mode
    run_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                   sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
