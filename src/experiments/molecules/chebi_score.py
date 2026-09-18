"""
Score a Tier-C checkpoint by generating captions, and select on the dev split.

Two jobs, and they are the same machinery pointed at different splits:

* **Selection.** A cosine run has no dev metric the Trainer can compute — BLEU
  needs generation, which `compute_metrics` never sees — so the run saves several
  checkpoints and this picks the best on **val**. Selecting on ``eval_loss``
  instead would optimise a different thing than the number reported, which is a
  defect this package has already recorded once (`train.py::TIER_METRIC`).
* **Reporting.** The winner is then scored on **test**, over the benchmark's whole
  3,300-molecule split, with predictions dumped so they can be rescored offline
  under the published metric protocol (`chebi_lit_metrics.py`).

Usage (GPU, through Slurm — never on the login node)::

    python -m src.experiments.molecules.chebi_score \\
        --run-dir results/<sweep>/<run> --split val --max-samples 500
    python -m src.experiments.molecules.chebi_score \\
        --checkpoint <ckpt> --split test --max-samples 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

from .chebi import BENCHMARK_SIZES, CHEBI_TASK, excluded_cids

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "chebi")

#: Heavy-atom buckets for the size ladder. The last two are the rows a cap-64
#: build never attempted, which is a different finding from failing on them.
BUCKETS = ((0, 20), (21, 40), (41, 64), (65, 128), (129, 10**6))


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", default=None,
                   help="score every checkpoint under this directory and pick "
                        "the best (selection mode).")
    p.add_argument("--checkpoint", default=None,
                   help="score exactly this checkpoint (reporting mode).")
    p.add_argument("--runs-jsonl", required=True,
                   help="the sweep's runs.jsonl; the run's own record is what "
                        "this rebuilds its config from, so a scoring pass cannot "
                        "drift from the run it scores.")
    p.add_argument("--run-name", required=True,
                   help="which record in that file (its `sweep_run`).")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 scores the whole split.")
    p.add_argument("--batch-tokens", type=int, default=2048,
                   help="see generate.DEFAULT_BATCH_TOKENS: the cap-128 "
                        "build's N^2 bias, not the sequence length, is what "
                        "sets the peak.")
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args(argv)


def config_from_record(runs_jsonl: str, run_name: str):
    """Rebuild the `RunConfig` of a finished run from its own training record.

    Scoring a checkpoint with a config that differs from the one that trained it
    is a way to report a number for a model that was never built — a different
    encoding, a different `max_spd`, a different ChEBI screen. Reading the run's
    own record makes that impossible by construction.
    """
    from .config import RunConfig

    record = None
    with open(runs_jsonl) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sweep_run") == run_name or row.get("run_name") == run_name:
                record = row
    if record is None:
        raise SystemExit(f"no record named {run_name!r} in {runs_jsonl}")

    fields = set(RunConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in record.items() if k in fields and v is not None}
    kwargs["pool"] = tuple(kwargs.get("pool") or ())
    for key in ("len_buckets", "node_buckets"):
        if kwargs.get(key):
            kwargs[key] = tuple(kwargs[key])
    return RunConfig(**kwargs)


def _bucket(n: int) -> str:
    for low, high in BUCKETS:
        if low <= n <= high:
            return f"{low}-{high}" if high < 10**6 else f"{low}+"
    return "?"


def checkpoints_of(run_dir: str) -> list:
    """Every complete checkpoint under ``run_dir``, in step order."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "adapter_model.safetensors")):
            try:
                out.append((int(path.rsplit("-", 1)[1]), path))
            except ValueError:
                continue
    return [p for _s, p in sorted(out)]


def score_checkpoint(checkpoint, cfg, split, max_samples, batch_tokens):
    """``(metrics, rows)`` — generate on ``split`` and score with our own metrics."""
    import torch

    from .captions import caption_metrics
    from .dataset import load_data
    from .generate import build_model_for_eval, generate_captions

    model, tokenizer, collator, device = build_model_for_eval(cfg, checkpoint)
    _train, val, test = load_data(cfg)
    source = val if split == "val" else test

    start = time.time()
    with torch.no_grad():
        predictions, targets, keys = generate_captions(
            model, tokenizer, collator, source, device=device,
            max_samples=max_samples or None, batch_tokens=batch_tokens)
    elapsed = time.time() - start

    rows = []
    for prediction, target, key in zip(predictions, targets, keys):
        rows.append({"key": key, "target": target,
                     "prediction": (prediction or "").strip(),
                     "heavy_atoms": key.get("heavy_atoms", -1)
                     if isinstance(key, dict) else -1})
    metrics = dict(caption_metrics(predictions, targets))
    metrics["generate_seconds"] = round(elapsed)
    return metrics, rows


def main(argv=None) -> int:
    args = _args(argv)
    os.makedirs(args.out, exist_ok=True)
    cfg = config_from_record(args.runs_jsonl, args.run_name)

    if bool(args.run_dir) == bool(args.checkpoint):
        raise SystemExit("pass exactly one of --run-dir (select) or "
                         "--checkpoint (report)")

    if args.run_dir:
        best, results = None, []
        for ckpt in checkpoints_of(args.run_dir):
            metrics, _rows = score_checkpoint(ckpt, cfg, args.split,
                                              args.max_samples, args.batch_tokens)
            step = int(ckpt.rsplit("-", 1)[1])
            results.append({"step": step, "checkpoint": ckpt, **metrics})
            print(f"[select] step {step:>6}  bleu2={metrics['bleu2']:.4f}  "
                  f"rouge_l={metrics['rouge_l']:.4f}  n={metrics['n']}")
            if best is None or metrics["bleu2"] > best["bleu2"]:
                best = results[-1]
        if best is None:
            raise SystemExit(f"no checkpoints under {args.run_dir}")
        print(f"\n[select] BEST on {args.split}: step {best['step']} "
              f"bleu2={best['bleu2']:.4f}")
        stem = os.path.join(args.out,
                            os.path.basename(args.run_dir.rstrip("/")) + "-select")
        with open(stem + ".json", "w") as fh:
            json.dump({"run_dir": args.run_dir, "split": args.split,
                       "max_samples": args.max_samples,
                       "candidates": results, "best": best}, fh, indent=2)
        print(f"wrote {stem}.json")
        return 0

    metrics, rows = score_checkpoint(args.checkpoint, cfg, args.split,
                                     args.max_samples, args.batch_tokens)
    excluded = excluded_cids(args.split,
                             heavy_atom_cap=cfg.chebi_heavy_atom_cap,
                             allow_disconnected=cfg.chebi_allow_disconnected)
    benchmark = BENCHMARK_SIZES[args.split]
    coverage = len(rows) / benchmark if args.max_samples == 0 else None
    print(f"\n{CHEBI_TASK} {args.split}: {len(rows)} scored, {len(excluded)} "
          f"excluded, benchmark {benchmark}"
          + (f", coverage {coverage:.4f}" if coverage is not None else ""))
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                           if k in ("bleu2", "bleu4", "rouge_l", "meteor")))

    stem = os.path.join(args.out,
                        f"{os.path.basename(os.path.dirname(args.checkpoint))}"
                        f"-{os.path.basename(args.checkpoint)}-{args.split}")
    with open(stem + ".jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with open(stem + ".json", "w") as fh:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint),
                   "split": args.split, "n_scored": len(rows),
                   "n_benchmark": benchmark, "coverage": coverage,
                   "excluded_cids": [e["cid"] for e in excluded],
                   "excluded_reasons": {r: sum(1 for e in excluded
                                               if e["reason"] == r)
                                        for r in {e["reason"] for e in excluded}},
                   "chebi_heavy_atom_cap": cfg.chebi_heavy_atom_cap,
                   "chebi_allow_disconnected": cfg.chebi_allow_disconnected,
                   "model_name": cfg.model_name, "seed": cfg.seed,
                   "arm": cfg.arm, "lr": cfg.lr,
                   "ours": metrics}, fh, indent=2)
    print(f"wrote {stem}.jsonl / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
