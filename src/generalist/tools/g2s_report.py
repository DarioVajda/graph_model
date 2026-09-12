"""What a checkpoint actually writes on graph-to-SMILES, and for which molecules.

`MOLECULE_GENERALIST.md` §5's three metrics come back as three numbers over a
500-row subsample, and on the graph arm two of the three have been exactly
0.0000 in every cell of the campaign. A zero is the least informative number a
generative task can return: it is the same whether the model emits nothing, emits
a plausible molecule that is not the right one, or emits the right skeleton with
one ring closure wrong. This tool opens it.

Two things it adds to `in_mixture`'s row:

* **The predictions themselves**, dumped one row per molecule with the target
  beside them, so the failure has a shape rather than a rate.
* **A size ladder.** Serializing a graph is a traversal, and a traversal gets
  harder with every atom. "Cannot do this at all" and "can do this for small
  molecules and loses the thread on large ones" are different findings that
  collapse to the same 0.0000 when the split is scored whole. Buckets are on the
  target's heavy-atom count.

The subsample is `eval_indices`, the same seeded one the campaign's `in_mixture`
firings used, so ``--max-samples 500`` reproduces the reported row exactly and
anything larger is a superset of it.

Usage (GPU, through Slurm — never on the login node):

    RUNMOD=src.generalist.tools.g2s_report GPU=1 src/generalist/tools/run_cli.sh \
        --run-config src/generalist/configs/probes/006_molecule_generalist_2x.jsonc \
        --cell molecule_generalist_graph_2x_s0 \
        --checkpoint src/generalist/results/runs/<run>-anneal-11140/anneal/checkpoint-12255

It builds nothing: a trained checkpoint implies its sources exist, and `build`
rewrites a shared `manifest.json`, so a scoring job that also builds races any
other one that is running (`tools/notation_probe.py` says the same).
"""

from __future__ import annotations

import argparse
import json
import os
import time

#: Heavy-atom buckets for the size ladder. Open-ended on the right, because the
#: campaign's pool runs to molecules the node budget only just admits.
BUCKETS = ((0, 10), (11, 15), (16, 20), (21, 30), (31, 10**6))

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "g2s")


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-config", required=True,
                   help="the config the checkpoint was trained under.")
    p.add_argument("--cell", default=None, help="which cell of that config.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=("test", "val"))
    p.add_argument("--max-samples", type=int, default=500,
                   help="0 scores the whole split; 500 is the campaign's row.")
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args(argv)


def _bucket(n: int) -> str:
    for low, high in BUCKETS:
        if low <= n <= high:
            return f"{low}-{high}" if high < 10**6 else f"{low}+"
    return "?"


def _heavy_atoms(smiles: str) -> int:
    """Heavy atoms in a *target*, which is canonical and always parses."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else -1


def rows_for(predictions, targets) -> list:
    """One dict per molecule: the pair, its size, and the three §5 verdicts.

    The verdicts are recomputed here rather than read off `smiles_scores`, which
    returns rates and not a per-row breakdown. They are the same three
    predicates, taken from the same place, and `check_totals_match` asserts that
    the rates they aggregate to are the ones `smiles_scores` reports — so this
    stays a decomposition of the reported number rather than a second opinion
    about it.
    """
    from rdkit import Chem, RDLogger

    from ..adapters.molecules import STEREO_MARKS

    RDLogger.DisableLog("rdApp.*")
    out = []
    for prediction, target in zip(predictions, targets):
        prediction = (prediction or "").strip()
        mol = Chem.MolFromSmiles(prediction) if prediction else None
        out.append({
            "target": target,
            "prediction": prediction,
            "heavy_atoms": _heavy_atoms(target),
            "bucket": _bucket(_heavy_atoms(target)),
            "valid": mol is not None,
            "roundtrip": (mol is not None
                          and Chem.MolToSmiles(mol, canonical=True) == target),
            "exact": prediction == target,
            "stereo": any(mark in prediction for mark in STEREO_MARKS),
            "target_chars": len(target),
            "prediction_chars": len(prediction),
        })
    return out


def check_totals_match(rows, scores) -> None:
    """The per-row verdicts must aggregate to `smiles_scores`' rates.

    Cheap, and it is the whole licence for reporting a breakdown: a ladder that
    does not add up to the reported number is a second measurement of the task,
    and nobody would know which of the two to believe.
    """
    n = len(rows) or 1
    for key, field in (("validity", "valid"), ("roundtrip_match", "roundtrip"),
                       ("exact_match", "exact")):
        mine = sum(1 for r in rows if r[field]) / n
        if abs(mine - scores[key]) > 1e-9:
            raise SystemExit(
                f"g2s_report: {key} is {scores[key]:.6f} from smiles_scores and "
                f"{mine:.6f} from the per-row verdicts; the breakdown is not a "
                "decomposition of the reported number and must not be published")


def ladder(rows) -> list:
    """The size ladder: one line per heavy-atom bucket, plus the total."""
    order = [f"{low}-{high}" if high < 10**6 else f"{low}+"
             for low, high in BUCKETS]
    out = []
    for name in order + ["all"]:
        got = rows if name == "all" else [r for r in rows if r["bucket"] == name]
        if not got:
            continue
        n = len(got)
        out.append({
            "bucket": name, "n": n,
            "validity": sum(r["valid"] for r in got) / n,
            "roundtrip_match": sum(r["roundtrip"] for r in got) / n,
            "exact_match": sum(r["exact"] for r in got) / n,
            "stereo_marks_emitted": sum(r["stereo"] for r in got) / n,
            "mean_target_chars": sum(r["target_chars"] for r in got) / n,
            "mean_prediction_chars": sum(r["prediction_chars"] for r in got) / n,
            "empty": sum(1 for r in got if not r["prediction"]) / n,
        })
    return out


def table(report) -> str:
    head = (f"  {'heavy atoms':<12}{'n':>6}{'valid':>9}{'roundtrip':>11}"
            f"{'exact':>9}{'empty':>8}{'chars':>8}{'target':>8}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for row in report:
        lines.append(
            f"  {row['bucket']:<12}{row['n']:>6}{row['validity']:>9.4f}"
            f"{row['roundtrip_match']:>11.4f}{row['exact_match']:>9.4f}"
            f"{row['empty']:>8.4f}{row['mean_prediction_chars']:>8.1f}"
            f"{row['mean_target_chars']:>8.1f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import torch

    from ..adapters import molecules
    from ..evaluate.scorers import eval_indices, generate_predictions
    from .notation_probe import build_trained_model

    args = _args(argv)
    os.makedirs(args.out, exist_ok=True)

    model, tokenizer, collator, config, device = build_trained_model(
        args.run_config, os.path.abspath(args.checkpoint),
        os.path.join(args.out, "scratch"), args.cell)
    spec = molecules.task_specs(config.adapter_config(), config.arm)["mol/g2s"]
    source = molecules.load("mol/g2s", args.split, config.arm, 0,
                            config.adapter_config())
    indices = eval_indices(len(source), args.max_samples or None)
    print(f"run {config.run_name}  arm {config.arm}  split {args.split}  "
          f"{len(indices)} of {len(source)} rows")

    start = time.time()
    with torch.no_grad():
        predictions, targets = generate_predictions(
            model, tokenizer, collator, source, indices,
            max_new_tokens=spec.max_new_tokens or 256, device=device,
            batch_tokens=args.batch_tokens)
    print(f"[generate] {len(predictions)} rows in {time.time() - start:.0f}s")

    rows = rows_for(predictions, targets)
    check_totals_match(rows, molecules.smiles_scores(predictions, targets))
    report = ladder(rows)
    print()
    print(table(report))

    stem = f"{config.run_name}-{os.path.basename(args.checkpoint)}-{args.split}"
    with open(os.path.join(args.out, f"{stem}.jsonl"), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(args.out, f"{stem}.json"), "w") as fh:
        json.dump({"run_name": config.run_name, "arm": config.arm,
                   "checkpoint": os.path.abspath(args.checkpoint),
                   "split": args.split, "n": len(rows), "ladder": report},
                  fh, indent=2, sort_keys=True)
    print(f"\nwrote {os.path.join(args.out, stem)}.jsonl / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
