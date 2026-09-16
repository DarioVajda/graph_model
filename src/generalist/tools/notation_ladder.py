"""The notation ladder's read-out — the notation gradient, assembled from per-run scorings.

`MOLECULE_GENERALIST.md` §8.3. `notation_probe.py --checkpoint` scores one
trained run and writes `trained_<run>.json`; this merges those, adds the SMILES
leg from arm 2 and the graph arm from §8.2, and prints the table the section asks
for — property ROC-AUC per notation, and the flat-minus-graph gap that is the
actual deliverable.

**The gradient is the result, not a winner.** If the gap shrinks as the
backbone's exposure to the notation falls, §8's pretraining reading holds. If it
is flat across notations, the flat arm's edge is architectural. Both are results;
neither depends on having picked a favourable baseline, which is why all three
legs are printed and none is selected.

Usage:

    python3 -m src.generalist.tools.notation_ladder \\
        --dir src/generalist/results/notation_probe
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

#: The five property sets, in the order §8's tables use them.
TASKS = ("bace", "bbbp", "hiv", "tox21", "sider")

#: Notation exposure, high to low, as the corpus-frequency argument orders it —
#: SMILES is vastly more common in text than SELFIES or InChI. The zero-shot probe could not
#: measure this ordering (every arm read at chance), so it rests on that argument
#: and the write-up says so rather than implying a measurement.
ARM_ORDER = ("flat", "flat_inchi", "flat_selfies", "graph")

ARM_LABEL = {"flat": "SMILES", "flat_inchi": "InChI",
             "flat_selfies": "SELFIES", "graph": "graph"}


def load_rows(directory: str) -> list:
    """Every scored row under ``directory``, tagged with its run and seed."""
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "trained_*.json"))):
        record = json.load(open(path))
        run = os.path.basename(path)[len("trained_"):-len(".json")]
        for row in record["rows"]:
            row = dict(row)
            row["run"] = run
            row["seed"] = _seed_of(run)
            row["trained"] = record.get("trained", True)
            rows.append(row)
    return rows


def _seed_of(run: str) -> int:
    tail = run.rsplit("_s", 1)
    try:
        return int(tail[-1])
    except (ValueError, IndexError):
        return -1


def aggregate(rows) -> dict:
    """``(arm, task) -> (mean, sd, n_seeds)`` over the seeds present."""
    buckets: dict = {}
    for row in rows:
        if row.get("roc_auc") is None:
            continue
        buckets.setdefault((row["arm"], row["task"]), []).append(row["roc_auc"])
    out = {}
    for key, values in buckets.items():
        out[key] = (st.mean(values),
                    st.stdev(values) if len(values) > 1 else float("nan"),
                    len(values))
    return out


def table(agg: dict) -> str:
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in agg)]
    width = 16
    lines = ["", "Notation ladder — property ROC-AUC by notation (mean over seeds)",
             f"{'set':8s}" + "".join(f"{ARM_LABEL[a]:>{width}s}" for a in arms)]
    lines.append("-" * len(lines[-1]))
    for task in TASKS:
        cells = ""
        for arm in arms:
            got = agg.get((arm, task))
            cells += (f"{'  n/a':>{width}s}" if got is None
                      else f"{got[0]:>{width - 6}.4f} ±{got[1]:.3f}"
                      if got[2] > 1 else f"{got[0]:>{width}.4f}")
        lines.append(f"{task:8s}{cells}")

    lines.append("")
    means = {}
    for arm in arms:
        vals = [agg[(arm, t)][0] for t in TASKS if (arm, t) in agg]
        if vals:
            means[arm] = st.mean(vals)
    lines.append("five-set mean")
    lines.append(f"{'':8s}" + "".join(
        f"{means.get(a, float('nan')):>{width}.4f}" for a in arms))

    if "graph" in means:
        lines.append("")
        lines.append("flat - graph, the gradient this tier exists to produce")
        for arm in arms:
            if arm == "graph" or arm not in means:
                continue
            lines.append(f"  {ARM_LABEL[arm]:<10s} {means[arm] - means['graph']:+.4f}")
        lines.append("")
        lines.append("  Shrinking down this list as exposure falls -> §8's "
                     "pretraining reading holds.")
        lines.append("  Flat across it -> the flat arm's edge is architectural.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "notation_probe"))
    args = parser.parse_args(argv)

    rows = load_rows(args.dir)
    if not rows:
        raise SystemExit(f"no trained_*.json under {args.dir}")

    from .notation_probe import check_arms_agree_on_labels

    # The same assertion the probe makes, across runs this time: every arm scores
    # the same molecules, so the label base rate is a property of the task. It is
    # checked per seed, because a seed is one scoring pass.
    for seed in sorted({r["seed"] for r in rows}):
        check_arms_agree_on_labels([r for r in rows if r["seed"] == seed])

    agg = aggregate(rows)
    print(f"runs: {sorted({r['run'] for r in rows})}")
    print(table(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
