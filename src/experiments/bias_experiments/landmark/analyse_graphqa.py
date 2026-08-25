"""Read sweep 045 and report landmark's GraphQA k-scaling against reused controls.

    python -m src.experiments.bias_experiments.landmark.analyse_graphqa

Separate from `analyse.py` rather than a flag on it: GraphQA scores exact-answer
`test_accuracy` over a `task` axis, WebQSP scores generation F1 over a dimension
axis, and the reused controls live in different sweeps. Forcing one table to carry
both would make each half harder to read than either is alone.

Reporting rules follow the repo's, and `analyse.py`'s:

* **median seed** over the three seeds (42/43/44), not mean.
* **per-cell best bias_lr**, with the winning LR recorded and tagged when it sits
  at the edge of the sampled bracket — a cell whose optimum is outside the bracket
  under-reads its arm, which is the mistake 040/042 made on WebQSP.
* **controls are REUSED, not re-run**: the no-bias floor, `magnetic` and
  `magnetic_linear` (masked and unmasked) all come from 013/017 at this identical
  recipe.

Landmark is compared to `magnetic` and `magnetic_linear` INDIVIDUALLY, against a
common no-bias floor. On GraphQA the biases are not a marginal effect — magnetic
takes `node_degree` from 0.086 to 0.984 — so the interesting question is not
"does a bias help" but "how much of that does landmark recover".

### The k=16 caveat is enforced here, not left to the reader

GraphQA graphs average 12.9 nodes, so at k=16 nearly every node is an anchor and
the landmark oracle is the exact all-pairs distance matrix rather than an
approximation of it. On `shortest_path` that is close to encoding the label, which
is why every GraphQA bias config in this directory runs `spd: false`. So the k=16
row is printed as CEILING, and on `shortest_path` it is printed as
CEILING/LABEL-ADJACENT — the honest landmark result is the k=4 and k=8 rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

_HERE = os.path.dirname(__file__)
RESULTS = os.path.join(_HERE, "results")
_BIAS_EXP = os.path.dirname(_HERE)

ACC, VAL = "test_accuracy", "best_val_accuracy"
TASKS = ("node_degree", "shortest_path", "edge_count")

# Reused controls, by (sweep runs.jsonl, arm label).
#
# READ THE LABELS CAREFULLY — the `arm()` convention names what is OFF, so
# "no-spd+rrwp" means spd and rrwp are off and MAGNETIC IS ON. The arm with no
# graph bias at all is "no-spd+rrwp+magnetic". Getting this backwards inverts
# every comparison in this file: it swaps the strongest bias arm for the floor.
#
# 013 sweeps the masked variants; 017 adds the `+selfnode` (unmasked) ones.
# Landmark cannot mask its diagonal (an inner product cannot be forced to zero on
# i=j), so it runs unmasked, and the LIKE-FOR-LIKE comparators are 017's
# `+selfnode` arms. The masked ones are carried too, since that is the
# configuration 013 reported and the difference is worth seeing.
_L013 = os.path.join(_BIAS_EXP, "linear_bias", "results",
                     "013_graphqa_linear", "runs.jsonl")
_L017 = os.path.join(_BIAS_EXP, "linear_bias", "results",
                     "017_graphqa_selfnode", "runs.jsonl")

# (label, path, arm, unmasked?) — printed in this order, floor first.
_CONTROLS = [
    ("floor (no bias)",      _L013, "no-spd+rrwp+magnetic",           False),
    ("magnetic (MLP)",       _L013, "no-spd+rrwp",                    False),
    ("magnetic +selfnode",   _L017, "no-spd+rrwp+selfnode",           True),
    ("magnetic_linear",      _L013, "mag-linear+no-spd+rrwp",         False),
    ("mag_linear +selfnode", _L017, "mag-linear+no-spd+rrwp+selfnode", True),
]
_FLOOR = (_L013, "no-spd+rrwp+magnetic")


_K_RE = re.compile(r"landmark_k_collate(\d+)")


def _k_of(r: dict):
    """The anchor count this run actually used.

    Runs launched before `landmark_k_collate` was added to GraphQA's result record
    do not carry it, and k is the ONLY thing separating the cells of the sweep —
    absent, every k lands in one bucket and the scaling curve reads as flat rather
    than as missing. The sweep's own run name encodes it, so recover it from there
    rather than defaulting to a value that silently merges the cells.
    """
    k = r.get("landmark_k_collate") or r.get("landmark_k")
    if k:
        return int(k)
    m = _K_RE.search(r.get("sweep_run") or "")
    if m:
        return int(m.group(1))
    raise SystemExit(
        f"run {r.get('sweep_run')!r} has no landmark_k_collate and none in its "
        "name — refusing to guess, since a wrong k merges distinct cells.")


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _control(spec, task, with_spread=False):
    """Median-seed accuracy of a reused control arm on `task`, best over bias_lr.

    With `with_spread`, also returns the winning cell's seed min-max range, which
    is what decides whether a task has enough headroom to resolve anything.
    """
    path, arm = spec
    rows = [r for r in load(path) if r.get("arm") == arm and r.get("task") == task]
    if not rows:
        return (None, None) if with_spread else None
    by_lr = defaultdict(list)
    for r in rows:
        by_lr[r.get("bias_lr")].append(r.get(ACC))
    cells = [(median(v), max(v) - min(v)) for v in by_lr.values() if median(v) is not None]
    if not cells:
        return (None, None) if with_spread else None
    best = max(cells, key=lambda c: c[0])
    return best if with_spread else best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=os.path.join(RESULTS, "045_graphqa_landmark"))
    args = ap.parse_args()

    runs = load(os.path.join(args.sweep, "runs.jsonl"))
    if not runs:
        raise SystemExit(f"no runs in {args.sweep} — has the sweep finished?")
    print(f"[{os.path.basename(args.sweep)}] {len(runs)} runs")

    cells = defaultdict(list)
    for r in runs:
        cells[(r.get("task"), _k_of(r), r.get("bias_lr"))].append(r)

    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
    print(f"\n{'task':15s} {'k':>3s} {'dims':>5s} {'bias_lr':>8s} {'n':>2s} "
          f"{'acc(med)':>9s} {'acc(min)':>9s} {'acc(max)':>9s} {'val':>8s}")
    rows = []
    for (task, k, lr), rs in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0, kv[0][2] or 0)):
        accs = [r.get(ACC) for r in rs]
        row = dict(task=task, k=k, dims=3 * (k or 0), bias_lr=lr, n=len(rs),
                   acc=median(accs), acc_min=min(accs), acc_max=max(accs),
                   val=median([r.get(VAL) for r in rs]))
        rows.append(row)
        print(f"{str(task):15s} {k or 0:3d} {row['dims']:5d} {lr if lr else 0:8.0e} "
              f"{len(rs):2d} {fmt(row['acc']):>9s} {fmt(row['acc_min']):>9s} "
              f"{fmt(row['acc_max']):>9s} {fmt(row['val']):>8s}")

    # ── the comparators, and how much room there is above the floor ────────
    # On GraphQA a graph bias is worth 40-90 pp, not 1-2, so the floor and the
    # incumbents are printed together: "beats the floor" is a low bar here and
    # "closes the gap to magnetic" is the one that means something.
    print("\n── reused controls (median seed, best bias_lr) ──")
    hdr = f"{'arm':22s}" + "".join(f"{t:>16s}" for t in TASKS)
    print(hdr)
    ctl = {}
    for label, path, arm, unmasked in _CONTROLS:
        vals = [_control((path, arm), t) for t in TASKS]
        ctl[label] = dict(zip(TASKS, vals))
        print(f"{label:22s}" + "".join(f"{fmt(v):>16s}" for v in vals))
    floor = ctl["floor (no bias)"]
    best_incumbent = {}
    for t in TASKS:
        cands = [(lab, ctl[lab][t]) for lab, _, _, _ in _CONTROLS
                 if lab != "floor (no bias)" and ctl[lab][t] is not None]
        best_incumbent[t] = max(cands, key=lambda c: c[1]) if cands else (None, None)
    print(f"{'headroom over floor':22s}"
          + "".join(f"{fmt((best_incumbent[t][1] - floor[t]) if (best_incumbent[t][1] is not None and floor[t] is not None) else None):>16s}"
                    for t in TASKS))
    print(f"{'  best incumbent':22s}"
          + "".join(f"{str(best_incumbent[t][0])[:15]:>16s}" for t in TASKS))

    print(f"\n── landmark k-scaling per task (best bias_lr per cell) ──")
    print(f"{'task':14s} {'k':>3s} {'acc':>8s} {'bias_lr':>8s} {'vs floor':>9s} "
          f"{'vs magnet':>10s} {'vs maglin':>10s} {'gap closed':>11s} "
          f"{'spread':>7s} {'LRedge':>9s}  note")
    summary = {}
    for task in TASKS:
        f0 = floor[task]
        # The two arms the user asked landmark to be compared against, in the
        # configuration landmark itself runs (unmasked) where 017 provides it.
        mag = ctl["magnetic +selfnode"][task] or ctl["magnetic (MLP)"][task]
        mlin = ctl["mag_linear +selfnode"][task] or ctl["magnetic_linear"][task]
        for k in sorted({r["k"] or 0 for r in rows if r["task"] == task}):
            cand = [r for r in rows
                    if r["task"] == task and (r["k"] or 0) == k and r["acc"] is not None]
            if not cand:
                continue
            b = max(cand, key=lambda r: r["acc"])
            summary[f"{task}@k{k}"] = b
            lrs = sorted({r["bias_lr"] for r in cand if r["bias_lr"]})
            edge = ("" if len(lrs) < 2 or b["bias_lr"] not in (lrs[0], lrs[-1])
                    else ("LOW-EDGE" if b["bias_lr"] == lrs[0] else "HIGH-EDGE"))
            note = ""
            if (k or 0) >= 16:
                # All THREE tasks are functions of the exact distance matrix:
                # shortest_path is d(u,v); edge_count is #{d=1}/2; node_degree is
                # #{v : d(u,v)=1}. So at k=16, where nearly every node is an
                # anchor and the oracle is exact, every one of them is
                # label-adjacent — not just shortest_path, as this note used to
                # say. The k=16 column measures what a model does with near-exact
                # APSP handed to it, which is not a landmark result.
                note = "CEILING/LABEL-ADJACENT (oracle ~exact at k>=16)"
            d = lambda c: f"{b['acc'] - c:+.4f}" if c is not None else "-"
            # The headline number: what fraction of the floor -> best-incumbent
            # gap landmark recovers. 1.0 means it matches the best existing bias.
            top = best_incumbent[task][1]
            gap = ((b["acc"] - f0) / (top - f0)
                   if None not in (f0, top) and top > f0 else None)
            print(f"{task:14s} {k:3d} {b['acc']:8.4f} {b['bias_lr']:8.0e} "
                  f"{d(f0):>9s} {d(mag):>10s} {d(mlin):>10s} "
                  f"{(f'{gap:.2f}' if gap is not None else '-'):>11s} "
                  f"{b['acc_max'] - b['acc_min']:7.4f} {edge:>9s}  {note}")

    print("\nAll comparators are 013/017 reused at this identical recipe, not re-run.\n"
          "'gap closed' = (landmark - floor) / (best incumbent - floor): 1.00 means\n"
          "landmark matches the strongest existing bias on that task, 0.00 means it\n"
          "recovers none of what a graph bias is worth there. An *-EDGE tag means the\n"
          "winning LR is the end of the sampled bracket, so the cell under-reads the arm.\n"
          "k=16 rows are a CEILING, not a method result: see this module's docstring.")

    out = os.path.join(RESULTS, "scaling_graphqa.json")
    with open(out, "w") as f:
        json.dump({"cells": rows, "best": summary, "controls": ctl,
                   "best_incumbent": {t: best_incumbent[t][0] for t in TASKS}},
                  f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
