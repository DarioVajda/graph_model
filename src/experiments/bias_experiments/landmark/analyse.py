"""Read the landmark sweeps and report the structural-dimension scaling law.

    python -m src.experiments.bias_experiments.landmark.analyse [sweep_dir ...]

Both factorized bias types are swept on a MATCHED dimension axis (040), so the
two curves are directly comparable in the currency the deferred fused backbone
actually spends: appended head width. `magnetic_linear` costs 2M dims, `landmark`
costs 3k, and this maps each run back to its dims before comparing anything.

Reporting rules follow the repo's:

* **median seed**, not mean — the convention every KGQA sweep here uses.
* **per-arm bias_lr**, not a shared one. `linear_bias` Conclusion 5: the linear
  head wanted ~4x the MLP head's LR and "any B-vs-C comparison at one shared LR
  prices optimization, not math". Each (type, dims) cell reports its better LR and
  the table records which one won, so an LR artifact stays visible.
* **headroom fractions against a same-week floor** (041), not against a borrowed
  denominator.

**F1 is the headline, and EM/hit\\* are printed beside it — never instead of it.**
Every prior number in this line of work is F1, so swapping metrics to find a
favourable one would make this sweep incomparable to `linear_bias`,
`magnetic_content` and `bias_sharing`.

EM is printed beside F1 because for landmark the two move in OPPOSITE directions,
and that divergence is the arm's main diagnostic. They are not two views of one
prediction and neither contains the other:

* `em_accuracy` (`src/train/eval.py:13`) is TEACHER-FORCED — argmax at each answer
  position against gold, with the model conditioned on the GOLD prefix. No
  generation happens.
* `f1`/`hits1`/`hit_star` (`kgqa/evaluate.py:150`) come from real autoregressive
  `model.generate(...)`, parsed and matched GNN-RAG style.

So the columns split 3-1: the three generative metrics always move together, and
EM is free to move against them. For landmark every cell has F1/hits@1/hit\\* down
and EM up; for `magnetic_linear` all four peak in the same cell. Reading EM as if
it were a generative metric is what produced the (wrong) "EM ⊆ hit\\*, so this is a
sharpened distribution" claim in an earlier revision of README §9 — the subset
relation does not hold. See §9 for the mechanism (a constant query factor at the
isolated prompt node, which helps continuation and cannot do retrieval).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), "results")
F1, HITS = "test_webqsp_f1", "test_webqsp_hits1"
EM, HITSTAR = "test_webqsp_em_accuracy", "test_webqsp_hit_star"
SEL = "eval_sel_f1"

# 040's landmark cells ran the UNNORMALIZED form and are superseded by 042/043 as
# a measurement of the bias; they are kept only as the measured cost of that form.
# 040's `magnetic_linear` cells never touched that code path and stand.
#
# 044 is `magnetic_linear` at 043's low LRs and MUST be included: without it the
# per-cell "best bias_lr" is chosen from {5e-3, 2e-2} for magnetic and from
# {3e-4, ..., 2e-2} for landmark, i.e. one arm gets a 17x-wider LR search than the
# other. That is precisely the `linear_bias` Conclusion 5 error — pricing
# optimization instead of math — with the asymmetry pointing at the incumbent.
_DEFAULT_SWEEPS = ["040_webqsp_dimsweep", "041_webqsp_floor",
                   "042_webqsp_landmark_norm", "043_webqsp_landmark_lowlr",
                   "044_webqsp_magnetic_lowlr"]


def load(sweep_dir: str) -> list[dict]:
    p = os.path.join(sweep_dir, "runs.jsonl")
    if not os.path.exists(p):
        return []
    sweep = os.path.basename(sweep_dir)
    with open(p) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        r["_sweep"] = sweep
    return rows


# Sweeps that predate `landmark_norm` being written into the record, and what it
# was. Runs launched before the field was added carry the old format, so the flag
# has to be recoverable from the sweep it belongs to — otherwise `.get(..., False)`
# silently relabels every normalized run as unnormalized and pools 042/043 into
# 040, averaging the defect into the fix.
_NORM_BY_SWEEP = {"040_webqsp_dimsweep": False,
                  "042_webqsp_landmark_norm": True,
                  "043_webqsp_landmark_lowlr": True}


def _is_normalized(r: dict) -> bool:
    if "landmark_norm" in r:
        return bool(r["landmark_norm"])
    sweep = r.get("_sweep")
    if sweep in _NORM_BY_SWEEP:
        return _NORM_BY_SWEEP[sweep]
    raise SystemExit(
        f"run from sweep {sweep!r} has no `landmark_norm` and no known default — "
        "refusing to guess, since guessing wrong pools two different models into "
        "one cell. Add it to _NORM_BY_SWEEP.")


def arm_of(r: dict):
    """(type, struct_dims). `None` dims for the floor, which appends nothing.

    `landmark` and `landmark_unnorm` are kept as SEPARATE arms. They are different
    models — one has a Cauchy-Schwarz bound on |b| and the other provably does not
    — so pooling them into one (type, dims, lr) cell would average 040's runaway
    into 042/043's numbers and quietly understate the fixed form.
    """
    if r.get("landmark"):
        k = r.get("landmark_k_collate") or 32
        norm = _is_normalized(r)
        return ("landmark" if norm else "landmark_unnorm"), 3 * k
    if r.get("magnetic_linear"):
        m = r.get("magnetic_m_collate") or r.get("magnetic_m") or 0
        return "magnetic_linear", 2 * m
    return "none", 0


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="*",
                    default=[os.path.join(RESULTS, s) for s in _DEFAULT_SWEEPS])
    args = ap.parse_args()

    runs = []
    for s in args.sweeps:
        got = load(s)
        print(f"[{os.path.basename(s)}] {len(got)} runs")
        runs += got
    if not runs:
        raise SystemExit("no runs found — has the sweep finished?")

    cells = defaultdict(list)
    for r in runs:
        t, d = arm_of(r)
        cells[(t, d, r.get("bias_lr"))].append(r)

    floor_rs = [r for r in runs if arm_of(r)[0] == "none"]
    floor = median([r.get(F1) for r in floor_rs])
    floor_em = median([r.get(EM) for r in floor_rs])
    floor_hs = median([r.get(HITSTAR) for r in floor_rs])
    print(f"\nfloor (none, median seed): F1 = {floor}  EM = {floor_em}  "
          f"hit* = {floor_hs}")

    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
    print(f"\n{'type':16s} {'dims':>5s} {'bias_lr':>8s} {'n':>2s} "
          f"{'F1(med)':>8s} {'F1(min)':>8s} {'F1(max)':>8s} "
          f"{'EM':>7s} {'Hits@1':>7s} {'hit*':>7s} {'sel':>7s}")
    rows = []
    for (t, d, lr), rs in sorted(cells.items()):
        f1s = [r.get(F1) for r in rs]
        row = dict(type=t, dims=d, bias_lr=lr, n=len(rs),
                   f1=median(f1s), f1_min=min(f1s), f1_max=max(f1s),
                   em=median([r.get(EM) for r in rs]),
                   hits=median([r.get(HITS) for r in rs]),
                   hit_star=median([r.get(HITSTAR) for r in rs]),
                   sel=median([r.get(SEL) for r in rs]),
                   sweeps=sorted({r.get("_sweep") for r in rs}))
        rows.append(row)
        print(f"{t:16s} {d:5d} {lr if lr else 0:8.0e} {len(rs):2d} "
              f"{fmt(row['f1']):>8s} {fmt(row['f1_min']):>8s} {fmt(row['f1_max']):>8s} "
              f"{fmt(row['em']):>7s} {fmt(row['hits']):>7s} {fmt(row['hit_star']):>7s} "
              f"{fmt(row['sel']):>7s}")

    # Per-(type, dims): the better LR wins the cell, and which one is recorded.
    print(f"\n── scaling law (best bias_lr per cell) ──")
    print(f"{'type':16s} {'dims':>5s} {'F1':>8s} {'bias_lr':>8s} {'dF1':>9s} "
          f"{'dEM':>9s} {'seed spread':>12s} {'LR at edge?':>12s}")
    best = {}
    for (t, d), group in defaultdict(list, {
            (r["type"], r["dims"]): [] for r in rows}).items():
        cand = [r for r in rows if r["type"] == t and r["dims"] == d and r["f1"] is not None]
        if not cand:
            continue
        b = max(cand, key=lambda r: r["f1"])
        best[(t, d)] = b
        hr = ((b["f1"] - floor) if floor is not None else None)
        dem = ((b["em"] - floor_em)
               if (b["em"] is not None and floor_em is not None) else None)
        # If the winning LR is the smallest or largest sampled for this arm, the
        # optimum is outside the bracket and the cell is a lower bound on the arm,
        # not a measurement of it. That is exactly what 040/042 got wrong.
        lrs = sorted({r["bias_lr"] for r in cand if r["bias_lr"]})
        edge = ("" if len(lrs) < 2 or b["bias_lr"] not in (lrs[0], lrs[-1])
                else ("LOW-EDGE" if b["bias_lr"] == lrs[0] else "HIGH-EDGE"))
        print(f"{t:16s} {d:5d} {b['f1']:8.4f} {b['bias_lr']:8.0e} "
              f"{(f'{hr:+.4f}' if hr is not None else '-'):>9s} "
              f"{(f'{dem:+.4f}' if dem is not None else '-'):>9s} "
              f"{b['f1_max'] - b['f1_min']:12.4f} {edge:>12s}")
    print("\ndF1/dEM are vs the 041 floor. An *-EDGE tag means the best LR is the "
          "end of the\nsampled bracket, so the arm's optimum is outside it and the "
          "cell under-reads the arm.")

    out = os.path.join(RESULTS, "scaling.json")
    with open(out, "w") as f:
        json.dump({"floor": floor, "floor_em": floor_em, "floor_hit_star": floor_hs,
                   "cells": rows,
                   "best": {f"{t}@{d}": v for (t, d), v in best.items()}}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
