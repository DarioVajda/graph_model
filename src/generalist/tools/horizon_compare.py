"""The doubled horizon against the campaign it doubles, task by task.

§9 Tier 2 asks one question: **does §8's property-prediction gap survive once HIV
and Tox21 train past one epoch?** Answering it needs the two horizons side by
side on every task, not just on the five-set mean — a gap that closes because the
graph arm improved is a different finding from one that closes because the flat
arm regressed, and only the per-task rows separate them.

Two instruments, and they are not interchangeable:

* **Property sets** come from `tools/notation_probe.py --checkpoint`, which
  excludes the union of truncated rows from every arm so the arms score the
  identical molecules. This is the one to quote.
* **Everything else** — the exact-match probes, ChEBI-20, g2s — comes from the
  anneal's own `result.json`, which is the only place they are measured.

    src/generalist/tools/horizon_compare.py

Reads whatever is on disk and says what is missing rather than failing, so it is
useful while the scoring is still running.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

PROBE_DIR = "src/generalist/results/notation_probe"
RUNS_DIR = "src/generalist/results/runs"
PROPERTY_TASKS = ("bace", "bbbp", "hiv", "sider", "tox21")

#: ``{comparison: {arm: [(label, run-name stem, anneal step), ...]}}``.
#:
#: One tool, because the question is always the same shape — two campaigns that
#: differ in exactly one thing, read task by task — and a second copy of the
#: reading code is a second place for the metric keys to drift.
#:
#: ``backbone`` compares base `Llama-3.2-1B` in ``Q:/A:`` formatting against
#: `Llama-3.2-1B-Instruct` in chat formatting, at the same 11,140 steps over the
#: same molecules. Note what is NOT comparable across those two legs: the g2s and
#: ChEBI-20 rows of the base leg were measured with no stop token, so they score
#: whether the model stopped (§8). Their base column is a floor, not a baseline.
LEG_SETS = {
    "horizon": {
        "graph": [("1x", "molecule_generalist_graph", 5599),
                  ("2x", "molecule_generalist_graph_2x", 11140)],
        "SMILES": [("1x", "molecule_generalist_flat", 5599),
                   ("2x", "molecule_generalist_flat_2x", 11140)],
    },
    "backbone": {
        "graph": [("base", "molecule_generalist_graph_2x", 11140),
                  ("instr", "molecule_generalist_instruct_graph", 11140)],
        "SMILES": [("base", "molecule_generalist_flat_2x", 11140),
                   ("instr", "molecule_generalist_instruct_flat", 11140)],
    },
}
LEGS = LEG_SETS["horizon"]
SEEDS = (0, 1, 2)


def probe_rows(stem, seed):
    path = os.path.join(PROBE_DIR, f"trained_{stem}_s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r["task"]: r["roc_auc"] for r in json.load(f)["rows"]}


def suite_metrics(stem, seed, step):
    """The anneal's own final metrics, for the tasks the probe does not score.

    Normally `result.json`. Two of the 2x graph anneals trained to completion and
    then lost that file to a walltime kill during the final evaluation, so a
    standalone `eval` on the surviving checkpoint is read as an equivalent
    source — the same validators over the same checkpoint, just scored in a
    second job rather than at the end of the first.
    """
    run_dir = os.path.join(RUNS_DIR, f"{stem}_s{seed}-anneal-{step}")
    path = os.path.join(run_dir, "result.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["legs"]["anneal"]["history"][-1][1]
    recovered = sorted(glob.glob(os.path.join(run_dir, "anneal_eval.json",
                                              "eval_step*.json")))
    if not recovered:
        return None
    with open(recovered[-1]) as f:
        return json.load(f)["eval"]["metrics"]


def mean_sd(values):
    if not values:
        return None
    mu = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 2 else 0.0
    return mu, sd


def fmt(stat):
    return "--" if stat is None else f"{stat[0]:.4f} ±{stat[1]:.4f}"


def collect(getter, legs=None):
    """``{arm: {leg: {key: [per-seed values]}}}``, skipping what is absent."""
    out, missing = {}, []
    for arm, legs_for_arm in (legs or LEGS).items():
        out[arm] = {}
        for horizon, stem, step in legs_for_arm:
            per_key = {}
            for seed in SEEDS:
                got = getter(stem, seed, step)
                if got is None:
                    missing.append(f"{stem}_s{seed}")
                    continue
                for key, value in got.items():
                    per_key.setdefault(key, []).append(value)
            out[arm][horizon] = per_key
    return out, missing


def table(title, rows, data, keyfmt=lambda k: k, labels=("1x", "2x")):
    a, b = labels
    print(f"\n### {title}\n")
    print(f"| task | graph {a} | graph {b} | Δ | SMILES {a} | SMILES {b} | Δ |")
    print("|---|---|---|--:|---|---|--:|")
    for key in rows:
        cells, deltas = [], []
        for arm in ("graph", "SMILES"):
            stats = [mean_sd(data[arm][h].get(key, [])) for h in labels]
            cells.append([fmt(s) for s in stats])
            deltas.append(None if None in stats else stats[1][0] - stats[0][0])
        line = f"| {keyfmt(key)} |"
        for (one, two), delta in zip(cells, deltas):
            line += f" {one} | {two} | {'--' if delta is None else f'{delta:+.4f}'} |"
        print(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs", default="horizon", choices=sorted(LEG_SETS),
                    help="which two campaigns to put side by side.")
    ap.add_argument("--suite-only", action="store_true",
                    help="skip the notation-probe table, which needs its own "
                         "scoring pass per cell.")
    args = ap.parse_args(argv)
    legs = LEG_SETS[args.legs]
    labels = tuple(label for label, _stem, _step in legs["graph"])

    prop, missing_prop = collect(lambda s, seed, step: probe_rows(s, seed), legs)
    if missing_prop:
        print(f"[probe] not scored yet: {', '.join(sorted(set(missing_prop)))}",
              file=sys.stderr)
    if not args.suite_only:
        table("Property prediction — ROC-AUC (clean instrument)",
              PROPERTY_TASKS, prop, keyfmt=str.upper, labels=labels)

    # The five-set mean, per seed, so the comparison stays paired.
    print()
    for arm in ("graph", "SMILES"):
        line = f"  {arm:7s}"
        for horizon in labels:
            per = prop[arm][horizon]
            if not all(per.get(t) for t in PROPERTY_TASKS):
                line += f"  {horizon}: --"
                continue
            n = len(per[PROPERTY_TASKS[0]])
            means = [statistics.fmean(per[t][i] for t in PROPERTY_TASKS)
                     for i in range(n)]
            line += f"  {horizon}: {statistics.fmean(means):.4f}"
            line += f" (seeds {', '.join(f'{m:.4f}' for m in means)})"
        print(line)

    suite, _ = collect(suite_metrics, legs)
    keys = set()
    for arm in suite:
        for horizon in suite[arm]:
            keys |= set(suite[arm][horizon])

    em = sorted(k for k in keys if k.endswith("/em_accuracy"))
    gen = sorted(k for k in keys
                 if any(k.endswith(m) for m in
                        ("/bleu2", "/bleu4", "/rouge_l", "/meteor",
                         "/exact_match", "/roundtrip_match", "/validity")))
    short = lambda k: k.replace("in_mixture/mol/", "").replace("held_out/mol/", "") \
                       .replace("/test", "").replace("/held_out", " (held out)")
    table("Structural probes — exact match",
          [k for k in em if "/val/" not in k], suite, keyfmt=short, labels=labels)
    table("Generation", [k for k in gen if "/val/" not in k], suite,
          keyfmt=short, labels=labels)
    if args.legs == "backbone":
        print("\nThe generation rows' `base` column was measured with no stop "
              "token (§8): it scores whether the model stopped, so it is a floor "
              "and not a baseline. The property rows are unaffected — they never "
              "generate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
