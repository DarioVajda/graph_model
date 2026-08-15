"""
Aggregate the three `bias_sharing` G-sweeps into the tables quoted in README §4.

Accuracy comes from ``results/<sweep>/runs.jsonl``. **Step time does not**: only
the GraphQA runner writes a timing column (``train_steps_per_second``), so for
WebQSP and context-4k the per-step cost is recovered from the tqdm progress
stamps in the slurm logs — see :func:`step_time_from_log`.

    python3 -m src.experiments.bias_experiments.bias_sharing.analyse                # all sweeps
    python3 -m src.experiments.bias_experiments.bias_sharing.analyse --sweep 002_webqsp_g_sweep

Every number in README §4 is produced here; regenerate rather than hand-edit.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import statistics as st
from collections import defaultdict
from typing import Optional

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# eval_steps per sweep — the tqdm sampling window is split on these boundaries so
# a span containing a (much slower) evaluation never enters the step-time median.
EVAL_STEPS = {
    "001_graphqa_g_sweep": 20,
    "002_webqsp_g_sweep": 200,
    "003_context4k_g_sweep": 500,
}

ARMS = [0, 1, 2, 4, 8, 16]


# ── run records ───────────────────────────────────────────────────────────────

def load_runs(sweep: str) -> list[dict]:
    path = os.path.join(RESULTS, sweep, "runs.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_sd(values) -> tuple[float, float]:
    values = list(values)
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def permutation_p(a: list[float], b: list[float]) -> float:
    """One-sided p for ``mean(b) - mean(a)`` under exchangeability (exact).

    With 3 vs 3 the smallest attainable p is 1/20 = 0.05, so a reported 0.05 means
    *perfect separation*, not a marginal result.
    """
    obs = st.mean(b) - st.mean(a)
    pool = a + b
    hits = total = 0
    for combo in itertools.combinations(range(len(pool)), len(a)):
        A = [pool[i] for i in combo]
        B = [pool[i] for i in range(len(pool)) if i not in combo]
        hits += (st.mean(B) - st.mean(A)) >= obs - 1e-12
        total += 1
    return hits / total


# ── step time from slurm logs ─────────────────────────────────────────────────

_BAR = re.compile(r"(\d+)/(\d+) \[(\d+):(\d\d):(\d\d)<")


def _bar_samples(path: str) -> tuple[Optional[int], list[tuple[int, int]]]:
    """``(total_steps, [(step, elapsed_s), …])`` for the *main* tqdm bar.

    A training log carries several interleaved bars (train, and one per
    evaluation). The training bar is the one that prints most often; each step is
    taken at its first sighting so a redrawn line cannot skew the elapsed time.
    """
    raw, seen_totals = [], defaultdict(int)
    with open(path, errors="ignore") as f:
        for line in f:
            for m in _BAR.finditer(line):
                step, total = int(m.group(1)), int(m.group(2))
                h, mi, s = int(m.group(3)), int(m.group(4)), int(m.group(5))
                raw.append((total, step, h * 3600 + mi * 60 + s))
                seen_totals[total] += 1
    if not raw:
        return None, []
    main = max(seen_totals, key=lambda t: seen_totals[t])
    first: dict[int, int] = {}
    for total, step, elapsed in raw:
        if total == main:
            first.setdefault(step, elapsed)
    return main, sorted(first.items())


def step_time_from_log(path: str, eval_steps: int, min_span: int = 60) -> Optional[float]:
    """Median seconds/step over evaluation-free spans of the training bar.

    tqdm stamps elapsed time at 1-second resolution, so consecutive samples (one
    step apart) quantize to a useless 0/1 s. Instead each ``eval_steps`` window
    contributes ONE rate, measured between its first and last sample — at least
    ``min_span`` steps apart, which puts the quantization error under ~1%.
    """
    _, points = _bar_samples(path)
    if not points:
        return None
    windows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for step, elapsed in points:
        windows[step // eval_steps].append((step, elapsed))
    rates = []
    for samples in windows.values():
        samples.sort()
        (s0, e0), (s1, e1) = samples[0], samples[-1]
        if s1 - s0 >= min_span and e1 > e0:
            rates.append((e1 - e0) / (s1 - s0))
    return st.median(rates) if rates else None


def step_times_by_arm(sweep: str) -> dict[int, list[float]]:
    """``{G: [s/it per run]}``, keyed off the sweep's array map."""
    base = os.path.join(RESULTS, sweep)
    array_map = {}
    with open(os.path.join(base, "array_map.tsv")) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            task, name = line.split()
            array_map[int(task)] = name

    out: dict[int, list[float]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(base, "logs", "*.slurm.out"))):
        task = int(path.rsplit("_", 1)[1].split(".")[0])
        name = array_map.get(task)
        if name is None:
            continue
        g = int(re.search(r"magnetic_groups(\d+)", name).group(1))
        rate = step_time_from_log(path, EVAL_STEPS[sweep])
        if rate is not None:
            out[g].append(rate)
    return out


# ── per-sweep reports ─────────────────────────────────────────────────────────

def _accuracy_block(rows: list[dict], metric: str, group_by: Optional[str] = None,
                    scale: float = 100.0) -> None:
    keys = sorted({r[group_by] for r in rows}) if group_by else [None]
    header = f'{"":16}' if group_by else ""
    print(header + "".join(f"  G={g:<13}" for g in ARMS))
    for key in keys:
        subset = [r for r in rows if group_by is None or r[group_by] == key]
        cells = []
        for g in ARMS:
            values = [r[metric] * scale for r in subset if r["magnetic_groups"] == g]
            m, sd = mean_sd(values)
            cells.append(f"{m:5.2f}±{sd:4.2f}(n{len(values)})")
        label = f"{key:16}" if group_by else ""
        print(label + "  ".join(cells))


def report(sweep: str) -> None:
    rows = load_runs(sweep)
    print("=" * 78)
    print(f"{sweep}   {len(rows)} runs, {len({r['sweep_run'] for r in rows})} unique")
    print("=" * 78)

    if sweep == "001_graphqa_g_sweep":
        print("\ntest_accuracy (%) by task")
        _accuracy_block(rows, "test_accuracy", group_by="task")
        print("\nbest_val_accuracy (%) by task — the model-selection metric")
        _accuracy_block(rows, "best_val_accuracy", group_by="task")
        print("\ntrain_steps_per_second by task (recorded by the runner)")
        print(f'{"":16}' + "".join(f"  G={g:<13}" for g in ARMS))
        for task in sorted({r["task"] for r in rows}):
            base, _ = mean_sd(r["train_steps_per_second"] for r in rows
                              if r["task"] == task and r["magnetic_groups"] == 0)
            cells = []
            for g in ARMS:
                m, sd = mean_sd(r["train_steps_per_second"] for r in rows
                                if r["task"] == task and r["magnetic_groups"] == g)
                cells.append(f"{m:5.3f}±{sd:.3f} ({m / base:4.2f}x)")
            print(f"{task:16}" + "  ".join(cells))
        a = [r["test_accuracy"] for r in rows if r["task"] == "edge_count" and r["magnetic_groups"] == 0]
        b = [r["test_accuracy"] for r in rows if r["task"] == "edge_count" and r["magnetic_groups"] > 0]
        av = [r["best_val_accuracy"] for r in rows if r["task"] == "edge_count" and r["magnetic_groups"] == 0]
        bv = [r["best_val_accuracy"] for r in rows if r["task"] == "edge_count" and r["magnetic_groups"] > 0]
        print(f"\nedge_count G=0 vs pooled G>=1:  test p={permutation_p(a, b):.4f}  "
              f"val p={permutation_p(av, bv):.4f}")

    elif sweep == "002_webqsp_g_sweep":
        for metric in ("test_webqsp_f1", "test_webqsp_hits1", "test_webqsp_hit_star"):
            print(f"\n{metric} (%)")
            _accuracy_block(rows, metric)
        f1 = lambda g: [r["test_webqsp_f1"] for r in rows if r["magnetic_groups"] == g]  # noqa: E731
        print("\npermutation tests on test_webqsp_f1 (p=0.05 is the n=3 floor):")
        for lo, hi in [(0, 16), (1, 16), (2, 16), (4, 16), (8, 16)]:
            print(f"  G={lo:<2} vs G={hi:<2}  diff {100*(st.mean(f1(hi)) - st.mean(f1(lo))):+5.2f}  "
                  f"p={permutation_p(f1(lo), f1(hi)):.4f}")

    elif sweep == "003_context4k_g_sweep":
        print("\ndev_em (%) — in-distribution mixture, teacher-forced")
        _accuracy_block(rows, "dev_em")
        print("\nper-seed dev_em, to expose the single collapsed run:")
        for g in ARMS:
            per_seed = sorted((r["seed"], round(r["dev_em"] * 100, 1))
                              for r in rows if r["magnetic_groups"] == g)
            print(f"  G={g:<3} {per_seed}")

    # GraphQA's runner records `train_steps_per_second` directly (printed above);
    # its eval_steps=20 windows are also too short to sample a 60-step span, so
    # log scraping is both unnecessary and impossible there.
    times = {} if sweep == "001_graphqa_g_sweep" else step_times_by_arm(sweep)
    if times:
        print("\nmedian training s/it (from slurm tqdm stamps, evaluation-free spans)")
        base, _ = mean_sd(times[0]) if 0 in times else (None, None)
        for g in ARMS:
            if g not in times:
                continue
            m, sd = mean_sd(times[g])
            speedup = f"  {base / m:4.3f}x vs G=0" if base else ""
            print(f"  G={g:<3} n={len(times[g])}  {m:6.3f} ± {sd:.3f}{speedup}")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="append", choices=sorted(EVAL_STEPS),
                   help="repeatable; default is all three")
    args = p.parse_args(argv)
    for sweep in (args.sweep or sorted(EVAL_STEPS)):
        report(sweep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
