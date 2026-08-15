"""Aggregate Phase 0 results across seeds into the tables LINEAR_BIAS.md §4 asks for.

    python -m src.experiments.bias_experiments.linear_bias.analyse

Reports, per dataset and per ``M``: the linear head's R^2 against the trained MLP
head (median over seeds, per the campaign's median-seed rule), the residual as a
fraction of the bias's own std — the interpretable unit — and the worst layer,
because a single bad layer is what a mean would hide.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def _load(pattern: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        tag = name.split("_")[1]                       # p0_<tag>_seed<N>.json
        out.setdefault(tag, []).append(json.load(open(path)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="src/experiments/bias_experiments/linear_bias/results")
    a = p.parse_args(argv)

    groups = _load(os.path.join(a.results, "p0_*_seed*.json"))
    if not groups:
        raise SystemExit(f"No Phase 0 results under {a.results}")

    for tag, runs in groups.items():
        meta = runs[0]["meta"]
        print(f"\n=== {tag}  ({len(runs)} seeds, {meta['n_batches']}x{meta['batch_size']} "
              f"graphs, {meta['pairs_per_graph']} pairs/graph, d_mag={meta['d_mag']}, "
              f"H={meta['n_heads']})")
        print(f"{'M':>5} {'R2 (median seed)':>18} {'resid/std':>11} "
              f"{'worst layer R2':>15} {'worst':>6} {'max cond':>10}")

        m_keys = sorted({m for r in runs for m in r["p0a"]}, key=int)
        for m in m_keys:
            per_seed = [r["p0a"][m] for r in runs if m in r["p0a"]]
            if not per_seed:
                continue
            r2 = st.median([s["r2_mean_over_layers"] for s in per_seed])
            worst = st.median([s["r2_worst_layer"] for s in per_seed])
            # Identify which layer is worst in the median seed's run.
            ref = min(per_seed, key=lambda s: abs(s["r2_worst_layer"] - worst))
            wl = min(ref["per_layer"].items(), key=lambda kv: kv[1]["r2_mean"])[0]
            # resid/std is measured, not derived from R^2: the two agree only when
            # the fit is well-conditioned, and the point of printing both is to
            # see when it is not.
            resid = st.median([s["resid_mean_over_layers"] for s in per_seed]) \
                if "resid_mean_over_layers" in per_seed[0] else (max(1 - r2, 0)) ** 0.5
            cond = max(v.get("gram_cond", float("nan"))
                       for s in per_seed for v in s["per_layer"].values())
            print(f"{m:>5} {r2:>18.4f} {resid:>11.3f} {worst:>15.4f} {wl:>6} {cond:>10.2e}")

        spec = runs[0].get("p0b") or {}
        if spec:
            e = st.median([v["energy_in_cap"] for v in spec.values()])
            r90 = st.median([v["rank_90"] for v in spec.values()])
            r99 = st.median([v["rank_99"] for v in spec.values()])
            n_sv = list(spec.values())[0]["n_singular"]
            print(f"  P0b: energy within rank-2M cap {e:.4f} | "
                  f"rank@90% {r90:.0f} | rank@99% {r99:.0f} | matrix size {n_sv}")


if __name__ == "__main__":
    main()
