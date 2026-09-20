"""The §9.1 table: what each replay variant did to the molecules and to the text.

One row per run, one column per number the screen is decided on. It reads the
same two files every other report reads — a fork's `result.json` for the molecule
validators and `text_behaviour/<run>/eval_step*.json` for the text ones — so the
table cannot drift from the runs it describes.

    RUNMOD=src.generalist.tools.replay_report src/generalist/tools/run_cli.sh \
        --baseline molecule_generalist_instruct_graph_s0 \
        --seeds molecule_generalist_instruct_graph_s1 \
                molecule_generalist_instruct_graph_s2 \
        --runs replay_screen_graph replay08_screen_graph \
               replay_anneal15_graph_s0 replay_anneal40_graph_s0

`--baseline` is the run every delta is taken against (seed 0 of the campaign, the
only seed the variants share a data seed with). `--seeds` names the baseline's
sibling seeds, which is where the sd that the property bar is stated in comes
from — one seed's spread, not an error bar on the mean.

The bars are §9.1's, pre-registered: `caption_rate` <= 0.03 with a lower
`kl_mean`, the five-set property mean within one seed-sd, every in-mixture probe
within +-0.02, and g2s `exact_match` within 0.03.
"""

import argparse
import glob
import json
import os
import statistics
import sys

RESULTS = "src/generalist/results"

PROPERTY_SETS = ("bace", "bbbp", "hiv", "sider", "tox21")

PROBES = tuple(f"in_mixture/mol/{t}/test/em_accuracy" for t in (
    "aromatic_ring", "ring_count", "ring_size", "ring_membership",
    "fg_count", "fg_presence", "fg_atom_membership",
    "stereo_assigned", "stereo_potential"))

#: §9.1's bars.
CAPTION_BAR = 0.03
PROBE_BAR = 0.02
G2S_BAR = 0.03


def _final_metrics(path: str) -> dict:
    """Every metric at the highest step it was recorded at, from a nested record."""
    out = {}

    def walk(node):
        if isinstance(node, dict):
            metrics = node.get("metrics")
            if isinstance(metrics, dict):
                step = node.get("step", -1)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if key not in out or step >= out[key][0]:
                            out[key] = (step, value)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    with open(path) as f:
        walk(json.load(f))
    return {key: value for key, (_step, value) in out.items()}


def _anneal_result(root: str, run: str) -> dict:
    """A run's annealed metrics, whichever anneal directory it wrote."""
    for pattern in (f"{run}-anneal-*/result.json", f"{run}/result.json"):
        matches = sorted(glob.glob(os.path.join(root, "runs", pattern)))
        if matches:
            return _final_metrics(matches[-1])
    raise SystemExit(f"no anneal result.json for {run} under {root}/runs")


def _text_result(root: str, run: str) -> dict:
    matches = sorted(glob.glob(os.path.join(root, "text_behaviour", run,
                                            "eval_step*.json")))
    if not matches:
        return {}
    return _final_metrics(matches[-1])


def _property_mean(metrics: dict):
    values = [metrics.get(f"in_mixture/mol/{task}/test/roc_auc")
              for task in PROPERTY_SETS]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def _row(name: str, metrics: dict, text: dict, base, base_text, sd):
    mean = _property_mean(metrics)
    row = {"run": name, "property_mean": mean}
    if mean is not None and base is not None:
        row["property_delta"] = mean - base
        row["property_sd"] = (mean - base) / sd if sd else None
    for task in PROPERTY_SETS:
        key = f"in_mixture/mol/{task}/test/roc_auc"
        if key in metrics:
            row[task] = metrics[key]
    row["g2s"] = metrics.get("in_mixture/mol/g2s/test/exact_match")
    row["chebi_rouge_l"] = metrics.get("in_mixture/mol/chebi20/test/rouge_l")
    row["caption_plain"] = text.get("text_behaviour/plain/on/caption_rate")
    row["caption_system"] = text.get("text_behaviour/system/on/caption_rate")
    row["kl_plain"] = text.get("text_behaviour/plain/divergence/kl_mean")
    row["chars_on"] = text.get("text_behaviour/plain/on/chars_mean")
    row["chars_off"] = text.get("text_behaviour/plain/off/chars_mean")
    row["_worst_probe"] = ("", 0.0)
    return row


def _worst_probe(metrics: dict, baseline: dict):
    """The Tier-A probe that moved furthest from the baseline, and by how much."""
    worst, name = 0.0, ""
    for key in PROBES:
        if key in metrics and key in baseline:
            delta = metrics[key] - baseline[key]
            if abs(delta) > abs(worst):
                worst, name = delta, key.split("/")[2]
    return name, worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seeds", nargs="*", default=[],
                        help="the baseline's sibling seeds, for the seed sd")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.results)
    baseline = _anneal_result(root, args.baseline)
    baseline_text = _text_result(root, args.baseline)
    base_mean = _property_mean(baseline)

    means = [base_mean] + [_property_mean(_anneal_result(root, seed))
                           for seed in args.seeds]
    means = [m for m in means if m is not None]
    sd = statistics.stdev(means) if len(means) > 1 else None

    rows = []
    for name in [args.baseline] + list(args.runs):
        metrics = baseline if name == args.baseline else _anneal_result(root, name)
        text = baseline_text if name == args.baseline else _text_result(root, name)
        row = _row(name, metrics, text, base_mean, baseline_text, sd)
        row["_worst_probe"] = _worst_probe(metrics, baseline)
        row["g2s_delta"] = (None if row["g2s"] is None or baseline.get(
            "in_mixture/mol/g2s/test/exact_match") is None else
            row["g2s"] - baseline["in_mixture/mol/g2s/test/exact_match"])
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=1, sort_keys=True, default=str))
        return 0

    print(f"baseline {args.baseline}; property mean {base_mean:.4f}"
          + (f", seed sd {sd:.4f} over {len(means)} seeds" if sd else ""))
    header = (f"{'run':28s}{'prop':>8}{'d':>8}{'sd':>7}{'g2s':>8}{'d':>8}"
              f"{'rougeL':>8}{'cap':>7}{'capsys':>8}{'kl':>7}{'chars':>7}")
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(key, width=8, places=4):
            value = row.get(key)
            return f"{value:{width}.{places}f}" if isinstance(value, float) else f"{'-':>{width}}"
        print(f"{row['run'][:28]:28s}{fmt('property_mean')}{fmt('property_delta')}"
              f"{fmt('property_sd', 7, 1)}{fmt('g2s')}{fmt('g2s_delta')}"
              f"{fmt('chebi_rouge_l')}{fmt('caption_plain', 7, 3)}"
              f"{fmt('caption_system')}{fmt('kl_plain', 7, 3)}"
              f"{fmt('chars_on', 7, 0)}")

    print()
    print("per-set ROC-AUC, as deltas against the baseline")
    print(f"{'run':28s}" + "".join(f"{task:>9}" for task in PROPERTY_SETS)
          + f"{'worst probe':>22}")
    for row in rows:
        cells = ""
        for task in PROPERTY_SETS:
            value, base = row.get(task), baseline.get(
                f"in_mixture/mol/{task}/test/roc_auc")
            cells += (f"{value - base:+9.4f}" if isinstance(value, float)
                      and isinstance(base, float) else f"{'-':>9}")
        name, delta = row["_worst_probe"]
        print(f"{row['run'][:28]:28s}{cells}"
              + (f"{name + ' ' + format(delta, '+.3f'):>22}" if name else f"{'-':>22}"))

    print()
    print("verdict against §9.1's bars")
    for row in rows[1:]:
        checks = []
        for key, label in (("caption_plain", "caption plain"),
                           ("caption_system", "caption system")):
            value = row.get(key)
            if isinstance(value, float):
                checks.append(f"{label} {'pass' if value <= CAPTION_BAR else 'FAIL'}"
                              f" ({value:.3f})")
        kl, base_kl = row.get("kl_plain"), rows[0].get("kl_plain")
        if isinstance(kl, float) and isinstance(base_kl, float):
            checks.append(f"kl {'pass' if kl < base_kl else 'FAIL'} ({kl:.3f} vs {base_kl:.3f})")
        if isinstance(row.get("property_sd"), float):
            checks.append(f"property {'pass' if abs(row['property_sd']) <= 1 else 'FAIL'}"
                          f" ({row['property_sd']:+.1f} sd)")
        if isinstance(row.get("g2s_delta"), float):
            checks.append(f"g2s {'pass' if abs(row['g2s_delta']) <= G2S_BAR else 'FAIL'}"
                          f" ({row['g2s_delta']:+.3f})")
        name, delta = row["_worst_probe"]
        if name:
            checks.append(f"probes {'pass' if abs(delta) <= PROBE_BAR else 'FAIL'}"
                          f" (worst {name} {delta:+.3f})")
        print(f"  {row['run']}: " + "; ".join(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
