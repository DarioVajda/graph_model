"""What `text_behaviour` measured, per arm, adapter-on against adapter-off.

Reads whatever `tools/text_behaviour_all.py` has produced and says what is
missing rather than failing, so it is useful while the jobs are still running.

    src/generalist/tools/text_report.py
    src/generalist/tools/text_report.py --config <cfg>

**Read the `off` column first.** It is the same base model in every cell — the
backbone is frozen and `base_exact` says so to 0.0 — so any spread across cells
in that column is measurement noise, and it is the scale against which a
difference in the `on` column means anything. A 20-character gap between arms is
nothing if the two `off` columns already differ by 20.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR = "src/generalist/results/text_behaviour"
DEFAULT_CONFIG = "src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc"

#: ``{label: run-name stem}``. The arm names the write-up uses, not the config's.
ARMS = {"graph": "molecule_generalist_instruct_graph",
        "SMILES": "molecule_generalist_instruct_flat"}
SEEDS = (0, 1, 2)

CONDITIONS = ("plain", "system")

#: ``(leaf, heading, format)``. Order is the report's order.
REGISTER_ROWS = (
    ("caption_rate", "**answered with a molecule caption**", "{:.3f}"),
    ("chars_mean", "characters", "{:.1f}"),
    ("new_tokens_mean", "new tokens", "{:.1f}"),
    ("stop_rate", "stopped", "{:.3f}"),
    ("single_token_rate", "<=1 token", "{:.3f}"),
    ("empty_rate", "empty", "{:.3f}"),
)

DIVERGENCE_ROWS = (
    ("base_continuation_nll", "NLL of the backbone's own words", "{:.4f}"),
    ("kl_mean", "mean KL(off || on) per token", "{:.4f}"),
)


def metrics_for(cell: str) -> dict:
    """The validator's metrics for one cell, or ``{}`` if it has not run.

    The newest `eval_step*.json` wins: re-running the measurement on the same
    checkpoint should report the re-run, not the first attempt.
    """
    files = sorted(glob.glob(os.path.join(REPO, OUT_DIR, cell, "eval_step*.json")),
                   key=os.path.getmtime)
    if not files:
        return {}
    with open(files[-1]) as fh:
        record = json.load(fh)
    return dict((record.get("eval") or {}).get("metrics") or {})


def collect() -> tuple:
    """``({(arm, condition, state, leaf): [per-seed values]}, missing)``."""
    out, missing = {}, []
    for arm, stem in ARMS.items():
        for seed in SEEDS:
            cell = f"{stem}_s{seed}"
            metrics = metrics_for(cell)
            if not metrics:
                missing.append(cell)
                continue
            for key, value in metrics.items():
                parts = key.split("/")
                if len(parts) != 4 or parts[0] != "text_behaviour":
                    continue
                _name, condition, state, leaf = parts
                if not isinstance(value, (int, float)):
                    continue
                out.setdefault((arm, condition, state, leaf), []).append(float(value))
    return out, missing


def cell(values, fmt) -> str:
    if not values:
        return "--"
    mu = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{fmt.format(mu)} ±{fmt.format(sd)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="named for the record; the cells are read off disk.")
    args = ap.parse_args(argv)

    data, missing = collect()
    if missing:
        print(f"[text] not measured yet: {', '.join(sorted(set(missing)))}",
              file=sys.stderr)
    if not data:
        print("nothing to report", file=sys.stderr)
        return 0

    print(f"`text_behaviour` over {args.config}")
    print("mean ±sd over seeds 0/1/2. `off` is the base model exactly "
          "(the backbone is frozen), so it is the control and the noise floor.")

    for condition in CONDITIONS:
        present = any(k[1] == condition for k in data)
        if not present:
            continue
        print(f"\n### {condition}\n")
        print("| | graph on | graph off | SMILES on | SMILES off |")
        print("|---|---|---|---|---|")
        for leaf, heading, fmt in REGISTER_ROWS:
            row = [cell(data.get((arm, condition, state, leaf), []), fmt)
                   for arm in ARMS for state in ("on", "off")]
            print(f"| {heading} | " + " | ".join(row) + " |")

        print()
        print("| divergence from the backbone | graph | SMILES |")
        print("|---|---|---|")
        for leaf, heading, fmt in DIVERGENCE_ROWS:
            row = [cell(data.get((arm, condition, "divergence", leaf), []), fmt)
                   for arm in ARMS]
            print(f"| {heading} | " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
