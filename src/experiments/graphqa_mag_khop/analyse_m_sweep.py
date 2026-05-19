"""
Analyse the M-sweep results and write a markdown table to m_sweep_results.md.

Reads results.json, groups runs by (K, M), and reports mean ± std EM accuracy.
M=0 means all eigenvectors were used (no truncation).

Usage:
    python -m src.experiments.graphqa_mag_khop.analyse_m_sweep
"""

import json
import os
import re
import statistics

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH   = os.path.join(EXPERIMENT_DIR, "results.json")
OUTPUT_PATH    = os.path.join(EXPERIMENT_DIR, "m_sweep_results.md")

# Pattern: GraphQA_shortest_path_K=<k>_M=<m>[_v<n>]
RUN_RE = re.compile(r"GraphQA_shortest_path_K=(\d+)_M=(\d+)(?:_v\d+)?$")


def load_results():
    with open(RESULTS_PATH) as f:
        raw = json.load(f)

    groups: dict[tuple[int, int], list[float]] = {}
    skipped = []

    for run_name, metrics in raw.items():
        m = RUN_RE.match(run_name)
        if not m:
            skipped.append(run_name)
            continue
        k, mag_m = int(m.group(1)), int(m.group(2))
        acc = metrics.get("test_em_accuracy")
        if acc is None:
            skipped.append(run_name)
            continue
        groups.setdefault((k, mag_m), []).append(acc)

    if skipped:
        print(f"Skipped {len(skipped)} unrecognised entries: {skipped}")

    return groups


def format_cell(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1)"
    mean = statistics.mean(values)
    std  = statistics.stdev(values)
    return f"{mean:.3f} ± {std:.3f} (n={len(values)})"


def build_table(groups: dict[tuple[int, int], list[float]]) -> str:
    k_values   = sorted({k for k, _ in groups})
    m_values   = sorted({m for _, m in groups})

    # Header
    col_header = " | ".join(f"K={k}" for k in k_values)
    header     = f"| M | {col_header} |"
    separator  = "| " + " | ".join(["---"] * (len(k_values) + 1)) + " |"

    rows = [header, separator]
    for m in m_values:
        label = "all" if m == 0 else str(m)
        cells = []
        for k in k_values:
            vals = groups.get((k, m), [])
            cells.append(format_cell(vals) if vals else "—")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def main():
    groups = load_results()

    # Summary stats to stdout
    print("\nRaw groups (K, M) → accuracies:")
    for (k, m), vals in sorted(groups.items()):
        label = "all" if m == 0 else str(m)
        mean  = statistics.mean(vals)
        std   = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  K={k}, M={label:>3}: {mean:.3f} ± {std:.3f}  (runs: {[round(v, 3) for v in sorted(vals, reverse=True)]})")

    table = build_table(groups)

    md = f"""# M-sweep Results

Effect of truncating the magnetic Laplacian eigenvectors to M on GraphQA
shortest-path exact-match accuracy. M=all means no truncation.
Graph sizes range from 6 to 20 nodes (mean 13), so M=16 ≈ M=all for most graphs.

{table}

*Values are mean ± std over multiple seeds. Single runs shown as bare accuracy.*
"""

    with open(OUTPUT_PATH, "w") as f:
        f.write(md)

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
