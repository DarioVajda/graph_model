"""
Render the paper's GraphQA tables (LaTeX) from a sweep's ``runs.jsonl``.

    python3 -m src.experiments.graphqa.analysis.prep_table src/experiments/graphqa/results/003_ablation

Table 1 compares GTLM (both graph encodings) against the published baselines; Table 2
is the bias ablation. Cells are mean ± sample std of **test** exact match over seeds,
in percent; a cell with fewer than ``--min-seeds`` runs prints "?" rather than a
number computed from too few points.

Runs are grouped on the record's own ``arm`` / ``task`` / ``graph_type`` fields. The
previous version of this script recovered those by regex-matching run *names* against
``results.json``, which meant a renamed run silently dropped out of the table; the
fields are now written by ``train.py`` into each record.
"""

import argparse
import math
import os
import sys
from collections import defaultdict

from sweep.report import load_runs

# Published baselines: Zero-Shot LLM, then the top-3 GraphToken graph encodings.
BASELINES = {
    'node_count':        [21.7, "79.2\\llnote{MPNN}", "91.2\\llnote{MHA}", "99.6\\llnote{NS}"],
    'edge_count':        [12.4, "26.4\\llnote{MHA}", "36.8\\llnote{MPNN}", "42.6\\llnote{ES}"],
    'cycle_check':       [76.0, "96.2\\llnote{MHA}", "96.4\\llnote{GCN}", "96.4\\llnote{ES}"],
    'triangle_counting': [1.5, "23.4\\llnote{HGT}", "26.6\\llnote{MHA}", "34.8\\llnote{MPNN}"],
    'node_degree':       [14.0, "26.6\\llnote{HGT}", "55.2\\llnote{MHA}", "96.2\\llnote{MPNN}"],
    'connected_nodes':   [14.7, "24.4\\llnote{MHA}", "25.0\\llnote{MPNN}", "26.4\\llnote{GCN}"],
    'reachability':      [84.9, "93.2\\llnote{MHA}", "94.2\\llnote{NS}", "94.4\\llnote{HGT}"],
    'edge_existence':    [44.5, "68.0\\llnote{GCN}", "71.8\\llnote{HGT}", "73.8\\llnote{MHA}"],
    'shortest_path':     [11.5, "60.4\\llnote{GCN}", "60.8\\llnote{MHA}", "63.8\\llnote{MPNN}"],
}

T1A_TASKS = ['node_count', 'edge_count', 'cycle_check', 'triangle_counting']
T1B_TASKS = ['node_degree', 'connected_nodes', 'reachability', 'edge_existence', 'shortest_path']
ABLATION_TASKS = T1A_TASKS + T1B_TASKS
ABLATION_ARMS = [("Standard", "base"), ("w/o SPD", "no-spd"),
                 ("w/o RRWP", "no-rrwp"), ("w/o Magnetic", "no-magnetic")]


def collect(runs, metric="test_accuracy", min_seeds=3):
    """Group runs into ``(arm, task, graph_type) -> (mean%, std%)`` over seeds."""
    values = defaultdict(list)
    for r in runs:
        if r.get("mode") != "train" or r.get(metric) is None:
            continue
        key = (r.get("arm"), r.get("task"), r.get("graph_type"))
        if None in key:
            continue
        values[key].append(r[metric])

    stats = {}
    for key, vals in values.items():
        if len(vals) < min_seeds:
            stats[key] = "?"
            continue
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
        stats[key] = (mean * 100, math.sqrt(variance) * 100)
    return stats, {k: len(v) for k, v in values.items()}


def parse_baseline(b):
    """Parse a baseline entry into ``{'val': float, 'suffix': str}``."""
    if isinstance(b, (int, float)):
        return {'val': float(b), 'suffix': ''}
    import re
    m = re.match(r"([0-9.]+)(.*)", str(b))
    return {'val': float(m.group(1)), 'suffix': m.group(2)}


def build_table1_columns(stats):
    """Per task: the 4 baseline rows + GTLM incidence + GTLM standard, rank-formatted."""
    cols = {}
    for task in BASELINES:
        cells = [parse_baseline(b) for b in BASELINES[task]]
        for graph_type in ("incidence", "standard"):
            res = stats.get(("base", task, graph_type), "?")
            if res == "?":
                cells.append({'val': None, 'suffix': ''})
            else:
                cells.append({'val': res[0], 'suffix': f"\\std{{{res[1]:.1f}}}"})
        cols[task] = cells

    # Bold the best and underline the runner-up, per task column.
    for task, cells in cols.items():
        vals = sorted({round(c['val'], 1) for c in cells if c['val'] is not None}, reverse=True)
        best = vals[0] if vals else None
        second = vals[1] if len(vals) > 1 else None
        for c in cells:
            if c['val'] is None:
                c['fmt'] = "?"
                continue
            rnd = round(c['val'], 1)
            s = "100" if rnd == 100.0 else f"{rnd:.1f}"
            if best is not None and rnd == best:
                s = f"\\textbf{{{s}}}"
            elif second is not None and rnd == second:
                s = f"\\underline{{{s}}}"
            c['fmt'] = s + c['suffix']
    return cols


def render(stats):
    cols = build_table1_columns(stats)

    def row_t1(idx, tasks):
        return " & ".join(cols[t][idx]['fmt'] for t in tasks)

    def cell_t2(arm, task):
        res = stats.get((arm, task, "standard"), "?")
        if res == "?":
            return "?"
        mean, std = res
        val = "100" if round(mean, 1) == 100.0 else f"{mean:.1f}"
        return f"{val} $\\pm$ {std:.1f}"

    def row_t2(arm):
        return " & ".join(cell_t2(arm, t) for t in ABLATION_TASKS)

    table_1 = r"""\begin{table}[ht]
    \centering
    \footnotesize % Scales down the text cleanly
    \renewcommand{\arraystretch}{0.9} % Compresses vertical space between rows
    \setlength{\tabcolsep}{4pt} % Tightens horizontal space between columns
    \caption{Accuracy ($\%$) Comparison on \textbf{GraphQA} tasks. The baselines are a Zero-Shot LLM and the top-3 highest scoring graph encoding methods used by GraphToken. The highest score in each task is \textbf{bold} and the second highest is \underline{underlined}.}
    \label{tab:graphqa-results}

    % --- FIRST PART: GRAPH TASKS ---
    \textbf{(a) Graph Tasks} \\
    \vspace{0.2em}
    \begin{tabular}{l | c c c c}
        \toprule
        \textbf{Method} & \textbf{Node Count} & \textbf{Edge Count} & \textbf{Cycle Check} & \textbf{Triangle Counting} \\
        \midrule
        Zero-Shot LLM       & """ + row_t1(0, T1A_TASKS) + r""" \\
        3rd GraphToken      & """ + row_t1(1, T1A_TASKS) + r""" \\
        2nd GraphToken      & """ + row_t1(2, T1A_TASKS) + r""" \\
        1st GraphToken      & """ + row_t1(3, T1A_TASKS) + r""" \\
        \midrule
        \textbf{GTLM} {\scriptsize(Incidence)} & """ + row_t1(4, T1A_TASKS) + r""" \\
        \textbf{GTLM} {\scriptsize(Standard)}  & """ + row_t1(5, T1A_TASKS) + r""" \\
        \bottomrule
    \end{tabular}

    \vspace{1em} % Minimal gap between sub-tables

    % --- SECOND PART: NODE & EDGE TASKS ---
    \textbf{(b) Node and Edge Tasks} \\
    \vspace{0.2em}
    \begin{tabular}{l | c c | c c c}
        \toprule
        & \multicolumn{2}{c|}{\textbf{Node Tasks}} & \multicolumn{3}{c}{\textbf{Edge Tasks}} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-6}
        \textbf{Method} & \textbf{Node Degree} & \textbf{Connected Nodes \;\;} & \textbf{Reachability} & \textbf{Edge Existence} & \textbf{Shortest Path} \\
        \midrule
        Zero-Shot LLM       & """ + row_t1(0, T1B_TASKS) + r""" \\
        3rd GraphToken      & """ + row_t1(1, T1B_TASKS) + r""" \\
        2nd GraphToken      & """ + row_t1(2, T1B_TASKS) + r""" \\
        1st GraphToken      & """ + row_t1(3, T1B_TASKS) + r""" \\
        \midrule
        \textbf{GTLM} {\scriptsize(Incidence)} & """ + row_t1(4, T1B_TASKS) + r""" \\
        \textbf{GTLM} {\scriptsize(Standard)}  & """ + row_t1(5, T1B_TASKS) + r""" \\
        \bottomrule
    \end{tabular}
\end{table}"""

    ablation_rows = "\n".join(
        f"{label:12s} & " + row_t2(arm) + r" \\" + ("\n\\midrule" if arm == "base" else "")
        for label, arm in ABLATION_ARMS)

    table_2 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Ablation study} results, comparing the accuracies ($\%$) of standard GTLM to its variants without each attention bias type individually.}
\label{tab:ablation-study}
\resizebox{\textwidth}{!}{
\begin{tabular}{l | *{4}{c} | *{2}{c} | *{3}{c}}
\toprule
& \multicolumn{4}{c}{\textbf{Graph Tasks}} & \multicolumn{2}{c}{\textbf{Node Tasks}} & \multicolumn{3}{c}{\textbf{Edge Tasks}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}
\textbf{Method} & \textbf{Node Count} & \textbf{Edge Count} & \textbf{Cycle Check} & \textbf{Tri. Count} & \textbf{Node Deg.} & \textbf{Conn. Nodes} & \textbf{Reachability} & \textbf{Edge Exist.} & \textbf{Shortest Path} \\

\midrule
""" + ablation_rows + r"""

\bottomrule
\end{tabular}
}
\end{table}"""
    return table_1, table_2


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.graphqa.analysis.prep_table",
        description="Render the GraphQA paper tables from a sweep's runs.jsonl.")
    p.add_argument("sweep_dir", help="a sweep directory containing runs.jsonl")
    p.add_argument("--metric", default="test_accuracy")
    p.add_argument("--min-seeds", type=int, default=3,
                   help="cells with fewer runs print '?' instead of a number.")
    args = p.parse_args(argv)

    runs = load_runs(args.sweep_dir)
    if not runs:
        print(f"No runs found in {os.path.join(args.sweep_dir, 'runs.jsonl')}", file=sys.stderr)
        return 1

    stats, counts = collect(runs, metric=args.metric, min_seeds=args.min_seeds)
    missing = [k for k, v in stats.items() if v == "?"]
    print(f"% {len(runs)} runs; {len(stats)} (arm, task, graph_type) cells; "
          f"{len(missing)} under {args.min_seeds} seeds", file=sys.stderr)
    for key in sorted(missing):
        print(f"%   thin cell {key}: {counts[key]} run(s)", file=sys.stderr)

    table_1, table_2 = render(stats)
    print("=" * 60)
    print("TABLE 1: GraphQA Results")
    print("=" * 60)
    print(table_1)
    print("\n" * 3)
    print("=" * 60)
    print("TABLE 2: Ablation Study")
    print("=" * 60)
    print(table_2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
