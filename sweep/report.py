"""
Post-hoc aggregation over a finished (or partial) sweep.

Because every run logs its own line independently (subprocesses / separate sbatch
jobs), no single process sees the whole sweep — aggregation is this separate pass
over ``<sweep_dir>/runs.jsonl``. The runner invokes :func:`write_report`
automatically after a *local* sweep; for sbatch sweeps you run it once the jobs
finish::

    python -m sweep.report <sweep_dir>

This module is experiment-agnostic: it prints one row per run over the union of
columns. An experiment that knows its own metrics (which column is "accuracy",
which axes to average over) can build a richer table on top of :func:`load_runs`
and :func:`format_table`.
"""

import json
import os
import sys

# Columns injected by the runner's bookkeeping; not interesting in the table.
_HIDDEN = {"timestamp"}


def load_runs(sweep_dir):
    """Load a sweep's ``runs.jsonl`` (``[]`` if none yet)."""
    path = os.path.join(sweep_dir, "runs.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _columns(runs):
    """Ordered union of keys across runs (first-seen order), minus hidden ones."""
    cols = []
    for r in runs:
        for k in r:
            if k not in cols and k not in _HIDDEN:
                cols.append(k)
    return cols


def _cell(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return "" if v is None else str(v)


def format_table(rows, columns=None):
    """Render a list of dicts as a fixed-width text table."""
    if not rows:
        return "(no runs)"
    columns = columns or _columns(rows)
    widths = {c: len(c) for c in columns}
    cells = []
    for r in rows:
        row = {c: _cell(r.get(c)) for c in columns}
        for c in columns:
            widths[c] = max(widths[c], len(row[c]))
        cells.append(row)
    line = lambda row: "  ".join(row[c].ljust(widths[c]) for c in columns)
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    return "\n".join([header, sep] + [line(r) for r in cells])


def write_report(sweep_dir):
    """Write ``<sweep_dir>/report.md`` from ``runs.jsonl`` and return its text."""
    runs = load_runs(sweep_dir)
    name = os.path.basename(os.path.normpath(sweep_dir))
    table = format_table(runs)
    text = (f"# Sweep report: {name}\n\n"
            f"{len(runs)} run(s) recorded.\n\n"
            f"```\n{table}\n```\n")
    with open(os.path.join(sweep_dir, "report.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m sweep.report <sweep_dir>", file=sys.stderr)
        return 2
    sweep_dir = argv[0]
    runs = load_runs(sweep_dir)
    print(format_table(runs))
    write_report(sweep_dir)
    print(f"\n[report] wrote {os.path.join(sweep_dir, 'report.md')} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
