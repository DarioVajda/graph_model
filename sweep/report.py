"""
Post-hoc aggregation over a finished (or partial) sweep.

Because every run logs its own line independently (subprocesses / separate sbatch
jobs), no single process sees the whole sweep — aggregation is this separate pass
over ``<sweep_dir>/runs.jsonl``. The runner invokes :func:`write_report`
automatically after a *local* sweep; for sbatch sweeps you run it once the jobs
finish::

    python -m sweep.report <sweep_dir>

This module is experiment-agnostic: rows are sorted by run id (``runs.jsonl``
is in *completion* order — parallel sbatch jobs finish shuffled), columns whose
value is identical across all runs are pulled out into a "shared config" list,
and the table shows one row per run over the remaining (varying) columns. An
experiment that knows its own metrics (which column is "accuracy", which axes
to average over) can build a richer table on top of :func:`load_runs` and
:func:`format_table`.
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
        # Fixed-point at 4 decimals reads best for the metrics that dominate these
        # tables (accuracies, losses), but silently renders small-magnitude values
        # as "0.0000" — learning rates especially (3e-5 is a real setting, not a
        # zero). Fall back to significant-figure notation below the point where
        # 4 decimals stop carrying information.
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.3g}"
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return "" if v is None else str(v)


def _sort_key(run):
    """Stable run identity: sweep_run ("0003_..."), else run_name, else last."""
    return (0, run["sweep_run"]) if "sweep_run" in run else \
           (1, run.get("run_name") or "")


def split_constant_columns(runs, columns=None):
    """Partition columns into ``(varying, constants)``.

    ``constants`` maps column -> rendered value for every column whose value is
    identical across all runs (a column missing from some runs counts as
    varying). With fewer than two runs nothing is "constant", so everything
    stays in the table.
    """
    columns = columns if columns is not None else _columns(runs)
    if len(runs) < 2:
        return columns, {}
    varying, constants = [], {}
    for c in columns:
        vals = {_cell(r.get(c)) for r in runs}
        if len(vals) == 1 and all(c in r for r in runs):
            constants[c] = vals.pop()
        else:
            varying.append(c)
    return varying, constants


def format_markdown_table(rows, columns=None):
    """Render a list of dicts as a GitHub-flavored markdown pipe table.

    Columns whose every non-empty cell is numeric are right-aligned.
    """
    if not rows:
        return "(no runs)"
    columns = columns if columns is not None else _columns(rows)
    cells = [{c: _cell(r.get(c)).replace("|", "\\|") for c in columns} for r in rows]

    def _numeric(col):
        vals = [r.get(col) for r in rows if r.get(col) is not None]
        return bool(vals) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                  for v in vals)

    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(" ---: " if _numeric(c) else " --- " for c in columns) + "|"
    body = ["| " + " | ".join(row[c] for c in columns) + " |" for row in cells]
    return "\n".join([header, sep] + body)


def format_table(rows, columns=None):
    """Render a list of dicts as a fixed-width text table (for terminal use)."""
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
    runs = sorted(load_runs(sweep_dir), key=_sort_key)
    name = os.path.basename(os.path.normpath(sweep_dir))
    varying, constants = split_constant_columns(runs)
    text = (f"# Sweep report: {name}\n\n"
            f"{len(runs)} run(s) recorded.\n\n")
    if constants:
        shared = "\n".join(f"- `{k}` = {v}" for k, v in constants.items())
        text += f"Shared across all runs:\n\n{shared}\n\n"
    text += format_markdown_table(runs, columns=varying) + "\n"
    with open(os.path.join(sweep_dir, "report.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m sweep.report <sweep_dir>", file=sys.stderr)
        return 2
    sweep_dir = argv[0]
    text = write_report(sweep_dir)
    print(text)
    print(f"[report] wrote {os.path.join(sweep_dir, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
