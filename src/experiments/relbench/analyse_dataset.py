"""Structural survey of a RelBench dataset, to size the neighborhood before building it.

PLAN.md 4: "Nothing downstream is designed until these numbers exist. In particular
`max_nodes` and the fanout are chosen *from* the degree distribution, not guessed."

Three sections, in ascending order of how much they constrain the design:

**Schema** -- per table: rows, dtypes, null fractions, time column, foreign keys, and the
p50/p95/max character length of every text-ish column. The length percentiles decide
`max_node_chars` (PLAN.md 6.1) and warn about a free-text column that would blow the token
budget on its own.

**Tasks** -- per task: split sizes and timestamp ranges, `timedelta`, and either the class
balance (binary) or the target quantiles (regression). The quantiles are also what a
`bucket` readout would discretize on (PLAN.md 7.2), so they are recorded even for tasks we
score with `numeric_text`.

**Temporally-filtered degrees** -- the one that actually sets the budget. For a sample of
real (entity, timestamp) seeds drawn from the train table, how many child rows are
*eligible* at that timestamp, per relation? Raw fkey counts would badly overstate this: a
driver has ~300 career results but only the ones before the seed date are visible, and
early-career seeds see almost none. The p50/p90/p99 of this distribution is what
`num_neighbors` and `max_nodes` have to cover.

Everything here is derived from relbench metadata -- no table names, column names or task
names are hard-coded, per PLAN.md 5.0 (A). Run it against any dataset:

    RELBENCH_CACHE_DIR=src/experiments/relbench/raw_data \\
        .venv/bin/python src/experiments/relbench/analyse_dataset.py --dataset rel-f1
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

os.environ.setdefault("RELBENCH_CACHE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "raw_data"))

from relbench.base import TaskType                      # noqa: E402
from relbench.datasets import get_dataset               # noqa: E402
from relbench.tasks import get_task, get_task_names     # noqa: E402


# Columns whose *content* is text a language model would read, as opposed to identifiers.
# Detected by dtype plus a cardinality rule, never by name -- see PLAN.md 5.0 (A).
_TEXT_DTYPES = ("object", "string")


def _jsonable(x):
    """numpy/pandas scalars -> plain Python, so json.dump doesn't choke."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def analyse_schema(db):
    """Per-table shape, null fractions, fkeys, and text-column length percentiles."""
    out = {}
    for name, table in db.table_dict.items():
        df = table.df
        cols = {}
        for col in df.columns:
            series = df[col]
            info = {
                "dtype": str(series.dtype),
                "null_frac": round(float(series.isna().mean()), 4),
                "n_unique": int(series.nunique(dropna=True)),
            }
            # A column is "text-ish" if it is object/string dtype and is not simply an
            # identifier (cardinality == row count would make it a free-form id, which
            # PLAN.md 6.1 drops).
            if str(series.dtype) in _TEXT_DTYPES:
                lengths = series.dropna().astype(str).str.len()
                if len(lengths):
                    info["chars_p50"] = int(lengths.quantile(0.50))
                    info["chars_p95"] = int(lengths.quantile(0.95))
                    info["chars_max"] = int(lengths.max())
            cols[col] = info

        out[name] = {
            "n_rows": int(len(df)),
            "pkey_col": table.pkey_col,
            "time_col": table.time_col,
            "fkeys": dict(table.fkey_col_to_pkey_table),
            "columns": cols,
        }
        if table.time_col is not None:
            t = df[table.time_col]
            out[name]["time_min"] = t.min()
            out[name]["time_max"] = t.max()
    return out


def print_schema(schema):
    print("\n" + "=" * 78)
    print("SCHEMA")
    print("=" * 78)
    for name, s in schema.items():
        span = ""
        if s.get("time_min") is not None:
            span = f"  time[{s['time_col']}]: {s['time_min']:%Y-%m-%d}..{s['time_max']:%Y-%m-%d}"
        print(f"\n{name}  ({s['n_rows']:,} rows)  pkey={s['pkey_col']}{span}")
        if s["fkeys"]:
            print(f"  fkeys: " + ", ".join(f"{c}->{t}" for c, t in s["fkeys"].items()))
        for col, info in s["columns"].items():
            flags = []
            if col == s["pkey_col"]:
                flags.append("PKEY")
            if col in s["fkeys"]:
                flags.append("FKEY")
            if col == s["time_col"]:
                flags.append("TIME")
            # PLAN.md 6.1's auto-derivation drops these three, plus >95% null.
            if info["null_frac"] > 0.95:
                flags.append("NULL>95%")
            if info["n_unique"] == s["n_rows"] and col != s["pkey_col"]:
                flags.append("FREEFORM-ID")
            chars = ""
            if "chars_p50" in info:
                chars = f"  chars p50/p95/max={info['chars_p50']}/{info['chars_p95']}/{info['chars_max']}"
            print(f"    {col:26s} {info['dtype']:12s} null={info['null_frac']:<6.3f} "
                  f"uniq={info['n_unique']:<7d}{chars}  {' '.join(flags)}")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _task_docstring(task):
    """The task's own one-line description, or None if it has no usable one.

    Shared with the question template (PLAN.md 5.0 A) -- this is the single rule that
    decides whether a question can quote relbench's own words or must be built from
    `entity_table`/`target_col`/`task_type`/`timedelta` alone.
    """
    cls = type(task)
    if not cls.__module__.startswith("relbench.tasks."):
        return {"has_docstring": False, "docstring": None,
                "docstring_rejected": f"framework class ({cls.__module__})"}
    doc = " ".join((cls.__dict__.get("__doc__") or "").split())
    if not doc:
        return {"has_docstring": False, "docstring": None,
                "docstring_rejected": "class defines no __doc__"}
    return {"has_docstring": True, "docstring": doc, "docstring_rejected": None}


def analyse_task(dataset_name, task_name):
    task = get_task(dataset_name, task_name, download=True)
    info = {
        "task_type": str(task.task_type),
        "entity_table": getattr(task, "entity_table", None),
        "entity_col": getattr(task, "entity_col", None),
        "target_col": getattr(task, "target_col", None),
        "time_col": getattr(task, "time_col", None),
        "timedelta_days": task.timedelta.days,
        "metrics": [m.__name__ for m in task.metrics],
        # PLAN.md 5.0 (A): the question template falls back to schema metadata when no
        # usable docstring exists. Two ways that happens, both real:
        #   * no docstring at all -- all 14 tasks in `relbench/tasks/dbinfer.py`;
        #   * a docstring that documents the *machinery* rather than the task -- e.g.
        #     `results-position` IS `AutoCompleteTask`, a generic per-column factory whose
        #     docstring reads "Args: dataset: The dataset object...". Splicing that into a
        #     question is worse than having nothing.
        # The discriminator is the defining module: concrete tasks live in
        # `relbench.tasks.*`, framework classes in `relbench.base.*`. No task names.
        "defining_module": type(task).__module__,
        **_task_docstring(task),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        tbl = task.get_table(split, mask_input_cols=False)
        df = tbl.df
        s = {"n_rows": int(len(df))}
        if tbl.time_col is not None and tbl.time_col in df:
            s["time_min"] = df[tbl.time_col].min()
            s["time_max"] = df[tbl.time_col].max()
            s["n_timestamps"] = int(df[tbl.time_col].nunique())
        if task.target_col in df:
            y = df[task.target_col].dropna()
            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                s["positive_rate"] = round(float(y.mean()), 4)
            elif task.task_type == TaskType.REGRESSION:
                s["target_quantiles"] = {
                    f"q{int(q*100):02d}": round(float(y.quantile(q)), 4)
                    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
                s["target_mean"] = round(float(y.mean()), 4)
        info["splits"][split] = s
    return task, info


def print_tasks(tasks):
    print("\n" + "=" * 78)
    print("TASKS")
    print("=" * 78)
    for name, info in tasks.items():
        print(f"\n{name}  [{info['task_type']}]  timedelta={info['timedelta_days']}d")
        print(f"  entity: {info['entity_table']}.{info['entity_col']}  "
              f"target: {info['target_col']}")
        print(f"  metrics: {', '.join(info['metrics'])}")
        doc = info["docstring"] or (
            f"(NONE: {info['docstring_rejected']} -- question falls back to metadata)")
        print(f"  docstring: {doc}")
        for split, s in info["splits"].items():
            extra = ""
            if "positive_rate" in s:
                extra = f"  pos_rate={s['positive_rate']}"
            elif "target_quantiles" in s:
                q = s["target_quantiles"]
                extra = f"  target q10/q50/q90={q['q10']}/{q['q50']}/{q['q90']}"
            span = ""
            if s.get("time_min") is not None:
                span = (f"  {s['time_min']:%Y-%m-%d}..{s['time_max']:%Y-%m-%d} "
                        f"({s['n_timestamps']} timestamps)")
            print(f"    {split:5s} n={s['n_rows']:<7,d}{span}{extra}")


# ---------------------------------------------------------------------------
# Temporally-filtered degrees -- the numbers that set the budget
# ---------------------------------------------------------------------------

def child_relations(db, parent_table):
    """Every (child_table, fkey_col) pointing at `parent_table`. Schema-derived."""
    rels = []
    for name, table in db.table_dict.items():
        for fkey_col, parent in table.fkey_col_to_pkey_table.items():
            if parent == parent_table:
                rels.append((name, fkey_col))
    return rels


def parent_relations(db, entity_table):
    """Every (fkey_col, parent_table) the entity table itself points at.

    Needed because not every entity is a dimension row. `results-position` seeds on a
    *fact* table, which has no children at all -- its neighborhood is entirely upward,
    through its own fkeys. Counting only children would report degree 0 and silently
    suggest the neighborhood is empty.
    """
    table = db.table_dict[entity_table]
    return list(table.fkey_col_to_pkey_table.items())


def analyse_degrees(db, task, n_seeds=500, seed=0):
    """How many child rows is a seed actually allowed to see, per relation?

    Sampled from the *train* table so the seeds are real (entity, timestamp) pairs with the
    real distribution over career stage -- a rookie's first race and a veteran's last one
    have wildly different eligible counts, and the budget has to cover both.
    """
    entity_table = task.entity_table
    rels = child_relations(db, entity_table)
    train = task.get_table("train", mask_input_cols=False).df

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(n_seeds, len(train)), replace=False)
    seeds = train.iloc[idx]

    # Pre-index each child relation by fkey value so the per-seed work is a dict lookup
    # plus a comparison, not a scan of the whole table.
    prepared = {}
    for child_table, fkey_col in rels:
        ct = db.table_dict[child_table]
        cdf = ct.df
        times = (cdf[ct.time_col].to_numpy() if ct.time_col is not None else None)
        groups = defaultdict(list)
        for pos, fk in enumerate(cdf[fkey_col].to_numpy()):
            if not pd.isna(fk):
                groups[int(fk)].append(pos)
        prepared[(child_table, fkey_col)] = (
            {k: np.asarray(v) for k, v in groups.items()}, times, ct.time_col)

    counts = {rel: [] for rel in rels}
    for _, row in seeds.iterrows():
        eid = int(row[task.entity_col])
        ts = row[task.time_col]
        for rel, (groups, times, time_col) in prepared.items():
            pos = groups.get(eid)
            if pos is None:
                counts[rel].append(0)
                continue
            if times is None:
                # Static child table: every row is always eligible (PLAN.md 5.1).
                counts[rel].append(len(pos))
            else:
                counts[rel].append(int((times[pos] <= np.datetime64(ts)).sum()))

    out = {}
    for rel, vals in counts.items():
        a = np.asarray(vals)
        out[f"{rel[0]}.{rel[1]}"] = {
            "child_table": rel[0], "fkey_col": rel[1],
            "mean": round(float(a.mean()), 2),
            "p50": int(np.percentile(a, 50)),
            "p90": int(np.percentile(a, 90)),
            "p99": int(np.percentile(a, 99)),
            "max": int(a.max()),
            "zero_frac": round(float((a == 0).mean()), 4),
        }
    if rels:
        totals = np.sum([counts[r] for r in rels], axis=0)
        out["_total_hop1_children"] = {
            "mean": round(float(totals.mean()), 2),
            "p50": int(np.percentile(totals, 50)),
            "p90": int(np.percentile(totals, 90)),
            "p99": int(np.percentile(totals, 99)),
        }
    # Upward relations are a fixed, tiny count (one parent row per fkey) but they are the
    # *entire* neighborhood when the entity is itself a fact row -- see `parent_relations`.
    out["_parents"] = {
        "n_relations": len(parent_relations(db, entity_table)),
        "relations": [f"{c}->{p}" for c, p in parent_relations(db, entity_table)],
    }
    return out


def print_degrees(task_name, deg, n_seeds):
    print(f"\n  {task_name}  (over {n_seeds} sampled train seeds)")
    print(f"    {'relation':34s} {'mean':>8s} {'p50':>6s} {'p90':>6s} {'p99':>6s} "
          f"{'max':>7s} {'zero%':>7s}")
    for key, d in deg.items():
        if key.startswith("_"):
            continue
        print(f"    {key:34s} {d['mean']:8.2f} {d['p50']:6d} {d['p90']:6d} {d['p99']:6d} "
              f"{d['max']:7d} {100*d['zero_frac']:6.1f}%")
    t = deg.get("_total_hop1_children")
    if t:
        print(f"    {'TOTAL hop-1 children':34s} {t['mean']:8.2f} {t['p50']:6d} "
              f"{t['p90']:6d} {t['p99']:6d}")
    else:
        print(f"    (no child relations -- entity is a fact row)")
    p = deg["_parents"]
    print(f"    upward: {p['n_relations']} parent relation(s): {', '.join(p['relations']) or '-'}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="default: every entity task relbench lists for the dataset")
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--out", default=None,
                    help="default: <this dir>/analysis/<dataset>_stats.json")
    args = ap.parse_args()

    print(f"cache dir: {os.environ['RELBENCH_CACHE_DIR']}")
    db = get_dataset(args.dataset, download=True).get_db()

    schema = analyse_schema(db)
    print_schema(schema)

    task_names = args.tasks or list(get_task_names(args.dataset))
    tasks, degrees, skipped = {}, {}, {}
    for name in task_names:
        try:
            task, info = analyse_task(args.dataset, name)
        except Exception as exc:                      # noqa: BLE001
            skipped[name] = f"{type(exc).__name__}: {exc}"
            continue
        # Entity tasks only: link prediction needs a different eval architecture and is an
        # explicit non-goal for phase 1 (PLAN.md 1).
        if not hasattr(task, "entity_table"):
            skipped[name] = "not an EntityTask"
            continue
        tasks[name] = info
        degrees[name] = analyse_degrees(db, task, n_seeds=args.n_seeds)

    print_tasks(tasks)
    if skipped:
        print("\n  skipped: " + ", ".join(f"{k} ({v})" for k, v in skipped.items()))

    print("\n" + "=" * 78)
    print("TEMPORALLY-FILTERED HOP-1 DEGREES  (this is what sizes max_nodes / fanout)")
    print("=" * 78)
    for name, deg in degrees.items():
        print_degrees(name, deg, args.n_seeds)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analysis",
        f"{args.dataset}_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(_jsonable({"dataset": args.dataset, "schema": schema,
                             "tasks": tasks, "degrees": degrees,
                             "skipped": skipped, "n_seeds": args.n_seeds}), fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
