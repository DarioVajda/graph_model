"""relbench ``Database`` -> PyG ``HeteroData``: topology and time, nothing else.

Deliberately not ``relbench.modeling.graph.make_pkey_fkey_graph``. That builds a
``torch_frame.TensorFrame`` of encoded features for every column, which is the exact
compression step this experiment exists to remove: our columns become *text* (``row_text``),
never tensors. Bypassing it also drops a ``torch_frame`` dependency we would otherwise carry
for nothing.

What we keep is the part that matters for sampling:

* one node type per table, node id == row index (relbench guarantees ``pkey`` is a
  consecutive ``0..n-1`` range, so no id maps are needed anywhere in the pipeline);
* ``data[table].time``, int64 unix seconds, which drives the temporal filter;
* one edge type per foreign key, oriented **child -> parent** (a fact row referencing its
  dimension row), then ``ToUndirected`` so the sampler can walk both ways.

The graph is a pure function of the database, so it is cached once per dataset and shared by
every task, split, arm and seed.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# Time assigned to rows of tables with no `time_col` -- the static dimension tables
# (`drivers`/`circuits`/`constructors` in rel-f1; `conditions`/`sponsors`/`interventions`/
# `facilities` in rel-trial). They carry no event time, so they must be eligible at every
# seed timestamp, which means a value below every real timestamp.
#
# Not `iinfo(int64).min`: PyG and anything downstream may take a difference of two times, and
# the minimum would overflow. -(2**40) seconds is roughly year -32,800 -- far below any real
# data, with 23 orders of headroom before overflow.
STATIC_TIME = -(2 ** 40)

# Edge-type name for a foreign key. Kept distinct per fkey column because one child table can
# reference the same parent twice (rel-trial's junction tables point at `studies` and at a
# dimension table; a schema with `from_account`/`to_account` would collide otherwise).
def fkey_edge_name(fkey_col):
    return f"f2p_{fkey_col}"


def link_tables(db):
    """Tables whose every column is a pkey, fkey or timestamp -- pure many-to-many joins.

    rel-trial has three (`conditions_studies`, `facilities_studies`, `interventions_studies`);
    rel-f1 has none. A row of one carries *no information beyond the link itself*: the
    content lives one hop further out, in `conditions.mesh_term` or `facilities.name`.

    The sampler needs this because a node budget spent per-row would be consumed entirely by
    junction rows on a schema like rel-trial's, starving exactly the dimension rows that
    carry the text (PLAN.md 4.2). Derived from the schema alone -- no table names anywhere.
    """
    out = set()
    for name, table in db.table_dict.items():
        structural = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
        structural.discard(None)
        if not [c for c in table.df.columns if c not in structural]:
            out.add(name)
    return out


def _time_seconds(table):
    """int64 unix seconds for a table's rows, or the static sentinel if it has no time."""
    n = len(table.df)
    if table.time_col is None:
        return torch.full((n,), STATIC_TIME, dtype=torch.int64)

    col = table.df[table.time_col]
    if not pd.api.types.is_datetime64_any_dtype(col):
        raise TypeError(
            f"time column {table.time_col!r} is {col.dtype}, expected datetime64. "
            "relbench normally normalizes this; investigate rather than coercing.")

    secs = col.astype("int64") // 10 ** 9
    # NaT becomes a huge negative number under astype, which would silently read as "very
    # old" and therefore "always eligible" -- the exact direction that leaks nothing but
    # quietly corrupts recency ordering. Make it the explicit sentinel instead.
    nat = col.isna().to_numpy()
    out = torch.from_numpy(secs.to_numpy().copy())
    if nat.any():
        out[torch.from_numpy(nat)] = STATIC_TIME
    return out


def build_hetero_data(db, verbose=True):
    """Build the (undirected, time-annotated) heterogeneous graph for a whole database."""
    data = HeteroData()

    for name, table in db.table_dict.items():
        n = len(table.df)
        data[name].num_nodes = n
        data[name].time = _time_seconds(table)

        # The pipeline treats a row index as a node id everywhere -- `n_id` straight back to
        # `df.iloc[...]`. That is only valid because relbench reindexes primary keys to a
        # consecutive range in `validate_and_correct_db`. Assert rather than trust: a silent
        # violation would misalign every row's text with its topology.
        if table.pkey_col is not None:
            pkeys = table.df[table.pkey_col].to_numpy()
            if not np.array_equal(pkeys, np.arange(n)):
                raise ValueError(
                    f"table {name!r} has a non-consecutive pkey {table.pkey_col!r}; "
                    "row index != node id, so every downstream lookup would be wrong")

    n_edges = 0
    for name, table in db.table_dict.items():
        for fkey_col, parent in table.fkey_col_to_pkey_table.items():
            vals = table.df[fkey_col]
            # relbench nulls fkeys that dangle. Those rows simply have no edge; they are
            # still nodes and can still be sampled from the other direction.
            keep = ~vals.isna().to_numpy()
            child_idx = np.arange(len(table.df), dtype=np.int64)[keep]
            parent_idx = vals.to_numpy()[keep].astype(np.int64)

            n_parent = data[parent].num_nodes
            if parent_idx.size and (parent_idx.min() < 0 or parent_idx.max() >= n_parent):
                raise ValueError(
                    f"{name}.{fkey_col} references {parent} rows outside [0, {n_parent})")

            data[name, fkey_edge_name(fkey_col), parent].edge_index = torch.stack([
                torch.from_numpy(child_idx), torch.from_numpy(parent_idx)])
            n_edges += int(keep.sum())
            if verbose:
                print(f"  edge {name}.{fkey_col} -> {parent}: {int(keep.sum()):,} "
                      f"({int((~keep).sum()):,} dangling)")

    # The sampler needs to walk parent -> child (a driver to its results) as well as
    # child -> parent. `directional` sampling later records only the edges actually
    # traversed, and `sampler.py` re-orients the output back to child -> parent, which is
    # the orientation the magnetic-Laplacian bias reads.
    data = T.ToUndirected()(data)
    if verbose:
        print(f"  {len(data.node_types)} node types, {len(data.edge_types)} edge types "
              f"(after ToUndirected), {n_edges:,} directed fkey edges")
    return data


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def graph_path(processed_dir, dataset):
    return os.path.join(processed_dir, dataset, "graph.pt")


def build_and_cache(db, dataset, processed_dir, db_version="v1", rebuild=False,
                    verbose=True):
    """Build the graph once per dataset and reuse it for every task/split/arm/seed."""
    path = graph_path(processed_dir, dataset)
    meta_path = path + ".json"

    if os.path.exists(path) and os.path.exists(meta_path) and not rebuild:
        with open(meta_path) as fh:
            meta = json.load(fh)
        if meta.get("db_version") == db_version and meta.get("static_time") == STATIC_TIME:
            if verbose:
                print(f"[graph_build] loading cached graph from {path}")
            return torch.load(path, weights_only=False)
        if verbose:
            print(f"[graph_build] cache key changed ({meta.get('db_version')} -> "
                  f"{db_version}); rebuilding")

    if verbose:
        print(f"[graph_build] building {dataset} graph")
    data = build_hetero_data(db, verbose=verbose)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    with open(meta_path, "w") as fh:
        json.dump({
            "dataset": dataset,
            "db_version": db_version,
            "static_time": STATIC_TIME,
            "node_types": {k: int(data[k].num_nodes) for k in data.node_types},
            "edge_types": ["__".join(e) for e in data.edge_types],
        }, fh, indent=2)
    if verbose:
        print(f"[graph_build] wrote {path}")
    return data
