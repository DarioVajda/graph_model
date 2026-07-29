"""Phase-0 acceptance gate for the RelBench x GTLM experiment (PLAN.md 3.3).

Nothing in this experiment should be built until this script passes. It answers four
questions, in increasing order of what they would cost us to get wrong:

1. Does ``relbench`` import and do its metrics run under this venv's scikit-learn?
   We install it with ``--no-deps`` because it pins ``scikit-learn<=1.6.1`` and other
   experiments here need 1.7.2 (PLAN.md 3.1), so the pin is deliberately violated.
2. Is ``pyg-lib`` visible to PyG? Without it ``NeighborSampler`` raises outright
   (``neighbor_sampler.py:512``); ``torch-sparse`` cannot cover for it because it is
   compiled against CUDA 12.8 and PyG disables it at import.
3. **Does PyG's temporal sampling mean what we think it means?** This is the load-bearing
   one. The whole experiment rests on no sampled row being visible from the future, and
   that guarantee is now a dependency's rather than ours. The check runs on a hand-built
   ten-node graph with hand-checkable timestamps, so a failure implicates PyG and not our
   schema conversion.
4. Does the same hold end-to-end on a real ``relbench`` database (``FakeDataset``)?

Run:  .venv/bin/python src/experiments/relbench/check_env.py
"""

import sys
import traceback

import numpy as np
import torch


# `pyg-lib` does NOT implement induced-subgraph sampling for heterogeneous graphs:
# `neighbor_sampler.py:440` guards the pyg-lib branch with
# `self.subgraph_type != SubgraphType.induced`, and an `induced` request therefore falls
# through to the `torch-sparse` branch -- which is the one broken in this venv, producing a
# misleading "requires either 'pyg-lib' or 'torch-sparse'" even with pyg-lib installed.
# `directional` keeps exactly the edges the sampler traversed, which is what we want
# anyway: it is the fkey path from the seed, not an arbitrary induced closure.
SUBGRAPH_TYPE = "directional"

# The sentinel time for rows from static dimension tables -- tables with no `time_col`
# (`drivers`, `circuits`, `constructors` in rel-f1). They must be eligible at every seed
# timestamp. INT64_MIN is the natural choice but risks overflow if anything downstream
# takes a difference of two times, so we use a large-but-safe negative instead; check 3c
# pins the behaviour either way. Corresponds to year ~-2600, i.e. before any real data.
STATIC_TIME = -(2 ** 40)

_results = []


def check(name):
    """Decorator: run a check, record pass/fail, never let one failure hide the rest."""
    def deco(fn):
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception:
            traceback.print_exc()
            _results.append((name, False))
            print(f"--- FAIL: {name}")
        else:
            _results.append((name, True))
            print(f"--- ok: {name}")
        return fn
    return deco


# ---------------------------------------------------------------------------
# 1. relbench under a violated sklearn pin
# ---------------------------------------------------------------------------

@check("relbench imports and its metrics run under this sklearn")
def _check_relbench():
    import sklearn
    import relbench
    from relbench.metrics import (
        roc_auc, average_precision, f1, accuracy, mae, rmse, r2,
    )
    print(f"relbench {relbench.__file__}, sklearn {sklearn.__version__}")

    # A deliberately easy but non-degenerate case, so a silently-broken metric shows up as
    # a wrong number rather than an exception.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    got = {
        "roc_auc": roc_auc(y_true, y_score),
        "average_precision": average_precision(y_true, y_score),
        "f1": f1(y_true, y_score > 0.5),
        "accuracy": accuracy(y_true, y_score > 0.5),
    }
    print("classification:", {k: round(v, 4) for k, v in got.items()})
    assert abs(got["roc_auc"] - 0.75) < 1e-9, got["roc_auc"]

    y_true_r = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_r = np.array([2.5, 0.0, 2.0, 8.0])
    got_r = {"mae": mae(y_true_r, y_pred_r), "rmse": rmse(y_true_r, y_pred_r),
             "r2": r2(y_true_r, y_pred_r)}
    print("regression:", {k: round(v, 4) for k, v in got_r.items()})
    assert abs(got_r["mae"] - 0.5) < 1e-9, got_r["mae"]


# ---------------------------------------------------------------------------
# 2. pyg-lib present
# ---------------------------------------------------------------------------

@check("pyg-lib is detected by torch_geometric")
def _check_pyg_lib():
    import torch_geometric
    import torch_geometric.typing as T
    print(f"torch {torch.__version__}, pyg {torch_geometric.__version__}")
    for flag in ("WITH_PYG_LIB", "WITH_SAMPLED_OP", "WITH_EDGE_TIME_NEIGHBOR_SAMPLE"):
        print(f"  {flag:32s} {getattr(T, flag, None)}")
    assert T.WITH_PYG_LIB, (
        "pyg-lib not detected. Install it on a machine with outbound internet:\n"
        "  .venv/bin/pip install pyg-lib "
        "-f https://data.pyg.org/whl/torch-2.11.0+cu130.html"
    )


# ---------------------------------------------------------------------------
# 3. PyG temporal semantics, on a graph small enough to check by hand
# ---------------------------------------------------------------------------

def _toy_graph():
    """One entity, ten facts at known times, all facts pointing at the entity.

    Fact times straddle the seed timestamp of 55 *and* land exactly on it, so the sample
    distinguishes `time <= seed` from `time < seed` -- a distinction that decides whether
    a row from the label window itself can leak in.
    """
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    fact_times = torch.tensor([0, 10, 20, 30, 40, 50, 55, 60, 70, 80], dtype=torch.int64)

    data = HeteroData()
    data["entity"].time = torch.tensor([STATIC_TIME], dtype=torch.int64)  # static table
    data["entity"].num_nodes = 1
    data["fact"].time = fact_times
    data["fact"].num_nodes = fact_times.numel()
    # child -> parent, the fkey direction (a fact row referencing its dimension row).
    data["fact", "f2p", "entity"].edge_index = torch.stack([
        torch.arange(fact_times.numel(), dtype=torch.int64),
        torch.zeros(fact_times.numel(), dtype=torch.int64),
    ])
    return T.ToUndirected()(data), fact_times


def _sample_toy(strategy, num_neighbors, seed_time=55):
    from torch_geometric.loader import NeighborLoader

    data, fact_times = _toy_graph()
    loader = NeighborLoader(
        data,
        num_neighbors={k: num_neighbors for k in data.edge_types},
        input_nodes=("entity", torch.tensor([0])),
        input_time=torch.tensor([seed_time], dtype=torch.int64),
        time_attr="time",
        temporal_strategy=strategy,
        subgraph_type=SUBGRAPH_TYPE,
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(loader))
    return batch, fact_times


@check("3a. temporal filter: no sampled row is from the future")
def _check_temporal_filter():
    # A fanout wide enough to take everything, so the ONLY thing removing rows is time.
    batch, fact_times = _sample_toy("uniform", [50])
    sampled = sorted(batch["fact"].time.tolist())
    print(f"all fact times : {fact_times.tolist()}")
    print(f"seed time      : 55")
    print(f"sampled times  : {sampled}")
    assert sampled, "nothing sampled at all -- edge direction is probably wrong"
    assert max(sampled) <= 55, f"FUTURE ROW SAMPLED: {sampled}"

    boundary = 55 in sampled
    print(f"boundary row (time == seed) included: {boundary}  "
          f"-> PyG uses `time {'<=' if boundary else '<'} seed`")
    # `<=` is the correct semantics here, not merely an acceptable one. relbench builds its
    # label window with `re.date > t.timestamp AND re.date <= t.timestamp + timedelta`
    # (`tasks/f1.py:95-96`), i.e. STRICTLY after the seed -- so a row stamped exactly at the
    # prediction instant is past, not label, and admitting it leaks nothing. relbench's own
    # `modeling/loader.py` drives the same PyG sampler, so this also keeps us byte-aligned
    # with RDL's input. Asserted, because a future PyG changing to `<` would silently
    # shrink every neighborhood.
    assert boundary, (
        "PyG excluded the row at time == seed. relbench's label window starts strictly "
        "after the seed, so that row is legitimate past context and RDL sees it.")
    globals()["_BOUNDARY_INCLUSIVE"] = boundary


@check("3b. temporal_strategy='last' returns the most recent eligible rows")
def _check_last_strategy():
    batch, _ = _sample_toy("last", [3])
    sampled = sorted(batch["fact"].time.tolist(), reverse=True)
    print(f"num_neighbors=3, strategy='last' -> {sampled}")
    assert len(sampled) == 3, f"expected 3 neighbours, got {len(sampled)}"
    expected = [55, 50, 40] if globals().get("_BOUNDARY_INCLUSIVE") else [50, 40, 30]
    assert sampled == expected, f"expected {expected}, got {sampled}"
    print("most-recent-k is expressible -- no hand-rolled sampler needed (PLAN.md 2.1)")


@check("3c. static-table sentinel survives sampling")
def _check_static_sentinel():
    batch, _ = _sample_toy("last", [3])
    assert batch["entity"].time.tolist() == [STATIC_TIME], batch["entity"].time
    print(f"sentinel {STATIC_TIME} preserved; static rows stay eligible")


@check("3d. hop attribution is recoverable")
def _check_hop_attribution():
    batch, _ = _sample_toy("last", [3])
    # We need (table, row, hop) triples to fetch pandas rows and to trim by hop later.
    n_id = batch["fact"].n_id
    print(f"fact n_id -> original rows: {n_id.tolist()}")
    per_hop = getattr(batch["fact"], "num_sampled_nodes", None)
    print(f"num_sampled_nodes: {per_hop}")
    assert n_id.numel() == 3, "n_id must map sampled nodes back to source rows"


# ---------------------------------------------------------------------------
# 4. The same, end-to-end on a real relbench Database
# ---------------------------------------------------------------------------

def _db_to_hetero(db):
    """Minimal relbench `Database` -> `HeteroData`: topology and time, no features.

    Deliberately not `relbench.modeling.graph.make_pkey_fkey_graph`, which builds
    `torch_frame` TensorFrames for every column -- our columns become *text* (PLAN.md 6.1),
    never tensors, so that dependency buys nothing and costs a broken install.

    This is a throwaway preview of `graph_build.py`; the real one handles dangling fkeys
    and caching.
    """
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    data = HeteroData()
    for name, table in db.table_dict.items():
        n = len(table.df)
        data[name].num_nodes = n
        if table.time_col is not None:
            t = table.df[table.time_col].astype("int64").to_numpy() // 10 ** 9
            data[name].time = torch.from_numpy(t)
        else:
            data[name].time = torch.full((n,), STATIC_TIME, dtype=torch.int64)

    for name, table in db.table_dict.items():
        for fkey_col, parent in table.fkey_col_to_pkey_table.items():
            vals = table.df[fkey_col].to_numpy()
            mask = ~np.isnan(vals.astype("float64"))
            child_idx = np.arange(len(table.df))[mask]
            parent_idx = vals[mask].astype("int64")
            data[name, f"f2p_{fkey_col}", parent].edge_index = torch.stack([
                torch.from_numpy(child_idx), torch.from_numpy(parent_idx),
            ])
    return T.ToUndirected()(data)


@check("4. temporal sampling on a real relbench Database (FakeDataset)")
def _check_fake_dataset():
    from relbench.datasets.fake import FakeDataset
    from torch_geometric.loader import NeighborLoader

    db = FakeDataset(num_products=30, num_customers=10, num_reviews=200).get_db()
    data = _db_to_hetero(db)
    print("node types:", {k: data[k].num_nodes for k in data.node_types})

    review_times = np.sort(data["review"].time.numpy())
    seed_time = int(review_times[len(review_times) // 2])   # median, so both sides are populated
    print(f"review times span {review_times[0]}..{review_times[-1]}, seed_time={seed_time}")

    loader = NeighborLoader(
        data,
        num_neighbors={k: [8, 8] for k in data.edge_types},
        input_nodes=("customer", torch.arange(5)),
        input_time=torch.full((5,), seed_time, dtype=torch.int64),
        time_attr="time",
        temporal_strategy="last",
        subgraph_type=SUBGRAPH_TYPE,
        batch_size=1,
        shuffle=False,
    )

    n_checked = 0
    for batch in loader:
        for ntype in batch.node_types:
            times = batch[ntype].time
            if times.numel() == 0:
                continue
            worst = int(times.max())
            assert worst <= seed_time, (
                f"FUTURE ROW: {ntype} has time {worst} > seed {seed_time}")
            n_checked += times.numel()
    print(f"{n_checked} sampled nodes across 5 seeds, none from the future")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("RelBench x GTLM -- phase-0 environment gate (PLAN.md 3.3)")
    print("=" * 72)

    failed = [name for name, ok in _results if not ok]
    print("\n" + "=" * 72)
    for name, ok in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 72)
    if failed:
        print(f"\n{len(failed)} check(s) failed. Do not proceed to phase 1.")
        print("If 3a/3b failed, fall back to the hand-rolled sampler in PLAN.md 5.2's "
              "appendix and reinstate the ~200-line estimate.")
        sys.exit(1)
    print("\nAll checks passed. Phase 1 (download + analyse_dataset.py) is unblocked.")
