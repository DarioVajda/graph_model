"""The five invariants the RelBench sampler must hold (PLAN.md 5.3).

Test 1 -- **no sampled row is from the future** -- is the single most important test in the
experiment. A violation makes every task trivial, inflates every number, and fails silently:
nothing downstream can detect it. It is asserted here even though PyG rather than we do the
filtering, because trusting a dependency is not the same as assuming it, and a PyG upgrade
that changed `input_time` semantics would otherwise land unnoticed.

These run on a hand-built database and on `relbench.datasets.fake.FakeDataset`, so no
download is needed and CI stays offline.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from relbench.base import Database, Table
from relbench.datasets.fake import FakeDataset

from src.experiments.relbench.graph_build import (
    STATIC_TIME, build_hetero_data, link_tables,
)
from src.experiments.relbench.sampler import TemporalSampler


DAY = 86_400


# ---------------------------------------------------------------------------
# A tiny database whose every timestamp is checkable by hand.
# ---------------------------------------------------------------------------

def toy_db():
    """Two entities; ten `event` rows each, one per day; one static `kind` dimension.

    `event` day d has time `d * DAY`, so a seed at `5 * DAY` must see days 0..5 and nothing
    later. `kind` has no time column and must therefore always be eligible.
    """
    n_ent, n_day = 2, 10
    entity = pd.DataFrame({"entity_id": range(n_ent), "label": [f"e{i}" for i in range(n_ent)]})
    kind = pd.DataFrame({"kind_id": [0, 1], "name": ["alpha", "beta"]})

    rows = []
    for e in range(n_ent):
        for d in range(n_day):
            rows.append({"event_id": len(rows), "entity_id": e, "kind_id": d % 2,
                         "day": d, "ts": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d)})
    event = pd.DataFrame(rows)

    return Database({
        "entity": Table(df=entity, fkey_col_to_pkey_table={}, pkey_col="entity_id"),
        "kind": Table(df=kind, fkey_col_to_pkey_table={}, pkey_col="kind_id"),
        "event": Table(df=event,
                       fkey_col_to_pkey_table={"entity_id": "entity", "kind_id": "kind"},
                       pkey_col="event_id", time_col="ts"),
    })


@pytest.fixture(scope="module")
def toy():
    db = toy_db()
    return db, build_hetero_data(db, verbose=False)


def _ts(day):
    return int(pd.Timestamp("2020-01-01").timestamp()) + day * DAY


# ---------------------------------------------------------------------------
# 1. No future rows
# ---------------------------------------------------------------------------

def test_no_future_rows_toy(toy):
    """Hand-checkable: a seed on day 5 sees days 0..5, never 6..9."""
    db, data = toy
    events = db.table_dict["event"].df
    sampler = TemporalSampler(data, "entity", max_nodes=64)

    for seed_day in range(10):
        graph = sampler.sample(0, _ts(seed_day))
        days = [events.iloc[row]["day"] for table, row, _ in graph.nodes if table == "event"]
        assert all(d <= seed_day for d in days), (
            f"seed day {seed_day} sampled future events {sorted(days)}")
        # And it really did retrieve the past, rather than passing by returning nothing.
        assert len(days) == seed_day + 1, (
            f"seed day {seed_day} expected {seed_day + 1} events, got {sorted(days)}")


def test_no_future_rows_fake_dataset():
    """The same invariant end to end on a real `relbench` Database."""
    db = FakeDataset(num_products=30, num_customers=10, num_reviews=300).get_db()
    data = build_hetero_data(db, verbose=False)
    times = {name: t.df[t.time_col].astype("int64") // 10 ** 9
             for name, t in db.table_dict.items() if t.time_col is not None}

    review_times = np.sort(times["review"].to_numpy())
    sampler = TemporalSampler(data, "customer", max_nodes=32)

    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        seed_ts = int(review_times[int(q * (len(review_times) - 1))])
        for customer in range(10):
            graph = sampler.sample(customer, seed_ts)
            for table, row, _ in graph.nodes:
                if table in times:
                    assert int(times[table].iloc[row]) <= seed_ts, (
                        f"{table} row {row} is from after seed {seed_ts}")


def test_static_rows_are_always_eligible(toy):
    """A table with no `time_col` must never be excluded by the temporal filter."""
    db, data = toy
    assert (data["kind"].time == STATIC_TIME).all()
    graph = TemporalSampler(data, "entity", max_nodes=64).sample(0, _ts(9))
    assert any(t == "kind" for t, _, _ in graph.nodes), (
        "static dimension rows were filtered out; the sentinel is not below real timestamps")


# ---------------------------------------------------------------------------
# 2. Budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_nodes", [1, 2, 4, 8, 16])
def test_budget_respected(toy, max_nodes):
    db, data = toy
    sampler = TemporalSampler(data, "entity", max_nodes=max_nodes)
    graph = sampler.sample(0, _ts(9))
    assert len(graph) <= max_nodes
    assert len(graph) >= 1                       # the seed always survives


def test_link_rows_do_not_consume_budget():
    """On a junction schema the budget must buy *content*, not join rows (PLAN.md 4.2).

    Built as a mirror of rel-trial: `entity` -- `bridge` (pure link) -- `dimension`. Without
    the exemption the whole budget goes to `bridge` rows and no `dimension` row -- the one
    carrying the text -- ever appears.
    """
    n = 12
    entity = pd.DataFrame({"entity_id": [0], "label": ["seed"]})
    dimension = pd.DataFrame({"dim_id": range(n), "name": [f"name-{i}" for i in range(n)]})
    bridge = pd.DataFrame({
        "bridge_id": range(n), "entity_id": [0] * n, "dim_id": range(n),
        "ts": [pd.Timestamp("2020-01-01")] * n})

    db = Database({
        "entity": Table(df=entity, fkey_col_to_pkey_table={}, pkey_col="entity_id"),
        "dimension": Table(df=dimension, fkey_col_to_pkey_table={}, pkey_col="dim_id"),
        "bridge": Table(df=bridge,
                        fkey_col_to_pkey_table={"entity_id": "entity", "dim_id": "dimension"},
                        pkey_col="bridge_id", time_col="ts"),
    })
    assert link_tables(db) == {"bridge"}, "schema rule failed to spot the pure join table"

    data = build_hetero_data(db, verbose=False)
    seed_ts = int(pd.Timestamp("2020-06-01").timestamp())

    # `collapse_links=False` isolates the budget question from the contraction question.
    naive = TemporalSampler(data, "entity", max_nodes=6,
                            collapse_links=False).sample(0, seed_ts)
    aware = TemporalSampler(data, "entity", max_nodes=6, link_tables={"bridge"},
                            collapse_links=False).sample(0, seed_ts)

    n_dim_naive = sum(t == "dimension" for t, _, _ in naive.nodes)
    n_dim_aware = sum(t == "dimension" for t, _, _ in aware.nodes)

    # Naive: `bridge` and `dimension` are two tables competing for one budget, so the join
    # rows take roughly half of it despite carrying nothing.
    assert n_dim_naive < 5, (
        f"precondition: join rows should take a share of the naive budget, "
        f"but {n_dim_naive}/5 content rows survived")
    # Link-aware: the whole budget (max_nodes minus the seed) buys content.
    assert n_dim_aware == 5, f"expected the budget to buy 5 dimension rows, got {n_dim_aware}"
    assert n_dim_aware > n_dim_naive
    # The join rows are still present -- they are topology, just not billed for.
    assert sum(t == "bridge" for t, _, _ in aware.nodes) == 5, (
        "link rows connecting a kept dimension row must be retained")


def test_link_collapse_contracts_join_rows():
    """`collapse_links` removes contentless join rows but keeps what they joined.

    A `conditions_studies` row renders as `date: 158d before` and nothing else -- no content,
    and a date it already shares with the study it hangs off. Six of nineteen nodes in a real
    rel-trial document were exactly this.
    """
    n = 4
    entity = pd.DataFrame({"entity_id": [0], "label": ["seed"]})
    dimension = pd.DataFrame({"dim_id": range(n), "name": [f"name-{i}" for i in range(n)]})
    bridge = pd.DataFrame({
        "bridge_id": range(n), "entity_id": [0] * n, "dim_id": range(n),
        "ts": [pd.Timestamp("2020-01-01")] * n})

    db = Database({
        "entity": Table(df=entity, fkey_col_to_pkey_table={}, pkey_col="entity_id"),
        "dimension": Table(df=dimension, fkey_col_to_pkey_table={}, pkey_col="dim_id"),
        "bridge": Table(df=bridge,
                        fkey_col_to_pkey_table={"entity_id": "entity", "dim_id": "dimension"},
                        pkey_col="bridge_id", time_col="ts"),
    })
    data = build_hetero_data(db, verbose=False)
    seed_ts = int(pd.Timestamp("2020-06-01").timestamp())

    graph = TemporalSampler(data, "entity", max_nodes=16, link_tables={"bridge"},
                            collapse_links=True).sample(0, seed_ts)

    assert not any(t == "bridge" for t, _, _ in graph.nodes), "join rows must be contracted"
    assert sum(t == "dimension" for t, _, _ in graph.nodes) == n, "content must survive"
    # Each dimension row is now joined straight to the seed, oriented back toward it.
    assert sorted(graph.edges) == [(i, 0) for i in range(1, n + 1)], graph.edges


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

def test_determinism_and_versions(toy):
    db, data = toy
    sampler = TemporalSampler(data, "entity", max_nodes=8, strategy="uniform")

    a = sampler.sample(0, _ts(9), version=0, identity=("toy", "t", "train"))
    b = sampler.sample(0, _ts(9), version=0, identity=("toy", "t", "train"))
    assert a.nodes == b.nodes and a.edges == b.edges, "same identity must reproduce exactly"

    # Interleave another draw: the per-call seeding must make the result independent of
    # whatever else touched the global RNG in between.
    sampler.sample(1, _ts(4), version=3)
    c = sampler.sample(0, _ts(9), version=0, identity=("toy", "t", "train"))
    assert a.nodes == c.nodes, "sampling is order-dependent; the RNG is not being reseeded"


def test_recency_order_under_last_strategy(toy):
    """`temporal_strategy='last'` must return the k most recent, not an arbitrary k."""
    db, data = toy
    events = db.table_dict["event"].df
    graph = TemporalSampler(data, "entity", max_nodes=4).sample(0, _ts(9))
    days = sorted((events.iloc[row]["day"] for t, row, _ in graph.nodes if t == "event"),
                  reverse=True)
    assert days == sorted(range(10 - len(days), 10), reverse=True), (
        f"expected the most recent days, got {days}")


# ---------------------------------------------------------------------------
# 4. Structure
# ---------------------------------------------------------------------------

def test_seed_is_node_zero_and_graph_is_connected(toy):
    db, data = toy
    sampler = TemporalSampler(data, "entity", max_nodes=16)

    for seed_day in (0, 3, 9):
        graph = sampler.sample(0, _ts(seed_day))
        assert graph.nodes[0] == ("entity", 0, 0), "the seed must be node 0"

        adj = {i: set() for i in range(len(graph))}
        for i, j in graph.edges:
            adj[i].add(j)
            adj[j].add(i)
        seen, stack = {0}, [0]
        while stack:
            for nxt in adj[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        assert len(seen) == len(graph), (
            f"{len(graph) - len(seen)} node(s) unreachable from the seed")


def test_edges_are_child_to_parent(toy):
    """Direction is load-bearing: the magnetic-Laplacian bias is the one channel that
    carries it, and child -> parent (fact -> dimension) is the meaningful orientation."""
    db, data = toy
    graph = TemporalSampler(data, "entity", max_nodes=16).sample(0, _ts(9))
    for i, j in graph.edges:
        child, parent = graph.nodes[i][0], graph.nodes[j][0]
        assert child == "event" and parent in ("entity", "kind"), (
            f"edge {child} -> {parent} is not oriented child -> parent")


def test_no_duplicate_edges(toy):
    """Both orientations of each fkey edge appear in the raw sampler output."""
    db, data = toy
    graph = TemporalSampler(data, "entity", max_nodes=16).sample(0, _ts(9))
    assert len(graph.edges) == len(set(graph.edges))
    assert all(i != j for i, j in graph.edges), "self-loop in the sampled graph"


def test_empty_neighborhood_is_valid(toy):
    """A seed with no eligible history yields the seed alone, not an error.

    Real: rel-f1 has drivers whose first prediction date precedes any of their races.
    """
    db, data = toy
    graph = TemporalSampler(data, "entity", max_nodes=16).sample(
        0, int(pd.Timestamp("2019-01-01").timestamp()))
    assert len(graph) == 1 and graph.nodes[0] == ("entity", 0, 0)
    assert graph.edges == []
