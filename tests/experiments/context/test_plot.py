"""The figure must not average two evaluation conditions into one cell.

``grid.jsonl`` carries one record per (N, T, **k**). Aggregating on (N, T) alone
looks harmless — every record still contributes — but it silently averages a
condition the graph arm solves perfectly (k=1) with one it does not (k=4) and then
labels the mean as a single accuracy. Nothing raises; the figure is just wrong, and
wrong in the direction that flatters the result. These pin the keying.
"""

import json

import pytest

pytest.importorskip("matplotlib")

from src.experiments.context.analysis.plot import (  # noqa: E402
    load_grid, pick_metric, write_table,
)


def _rec(n, t, k, acc, **kw):
    r = {"n_nodes": n, "tokens_per_node": t, "hops": k, "n": 200,
         "em": acc, "code_acc": acc, "packed_len": (n - 2) * t,
         "distractor_rate": 0.0, "code_no_eos_rate": 0.0, "malformed_rate": 1 - acc,
         "in_train_distribution": True, "max_train_len": 16384}
    r.update(kw)
    return r


def _write(tmp_path, records):
    path = tmp_path / "grid.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


# ── the bug this file exists for ──────────────────────────────────────────────

def test_hop_counts_are_not_pooled_into_one_cell(tmp_path):
    """One cell, four k values, wildly different accuracy -> four groups, not one."""
    path = _write(tmp_path, [_rec(16, 64, k, acc)
                             for k, acc in [(1, 1.0), (2, 1.0), (3, 0.5), (4, 0.1)]])
    by_k = load_grid(path)
    assert sorted(by_k) == [1, 2, 3, 4]
    assert [by_k[k][(16, 64)]["em"] for k in (1, 2, 3, 4)] == [1.0, 1.0, 0.5, 0.1]
    # The pooled mean (0.65) must appear nowhere.
    assert all(c["em"] != pytest.approx(0.65)
               for cells in by_k.values() for c in cells.values())


def test_each_hop_group_keeps_the_full_cell_grid(tmp_path):
    cells = [(16, 64), (16, 128), (128, 64), (128, 128)]
    path = _write(tmp_path, [_rec(n, t, k, 0.9) for n, t in cells for k in (1, 2)])
    by_k = load_grid(path)
    assert set(by_k) == {1, 2}
    for k in (1, 2):
        assert set(by_k[k]) == set(cells)


def test_seeds_within_one_condition_still_pool(tmp_path):
    """Pooling over seeds is correct and must survive the k split."""
    path = _write(tmp_path, [_rec(16, 64, 1, 1.0, seed=0), _rec(16, 64, 1, 0.0, seed=1)])
    cell = load_grid(path)[1][(16, 64)]
    assert cell["seeds"] == 2
    assert cell["n"] == 400
    assert cell["em"] == pytest.approx(0.5)


# ── records that predate the k mixture ────────────────────────────────────────

def test_records_without_hops_form_a_single_group(tmp_path):
    """The star grid has no `hops` field; it must behave exactly as before."""
    records = [_rec(16, 64, 0, 1.0), _rec(16, 128, 0, 1.0)]
    for r in records:
        del r["hops"]
    by_k = load_grid(_write(tmp_path, records))
    assert list(by_k) == [0]
    assert set(by_k[0]) == {(16, 64), (16, 128)}


# ── which accuracy column ─────────────────────────────────────────────────────

def test_code_acc_wins_when_every_record_has_it():
    assert pick_metric([{"code_acc": 1.0, "em": 0.0}]) == "code_acc"


def test_falls_back_to_em_when_any_record_predates_code_acc():
    assert pick_metric([{"code_acc": 1.0, "em": 1.0}, {"em": 1.0}]) == "em"


def test_explicit_metric_overrides_the_auto_choice(tmp_path):
    path = _write(tmp_path, [_rec(16, 64, 1, 0.5, em=0.25)])
    assert load_grid(path, metric="em")[1][(16, 64)]["em"] == pytest.approx(0.25)
    assert load_grid(path, metric="code_acc")[1][(16, 64)]["em"] == pytest.approx(0.5)


# ── the table view ────────────────────────────────────────────────────────────

def test_table_emits_one_row_per_condition_with_k_first(tmp_path):
    path = _write(tmp_path, [_rec(n, 64, k, 0.9)
                             for n in (16, 32) for k in (1, 2)])
    text = write_table(load_grid(path), str(tmp_path / "t.md"))
    body = [ln for ln in text.splitlines() if ln.startswith("| ") and "--:" not in ln][1:]
    assert len(body) == 4
    assert [ln.split("|")[1].strip() for ln in body] == ["1", "1", "2", "2"]
    assert "| k |" in text.splitlines()[0]
