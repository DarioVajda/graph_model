"""Pin the positional-alignment invariant that `task.evaluate` silently depends on.

`EntityTask.evaluate` does `target_table.df[target_col].to_numpy()` and compares it to the
prediction vector by position -- it checks only the *length*. So a val/test cache that is
reordered, strided, or partial produces a plausible-looking AUROC computed against the wrong
rows. Nothing in relbench or in HF raises. These tests are the only thing standing between a
strided smoke cache and a reported headline number.
"""

import networkx as nx
import pytest

from src.experiments.relbench.data import assert_row_order


class _FakeDataset:
    """The two attributes `assert_row_order` reads off a built `TextGraphDataset`."""

    def __init__(self, row_ids):
        self.graphs = []
        for r in row_ids:
            g = nx.DiGraph()
            if r is not None:
                g.graph["row_id"] = r
            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)


def test_complete_split_passes_both_checks():
    assert_row_order(_FakeDataset(range(5)), "test", contiguous=True)


def test_missing_row_id_is_rejected():
    """A cache built before the invariant existed. Predictions cannot be aligned at all."""
    with pytest.raises(ValueError, match="missing `row_id`"):
        assert_row_order(_FakeDataset([0, 1, None, 3]), "test")


def test_reordered_split_is_rejected():
    with pytest.raises(ValueError, match="not monotonic"):
        assert_row_order(_FakeDataset([0, 2, 1, 3]), "val")


def test_strided_split_is_monotonic_but_not_contiguous():
    """The failure that actually happened: a 201-of-11,411 train and 114-of-566 val build
    whose row_ids are strided. Monotonicity alone accepts it."""
    strided = _FakeDataset([0, 5, 10, 15])
    assert_row_order(strided, "val", contiguous=False)          # monotonic: passes
    with pytest.raises(ValueError, match="strided or partial"):
        assert_row_order(strided, "val", contiguous=True)


def test_truncated_split_is_rejected():
    """A prefix of the split is contiguous from 0 but short; only `task.evaluate`'s length
    check would catch it, and only if someone remembered to pass the full target table."""
    assert_row_order(_FakeDataset(range(3)), "test", contiguous=True)   # self-consistent
    with pytest.raises(ValueError, match="strided or partial"):
        assert_row_order(_FakeDataset([1, 2, 3]), "test", contiguous=True)
