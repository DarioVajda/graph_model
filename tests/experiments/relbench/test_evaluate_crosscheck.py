"""Prove the `task.evaluate` cross-check can actually fail.

`evaluate_split` recomputes the metric two ways -- once from the answer token we gathered,
once through relbench's own `task.evaluate` against the task table -- and raises if they
disagree. That guard is the last thing standing between a misaligned cache and a plausible
headline number, and a guard that never fires is decoration. So: feed it misaligned data and
require it to raise.
"""

import copy

import numpy as np
import pytest

from src.experiments.relbench.evaluate import evaluate_split

YES, NO = 10035, 912


class _Table:
    def __init__(self, df):
        self.df = df


class _Task:
    """Scores positionally and checks only length -- relbench's actual behaviour."""

    target_col = "y"

    def __init__(self, targets):
        from relbench.base import TaskType
        from relbench.metrics import roc_auc
        self.task_type = TaskType.BINARY_CLASSIFICATION
        self.metrics = [roc_auc]
        import pandas as pd
        self._df = pd.DataFrame({"y": targets})

    def get_table(self, split, mask_input_cols=False):
        return _Table(self._df)

    def evaluate(self, pred, target_table=None, metrics=None):
        target = (target_table or self.get_table("test")).df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(f"length mismatch: {len(pred)} vs {len(target)}")
        from relbench.metrics import roc_auc
        return {"roc_auc": float(roc_auc(target, pred))}


class _Trainer:
    """Returns fixed (logit_yes, logit_no, true_token) rows."""

    def __init__(self, preds, reported):
        self._preds = np.asarray(preds, dtype=np.float64)
        self._reported = reported

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval"):
        return {f"{metric_key_prefix}_{k}": v for k, v in self._reported.items()}

    def predict(self, dataset, metric_key_prefix="pred"):
        class _Out:
            predictions = self._preds
        return _Out()


def _preds(margins, truths):
    return [[m, 0.0, YES if t else NO] for m, t in zip(margins, truths)]


def test_agreeing_metrics_pass():
    margins, truths = [3.0, 1.0, -1.0, -3.0], [1, 1, 0, 0]
    task = _Task(truths)
    trainer = _Trainer(_preds(margins, truths), {"roc_auc": 1.0})
    official, _ = evaluate_split(trainer, None, task, "test", YES)
    assert official["test_roc_auc"] == pytest.approx(1.0)


def test_disagreement_raises():
    """The guard's whole purpose: our number and relbench's must not silently differ."""
    margins, truths = [3.0, 1.0, -1.0, -3.0], [1, 1, 0, 0]
    task = _Task(truths)
    trainer = _Trainer(_preds(margins, truths), {"roc_auc": 0.42})   # wrong on purpose
    with pytest.raises(ValueError, match="disagrees with task.evaluate"):
        evaluate_split(trainer, None, task, "test", YES)


def test_shuffled_cache_is_caught():
    """The real failure mode: graphs built in a different order than the task table. Our
    metric (target read from the gathered answer token) stays self-consistent and looks
    fine; relbench's positional comparison does not agree."""
    margins = [3.0, 1.0, -1.0, -3.0]
    built_order = [1, 1, 0, 0]              # what our cache says, in cache order
    table_order = [0, 0, 1, 1]              # what the task table says, in table order
    task = _Task(table_order)
    trainer = _Trainer(_preds(margins, built_order), {"roc_auc": 1.0})
    with pytest.raises(ValueError, match="misaligned"):
        evaluate_split(trainer, None, task, "test", YES)


def test_capped_split_scores_against_its_own_rows():
    full = [1, 0, 0, 0, 1, 1, 0, 1]
    task = _Task(full)
    kept = [0, 1, 4, 6]                      # targets 1, 0, 1, 0
    margins = [5.0, -5.0, 4.0, -4.0]         # perfectly ranked for the KEPT rows
    trainer = _Trainer(_preds(margins, [full[i] for i in kept]), {})
    official, _ = evaluate_split(trainer, None, task, "val", YES, row_ids=kept)
    assert official["val_roc_auc"] == pytest.approx(1.0), (
        "scoring a capped split against the first N table rows would not give 1.0")
