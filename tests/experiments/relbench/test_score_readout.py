"""Synthetic logits -> known AUROC. The readout has no other test that can catch an error.

Every number this experiment reports passes through these two functions. A sign flip, an
off-by-one in the answer position, or a missing sigmoid all produce *plausible* metrics --
0.5-ish, or 0.9-ish, with nothing raising. So the readout is pinned against cases whose
correct answer is known by construction.
"""

import numpy as np
import pytest
import torch

from src.experiments.relbench.evaluate import (
    answer_token_ids, make_compute_metrics, make_margin_preprocessor,
)

YES, NO, V = 7, 9, 32


class _Task:
    """Stands in for a relbench EntityTask: the real metric functions, real task_type."""

    def __init__(self):
        from relbench.base import TaskType
        from relbench.metrics import accuracy, average_precision, f1, roc_auc
        self.task_type = TaskType.BINARY_CLASSIFICATION
        self.metrics = [average_precision, accuracy, f1, roc_auc]


def _batch(margins, truths, prefix=3):
    """One eval batch: `prefix` unsupervised tokens, then the answer token, then EOS.

    Built the way the collator actually lays it out, so the preprocessor's position
    arithmetic is exercised rather than assumed.
    """
    b, length = len(margins), prefix + 2
    logits = torch.zeros(b, length, V)
    labels = torch.full((b, length), -100)
    for i, (m, t) in enumerate(zip(margins, truths)):
        answer_at = prefix                       # index of the answer token
        logits[i, answer_at - 1, YES] = m        # logits at t-1 predict token t
        logits[i, answer_at - 1, NO] = 0.0
        labels[i, answer_at] = YES if t else NO
        labels[i, answer_at + 1] = 1             # EOS, also supervised
    return logits, labels


def _run(margins, truths):
    logits, labels = _batch(margins, truths)
    preds = make_margin_preprocessor(YES, NO)(logits, labels)
    return make_compute_metrics(_Task(), YES)((preds.numpy(), labels.numpy()))


# -- correctness --------------------------------------------------------------

def test_perfect_ranking_scores_one():
    assert _run([3.0, 2.0, -2.0, -3.0], [1, 1, 0, 0])["roc_auc"] == pytest.approx(1.0)


def test_inverted_ranking_scores_zero():
    """Catches a sign flip, which a 0.5-ish result would not."""
    assert _run([-3.0, -2.0, 2.0, 3.0], [1, 1, 0, 0])["roc_auc"] == pytest.approx(0.0)


def test_auroc_matches_sklearn_on_random_scores():
    rng = np.random.default_rng(0)
    margins = rng.normal(size=64) * 3
    truths = rng.integers(0, 2, size=64)
    if len(np.unique(truths)) < 2:
        pytest.skip("degenerate draw")
    from sklearn.metrics import roc_auc_score
    got = _run(list(margins), list(truths))["roc_auc"]
    assert got == pytest.approx(roc_auc_score(truths, margins))


def test_the_target_is_read_from_the_answer_token():
    """Truths are recovered from the gathered token id, not from position or order."""
    assert _run([1.0, 1.0, 1.0], [1, 0, 1])["pos_rate"] == pytest.approx(2 / 3)


def test_answer_position_is_the_first_supervised_token_not_the_eos():
    """EOS is supervised too. Scoring at EOS would read a logit that says nothing about the
    class, giving a stable ~0.5 that looks like an untrained model rather than a bug."""
    logits, labels = _batch([5.0, -5.0], [1, 0])
    # Make the EOS-scoring position say the opposite, so the two choices disagree.
    logits[0, 3, YES], logits[0, 3, NO] = -9.0, 9.0
    logits[1, 3, YES], logits[1, 3, NO] = 9.0, -9.0
    preds = make_margin_preprocessor(YES, NO)(logits, labels)
    assert preds[0, 0] - preds[0, 1] > 0, "example 0 should be scored at the answer token"
    assert preds[1, 0] - preds[1, 1] < 0


# -- the sigmoid, and what depends on it --------------------------------------

def test_threshold_metrics_use_the_probability_not_the_raw_margin():
    """All-negative predictions at a negative margin must not score f1 as all-positive.
    Without sigmoid, `f1` thresholds a raw margin at 0.5 and reports ~1.0 here."""
    out = _run([-3.0, -4.0, -2.0, -5.0], [0, 0, 0, 1])
    assert out["f1"] == pytest.approx(0.0)
    assert out["accuracy"] == pytest.approx(0.75)


def test_sigmoid_does_not_move_the_ranking_metrics():
    a = _run([3.0, 1.0, -1.0, -3.0], [1, 1, 0, 0])
    b = _run([300.0, 100.0, -100.0, -300.0], [1, 1, 0, 0])
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])
    assert a["average_precision"] == pytest.approx(b["average_precision"])


# -- the failure modes worth logging ------------------------------------------

def test_tied_scores_are_reported_as_such():
    """bf16 saturation collapsing every margin to one value is a known failure that looks
    like a bad model (PLAN.md 11)."""
    assert _run([1.0] * 6, [1, 0, 1, 0, 1, 0])["n_distinct"] == 1.0
    assert _run([1.0, 2.0, 3.0], [1, 0, 1])["n_distinct"] == 3.0


def test_single_class_split_does_not_raise():
    """A smoke run on a handful of rows legitimately hits this; sklearn would raise."""
    assert np.isnan(_run([1.0, 2.0], [1, 1])["roc_auc"])


def test_output_is_three_floats_per_example_not_the_vocabulary():
    """The whole point of doing this inside `preprocess_logits_for_metrics`."""
    logits, labels = _batch([1.0] * 5, [1, 0, 1, 0, 1])
    assert make_margin_preprocessor(YES, NO)(logits, labels).shape == (5, 3)


# -- label words --------------------------------------------------------------

def test_multi_token_label_words_are_rejected():
    class _Tok:
        name_or_path = "fake"

        def encode(self, w, add_special_tokens=False):
            return [1, 2] if w == " maybe" else [1]

    with pytest.raises(ValueError, match="1 tokens|2 tokens"):
        answer_token_ids(_Tok(), words=(" yes", " maybe"))


def test_llama_label_words_are_single_tokens():
    """The readout's premise, checked against the real tokenizer."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception:                                            # noqa: BLE001
        pytest.skip("tokenizer unavailable offline")
    yes, no = answer_token_ids(tok)
    assert yes != no
