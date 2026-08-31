"""Synthetic logits -> known AUROC. The Tier-B readout has no other test that can catch an error.

Every number Tier B reports passes through `make_margin_preprocessor` and
`make_margin_metrics`. A sign flip, an off-by-one in the answer position, or a missing
sigmoid all produce *plausible* metrics -- 0.5-ish, or 0.9-ish, with nothing raising. That
is the same argument PLAN.md §8 makes for the round-trip test being the highest-value test
in the campaign: a silent scoring bug looks like a mediocre model for weeks.

`evaluate.py` is a *port* of `src/experiments/relbench/evaluate.py`, and the tests were left
behind in the port. This file is the missing half, adapted to the molecules signatures
(`make_margin_metrics(yes_id)` rather than relbench's task-object form).
"""

import numpy as np
import pytest
import torch

from src.experiments.molecules.evaluate import (
    answer_token_ids,
    make_margin_metrics,
    make_margin_preprocessor,
    tied_pair_fraction,
)

YES, NO, V = 7, 9, 32


def _batch(margins, truths, prefix=3):
    """One eval batch: `prefix` unsupervised tokens, then the answer token, then EOS.

    Laid out the way the collator actually lays it out, so the preprocessor's position
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
    return make_margin_metrics(YES)((preds.numpy(), labels.numpy()))


# -- correctness --------------------------------------------------------------

def test_perfect_ranking_scores_one():
    assert _run([3.0, 2.0, -2.0, -3.0], [1, 1, 0, 0])["roc_auc"] == pytest.approx(1.0)


def test_inverted_ranking_scores_zero():
    """Catches a sign flip, which a 0.5-ish result would not."""
    assert _run([-3.0, -2.0, 2.0, 3.0], [1, 1, 0, 0])["roc_auc"] == pytest.approx(0.0)


def test_auroc_matches_sklearn_on_random_scores():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    margins = rng.normal(size=64) * 3
    truths = rng.integers(0, 2, size=64)
    if len(np.unique(truths)) < 2:
        pytest.skip("degenerate draw")
    got = _run(list(margins), list(truths))["roc_auc"]
    assert got == pytest.approx(roc_auc_score(truths, margins))


def test_the_target_is_read_from_the_answer_token():
    """Truths are recovered from the gathered token id, not from position or order."""
    assert _run([1.0, 1.0, 1.0], [1, 0, 1])["pos_rate"] == pytest.approx(2 / 3)


def test_answer_position_is_the_first_supervised_token_not_the_eos():
    """EOS is supervised too. Scoring at EOS would read a logit that says nothing about
    the class, giving a stable ~0.5 that looks like an untrained model rather than a bug."""
    logits, labels = _batch([5.0, -5.0], [1, 0])
    # Make the EOS-scoring position say the opposite, so the two choices disagree.
    logits[0, 3, YES], logits[0, 3, NO] = -9.0, 9.0
    logits[1, 3, YES], logits[1, 3, NO] = 9.0, -9.0
    preds = make_margin_preprocessor(YES, NO)(logits, labels)
    assert preds[0, 0] - preds[0, 1] > 0, "example 0 should be scored at the answer token"
    assert preds[1, 0] - preds[1, 1] < 0


def test_unsupervised_batch_raises_rather_than_scoring_noise():
    """Label masking breaking silently would score position 0 of every row."""
    logits, labels = _batch([1.0, 2.0], [1, 0])
    labels[:] = -100
    with pytest.raises(ValueError, match="label masking"):
        make_margin_preprocessor(YES, NO)(logits, labels)


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


# -- tie collapse: the HIV failure mode, pinned ------------------------------

def test_tied_scores_are_reported_as_such():
    """bf16 saturation collapsing every margin to one value is a known failure
    (`project-gtlm-margin-quantization`) that otherwise looks like a bad model."""
    assert _run([1.0] * 6, [1, 0, 1, 0, 1, 0])["n_distinct"] == 1.0
    assert _run([1.0, 2.0, 3.0], [1, 0, 1])["n_distinct"] == 3.0


def test_tied_pair_fraction_is_one_when_every_score_is_identical():
    """Every (pos, neg) pair tied => AUROC is entirely coin flips, and reads 0.5."""
    out = _run([1.0] * 6, [1, 0, 1, 0, 1, 0])
    assert out["tied_pair_fraction"] == pytest.approx(1.0)
    assert out["roc_auc"] == pytest.approx(0.5)


def test_tied_pair_fraction_is_zero_when_no_score_is_shared():
    assert _run([3.0, 2.0, -2.0, -3.0], [1, 1, 0, 0])["tied_pair_fraction"] == pytest.approx(0.0)


def test_tied_pair_fraction_counts_pairs_not_values():
    """Positives {1.0, 1.0} against negatives {1.0, 5.0}: 4 pairs, 2 of them tied.

    The denominator is pairs, not values -- a duplicated positive ties twice, and
    reporting 1/3 (distinct values) instead of 1/2 would understate the coin-flip share.
    """
    assert tied_pair_fraction([1.0, 1.0, 1.0, 5.0], [1, 1, 0, 0]) == pytest.approx(0.5)


def test_tied_pair_fraction_is_nan_on_a_single_class_split():
    assert np.isnan(tied_pair_fraction([1.0, 2.0], [1, 1]))


def test_the_imbalanced_case_is_measurable():
    """HIV is ~3.5% positive. A handful of distinct margins decides the whole AUROC,
    which is why `n_distinct` and `tied_pair_fraction` are in the record from run one."""
    margins = [1.0] * 100
    truths = [1] * 4 + [0] * 96
    out = _run(margins, truths)
    assert out["pos_rate"] == pytest.approx(0.04)
    assert out["n_distinct"] == 1.0
    assert out["tied_pair_fraction"] == pytest.approx(1.0)


# -- degenerate splits --------------------------------------------------------

def test_single_class_split_does_not_raise():
    """A smoke run on a handful of rows legitimately hits this; sklearn would raise."""
    assert np.isnan(_run([1.0, 2.0], [1, 1])["roc_auc"])


def test_output_is_three_floats_per_example_not_the_vocabulary():
    """The whole point of doing this inside `preprocess_logits_for_metrics`."""
    logits, labels = _batch([1.0] * 5, [1, 0, 1, 0, 1])
    assert make_margin_preprocessor(YES, NO)(logits, labels).shape == (5, 3)


def test_tuple_logits_are_unwrapped():
    """Some model wrappers return `(logits, ...)`; scoring the tuple would raise obscurely."""
    logits, labels = _batch([3.0, -3.0], [1, 0])
    preds = make_margin_preprocessor(YES, NO)((logits, None), labels)
    assert preds.shape == (2, 3)


# -- label words --------------------------------------------------------------

def test_multi_token_label_words_are_rejected():
    class _Tok:
        name_or_path = "fake"

        def encode(self, w, add_special_tokens=False):
            return [1, 2] if w == " maybe" else [1]

    with pytest.raises(ValueError, match="1 tokens|2 tokens"):
        answer_token_ids(_Tok(), words=(" yes", " maybe"))


def test_label_words_sharing_a_token_are_rejected():
    class _Tok:
        name_or_path = "fake"

        def encode(self, w, add_special_tokens=False):
            return [5]

    with pytest.raises(ValueError, match="share token id"):
        answer_token_ids(_Tok(), words=(" yes", " no"))


def test_llama_label_words_are_single_tokens():
    """The readout's premise, checked against the real tokenizer."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception:                                            # noqa: BLE001
        pytest.skip("tokenizer unavailable offline")
    yes, no = answer_token_ids(tok)
    assert yes != no
