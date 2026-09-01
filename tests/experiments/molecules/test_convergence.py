"""`_convergence` reads a training trajectory. This pins what it must not be read from.

The instrument was wrong for M3a's whole sweep and nothing caught it: `_eval_curve`
was called at the end of the run, by which point `load_best_model_at_end` had already
restored the best checkpoint and `trainer.evaluate(..., metric_key_prefix="eval")` had
logged that model's val score into the same history. The curve therefore ended on a
point equal to its own maximum, `tail_gain` became `max - values[-4]` (never negative),
and `still_improving` fired hardest on the runs that had overfit furthest — reporting
"needs more budget" for runs that were already past their peak.

14 of M3a's 16 runs were flagged `still_improving`; recomputed from the same stored
curves with the final point dropped, 2 were.
"""

import pytest

from src.experiments.molecules.train import _convergence

METRIC = "eval_roc_auc"


def curve(values):
    return [{"step": 50 * (i + 1), METRIC: v} for i, v in enumerate(values)]


# -- the reading itself --------------------------------------------------------

def test_a_rising_tail_is_still_improving():
    out = _convergence(curve([0.60, 0.65, 0.70, 0.75, 0.80]), METRIC)
    assert out["still_improving"] is True
    assert out["tail_gain"] == pytest.approx(0.15)


def test_a_flat_tail_is_converged():
    # tail_gain compares against values[-4], so 0.795 -> 0.801 is +0.006, under 1pp.
    out = _convergence(curve([0.60, 0.795, 0.80, 0.80, 0.801]), METRIC)
    assert out["still_improving"] is False


def test_a_falling_tail_is_not_still_improving():
    """Overfitting is the opposite of budget-limited and must not read as it."""
    out = _convergence(curve([0.60, 0.75, 0.80, 0.74, 0.70]), METRIC)
    assert out["still_improving"] is False
    assert out["tail_gain"] < 0


def test_a_gain_inside_eval_noise_does_not_count():
    """1pp over three evals is the threshold; just under it must not fire."""
    out = _convergence(curve([0.700, 0.700, 0.700, 0.700, 0.709]), METRIC)
    assert out["still_improving"] is False


def test_too_short_a_curve_reports_nothing_rather_than_guessing():
    out = _convergence(curve([0.6, 0.7, 0.8]), METRIC)
    assert out == {"tail_gain": None, "still_improving": None,
                   "best_eval_index": None, "peak_fraction": None}


# -- where the peak sits -------------------------------------------------------

def test_peak_fraction_locates_an_early_peak():
    """BBBP's real shape: peaks within the first fifth, then drifts down."""
    out = _convergence(curve([0.60, 0.96, 0.95, 0.94, 0.93, 0.93]), METRIC)
    assert out["best_eval_index"] == 1
    assert out["peak_fraction"] == pytest.approx(0.2)


def test_peak_fraction_is_one_when_the_run_was_interrupted():
    out = _convergence(curve([0.60, 0.65, 0.70, 0.75, 0.80]), METRIC)
    assert out["peak_fraction"] == 1.0


def test_the_first_of_several_equal_maxima_wins():
    """Otherwise a plateau reads as "peaked at the end" and hides spare budget."""
    out = _convergence(curve([0.60, 0.80, 0.80, 0.80, 0.80]), METRIC)
    assert out["best_eval_index"] == 1


# -- THE regression --------------------------------------------------------------

def test_the_flag_inverts_on_a_contaminated_curve():
    """The exact defect, reproduced.

    `values` is a run that peaked at 0.80 in the middle and decayed to 0.70 -- clearly
    converged and then overfit. Appending the post-reload re-evaluation of the best
    checkpoint (0.80, by construction the maximum) flips `still_improving` from False
    to True. If this ever stops flipping, the fix in `train_and_eval` -- snapshot the
    curve BEFORE `trainer.evaluate` -- has stopped being necessary and this test
    should be re-derived rather than deleted.
    """
    trajectory = [0.60, 0.72, 0.80, 0.76, 0.73, 0.70]
    clean = _convergence(curve(trajectory), METRIC)
    contaminated = _convergence(curve(trajectory + [max(trajectory)]), METRIC)

    assert clean["still_improving"] is False
    assert contaminated["still_improving"] is True
    assert contaminated["tail_gain"] > 0 > clean["tail_gain"]


def test_a_contaminated_curve_also_misplaces_the_peak_fraction():
    """The second casualty: the run looks like it peaked later than it did."""
    trajectory = [0.60, 0.90, 0.85, 0.80, 0.75]
    clean = _convergence(curve(trajectory), METRIC)
    contaminated = _convergence(curve(trajectory + [max(trajectory)]), METRIC)
    assert clean["peak_fraction"] == pytest.approx(0.25)
    # The duplicated maximum does not move `best_eval_index` (first max wins), but it
    # lengthens the curve, so the same peak reads as having come earlier.
    assert contaminated["peak_fraction"] == pytest.approx(0.2)
