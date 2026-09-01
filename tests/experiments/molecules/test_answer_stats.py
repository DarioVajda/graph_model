"""`_answer_stats` — the floor a score has to beat, and the case where there isn't one.

Admission criterion 3 (PLAN.md §3.2.4) is "well above the majority-class rate", which is
unreadable from a run record unless the record carries the rate. Two ways that has gone
wrong already: §8.1's Tier-B runs would have recorded `base_rate: null` silently, and
§3.2.10.1's pool widening pushed `stereo_assigned` to a test split with ONE answer in it,
where the majority rate is 1.000 and a constant predictor is perfect.
"""

from src.experiments.molecules.train import _answer_stats


def test_the_floor_comes_from_the_test_split_when_there_is_one():
    """Train and test disagree sharply under a scaffold split, and the headline is a
    test number, so the floor must be the test split's."""
    out = _answer_stats({
        "answers": {" Yes": 900, " No": 100},
        "answers_by_split": {"train": {" Yes": 800, " No": 20},
                             "test": {" Yes": 52, " No": 48}},
    })
    assert out["base_rate"] == 0.52
    assert out["base_rate_source"] == "test_split"


def test_it_falls_back_to_the_corpus_rate_for_older_artifacts():
    out = _answer_stats({"answers": {" Yes": 76, " No": 24}})
    assert out["base_rate"] == 0.76
    assert out["base_rate_source"] == "all_examples"


def test_empty_stats_report_nothing_rather_than_crashing():
    """A missing sidecar must not fail a run that already cost GPU-hours."""
    out = _answer_stats({})
    assert out["base_rate"] is None
    assert out["n_classes"] == 0
    assert out["degenerate_test_split"] is None


def test_none_is_accepted():
    assert _answer_stats(None)["base_rate"] is None


# -- the degenerate case -------------------------------------------------------

def test_a_single_answer_test_split_is_flagged_degenerate():
    """`014`'s real `stereo_assigned` split: 1000 test examples, all ' 0'.

    Accuracy is 1.000 for a model that has learned nothing, so the family cannot
    compare two arms and must be voided rather than reported.
    """
    out = _answer_stats({"answers_by_split": {"test": {" 0": 1000}}})
    assert out["degenerate_test_split"] is True
    assert out["base_rate"] == 1.0
    assert out["n_classes"] == 1


def test_two_answers_are_not_degenerate_however_skewed():
    """A 999/1 split is a terrible probe but it is still a comparison; only ONE
    class makes the arms formally incomparable. Do not conflate the two."""
    out = _answer_stats({"answers_by_split": {"test": {" 0": 999, " 1": 1}}})
    assert out["degenerate_test_split"] is False
    assert out["base_rate"] == 0.999


def test_the_flag_reads_the_test_split_not_the_corpus():
    """Train carrying several answers must not mask a single-answer test split --
    which is exactly the shape `stereo_assigned` has (train is 93% one class, test
    is 100%)."""
    out = _answer_stats({
        "answers": {" 0": 4000, " 1": 200},
        "answers_by_split": {"train": {" 0": 3718, " 1": 126, " 2": 53},
                             "test": {" 0": 1000}},
    })
    assert out["degenerate_test_split"] is True


def test_the_answer_distribution_is_ordered_by_frequency():
    out = _answer_stats({"answers_by_split": {"test": {" b": 10, " a": 90}}})
    assert list(out["answer_distribution"]) == [" a", " b"]
