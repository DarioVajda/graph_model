"""The compose stage's draw rules (`MOLECULE_GENERALIST.md` §9.4).

The selection rules are the whole safety argument for putting demonstrations in
front of an example, so they are tested on the selector rather than inferred
from a composed set: a demonstration must not be the target's own molecule, must
not state the target's facts, and must not tell you the target's answer by its
polarity.
"""

from __future__ import annotations

import random

from src.generalist.assistant import Fact, fact_polarity
from src.generalist.tools.assistant_compose import _agreement, _take


def _row(id_, key, value, kind="yesno", family="ring_membership"):
    fact = Fact(family, f"{family} is {value}.", value, kind)
    return {"id": id_, "key": key, "question": f"q for {id_}",
            "answer": f"a for {id_}", "facts": [fact.to_json()],
            "brief": {"format": ""}}


class TestTake:
    def test_it_honours_the_polarity_it_is_asked_for(self):
        ranked = [_row("a", "CCO", "yes"), _row("b", "CCC", "no")]
        assert _take(ranked, [], {}, "no")["id"] == "b"
        assert _take(ranked, [], {}, "yes")["id"] == "a"

    def test_an_empty_polarity_takes_the_best_ranked_row(self):
        ranked = [_row("a", "CCO", "yes"), _row("b", "CCC", "no")]
        assert _take(ranked, [], {}, "")["id"] == "a"

    def test_it_returns_none_when_the_polarity_cannot_be_met(self):
        ranked = [_row("a", "CCO", "yes")]
        assert _take(ranked, [], {}, "no") is None

    def test_a_row_already_chosen_is_not_chosen_twice(self):
        ranked = [_row("a", "CCO", "yes"), _row("b", "CCC", "yes")]
        chosen = [{"id": "a", "key": "CCO"}]
        assert _take(ranked, chosen, {}, "yes")["id"] == "b"

    def test_one_molecule_appears_once_however_many_rows_it_has(self):
        """Two accepted rows about one molecule are two ids and one structure;
        taking both would put the same component in the context twice."""
        ranked = [_row("a", "CCO", "yes"), _row("b", "CCO", "yes")]
        chosen = [{"id": "a", "key": "CCO"}]
        assert _take(ranked, chosen, {}, "yes") is None

    def test_the_reuse_ceiling_is_respected(self):
        ranked = [_row("a", "CCO", "yes"), _row("b", "CCC", "yes")]
        assert _take(ranked, [], {"a": 99}, "yes")["id"] == "b"


class TestAgreement:
    def test_only_rows_with_a_polarity_on_both_sides_are_counted(self):
        splits = {"train": [
            # yes target, one agreeing and one disagreeing demonstration
            {"facts": _row("t", "C", "yes")["facts"],
             "shots": [{"polarity": "yes"}, {"polarity": "no"}]},
            # a count target has no polarity and contributes nothing
            {"facts": _row("u", "CC", "3", kind="count")["facts"],
             "shots": [{"polarity": "yes"}]},
            # a demonstration with no polarity is skipped too
            {"facts": _row("v", "CCC", "no")["facts"],
             "shots": [{"polarity": ""}]},
        ]}
        assert _agreement(splits, fact_polarity) == {"pairs": 2, "agreed": 1,
                                                     "rate": 0.5}

    def test_no_countable_pairs_reports_no_rate_rather_than_zero(self):
        splits = {"train": [{"facts": [], "shots": []}]}
        assert _agreement(splits, fact_polarity)["rate"] is None


def test_the_balance_holds_over_a_skewed_pool():
    """The rule exists because the pool is skewed. A pool that is 90 % "yes"
    must still produce demonstrations that agree with their target about half
    the time, or the demonstration set is telling the model the answer."""
    rng = random.Random(0)
    pool = ([_row(f"y{i}", f"C{'C' * i}", "yes") for i in range(90)]
            + [_row(f"n{i}", f"O{'C' * i}", "no") for i in range(10)])
    used, agree, total = {}, 0, 0
    for trial in range(400):
        target = _row("t", "c1ccccc1", "yes" if trial % 2 else "no")
        want = rng.choice(("yes", "no"))
        picked = _take(pool, [], used, want) or _take(pool, [], used, "")
        assert picked is not None
        total += 1
        agree += fact_polarity(picked["facts"]) == fact_polarity(target["facts"])
    assert abs(agree / total - 0.5) < 0.1, (
        f"demonstrations agreed with their target {agree}/{total} of the time")
