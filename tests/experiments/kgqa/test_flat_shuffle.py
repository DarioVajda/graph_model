"""
Pin the E1 flat-order-shuffle diagnostic knob (2026-07-16, TODO.md).

Contract: flat_shuffle_lines=False is a byte-identical no-op (existing flat
caches/behavior untouched); =True scrambles each question's triple-line order
exactly once (shared across that question's `versions` rows, same scoping as
the answer-order augmentation) via an INDEPENDENT `line_rng` stream — NOT the
shared `rng` the answer-order augmentation draws from. That independence is
load-bearing, not cosmetic: consuming the shuffle from `rng` shifts every
subsequent `rng.shuffle(order)` draw, changing `target` between the shuffled
and unshuffled arms and confounding the diagnostic (caught by a bad first
implementation — see test_target_order_is_unaffected_by_line_shuffle below).
The flag is flat-arm-only and part of the flat cache key.
"""

import random

import pytest

from src.experiments.kgqa.config import RunConfig
from src.experiments.kgqa.flat_data import build_flat_rows, flat_data_config_key, triple_lines

NAMES = {"m.top": "Topic", "m.a": "A", "m.b": "B", "m.c": "C", "m.d": "D"}

# A topic entity with four outgoing triples, so a shuffle has 4! = 24 possible
# orders — vanishingly unlikely to coincide with the unshuffled order by chance.
REC = {
    "question": "who",
    "entities": ["m.top"],
    "answers": [{"kb_id": "m.a", "text": "A"}],
    "subgraph": {"tuples": [
        ["m.top", "r.1", "m.a"],
        ["m.top", "r.2", "m.b"],
        ["m.top", "r.3", "m.c"],
        ["m.top", "r.4", "m.d"],
    ]},
}

# Multiple present answers, so the answer-order augmentation (rng.shuffle)
# actually has something to permute.
REC_MULTI_ANSWER = {
    "question": "who",
    "entities": ["m.top"],
    "answers": [{"kb_id": "m.a", "text": "A"}, {"kb_id": "m.b", "text": "B"},
                {"kb_id": "m.c", "text": "C"}, {"kb_id": "m.d", "text": "D"}],
    "subgraph": REC["subgraph"],
}


def test_cache_key_suffix():
    d = RunConfig()
    assert "_shuf" not in flat_data_config_key(d, "webqsp")
    assert flat_data_config_key(RunConfig(flat_shuffle_lines=True), "webqsp").endswith("_shuf")


def test_validate_rejects_shuffle_on_graph_arm():
    with pytest.raises(ValueError):
        RunConfig(mode="train", flat_shuffle_lines=True).validate()
    # Flat modes are fine.
    RunConfig(mode="flat_train", flat_shuffle_lines=True).validate()
    RunConfig(mode="flat_data_prep", flat_shuffle_lines=True).validate()


def test_shuffle_off_is_byte_identical_to_unshuffled():
    cfg = RunConfig(flat_shuffle_lines=False)
    rng = random.Random("test:off")
    rows = build_flat_rows(REC, NAMES, cfg, versions=1, rng=rng, keep_unanswerable=False)
    assert rows[0]["lines"] == triple_lines(REC, NAMES, cfg)


def test_shuffle_on_reorders_but_preserves_the_multiset():
    cfg_off = RunConfig(flat_shuffle_lines=False)
    cfg_on = RunConfig(flat_shuffle_lines=True)
    unshuffled = triple_lines(REC, NAMES, cfg_off)

    rows = build_flat_rows(REC, NAMES, cfg_on, versions=1,
                           rng=random.Random("test:on"), keep_unanswerable=False,
                           line_rng=random.Random("test:on:lines"))
    shuffled = rows[0]["lines"]

    assert sorted(shuffled) == sorted(unshuffled)   # same triples, no data loss
    assert shuffled != unshuffled                   # a real reordering happened


def test_target_order_is_unaffected_by_line_shuffle():
    """The one thing this diagnostic must NOT change: the training target.
    Same `rng` seed, on vs. off, must yield an identical `target` string —
    otherwise a measured F1 delta would be confounded by two different
    augmentation samples, not isolated to the input triple order."""
    off = build_flat_rows(REC_MULTI_ANSWER, NAMES, RunConfig(flat_shuffle_lines=False),
                          versions=4, rng=random.Random("shared-seed"),
                          keep_unanswerable=False)
    on = build_flat_rows(REC_MULTI_ANSWER, NAMES, RunConfig(flat_shuffle_lines=True),
                         versions=4, rng=random.Random("shared-seed"),
                         keep_unanswerable=False, line_rng=random.Random("independent"))
    assert [r["target"] for r in off] == [r["target"] for r in on]
    # and the premise of the test isn't vacuous: the augmentation did vary something
    assert len({r["target"] for r in off}) > 1


def test_shuffle_is_shared_across_versions_of_one_question():
    """Mirrors the answer-order augmentation's scoping: the subgraph order is a
    property of the QUESTION, not of the version — only the target order
    varies per version."""
    cfg = RunConfig(flat_shuffle_lines=True)
    rows = build_flat_rows(REC, NAMES, cfg, versions=3,
                           rng=random.Random("test:versions"), keep_unanswerable=False,
                           line_rng=random.Random("test:versions:lines"))
    assert len(rows) == 3
    assert rows[0]["lines"] == rows[1]["lines"] == rows[2]["lines"]


def test_shuffle_is_deterministic_given_the_same_line_rng_seed():
    cfg = RunConfig(flat_shuffle_lines=True)
    rows_a = build_flat_rows(REC, NAMES, cfg, versions=1,
                             rng=random.Random("fixed-seed"), keep_unanswerable=False,
                             line_rng=random.Random("fixed-line-seed"))
    rows_b = build_flat_rows(REC, NAMES, cfg, versions=1,
                             rng=random.Random("fixed-seed"), keep_unanswerable=False,
                             line_rng=random.Random("fixed-line-seed"))
    assert rows_a[0]["lines"] == rows_b[0]["lines"]


def test_unanswerable_branch_also_shuffles():
    """The unanswerable (empty-target, dev/test-only) branch calls _lines() too
    — it must not bypass the shuffle."""
    rec = {**REC, "answers": [{"kb_id": "m.zzz", "text": "NotInGraph"}]}
    cfg = RunConfig(flat_shuffle_lines=True)
    rows = build_flat_rows(rec, NAMES, cfg, versions=1,
                           rng=random.Random("test:unanswerable"), keep_unanswerable=True,
                           line_rng=random.Random("test:unanswerable:lines"))
    assert len(rows) == 1 and rows[0]["unanswerable"] is True
    assert sorted(rows[0]["lines"]) == sorted(triple_lines(REC, NAMES, RunConfig()))
