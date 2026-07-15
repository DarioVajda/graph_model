"""Unit tests for the question_node knob (question-conditioned graph encoding).

Covers: config validation, cache-key stability (off = byte-identical keys),
CLI round-trip, and graph construction for all three edge modes on a synthetic
record (no data files / tokenizer needed).
"""

import random

import pytest

from src.experiments.kgqa.__main__ import build_parser, config_from_args
from src.experiments.kgqa.config import RunConfig, QUESTION_NODE_MODES
from src.experiments.kgqa.process_dataset import build_question_graphs


# ── synthetic record + names (no external data) ──────────────────────────────

RECORD = {
    "question": "where was q born",
    "entities": ["m.topic"],
    "answers": [{"kb_id": "m.ans", "text": "Answerville"}],
    "subgraph": {"tuples": [
        ["m.topic", "people.person.place_of_birth", "m.ans"],
        ["m.topic", "people.person.profession", "m.job"],
    ]},
    "paths": [],
}
NAMES = {"m.topic": "Q Person", "m.ans": "Answerville", "m.job": "Singer"}

EDGE_MODES = tuple(m for m in QUESTION_NODE_MODES if m != "off")


def build_one(question_node):
    cfg = RunConfig(question_node=question_node).for_dataset("webqsp")
    graphs = build_question_graphs(RECORD, NAMES, cfg, versions=1,
                                   rng=random.Random(0))
    assert len(graphs) == 1
    return graphs[0]


# ── config surface ────────────────────────────────────────────────────────────

def test_validate_accepts_all_modes():
    for qn in QUESTION_NODE_MODES:
        RunConfig(question_node=qn).validate()


@pytest.mark.parametrize("kwargs", [
    {"question_node": "bogus"},
    {"question_node": None},                   # "off" is the only disabled spelling
    {"question_node": "none"},                 # old name, renamed to "isolated"
    {"question_node": "all", "prompt_style": "chat"},
    {"question_node": "all", "mode": "flat_train"},
    {"question_node": "all", "mode": "flat_data_prep"},
])
def test_validate_rejects(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs).validate()


def test_cache_key_off_is_byte_identical_and_on_gets_suffix():
    base = RunConfig().data_config_key("webqsp")
    assert "_qn" not in base                       # historical keys untouched
    assert RunConfig(question_node="topics").data_config_key("webqsp") \
        == base + "_qntopics"


def test_cli_round_trip():
    args = build_parser().parse_args(["--question-node", "all"])
    assert config_from_args(args).question_node == "all"
    assert config_from_args(build_parser().parse_args([])).question_node == "off"


# ── graph construction ────────────────────────────────────────────────────────

def test_off_matches_historical_format():
    g = build_one("off")
    assert "QUESTION" not in g
    assert g.nodes["PROMPT"]["text"].startswith(RECORD["question"])


@pytest.mark.parametrize("mode", EDGE_MODES)
def test_question_node_structure(mode):
    g = build_one(mode)
    assert g.nodes["QUESTION"]["text"] == RECORD["question"]
    # prompt node is target-only; the question moved out of it
    assert g.nodes["PROMPT"]["text"].startswith("Answer:")
    assert RECORD["question"] not in g.nodes["PROMPT"]["text"]
    # PROMPT keeps its historical topic edges in every mode
    assert g.has_edge("PROMPT", "m.topic")
    # no in-edges to QUESTION: the swept knob is exactly its OUT-edge set
    assert not list(g.predecessors("QUESTION"))
    out = set(g.successors("QUESTION"))
    if mode == "all":
        assert out == set(g.nodes()) - {"PROMPT", "QUESTION"}
    elif mode == "topics":
        assert out == {"m.topic"}
    else:                                          # "isolated": no edges
        assert mode == "isolated" and not out


def test_unanswerable_row_keeps_question_node():
    rec = dict(RECORD, answers=[{"kb_id": "m.gone", "text": "Not Retrieved"}])
    cfg = RunConfig(question_node="topics").for_dataset("webqsp")
    (g,) = build_question_graphs(rec, NAMES, cfg, versions=1,
                                 rng=random.Random(0), keep_unanswerable=True)
    assert g.graph["unanswerable"]
    assert g.nodes["PROMPT"]["text"] == "Answer:"  # ends exactly at the anchor
    assert g.nodes["QUESTION"]["text"] == rec["question"]
