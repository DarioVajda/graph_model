"""Unit tests for the graph_construction knob (Levi vs. triplet-node graphs).

Covers: config validation, cache-key stability (levi = byte-identical keys),
CLI round-trip, and the triplet construction itself on a synthetic record —
node text identical to the flat serialization's lines, symmetric shared-entity
edges, PROMPT/QUESTION wiring, and targets byte-identical to the Levi arm's.
"""

import random

import pytest

from src.experiments.kgqa.__main__ import build_parser, config_from_args
from src.experiments.kgqa.config import RunConfig, GRAPH_CONSTRUCTIONS
from src.experiments.kgqa.flat_data import triple_lines
from src.experiments.kgqa.process_dataset import (
    build_base_triplet, build_question_graphs)


# ── synthetic record + names (no external data) ──────────────────────────────
# Four triples: T0/T1 share m.topic, T0/T2 share m.ans, T3 shares nothing.

RECORD = {
    "question": "where was q born",
    "entities": ["m.topic"],
    "answers": [{"kb_id": "m.ans", "text": "Answerville"}],
    "subgraph": {"tuples": [
        ["m.topic", "people.person.place_of_birth", "m.ans"],
        ["m.topic", "people.person.profession", "m.job"],
        ["m.ans", "location.location.containedby", "m.state"],
        ["m.lone", "people.person.gender", "m.gender"],
    ]},
    "paths": [],
}
NAMES = {"m.topic": "Q Person", "m.ans": "Answerville", "m.job": "Singer",
         "m.state": "Some State", "m.lone": "Loner", "m.gender": "Male"}

T = [("T", i) for i in range(4)]


def build_one(question_node="isolated", record=RECORD, **kwargs):
    cfg = RunConfig(graph_construction="triplet", question_node=question_node,
                    **kwargs).for_dataset("webqsp")
    graphs = build_question_graphs(record, NAMES, cfg, versions=1,
                                   rng=random.Random(0))
    assert len(graphs) == 1
    return graphs[0]


# ── config surface ────────────────────────────────────────────────────────────

def test_validate_accepts_all_constructions():
    for gc in GRAPH_CONSTRUCTIONS:
        RunConfig(graph_construction=gc).validate()


@pytest.mark.parametrize("kwargs", [
    {"graph_construction": "bogus"},
    {"graph_construction": None},
    {"graph_construction": "triplet", "mode": "flat_train"},
    {"graph_construction": "triplet", "mode": "flat_data_prep"},
    {"graph_construction": "triplet", "cvt_collapse": True},
    {"graph_construction": "triplet", "cvt_collapse": False},
])
def test_validate_rejects(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs).validate()


def test_cache_key_levi_is_byte_identical_and_triplet_gets_suffix():
    base = RunConfig().data_config_key("webqsp")
    assert "_gc" not in base                       # historical keys untouched
    assert RunConfig(graph_construction="triplet").data_config_key("webqsp") \
        == base + "_gctriplet"
    # composes with the question_node suffix in fixed order
    assert RunConfig(graph_construction="triplet",
                     question_node="isolated").data_config_key("webqsp") \
        == base + "_qnisolated_gctriplet"


def test_cli_round_trip():
    args = build_parser().parse_args(["--graph-construction", "triplet"])
    assert config_from_args(args).graph_construction == "triplet"
    assert config_from_args(build_parser().parse_args([])).graph_construction == "levi"


# ── triplet construction ─────────────────────────────────────────────────────

def test_node_text_matches_flat_serialization():
    cfg = RunConfig().for_dataset("webqsp")
    lines = triple_lines(RECORD, NAMES, cfg)
    g = build_base_triplet(RECORD, NAMES, cfg.rel_mode, cfg.max_nodes)
    assert [g.nodes[n]["text"] for n in T] == lines
    assert lines[0] == "Q Person | place of birth | Answerville"


def test_edges_are_symmetric_shared_entity_pairs():
    cfg = RunConfig().for_dataset("webqsp")
    g = build_base_triplet(RECORD, NAMES, cfg.rel_mode, cfg.max_nodes)
    expected = {(T[0], T[1]), (T[1], T[0]),        # share m.topic
                (T[0], T[2]), (T[2], T[0])}        # share m.ans
    assert set(g.edges()) == expected              # T3 shares nothing: isolated


def test_prompt_and_question_wiring():
    g = build_one("isolated")
    # PROMPT's topic edges go to the triples CONTAINING the topic entity
    assert set(g.successors("PROMPT")) == {T[0], T[1]}
    assert g.nodes["PROMPT"]["text"].startswith("Answer:")
    # QUESTION is a pure prefix node: question text, no edges either way
    assert g.nodes["QUESTION"]["text"] == RECORD["question"]
    assert not list(g.successors("QUESTION"))
    assert not list(g.predecessors("QUESTION"))


def test_question_node_topics_targets_topic_triples():
    g = build_one("topics")
    assert set(g.successors("QUESTION")) == {T[0], T[1]}


def test_targets_identical_to_levi_arm():
    def target(gc):
        cfg = RunConfig(graph_construction=gc,
                        question_node="isolated").for_dataset("webqsp")
        (g,) = build_question_graphs(RECORD, NAMES, cfg, versions=1,
                                     rng=random.Random(0))
        return g.nodes["PROMPT"]["text"], g.graph["gold_answers"]
    assert target("triplet") == target("levi")


def test_unanswerable_row_kept_with_empty_target():
    rec = dict(RECORD, answers=[{"kb_id": "m.gone", "text": "Not Retrieved"}])
    cfg = RunConfig(graph_construction="triplet",
                    question_node="isolated").for_dataset("webqsp")
    (g,) = build_question_graphs(rec, NAMES, cfg, versions=1,
                                 rng=random.Random(0), keep_unanswerable=True)
    assert g.graph["unanswerable"]
    assert g.nodes["PROMPT"]["text"] == "Answer:"


def test_levi_arm_unchanged():
    """The default construction still produces the historical Levi structure."""
    cfg = RunConfig(question_node="isolated").for_dataset("webqsp")
    (g,) = build_question_graphs(RECORD, NAMES, cfg, versions=1,
                                 rng=random.Random(0))
    assert g.has_edge("PROMPT", "m.topic")         # entity-id topic edges intact
    assert "m.ans" in g and ("T", 0) not in g
