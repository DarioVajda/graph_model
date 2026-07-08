"""
Pin the KGQA data-prep answer semantics (data-format v2) and the GNN-RAG scoring port.

Covers: literal-kb_id fallback for answer texts, keep-unanswerable eval rows
(empty target, benchmark denominators), and the verbatim GNN-RAG match/F1/Hits
functions in evaluate.py.
"""

import random

import pytest

from src.experiments.kgqa.config import RunConfig
from src.experiments.kgqa.process_dataset import (
    answer_text, build_question_graphs, full_gold_texts, ANSWER_DELIM,
)


NAMES = {"m.topic": "Topic Entity"}


def _record(answers, tuples):
    return {
        "question": "when did it happen",
        "entities": ["m.topic"],
        "answers": answers,
        "subgraph": {"tuples": tuples},
        "paths": [],
    }


# --------------------------------------------------------------------------- #
# answer_text: literal fallback
# --------------------------------------------------------------------------- #
def test_answer_text_prefers_text_field():
    assert answer_text({"kb_id": "m.x", "text": "Jamaican English"}) == "Jamaican English"


def test_answer_text_literal_fallback():
    assert answer_text({"kb_id": "1945-09-02", "text": None}) == "1945-09-02"
    assert answer_text({"kb_id": "AUD", "text": ""}) == "AUD"
    assert answer_text({"kb_id": "3.048"}) == "3.048"


def test_answer_text_decodes_encoded_literals():
    # Freebase URL-ish escaping: $XXXX -> chr(0xXXXX)
    assert answer_text({"kb_id": "Justin$002BBieber", "text": None}) == "Justin+Bieber"


def test_answer_text_mids_without_text_are_unscoreable():
    assert answer_text({"kb_id": "m.0abc", "text": None}) is None
    assert answer_text({"kb_id": "g.121wt37c", "text": ""}) is None


def test_full_gold_texts_includes_literals_and_mid_placeholders():
    # unnamed mids stay as raw-kb_id placeholders, mirroring RoG's answer lists
    rec = _record(
        [{"kb_id": "m.a", "text": "Named"}, {"kb_id": "1921-09", "text": None},
         {"kb_id": "g.dead", "text": ""}],
        [["m.topic", "r.r", "m.a"]],
    )
    assert full_gold_texts(rec) == ["Named", "1921-09", "g.dead"]


# --------------------------------------------------------------------------- #
# build_question_graphs: answerable / unanswerable / unscorable
# --------------------------------------------------------------------------- #
def test_literal_answer_in_graph_is_supervisable():
    rec = _record([{"kb_id": "1945-09-02", "text": None}],
                  [["m.topic", "time.event.end_date", "1945-09-02"]])
    gs = build_question_graphs(rec, NAMES, RunConfig(), versions=2, rng=random.Random(0))
    assert len(gs) == 2
    g = gs[0]
    assert g.nodes["PROMPT"]["text"] == f"when did it happen{ANSWER_DELIM} 1945-09-02"
    assert g.graph["unanswerable"] is False
    assert g.graph["gold_answers"] == ["1945-09-02"]


def test_unanswerable_dropped_for_train_kept_for_eval():
    # gold has text but its entity is NOT in the graph (retrieval failure)
    rec = _record([{"kb_id": "m.gone", "text": "Missing Answer"}],
                  [["m.topic", "r.r", "m.other"]])
    assert build_question_graphs(rec, NAMES, RunConfig(), 1, random.Random(0)) == []
    gs = build_question_graphs(rec, NAMES, RunConfig(), 1, random.Random(0),
                               keep_unanswerable=True)
    assert len(gs) == 1
    g = gs[0]
    # empty target: prompt ends exactly at the delimiter (no trailing space)
    assert g.nodes["PROMPT"]["text"] == f"when did it happen{ANSWER_DELIM}"
    assert g.graph["unanswerable"] is True
    assert g.graph["gold_answers"] == ["Missing Answer"]


def test_unnamed_mid_gold_dropped_for_train_kept_for_eval():
    # only gold is an unnamed mid: no supervision target exists (train drops it),
    # but eval keeps the question with the raw-mid placeholder gold (RoG parity)
    rec = _record([{"kb_id": "g.dead", "text": ""}],
                  [["m.topic", "r.r", "g.dead"]])
    assert build_question_graphs(rec, NAMES, RunConfig(), 1, random.Random(0)) == []
    gs = build_question_graphs(rec, NAMES, RunConfig(), 1, random.Random(0),
                               keep_unanswerable=True)
    assert len(gs) == 1
    assert gs[0].graph["unanswerable"] is True
    assert gs[0].graph["gold_answers"] == ["g.dead"]


# --------------------------------------------------------------------------- #
# GNN-RAG scoring port (evaluate.py) — heavier imports, keep at the end
# --------------------------------------------------------------------------- #
ev = pytest.importorskip("src.experiments.kgqa.evaluate")


def test_match_is_normalized_substring():
    assert ev.match("the Jamaican English language", "Jamaican English")
    assert not ev.match("jamaican creole english language", "Jamaican English")
    assert ev.match("It ended on 1945-09-02.", "1945-09-02")


def test_eval_f1_matches_gnnrag_semantics():
    # matched counts golds found in the JOINED string; precision denom = #parts
    f1, prec, rec = ev.eval_f1(["Xyz", "Wrong"], ["Xyz"])
    assert (prec, rec) == (0.5, 1.0)
    assert abs(f1 - 2 * 0.5 * 1.0 / 1.5) < 1e-9
    assert ev.eval_f1([], ["Xyz"]) == (0, 0, 0)


def test_eval_hits():
    assert ev.eval_hit1(["Xyz", "Qrs"], ["Qrs"]) == 0     # first part only
    assert ev.eval_hit1(["Qrs", "Xyz"], ["Qrs"]) == 1
    assert ev.eval_hit(["Xyz", "Qrs"], ["Qrs"]) == 1      # any part


def test_parse_answer_list_keeps_raw_parts():
    assert ev.parse_answer_list(" Jamaican English, Jamaican Creole ,") == \
        ["Jamaican English", "Jamaican Creole"]


# --------------------------------------------------------------------------- #
# data-format v3: newline answer separator (comma-in-gold fix)
# --------------------------------------------------------------------------- #
def test_answer_sep_tracks_data_format_version():
    assert RunConfig(data_format_version=2).answer_sep == ", "
    assert RunConfig(data_format_version=2).answer_parse_sep == ","
    assert RunConfig(data_format_version=3).answer_sep == "\n"
    assert RunConfig(data_format_version=3).answer_parse_sep == "\n"


def test_data_format_version_in_cache_key_and_validated():
    assert RunConfig(data_format_version=2).data_config_key().endswith("_dfv2")
    assert RunConfig(data_format_version=3).data_config_key().endswith("_dfv3")
    with pytest.raises(ValueError):
        RunConfig(data_format_version=1).validate()


def test_v3_target_newline_joined_and_comma_gold_roundtrips():
    # the motivating case: a gold with an internal comma survives join+parse
    rec = _record(
        [{"kb_id": "m.a", "text": "Return J. Meigs, Jr."},
         {"kb_id": "m.b", "text": "Second Answer"}],
        [["m.topic", "r.r", "m.a"], ["m.topic", "r.r", "m.b"]],
    )
    names = {**NAMES, "m.a": "Return J. Meigs, Jr.", "m.b": "Second Answer"}
    cfg = RunConfig(data_format_version=3)
    gs = build_question_graphs(rec, names, cfg, versions=1, rng=random.Random(0))
    target = gs[0].nodes["PROMPT"]["text"].split(ANSWER_DELIM)[1]
    parsed = ev.parse_answer_list(target, sep=cfg.answer_parse_sep)
    assert sorted(parsed) == ["Return J. Meigs, Jr.", "Second Answer"]


def test_v3_parse_drops_blank_segments():
    # "\n\n" drift in generations must not create empty predictions
    assert ev.parse_answer_list(" A1\n\nA2\n", sep="\n") == ["A1", "A2"]
