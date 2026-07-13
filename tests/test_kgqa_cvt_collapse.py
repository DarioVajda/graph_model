"""
Pin the 2x2 CVT-collapse ablation knob (2026-07-09).

Contract: cvt_collapse=None resolves to the arm default (graph: collapse,
flat: raw triples) so every pre-existing cache keeps its key; explicit values
add the _nocvt / _cvt suffixes; build_base_levi(cvt_collapse=False) keeps
mediator nodes; collapsed_triple_lines composes relation texts through
collapsed mediators exactly like the graph contraction; target sets are
collapse-invariant (collapse never removes a nameable answer node).
"""

from src.experiments.kgqa.config import RunConfig
from src.experiments.kgqa.process_dataset import (
    UNNAMED, build_base_levi, present_answer_texts)
from src.experiments.kgqa.flat_data import (
    collapsed_triple_lines, triple_lines, flat_data_config_key)

NAMES = {"m.p": "Parent", "m.leaf": "Leaf", "m.top": "Topic"}

# Topic -> rel0 -> (unnamed single-parent mediator) -> rel1 -> Leaf
REC = {
    "question": "who",
    "entities": ["m.top"],
    "answers": [{"kb_id": "m.leaf", "text": "Leaf"}],
    "subgraph": {"tuples": [
        ["m.top", "a.b.arrest", "m.cvt1"],
        ["m.cvt1", "a.b.source", "m.leaf"],
    ]},
}


def test_resolution_and_cache_keys():
    d = RunConfig()
    assert d.resolved_cvt_collapse("graph") is True
    assert d.resolved_cvt_collapse("flat") is False
    assert "_nocvt" not in d.data_config_key("webqsp")
    assert "_cvt" not in flat_data_config_key(d, "webqsp")
    assert RunConfig(cvt_collapse=False).data_config_key("webqsp").endswith("_nocvt")
    assert flat_data_config_key(RunConfig(cvt_collapse=True), "webqsp").endswith("_cvt")


def test_uncollapsed_graph_keeps_mediator():
    g_cvt = build_base_levi(REC, NAMES, "last_1", 512, cvt_collapse=True)
    g_raw = build_base_levi(REC, NAMES, "last_1", 512, cvt_collapse=False)
    assert "m.cvt1" not in g_cvt
    assert "m.cvt1" in g_raw and g_raw.nodes["m.cvt1"]["text"] == UNNAMED
    # collapse rewired rel0 -> rel1
    assert g_cvt.has_edge(("R", 0), ("R", 1))
    assert not g_raw.has_edge(("R", 0), ("R", 1))


def test_collapsed_lines_compose_relations():
    cfg = RunConfig(cvt_collapse=True)
    lines = collapsed_triple_lines(REC, NAMES, cfg)
    assert "Topic | arrest source | Leaf" in lines          # composed through cvt
    raw = triple_lines(REC, NAMES, RunConfig())
    assert raw == [f"Topic | arrest | {UNNAMED}", f"{UNNAMED} | source | Leaf"]


def test_multi_parent_mediator_not_collapsed_either_way():
    rec = {**REC, "subgraph": {"tuples": [
        ["m.top", "a.b.arrest", "m.cvt1"],
        ["m.p", "a.b.other", "m.cvt1"],          # second parent
        ["m.cvt1", "a.b.source", "m.leaf"],
    ]}}
    g = build_base_levi(rec, NAMES, "last_1", 512, cvt_collapse=True)
    assert "m.cvt1" in g                                     # ambiguous -> kept
    lines = collapsed_triple_lines(rec, NAMES, RunConfig(cvt_collapse=True))
    assert f"{UNNAMED} | source | Leaf" in lines             # mediator stays an endpoint


def test_targets_are_collapse_invariant():
    for flag in (True, False):
        g = build_base_levi(REC, NAMES, "last_1", 512, cvt_collapse=flag)
        assert present_answer_texts(g, REC) == ["Leaf"]
