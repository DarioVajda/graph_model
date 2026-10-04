"""
The graph domains (`src/generalist/adapters/_graph.py` and the six adapters on
it) and what wiring them in must not move.

What is checked here needs no raw data and no tokenizer:

* **The format.** `assemble` gives every domain the molecule layout — content
  nodes, an edge-free question node, a prompt node last that points at its
  targets — with one leading space on the answer and the stop suffix on the
  terminated kinds only. `split_prompt` takes a specialist's own prompt node back
  off and keeps where it pointed.
* **Generator determinism.** A `_Stream` yields the same draws however it is
  chunked and whatever happens to the global generators in between, including a
  module that reseeds them at import.
* **Keys.** `graph_key` ignores node ids and construction order.
* **Scoring.** `entity_scores` is GNN-RAG's protocol over the full gold list.
* **Wiring.** The prefixes dispatch, `adapter_options` is validated and hashed
  only for domains the mixture names, and a molecules-only config's hash and
  registry do not move.

The builds themselves are checked on real data by
`src/generalist/tools/checks/graph_domains_smoke.py`, which needs a GPU node.
"""

import random

import networkx as nx
import numpy as np
import pytest

from src.generalist.adapters import GRAPH_DOMAINS, PREFIXES, adapter_for, get_adapter
from src.generalist.adapters import _graph as G
from src.generalist.config import MIXTURES, ConfigError, RunConfig
from src.generalist.evaluate.scorers import (GENERATED_KINDS, METRIC_KEYS,
                                             entity_scores, scoring_cost)
from src.generalist.schema import ANSWER_KINDS


class _Fmt:
    """A prompt format with a stop suffix, like the chat formats."""

    answer_prefix = "Answer:"
    answer_suffix = "<|eot_id|>"

    @staticmethod
    def question(q):
        return f"<q>{q}</q>"


def _draw(answer=" Yes", targets=(1, 2)):
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    for n in graph.nodes:
        graph.nodes[n]["text"] = f"node {n}"
    return G.Draw(graph=graph, question="Are 1 and 2 linked?", answer=answer,
                  targets=targets, key="k")


# ── format ───────────────────────────────────────────────────────────────────

def test_assemble_layout_question_isolated_prompt_last():
    out = G.assemble(_draw(), "yesno", _Fmt)
    n = out.number_of_nodes()
    q, p = out.graph["question_node"], out.graph["prompt_node"]
    assert (q, p) == (n - 2, n - 1)
    assert out.degree(q) == 0
    assert sorted(out.successors(p)) == [1, 2]
    assert out.in_degree(p) == 0
    assert out.nodes[q]["text"] == "<q>Are 1 and 2 linked?</q>"
    # yesno is stored with its space and is never terminated.
    assert out.nodes[p]["text"] == "Answer: Yes"
    # The undirected content graph became a symmetric digraph.
    assert out.has_edge(0, 1) and out.has_edge(1, 0)


@pytest.mark.parametrize("kind", ["span", "entities", "text", "smiles"])
def test_assemble_terminated_kinds_get_space_and_suffix(kind):
    out = G.assemble(_draw(answer="8."), kind, _Fmt)
    assert out.nodes[out.graph["prompt_node"]]["text"] == "Answer: 8.<|eot_id|>"


def test_assemble_rejects_bad_targets_and_textless_nodes():
    with pytest.raises(G.GraphBuildError):
        G.assemble(_draw(targets=(7,)), "yesno", _Fmt)
    draw = _draw()
    del draw.graph.nodes[0]["text"]
    with pytest.raises(G.GraphBuildError):
        G.assemble(draw, "yesno", _Fmt)


def test_split_prompt_keeps_targets_and_drops_source_question():
    graph = nx.DiGraph()
    graph.add_nodes_from([(0, {"text": "a"}), (1, {"text": "b"}),
                          ("q", {"text": "question"}), ("p", {"text": "A: Yes"})])
    graph.add_edges_from([(0, 1), ("p", 1)])
    graph.graph = {"prompt_node": "p", "question_node": "q"}
    content, text, targets = G.split_prompt(graph)
    assert text == "A: Yes"
    assert targets == (1,)
    assert sorted(content.nodes) == [0, 1]
    assert content.graph == {}
    assert "p" in graph and "q" in graph          # the input is not mutated


# ── determinism ──────────────────────────────────────────────────────────────

def _random_draws():
    while True:
        yield (random.random(), float(np.random.rand()))


def test_stream_is_independent_of_chunking_and_global_state():
    whole = G._Stream(_random_draws, "seed").take(12)
    chunked = G._Stream(_random_draws, "seed")
    pieces = []
    for n in (5, 1, 6):
        random.seed(999)
        np.random.seed(999)
        pieces += chunked.take(n)
    assert pieces == whole
    assert G._Stream(_random_draws, "other").take(12) != whole


def test_stream_survives_a_factory_that_reseeds_globals():
    def reseeding_factory():
        random.seed(42)                     # what `our_tests` does at import
        np.random.seed(42)
        return _random_draws()

    assert (G._Stream(reseeding_factory, "s").take(4)
            == G._Stream(_random_draws, "s").take(4))


def test_stream_restores_the_callers_generators():
    random.seed(7)
    expected = random.random()
    random.seed(7)
    G._Stream(_random_draws, "s").take(3)
    assert random.random() == expected


# ── keys ─────────────────────────────────────────────────────────────────────

def test_graph_key_ignores_ids_and_order_but_not_content():
    a = nx.Graph()
    a.add_nodes_from([(0, {"text": "x"}), (1, {"text": "y"}), (2, {"text": "z"})])
    a.add_edges_from([(0, 1), (1, 2)])
    b = nx.relabel_nodes(a, {0: 2, 1: 0, 2: 1})
    assert G.graph_key(a) == G.graph_key(b)
    assert G.graph_key(a, extra="q1") != G.graph_key(a, extra="q2")
    c = a.copy()
    c.add_edge(0, 2)
    assert G.graph_key(a) != G.graph_key(c)


def test_expressiveness_sizes_are_log_uniform():
    from src.generalist.adapters.expressiveness import log_uniform_size

    random.seed(0)
    sizes = [log_uniform_size(10, 1000) for _ in range(20_000)]
    assert min(sizes) >= 10 and max(sizes) <= 1000
    assert max(sizes) > 900 and min(sizes) < 12
    # Equal mass per factor of ten: 10-100 and 100-1000 each hold about half.
    below = sum(s < 100 for s in sizes) / len(sizes)
    assert 0.47 < below < 0.53
    with pytest.raises(ValueError):
        log_uniform_size(3, 10)


def test_magnetic_m_cap():
    domain = get_adapter("expressiveness").DOMAIN_SPEC
    info = domain.info("hard")
    cfg = domain.config_class()
    assert info.magnetic_m_cap == 128
    for run_m, expected in ((0, 128), (64, 64), (500, 128)):
        cfg.magnetic_m = run_m
        assert domain.magnetic_m(cfg, info) == expected


# ── scoring ──────────────────────────────────────────────────────────────────

def test_new_kinds_are_declared_and_scored():
    assert {"span", "entities"} <= set(ANSWER_KINDS)
    assert {"span", "entities"} <= set(METRIC_KEYS)
    assert "entities" in GENERATED_KINDS and "span" not in GENERATED_KINDS

    class Spec:
        answer_kind = "entities"
        max_new_tokens = 128

    assert scoring_cost(Spec, 10) == 1280.0
    Spec.answer_kind = "span"
    assert scoring_cost(Spec, 10) == 10.0


def test_entity_scores_follow_gnn_rag():
    golds = [["Paris", "Lyon"], ["Berlin"], ["Rome"]]
    preds = ["Paris\nMarseille", "berlin", ""]
    out = entity_scores(preds, golds)
    assert out["n"] == 3
    # Row 1: 1 of 2 golds, 1 of 2 items -> f1 0.5; row 2 exact; row 3 empty -> 0.
    assert out["f1"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert out["hit1"] == pytest.approx(2 / 3)
    assert out["hit"] == pytest.approx(2 / 3)
    assert set(out) == set(METRIC_KEYS["entities"])


# ── wiring ───────────────────────────────────────────────────────────────────

def test_prefix_dispatch():
    for domain in GRAPH_DOMAINS:
        assert PREFIXES[f"{domain}/"] == domain
        assert adapter_for(f"{domain}/anything") == domain
    assert adapter_for("mol/bace") == "molecules"


def test_every_domain_declares_its_tasks():
    seen = set()
    for domain in GRAPH_DOMAINS:
        module = get_adapter(domain)
        spec = module.DOMAIN_SPEC
        assert spec.name == domain and spec.prefix == f"{domain}/"
        for info in spec.tasks:
            assert info.answer_kind in ANSWER_KINDS
            assert info.kind in ("corpus", "generator")
            name = spec.full_name(info.name)
            assert name not in seen
            seen.add(name)
            if info.kind == "generator":
                assert all((info.sizes or {}).get(s) for s in info.built_splits())
        # `GRAPH_GENERALIST.md` §2: expressiveness is one task and KGQA's two
        # datasets share seeds, so neither holds a task out.
        held = any(info.held_out for info in spec.tasks)
        assert held == (domain not in ("expressiveness", "kgqa")), domain


def test_graph_smoke_mixture_covers_every_domain():
    names = [e["name"] for e in MIXTURES["graph_smoke"]]
    assert sorted({adapter_for(n) for n in names}) == sorted(GRAPH_DOMAINS)


def test_adapter_options_validated_and_layered():
    config = RunConfig(mixture="graph_smoke",
                       adapter_options={"kgqa": {"strict_cross_dataset": True,
                                                 "limit": 8}})
    assert config.graph_domains() == tuple(sorted(GRAPH_DOMAINS))
    kgqa = config.domain_adapter_config("kgqa")
    assert kgqa.strict_cross_dataset is True and kgqa.limit == 8
    assert kgqa.model_name == config.model_name
    # Each domain keeps its own node length; the run's is the molecule one.
    assert config.domain_adapter_config("expressiveness").max_length == 64
    bad = RunConfig(mixture="graph_smoke", adapter_options={"tag": {"nope": 1}})
    with pytest.raises(ConfigError):
        bad.domain_adapter_config("tag")


def test_adapter_options_hashed_only_for_mixture_domains():
    base = RunConfig()
    assert base.graph_domains() == ()
    with_unused = RunConfig(adapter_options={"kgqa": {"limit": 8}})
    assert with_unused.config_hash() == base.config_hash()
    assert "adapter_options" not in base.hash_payload()

    smoke = RunConfig(mixture="graph_smoke")
    limited = RunConfig(mixture="graph_smoke", adapter_options={"kgqa": {"limit": 8}})
    empty = RunConfig(mixture="graph_smoke", adapter_options={"kgqa": {}})
    assert limited.config_hash() != smoke.config_hash()
    assert empty.config_hash() == smoke.config_hash()
