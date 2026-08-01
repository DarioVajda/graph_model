"""Invariants for the k-hop pointer-chasing task (README §A.1).

The chain task exists because the lookup task was solvable by one literal string
match, so the properties that matter are the ones that keep it *unsolvable* that
way:

  * the QUESTION names the START node and never the answer — otherwise the
    traversal is bypassable and we are back to README §3.1;
  * every content node holds both a code and a pointer, so the answer node cannot
    be spotted as "the one that mentions a code";
  * walking the pointers ``hops`` times from the start really does land on the
    node whose code is the gold answer;
  * SPD from the QUESTION does NOT single out the answer. This is the subtle one:
    with a lone QUESTION->start edge the answer sits at the unique distance
    hops+1, and the graph arm could read it off the bias without traversing —
    measuring "we labelled the answer" rather than "structure helps";
  * the chain is identical across cells, so N and T remain paired axes.
"""

import re

import networkx as nx
import pytest

from src.experiments.context.config import RunConfig

transformers = pytest.importorskip("transformers")

from src.experiments.context.data import (  # noqa: E402
    build_code_pool, build_id_pool, chain_slots, load_corpus, make_blueprint,
    realize_chain,
)
from src.experiments.context.process_dataset import RAW_DATA_DIR  # noqa: E402

HOPS = 3
CFG = RunConfig(hops=HOPS, node_counts=(16, 64), token_counts=(64, 128))


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(CFG.model_name)


@pytest.fixture(scope="module")
def corpus(tokenizer):
    import os
    path = os.path.join(
        RAW_DATA_DIR,
        f"filler_wikitext-103-raw-v1_{CFG.model_name.split('/')[-1]}_{RunConfig().corpus_tokens}.npy")
    if not os.path.exists(path):
        pytest.skip("filler corpus not built; run --mode data_prep first")
    return load_corpus(tokenizer, RAW_DATA_DIR, RunConfig().corpus_tokens, verbose=False)


@pytest.fixture(scope="module")
def pools(tokenizer):
    return (build_code_pool(tokenizer, CFG.code_len, CFG.id_pool, seed=CFG.data_seed),
            build_id_pool(CFG.id_pool))




@pytest.fixture(scope="module")
def make(tokenizer, corpus, pools):
    codes, ids = pools

    def _make(n=64, t=128, graph_id=0, cfg=CFG):
        bp = make_blueprint(graph_id, cfg, codes, ids, len(corpus), split="test")
        return realize_chain(bp, cfg, n, t, tokenizer, corpus, split="test")
    return _make


def _content(g):
    qn, pn = g.graph["question_node"], g.graph["prompt_node"]
    return [k for k in g.nodes if k not in (qn, pn)]


def _definer(g, node_id):
    """The single node whose OWN kv sentence defines ``node_id``."""
    hits = [k for k in _content(g) if f"access code for {node_id} is" in g.nodes[k]["text"]]
    assert len(hits) == 1, f"{node_id} defined by {len(hits)} nodes"
    return hits[0]


def _pointer_of(g, node):
    m = re.search(r"Continue at (NODE-\d+)\.", g.nodes[node]["text"])
    assert m, f"node {node} has no pointer"
    return m.group(1)


# ── the traversal is real ─────────────────────────────────────────────────────

def test_walking_the_pointers_lands_on_the_gold_code(make):
    for graph_id in (0, 5, 17):
        g = make(graph_id=graph_id)
        cur = g.graph["start_id"]
        walked = [cur]
        for _ in range(HOPS):
            cur = _pointer_of(g, _definer(g, cur))
            walked.append(cur)
        assert walked == g.graph["chain_ids"]
        assert cur == g.graph["gold_id"]
        code = re.search(rf"access code for {cur} is (\w+)\.", g.nodes[_definer(g, cur)]["text"])
        assert code.group(1) == g.graph["gold_code"]


def test_question_names_the_start_and_never_the_answer(make):
    """If the QUESTION named the answer the chain would be bypassable."""
    for graph_id in (0, 3, 9):
        g = make(graph_id=graph_id)
        q = g.nodes[g.graph["question_node"]]["text"]
        assert g.graph["start_id"] in q
        assert g.graph["gold_id"] not in q
        assert str(HOPS) in q


def test_the_chain_never_revisits_a_node(make):
    for graph_id in range(6):
        ids = make(graph_id=graph_id).graph["chain_ids"]
        assert len(set(ids)) == len(ids) == HOPS + 1


# ── no shortcuts ──────────────────────────────────────────────────────────────

def test_every_content_node_has_both_a_code_and_a_pointer(make):
    """Decoys must be indistinguishable from chain nodes by surface form."""
    g = make()
    for k in _content(g):
        assert "access code for" in g.nodes[k]["text"]
        assert "Continue at" in g.nodes[k]["text"]


def test_the_answer_node_also_has_an_outgoing_pointer(make):
    """Otherwise it is the unique node with no successor — a dead giveaway."""
    g = make()
    assert "Continue at" in g.nodes[_definer(g, g.graph["gold_id"])]["text"]
    # Exactly fan_out, not ">= 1": a uniform out-degree is what stops degree
    # itself from marking the chain, and ">= 1" would pass a build that gave the
    # chain nodes extra edges.
    assert all(g.out_degree(k) == CFG.fan_out for k in _content(g))


# ── decoy references (fan_out > 1) ────────────────────────────────────────────
# At fan_out=1 the content subgraph is functional, so the answer is the UNIQUE
# node at distance hops from the start and SPD hands the graph arm the answer
# without any traversal (measured on the built data: 100% of graphs). These pin
# the property that makes fan_out>1 worth building.

FAN = 2
CFG_FAN = RunConfig(hops=HOPS, node_counts=(16, 64), token_counts=(64, 128), fan_out=FAN)


@pytest.fixture(scope="module")
def make_fan(tokenizer, corpus, pools):
    codes, ids = pools

    def _make(n=64, t=128, graph_id=0):
        bp = make_blueprint(graph_id, CFG_FAN, codes, ids, len(corpus), split="test")
        return realize_chain(bp, CFG_FAN, n, t, tokenizer, corpus, split="test")
    return _make


def test_fan_out_gives_every_content_node_the_same_out_degree(make_fan):
    g = make_fan()
    assert {g.out_degree(k) for k in _content(g)} == {FAN}


def test_fan_out_stops_spd_from_identifying_the_answer(make_fan):
    """THE point of the decoy build.

    With fan_out=1 the answer is alone at distance `hops` from the start, so the
    graph arm can read it off the distance bias. With decoys, ~fan_out**hops nodes
    share that distance and topology can only prune the candidate set.
    """
    lonely = 0
    for graph_id in range(8):
        g = make_fan(graph_id=graph_id)
        start = _definer(g, g.graph["start_id"])
        sub = g.subgraph(_content(g))
        dist = nx.single_source_shortest_path_length(sub, start)
        at_k = [k for k, d in dist.items() if d == HOPS]
        assert len(at_k) >= 1
        lonely += (len(at_k) == 1)
    # Allowed to happen occasionally by chance; must not be the rule.
    assert lonely <= 2, f"answer alone at distance {HOPS} in {lonely}/8 graphs"


def test_decoy_edges_are_indistinguishable_from_real_ones_in_the_graph(make_fan):
    """If a decoy edge carried an attribute, the topology would leak the chain."""
    g = make_fan()
    assert all(not d for _, _, d in g.edges(data=True))


def test_the_text_still_identifies_the_real_chain_under_fan_out(make_fan):
    """Structure prunes; TEXT must still resolve. Walking 'Continue at' works."""
    for graph_id in (0, 5, 17):
        g = make_fan(graph_id=graph_id)
        cur = g.graph["start_id"]
        for _ in range(HOPS):
            cur = _pointer_of(g, _definer(g, cur))
        assert cur == g.graph["gold_id"]


def test_a_decoy_is_never_the_real_successor_or_the_node_itself(make_fan):
    """A decoy pointing where the real one points would silently lower fan-out."""
    g = make_fan()
    for k in _content(g):
        text = g.nodes[k]["text"]
        real = re.search(r"Continue at (NODE-\d+)\.", text).group(1)
        found = re.findall(r"Decoy reference: (NODE-\d+)\.", text)
        assert len(found) == FAN - 1
        assert real not in found
        own = re.search(r"access code for (NODE-\d+) is", text).group(1)
        assert own not in found


def test_fan_out_one_is_byte_identical_to_the_original_build(make, make_fan):
    """fan_out=1 must not perturb the existing datasets in any way.

    Decoy targets come from their own rng stream precisely so that turning the
    knob on cannot shift the real pointers.
    """
    g = make()
    assert all("Decoy reference" not in g.nodes[k]["text"] for k in _content(g))
    assert "ignoring any decoy" not in g.nodes[g.graph["question_node"]]["text"]
    # and the decoy build really does differ
    assert "ignoring any decoy" in make_fan().nodes[
        make_fan().graph["question_node"]]["text"]


def test_fan_out_forks_the_cache_key_only_when_hops_are_on():
    base = RunConfig(hops=2)
    assert RunConfig(hops=2, fan_out=2).data_config_key() != base.data_config_key()
    # at hops=0 fan_out reaches no code path, so it must not orphan lookup caches
    assert RunConfig(fan_out=2).data_config_key() == RunConfig().data_config_key()


def test_spd_from_the_question_does_not_single_out_the_answer(make):
    """The subtle one — see the module docstring.

    A lone QUESTION->start edge puts the answer at the unique distance hops+1.
    The QUESTION therefore fans out to every content node, so distance from it
    is 1 for all of them and carries no information about the chain.
    """
    for graph_id in range(8):
        g = make(graph_id=graph_id)
        qn = g.graph["question_node"]
        dist = nx.single_source_shortest_path_length(g, qn)
        assert {dist[k] for k in _content(g)} == {1}


def test_chain_distance_survives_among_the_content_nodes(make):
    """Fanning the QUESTION out must not erase the structure the task needs."""
    g = make()
    start, answer = _definer(g, g.graph["start_id"]), _definer(g, g.graph["gold_id"])
    assert nx.shortest_path_length(g, start, answer) == HOPS


def test_graph_is_directed(make):
    """Direction is what magnetic_q encodes; the star task had none to encode."""
    g = make()
    assert g.is_directed()


# ── pairing across cells ──────────────────────────────────────────────────────

def test_the_chain_is_identical_across_cells(make):
    """Same blueprint -> same chain and same answer at every (N, T)."""
    ref = make(n=64, t=128)
    for (n, t) in ((16, 64), (16, 128), (64, 64)):
        g = make(n=n, t=t)
        assert g.graph["chain_ids"] == ref.graph["chain_ids"]
        assert g.graph["gold_code"] == ref.graph["gold_code"]
        assert g.graph["start_id"] == ref.graph["start_id"]


def test_content_nodes_are_exactly_t_tokens(make, tokenizer):
    for t in (64, 128):
        g = make(t=t)
        lengths = {len(tokenizer(g.nodes[k]["text"], add_special_tokens=False)["input_ids"])
                   for k in _content(g)}
        assert lengths == {t}


def test_a_cell_too_small_for_the_chain_is_rejected(make):
    """N-2 < hops+1 must fail loudly rather than build a truncated chain."""
    cfg = RunConfig(hops=8, node_counts=(8, 64), token_counts=(128,))
    with pytest.raises(ValueError, match="too few for a 8-hop chain"):
        make(n=8, t=128, cfg=cfg)


# ── config guards ─────────────────────────────────────────────────────────────

def test_validate_rejects_hops_the_smallest_cell_cannot_hold():
    with pytest.raises(ValueError, match="chain nodes"):
        RunConfig(hops=8, node_counts=(8, 64), token_counts=(128,)).validate()


def test_hops_zero_keeps_the_lookup_cache_key_unchanged():
    """hops=0 must not orphan the datasets README §3.1 already built."""
    assert "_h" not in RunConfig(hops=0).data_config_key()
    assert "_h3_" in RunConfig(hops=3).data_config_key()


def test_hops_changes_the_cache_key():
    keys = {RunConfig(hops=k).data_config_key() for k in (0, 1, 2, 3)}
    assert len(keys) == 4


def test_chain_slots_are_taken_from_the_front_of_the_nesting_order(make, pools, corpus):
    """That is what makes the chain present in every cell."""
    codes, ids = pools
    bp = make_blueprint(0, CFG, codes, ids, len(corpus), split="test")
    assert chain_slots(bp, CFG) == bp.slot_order[:HOPS + 1]
