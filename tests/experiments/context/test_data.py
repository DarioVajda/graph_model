"""Build invariants for the Needle-in-a-Graph dataset.

These are the four things that, if broken, produce a perfectly plausible heatmap of
a build bug rather than of the model (README §A.4):

  * a content node is **exactly** T tokens (T is the x-axis of the figure);
  * the gold code and gold id live in exactly one content node, and the QUESTION
    node names that id;
  * cells are **paired** — the same blueprint across all 25 of them, nested node
    subsets along N, and only the within-node needle offset redrawn along T;
  * the supervised span is exactly ``code_len`` tokens (+ EOS), so exact match
    compares the same quantity in every cell.

The builder asserts the first two on every graph it makes (``check_split``); this
file covers all four independently, and the pairing ones cannot be checked inside
a single split at all.
"""

import os
import re

import pytest

from src.experiments.context.config import RunConfig
from src.experiments.context.data import (
    ANSWER_PREFIX, answer_prefix_len, build_code_pool, build_id_pool, fit_node_text,
    load_corpus, make_blueprint, needle_offsets, realize,
)
from src.experiments.context.process_dataset import RAW_DATA_DIR

transformers = pytest.importorskip("transformers")

CFG = RunConfig(node_counts=(8, 16, 32), token_counts=(32, 64, 128),
                magnetic_m=32, id_pool=512)
CORPUS_TOKENS = 1_000_000


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(CFG.model_name)


@pytest.fixture(scope="module")
def corpus(tokenizer):
    """The cached filler stream; skip rather than download inside the test suite."""
    path = os.path.join(
        RAW_DATA_DIR,
        f"filler_wikitext-103-raw-v1_{CFG.model_name.split('/')[-1]}_{RunConfig().corpus_tokens}.npy")
    if not os.path.exists(path):
        pytest.skip(f"filler corpus not built ({path}); run --mode data_prep first")
    return load_corpus(tokenizer, RAW_DATA_DIR, RunConfig().corpus_tokens, verbose=False)


@pytest.fixture(scope="module")
def pools(tokenizer):
    return (build_code_pool(tokenizer, CFG.code_len, CFG.id_pool, seed=CFG.data_seed),
            build_id_pool(CFG.id_pool))


def _blueprint(corpus, pools, graph_id=0):
    codes, ids = pools
    return make_blueprint(graph_id, CFG, codes, ids, len(corpus), split="test")


def _content_nodes(g, n):
    return [k for k in g.nodes if k not in (0, n - 1)]


def _node_id_of(text):
    m = re.search(r"NODE-\d+", text)
    return m.group() if m else None


# ── exact token counts ────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,t", [(8, 32), (16, 64), (32, 128)])
def test_content_nodes_are_exactly_t_tokens(tokenizer, corpus, pools, n, t):
    g = realize(_blueprint(corpus, pools), CFG, n, t, tokenizer, corpus, split="test")
    lengths = {len(tokenizer(g.nodes[k]["text"], add_special_tokens=False)["input_ids"])
               for k in _content_nodes(g, n)}
    assert lengths == {t}


def test_code_pool_is_fixed_token_length(tokenizer, pools):
    codes, _ = pools
    for code in codes[:200]:
        assert len(tokenizer(" " + code, add_special_tokens=False)["input_ids"]) == CFG.code_len


def test_code_pool_is_fixed_character_length(pools):
    """Equal-length distinct strings cannot contain one another.

    Regression: v1 filtered only on TOKEN length, so 4-, 5- and 6-character codes
    shared one pool and gold "IREO" sat inside distractor "OIREO" — an ambiguous
    item that the build assertion caught on graph 113 of the first full build.
    """
    codes, _ = pools
    assert len({len(c) for c in codes}) == 1


def test_codes_cannot_occur_in_english_prose(pools):
    """An interior digit flanked by letters ("AB1CD") does not appear in wikitext.

    Regression: 1.0% of the v1 pool was all-digit ("8192", "0595"), which occur
    freely in the filler corpus as years and quantities — making the needle
    findable in a non-gold node.
    """
    codes, _ = pools
    for code in codes[:500]:
        assert any(c.isdigit() for c in code) and any(c.isalpha() for c in code)
        assert code[0].isalpha() and code[-1].isalpha()
        assert not code.isdigit()


def test_no_code_is_a_substring_of_another(pools):
    """The property the two tests above exist to guarantee, checked directly."""
    codes = sorted(pools[0][:600], key=len)
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            assert a not in b, f"{a!r} is a substring of {b!r}"


def test_fit_node_text_keeps_the_needle_and_the_length(tokenizer, corpus):
    kv = "The access code for NODE-00001 is ABC."
    for t in (32, 64, 128):
        for offset in (0, 3, t // 2):
            text, ids = fit_node_text(tokenizer, corpus, 1234, kv, offset, t)
            assert len(ids) == t
            assert kv in text


# ── the needle is unique and findable ─────────────────────────────────────────

def test_gold_code_and_id_appear_in_exactly_one_content_node(tokenizer, corpus, pools):
    n, t = 16, 64
    g = realize(_blueprint(corpus, pools), CFG, n, t, tokenizer, corpus, split="test")
    texts = {k: g.nodes[k]["text"] for k in _content_nodes(g, n)}
    assert sum(g.graph["gold_code"] in v for v in texts.values()) == 1
    assert sum(g.graph["gold_id"] in v for v in texts.values()) == 1
    assert g.graph["gold_id"] in g.nodes[0]["text"]          # QUESTION names it
    assert g.graph["gold_code"] in g.nodes[n - 1]["text"]    # PROMPT answers it


def test_distractor_codes_are_distinct_from_the_gold(tokenizer, corpus, pools):
    g = realize(_blueprint(corpus, pools), CFG, 32, 32, tokenizer, corpus, split="test")
    codes = g.graph["codes"]
    assert len(set(codes)) == len(codes)
    assert codes.count(g.graph["gold_code"]) == 1


def test_topology_is_a_star_with_an_isolated_prompt(tokenizer, corpus, pools):
    n, t = 16, 32
    g = realize(_blueprint(corpus, pools), CFG, n, t, tokenizer, corpus, split="test")
    assert g.number_of_nodes() == n
    assert g.degree(0) == n - 2                    # QUESTION is the star centre
    assert g.degree(n - 1) == 0                    # PROMPT is isolated
    assert all(g.degree(k) == 1 for k in _content_nodes(g, n))


# ── pairing across the grid ───────────────────────────────────────────────────

def test_node_subsets_are_nested_along_n(tokenizer, corpus, pools):
    bp = _blueprint(corpus, pools)
    t = 64
    sets = []
    for n in (8, 16, 32):
        g = realize(bp, CFG, n, t, tokenizer, corpus, split="test")
        sets.append({_node_id_of(g.nodes[k]["text"]) for k in _content_nodes(g, n)})
    assert sets[0] < sets[1] < sets[2]


def test_gold_is_present_in_every_cell(tokenizer, corpus, pools):
    bp = _blueprint(corpus, pools)
    for n in (8, 16, 32):
        for t in (32, 64, 128):
            g = realize(bp, CFG, n, t, tokenizer, corpus, split="test")
            ids = {_node_id_of(g.nodes[k]["text"]) for k in _content_nodes(g, n)}
            assert g.graph["gold_id"] in ids
            assert g.graph["gold_code"] == bp.codes[bp.gold_slot]


def test_only_the_needle_offset_changes_along_t(tokenizer, corpus, pools):
    """Same nodes, same codes, same gold — a different within-node position."""
    bp = _blueprint(corpus, pools)
    a = realize(bp, CFG, 16, 32, tokenizer, corpus, split="test")
    b = realize(bp, CFG, 16, 128, tokenizer, corpus, split="test")
    ids_a = {_node_id_of(a.nodes[k]["text"]) for k in _content_nodes(a, 16)}
    ids_b = {_node_id_of(b.nodes[k]["text"]) for k in _content_nodes(b, 16)}
    assert ids_a == ids_b
    assert a.graph["gold_code"] == b.graph["gold_code"]
    assert needle_offsets(bp, CFG, 32, "test") != needle_offsets(bp, CFG, 128, "test")


def test_needle_offsets_do_not_depend_on_n(tokenizer, corpus, pools):
    """Offsets are drawn for every slot, so a cell's needle position is N-independent."""
    bp = _blueprint(corpus, pools)
    offsets = needle_offsets(bp, CFG, 64, "test")
    assert len(offsets) == CFG.n_content_max()
    assert needle_offsets(bp, CFG, 64, "test") == offsets      # deterministic


def test_blueprints_are_deterministic(corpus, pools):
    a = _blueprint(corpus, pools, graph_id=5)
    b = _blueprint(corpus, pools, graph_id=5)
    assert (a.codes, a.node_ids, a.slot_order, a.filler_at) == \
           (b.codes, b.node_ids, b.slot_order, b.filler_at)
    assert _blueprint(corpus, pools, graph_id=6).codes != a.codes


# ── the supervised span ───────────────────────────────────────────────────────

def test_prompt_node_supervises_exactly_the_code_and_eos(tokenizer, corpus, pools):
    g = realize(_blueprint(corpus, pools), CFG, 8, 32, tokenizer, corpus, split="test")
    prompt_text = g.nodes[7]["text"]
    assert prompt_text.startswith(ANSWER_PREFIX)
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    ids.append(tokenizer.eos_token_id)             # what tokenize(add_eos=True) does
    supervised = len(ids) - answer_prefix_len(tokenizer)
    assert supervised == CFG.code_len + 1
