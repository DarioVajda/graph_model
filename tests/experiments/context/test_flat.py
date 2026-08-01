"""Invariants for the flat-text arm (README §3.1).

The flat arm exists to be compared against the graph arm, so what matters is that
the two differ in EXACTLY one thing — the input representation — and in nothing
else that could explain a gap:

  * the supervised span is the same ``code_len + 1`` tokens (code + EOS), so EM
    compares the same quantity in both arms;
  * the needle is present exactly once in the context and the gold id names it;
  * content-node order is shuffled, deterministic per item, and carries no
    positional cue about where the gold sits;
  * every content node survives serialization (nothing is silently dropped).

The label-alignment one is the dangerous invariant: ``FlatCollator`` masks all but
the final ``code_len + 1`` positions, which is only correct if the code really
tokenizes to ``code_len`` tokens at the end of the sequence. If that slipped, the
arm would score a wrong span and quietly report garbage.
"""

import pytest

from src.experiments.context.config import RunConfig
from src.experiments.context.data import ANSWER_PREFIX

transformers = pytest.importorskip("transformers")

from src.experiments.context.flat import (  # noqa: E402
    ARTICLE_HEADER, FlatCellView, FlatCollator, content_order, serialize_graph,
)

CFG = RunConfig(mode="flat_grid")


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(CFG.model_name)


@pytest.fixture(scope="module")
def split():
    """A real built cell; skip when the dataset has not been built."""
    from src.experiments.context.process_dataset import cell_split_name, load_split
    try:
        return load_split(CFG, cell_split_name(8, 32))
    except FileNotFoundError:
        pytest.skip("dataset not built; run --mode data_prep first")


@pytest.fixture(scope="module")
def collator(tokenizer):
    return FlatCollator(tokenizer, CFG.code_len, data_seed=CFG.data_seed)


# ── the supervised span (the dangerous one) ───────────────────────────────────

def test_supervised_span_is_exactly_the_code_and_eos(split, collator, tokenizer):
    view = FlatCellView(split)
    for i in (0, 7, 42):
        batch = collator([view[i]])
        ids, labels = batch["input_ids"][0], batch["labels"][0]
        sup = labels != -100
        assert int(sup.sum()) == CFG.code_len + 1
        sup_ids = ids[sup].tolist()
        assert sup_ids[-1] == tokenizer.eos_token_id
        decoded = tokenizer.decode(sup_ids[:-1]).strip()
        assert decoded == view.graphs[i].graph["gold_code"]


def test_supervised_span_sits_at_the_very_end(split, collator):
    """The mask is positional, so the code must be the final content."""
    view = FlatCellView(split)
    labels = collator([view[0]])["labels"][0]
    sup = (labels != -100).nonzero().flatten().tolist()
    assert sup == list(range(len(labels) - (CFG.code_len + 1), len(labels)))


def test_labels_and_input_ids_agree_on_the_span(split, collator):
    view = FlatCellView(split)
    batch = collator([view[3]])
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    sup = labels != -100
    assert ids[sup].tolist() == labels[sup].tolist()


# ── the needle is present and unique ──────────────────────────────────────────

def test_gold_code_appears_once_in_context_and_once_as_the_answer(split, collator, tokenizer):
    view = FlatCellView(split)
    for i in (0, 11):
        text = tokenizer.decode(collator([view[i]])["input_ids"][0].tolist())
        g = view.graphs[i].graph
        assert text.count(g["gold_code"]) == 2      # gold node + the answer
        assert text.count(g["gold_id"]) == 2        # gold node + the QUESTION
        assert text.rstrip().endswith(g["gold_code"] + tokenizer.eos_token)


def test_every_content_node_survives_serialization(split, tokenizer):
    view = FlatCellView(split)
    g = view.graphs[0]
    n = g.graph["cell_n"]
    text = serialize_graph(g, content_order(g, 0))
    assert text.count("## Article") == n - 2
    for node in g.nodes:
        if node not in (g.graph["question_node"], g.graph["prompt_node"]):
            assert g.nodes[node]["text"] in text


def test_serialization_ends_with_the_answer_prefix(split):
    view = FlatCellView(split)
    g = view.graphs[0]
    assert serialize_graph(g, content_order(g, 0)).endswith(ANSWER_PREFIX)


# ── the shuffle ───────────────────────────────────────────────────────────────

def test_order_is_deterministic_per_item_but_varies_across_items(split):
    view = FlatCellView(split)
    g = view.graphs[0]
    assert content_order(g, 5) == content_order(g, 5)
    assert content_order(g, 5) != content_order(g, 6)


def test_order_is_a_permutation_of_the_content_nodes(split):
    view = FlatCellView(split)
    g = view.graphs[0]
    expected = {k for k in g.nodes
                if k not in (g.graph["question_node"], g.graph["prompt_node"])}
    assert set(content_order(g, 0)) == expected
    assert len(content_order(g, 0)) == len(expected)


def test_gold_position_is_not_pinned(split):
    """Across items the gold node lands at many different serialized positions.

    If the shuffle were broken the gold would sit at a fixed slot and the arm
    would measure position-copying rather than retrieval.
    """
    view = FlatCellView(split)
    positions = set()
    for i in range(40):
        g = view.graphs[i]
        order = content_order(g, CFG.data_seed + i)
        gold_id = g.graph["gold_id"]
        for slot, node in enumerate(order):
            if gold_id in g.nodes[node]["text"]:
                positions.add(slot)
                break
    assert len(positions) > 1


# ── the header cannot be confused with a code ─────────────────────────────────

def test_article_header_cannot_collide_with_a_code(split, tokenizer):
    """Codes are LLDLL; the header is 'Article <n>'. No header renders a code."""
    from src.experiments.context.data import build_code_pool
    codes = set(build_code_pool(tokenizer, CFG.code_len, 256, seed=CFG.data_seed))
    for i in range(1, 200):
        assert ARTICLE_HEADER.format(i=i) not in codes
        assert not any(c in ARTICLE_HEADER.format(i=i) for c in codes)


# ── batching ──────────────────────────────────────────────────────────────────

def test_multi_row_batch_left_pads_and_keeps_the_span_at_the_end(split, collator):
    view = FlatCellView(split)
    batch = collator([view[0], view[1]])
    assert batch["input_ids"].shape[0] == 2
    for row in range(2):
        labels = batch["labels"][row]
        sup = (labels != -100).nonzero().flatten().tolist()
        assert sup == list(range(len(labels) - (CFG.code_len + 1), len(labels)))
        # padding is on the LEFT, so attention_mask starts 0s and ends 1s
        attn = batch["attention_mask"][row].tolist()
        assert attn[-1] == 1
        assert attn == sorted(attn)
