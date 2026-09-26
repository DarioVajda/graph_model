"""Invariants for the placement probe (README §3.4).

The probe compares conditions paired per graph, so a condition must differ from
``random`` in the ORDER of the node texts and in nothing else:

  * the chain is found correctly, and the block is contiguous, in the stated direction,
    at the stated place, with the other nodes' relative order untouched;
  * ``random`` is byte-identical to the §3.3 flat serialization, so its scores are
    directly checkable against ``flat_trained_grid.jsonl``;
  * the graph-arm permutation moves every node-indexed field together — a field left
    in the old order would hand the model a corrupted graph and read as a position
    effect.
"""

import pytest
import torch

from src.experiments.context.config import RunConfig
from src.experiments.context.data import KV_TEMPLATE

transformers = pytest.importorskip("transformers")

from src.experiments.context.flat import (  # noqa: E402
    FlatCellView, FlatCollator, content_order, serialize_graph,
)
from src.experiments.context.placement import (  # noqa: E402
    CONDITIONS, PLACEMENTS, PlacedFlatCollator, PlacedGraphView, answer_position,
    chain_nodes, condition_order, graph_base_order, parse_condition, permute_item,
    placed_order,
)

# The §3.3 main-sweep build: the only one with k-mixture test splits.
CFG = RunConfig(
    mode="placement", checkpoint_path="unused", hop_counts=(1, 2, 3, 4), fan_out=2,
    node_counts=(16, 32, 64, 128), token_counts=(64, 128, 256, 512),
    max_train_len=16384, n_train=16000, n_dev=200, n_test=200, code_len=3,
    id_pool=4096, data_seed=42, spd=True, max_spd=8, rrwp=False, magnetic=True,
    magnetic_dim=128, magnetic_m=128, magnetic_q=0.25,
)


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(CFG.model_name)


@pytest.fixture(scope="module", params=[2, 4])
def split(request):
    from src.experiments.context.process_dataset import cell_split_name, load_split
    try:
        return load_split(CFG, cell_split_name(16, 64, request.param))
    except FileNotFoundError:
        pytest.skip("main-sweep dataset not built; run sbatch_mainsweep_data.sh first")


# ── the chain ──────────────────────────────────────────────────────────────────

def test_chain_nodes_follow_the_pointers_to_the_gold_code(split):
    for g in split.graphs[:20]:
        chain = chain_nodes(g)
        assert len(chain) == g.graph["hops"] + 1
        for a, b in zip(chain, chain[1:]):
            succ_id = g.graph["chain_ids"][chain.index(b)]
            assert f"Continue at {succ_id}." in g.nodes[a]["text"]
        answer_kv = KV_TEMPLATE.format(node_id=g.graph["gold_id"], code=g.graph["gold_code"])
        assert answer_kv in g.nodes[chain[-1]]["text"]


# ── the order ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("direction", ["fwd", "rev"])
@pytest.mark.parametrize("p", PLACEMENTS)
def test_placed_order_is_a_contiguous_block_at_p(direction, p):
    base = list(range(20))
    chain = [13, 2, 7]
    order = placed_order(base, chain, direction, p)
    assert sorted(order) == base
    at = order.index(chain[0] if direction == "fwd" else chain[-1])
    block = chain if direction == "fwd" else chain[::-1]
    assert order[at:at + len(chain)] == block
    assert at == round(p * (len(base) - len(chain)))
    assert [j for j in order if j not in chain] == [j for j in base if j not in chain]


def test_the_extremes_put_the_block_first_and_last():
    base, chain = list(range(10)), [4, 8]
    assert placed_order(base, chain, "fwd", 0.0)[:2] == [4, 8]
    assert placed_order(base, chain, "fwd", 1.0)[-2:] == [4, 8]
    assert placed_order(base, chain, "rev", 1.0)[-2:] == [8, 4]


def test_answer_position_reaches_both_ends(split):
    g = split.graphs[0]
    base = graph_base_order(g)
    assert answer_position(condition_order(g, base, "rev@0.00"), g) == 0.0
    assert answer_position(condition_order(g, base, "fwd@1.00"), g) == 1.0


def test_parse_condition_round_trips_every_condition():
    for cond in CONDITIONS:
        direction, p = parse_condition(cond)
        assert cond == "random" if p is None else cond == f"{direction}@{p:.2f}"
    with pytest.raises(ValueError):
        parse_condition("sideways@0.5")


# ── the flat arm ───────────────────────────────────────────────────────────────

def test_random_is_byte_identical_to_the_grid_serialization(split, tokenizer):
    view = FlatCellView(split)
    plain = FlatCollator(tokenizer, CFG.code_len, data_seed=CFG.data_seed)
    placed = PlacedFlatCollator(tokenizer, CFG.code_len, data_seed=CFG.data_seed,
                                condition="random")
    for i in (0, 5, 17):
        a, b = plain([view[i]]), placed([view[i]])
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["labels"], b["labels"])


@pytest.mark.parametrize("cond", CONDITIONS[1:])
def test_a_condition_changes_the_order_and_nothing_else(split, tokenizer, cond):
    g = split.graphs[3]
    base = content_order(g, CFG.data_seed + 3)
    order = condition_order(g, base, cond)
    assert sorted(order) == sorted(base)
    a, b = serialize_graph(g, base), serialize_graph(g, order)
    # Same node texts, same numbered headers, same question and answer prefix: only
    # the sequence differs.
    assert a != b or order == base
    assert sorted(a.split("\n")) == sorted(b.split("\n"))
    assert a.split("\n")[-2:] == b.split("\n")[-2:]
    view = FlatCellView(split)
    batch = PlacedFlatCollator(tokenizer, CFG.code_len, data_seed=CFG.data_seed,
                               condition=cond)([view[3]])
    sup = batch["labels"][0] != -100
    assert tokenizer.decode(batch["input_ids"][0][sup][:-1]).strip() == g.graph["gold_code"]


# ── the graph arm ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cond", ["random", "fwd@0.00", "rev@0.50", "fwd@1.00"])
def test_permute_item_moves_every_node_field_together(split, cond):
    g = split.graphs[1]
    item = split[1]
    content = condition_order(g, graph_base_order(g), cond)
    new_to_old = [g.graph["question_node"]] + content + [g.graph["prompt_node"]]
    out = permute_item(item, g, content)

    assert out["prompt_node"] == len(new_to_old) - 1
    for new, old in enumerate(new_to_old):
        assert out["text"][new] == item["text"][old]
        assert out["input_ids"][new] == item["input_ids"][old]
        assert torch.equal(out["magnetic_V"][new], item["magnetic_V"][old])
        for new2, old2 in enumerate(new_to_old):
            assert out["shortest_path_dists"][new, new2] == item["shortest_path_dists"][old, old2]
    old_edges = set(item["edges"])
    assert {(new_to_old[u], new_to_old[v]) for u, v in out["edges"]} == old_edges
    assert torch.equal(out["magnetic_lambdas"], item["magnetic_lambdas"])
    assert torch.equal(out["labels"], item["labels"])


def test_graph_random_is_the_stored_order(split):
    g = split.graphs[0]
    out = PlacedGraphView(split, "random")[0]
    assert out["text"] == split[0]["text"]
    assert out["edges"] == split[0]["edges"]


def test_permute_item_refuses_a_field_it_cannot_reorder(split):
    g = split.graphs[0]
    item = dict(split[0], rwse=torch.zeros(g.number_of_nodes(), 4))
    with pytest.raises(ValueError, match="rwse"):
        permute_item(item, g, graph_base_order(g))


# ── config ─────────────────────────────────────────────────────────────────────

def test_only_conditions_filters_in_canonical_order():
    cfg = RunConfig(**{**CFG.__dict__, "only_conditions": "rev@0.50,random"})
    assert cfg.selected_conditions() == ("random", "rev@0.50")
    assert CFG.selected_conditions() == CONDITIONS


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="unknown conditions"):
        RunConfig(**{**CFG.__dict__, "only_conditions": "fwd@0.33"}).validate()


def test_placement_requires_a_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        RunConfig(**{**CFG.__dict__, "checkpoint_path": None}).validate()
