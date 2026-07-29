"""Pin where the QUESTION and PROMPT nodes sit, and that attaching PROMPT is not a leak.

The PROMPT node carries the answer. Attaching it to the seed row (kgqa's convention --
`process_dataset.py:434` wires PROMPT to the topic entities unconditionally) changes the SPD
and magnetic tensors, which is the point. What it must *not* change is who can read the
answer span: the attention mask keys off prompt-node identity, never off edges. If that ever
stops being true, every AUROC this experiment produces is meaningless, and it will look like
a triumph rather than a failure.
"""

import pytest

from src.experiments.relbench.config import RunConfig
from src.experiments.relbench.data import (
    PROMPT_NODE_ID, QUESTION_NODE_ID, build_graph,
)


class _Sampled:
    """Three rows: seed `drivers` #0, a `results` child, a `races` grandparent."""

    nodes = [("drivers", 0, 0), ("results", 7, 1), ("races", 3, 2)]
    edges = [(1, 0), (1, 2)]


class _Renderer:
    def render(self, table, row, seed_ts, aligned=False):
        return f"#{row}" if aligned else f"{table} #{row}"

    def header(self, table):
        return f"TABLE {table} | id"

    def cheaper_as_schema_node(self, table, rows, seed_ts):
        return len(rows) > 1


def _build(**kw):
    return build_graph(_Sampled(), _Renderer(), 0, "QUESTION | q", "Answer: yes", **kw)


def test_prompt_attaches_to_the_seed_by_default():
    g = _build()
    assert (PROMPT_NODE_ID, 0) in g.edges, "default is kgqa's convention: PROMPT -> seed"


def test_prompt_can_be_isolated_for_the_ablation():
    g = _build(prompt_node="isolated")
    assert g.degree(PROMPT_NODE_ID) == 0


def test_prompt_edge_points_at_the_seed_not_at_an_arbitrary_row():
    """`add_edge(PROMPT, 0)` would silently attach to whatever row landed at index 0 if the
    sampler ever stopped putting the seed there."""
    g = _build()
    assert list(g.successors(PROMPT_NODE_ID)) == [0]
    assert g.nodes[0]["text"].startswith("TARGET "), "node 0 must be the seed row"


def test_attaching_the_prompt_does_not_connect_it_to_the_question():
    """The question node stays isolated by default; the prompt edge must not bridge them."""
    g = _build()
    assert g.degree(QUESTION_NODE_ID) == 0


def test_prompt_edge_does_not_change_the_supervised_span():
    """Topology feeds the bias; the answer text is identical either way."""
    attached, isolated = _build(), _build(prompt_node="isolated")
    assert attached.nodes[PROMPT_NODE_ID]["text"] == isolated.nodes[PROMPT_NODE_ID]["text"]
    assert attached.graph["prompt_node"] == isolated.graph["prompt_node"] == PROMPT_NODE_ID


def test_prompt_node_is_a_construction_knob():
    """It changes SPD and magnetic, so two settings must not share a built cache."""
    a = RunConfig().validate()
    b = RunConfig(prompt_node="isolated").validate()
    assert a.prompt_node == "seed", "attached is the default"
    assert a.data_config_key() != b.data_config_key()


def test_k_hop_is_rejected_when_the_prompt_is_isolated():
    """An edgeless prompt node reaches nothing, so the K-hop mask blinds the readout --
    kgqa's diagnosed k_hop collapse. Fail before the GPU, not after."""
    with pytest.raises(ValueError, match="blinds the readout"):
        RunConfig(k_hop=2, prompt_node="isolated").validate()
    RunConfig(k_hop=2, prompt_node="seed").validate()      # fine: PROMPT -> seed exists
