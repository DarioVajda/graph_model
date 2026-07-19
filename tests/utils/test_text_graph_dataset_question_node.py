"""
Graph-level attributes naming a node must survive relabeling.

TextGraphDataset relabels nodes twice (to 0..N-1, then into RCM order). NetworkX only
remaps the nodes, not graph-level attributes that *name* a node, so each such attribute
has to be remapped explicitly. `prompt_node` always was; `question_node` — set by the
experiments that give the question its own prefix node — was not, leaving a stale label
that raises KeyError for anything that trusts it.
"""

import networkx as nx

from src.utils.text_graph_dataset import TextGraphDataset


def _graph():
    g = nx.DiGraph()
    for name in ("alice", "bob", "carol"):
        g.add_node(name, text=name)
    g.add_edge("alice", "bob")
    g.add_edge("bob", "carol")
    g.add_node("QUESTION", text="Q: who?")
    g.add_node("PROMPT", text="A: bob")
    g.add_edge("PROMPT", "alice")
    g.graph["prompt_node"] = "PROMPT"
    g.graph["question_node"] = "QUESTION"
    return g


def test_question_node_is_remapped_to_an_index():
    ds = TextGraphDataset([_graph()], per_graph_versions=1)
    built = ds.graphs[0]

    qn = built.graph["question_node"]
    pn = built.graph["prompt_node"]

    # both must be valid indices into the relabeled graph ...
    assert qn in built.nodes, f"stale question_node label {qn!r}"
    assert pn in built.nodes
    assert qn != pn
    # ... pointing at the nodes they named before relabeling
    assert built.nodes[qn]["text"] == "Q: who?"
    assert built.nodes[pn]["text"] == "A: bob"


def test_question_node_absent_is_left_alone():
    g = _graph()
    del g.graph["question_node"]
    ds = TextGraphDataset([g], per_graph_versions=1)
    assert "question_node" not in ds.graphs[0].graph
