"""The assistant arm's few-shot graph (`MOLECULE_GENERALIST.md` §9.4).

A demonstration is a different molecule, and the whole claim of this encoding is
that the graph says so: the demonstrations are their own components, the prompt
reaches them only through their own node, and the target molecule is the one
thing the prompt touches directly. Those are structural properties, not
formatting ones, so they are tested on the graph rather than on the text.
"""

from __future__ import annotations

import networkx as nx
import pytest
from rdkit import Chem

from src.experiments.molecules.config import RunConfig
from src.experiments.molecules.dataset import (build_assistant_example,
                                               build_graph_example)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PHENOL = "Oc1ccccc1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"


@pytest.fixture
def cfg():
    return RunConfig(task="ring_membership").validate()


def _shots(*smiles):
    return [(Chem.MolFromSmiles(s), f"Question about {s}?", f" Answer {i}")
            for i, s in enumerate(smiles)]


def _build(cfg, shots, question="Question: is atom 2 part of a ring?",
           answer=" Yes", named=(1,)):
    return build_assistant_example(Chem.MolFromSmiles(ASPIRIN), question, answer,
                                   list(named), shots, cfg)


def test_with_no_demonstrations_it_is_the_ordinary_graph_example(cfg):
    """The few-shot builder has to be a strict extension, or the 78 % of the set
    that carries no demonstrations would be encoded differently from every other
    molecule task in the campaign.

    `cfg.task` here is atom-level, so `build_graph_example` labels its atoms and
    wires the prompt to the named one; passing the same named atom and
    ``atom_labels=True`` is what makes the two comparable."""
    mol = Chem.MolFromSmiles(ASPIRIN)
    question, answer = "Question: is atom 2 part of a ring?", " Yes"
    plain = build_graph_example(mol, question, answer, [1], cfg)
    composed = build_assistant_example(mol, question, answer, [1], [], cfg)

    assert composed.number_of_nodes() == plain.number_of_nodes()
    assert set(composed.edges) == set(plain.edges)
    assert composed.graph["prompt_node"] == plain.graph["prompt_node"]
    assert composed.graph.get("question_node") == plain.graph.get("question_node")
    for i in range(plain.number_of_nodes()):
        assert composed.nodes[i]["text"] == plain.nodes[i]["text"]


def test_each_demonstration_is_its_own_component(cfg):
    """Cut the prompt node and the molecules must fall apart. A bond between two
    molecules would be a chemical claim nothing computed."""
    graph = _build(cfg, _shots(PHENOL, ETHANOL))
    without_prompt = graph.copy()
    without_prompt.remove_node(graph.graph["prompt_node"])
    without_prompt.remove_node(graph.graph["question_node"])

    components = list(nx.connected_components(without_prompt.to_undirected()))
    # One for the target, one for each demonstration (its molecule plus its node).
    assert len(components) == 3
    sizes = sorted(len(c) for c in components)
    # Aspirin is the largest; phenol and ethanol carry their demonstration node.
    assert sizes[0] < sizes[-1]


def test_a_demonstration_sits_exactly_one_hop_behind_the_target(cfg):
    """The distances are the point, and they are the target's own distances + 1.

    Under `rich_levi` the prompt reaches the target's *atoms* in one hop and its
    Levi bond nodes in two. A demonstration reproduces that pattern one hop
    further out — its node at 1, its atoms at 2, its bond nodes at 3 — because
    the demonstration node anchors its molecule exactly as the prompt node
    anchors the target's. So the bias features see the same local structure
    around a demonstration as around the target, shifted by one, and "which
    molecule is the question about" is answerable from the prompt's SPD row.

    Wiring the prompt straight to every atom in the graph would collapse the
    target and the demonstrations to a single distance and leave that question
    answerable only from the text.
    """
    graph = _build(cfg, _shots(PHENOL), named=())
    prompt = graph.graph["prompt_node"]
    lengths = nx.single_source_shortest_path_length(graph, prompt)

    shot_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "shot"]
    assert len(shot_nodes) == 1
    assert lengths[shot_nodes[0]] == 1

    demo = set(nx.descendants(graph, shot_nodes[0]))
    assert demo, "the demonstration node must point at its own molecule"
    target = (set(lengths) - demo - {prompt, shot_nodes[0],
                                     graph.graph["question_node"]})
    assert target, "the prompt must reach the target molecule"

    def _by_kind(nodes, kind):
        return [n for n in nodes if graph.nodes[n].get("kind") == kind]

    for kind, target_distance in (("atom", 1), ("bond", 2)):
        mine = _by_kind(target, kind)
        theirs = _by_kind(demo, kind)
        assert mine and theirs, f"both molecules need {kind} nodes to compare"
        assert {lengths[n] for n in mine} == {target_distance}
        assert {lengths[n] for n in theirs} == {target_distance + 1}


def test_two_demonstrations_of_the_same_molecule_stay_two_components(cfg):
    """Namespacing by demonstration index, checked. Shared node keys would weld
    the two molecules into one and silently halve the context."""
    graph = _build(cfg, _shots(BENZENE, BENZENE))
    without_prompt = graph.copy()
    without_prompt.remove_node(graph.graph["prompt_node"])
    without_prompt.remove_node(graph.graph["question_node"])
    assert len(list(nx.connected_components(without_prompt.to_undirected()))) == 3


def test_demonstrations_are_read_before_the_question_that_refers_to_them(cfg):
    """`TextGraphDataset` reads node text in index order, so index order is
    reading order: the worked examples come before the question."""
    graph = _build(cfg, _shots(PHENOL, ETHANOL))
    shot_indices = [n for n, d in graph.nodes(data=True) if d.get("kind") == "shot"]
    target_indices = [n for n, d in graph.nodes(data=True)
                      if d.get("kind") == "atom" and n not in _demo_nodes(graph)]

    assert max(shot_indices) < min(target_indices)
    assert max(target_indices) < graph.graph["question_node"]
    assert graph.graph["question_node"] < graph.graph["prompt_node"]
    assert graph.graph["prompt_node"] == graph.number_of_nodes() - 1


def _demo_nodes(graph) -> set:
    """Every node belonging to a demonstration, found through its own node."""
    out = set()
    for node, data in graph.nodes(data=True):
        if data.get("kind") == "shot":
            out |= set(nx.descendants(graph, node)) | {node}
    return out


def test_the_scored_position_is_untouched_by_the_demonstrations(cfg):
    """The supervised token is the last token of the prompt node. If attaching
    demonstrations moved it, the few-shot rows would be scored somewhere else."""
    plain = _build(cfg, [])
    composed = _build(cfg, _shots(PHENOL, ETHANOL))
    assert (composed.nodes[composed.graph["prompt_node"]]["text"]
            == plain.nodes[plain.graph["prompt_node"]]["text"] == "\nA: Yes")


def test_the_named_atom_is_the_only_one_the_prompt_wires_to(cfg):
    """An assistant example whose facts are all about atom 2 points there. The
    indices are RDKit's own, 0-based — `assistant.named_atoms_for` converts from
    the 1-based numbering the fact sheets and Tier-A questions use."""
    graph = _build(cfg, [], named=(1,))
    prompt = graph.graph["prompt_node"]
    wired = [n for n in graph.successors(prompt)]
    assert len(wired) == 1
    assert "atom2 " in graph.nodes[wired[0]]["text"]


def test_molecule_level_facts_wire_the_prompt_to_the_whole_molecule(cfg):
    graph = _build(cfg, [], named=())
    prompt = graph.graph["prompt_node"]
    atoms = [n for n, d in graph.nodes(data=True) if d.get("kind") == "atom"]
    assert set(graph.successors(prompt)) == set(atoms)


def test_atoms_are_labelled_so_a_question_can_name_one(cfg):
    """Most of §9.4's fact families name an atom, and an unlabelled graph leaves
    "atom 20" pointing at nothing."""
    graph = _build(cfg, _shots(PHENOL), named=())
    labelled = [d["text"] for _n, d in graph.nodes(data=True)
                if d.get("kind") == "atom"]
    assert all(t.startswith("atom") for t in labelled)
    # …including the demonstration's own molecule, whose question names atoms too.
    assert sum(t.startswith("atom1 ") for t in labelled) == 2


def test_a_demonstrations_text_carries_its_question_and_answer(cfg):
    graph = _build(cfg, _shots(PHENOL))
    text = [d["text"] for _n, d in graph.nodes(data=True)
            if d.get("kind") == "shot"][0]
    assert f"Question about {PHENOL}?" in text
    assert "Answer 0" in text
