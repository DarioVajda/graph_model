"""Round-trip fidelity of the molecule encodings.

PLAN.md §8 calls this the highest-value test in the campaign, for a specific reason:
an encoding bug is otherwise completely silent. A dropped aromatic flag or a bond
wired to the wrong Levi node still trains, still shows a falling loss, and produces a
merely mediocre number that reads as an architectural limitation weeks later.

Each encoding is checked at the level it *claims* to preserve (`ROUNDTRIP_LEVEL`), so
a pass is a statement about information content — and the declared losses of the
cheaper encodings are asserted rather than assumed.
"""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from src.experiments.molecules.data import (  # noqa: E402
    ENCODINGS,
    REJECTED_ENCODINGS,
    atom_text,
    attach_question,
    flat_serialize,
    mol_to_graph,
    relabel_for_dataset,
    roundtrip_check,
    scaffold_split,
)


# A deliberately awkward spread: aromatics, fused rings, charges, stereo, halogens,
# a nitro group, a macrocycle-ish chain, and a single heavy atom.
SMILES = [
    "CCO",                                   # ethanol
    "c1ccccc1",                              # benzene
    "Oc1ccccc1",                             # phenol
    "CC(=O)Oc1ccccc1C(=O)O",                 # aspirin
    "C[C@@H](N)C(=O)O",                      # L-alanine, stereo
    "C[C@H](N)C(=O)O",                       # D-alanine, the mirror image
    "c1ccc2ccccc2c1",                        # naphthalene, fused
    "[NH4+].[Cl-]",                          # charged, disconnected
    "O=[N+]([O-])c1ccccc1",                  # nitrobenzene, formal charges
    "C/C=C/C",                               # trans-2-butene, bond stereo
    "C1CCCCCCCCC1",                          # cyclodecane
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",          # caffeine
    "O",                                     # water — single atom, no bonds
    "FC(F)(F)Br",                            # halogens
]


@pytest.fixture(params=SMILES, ids=lambda s: s[:24])
def mol(request):
    m = Chem.MolFromSmiles(request.param)
    assert m is not None, f"fixture SMILES failed to parse: {request.param}"
    return m


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_roundtrip_at_declared_level(mol, encoding):
    ok, level, expected, got = roundtrip_check(mol, encoding=encoding)
    assert ok, (f"{encoding} round-trip failed at level {level!r}\n"
                f"  expected: {expected}\n  got:      {got}")


@pytest.mark.parametrize("encoding", REJECTED_ENCODINGS)
def test_rejected_encoding_raises(encoding):
    """The fourth cell is refused at construction, not scored badly at eval."""
    m = Chem.MolFromSmiles("CC=CC")
    with pytest.raises(ValueError, match="rejected by construction"):
        mol_to_graph(m, encoding=encoding)


def test_terse_atom_only_would_lose_bond_order():
    """The reason the fourth cell is rejected, asserted rather than asserted-in-prose.

    Under a terse atom style with no bond nodes, butane and 2-butene would produce
    identical node text on identical topology.
    """
    single = Chem.MolFromSmiles("CCCC")
    double = Chem.MolFromSmiles("CC=CC")
    terse = [atom_text(a, "terse") for a in single.GetAtoms()]
    assert terse == [atom_text(a, "terse") for a in double.GetAtoms()]


def test_levi_doubles_the_node_count(mol):
    """Levi's cost, stated as a test so a silent change to it is caught."""
    levi = mol_to_graph(mol, encoding="rich_levi")
    plain = mol_to_graph(mol, encoding="rich_atom_only")
    assert levi.number_of_nodes() == plain.number_of_nodes() + mol.GetNumBonds()


def test_stereo_tags_switch_changes_only_chiral_atoms():
    """`stereo_tags: off` must remove the parity tag and nothing else.

    This is the switch gate A2 depends on: with it off, `stereo_assigned` has to sit
    at chance, and that is only meaningful if nothing else moved.
    """
    m = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")
    on = mol_to_graph(m, encoding="rich_levi", stereo_tags=True)
    off = mol_to_graph(m, encoding="rich_levi", stereo_tags=False)

    differing = [n for n in on.nodes
                 if on.nodes[n]["text"] != off.nodes[n]["text"]]
    assert differing, "the test molecule has a stereocenter; something should differ"
    for node in differing:
        assert "chiral" in on.nodes[node]["text"]
        assert "chiral" not in off.nodes[node]["text"]
        assert on.nodes[node]["text"].replace(" chiral cw", "").replace(" chiral ccw", "") \
            == off.nodes[node]["text"]


def test_cip_label_never_appears_in_node_text():
    """The line between information and answer-leakage (PLAN.md §1).

    The parity tag is standard GNN input. The CIP R/S label is the answer to
    `stereo_assigned` and must never reach the node text.
    """
    m = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    graph = mol_to_graph(m, encoding="rich_levi", stereo_tags=True)
    for _, attrs in graph.nodes(data=True):
        fields = attrs["text"].split()
        assert "R" not in fields and "S" not in fields, attrs["text"]


def test_enantiomers_differ_only_when_tags_are_on():
    """The two alanines are the same graph; only the parity tag separates them."""
    left = Chem.MolFromSmiles("C[C@@H](N)C(=O)O")
    right = Chem.MolFromSmiles("C[C@H](N)C(=O)O")

    def texts(m, tags):
        g = mol_to_graph(m, encoding="rich_levi", stereo_tags=tags)
        return sorted(d["text"] for _, d in g.nodes(data=True))

    assert texts(left, False) == texts(right, False), "graphs must be identical"
    assert texts(left, True) != texts(right, True), "parity tag must separate them"


def test_attach_question_prompt_edges(mol):
    graph = mol_to_graph(mol, encoding="rich_levi")
    n_atoms = mol.GetNumAtoms()

    named = attach_question(graph, "Is atom 0 in a ring?", " Yes",
                            named_atoms=[0], prompt_edges="named")
    assert named.out_degree(named.graph["prompt_node"]) == 1

    every = attach_question(graph, "Is it soluble?", " Yes", prompt_edges="all")
    assert every.out_degree(every.graph["prompt_node"]) == n_atoms

    none = attach_question(graph, "Is it soluble?", " Yes", prompt_edges="none")
    assert none.out_degree(none.graph["prompt_node"]) == 0


def test_question_node_is_edge_free(mol):
    """`question_node="on"` — the QUESTION node sits in the prefix with no edges.

    Edge-free is the point: the question is visible to every node through
    attention, but it contributes nothing to the SPD / magnetic features, so the
    structural bias still describes the molecule alone.
    """
    graph = attach_question(mol_to_graph(mol), "Q?", " Yes", prompt_edges="all")
    q = graph.graph["question_node"]
    assert graph.degree(q) == 0


def test_question_node_off_puts_the_question_in_the_prompt(mol):
    """The `off` layout has no QUESTION node at all; the prefix is query-blind."""
    graph = attach_question(mol_to_graph(mol), "Q?", " Yes",
                            prompt_edges="all", question_node="off")
    assert "question_node" not in graph.graph
    assert graph.nodes[graph.graph["prompt_node"]]["text"] == "Q?\nA: Yes"


def test_relabel_puts_prompt_last(mol):
    graph = relabel_for_dataset(
        attach_question(mol_to_graph(mol), "Q?", " Yes", prompt_edges="all"))
    n = graph.number_of_nodes()
    assert graph.graph["prompt_node"] == n - 1
    assert graph.graph["question_node"] == n - 2
    assert set(graph.nodes) == set(range(n))
    assert all("text" in d for _, d in graph.nodes(data=True))


def test_randomised_smiles_vary_but_denote_one_molecule(mol):
    """The §6 permutation experiment's premise: same molecule, different strings.

    Variation is expected only where the molecule is topologically asymmetric.
    Benzene and cyclodecane have a single atom symmetry class, so *every* traversal
    yields the same string and randomisation is a no-op — which is a real constraint
    on §6's effect size, not a defect: the flat arm's order-sensitivity can only show
    up on molecules that have distinguishable starting atoms.
    """
    canonical = flat_serialize(mol)
    variants = {flat_serialize(mol, canonical=False, seed=s) for s in range(10)}
    for variant in variants:
        assert Chem.MolToSmiles(Chem.MolFromSmiles(variant)) == canonical

    symmetry_classes = len(set(Chem.CanonicalRankAtoms(mol, breakTies=False)))
    if mol.GetNumAtoms() > 3 and symmetry_classes > 1:
        assert len(variants) > 1, "asymmetric molecule produced no SMILES variation"


def test_atom_labels_agree_between_the_two_arms(mol):
    """Both arms must resolve "atom N" to the same atom, or Tier A is rigged.

    RDKit reads atom map number 0 as "unmapped", so a 0-based scheme drops atom 0's
    label in the flat arm only — silently, and in the direction that would make the
    graph arm look better. Labels are 1-based in both arms for exactly this reason.
    """
    import re

    graph = mol_to_graph(mol, encoding="rich_levi", atom_labels=True)
    graph_labels = sorted(
        int(re.match(r"atom(\d+)", d["text"]).group(1))
        for _, d in graph.nodes(data=True) if d["kind"] == "atom")

    flat = flat_serialize(mol, atom_labels=True)
    flat_labels = sorted(int(n) for n in re.findall(r":(\d+)\]", flat))

    assert graph_labels == flat_labels == list(range(1, mol.GetNumAtoms() + 1))


def test_atom_labels_are_off_by_default(mol):
    """Molecule-level tasks name no atom; an index in the text would be noise."""
    for _, attrs in mol_to_graph(mol, encoding="rich_levi").nodes(data=True):
        assert not attrs["text"].startswith("atom")


def test_scaffold_split_is_deterministic_and_disjoint():
    smiles = SMILES * 8
    a = scaffold_split(smiles)
    b = scaffold_split(smiles)
    assert a == b, "scaffold split must be a function of the dataset, not of a seed"

    train, valid, test = a
    assert sorted(train + valid + test) == list(range(len(smiles)))
    assert not (set(train) & set(valid)) and not (set(valid) & set(test))


def test_scaffold_split_separates_scaffolds():
    """The property that makes the split worth using: no scaffold spans two splits."""
    from src.experiments.molecules.data import murcko_scaffold

    smiles = SMILES * 8
    train, valid, test = scaffold_split(smiles)
    by_split = {}
    for split_name, idx in (("train", train), ("valid", valid), ("test", test)):
        for i in idx:
            by_split.setdefault(murcko_scaffold(smiles[i]), set()).add(split_name)
    for scaffold, splits in by_split.items():
        assert len(splits) == 1, f"scaffold {scaffold!r} spans {splits}"
