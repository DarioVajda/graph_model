"""
Tier A — structural questions over real molecules, with RDKit as the oracle.

This is the `substructure` probe (`probes/README.md` Probe 2, explicitly *"Proxy
for: molecules"*) promoted from glued-together synthetic rings to real chemistry
against a real opponent. Labels are exact and free, so Tier A is a **generator**,
not a corpus: fresh examples every epoch, no repetition to overfit, and — per
`src/generalist/PLAN.md` §5 — exactly the kind of data the trunk prefers.

Every task returns a **single-token answer** (` Yes`/` No`, or a numeral 0-9), so
the shared last-token supervision in `dataset.py` lands exactly on the answer.
Counts above `MAX_COUNT` make the molecule ineligible rather than truncating the
label, so a "9" never silently means "9 or more".

Held out permanently (PLAN.md §4.1): `bond_path`. It is listed here because the
generator must be able to *build* it for the held-out evaluation — never for the
training mixture. `dataset.py` refuses to put it in a training split.
"""

from __future__ import annotations

import networkx as nx
from rdkit import Chem

from .data import HELD_OUT_TIER_A_TASKS

YES, NO = " Yes", " No"
MAX_COUNT = 9

TIER_A_TASKS = (
    "ring_membership",
    "aromatic_ring",
    "ring_count",
    "ring_size",
    "fg_presence",
    "fg_atom_membership",
    "fg_count",
    "bond_path",
    "longest_chain",
    "stereo_potential",
    "stereo_assigned",
)

#: Tasks whose question names one or more specific atoms. These require
#: `atom_labels=True` in both arms so "atom 14" resolves to the same atom in the
#: graph node text and in the flat arm's atom-mapped SMILES.
ATOM_LEVEL_TASKS = frozenset(
    {"ring_membership", "aromatic_ring", "ring_size", "bond_path",
     "fg_atom_membership"})

#: SMARTS patterns for `fg_presence` / `fg_count`. Chosen to be common enough in
#: drug-like molecules that both classes are reachable, and to be genuine subgraph
#: matches rather than single-atom lookups.
FUNCTIONAL_GROUPS = {
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "hydroxyl group": "[OX2H]",
    "primary amine": "[NX3;H2;!$(NC=O)]",
    "ketone": "[#6][CX3](=O)[#6]",
    "ether": "[OD2]([#6])[#6]",
    "nitro group": "[NX3](=O)=O",
    "amide": "[NX3][CX3](=[OX1])",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "nitrile": "[NX1]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
}

_SMARTS = {name: Chem.MolFromSmarts(p) for name, p in FUNCTIONAL_GROUPS.items()}


def _count(n: int) -> str | None:
    """Render a count as a single-token answer, or None if out of range."""
    return f" {n}" if 0 <= n <= MAX_COUNT else None


def _mol_nx(mol) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(a.GetIdx() for a in mol.GetAtoms())
    g.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Task generators
#
# Each returns ``(question, answer, named_atoms)`` or ``None`` when this molecule
# cannot yield a valid (and, for binary tasks, class-balanced) example. Returning
# None is normal: the caller draws another molecule.
# ─────────────────────────────────────────────────────────────────────────────

def _binary_atom_task(mol, rng, predicate, question_fmt):
    """Shared shape: split atoms by a predicate, pick a class 50/50, name an atom.

    Balancing per *example* (rather than filtering molecules) keeps the label
    distribution at 50/50 without biasing which molecules are seen — the same
    construction the probe suite uses.
    """
    positive = [a.GetIdx() for a in mol.GetAtoms() if predicate(a)]
    negative = [a.GetIdx() for a in mol.GetAtoms() if not predicate(a)]
    if not positive or not negative:
        return None
    label = rng.random() < 0.5
    idx = rng.choice(positive if label else negative)
    return question_fmt.format(atom=idx + 1), (YES if label else NO), [idx]


def ring_membership(mol, rng):
    return _binary_atom_task(
        mol, rng, lambda a: a.IsInRing(),
        "Question: is atom {atom} part of a ring?")


def aromatic_ring(mol, rng):
    return _binary_atom_task(
        mol, rng, lambda a: a.GetIsAromatic(),
        "Question: is atom {atom} part of an aromatic ring?")


def ring_count(mol, rng):
    answer = _count(mol.GetRingInfo().NumRings())
    if answer is None:
        return None
    return "Question: how many rings does this molecule have?", answer, []


def ring_size(mol, rng):
    """Smallest ring containing a named atom; `0` when the atom is not in a ring.

    Balanced between in-ring and not-in-ring atoms so `0` does not dominate.
    """
    info = mol.GetRingInfo()
    in_ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    off_ring = [a.GetIdx() for a in mol.GetAtoms() if not a.IsInRing()]
    if not in_ring or not off_ring:
        return None
    if rng.random() < 0.5:
        idx = rng.choice(in_ring)
        sizes = [len(r) for r in info.AtomRings() if idx in r]
        answer = _count(min(sizes)) if sizes else None
    else:
        idx = rng.choice(off_ring)
        answer = _count(0)
    if answer is None:
        return None
    return (f"Question: what is the size of the smallest ring containing "
            f"atom {idx + 1}?"), answer, [idx]


def fg_presence(mol, rng):
    name = rng.choice(list(FUNCTIONAL_GROUPS))
    present = mol.HasSubstructMatch(_SMARTS[name])
    return (f"Question: does this molecule contain a {name}?",
            YES if present else NO, [])


def fg_atom_membership(mol, rng):
    """`fg_presence`'s question, asked about a NAMED atom. The controlled twin.

    This exists to separate two explanations of why the graph arm loses every
    molecule-level family and wins every atom-level one (PLAN.md §3.2.5). It holds
    the chemistry fixed — same `FUNCTIONAL_GROUPS`, same SMARTS, same Yes/No answer
    as `fg_presence` — and changes exactly one thing: the question names an atom.

    That single change flips both candidate causes at once, which is the point:

    * it makes the prompt wiring `named` rather than `all`, so `spd[prompt, :]`
      becomes a graded distance profile instead of the two-valued row that carries
      no query information;
    * it forces `atom_labels` on BOTH arms, which inflates the flat arm's SMILES
      with `[cH:14]` atom maps and collapses the token-dilution ratio from ~5.3x
      (fg_presence) to ~2.2x.

    So `fg_atom_membership` vs `fg_presence` is the paired comparison: if the graph
    arm wins here and loses there on identical chemistry, the deficit is about how
    the question addresses the graph, not about the chemistry being hard.

    Balanced by construction: sample a group the molecule actually contains, then
    pick a member atom or a non-member atom with equal probability.
    """
    name = rng.choice(list(FUNCTIONAL_GROUPS))
    matches = mol.GetSubstructMatches(_SMARTS[name])
    if not matches:
        return None                      # nothing to be a member of; try another molecule
    members = {i for match in matches for i in match}
    outsiders = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in members]
    if not outsiders:
        return None                      # every atom is in the group; no negative exists
    if rng.random() < 0.5:
        idx, answer = rng.choice(sorted(members)), YES
    else:
        idx, answer = rng.choice(outsiders), NO
    return (f"Question: is atom {idx + 1} part of a {name}?", answer, [idx])


def fg_count(mol, rng):
    name = rng.choice(list(FUNCTIONAL_GROUPS))
    answer = _count(len(mol.GetSubstructMatches(_SMARTS[name])))
    if answer is None:
        return None
    return f"Question: how many {name}s does this molecule contain?", answer, []


def bond_path(mol, rng):
    """Bond distance between two named atoms. **HELD OUT** — see PLAN.md §4.1.

    Chosen as the held-out Tier-A family for the same reason `direction` was
    chosen among the probes: SPD *is* the answer by construction, so transfer
    here is unambiguous rather than a judgement call.
    """
    if mol.GetNumAtoms() < 2:
        return None
    graph = _mol_nx(mol)
    for _ in range(8):
        u, v = rng.sample(range(mol.GetNumAtoms()), 2)
        if not nx.has_path(graph, u, v):
            continue
        answer = _count(nx.shortest_path_length(graph, u, v))
        if answer is not None:
            return (f"Question: how many bonds separate atom {u + 1} and "
                    f"atom {v + 1}?"), answer, [u, v]
    return None


def longest_chain(mol, rng):
    """Longest unbranched chain of non-ring carbons.

    Restricted to non-ring carbons on purpose: that induced subgraph is a forest,
    so the longest path is the tree diameter and is computable exactly in linear
    time. Longest simple path in a general graph is NP-hard, and a task whose
    ground truth is expensive to verify is a bad task regardless of the model.
    """
    carbons = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() == "C" and not a.IsInRing()]
    if not carbons:
        return None
    sub = _mol_nx(mol).subgraph(carbons)
    longest = 0
    for component in nx.connected_components(sub):
        tree = sub.subgraph(component)
        # Tree diameter in nodes = (edge diameter + 1); double-BFS is exact here.
        far = max(nx.single_source_shortest_path_length(tree, next(iter(component))).items(),
                  key=lambda kv: kv[1])[0]
        depth = max(nx.single_source_shortest_path_length(tree, far).values())
        longest = max(longest, depth + 1)
    answer = _count(longest)
    if answer is None:
        return None
    return ("Question: how many carbon atoms are in the longest unbranched chain "
            "of non-ring carbons?"), answer, []


def _chiral_centers(mol, unassigned):
    return Chem.FindMolChiralCenters(
        mol, includeUnassigned=unassigned, useLegacyImplementation=False)


def stereo_potential(mol, rng):
    """How many atoms *could* be stereocentres — pure connectivity.

    Determined entirely by the graph: an atom qualifies when its four branches are
    constitutionally distinct. The graph arm should **win** this (PLAN.md gate A2).
    """
    answer = _count(len(_chiral_centers(mol, unassigned=True)))
    if answer is None:
        return None
    return ("Question: how many atoms in this molecule could be stereocenters?",
            answer, [])


def stereo_assigned(mol, rng):
    """How many stereocentres have a **defined** configuration.

    The negative control. This answer lives in the SMILES `@`/`@@` tags and is not
    in a plain atom-bond graph, so with `stereo_tags: off` the graph arm must sit
    at chance; with the parity tag on it becomes solvable but still requires CIP
    ranking over the graph. The gap between those two runs is the measurement
    (PLAN.md §1) — and the CIP *label* is never placed in node text.
    """
    assigned = [c for c in _chiral_centers(mol, unassigned=True) if c[1] != "?"]
    answer = _count(len(assigned))
    if answer is None:
        return None
    return ("Question: how many stereocenters in this molecule have a defined "
            "configuration?"), answer, []


TASK_GENERATORS = {
    "ring_membership": ring_membership,
    "aromatic_ring": aromatic_ring,
    "ring_count": ring_count,
    "ring_size": ring_size,
    "fg_presence": fg_presence,
    "fg_atom_membership": fg_atom_membership,
    "fg_count": fg_count,
    "bond_path": bond_path,
    "longest_chain": longest_chain,
    "stereo_potential": stereo_potential,
    "stereo_assigned": stereo_assigned,
}

#: The answer vocabulary, for label-distribution reporting and for asserting
#: single-token-ness at build time.
ANSWER_VOCAB = (YES, NO) + tuple(f" {i}" for i in range(MAX_COUNT + 1))


def assert_tier_a_wired():
    """Every declared task has a generator, and the held-out one is real."""
    missing = set(TIER_A_TASKS) - set(TASK_GENERATORS)
    if missing:
        raise AssertionError(f"tasks declared but not implemented: {sorted(missing)}")
    unknown = set(HELD_OUT_TIER_A_TASKS) - set(TIER_A_TASKS)
    if unknown:
        raise AssertionError(
            f"held-out task {sorted(unknown)} is not a Tier-A task; the held-out "
            "declaration in PLAN.md §4.1 would be silently vacuous")
    return True
