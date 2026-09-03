"""
Molecule -> GTLM graph encoding, the flat (SMILES) control, scaffold splitting,
and the round-trip fidelity test.

This module is deliberately **free of torch and of `TextGraphDataset`** so that it
imports and runs on the login node: M0's dataset statistics (`analyse_dataset.py`)
must be measurable before any GPU is committed. Dataset assembly lives elsewhere.

The three encodings are the three usable cells of PLAN.md §3.2:

    rich_levi        atoms carry full per-atom text; every bond is its own node
    terse_levi       atoms carry only an element word; every bond is its own node
    rich_atom_only   no bond nodes; bond orders summarised into the atom's text

The fourth cell (`terse_atom_only`) is rejected at construction, not measured: with
no bond node *and* no bond summary, a double bond is indistinguishable from a single
one, so the encoding is information-destroying rather than merely weak.

Node text examples:

    rich atom   "carbon aromatic ring deg3 H1"          (+ " charge+1", + " chiral cw")
    terse atom  "carbon"
    rich bond   "double bond in ring"
    terse bond  "double"

`stereo_tags` controls whether the *parity* tag (RDKit's chiral tag) enters the atom
text. It is standard GNN input — OGB's `atom_to_feature_vector` carries chirality as
its second feature — so `True` is the Tier-B and trunk default. The CIP R/S *label*
is never emitted: that is the answer to the `stereo_assigned` task, not information
about the molecule (PLAN.md §1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os

import networkx as nx
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit is loud about valence and kekulisation on public molecule sets; the parse
# failures we care about are counted explicitly by `load_tier_b`, not read off stderr.
RDLogger.DisableLog("rdApp.*")


ENCODINGS = ("rich_levi", "terse_levi", "rich_atom_only")
#: Rejected by construction — see the module docstring.
REJECTED_ENCODINGS = ("terse_atom_only",)

#: Where the prompt node's *edges* go. Edges do not control token visibility (the
#: bidirectional-prefix mask already lets the prompt read every prefix node); they
#: control the prompt node's row of the SPD / magnetic bias features. A prompt node
#: with no edges has a constant SPD row, i.e. the graph arm is structurally blank
#: exactly where the answer is generated — the failure diagnosed in the landmark
#: campaign. So molecule-level tasks wire to "all" rather than leaving it empty.
PROMPT_EDGE_MODES = ("named", "all", "none")

#: Where the question lives. ``"on"`` — its own edge-free prefix node, so the
#: question is IN the prefix and every atom and bond node attends to it, making
#: node representations question-conditioned. ``"off"`` — the question sits inside
#: the prompt node instead, leaving the prefix query-blind and forcing the molecule
#: to be encoded question-agnostically (the layout `probes` uses).
#:
#: ``"on"`` is the default and is settled; do not move it without a concrete
#: reason. graphqa and kgqa call this value ``"isolated"`` — there is exactly one
#: sensible placement here, so it is named for what it does rather than for how it
#: is built, and the synonym is rejected rather than aliased.
QUESTION_NODE_MODES = ("on", "off")

_ELEMENT_WORDS = {
    "C": "carbon", "N": "nitrogen", "O": "oxygen", "S": "sulfur", "P": "phosphorus",
    "F": "fluorine", "Cl": "chlorine", "Br": "bromine", "I": "iodine", "B": "boron",
    "Si": "silicon", "Se": "selenium", "H": "hydrogen",
}

_BOND_WORDS = {
    Chem.BondType.SINGLE: "single",
    Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple",
    Chem.BondType.AROMATIC: "aromatic",
}

_CHIRAL_WORDS = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: "cw",
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: "ccw",
}

_BOND_STEREO_WORDS = {
    Chem.BondStereo.STEREOE: "E",
    Chem.BondStereo.STEREOZ: "Z",
}


# ─────────────────────────────────────────────────────────────────────────────
# Node text
# ─────────────────────────────────────────────────────────────────────────────

_WORD_ELEMENTS = {word: symbol for symbol, word in _ELEMENT_WORDS.items()}


def _element_word(symbol: str) -> str:
    return _ELEMENT_WORDS.get(symbol, symbol.lower())


def _symbol_from_word(word: str) -> str:
    """Inverse of `_element_word`, for decoding node text back to a molecule."""
    return _WORD_ELEMENTS.get(word, word.capitalize())


def atom_text(atom, style: str, stereo_tags: bool = True,
              bond_summary: bool = False) -> str:
    """Render one atom as node text.

    ``style="terse"`` emits the element word alone. ``style="rich"`` adds the fields
    a standard GNN featuriser receives: aromaticity, ring membership, heavy-atom
    degree, implicit+explicit hydrogens, formal charge, and (when ``stereo_tags``)
    the chiral parity tag.

    ``bond_summary=True`` appends the multiset of incident bond orders. It exists
    for ``rich_atom_only``, where there are no bond nodes to carry that information.
    """
    if style not in ("rich", "terse"):
        raise ValueError(f"atom style must be 'rich' or 'terse', got {style!r}")

    word = _element_word(atom.GetSymbol())
    if style == "terse":
        # Deliberately lossy: no charge, no hydrogens, no aromaticity. The
        # round-trip test asserts exactly this loss rather than pretending it away.
        return word

    parts = [word]
    if atom.GetIsAromatic():
        parts.append("aromatic")
    if atom.IsInRing():
        parts.append("ring")
    parts.append(f"deg{atom.GetDegree()}")
    parts.append(f"H{atom.GetTotalNumHs()}")

    charge = atom.GetFormalCharge()
    if charge:
        parts.append(f"charge{charge:+d}")

    # Isotope and radical count are rare (a few dozen molecules across all of Tier B:
    # technetium tracers, nitric oxide) but they are part of the molecule's identity,
    # and omitting them was silently costing `rich_levi` its exact round trip at M0.
    if atom.GetIsotope():
        parts.append(f"iso{atom.GetIsotope()}")
    if atom.GetNumRadicalElectrons():
        parts.append(f"rad{atom.GetNumRadicalElectrons()}")

    if stereo_tags:
        chiral = _CHIRAL_WORDS.get(atom.GetChiralTag())
        if chiral:
            parts.append(f"chiral {chiral}")

    if bond_summary:
        counts = defaultdict(int)
        for bond in atom.GetBonds():
            counts[_BOND_WORDS.get(bond.GetBondType(), "other")] += 1
        if counts:
            rendered = " ".join(f"{name}{counts[name]}"
                                for name in sorted(counts))
            parts.append(f"bonds {rendered}")

    return " ".join(parts)


class UnsupportedBondType(ValueError):
    """A bond this encoding cannot represent without silently losing information."""


def is_encodable(mol):
    """``(ok, reason)`` — whether every bond in ``mol`` has a faithful encoding.

    Measured over all of Tier B at M0: 1.63M bonds are single/double/triple/aromatic
    and **10 are dative**, all in organometallic iron complexes in HIV. Dative bonds
    are directional (donor -> acceptor) and our Levi encoding is undirected, so they
    cannot be represented without losing that direction. Ten molecules is not worth a
    directional special case in the bias path, so they are dropped and counted.
    Counted, not silently skipped: a dropped molecule bounds every metric computed
    downstream, exactly as a parse failure does.
    """
    for bond in mol.GetBonds():
        if bond.GetBondType() not in _BOND_WORDS:
            return False, str(bond.GetBondType())
    return True, ""


def bond_text(bond, style: str, stereo_tags: bool = True) -> str:
    """Render one bond as the text of its Levi node."""
    if style not in ("rich", "terse"):
        raise ValueError(f"bond style must be 'rich' or 'terse', got {style!r}")

    if bond.GetBondType() not in _BOND_WORDS:
        raise UnsupportedBondType(
            f"bond type {bond.GetBondType()!s} has no faithful text encoding; "
            "screen molecules with `is_encodable` before building graphs")
    word = _BOND_WORDS[bond.GetBondType()]
    if style == "terse":
        return word

    parts = [word, "bond"]
    if bond.IsInRing():
        parts.append("in ring")
    if stereo_tags:
        stereo = _BOND_STEREO_WORDS.get(bond.GetStereo())
        if stereo:
            parts.append(f"stereo {stereo}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Molecule -> graph
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_graph(mol, encoding: str = "rich_levi", stereo_tags: bool = True,
                 atom_labels: bool = False) -> nx.DiGraph:
    """Build the GTLM graph for one RDKit molecule.

    Returns a ``DiGraph`` (edges in both directions — molecules are undirected;
    see PLAN.md §3.3 for why that makes the magnetic bias a *spectral* channel here
    rather than a directional one) with, on every node:

    * ``text``      — what the model reads.
    * ``kind``      — ``"atom"`` or ``"bond"``; bookkeeping, not read by the model.
    * ``atom_idx``  — for atom nodes, the RDKit atom index. This is the handle a
      task uses to point a question at a specific atom, and it is what makes the
      round-trip test possible.

    ``atom_labels=True`` prefixes each atom's text with ``"atom<i> "``. Tier-A
    questions name a specific atom ("is atom 14 in a ring?"), and without a label
    there is nothing in the node text for the question to refer to. Use the matching
    ``flat_serialize(..., atom_labels=True)`` for the flat arm so both arms resolve
    "atom 14" to the *same* atom — otherwise the comparison is not a controlled one.
    Off by default: molecule-level Tier-B tasks name no atom, and an arbitrary index
    in the text is noise there.

    No prompt or question node is attached here; see `attach_question`.
    """
    if encoding in REJECTED_ENCODINGS:
        raise ValueError(
            f"encoding {encoding!r} is rejected by construction, not by measurement: "
            "with no bond node and no bond summary in the atom text, bond order is "
            "unrecoverable. See PLAN.md §3.2.")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")

    atom_style = "terse" if encoding.startswith("terse") else "rich"
    levi = encoding.endswith("_levi")

    graph = nx.DiGraph()

    for atom in mol.GetAtoms():
        text = atom_text(atom, atom_style, stereo_tags=stereo_tags,
                         bond_summary=not levi)
        if atom_labels:
            # 1-based: RDKit treats atom map number 0 as "unmapped", so a 0-based
            # scheme would silently drop the label for atom 0 in the flat arm while
            # keeping it in the graph arm — a mismatch that rigs Tier A invisibly.
            text = f"atom{atom.GetIdx() + 1} {text}"
        graph.add_node(
            ("atom", atom.GetIdx()),
            text=text,
            kind="atom",
            atom_idx=atom.GetIdx(),
        )

    for bond in mol.GetBonds():
        u = ("atom", bond.GetBeginAtomIdx())
        v = ("atom", bond.GetEndAtomIdx())
        if levi:
            node = ("bond", bond.GetIdx())
            graph.add_node(node,
                           text=bond_text(bond, atom_style, stereo_tags=stereo_tags),
                           kind="bond")
            # atom <-> bond <-> atom, both directions on both incidences
            for endpoint in (u, v):
                graph.add_edge(endpoint, node)
                graph.add_edge(node, endpoint)
        else:
            graph.add_edge(u, v)
            graph.add_edge(v, u)

    return graph


def attach_question(graph: nx.DiGraph, question: str, answer: str,
                    named_atoms=(), prompt_edges: str = "named",
                    question_node: str = "on",
                    answer_prefix: str = "\nA:") -> nx.DiGraph:
    """Attach the QUESTION prefix node and the PROMPT node.

    Mirrors `graphqa/process_dataset.py::example_to_graph` so the supervised span
    and generation anchor are byte-identical in shape to every other experiment:
    with ``question_node="on"`` (the default) the question body sits in its own
    edge-free prefix node and the prompt node holds ``answer_prefix + answer``.

    **Naming.** graphqa and kgqa spell these two values ``"isolated"`` / ``"off"``,
    because there they are two of several conceivable placements. Here there is
    only one placement worth having, so the value is simply ``"on"``: the question
    is in the prefix and the graph attends to it. ``QUESTION_NODE_MODES`` is the
    list, and ``"isolated"`` is not accepted — a silently-accepted synonym would
    put two spellings of one arm into the run records.

    ``prompt_edges`` decides where the PROMPT node's edges go — which feeds the
    bias features, not token visibility (see `PROMPT_EDGE_MODES`):

    * ``"named"`` — to the atoms the question is about. Atom-level Tier-A tasks.
    * ``"all"``   — to every atom node. Molecule-level Tier-B tasks, which have no
      named atom and would otherwise leave the prompt's SPD row constant.
    * ``"none"``  — no edges. A control, not a default.

    ``answer_prefix`` has **no trailing space**: every answer in `tasks.py` carries
    its own leading space, so the prompt tail is exactly ``"\\nA: Yes"`` here and in
    `dataset.build_flat_example`. The two arms must agree byte-for-byte on the
    tokens preceding the supervised one, or the comparison carries an uncontrolled
    difference at the only position that is scored.
    """
    if prompt_edges not in PROMPT_EDGE_MODES:
        raise ValueError(f"prompt_edges must be one of {PROMPT_EDGE_MODES}, "
                         f"got {prompt_edges!r}")
    if question_node not in QUESTION_NODE_MODES:
        raise ValueError(
            f"question_node must be one of {QUESTION_NODE_MODES}, got {question_node!r}. "
            "(graphqa/kgqa spell 'on' as 'isolated'; this experiment does not accept "
            "that synonym — see attach_question's docstring.)")

    graph = graph.copy()

    if question_node == "on":
        q_node = ("question", 0)
        graph.add_node(q_node, text=question, kind="question")
        graph.graph["question_node"] = q_node
        prompt_node = ("prompt", 0)
        graph.add_node(prompt_node, text=f"{answer_prefix}{answer}", kind="prompt")
    else:
        prompt_node = ("prompt", 0)
        graph.add_node(prompt_node, text=f"{question}{answer_prefix}{answer}",
                       kind="prompt")
    graph.graph["prompt_node"] = prompt_node

    if prompt_edges == "named":
        targets = [("atom", i) for i in named_atoms]
    elif prompt_edges == "all":
        targets = [n for n, d in graph.nodes(data=True) if d.get("kind") == "atom"]
    else:
        targets = []

    for target in targets:
        if target not in graph:
            raise ValueError(f"prompt edge target {target!r} is not in the graph")
        graph.add_edge(prompt_node, target)

    return graph


def relabel_for_dataset(graph: nx.DiGraph) -> nx.DiGraph:
    """Relabel the tuple keys to contiguous ints, prompt node last.

    `TextGraphDataset` / `GraphCollatorV2` pack the prompt node last regardless, but
    an integer-keyed graph with the prompt at ``N-1`` keeps the node ids readable in
    every debug dump, and keeps `graph.graph['prompt_node']` an index rather than a
    tuple.
    """
    prompt = graph.graph["prompt_node"]
    question = graph.graph.get("question_node")

    others = [n for n in graph.nodes if n not in (prompt, question)]
    order = others + ([question] if question is not None else []) + [prompt]
    mapping = {node: i for i, node in enumerate(order)}

    out = nx.relabel_nodes(graph, mapping, copy=True)
    out.graph = dict(graph.graph)
    out.graph["prompt_node"] = mapping[prompt]
    if question is not None:
        out.graph["question_node"] = mapping[question]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The flat control
# ─────────────────────────────────────────────────────────────────────────────

def flat_serialize(mol, canonical: bool = True, seed: int | None = None,
                   atom_labels: bool = False) -> str:
    """The flat twin's input: a SMILES string.

    ``canonical=False`` with a ``seed`` produces a *randomised* SMILES for the same
    molecule — the permutation-invariance experiment of PLAN.md §6. GTLM's answer is
    invariant to atom order by Property 1; the flat arm's is not, and the spread
    across randomisations is the measurement.

    ``atom_labels=True`` writes RDKit atom map numbers, so the SMILES renders as
    ``[cH:14]`` and "atom 14" in a Tier-A question resolves to the same atom the
    graph arm's ``atom14`` node names. Without this the flat arm is being asked a
    question it has no way to parse, which would make Tier A a rigged comparison
    rather than a measurement. Labels are **1-based** in both arms, because RDKit
    reads map number 0 as "unmapped" and would drop atom 0's label here only.
    """
    if atom_labels:
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    if canonical:
        return Chem.MolToSmiles(mol, canonical=True)
    if seed is None:
        raise ValueError("randomised SMILES needs a seed, so the eval is reproducible")
    # `MolToSmiles(..., randomSeed=)` was removed in rdkit 2026.03; the seeded
    # generator is the supported path and is what keeps §6's eval reproducible.
    return Chem.MolToRandomSmilesVect(mol, 1, randomSeed=seed)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip fidelity — the highest-value test in the plan (PLAN.md §8, M1)
# ─────────────────────────────────────────────────────────────────────────────

#: What each encoding is *expected* to preserve. An encoding bug is otherwise
#: silent: the model trains, the loss falls, the number is merely mediocre, and it
#: reads as an architectural limitation six weeks later.
ROUNDTRIP_LEVEL = {
    # canonical SMILES must match: elements, bond orders, charges, hydrogens,
    # isotopes and radicals all survive. Only stereo is compared stereo-free.
    "rich_levi": "exact",
    # elements + connectivity + bond orders, as a labelled graph. NOT as SMILES —
    # see the note below, this distinction is a measured finding.
    "terse_levi": "labelled_graph",
    # elements + connectivity only; bond orders are summarised into the atom text
    # rather than carried per-bond, so they are not individually recoverable.
    "rich_atom_only": "topology",
}

#: Measured at M0 over 6,467 molecule x encoding round trips across all nine Tier-B
#: datasets: `rich_levi` 1 failure, `rich_atom_only` 0, `terse_levi` 406 (6.3%) when
#: compared as SMILES — every one of them an aromatic heterocycle.
#:
#: **The terse finding is a result, not a bug.** Aromaticity perception needs
#: hydrogen counts: pyrrole's `[nH]` is aromatic only because that nitrogen carries
#: an H. `terse` emits "nitrogen" and drops the count, so RDKit cannot re-perceive
#: the ring and kekulises it into a saturated one. The *labelled graph* is intact —
#: elements, bonds and bond orders all round-trip — but the molecule is no longer
#: reconstructible as a chemical object.
#:
#: So `terse` is not merely "less verbose than rich": it is lossy for ~6% of
#: drug-like molecules, concentrated in N-heterocycles, which are among the most
#: common motifs in medicinal chemistry. That is a substantive prior for the
#: `terse × levi` arm of PLAN.md §3.2 and it was not visible before measuring.
TERSE_AROMATIC_LOSS = "aromaticity is not recoverable without hydrogen counts"


def _skeleton_smiles(mol) -> str:
    """Canonical SMILES of the charge-stripped, H-implicit, stereo-free skeleton."""
    stripped = Chem.RWMol(mol)
    for atom in stripped.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        # `terse` carries the element word and nothing else, so isotope and radical
        # state are declared losses too and the comparison must not ask for them.
        atom.SetIsotope(0)
        atom.SetNumRadicalElectrons(0)
    out = stripped.GetMol()
    # `RemoveStereochemistry` clears chiral tags, bond stereo AND bond *directions*;
    # clearing only the first two leaves the "/" marks in the emitted SMILES, so a
    # stereo-bearing molecule would fail a comparison that is not about stereo.
    Chem.RemoveStereochemistry(out)
    Chem.SanitizeMol(out, catchErrors=True)
    return Chem.MolToSmiles(out, canonical=True)


def _apply_rich_fields(atom, fields):
    """Set an atom's properties from the fields of its `rich` node text.

    Stereo parity (``chiral cw``) is deliberately *not* reapplied: reconstructing a
    CIP-consistent configuration needs the neighbour ordering the graph discards, so
    stereo is compared stereo-free (see `roundtrip_check`).
    """
    for field in fields:
        if field == "aromatic":
            atom.SetIsAromatic(True)
        elif field.startswith("H") and field[1:].isdigit():
            atom.SetNumExplicitHs(int(field[1:]))
            atom.SetNoImplicit(True)
        elif field.startswith("charge"):
            atom.SetFormalCharge(int(field[len("charge"):]))
        elif field.startswith("iso"):
            atom.SetIsotope(int(field[len("iso"):]))
        elif field.startswith("rad"):
            atom.SetNumRadicalElectrons(int(field[len("rad"):]))


def graph_to_mol(graph: nx.DiGraph, encoding: str, sanitize: bool = True):
    """Rebuild an RDKit molecule from the graph's *topology and node text alone*.

    Nothing is read from the source molecule — the element comes from the atom node's
    own word, the bond order from the bond node's own word, the endpoints from the
    graph. That is what makes this a test of the encoding rather than a copy of the
    input: a mis-rendered field or a bond wired to the wrong Levi node fails it.
    """
    rich = not encoding.startswith("terse")
    atom_nodes = sorted(n for n, d in graph.nodes(data=True) if d.get("kind") == "atom")

    rw = Chem.RWMol()
    index_map = {}
    for node in atom_nodes:
        fields = graph.nodes[node]["text"].split()
        atom = Chem.Atom(_symbol_from_word(fields[0]))
        if rich:
            _apply_rich_fields(atom, fields)
        index_map[node] = rw.AddAtom(atom)

    word_to_bond = {word: bond for bond, word in _BOND_WORDS.items()}
    aromatic_atoms = set()

    if encoding.endswith("_levi"):
        for node, attrs in graph.nodes(data=True):
            if attrs.get("kind") != "bond":
                continue
            endpoints = tuple(sorted(n for n in graph.successors(node)
                                     if graph.nodes[n].get("kind") == "atom"))
            if len(endpoints) != 2:
                raise ValueError(f"Levi bond node {node!r} has {len(endpoints)} atom "
                                 f"endpoints, expected 2")
            order = word_to_bond.get(attrs["text"].split()[0])
            if order is None:
                raise ValueError(f"unparseable bond text {attrs['text']!r}")
            rw.AddBond(index_map[endpoints[0]], index_map[endpoints[1]], order)
            if order is Chem.BondType.AROMATIC:
                aromatic_atoms.update(endpoints)
    else:
        seen = set()
        for u, v in graph.edges():
            if graph.nodes[u].get("kind") != "atom" or graph.nodes[v].get("kind") != "atom":
                continue
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            # atom_only carries no per-bond order; the topology-level check is the
            # strongest statement this encoding supports, and saying so is the point.
            rw.AddBond(index_map[u], index_map[v], Chem.BondType.SINGLE)

    # An aromatic *bond* implies aromatic endpoints. Without setting the flag RDKit
    # cannot perceive the ring and emits a kekulised, uppercase SMILES, which fails
    # the comparison for a reason that is about the decoder rather than the encoding.
    for node in aromatic_atoms:
        rw.GetAtomWithIdx(index_map[node]).SetIsAromatic(True)

    out = rw.GetMol()
    if sanitize:
        # Sanitization KEKULISES: where aromaticity cannot be perceived it rewrites
        # aromatic bonds as alternating single/double. That is a mutation of the bond
        # orders the encoding faithfully carried, so the labelled-graph comparison
        # must be made on the unsanitised molecule (see `roundtrip_check`).
        Chem.SanitizeMol(out, catchErrors=True)
    return out


def roundtrip_check(mol, encoding: str = "rich_levi", stereo_tags: bool = True):
    """Encode, decode, compare. Returns ``(ok, level, expected, got)``.

    The comparison is made at the level `ROUNDTRIP_LEVEL` declares for the encoding,
    so a *pass* is a statement about exactly what that encoding preserves — and the
    declared losses are asserted rather than hoped for.
    """
    level = ROUNDTRIP_LEVEL[encoding]
    graph = mol_to_graph(mol, encoding=encoding, stereo_tags=stereo_tags)
    rebuilt = graph_to_mol(graph, encoding, sanitize=(level != "labelled_graph"))

    if level == "exact":
        # Stereo is intentionally not reconstructed (the parity tag is in the text,
        # but rebuilding a CIP-consistent configuration needs the neighbour ordering
        # the graph discards), so compare the stereo-free forms. Everything else —
        # elements, bond orders, charges, hydrogens, isotopes, radicals — must match.
        expected = _flatten_stereo(Chem.MolToSmiles(mol, canonical=True))
        got = _flatten_stereo(Chem.MolToSmiles(rebuilt, canonical=True))

    elif level == "labelled_graph":
        expected = _labelled_graph_signature(mol)
        got = _labelled_graph_signature(rebuilt)

    else:  # topology
        expected = _degree_signature(mol)
        got = _degree_signature(rebuilt)

    return expected == got, level, expected, got


def _flatten_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def _labelled_graph_signature(mol) -> str:
    """Element-per-atom-index plus the set of ``(i, j, bond order)`` triples.

    Index-aligned rather than isomorphism-based: `graph_to_mol` adds atoms in sorted
    atom-node order, which is RDKit atom-index order, so atom *i* of the rebuild is
    atom *i* of the original. That makes this an exact comparison, and it is exactly
    what `terse` claims to preserve — no more (no hydrogens, charges, isotopes or
    aromatic perception) and no less.
    """
    atoms = "|".join(a.GetSymbol() for a in mol.GetAtoms())
    bonds = sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         str(b.GetBondType()))
        for b in mol.GetBonds())
    return atoms + "//" + ";".join(f"{i}-{j}:{t}" for i, j, t in bonds)


def _degree_signature(mol) -> str:
    """Element-and-degree multiset — what `atom_only` is expected to preserve."""
    return "|".join(sorted(f"{a.GetSymbol()}{a.GetDegree()}" for a in mol.GetAtoms()))


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold split
# ─────────────────────────────────────────────────────────────────────────────

def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles,
                                              includeChirality=include_chirality)


def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """Deterministic Bemis-Murcko scaffold split, DeepChem's ordering.

    Molecules are grouped by scaffold; groups are sorted largest-first (ties by
    first index) and poured into train, then valid, then test. No seed: the split is
    a function of the dataset, which is what makes it comparable across papers.

    Returns ``(train_idx, valid_idx, test_idx)``.
    """
    if abs(frac_train + frac_valid + frac_test - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")

    groups = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        try:
            key = murcko_scaffold(smiles)
        except Exception:
            key = ""  # unparseable scaffolds share one group, as DeepChem does
        groups[key].append(i)

    ordered = sorted(groups.values(), key=lambda idx: (len(idx), idx[0]), reverse=True)

    n = len(smiles_list)
    n_train, n_valid = frac_train * n, (frac_train + frac_valid) * n
    train, valid, test = [], [], []
    for group in ordered:
        if len(train) + len(group) > n_train:
            if len(train) + len(valid) + len(group) > n_valid:
                test += group
            else:
                valid += group
        else:
            train += group
    return train, valid, test


# ─────────────────────────────────────────────────────────────────────────────
# Tier B loading
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")


@dataclass(frozen=True)
class TierBSpec:
    filename: str
    smiles_col: str
    task_cols: tuple
    kind: str  # "classification" | "regression"


TIER_B = {
    "bace":    TierBSpec("bace.csv", "mol", ("Class",), "classification"),
    "bbbp":    TierBSpec("BBBP.csv", "smiles", ("p_np",), "classification"),
    "hiv":     TierBSpec("HIV.csv", "smiles", ("HIV_active",), "classification"),
    "tox21":   TierBSpec("tox21.csv.gz", "smiles", (), "classification"),
    "clintox": TierBSpec("clintox.csv.gz", "smiles",
                         ("FDA_APPROVED", "CT_TOX"), "classification"),
    "sider":   TierBSpec("sider.csv.gz", "smiles", (), "classification"),
    "esol":    TierBSpec("delaney-processed.csv", "smiles",
                         ("measured log solubility in mols per litre",), "regression"),
    "freesolv": TierBSpec("SAMPL.csv", "smiles", ("expt",), "regression"),
    "lipo":    TierBSpec("Lipophilicity.csv", "smiles", ("exp",), "regression"),
}

#: Held out from all molecule training, permanently. `bond_path` and `clintox`
#: were declared 2026-08-28 in PLAN.md §4.1, before any molecule run existed;
#: mirrored in `src/generalist/PLAN.md` §3.3. `longest_chain` joins them for the
#: generalist (`src/generalist/MOLECULE_GENERALIST.md` §4): two held-out Tier-A
#: families make the zero-shot readout a pair rather than a single point, and
#: `longest_chain` is the one whose specialist result is already recorded, so the
#: zero-shot number has a specialist number to sit against. Never add any of the
#: three to a training mixture, in any arm, including the specialist arm.
HELD_OUT_DATASETS = ("clintox",)
HELD_OUT_TIER_A_TASKS = ("bond_path", "longest_chain")


def load_tier_b(name: str, raw_dir: str = RAW_DIR):
    """Load one Tier-B dataset. Returns ``(records, spec, dropped)``.

    Each record is ``{"smiles": canonical, "mol": Mol, "targets": {col: value}}``.
    ``dropped`` is ``{"parse": n, "unsupported_bond": n}`` — molecules RDKit cannot
    parse, and molecules carrying a bond type with no faithful encoding
    (`is_encodable`). Both are first-class numbers rather than warnings: they bound
    every metric computed downstream.
    """
    import pandas as pd

    spec = TIER_B[name]
    df = pd.read_csv(os.path.join(raw_dir, spec.filename))

    task_cols = spec.task_cols or tuple(
        c for c in df.columns if c not in (spec.smiles_col, "mol_id"))

    missing = [c for c in (spec.smiles_col,) + tuple(task_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: columns {missing} absent; found {list(df.columns)[:12]}")

    records = []
    dropped = {"parse": 0, "unsupported_bond": 0}
    for row in df.to_dict("records"):
        mol = Chem.MolFromSmiles(row[spec.smiles_col])
        if mol is None:
            dropped["parse"] += 1
            continue
        # Heavy-atom graphs, as every molecular-ML baseline uses. A handful of Tier-B
        # SMILES carry an explicit `[H]` *atom* alongside the parent's own hydrogen
        # count (e.g. `[H]/[NH+]=C(\N)...` in clintox); encoding both double-counts
        # the hydrogen, and it was the single `rich_levi` round-trip failure at M0.
        # `RemoveAllHs`, not `RemoveHs`: the latter deliberately keeps an explicit H
        # that *defines* bond stereo, which is exactly the case here. The declared
        # cost is that double-bond stereo resting solely on such an H is lost — two
        # molecules in clintox, and we do not reconstruct stereo anyway.
        mol = Chem.RemoveAllHs(mol)
        if not is_encodable(mol)[0]:
            dropped["unsupported_bond"] += 1
            continue
        records.append({
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "mol": mol,
            "targets": {c: row[c] for c in task_cols},
        })
    return records, spec, dropped
