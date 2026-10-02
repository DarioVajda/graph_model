"""Checks on the person's turn that only make sense for a molecule.

The accept pass runs `TURN_CHECKS` over every turn, after its own generic turn
checks and in the order listed. Each entry is ``(reason, check)``, and
``check(turn, intent, licensed)`` returns None to pass, or the detail string the
rejection log records.
"""

from __future__ import annotations

import re


#: One "smallest ring containing ..." clause naming two or more atoms. The comma
#: is the tell: the ask phrase for a single `ring_size` fact ends at its atom, so
#: a second atom inside the same clause can only have got there by the two being
#: merged.
_JOINT_RING_CLAUSE = re.compile(
    r"smallest ring containing atom [^?.]*?,\s*(?:and\s+)?atom", re.IGNORECASE)


def asks_for_a_joint_ring(turn: str, intent) -> bool:
    """Has the turn collapsed several `ring_size` asks into one impossible one?

    Two `ring_size` facts go into the ask as two phrases — "the size of the
    smallest ring containing atom 23 (O) and the size of the smallest ring
    containing atom 15 (C)" — and the writer, paraphrasing that into something a
    person would type, compresses the repetition away: "the size of the smallest
    ring containing atom 14 (C), atom 15 (N), and atom 10 (O)". The shorter
    sentence is the natural one and it asks a different question, about a single
    ring holding all three atoms. Nothing computed that ring, the statements are
    per-atom, and no reply built from them can answer it.

    This is a writer defect that only the question shows, so neither the
    statement checks nor the judge is positioned to see it — the reply is
    faithful to the statements, and it is the question that moved. Deciding it
    here costs nothing: the turn is already stored.

    The `ring_size` count is what separates the two spellings. A turn that names
    several atoms in one clause but rests on one fact is a person asking loosely
    about one atom, which the reply can still answer.
    """
    if not _JOINT_RING_CLAUSE.search(turn or ""):
        return False
    facts = intent.get("facts") or []
    return sum(1 for fact in facts if fact.get("family") == "ring_size") > 1


#: A token that could be a structure: long enough, in the SMILES character set,
#: and carrying at least one ring closure, branch or bond symbol. The parse below
#: is what decides; this only keeps RDKit off every word in the turn.
_STRUCTURE_TOKEN = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#%/\\]{6,}")


def invents_a_structure(turn: str, known: str = "") -> bool:
    """Does the person's turn quote a structure nobody gave it?

    The person and the assistant are looking at the same molecule and the writer
    is never shown its SMILES, so any structure in the turn was invented — and an
    invented structure is a question about a different molecule than the one the
    answer is about. Observed shapes: a plausible-looking SMILES presented as
    "the molecule represented by …", and pseudo-labels like `anchor[0, C]`.

    Narrow on purpose (§9.4 rule 3): a token only counts if RDKit parses it to
    four or more heavy atoms, which no ordinary English word does.

    `known` is the text the render itself licenses — its claim and its
    statements. A `check_claim` on a `smiles` fact puts the structure in the
    person's mouth *by design*, so the four such rows in the third smoke build
    were the renderer's own SMILES being refused as an invention.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    for token in _STRUCTURE_TOKEN.findall(turn or ""):
        if not re.search(r"[\d\(\)\[\]=#]", token):
            continue
        if token in (known or ""):
            continue
        mol = Chem.MolFromSmiles(token)
        if mol is not None and mol.GetNumHeavyAtoms() >= 4:
            return True
    return False


def _structure(turn, intent, licensed):
    return "" if invents_a_structure(turn, licensed) else None


def _joint_ring(turn, intent, licensed):
    return turn[:60] if asks_for_a_joint_ring(turn, intent) else None


TURN_CHECKS = (
    ("turn_invents_a_structure", _structure),
    ("turn_asks_for_a_joint_ring", _joint_ring),
)
