"""The molecule fact sheet: everything §9.4 allows a reply to say about one molecule.

`MOLECULE_GENERALIST.md` §9.4 puts one rule above the rest: **every example is
written from a fact sheet computed by RDKit.** A language model asked to count
rings is wrong often enough to teach confident hallucination; asked to *phrase* a
fact it already holds, it is reliable.

What the fact sheet may hold is fixed by §9.4 and enforced in `fact_sheet`:

* the nine Tier-A families' answers for this molecule — ring counts and sizes,
  aromaticity, functional groups with counts, atom-level memberships, stereo
  potential and assignment;
* its canonical stereo-free SMILES (§5);
* the in-mixture Tier-B labels it carries, with the endpoint named in words;
* its ChEBI-20 caption, when it has one.

And what it may never hold: the traversal families (`bond_path`, `longest_chain`)
and anything from ClinTox. Those are §4's held-out set, and a fact sheet that
quoted them would spend the held-out measurement through the back door.
`HELD_OUT_PATTERNS` is the second line of that defence — it refuses an *accepted*
example that talks about them however the wording arrived.
"""

from __future__ import annotations

import random
import re

from ..facts import Fact, article

#: Counts above this are not single-token answers in the Tier-A families, so the
#: trunk was never trained to produce one. Mirrors `molecules.tasks.MAX_COUNT`.
MAX_COUNT = 20

#: §4, as text. An accepted example that matches any of these is rejected
#: whatever its facts said: the held-out families are held out in *language* too,
#: or the set teaches the answer to the question §9.3 is measuring.
HELD_OUT_PATTERNS = (
    r"\blongest\b", r"\bshortest\b", r"\bpath\b", r"\bchain\b", r"\bdistance\b",
    r"\bhops?\b", r"\bbonds? (?:apart|away|between)\b",
    r"\bclinical trial", r"\bfda\b", r"\bapproved\b", r"\btoxic",
)

_HELD_OUT_RE = re.compile("|".join(HELD_OUT_PATTERNS), re.IGNORECASE)


def mentions_held_out(text: str) -> bool:
    """§4's language filter. True means the example is rejected."""
    return _HELD_OUT_RE.search(text or "") is not None


#: The endpoints of the five in-mixture Tier-B corpora, in words. ClinTox is
#: absent by construction, and so is anything else the trunk did not train on.
#: Each endpoint as a pair: the clause for a positive label and the clause for a
#: negative one. Two clauses rather than one and a rule, because English does not
#: negate a verb phrase by prefix — "does not inhibits", "does not active in the
#: assay" — and a fact sheet is the one place in this pipeline where the wording
#: is guaranteed correct.
ENDPOINT_WORDS = {
    ("bace", "Class"): ("inhibits human beta-secretase 1 (BACE-1)",
                        "does not inhibit human beta-secretase 1 (BACE-1)"),
    ("bbbp", "p_np"): ("crosses the blood-brain barrier",
                       "does not cross the blood-brain barrier"),
    ("hiv", "HIV_active"): ("shows activity against HIV replication",
                            "shows no activity against HIV replication"),
}


def endpoint_words(corpus: str, endpoint: str, negative: bool = False) -> str:
    """The endpoint as a clause a sentence can be built around.

    Tox21 and SIDER carry dozens of columns apiece and are gradient rather than
    headline numbers (§1), so their endpoints are named as the column they are
    instead of being paraphrased one by one into claims about biology that the
    label does not quite support. Tox21's columns *are* assays; SIDER's are
    MedDRA system-organ classes, and calling one an assay — "it is active in the
    Investigations assay" — states something that was never measured.
    """
    named = ENDPOINT_WORDS.get((corpus, endpoint))
    if named:
        return named[1] if negative else named[0]
    pretty = endpoint.replace("_", " ").replace("-", " ").strip()
    if corpus == "sider":
        return (f"has no reported side effects in the {pretty} class"
                if negative else
                f"has reported side effects in the {pretty} class")
    return (f"is not active in the {pretty} assay" if negative
            else f"is active in the {pretty} assay")


def canonical_smiles(mol) -> str:
    """§5's target form: canonical, and stereo-free.

    The same function the g2s task's target goes through, so an assistant answer
    that quotes a SMILES is quoting the string the trunk was trained to write.
    """
    from ...adapters.molecules import g2s_target

    return g2s_target(mol)


def fact_sheet(mol, *, rng: random.Random, tier_b=(), caption: str = "",
               max_atom_facts: int = 3) -> list:
    """Every fact §9.4 allows about one molecule, as `Fact` objects.

    The atom-level families name specific atoms, and which atoms they name is the
    one thing here that is sampled rather than enumerated: a molecule has as many
    ring-membership facts as it has atoms, and a fact sheet that listed all of
    them would be a table, not something to write three sentences from.
    Everything else — ring count, the stereo pair, the groups it contains and
    their counts — is exhaustive, because those are one fact each.

    Atom indices are **1-based** in the text, which is the convention the Tier-A
    questions use ("atom 14"), so a fact sheet and a trained question address the
    same atom by the same number.
    """
    from rdkit import Chem

    from ....experiments.molecules.tasks import (
        _SMARTS, _chiral_centers, FUNCTIONAL_GROUPS)

    facts = []
    info = mol.GetRingInfo()

    n_rings = info.NumRings()
    if n_rings <= MAX_COUNT:
        facts.append(Fact("ring_count", f"It has {n_rings} ring(s).",
                          n_rings, "count"))

    aromatic_rings = sum(1 for ring in info.AtomRings()
                         if all(mol.GetAtomWithIdx(i).GetIsAromatic()
                                for i in ring))
    if aromatic_rings <= MAX_COUNT:
        facts.append(Fact("aromatic_ring",
                          f"{aromatic_rings} of its rings "
                          f"{'is' if aromatic_rings == 1 else 'are'} aromatic.",
                          aromatic_rings, "count"))

    potential = len(_chiral_centers(mol, unassigned=True))
    assigned = len([c for c in _chiral_centers(mol, unassigned=True)
                    if c[1] != "?"])
    if potential <= MAX_COUNT:
        facts.append(Fact("stereo_potential",
                          f"{potential} atom(s) could be stereocenters.",
                          potential, "count"))
    if assigned <= MAX_COUNT:
        facts.append(Fact("stereo_assigned",
                          f"{assigned} stereocenter(s) have a defined "
                          "configuration.", assigned, "count"))

    # Functional groups: every group it contains, with its count, and then
    # `fg_presence` drawn from both sides.
    #
    # **Both sides is the whole point, and taking only the absent ones is how
    # this went wrong the first time.** The guard against "the answer to 'does
    # it contain X' is always yes" was written by emitting `fg_presence` for
    # absent groups only — which inverted the bias rather than removing it and
    # made the family's value a constant. Measured on the set that built:
    # 1,180 `fg_presence` facts, yes-rate **0.000**, and a model that answers
    # "no" to every functional-group question it is asked while scoring 0.994
    # on the same question in its own validator. A family whose value never
    # varies teaches its prior and nothing else, so draw up to two from each
    # side and let the molecule decide how many there are to draw.
    present, absent = [], []
    members_by_group = {}
    for name in FUNCTIONAL_GROUPS:
        matches = mol.GetSubstructMatches(_SMARTS[name])
        members_by_group[name] = {i for match in matches for i in match}
        (present if matches else absent).append((name, len(matches)))
    for name, count in present:
        if count <= MAX_COUNT:
            facts.append(Fact("fg_count", f"It contains {count} {name}(s).",
                              count, "count"))
    for name, _ in rng.sample(present, min(2, len(present))):
        facts.append(Fact("fg_presence",
                          f"It contains {article(name)} {name}.", "yes",
                          "yesno"))
    for name, _ in rng.sample(absent, min(2, len(absent))):
        facts.append(Fact("fg_presence", f"It contains no {name}.", "no",
                          "yesno"))

    # Atom-level facts, on a sample of atoms. Each atom carries the three
    # atom-level families at once, so a writer can say something about an atom
    # rather than one disconnected fact per atom.
    indices = [a.GetIdx() for a in mol.GetAtoms()]
    for idx in rng.sample(indices, min(max_atom_facts, len(indices))):
        atom = mol.GetAtomWithIdx(idx)
        label = f"atom {idx + 1} ({atom.GetSymbol()})"
        in_ring = atom.IsInRing()
        facts.append(Fact("ring_membership",
                          f"{label} is {'in' if in_ring else 'not in'} a ring.",
                          "yes" if in_ring else "no", "yesno", atoms=[idx + 1]))
        aromatic = atom.GetIsAromatic()
        facts.append(Fact("aromatic_ring",
                          f"{label} is {'in' if aromatic else 'not in'} an "
                          "aromatic ring.",
                          "yes" if aromatic else "no", "yesno", atoms=[idx + 1]))
        sizes = [len(r) for r in info.AtomRings() if idx in r]
        smallest = min(sizes) if sizes else 0
        if smallest <= MAX_COUNT:
            # The "0 means no ring" gloss belongs only on the fact it explains.
            # Carried on a nonzero size it invites the answer to repeat both
            # halves — "has 6 atoms. Since it is in no ring, the ring size is 0"
            # is a real one — and a self-contradiction still passes containment.
            text = (f"{label} is in no ring." if smallest == 0 else
                    f"The smallest ring containing {label} has "
                    f"{smallest} atoms.")
            facts.append(Fact("ring_size", text, smallest, "count",
                              atoms=[idx + 1]))
        # One group per atom is enough, but *which* group has to be drawn rather
        # than taken. Taking `present[0]` made the choice a function of
        # `FUNCTIONAL_GROUPS`' dict order: measured on the set that built, 53 %
        # of 3,889 atom-level group facts asked about hydroxyl or ether, and
        # nitrile and sulfonamide together came to 2.8 %. Drawing also fixes the
        # family's polarity, which the same line held at a yes-rate of 0.149 —
        # a random atom is rarely inside one particular group, so half the draws
        # come from the groups that do contain this atom when any do.
        if present:
            covering = [name for name, _ in present if idx in members_by_group[name]]
            pool = ([name for name in covering] if covering and rng.random() < 0.5
                    else [name for name, _ in present])
            name = rng.choice(pool)
            member = idx in members_by_group[name]
            facts.append(Fact("fg_atom_membership",
                              f"{label} is {'part' if member else 'not part'} "
                              f"of {article(name)} {name}.",
                              "yes" if member else "no", "yesno",
                              atoms=[idx + 1]))

    smiles = canonical_smiles(mol)
    if smiles:
        facts.append(Fact("smiles", f"Its canonical SMILES is {smiles}.",
                          smiles, "smiles"))

    for corpus, endpoint, label in tier_b:
        clause = endpoint_words(corpus, endpoint, negative=not label)
        facts.append(Fact(f"tier_b/{corpus}", f"It {clause}.",
                          "yes" if label else "no", "yesno"))

    if caption:
        facts.append(Fact("caption", caption.strip(), caption.strip(), "text"))

    assert not any(f.family in ("bond_path", "longest_chain") for f in facts)
    return facts
