"""What the renderer may say about a molecule, and how it finds what a fact is about.

Every dictionary here is closed and hand written, because `render.py` is the only
place a claim enters an answer and these are the only words it has. Templates
fill `{anchor}` with the atom a fact names ("atom 14 (C)"), `{qualifier}` with
the functional group, `{a_qualifier}` with the group behind its article and
`{qualifiers}` with its plural.
"""

from __future__ import annotations

import re

from ..facts import clause


#: What a family *means*, in one clause, for the `explain` task. Closed and hand
#: written on purpose: a consequence a model supplies is a consequence nothing
#: computed, and this is the only place the set says anything general about
#: chemistry. Each entry has to be true of every molecule, not just the ones it
#: gets drawn for, because it is rendered verbatim.
GLOSS = {
    "ring_membership": "a ring atom is one that lies on a closed cycle of bonds",
    # Not "the shortest cycle it lies on": §4 holds `longest_chain` and
    # `bond_path` out of the *language* as well as the data, and
    # `mentions_held_out` matches `\bshortest\b`. This is the one rendered
    # sentence in the set that tripped its own filter, and the gloss is the
    # thing to change — a filter relaxed for the renderer stops being a filter.
    "ring_size": "the smallest ring containing an atom is the one with the "
                 "fewest atoms among the rings it belongs to",
    "ring_count": "the ring count is the number of smallest-set rings the "
                  "structure has",
    "aromatic_ring": "an aromatic ring is a cyclic system with delocalised pi "
                     "electrons",
    "fg_presence": "a functional group is a substructure that gives a molecule "
                   "a characteristic reactivity",
    "fg_count": "a functional group is a substructure that gives a molecule a "
                "characteristic reactivity",
    "fg_atom_membership": "an atom is part of a functional group when it is one "
                          "of the atoms the group's substructure covers",
    "stereo_potential": "a potential stereocentre is an atom whose substituents "
                        "could be arranged in more than one way",
    "stereo_assigned": "an assigned stereocentre is one whose arrangement the "
                       "structure actually specifies",
    "smiles": "SMILES is a line notation that writes a structure as a string",
    "caption": "a caption describes what the compound is and what it is used for",
}

#: Families that may be drawn for an `unanswerable` ask. Never a §4 held-out
#: family: teaching refusal on `bond_path` or `longest_chain` would train the
#: model out of precisely what §9.3 measures. These are in-scope families that
#: this particular molecule's sheet happens not to carry.
UNANSWERABLE_FAMILIES = (
    "ring_count", "ring_size", "aromatic_ring", "fg_count", "stereo_potential",
    "stereo_assigned", "caption",
)

#: Properties the sheet *never* carries, whatever the molecule, because nothing
#: computes them from a 2D graph without data from outside it. These are the
#: other half of the unanswerable pool and in practice the more useful half.
#:
#: The families above are only unanswerable when a particular sheet happens to
#: lack them, and in a measured smoke build that turned out to be `caption` 31
#: times out of 32 — every unanswerable turn was the sentence "Can you provide a
#: description of what this molecule is?", and the deduplicator then threw most
#: of them away. The twist had one cell in it.
#:
#: It is also the wrong refusal to teach. Describing a compound is something the
#: trunk *can* do, so refusing it trains a model to decline its own capability;
#: declining a melting point is the behaviour the fork is for.
OFF_SHEET_FAMILIES = (
    "melting_point", "boiling_point", "solubility", "pka", "logp",
    "iupac_name", "synthesis", "nmr", "ld50", "binding_affinity", "supplier",
)

#: How an unanswerable family is named in the reply, in the person's words rather
#: than the schema's.
FAMILY_WORDS = {
    "ring_count": "how many rings it has",
    "ring_size": "the size of the smallest ring around that atom",
    "aromatic_ring": "whether that ring is aromatic",
    "fg_count": "how many of that group it contains",
    "stereo_potential": "whether it has potential stereocentres",
    "stereo_assigned": "whether its stereocentres are assigned",
    "caption": "a description of what the compound is",
    "melting_point": "its melting point",
    "boiling_point": "its boiling point",
    "solubility": "how soluble it is in water",
    "pka": "its pKa",
    "logp": "its measured logP",
    "iupac_name": "its IUPAC name",
    "synthesis": "a route for making it",
    "nmr": "its NMR spectrum",
    "ld50": "its LD50",
    "binding_affinity": "what it binds and how tightly",
    "supplier": "where it can be bought",
}

#: The JSON type an unanswerable family's field would have carried, so that a
#: `fill_record` null sits in a typed schema next to the fields that do have
#: values. Anything absent is a string.
FIELD_KINDS = {
    "ring_count": "integer", "ring_size": "integer", "fg_count": "integer",
    "stereo_potential": "integer", "stereo_assigned": "integer",
    "aromatic_ring": "boolean",
    "melting_point": "number", "boiling_point": "number", "pka": "number",
    "logp": "number", "ld50": "number", "binding_affinity": "number",
}


#: What a fact is *asked about*, with its answer taken out. The writer of the
#: person's turn is given these and never the statements, so a question cannot
#: state its own answer — the defect that was the largest single rejection reason
#: across three builds is not caught here, it is unreachable.
ASK_PHRASES = {
    "ring_membership": "whether {anchor} is in a ring",
    "ring_size": "the size of the smallest ring containing {anchor}",
    "ring_count": "how many rings it has",
    "aromatic_ring": "whether {anchor} is in an aromatic ring",
    # The family covers two shapes. "2 of its rings are aromatic." carries no
    # atom, and falling back to `ring_count` there asked how many rings the
    # molecule has and answered how many of them are aromatic.
    "aromatic_ring/count": "how many of its rings are aromatic",
    # `{a_qualifier}` carries the article: `qualifier_of` strips it off the
    # sheet's wording, and "whether it contains nitrile" is not a sentence anyone
    # types. `{qualifiers}` is the plural, because "how many hydroxyl group
    # groups" is what you get from pasting "groups" onto a name that already
    # ends in one.
    "fg_presence": "whether it contains {a_qualifier}",
    "fg_count": "how many {qualifiers} it contains",
    "fg_atom_membership": "whether {anchor} is part of {a_qualifier}",
    # The tier_b fallback strips "is not" and nothing else, and this family's
    # negative wording hides its polarity in the verb: "It shows no activity
    # against HIV replication." went through untouched, so the ask carried the
    # answer and the false premise came out as "does not show no activity".
    "tier_b/hiv": "whether it shows activity against HIV replication",
    "stereo_potential": "how many atoms could be stereocentres",
    "stereo_assigned": "how many stereocentres are assigned",
    "smiles": "its canonical SMILES",
    "caption": "a description of what the compound is",
}

#: What a `decide` constraint calls the number it is about. "I need that number
#: to be at least 5" says which number only when the draw holds one, and a draw
#: of two counts is common: "How many ethers and how many hydroxyls does this
#: molecule contain? I need that number to be at least 5" names neither, while
#: the verdict behind it was computed on the first.
COUNT_SUBJECTS = {
    "ring_size": "the size of the smallest ring containing {anchor}",
    "ring_count": "the number of rings",
    "aromatic_ring/count": "the number of aromatic rings",
    "fg_count": "the number of {qualifiers}",
    "stereo_potential": "the number of atoms that could be stereocentres",
    "stereo_assigned": "the number of assigned stereocentres",
}

#: The sentence that points at the attached demonstrations. Several of them for
#: the reason every other axis here has several: §9.4 asks for no template
#: recognisable across the set, and a fixed preamble on a fifth of the examples
#: is the most recognisable template there could be.
SHOT_POINTERS = (
    "Worked examples on other molecules are attached — see those examples.",
    "Some solved examples for other molecules come with this one; see them.",
    "Attached are worked examples on different molecules. Use them as a guide.",
    "See the attached examples, which solve the same kind of question for other "
    "molecules.",
    "The attached examples answer questions like this one for other molecules.",
    "A few worked examples on unrelated molecules are attached for reference.",
)


_GROUP_RES = (
    re.compile(r"part of an? ([a-z][a-z ]*?)(?: group)?\.?$", re.IGNORECASE),
    re.compile(r"contains no ([a-z][a-z ]*?)\.?$", re.IGNORECASE),
    re.compile(r"contains \d+ ([a-z][a-z ]*?)\(s\)\.?$", re.IGNORECASE),
    re.compile(r"It contains an? ([a-z][a-z ]*?)\.?$", re.IGNORECASE),
)


def group_of(fact) -> str:
    """The functional group a fact is about, bare, or ""."""
    text = clause(fact)
    for pattern in _GROUP_RES:
        match = pattern.search(text)
        if match:
            # "hydroxyl group" and "hydroxyl" are the same group under two
            # spellings the sheet uses interchangeably; the bare name is the one
            # that takes an article and a plural cleanly.
            return re.sub(r"\s+group$", "", match.group(1).strip(),
                          flags=re.IGNORECASE)
    return ""


_ASSAY_RE = re.compile(r"\bin the ([A-Za-z0-9 _-]+?) assay", re.IGNORECASE)


def record_qualifier(fact) -> str:
    """What tells a fact's `fill_record` key apart where it names no atom.

    The family alone does not: a molecule has an `fg_count` per functional group
    and a `tier_b/tox21` per assay, so three assay facts all wanted the key
    `tier_b_tox21`, two of the three values were lost to the dict, and the record
    then failed verification for statements it had never had room for.
    """
    assay = _ASSAY_RE.search(fact.text or "")
    return assay.group(1) if assay else group_of(fact)


def ask_fallback(fact, text: str):
    """A Tier-B endpoint with no `ASK_PHRASES` entry, asked as a whether-clause.

    "It is not active in the SR ATAD5 assay." -> "whether it is active ...".
    """
    if not fact.family.startswith("tier_b/"):
        return None
    stripped = re.sub(r"\bis not\b", "is", text, count=1)
    stripped = re.sub(r"\b(shows?|has|have) no\b", r"\1", stripped, count=1)
    return f"whether {stripped[0].lower() + stripped[1:]}"
