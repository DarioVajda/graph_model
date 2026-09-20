"""The assistant set: fact sheets, style briefs, and the verifier (§9.4).

`MOLECULE_GENERALIST.md` §9.4 asks for a few thousand graph-conditioned questions
in free-form language with natural-language answers, and it puts one rule above
the rest: **every example is written from a fact sheet computed by RDKit.** A
language model asked to count rings is wrong often enough to teach confident
hallucination; asked to *phrase* a fact it already holds, it is reliable. So the
pipeline here is Python end to end and the writers only write:

    fact sheet  ->  style brief  ->  batch file  ->  (an agent writes)  ->  verify

This module is the Python half. It is deliberately import-light — RDKit and the
molecules package, nothing from the trainer — because two very different things
use it: `tools/intent_build.py`, which prepares batches on a login-node budget,
and `mol/assistant`'s ``verify``, which runs inside evaluation.

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

import json
import random
import re

# ─────────────────────────────────────────────────────────────────────────────
# Facts
# ─────────────────────────────────────────────────────────────────────────────

#: Counts above this are not single-token answers in the Tier-A families, so the
#: trunk was never trained to produce one. Mirrors `molecules.tasks.MAX_COUNT`.
MAX_COUNT = 20

#: Numbers spelled out. A fact's canonical form is the digit, and a writer that
#: says "three rings" has still stated the fact — the verifier accepts either,
#: and nothing else. Anything looser (any digit anywhere in the answer) would
#: pass an answer that got the fact wrong and happened to quote another number.
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)

#: Words that turn a statement into its negative. A `no` fact has to be stated as
#: one, and a `yes` fact has to not be.
NEGATIONS = ("no ", "not ", "n't", "lacks", "lacking", "without", "absent",
             "none", "free of", "neither", "nor ", "isn", "doesn")

#: §4, as text. An accepted example that matches any of these is rejected
#: whatever its facts said: the held-out families are held out in *language* too,
#: or the set teaches the answer to the question §9.3 is measuring.
HELD_OUT_PATTERNS = (
    r"\blongest\b", r"\bshortest\b", r"\bpath\b", r"\bchain\b", r"\bdistance\b",
    r"\bhops?\b", r"\bbonds? (?:apart|away|between)\b",
    r"\bclinical trial", r"\bfda\b", r"\bapproved\b", r"\btoxic",
)

_HELD_OUT_RE = re.compile("|".join(HELD_OUT_PATTERNS), re.IGNORECASE)

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


class Fact:
    """One statement about a molecule, and the string that proves it was used.

    ``value`` is the canonical form the verifier looks for in an answer; ``text``
    is the same fact in a sentence, which is what a writer is handed. ``family``
    is the Tier-A family, the corpus, or one of ``smiles`` / ``caption``, and it
    is what the correctness breakdown in §9.4's table is grouped by.
    """

    __slots__ = ("family", "text", "value", "kind", "atoms")

    def __init__(self, family: str, text: str, value: str, kind: str,
                 atoms=()):
        self.family = family
        self.text = text
        self.value = str(value)
        self.kind = kind                  # count | yesno | smiles | text
        self.atoms = tuple(atoms)

    def to_json(self) -> dict:
        out = {"family": self.family, "text": self.text, "value": self.value,
               "kind": self.kind}
        if self.atoms:
            out["atoms"] = list(self.atoms)
        return out

    @classmethod
    def from_json(cls, payload: dict) -> "Fact":
        return cls(payload["family"], payload["text"], payload["value"],
                   payload["kind"], payload.get("atoms", ()))

    def __repr__(self) -> str:
        return f"Fact({self.family}={self.value!r})"


def _article(word: str) -> str:
    """"a amide" and "a ether" are what a writer copies verbatim into the set,
    and both appear in v1. The group names are a closed list, so the vowel test
    is enough — none of them is a "european"."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def canonical_smiles(mol) -> str:
    """§5's target form: canonical, and stereo-free.

    The same function the g2s task's target goes through, so an assistant answer
    that quotes a SMILES is quoting the string the trunk was trained to write.
    """
    from .adapters.molecules import g2s_target

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

    from ..experiments.molecules.tasks import (
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

    # Functional groups: every group it contains, with its count, and two it does
    # not — a set written only from present groups teaches a model that the
    # answer to "does it contain X" is always yes.
    present, absent = [], []
    for name in FUNCTIONAL_GROUPS:
        matches = mol.GetSubstructMatches(_SMARTS[name])
        (present if matches else absent).append((name, len(matches)))
    for name, count in present:
        if count <= MAX_COUNT:
            facts.append(Fact("fg_count", f"It contains {count} {name}(s).",
                              count, "count"))
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
        for name, _count in present:
            members = {i for match in mol.GetSubstructMatches(_SMARTS[name])
                       for i in match}
            member = idx in members
            facts.append(Fact("fg_atom_membership",
                              f"{label} is {'part' if member else 'not part'} "
                              f"of {_article(name)} {name}.",
                              "yes" if member else "no", "yesno",
                              atoms=[idx + 1]))
            break                        # one group per atom is enough

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


# ─────────────────────────────────────────────────────────────────────────────
# Style briefs
# ─────────────────────────────────────────────────────────────────────────────

#: The axes a brief is drawn from. §9.4 asks for "as many phrasings, registers
#: and shapes as can be had — with no question template recognisable across the
#: set", and the way to get that from a model is to hand it one point of this
#: space per example rather than a general instruction to vary itself.
#:
#: There was a sixth axis, asking for a worked example about a *different*
#: molecule. It is gone: nothing computes facts for that other molecule, so the
#: verifier cannot check a word of it, and what came back was false — a smallest
#: ring of 4 atoms in 1,2-propanediol, which has no rings at all. An axis whose
#: content is unverifiable does not belong in a set whose whole claim is that
#: every fact in it was verified.
BRIEF_AXES = {
    "register": ("a chemist's shorthand", "a careful teacher", "a curious student",
                 "a terse colleague", "a patient explainer", "a lab notebook entry",
                 "a regulatory reviewer", "a textbook author"),
    "form": ("a direct question", "an instruction", "a scenario ending in a question",
             "a fill-in-the-blank request", "a request for a check of a claim",
             "a question with a stated reason for asking"),
    "length": ("under ten words", "one sentence", "two or three sentences"),
    "order": ("answer first, then the reason", "reason first, then the answer",
              "answer only"),
    "format": ("", "", "", "answer as JSON with the fact as the value",
               "answer in one word", "answer as a numbered list",
               "start the answer with the number"),
}

#: Forms that put the value in the question on purpose. Everywhere else a
#: question that already states its fact is a question with its answer in it,
#: and an example that teaches the model to read the answer out of the prompt
#: teaches it nothing about the molecule.
LEAKY_FORMS = ("a request for a check of a claim", "a fill-in-the-blank request")

#: Answers that talk about the fact *list* rather than the molecule. At
#: inference there is no list — there is a graph — so an answer reasoning from
#: "the facts provided" is teaching a move the model will never be able to make.
SHEET_TALK = (
    r"\bthe (?:given|provided|listed|above|supplied) (?:facts?|information|data)\b",
    r"\bthe (?:facts?|information|data) (?:given|provided|listed|supplied|above)\b",
    r"\baccording to the (?:facts?|list|information)\b",
    r"\bas (?:stated|listed|given|provided) above\b",
    r"\bfrom the list\b", r"\bthe fact sheet\b",
    r"\bfacts? \d+\b", r"\bthe (?:first|second|third) fact\b",
    r"\bstatements? \d+\b",
    # "as explicitly stated in the facts" and "based on the available data" are
    # the same move in clothes the patterns above do not fit.
    r"\b(?:as )?(?:explicitly |clearly )?"
    r"(?:stated|noted|mentioned|indicated|described) in the\b",
    r"\bin the facts?\b",
    r"\bthe facts? (?:states?|says?|shows?|indicates?|tells?)\b",
    r"\bbased on (?:the )?(?:provided|given|available|supplied|stated)\b",
    r"\b(?:provided|given|supplied) (?:above|here|data|information|results)\b",
)

_SHEET_TALK_RE = re.compile("|".join(SHEET_TALK), re.IGNORECASE)


def mentions_the_sheet(text: str) -> bool:
    """True when an answer reasons from the fact list instead of the molecule."""
    return _SHEET_TALK_RE.search(text or "") is not None

#: How many facts one example may use. One fact is the common case — a question
#: with one answer is what most of a case study looks like — and three is the
#: ceiling, because the verifier has to find every one of them in the answer.
FACT_COUNTS = (1, 1, 1, 2, 2, 3)


#: Formats whose answer is the fact's value and nothing else.
TERSE_FORMATS = ("answer in one word", "start the answer with the number")


def draw_brief(rng: random.Random) -> dict:
    """One point of `BRIEF_AXES`, plus how many facts the example uses.

    The axes are independent except where two of them contradict. An answer has
    to *contain* every fact it was written from, so "answer in one word" and
    "three facts" is a brief nothing can satisfy: the writer breaks the format or
    drops two facts, and the verifier is right to reject either. Drawing the
    combination at all just spends a generation pass to produce a rejection, so
    the terse formats take one fact.
    """
    brief = {axis: rng.choice(options) for axis, options in BRIEF_AXES.items()}
    brief["n_facts"] = rng.choice(FACT_COUNTS)
    if brief["format"] in TERSE_FORMATS:
        brief["n_facts"] = 1
    return brief


def brief_text(brief: dict) -> str:
    """The brief as the line a writer is given."""
    parts = [f"Voice: {brief['register']}.", f"Shape: {brief['form']}.",
             f"Question length: {brief['length']}.", f"Answer: {brief['order']}."]
    if brief.get("format"):
        parts.append(f"Format: {brief['format']}.")
    return " ".join(parts)


def brief_key(brief: dict) -> str:
    """The identity of a brief, for the per-brief ceiling in `accept`."""
    return "|".join(str(brief[axis]) for axis in sorted(BRIEF_AXES))


def select_facts(facts, brief: dict, rng: random.Random) -> list:
    """The facts one example is written from.

    A caption is never combined with another fact: it is a paragraph, and an
    example that has to contain both it and a ring count is an example whose
    answer is a list. `smiles` is treated the same way for the same reason.

    The terse formats narrow it further. A one-word answer cannot be a caption or
    a SMILES, and an answer that starts with the number needs a fact that *has* a
    number — a yes/no fact under that format is the same unsatisfiable brief as
    one word and three facts.

    **A multi-fact draw is coherent or it is not drawn.** Sampling freely across
    the sheet was the original rule and it produced examples like "How many
    ethers are present?" written from an ether count *and* two unrelated
    aromatic-ring facts, which came back as "There are 2 ethers. Atom 20 (O) is
    not in an aromatic ring, and atom 10 (C) is also not in an aromatic ring.
    Therefore, there are 2 ethers." Every fact is true and contained, and the
    "therefore" is a derivation nothing computed. Asked for several unrelated
    facts in one answer, a writer will either weld them with a false connective
    or list them without saying they are unrelated, and neither is a sentence I
    want the model to learn.

    So the other facts are drawn from the pivot's own neighbourhood — same
    family, or same atom — which is what "several facts about this molecule"
    should mean. Where the sheet has no such neighbours the example is written
    from the pivot alone, which is why this shortens some draws rather than
    failing them.
    """
    n = int(brief.get("n_facts", 1))
    terse = brief.get("format") in TERSE_FORMATS
    long_form = [f for f in facts if f.kind in ("text", "smiles")]
    short = [f for f in facts if f.kind not in ("text", "smiles")]
    if brief.get("format") == "start the answer with the number":
        counts = [f for f in short if f.kind == "count"]
        if counts:
            return [rng.choice(counts)]
    if long_form and not terse and rng.random() < 0.15:
        return [rng.choice(long_form)]
    if not short:
        return [rng.choice(facts)]
    pivot = rng.choice(short)
    if n <= 1:
        return [pivot]
    kin = [f for f in short
           if f is not pivot and (f.family == pivot.family
                                  or (f.atoms and set(f.atoms) & set(pivot.atoms)))]
    return [pivot] + rng.sample(kin, min(n - 1, len(kin)))


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot demonstrations
# ─────────────────────────────────────────────────────────────────────────────
#
# The axis §9.4 withdrew, reinstated on the one condition that withdrawal named.
# It was dropped because "a worked example is about a *different* molecule,
# nothing computes a fact sheet for that molecule, and so no part of the
# demonstration can be verified" — and what came back proved the point, a 4-atom
# smallest ring in a molecule with no rings.
#
# That objection is about where the demonstration comes from, not about
# few-shot. A demonstration drawn from an **already-accepted row** has a fact
# sheet, computed by RDKit for a real molecule from the same pool, and it has
# already been through every filter in `intent_accept.py`. So the shots are
# composed after acceptance, out of the set itself, and every word of every
# demonstration is verified to the same standard as the answer it precedes.
#
# The demonstration molecules go into the graph as their own **disconnected
# components** — which is also the only honest encoding, since they are
# different molecules and a bond between them would be a chemical claim. The
# prompt node carries a directed edge to each demonstration node, so the
# pointer the question makes in words ("worked examples are attached") is a
# real edge the structural bias can read. That puts the target's atoms at
# distance 1 from the prompt and a demonstration's atoms at distance 2, behind
# their own node, so "which molecule is the question about" is answerable from
# the SPD row rather than only from the text.

#: The share of accepted examples that get demonstrations.
SHOT_FRACTION = 0.22

#: How many demonstrations one example carries, when it carries any. Weighted
#: low: the point is to teach the shape of an answer, and a fourth demonstration
#: costs four more molecules' worth of context for very little of that.
SHOT_COUNTS = (1, 1, 1, 2, 2, 3, 4)

#: How many times one accepted row may serve as somebody else's demonstration.
#: Without a ceiling the selector's preference for a matching family and format
#: concentrates on whichever rows are easiest to match, and a handful of answers
#: would be shown to the model hundreds of times.
SHOT_REUSE_CEILING = 6

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


def draw_shot_count(rng: random.Random, fraction: float = None) -> int:
    """How many demonstrations this example gets. 0 for most of them."""
    share = SHOT_FRACTION if fraction is None else fraction
    if rng.random() >= share:
        return 0
    return rng.choice(SHOT_COUNTS)


def shot_pointer(rng: random.Random) -> str:
    """The line prefixed to a question whose example carries demonstrations."""
    return rng.choice(SHOT_POINTERS)


def shot_text(question: str, answer: str, index: int) -> str:
    """One demonstration, as the text of its node in the graph.

    Numbered, because the demonstrations are a *set* of components and the
    numbering is the only thing that lets an answer refer to one of them. The
    Q/A shape is deliberately not the prompt format the example itself uses:
    a demonstration is quoted material, and formatting it as a second turn
    would give the graph two assistant turns and no way to tell which one is
    being asked for.
    """
    return (f"Example {index + 1} (a different molecule)\n"
            f"Q: {question}\nA: {answer}")


def demo_leaks_target(demo_answer: str, target_facts) -> bool:
    """Does a demonstration's answer already state one of the target's facts?

    The copy shortcut, and the one way a demonstration can make an example
    easier than the example it demonstrates: a demonstration answering "3" to a
    ring-count question, in front of a target whose ring count is also 3,
    teaches that the answer is whatever the last example answered. The molecules
    differ and the demonstration is true of its own, so nothing here is *wrong*
    — it is just a row the model can get right without reading the graph, which
    is the same defect as a question that states its own answer.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in target_facts]
    if not facts:
        return False
    return any(facts_contained(demo_answer, facts).values())


def shot_candidates(target, pool) -> list:
    """`pool` rows that may demonstrate for `target`, best match first.

    Three hard rules and one preference. A demonstration may not be the target,
    may not be the same molecule under a different id (the partition key is the
    identity that matters — two rows about one molecule would put the target's
    own structure in its context twice), and may not state one of the target's
    facts. The preference is for a row that shares the target's *format* first
    and a fact *family* second, because a demonstration exists to show the shape
    of the wanted answer, and a JSON demonstration in front of a one-word
    question shows the wrong shape.
    """
    target_families = {f["family"] for f in target["facts"]}
    target_format = (target.get("brief") or {}).get("format", "")
    out = []
    for row in pool:
        if row["id"] == target["id"] or row["key"] == target["key"]:
            continue
        if demo_leaks_target(row["answer"], target["facts"]):
            continue
        families = {f["family"] for f in row["facts"]}
        row_format = (row.get("brief") or {}).get("format", "")
        score = (2 if row_format == target_format else 0) + \
                (1 if families & target_families else 0)
        out.append((score, row))
    out.sort(key=lambda pair: -pair[0])
    return [row for _score, row in out]


def fact_polarity(facts) -> str:
    """``"yes"``, ``"no"``, or ``""`` when a row's yes/no facts do not agree.

    The handle the polarity balance in `assistant_compose.py` is drawn on. It
    reads the *facts*, not the answer: the facts carry the canonical value
    already, and asking the text would mean re-deriving through the negation
    machinery something that was computed by RDKit two stages earlier.
    """
    values = {_fact_field(f, "value") for f in facts
              if _fact_field(f, "kind") == "yesno"}
    return values.pop() if len(values) == 1 else ""


def question_text(row) -> str:
    """The text of a composed row's question node: the pointer, then the exchange.

    The pointer is stored beside the question rather than folded into it so that
    ``question`` stays exactly the string the accept pass verified. A composed
    set therefore re-verifies as the accepted set it came from, and the sentence
    that refers to the attached components is recoverable as its own field. The
    same holds for the turns below: they are assembled here, at the consumer, and
    nothing on the row is rewritten.

    **A `needs_clarification` row is an exchange, and the exchange is the
    question.** Its `question` field holds only the person's opening turn, which
    is underspecified on purpose — "Is that atom part of a ring?" — and the
    clarifying turns that resolve it live in `turns`. Reading `question` alone
    asks the model something that has no answer while showing it one that names
    an atom the question never did, which teaches exactly the guess the twist
    exists to teach against. The middle turns are everything between the opening
    and the reply, so a single-turn row is unchanged and needs no transcript
    framing around it.
    """
    pointer = (row.get("pointer") or "").strip()
    question = row["question"]
    middle = (row.get("turns") or [])[1:-1]
    if middle:
        lines = [f"You: {question}"]
        lines += [f"{'Assistant' if t['role'] == 'assistant' else 'You'}: "
                  f"{t['text']}" for t in middle]
        question = "\n".join(lines)
    return f"{pointer}\n{question}" if pointer else question


def named_atoms_for(facts) -> list:
    """The atoms an assistant example is about, as **0-based RDKit indices**.

    Empty unless *every* drawn fact is atom-scoped. This mirrors
    `dataset.build_graph_example`: an atom-level question wires the prompt to
    the atoms it names, and a molecule-level one wires it to the whole
    molecule, because a prompt whose edges reach only one atom of a question
    about the whole molecule is pointing at the wrong thing. An assistant
    example can draw both kinds at once, and then the molecule is what it is
    about.

    **The indices are converted.** A fact sheet writes atoms 1-based, because
    that is the convention the Tier-A questions use ("atom 14"); the graph's
    node keys are RDKit's own 0-based indices. Wiring the prompt without
    subtracting one would point it at the neighbouring atom — silently, and
    only on the families that name an atom.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    if not facts or not all(f.atoms for f in facts):
        return []
    return sorted({a - 1 for f in facts for a in f.atoms})


def shot_molecules(row) -> list:
    """A composed row's demonstrations as ``(mol, question, answer)``.

    The molecule is rebuilt from the demonstration's partition key, which is the
    stereo-free canonical SMILES (`adapters.molecules.partition_key`) — the same
    string the §3 partition treats as the molecule's identity, so a demonstration
    cannot smuggle in a stereoisomer the partition holds at a different role.
    """
    from rdkit import Chem

    out = []
    for shot in row.get("shots") or []:
        mol = Chem.MolFromSmiles(shot["key"])
        if mol is None:                  # unparseable key: drop the shot, keep the row
            continue
        out.append((mol, shot["question"], shot["answer"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def _contains_number(answer: str, value: str) -> bool:
    n = int(value)
    if re.search(rf"(?<![\d.]){n}(?![\d.])", answer):
        return True
    if 0 <= n < len(NUMBER_WORDS):
        return re.search(rf"\b{NUMBER_WORDS[n]}\b", answer, re.IGNORECASE) is not None
    return False


#: Saying a ring size of 0 without writing a 0 — which is what the fact sheet
#: itself does.
_NO_RING = re.compile(r"\b(?:in no rings?"
                      r"|not (?:in|part of|a member of) (?:an?|any|the )?\s*rings?"
                      r"|belongs to no rings?"
                      # "Neither atom 13 (C) nor atom 14 (C) is in any ring."
                      r"|(?:neither|nor)\b[^.]*?\bin any rings?"
                      r"|no rings?\b)", re.IGNORECASE)


def _contains_count(answer: str, fact) -> bool:
    """A count is stated by its digit or its number word — or, for the one count
    whose own sentence has no digit in it, by that sentence.

    `ring_size` is 0 when the atom is in no ring, and what a writer is handed for
    it is "atom 1 (C) is in no ring." — the digit appears nowhere. 111 rows of
    the v3 write said exactly that and were refused for repeating the wording
    they were given. Every other zero count carries its digit ("It has 0
    ring(s)."), so the exemption stays on the family that needs it.
    """
    if _contains_number(answer, fact.value):
        return True
    return (_fact_field(fact, "family") == "ring_size"
            and str(_fact_field(fact, "value")) == "0"
            and _NO_RING.search(answer or "") is not None)


def _sentences(answer: str):
    return [s for s in re.split(r"(?<=[.!?;])\s+|\n+", answer) if s.strip()]


#: Openers that settle polarity on their own. "Yes" and "No" are not the whole
#: of it: under "answer in one word", and on the assay questions, what a writer
#: actually produces is "None", "Absent", "Inactive" — 21 rows of the v3 write
#: were refused for a gap in this vocabulary rather than for a mistake.
_YES_HEAD = re.compile(r"(yes\b|true\b|correct\b|it does\b|indeed\b|present\b|"
                       r"affirmative\b|active\b)")
_NO_HEAD = re.compile(r"(no\b|false\b|incorrect\b|it does not\b|it doesn't\b|"
                      r"none\b|absent\b|inactive\b|negative\b|nil\b|"
                      r"not present\b)")


def _json_values(answer: str):
    """The scalar values of a structured answer, or None if it is not one.

    The format axis asks for "answer as JSON with the fact as the value", so a
    writer that obeys it produces `{"atom_in_ring": true}`. The key there is the
    question restated and the value is the whole answer, and matching a prose
    subject against it can only ever fail: 163 rows of the v3 write were refused
    for writing the format they were asked for.
    """
    text = (answer or "").strip()
    if not text.startswith(("{", "[")):
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None

    def walk(node):
        if isinstance(node, dict):
            return [v for child in node.values() for v in walk(child)]
        if isinstance(node, list):
            return [v for child in node for v in walk(child)]
        return [node]

    return ["yes" if v is True else "no" if v is False else str(v).lower()
            for v in walk(payload)]


def _contains_yesno(answer: str, fact: Fact, siblings: int = 1) -> bool:
    """A yes/no fact is stated when the answer says so, or says the thing itself.

    "Yes" is not required: "atom 14 sits in a pyrimidine ring" states the fact,
    and demanding the word would flatten every register the briefs ask for into
    one. What is required is that the polarity is unambiguous — the subject named
    and the *clause* carrying this fact's own predicate negated iff the fact is
    `no`.

    The clause, and not the sentence, and this fact's predicate and not any
    clause about the subject. Polarity used to be read as "no negation anywhere
    in a sentence mentioning the atom", which is only right when an atom gets one
    fact and one sentence. Draw two facts of opposite polarity about one atom —
    which the sampler does constantly — and the writer answers them in one
    sentence, "Atom 24 (C) is in a ring, the smallest ring containing it has 6
    atoms, and it is not in an aromatic ring.", and the single `not` was taken to
    negate all three. That sentence is correct and complete, and it was the
    single largest source of `facts_missing`: of 18 such rejections read by hand
    across the writer A/B, 17 stated every fact they were given.

    A structured answer is read as its values, since that is where its content
    is; a value that is itself a sentence then falls through to the prose rules.
    `siblings` is how many facts in the example share this polarity, and a
    structured answer has to supply a value apiece: the brief asks for JSON "with
    the fact as the value", so one `false` beside one key states one fact, and
    `{"nr_er_lbd": false}` drawn from two assay facts has left one of them out.
    Prose is not held to the count — a bare "No." answering two negatives is the
    register the briefs ask for, and that latitude is older than this rule.
    """
    want_yes = fact.value == "yes"
    head_re = _YES_HEAD if want_yes else _NO_HEAD
    values = _json_values(answer)
    if values is not None:
        if sum(1 for v in values if head_re.match(v.strip())) >= siblings:
            return True
        lowered = ". ".join(values)
    else:
        lowered = (answer or "").lower()
        if head_re.match(lowered.strip()[:24]):
            return True
    subject = _subject_of(fact)
    if not subject:
        return False
    hits = _subject_clauses(lowered, subject)
    if not hits:
        return False
    predicate = _predicate_tokens(fact, subject)
    scored = [(len(predicate & set(re.findall(r"[a-z0-9]+", c))), c) for c in hits]
    # A fact whose wording is all common words has nothing to match on; it keeps
    # the older, looser read of "any clause about the subject".
    if predicate and not any(score for score, _ in scored):
        return False
    best = max(score for score, _ in scored)
    return any(_negated(clause) != want_yes
               for score, clause in scored if score == best)


#: Split an answer into the spans a polarity can attach to. Sentence boundaries
#: are not enough — "Atom 24 (C) is in a ring, ... and it is not in an aromatic
#: ring." is one sentence carrying two opposite polarities.
_ANSWER_CLAUSE_RE = re.compile(
    r"(?<=[.!?;:])\s+|\n+|,\s+|\s+(?:and|but|whereas|while|however|though|"
    r"although|yet)\s+", re.IGNORECASE)

#: Dropped before matching a clause against a fact's own wording. The negation is
#: the thing being measured, so letting it score the match would be circular: a
#: `no` fact would prefer a negated clause by construction and then find it
#: negated.
_POLARITY_BLIND = frozenset(("no", "not", "nor", "neither", "without", "lacks",
                             "lacking", "absent", "none", "isn", "doesn", "t"))

#: An atom reference, with the element it is usually written with.
_ATOM_REF_RE = re.compile(r"atom \d+(?:\s*\([a-z]{1,2}\))?", re.IGNORECASE)


def _answer_clauses(lowered: str) -> list:
    """The answer, split into the spans a polarity can attach to.

    "and" joins two predicates about as often as it joins two subjects, and
    splitting a compound subject leaves a fragment with nothing predicated of it:
    "Atom 5 (O) and atom 21 (C) are in a ring." would say nothing about atom 5. A
    span with no word of its own beyond its atom reference is therefore glued
    back onto the one that follows it.
    """
    parts = [c.strip() for c in _ANSWER_CLAUSE_RE.split(lowered) if c and c.strip()]
    out, carry = [], ""
    for part in parts:
        rest = _ATOM_REF_RE.sub(" ", part)
        if not [w for w in re.findall(r"[a-z]+", rest) if len(w) >= 3]:
            carry = f"{carry} {part}".strip()
            continue
        out.append(f"{carry} {part}".strip() if carry else part)
        carry = ""
    if carry:
        out.append(carry)
    return out


def _subject_clauses(lowered: str, subject: str) -> list:
    """The clauses that are *about* `subject`.

    An atom keeps the floor once it has it: prose names the atom and then says
    "it", so a clause with no atom reference of its own belongs to the last one
    named. Without that, "Atom 24 (C) is in a ring. It is not in an aromatic
    ring." loses the second fact for lack of a subject and the first one for the
    negation in a sentence it does not own. A clause that names several atoms is
    about all of them, and is not the property of the last one named.
    """
    clauses = _answer_clauses(lowered)
    if not _ATOM_REF_RE.fullmatch(subject.strip()):
        return [c for c in clauses if subject in c]
    number = re.search(r"\d+", subject).group(0)
    out, current = [], None
    for clause in clauses:
        refs = [re.search(r"\d+", r).group(0)
                for r in _ATOM_REF_RE.findall(clause)]
        if refs:
            current = refs[-1]
        if number in refs or (not refs and current == number):
            out.append(clause)
    return out


def _predicate_tokens(fact, subject: str) -> set:
    """What the fact says *about* its subject: its own wording, less the subject,
    less its polarity, and less the words every fact shares.

    What is left has to carry the meaning, because a clause is credited with
    stating the fact when it matches one of these. Scoring on the common words
    instead would credit "Atom 14 (C) is a carbon." with "atom 14 (C) is in a
    ring." on the strength of "is" and "a".
    """
    text = _ATOM_REF_RE.sub(" ", (_fact_field(fact, "text") or "").lower())
    if subject:
        text = text.replace(subject, " ")
    words = set(re.findall(r"[a-z0-9]+", text)) - _POLARITY_BLIND
    return {w for w in words if len(w) >= 4 and w not in _PREDICATE_STOP}


def _negated(clause: str) -> bool:
    return any(word in clause for word in NEGATIONS)


def _subject_of(fact) -> str:
    """What a yes/no fact is *about*, lowercased: the group, or the atom.

    Taken from the fact's own text rather than re-derived, so the two cannot
    drift apart: the text is what the writer was given.
    """
    text = _fact_field(fact, "text").lower()
    match = re.search(r"part of an? ([a-z ]+?)\.", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"contains no ([a-z ]+?)\.", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"^atom (\d+)", text)
    if match:
        return f"atom {match.group(1)}"
    match = re.search(r"it (?:does not )?(.+?)\.", text)
    if match:
        return match.group(1).strip()[:40]
    return ""


def _contains_smiles(answer: str, value: str) -> bool:
    """A quoted SMILES must parse and canonicalize to the molecule's own.

    §9.4's rule, and it is stricter than string equality in the direction that
    matters: a writer may quote a valid alternative spelling of the same molecule
    and be right, and may quote a string that looks similar and be wrong.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    if value in answer:
        return True
    for candidate in re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\.]{6,}", answer):
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            continue
        if canonical_smiles(mol) == value:
            return True
    return False


def _names_its_atom(answer: str, fact) -> bool:
    """Whether the answer names the atom an atom-scoped fact is about."""
    atoms = _fact_field(fact, "atoms") or []
    return all(re.search(rf"\b{int(a)}\b", answer or "") for a in atoms)


def _ambiguous(facts) -> set:
    """Indices of atom-scoped facts another fact in the same example could be
    mistaken for: same family, same value, different atoms.

    One sentence otherwise states both. "Atom 2 is in no ring" was counted as
    also stating the smallest ring containing atom 22, because the number the
    check looks for is "0" either way. Only these facts have to name their atom
    — demanding it of every fact would fail "Yes", which is a whole register the
    briefs ask for and a format ("answer in one word") that permits nothing else.

    Family and not `kind`, because `kind` is one of four coarse buckets: grouping
    by it made "atom 17 is in a ring" and "atom 17 is in an aromatic ring"
    confusable, both being `yesno`/`yes`, and so made each name an atom the other
    already fixes — refusing "Yes. It is in an aromatic ring." for stating both.
    Atoms have to differ too. Two facts about one atom are not confusable at all,
    whatever their family; one sentence about that atom states both.
    """
    seen = {}
    for i, fact in enumerate(facts):
        atoms = tuple(_fact_field(fact, "atoms") or ())
        if not atoms:
            continue
        seen.setdefault((_fact_field(fact, "family"),
                         str(_fact_field(fact, "value"))), []).append((i, atoms))
    return {i for group in seen.values() if len({a for _, a in group}) > 1
            for i, _ in group}


def facts_contained(answer: str, facts) -> dict:
    """``{fact index: whether the answer states it}`` — §9.4's correctness read."""
    out = {}
    ambiguous = _ambiguous(facts)
    polarities = {}
    for fact in facts:
        if fact.kind == "yesno":
            polarities[fact.value] = polarities.get(fact.value, 0) + 1
    for i, fact in enumerate(facts):
        if i in ambiguous and not _names_its_atom(answer, fact):
            out[i] = False
            continue
        if fact.kind == "count":
            out[i] = _contains_count(answer, fact)
        elif fact.kind == "yesno":
            out[i] = _contains_yesno(answer, fact, polarities[fact.value])
        elif fact.kind == "smiles":
            out[i] = _contains_smiles(answer, fact.value)
        else:
            # A caption is prose; requiring it verbatim would reject every
            # paraphrase, which is the whole point of the set. Require that the
            # answer is not shorter than a clause and shares vocabulary with it.
            out[i] = _overlaps(answer, fact.value)
    return out


def _overlaps(answer: str, reference: str, floor: float = 0.25) -> bool:
    a = set(re.findall(r"[a-z0-9]+", answer.lower()))
    b = set(re.findall(r"[a-z0-9]+", reference.lower()))
    if not b:
        return False
    return len(a & b) / len(b) >= floor


def format_met(answer: str, brief: dict, facts=()) -> bool:
    """The brief's format instruction, checked. No instruction, nothing to check.

    ``facts`` is needed for one rule only: "start the answer with the number"
    means the fact's number, and a leading "1." from a numbered list satisfies a
    bare leading-digit test while answering with a list marker.
    """
    instruction = (brief or {}).get("format") or ""
    if not instruction:
        return True
    if "JSON" in instruction:
        try:
            json.loads(_json_span(answer))
        except (ValueError, TypeError):
            return False
        return True
    if "one word" in instruction:
        return len(answer.strip().strip(".").split()) == 1
    if "numbered list" in instruction:
        # A list written "1. … 2. …" on one line is a numbered list. Anchoring
        # the items to line starts only tests whether the writer used newlines.
        return len(re.findall(r"(?:^|\s)\d+[.)]\s+", answer)) >= 2
    if "start the answer with the number" in instruction:
        leading = re.match(r"^\W*(\d+)", answer.strip())
        if leading is None:
            return False
        counts = [f.value for f in facts if getattr(f, "kind", "") == "count"]
        return not counts or int(leading.group(1)) == int(counts[0])
    return True


def _json_span(answer: str) -> str:
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        start, end = answer.find("["), answer.rfind("]")
    return answer[start:end + 1] if start >= 0 and end > start else answer


def mentions_held_out(text: str) -> bool:
    """§4's language filter. True means the example is rejected."""
    return _HELD_OUT_RE.search(text or "") is not None


#: Chemistry a fact sheet never states, and therefore a writer never has grounds
#: for. Named ring systems and compound classes are the first list; group names
#: outside `FUNCTIONAL_GROUPS` are the second. Both are invention by
#: construction — the sheet is exhaustive over rings, aromaticity, stereocentres
#: and the ten groups, so anything here came from the writer's own chemistry
#: rather than from the molecule in front of it.
NEVER_ON_A_SHEET = (
    "benzene", "phenyl", "naphthalene", "anthracene", "pyridine", "pyrimidine",
    "pyrrole", "imidazole", "indole", "furan", "thiophene", "piperidine",
    "piperazine", "morpholine", "quinoline", "purine", "steroid", "peptide",
    "alkaloid", "sugar", "nucleoside",
    "ester", "alcohol", "aldehyde", "thiol", "sulfide", "disulfide",
    "carbamate", "carbamoyl", "urea", "imine", "azo", "thiocarbonyl",
    "phosphate", "sulfate", "epoxide", "anhydride", "acetyl", "methoxy",
)

_NEVER_RE = re.compile(
    "|".join(rf"\b{re.escape(word)}s?\b" for word in NEVER_ON_A_SHEET),
    re.IGNORECASE)


def unsupported_claims(answer: str, sheet) -> list:
    """Chemistry the answer asserts that its molecule's fact sheet does not.

    Containment asks whether the facts the example was written from are *in* the
    answer. This asks the other half, and it is the half a writer actually fails:
    handed one SMILES fact, a 12B model will describe two benzene rings and a
    hydroxy group it inferred, and every one of those claims lands in the
    training set as if it had been computed.

    It cannot prove an answer adds nothing — prose has too many ways to assert.
    What it does cover is the vocabulary: a group name the sheet never mentions,
    and any named ring system or compound class, neither of which a sheet can
    ever license. Returns the offending words, so the rejection log says which.
    """
    from ..experiments.molecules.tasks import FUNCTIONAL_GROUPS

    text = (answer or "").lower()
    sheet_text = " ".join(
        (f.text if isinstance(f, Fact) else f.get("text", "")) for f in sheet
    ).lower()

    found = []
    for name in FUNCTIONAL_GROUPS:
        if re.search(rf"\b{re.escape(name)}s?\b", text) and name not in sheet_text:
            found.append(name)
    for match in _NEVER_RE.finditer(text):
        word = match.group(0).lower().rstrip("s")
        if word not in sheet_text and word not in found:
            found.append(word)
    return found


#: A claim about chemistry in general, which a sheet about one molecule cannot
#: license however true it happens to be. "1, because there is 1 ether and all
#: ethers contain exactly 2 oxygen atoms" is the shape: the number is right, the
#: generalisation is invented, and containment sees only the number.
_UNIVERSAL_RE = re.compile(
    r"\b(?:all|every|any)\s+[a-z][a-z-]*s?\s+"
    r"(?:contain|contains|have|has|are|is|consist|consists|include|includes)\b",
    re.IGNORECASE)

#: An atom's index is not a property of the atom. A writer that reads "atom 17"
#: as atomic number 17 produces a claim that is sometimes even right — atom 17 of
#: that molecule was a chlorine — and always unfounded.
_ATOM_PROPERTY_RE = re.compile(r"\batomic (?:number|weight|mass)\b",
                               re.IGNORECASE)


def _fact_field(fact, name: str, default=""):
    """Facts reach the verifier as dataclasses when written and as dicts when
    read back from JSONL. Both are read the same way here."""
    if isinstance(fact, Fact):
        return getattr(fact, name, default)
    return fact.get(name, default)


def ungrounded_claims(answer: str, facts) -> list:
    """Claims whose *scope* outruns the fact that supports them.

    `unsupported_claims` polices vocabulary — a group the sheet never names.
    This polices the three moves that stay inside the vocabulary and are still
    wrong: a universal law of chemistry, an atom index read as an atomic number,
    and a per-atom negative widened into a statement about the whole molecule.

    The last is the one that matters most. "atom 14 is not part of an ether" is
    a fact about atom 14; "this molecule contains no ether group" is a different
    claim, false for the molecule that fact was drawn from, and the two are one
    sentence apart. Only a `fg_presence` fact can license the wide form.
    """
    text = answer or ""
    found = []
    match = _UNIVERSAL_RE.search(text)
    if match:
        found.append(f"universal: {match.group(0).strip().lower()}")
    if _ATOM_PROPERTY_RE.search(text):
        found.append("atom index read as an atomic property")

    wide = {_subject_of(f) for f in facts
            if _fact_field(f, "family") == "fg_presence"}
    for fact in facts:
        if _fact_field(fact, "family") != "fg_atom_membership":
            continue
        if _fact_field(fact, "value") != "no":
            continue
        group = _subject_of(fact)
        if not group or group in wide or group.startswith("atom "):
            continue
        if re.search(rf"\b(?:contains?|has|have|with)\s+no\s+{re.escape(group)}",
                     text, re.IGNORECASE) or re.search(
                         rf"\bis\s+not\s+an?\s+{re.escape(group)}\b",
                         text, re.IGNORECASE):
            found.append(f"molecule-wide: {group}")
    return found


def opener_contradicts(answer: str, question: str, facts) -> bool:
    """True when the answer opens "Yes" and then says the opposite.

    "Yes, atom 8, which is a nitrogen atom, is not part of a hydroxyl group"
    verifies: the subject is named and the polarity of the clause matches the
    fact. The opener is what a reader takes away, and it says the other thing.

    A question that is itself negative ("is it true that it is not aromatic?")
    can be confirmed with "yes" and a negated clause, so a negated question
    disarms the check rather than tripping it.

    **Two ways this over-fired, both measured on the v2 smoke and both costing
    correct rows.** Together they were 20 of 184 written rows.

    * A bare "No." has no clause after the opener, so there is nothing for the
      opener to contradict — but an empty rest reads as un-negated, and an
      un-negated rest under a "no" opener was the failing combination. Every
      "No." answer to a `no` fact was being thrown away, which is the single
      most ordinary correct answer in the set.
    * "Yes" is not always a polarity claim. Asked "what can be inferred about
      its composition?", a writer opens "Yes, it can be inferred that the
      molecule lacks any amide group" — where "Yes" is a discourse marker and
      "lacks" is the fact, stated correctly. The check only means anything when
      the question was a yes/no question in the first place.
    """
    head = (answer or "").strip().lower()[:6]
    if not re.match(r"^(yes\b|no\b)", head):
        return False
    if any(word in (question or "").lower() for word in NEGATIONS):
        return False
    if not _is_polar_question(question):
        return False
    if not any(_fact_field(f, "kind") == "yesno" for f in facts):
        return False
    first = (_sentences(answer or "") or [""])[0].lower()
    rest = re.sub(r"^\W*(yes|no)\b[,.\s]*", "", first).strip()
    if not rest:
        return False
    negated = any(word in rest for word in NEGATIONS)
    return negated == head.startswith("yes")


#: A question a "yes" or a "no" actually answers. Anywhere else an opening
#: "Yes," is a discourse marker and carries no polarity to contradict.
_POLAR_RE = re.compile(
    r"^\W*(?:is|are|was|were|does|do|did|can|could|has|have|had|will|would|"
    r"should|may|might)\b|\bis it true\b|\bcorrect\?|\bright\?",
    re.IGNORECASE)

#: A yes/no question asked without an interrogative lead. The brief's "an
#: instruction" shape asks for exactly this — "Determine if atom 13 is part of
#: a ring" — and it is as neutral as the interrogative it paraphrases. The
#: window between the verb and the complementiser is bounded so that a verb in
#: one clause cannot reach an "if" in the next.
_POLAR_IMPERATIVE_RE = re.compile(
    r"\b(?:determine|check|verify|confirm|identify|specify|state|say|assess|"
    r"evaluate|examine|establish|indicate|report|tell\s+me)\b[^.?!]{0,48}?"
    r"\b(?:if|whether)\b", re.IGNORECASE)

#: Where one clause ends and the next may begin with its own interrogative.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|,\s+(?=\w)")


def _question_clauses(question: str) -> list:
    """``question`` split where one clause ends and the next may begin.

    Not `_clauses`, which splits an *answer* at "because"/"which" to isolate a
    definition riding inside a sentence. Two different jobs, and the names
    collided silently: the later definition won at module scope and this one was
    never called, so a preamble ending in a comma stayed glued to the question
    it preceded.
    """
    return [c for c in _CLAUSE_SPLIT_RE.split(question or "") if c.strip()]


def _is_polar_clause(clause: str) -> bool:
    """One clause, asking something a yes or a no answers."""
    return (_POLAR_RE.search(clause) is not None
            or _POLAR_IMPERATIVE_RE.search(clause) is not None)


def _is_polar_question(question: str) -> bool:
    """Does any clause of ``question`` ask something a yes or a no answers?

    **The lead has to be looked for clause by clause.** Anchoring it at the
    start of the whole string is the same mistake as reading only the first
    sentence: every brief that asks for a preamble — the scenario shape, the
    stated reason for asking, the regulatory voice — puts the interrogative
    second, and "To map the structure, are atoms 10 and 20 primary amines?" is
    as polar as "Are atoms 10 and 20 primary amines?". Measured on one 400-row
    arm, the anchored version called **0 of 23** leak rejections polar, so the
    yes-fact exemption this function exists to enable never once applied and
    every plain question about a `yes` fact was refused for containing its own
    subject.

    The clause still has to *begin* with the auxiliary, which is what keeps the
    exemption honest: "is atom 1 in a ring?" is a question and "atom 1 is in a
    ring" is an assertion, and only the first one starts with one.
    """
    return any(_is_polar_clause(clause)
               for clause in _question_clauses(question))


#: Words by which a question commits to an answer it is nominally asking for.
#: "Why is atom 1 not in a ring?" does not ask whether atom 1 is in a ring; it
#: presupposes that it is not, and the answer is then in the prompt.
_PRESUPPOSING = (r"\bwhy\b", r"\bhow come\b", r"\bwhat makes\b",
                 r"\bexplain\b", r"\breason (?:that|why)\b")
_PRESUPPOSING_RE = re.compile("|".join(_PRESUPPOSING), re.IGNORECASE)


def question_leaks(question: str, facts, form: str = "") -> bool:
    """Does the question already state its own answer?

    An example whose question carries its answer teaches the model to copy the
    prompt instead of reading the graph, so it is refused — outside the two
    brief forms that ask for exactly that (`LEAKY_FORMS`).

    **A neutral yes/no question is not a leak, and containment cannot tell.**
    "Is atom 3 (C) in a ring?" contains its own yes-fact by every test this
    module has: the subject is named and no negation appears, which is exactly
    what asserting it would look like. But the question commits to nothing —
    it is the plainest question in the set, and rejecting it was 2 of every 8
    leak rejections. So a *yes* fact is exempt when the question is polar and
    neither negates nor presupposes. A "no" fact gets no such exemption: the
    only way a question reaches the negative is to put it there.
    """
    if form in LEAKY_FORMS or not facts:
        return False
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    text = question or ""
    neutral = (not _PRESUPPOSING_RE.search(text)
               and not any(word in text.lower() for word in NEGATIONS))
    clauses = _question_clauses(text) if neutral else []
    contained = facts_contained(text, facts)
    for i, fact in enumerate(facts):
        if not contained[i]:
            continue
        if fact.kind == "yesno" and fact.value == "yes" and _asks_for(fact, clauses):
            continue
        return True
    return False


def _asks_for(fact, clauses) -> bool:
    """Is this fact asked about rather than asserted?

    **The polarity has to belong to the clause the fact is in.** "My friend
    found a molecule where atom 3 (C) is in a ring. Should I tell them?" has a
    polar clause and still leaks, because the clause carrying the fact is a
    statement and the polar one is about something else. So the exemption is
    granted per fact, by the clause that holds it:

    * a clause holds the fact and asks about it -> asked, not asserted;
    * no clause holds the whole fact, so nothing states it in one breath —
      "Check the connectivity for atom 15. Confirm if it's part of a cyclic
      system" spreads the atom and the predicate over two clauses — and a
      polar clause somewhere is then enough;
    * a clause holds the fact and none that holds it is polar -> asserted.
    """
    if not clauses:
        return False
    holding = [c for c in clauses
               if facts_contained(c, [fact])[0] and _states_predicate(c, fact)]
    if not holding:
        return any(_is_polar_clause(c) for c in clauses)
    return any(_is_polar_clause(c) for c in holding)


#: Words in a fact's own sentence that carry no predicate. Everything else in it
#: is what the fact actually claims.
_PREDICATE_STOP = frozenset((
    "atom", "atoms", "this", "that", "molecule", "part", "with", "have", "has",
    "its", "it", "is", "are", "in", "of", "the", "a", "an", "no", "not", "and",
    "to", "be", "there", "contains", "containing", "could", "some", "any",
))


def _states_predicate(clause: str, fact) -> bool:
    """Does ``clause`` carry what the fact claims, not merely its subject?

    Clause-level containment is weaker than it looks: `_subject_of` an
    atom-scoped fact is the atom reference alone, so "I'm looking at atom 16."
    counts as stating "atom 16 is in an aromatic ring" purely by naming atom 16
    without negating it. Over a whole answer that is the right reading; over one
    clause of a question it is not, and it was refusing every question that
    introduced its atom in one breath and asked about it in the next.

    So an assertion has to carry a content word from the fact's own sentence.
    Short tokens are dropped because a fact's element letter — the "(C)" in
    "atom 16 (C)" — matches almost any text.
    """
    words = {w for w in re.findall(r"[a-z]+", (_fact_field(fact, "text") or "").lower())
             if len(w) >= 4 and w not in _PREDICATE_STOP}
    if not words:
        return True
    lowered = clause.lower()
    return any(word in lowered for word in words)


#: A question whose answer has to be a number. "How many", and the handful of
#: ways of asking it without those two words.
_COUNT_QUESTION_RE = re.compile(
    r"\bhow many\b|\bwhat (?:is|was) the (?:number|count|total)\b|"
    r"\bthe number of\b|\bhow large\b|\bwhat size\b|\bhow big\b",
    re.IGNORECASE)


def question_mismatches_facts(question: str, answer: str, facts) -> str:
    """A question the drawn fact cannot answer, as a reason tag or ``""``.

    The residual defect after everything else, and the one §9.4 named without
    being able to check: "questions that ask something the drawn fact does not
    answer". Containment never sees it, because the *value* does turn up in the
    answer — it is the question that has wandered off. Two shapes, both from the
    v2 hand review:

    * a count question answered from a yes/no fact — "How many atoms are in an
      aromatic ring containing atom 24?" answered "yes";
    * a question about something nothing computed, answered with the drawn
      fact's number anyway — "Is bromine present in the molecule?" answered "0,
      because bromine is not present", written from a fact about a ring size.
      Nothing in this pipeline knows whether that molecule contains bromine.

    The second is caught by the atom the fact is scoped to: a fact about atom 20
    that neither the question nor the answer ever mentions is a fact the example
    is not really about. That is a weaker claim than checking the topic, and it
    is one the fact sheet actually supports.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    if not facts:
        return ""

    if _COUNT_QUESTION_RE.search(question or ""):
        if not any(f.kind == "count" for f in facts):
            return "a count question written from no count fact"

    scoped = [f for f in facts if f.atoms]
    if scoped and not any(_names_its_atom(f"{question} {answer}", f)
                          for f in scoped):
        atoms = sorted({a for f in scoped for a in f.atoms})
        return f"never mentions the atom the fact is about: {atoms}"
    return ""


#: Subjects a question can raise, and the families that can answer one. The
#: check this drives is one-directional: raising a subject with no fact behind it
#: is a question the example cannot answer, while *not* raising one is fine —
#: most questions name their subject in words the sheet never uses.
#:
#: Functional groups are added from `FUNCTIONAL_GROUPS` on first use, so the two
#: lists cannot drift apart.
_SUBJECTS = {
    "stereochemistry": (r"\bstereo|\bchiral|\bchirality\b|\benantiomer|"
                        r"\bstereocent|\bstereoisomer|\bR/S\b",
                        ("stereo_potential", "stereo_assigned")),
    "aromaticity": (r"\baromatic|\baromaticity\b",
                    ("aromatic_ring",)),
    "rings": (r"\bring\b|\brings\b|\bcyclic\b|\bring system",
              ("ring_count", "ring_size", "ring_membership", "aromatic_ring")),
    "SMILES": (r"\bSMILES\b", ("smiles",)),
    # No family holds an element census or a molecular formula, so this subject
    # has no family that can answer it and every question raising it is refused.
    # "Does this molecule have three carbons?" answered "Yes … the SMILES starts
    # with C, which represents a carbon atom" is the shape: the SMILES fact is
    # contained, the count is invented, and the molecule has twenty of them.
    # Two patterns only, and deliberately so. A bare "how many atoms" cannot be
    # the trigger: "How many atoms are in the smallest ring containing atom 24?"
    # is a `ring_size` question and one of the commonest shapes in the set, and
    # "How many atoms in this molecule can be stereocentres?" is
    # `stereo_potential`. Every attempt to separate those by lookahead traded one
    # false positive for another, so the pattern stays on the two phrasings no
    # family can answer under any reading — a formula or mass, and a count of a
    # named element. The element half is plural for the same reason: "the atom
    # labelled 15 is a carbon atom" names one atom, it does not ask for a census.
    "atom census": (r"\bmolecular (?:weight|formula|mass)\b|"
                    r"\batomic (?:weight|mass|number)\b|"
                    r"\b(?:carbon|nitrogen|oxygen|hydrogen|sulfur|sulphur|"
                    r"chlorine|bromine|fluorine|iodine|halogen)s? atoms\b",
                    ()),
}

_SUBJECT_RE = {}


def _subject_res() -> dict:
    """The compiled subject patterns, with the functional groups filled in."""
    if not _SUBJECT_RE:
        from ..experiments.molecules.tasks import FUNCTIONAL_GROUPS
        groups = "|".join(re.escape(name) for name in sorted(FUNCTIONAL_GROUPS))
        subjects = dict(_SUBJECTS)
        subjects["functional group"] = (
            rf"\bfunctional group|\b(?:{groups})(?:e?s)?\b",
            ("fg_count", "fg_presence", "fg_atom_membership"))
        for name, (pattern, families) in subjects.items():
            _SUBJECT_RE[name] = (re.compile(pattern, re.IGNORECASE), families)
    return _SUBJECT_RE


def question_changes_subject(question: str, facts) -> str:
    """A question whose subject no drawn fact is about, as a reason or ``""``.

    The defect this closes, from the v2 review: *"What is the maximum number of
    chiral centers in this molecule?"* answered *"2, because it has two rings."*
    The fact is a ring count, correct and contained, and every other check passes
    — `question_mismatches_facts` sees a count question and a count fact and is
    satisfied. Nothing in this pipeline computed a chiral-centre count for that
    molecule, so the answer is a guess wearing a verified number.

    Containment can only ask whether the answer used the fact. This asks the
    other half: whether the question was about it. A question that raises
    stereochemistry with no stereo fact drawn, or rings with no ring fact, is
    unanswerable from the sheet however good the number in the answer looks.

    Only subjects the sheet has a family for are checked, and each maps to every
    family that could answer it, so the check fires on a genuinely missing
    subject rather than on a phrasing.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    if not facts:
        return ""
    # A caption is a paragraph about the molecule and may raise anything. A
    # SMILES fact is *not* the same licence, though the first version treated it
    # as one: the string encodes the structure but nothing here read a count out
    # of it, so a SMILES fact answers a question about SMILES and no other.
    if any(f.kind == "text" for f in facts):
        return ""
    used = {f.family for f in facts}
    for subject, (pattern, families) in _subject_res().items():
        if pattern.search(question or "") and not (used & set(families)):
            return subject
    return ""


#: "atom 15", and also "atoms 16, 17 and 15" — a list is the phrasing an
#: invented atom most often arrives in, so the run after the plural is read to
#: its end rather than stopping at the first number.
_ATOM_REFERENCE_RE = re.compile(
    r"\batoms?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)", re.IGNORECASE)


def answer_invents_atom(answer: str, facts) -> str:
    """Atom numbers the answer names that no drawn fact is about.

    From the v2 review: *"which atom is in an aromatic ring?"* answered *"1.
    Atom 15 (C) is in an aromatic ring. 2. Atom 16 (C) is in an aromatic ring."*
    written from a single fact about atom 15. Atom 16 is invented, and the
    invention is invisible to containment because the fact's own value is right
    there in sentence one.

    An atom index has exactly one legitimate source here — the fact sheet — so
    this needs no chemistry to check, and it is exact rather than heuristic. The
    numbering is the sheet's 1-based one, which is also what the answer sees.

    A caption or a SMILES answer is exempt: both are prose the sheet did not
    write and neither is scoped to an atom.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    if any(f.kind in ("text", "smiles") for f in facts):
        return ""
    allowed = {a for f in facts for a in f.atoms}
    named = {int(n) for run in _ATOM_REFERENCE_RE.findall(answer or "")
             for n in re.findall(r"\d+", run)}
    invented = sorted(named - allowed)
    return ",".join(str(a) for a in invented)


#: Quantifiers that put a question to the whole molecule rather than to an atom.
_MOLECULE_SCOPE_RE = re.compile(
    r"\bthis molecule\b|\bthe molecule\b|\bthis compound\b|\bthe compound\b|"
    r"\bany\b|\bit\b", re.IGNORECASE)


def question_widens_scope(question: str, facts) -> str:
    """A molecule-level question answered from an atom-level fact.

    "Is this molecule an ether?" answered "No, because atom 15 (C) is not part
    of an ether." The fact is about atom 15 and the question is about the
    molecule, so the "No" is a generalisation from one atom to all of them —
    false for any molecule with an ether somewhere else. `ungrounded_claims`
    already refuses the widening when the *answer* spells it out ("contains no
    ether"), but here the widening rides on the question and the answer only has
    to say "No".

    So the check is on scope, not on wording: every drawn fact scoped to an atom,
    a question that names no atom, and a question that quantifies over the
    molecule. A question that names its atom is exempt however it is phrased, and
    a molecule-level fact is not this check's business.

    **One witness proves existence.** "Are any atoms in an aromatic ring?"
    answered "Yes. Atom 8 is in an aromatic ring" is sound, and the check refused
    it in its first form. The asymmetry is the same one `question_leaks` runs on:
    an existential question is settled by a single positive instance, while no
    number of atom-level facts settles it in the negative — "atom 15 is not part
    of an ether" leaves every other atom unexamined. So an existential question
    is exempt exactly when every atom-scoped fact is positive.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    if not facts or not all(f.atoms for f in facts):
        return ""
    if _ATOM_REFERENCE_RE.search(question or ""):
        return ""
    match = _MOLECULE_SCOPE_RE.search(question or "")
    if not match:
        return ""
    if (_EXISTENTIAL_RE.search(question or "")
            and all(f.value.lower() == "yes" for f in facts)):
        return ""
    return match.group(0).strip().lower()


#: Questions a single positive fact settles: they ask whether something occurs
#: at all, not what is true of the molecule as a whole.
_EXISTENTIAL_RE = re.compile(
    r"\bany\b|\bis there\b|\bare there\b|\bdoes (?:it|this|the) \w+ (?:contain|have)\b|"
    r"\bdoes (?:it|this molecule|the molecule) (?:contain|have)\b|"
    r"\bcontains? an?\b|\bis part of\b|\bare part of\b", re.IGNORECASE)


#: Adjacency, in the vocabulary a writer reaches for. **No family in the sheet
#: holds connectivity** — not bonds, not neighbours, not degree — so any of these
#: about the molecule at hand is invented, and inventing it is worse here than
#: elsewhere: bonds are exactly what the graph arm is supposed to read off the
#: structure rather than recite from text.
_CONNECTIVITY_RE = re.compile(
    r"\bbonded\b|\bbonds to\b|\bconnected to\b|\bconnects to\b|\battached to\b|"
    r"\badjacent to\b|\bneighbou?ring\b|\bneighbou?r of\b|\blinked to\b|"
    r"\b(?:single|double|triple) bond\b|\bcovalent bond\b",
    re.IGNORECASE)


def claims_connectivity(answer: str, facts) -> str:
    """Adjacency asserted in an answer, which no fact sheet here supports.

    From the review: "Yes, atom 13 is aromatic. This is because the atom is
    bonded to four atoms, and one of them is an aromatic atom", and "1, because
    this molecule has one oxygen atom connected to two carbon atoms". Both state
    a true value and justify it with a bond structure nobody computed. The value
    passes containment and the justification is the part a reader learns from.

    `invented_atom` catches this only when the invented bond names an index.
    This catches it by vocabulary instead, which is blunt, and on the rejection
    log a third of the first version's catches were **generic definitions** —
    "A ketone is a compound whose central carbon is bonded to …", "Amides are
    characterised by a carbonyl group bonded to a nitrogen". Those state no
    connectivity about the molecule at hand, so a clause in definitional shape is
    dropped before the search.

    What is left is deliberately blunt, and that is the right side to err on: an
    answer that reasons from a definition the pipeline never checked applies is
    not factually wrong so much as wrong about where its answer came from, and
    the set is drawn from several times more candidates than it keeps.

    A caption or a SMILES answer is exempt: a caption is prose the sheet did not
    write, and a SMILES string is connectivity, stated in the one notation this
    pipeline actually computes.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    if any(f.kind in ("text", "smiles") for f in facts):
        return ""
    for clause in _clauses(answer or ""):
        if _DEFINITIONAL_RE.match(clause.strip()):
            continue
        match = _CONNECTIVITY_RE.search(clause)
        if match:
            return match.group(0).strip().lower()
    return ""


#: A clause opening in definitional shape — "A ketone is …", "Amides are …",
#: "The definition of a hydroxyl group is …" — says what a group is in general,
#: not what this molecule looks like.
_DEFINITIONAL_RE = re.compile(
    r"(?:an?\s+[\w-]+(?:\s+[\w-]+)?|[\w-]+s|the definition of [\w\s-]+)\s+"
    r"(?:is|are)\b", re.IGNORECASE)


def _clauses(text: str) -> list:
    """Sentences, split further at "because" and "which" so a definition riding
    inside a larger sentence is still its own clause."""
    parts = []
    for sentence in _sentences(text):
        parts.extend(re.split(r",?\s+(?:because|since|as|which|where)\s+",
                              sentence))
    return parts


#: The families a JSON answer may name its value after. A key naming a family
#: the example was not written from is a mislabel, not a stylistic choice:
#: `{"ring_count": 0}` written from "the smallest ring containing atom 25 has 0
#: atoms" teaches that a molecule with rings has none.
_FAMILY_KEYS = ("ring_count", "ring_size", "aromatic_ring", "fg_count",
                "fg_presence", "fg_atom_membership", "stereo_potential",
                "stereo_assigned", "smiles", "caption")


def json_key_misnames(answer: str, brief: dict, facts) -> str:
    """The family a JSON answer's key claims, when that is not the family it was
    written from. Empty when the key is free-form, which is allowed."""
    if "JSON" not in (brief or {}).get("format", ""):
        return ""
    used = {_fact_field(f, "family") for f in facts}
    for key in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', answer or ""):
        lowered = key.lower()
        if lowered in _FAMILY_KEYS and lowered not in used:
            return lowered
    return ""


_ATOM_NAMED_RE = re.compile(r"\batom\s+#?(\d+)", re.IGNORECASE)


def question_drifts(question: str, seed: str, facts) -> str:
    """What a rewrite of `seed` lost or invented, or "" if it asks the same thing.

    Only for the rephrase pipeline, where the question arrives instantiated from
    a template and the model's job is to reword it. The template guarantees the
    question corresponds to its facts; this guarantees the rewrite still is the
    question the template wrote.

    **It checks anchors, not wording.** A rewrite that changes nothing has added
    no variety and a rewrite that changes everything is the invention this whole
    step removed, so what is enforced is the small set of things that cannot
    change: which atoms are named, and which groups. Whether the rewrite still
    asks for the right *quantity* is left to `question_mismatches_facts` and
    `question_widens_scope`, which were built for exactly that and now have only
    a reworded question to catch rather than a freely invented one.
    """
    wanted = set(_ATOM_NAMED_RE.findall(seed or ""))
    named = set(_ATOM_NAMED_RE.findall(question or ""))
    loose = set(re.findall(r"\b\d+\b", question or ""))
    missing = wanted - (named | loose)
    if missing:
        return "drops atom " + ", ".join(sorted(missing, key=int))
    extra = named - wanted
    if extra:
        return "names atom " + ", ".join(sorted(extra, key=int)) + " unasked"

    lowered = (question or "").lower()
    for fact in facts:
        family = _fact_field(fact, "family")
        if not str(family).startswith("fg_"):
            continue
        # The group is the one word the rewrite may not paraphrase: "does it
        # have an -OH" instead of "a hydroxyl group" is a different question to
        # a verifier that looks for the family by name.
        # Anchored on the verb, not just on the article: an unanchored search
        # finds the "12" in "atom 12 (O) is part of an ether" and reads the rest
        # of the sentence as the group's name.
        match = re.search(r"(?:part of|contains?)\s+(?:no|an?|the|\d+)\s+"
                          r"(.+?)(?:\(s\))?\.$",
                          _fact_field(fact, "text") or "")
        name = (match.group(1).strip().lower() if match else "")
        if name and name in (seed or "").lower() and name not in lowered:
            return f"drops the group {name}"
    return ""


def four_grams(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def verify(answer: str, facts, brief=None) -> dict:
    """The task's ``verify`` (D2) and the correctness metric, in one call.

    An example passes when every fact it was written from is stated in the answer
    and the brief's format instruction is met. The per-fact detail is returned
    beside the verdict because §9.4 reports correctness broken down by fact
    family and by brief, and that breakdown is not recoverable from a boolean.
    """
    facts = [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]
    contained = facts_contained(answer, facts)
    formatted = format_met(answer, brief or {}, facts)
    return {
        "passed": all(contained.values()) and formatted,
        "format_met": formatted,
        "facts": [{"family": f.family, "value": f.value, "contained": contained[i]}
                  for i, f in enumerate(facts)],
    }
