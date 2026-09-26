"""The answer's content, rendered from the fact sheet — never written by a model.

§9.4's rule in its strong form. Earlier builds handed a model a fact sheet and
asked it to compose a reply; three rounds of filters then tried to establish
whether what came back was true. This module composes the reply, in a plain
register, out of statements RDKit computed, and the model's whole remaining job
is to change that reply's voice. Invented chemistry stops being a category that
can occur, because no chemistry is generated.

Two things come out of a render and they have different jobs:

* ``statements`` is the list of claims the reply makes. It is what verification
  checks survived the re-voicing, and nothing else in the pipeline may add to it.
* ``reply`` is those statements welded into a plain-register answer shaped by the
  task. It is what the writer is handed.

Everything here is deterministic given (facts, task, twist). Nothing in this
module imports a model, and nothing in it is allowed to say anything the fact
sheet does not license — which is why `GLOSS` is a closed, readable dictionary
rather than anything generated.
"""

import json
import re

from .assistant import Fact, _article


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

#: The twists, and which tasks each one is available on. A twist changes what the
#: reply has to *do*, which is the axis the earlier builds had no room for.
TWISTS = ("none", "false_premise", "unanswerable", "needs_clarification")

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

#: The JSON type an off-sheet field would have carried, so that a `fill_record`
#: null sits in a typed schema next to the fields that do have values.
_OFF_SHEET_KINDS = {
    "melting_point": "number", "boiling_point": "number", "pka": "number",
    "logp": "number", "ld50": "number", "binding_affinity": "number",
}


#: What a fact is *asked about*, with its answer taken out. The writer of the
#: person's turn is given these and never the statements, so a question cannot
#: state its own answer — the defect that was the largest single rejection reason
#: across three builds is not caught here, it is unreachable.
ASK_PHRASES = {
    "ring_membership": "whether {atom} is in a ring",
    "ring_size": "the size of the smallest ring containing {atom}",
    "ring_count": "how many rings it has",
    "aromatic_ring": "whether {atom} is in an aromatic ring",
    # The family covers two shapes. "2 of its rings are aromatic." carries no
    # atom, and falling back to `ring_count` there asked how many rings the
    # molecule has and answered how many of them are aromatic.
    "aromatic_ring/count": "how many of its rings are aromatic",
    # `{a_group}` carries the article: `_group_of` strips it off the sheet's
    # wording, and "whether it contains nitrile" is not a sentence anyone types.
    # `{groups}` is the plural, because "how many hydroxyl group groups" is what
    # you get from pasting "groups" onto a name that already ends in one.
    "fg_presence": "whether it contains {a_group}",
    "fg_count": "how many {groups} it contains",
    "fg_atom_membership": "whether {atom} is part of {a_group}",
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

_GROUP_RES = (
    re.compile(r"part of an? ([a-z][a-z ]*?)(?: group)?\.?$", re.IGNORECASE),
    re.compile(r"contains no ([a-z][a-z ]*?)\.?$", re.IGNORECASE),
    re.compile(r"contains \d+ ([a-z][a-z ]*?)\(s\)\.?$", re.IGNORECASE),
    re.compile(r"It contains an? ([a-z][a-z ]*?)\.?$", re.IGNORECASE),
)


def _group_of(fact) -> str:
    text = _clause(fact)
    for pattern in _GROUP_RES:
        match = pattern.search(text)
        if match:
            # "hydroxyl group" and "hydroxyl" are the same group under two
            # spellings the sheet uses interchangeably; the bare name is the one
            # that takes an article and a plural cleanly.
            return re.sub(r"\s+group$", "", match.group(1).strip(),
                          flags=re.IGNORECASE)
    return ""


#: What a `decide` constraint calls the number it is about. "I need that number
#: to be at least 5" says which number only when the draw holds one, and a draw
#: of two counts is common: "How many ethers and how many hydroxyls does this
#: molecule contain? I need that number to be at least 5" names neither, while
#: the verdict behind it was computed on the first.
COUNT_SUBJECTS = {
    "ring_size": "the size of the smallest ring containing {atom}",
    "ring_count": "the number of rings",
    "aromatic_ring/count": "the number of aromatic rings",
    "fg_count": "the number of {groups}",
    "stereo_potential": "the number of atoms that could be stereocentres",
    "stereo_assigned": "the number of assigned stereocentres",
}


def _count_subject(fact) -> str:
    """The counted thing, named, for a constraint stated mid-sentence.

    Falls back to "that number" wherever the name cannot be filled in, which is
    the old wording and no worse than it: an unnamed number is ambiguous beside
    a second count, never wrong.
    """
    template = COUNT_SUBJECTS.get(f"{fact.family}/{fact.kind}") \
        or COUNT_SUBJECTS.get(fact.family)
    if template is None:
        return "that number"
    atom, group = _atom_ref(fact), _group_of(fact)
    if ("{atom}" in template and not atom) or ("{groups}" in template
                                               and not group):
        return "that number"
    return template.format(atom=atom, groups=_plural(group) if group else "")


def _plural(group: str) -> str:
    if group.endswith("s"):
        return group
    if re.search(r"(?:ch|sh|x|z)$", group):
        return group + "es"
    if re.search(r"[^aeiou]y$", group):
        return group[:-1] + "ies"
    return group + "s"


def ask_phrase(fact) -> str:
    """What this fact answers, phrased as the thing a person would ask for.

    Built from the sheet's own wording with the polarity removed, so it names the
    same atom and the same group as the fact and commits to nothing about the
    value. A family with no entry falls back to the sentence with its negation
    and its number stripped, which is coarse but still says nothing.
    """
    atom = _atom_ref(fact)
    group = _group_of(fact)
    template = ASK_PHRASES.get(f"{fact.family}/{fact.kind}") \
        or ASK_PHRASES.get(fact.family)
    if template is None and fact.family.startswith("tier_b/"):
        # "It is not active in the SR ATAD5 assay." -> "whether it is active ..."
        stripped = re.sub(r"\bis not\b", "is", _clause(fact), count=1)
        stripped = re.sub(r"\b(shows?|has|have) no\b", r"\1", stripped, count=1)
        return f"whether {stripped[0].lower() + stripped[1:]}"
    if template is None:
        stripped = re.sub(r"\b(?:not|no)\b", "", _clause(fact))
        stripped = re.sub(r"\b\d+\b", "", stripped)
        return re.sub(r"\s{2,}", " ", stripped).strip()
    if "{atom}" in template and not atom:
        return ASK_PHRASES.get("ring_count", template)
    if not group and any(slot in template
                         for slot in ("{group}", "{a_group}", "{groups}")):
        return template.format(atom=atom or "that atom", group="that group",
                               a_group="that group", groups="of those groups")
    return template.format(atom=atom or "that atom", group=group,
                           groups=_plural(group) if group else "",
                           a_group=f"{_article(group)} {group}" if group else "")


def answer_token(fact):
    """The one thing a terse reply to this fact has to carry.

    A count answers with its number and a yes/no with its polarity; anything else
    has no token and can only be checked as prose. This is the renderer's answer
    to "what did the reply have to say", and it exists because the accept pass
    was re-deriving it from the *sentence* and getting it wrong in four different
    ways — an assay called `SR ATAD5` read as the number 5, an atom index read as
    a value, "Two" not matching 2, and "rings" not matching "ring".
    """
    if fact.kind == "count":
        try:
            return str(int(fact.value))
        except (TypeError, ValueError):
            return None
    if fact.kind == "yesno":
        return "yes" if fact.value == "yes" else "no"
    return None


class Render:
    """What a renderer returns: the claims, the plain reply, and the turns."""

    __slots__ = ("statements", "reply", "turns", "ask", "skeleton", "kinds",
                 "answers", "verdict", "gloss")

    def __init__(self, statements, reply, turns, ask, skeleton=None, kinds=(),
                 answers=None, verdict=None, gloss=None):
        self.statements = list(statements)
        self.reply = reply
        self.turns = list(turns)          # ("person"|"assistant", text)
        self.ask = ask                    # what the person wants, for the writer
        self.skeleton = skeleton          # declared JSON shape, or None
        self.kinds = tuple(kinds)         # the drawn facts' kinds, for the style
        # One token per statement, in the same order, or None where the statement
        # has no scalar answer. A list-format reply answers by position, so this
        # is what each position has to carry.
        self.answers = list(answers if answers is not None
                            else [None] * len(self.statements))
        # Where the whole reply is a single verdict — `check_claim` and `decide`
        # answer "yes"/"no" about the *claim*, not about the facts — a terse
        # reply carries this and none of the answers above.
        self.verdict = verdict
        # The general sentence an `explain` reply owes. Alongside the verdict
        # rather than inside `ask` because it is the same kind of thing: content
        # the reply must carry that no statement licenses, and so invisible to
        # every check that enumerates statements.
        self.gloss = gloss

    def to_json(self) -> dict:
        out = {"statements": self.statements, "reply": self.reply,
               "turns": [{"role": r, "text": t} for r, t in self.turns],
               "ask": self.ask, "kinds": list(self.kinds),
               "answers": self.answers}
        if self.skeleton is not None:
            out["skeleton"] = self.skeleton
        if self.verdict is not None:
            out["verdict"] = self.verdict
        if self.gloss is not None:
            out["gloss"] = self.gloss
        return out


def _sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _clause(fact) -> str:
    """A fact's sentence with its full stop removed, so it can be joined."""
    return (fact.text or "").strip().rstrip(".")


def _lower_first(text: str) -> str:
    """A clause put mid-sentence. Atom facts already read lower-case; the
    molecule-level ones start "It", which reads wrong after "if"."""
    return text[0].lower() + text[1:] if text else text


def _atom_ref(fact) -> str:
    """The atom a fact is about, as it is written on the sheet.

    Searched anywhere in the sentence rather than anchored at the front: a
    `ring_size` fact reads "The smallest ring containing atom 7 (C) has 6 atoms",
    and anchoring found nothing there, which is how a clarification turn came to
    ask "Which atom do you mean?" and be answered "The atom."
    """
    match = re.search(r"(atom \d+(?:\s*\([A-Za-z]{1,2}\))?)", fact.text or "",
                      re.IGNORECASE)
    return match.group(1) if match else ""


def _one_atom(facts) -> str:
    """The single atom every fact is about, or "" if they differ or there is none."""
    refs = {_atom_ref(f).lower() for f in facts}
    if len(refs) != 1:
        return ""
    ref = refs.pop()
    return _atom_ref(facts[0]) if ref else ""


def _negate(fact) -> str:
    """The fact's own sentence with its polarity flipped, for a false premise.

    Only ever built from the sheet's own wording, so the false claim a person
    brings is the exact contradiction of something computed rather than a
    plausible-sounding invention.
    """
    text = _clause(fact)
    if fact.kind == "yesno":
        if re.search(r"\bis not\b", text):
            return re.sub(r"\bis not\b", "is", text, count=1)
        if re.search(r"\bare not\b", text):
            return re.sub(r"\bare not\b", "are", text, count=1)
        if re.search(r"\bcontains no\b", text):
            # "contains no primary amine" -> "contains a primary amine", not
            # "contains primary amine". The constraint this builds goes into the
            # person's turn verbatim, so the article is not cosmetic.
            group = _group_of(fact)
            article = _article(group) if group else "a"
            return re.sub(r"\bcontains no\b", f"contains {article}", text,
                          count=1)
        if re.search(r"\bcontains an?\b", text):
            # The same family's positive form, and the article has to go with
            # the verb: falling through to the bare `contains` branch below
            # yields "contains no an ether".
            return re.sub(r"\bcontains an?\b", "contains no", text, count=1)
        # A polarity carried by the verb rather than by a "not": the HIV family
        # reads "It shows no activity against HIV replication", and falling
        # through to the wrapper below turned its false premise into the double
        # negative "it is not the case that it shows no activity".
        if re.search(r"\b(?:shows?|has|have) no\b", text):
            return re.sub(r"\b(shows?|has|have) no\b", r"\1", text, count=1)
        if re.search(r"\bis\b", text):
            return re.sub(r"\bis\b", "is not", text, count=1)
        if re.search(r"\bcontains\b", text):
            return re.sub(r"\bcontains\b", "contains no", text, count=1)
        if re.search(r"\bshows\b", text):
            return re.sub(r"\bshows\b", "shows no", text, count=1)
        return f"it is not the case that {text[0].lower() + text[1:]}"
    if fact.kind == "count":
        try:
            wrong = int(fact.value) + 1
        except (TypeError, ValueError):
            return f"it is not the case that {text[0].lower() + text[1:]}"
        flipped = re.sub(rf"\b{re.escape(str(fact.value))}\b", str(wrong), text,
                         count=1)
        # A count whose sentence does not contain its own digit. `ring_size` 0
        # reads "atom 24 (O) is in no ring" and nothing there says "0", so the
        # substitution above was a no-op and the false premise came out as the
        # statement verbatim — 39 rows of the final build had a person assert
        # something true and get told it was wrong. Fall through to the wrapper,
        # which contradicts the sentence whatever its wording; `false_premise`
        # also refuses this pivot outright, and the two are belt and braces.
        if flipped != text:
            return flipped
    return f"it is not the case that {text[0].lower() + text[1:]}"


def _polarised(fact, positive: bool) -> str:
    """The fact's predication in the asked-for polarity, whatever its value.

    `_clause` renders a fact as the sheet states it, so it already carries the
    *value's* polarity, and `_negate` flips that. Neither is "the positive form"
    on its own, and reading them as if they were is what made a `decide`
    constraint contradict the statement it was decided against: on a fact whose
    value is "no", "I only want it if X" came out as "I only want it if not X"
    while the verdict had been computed for "X", and the rendered reply then read
    "Yes, that one qualifies. Atom 9 (C) is not in an aromatic ring."
    """
    return _clause(fact) if (fact.value == "yes") == positive else _negate(fact)


# --------------------------------------------------------------------------
# the tasks
# --------------------------------------------------------------------------

def _report(facts, twist, rng, spare_family=None):
    if twist == "unanswerable":
        words = FAMILY_WORDS.get(spare_family, "that property")
        anchor = facts[0] if facts else None
        known = _clause(anchor) if anchor is not None else ""
        statements = [f"{_sentence(known)}." ] if known else []
        reply = (f"I can't tell you {words} — that isn't something I have for "
                 f"this structure.")
        if known:
            reply += f" What I do have: {known}."
        return statements, reply, {"wants": words, "answerable": False,
                                   "_answers": [answer_token(anchor)]
                                   if known else []}

    statements = [f"{_sentence(_clause(f))}." for f in facts]
    reply = " ".join(statements)
    return statements, reply, {"wants": "the value", "answerable": True,
                               "_answers": [answer_token(f) for f in facts]}


def _compare(facts, twist, rng, spare_family=None):
    left, right = facts[0], facts[1]
    statements = [f"{_sentence(_clause(left))}.", f"{_sentence(_clause(right))}."]
    # "whereas" is a contrast, and a contrast between two facts that agree is
    # false connective tissue of exactly the kind §9.4 rejects: "Atom 5 (C) is in
    # an aromatic ring, whereas atom 15 (C) is in an aromatic ring." The
    # connective is therefore chosen from the values, which the sheet knows.
    # `_lower_first` on the right-hand clause for the same reason `_decide` needs
    # it: the sheet capitalises a molecule-level fact and `ring_size` opens with
    # "The", so joining the clause raw put a capital mid-sentence — "Atom 1 (C)
    # is in no ring, whereas The smallest ring containing atom 19 (C) has 6
    # atoms."
    if left.value == right.value:
        reply = (f"{_sentence(_clause(left))}, "
                 f"and {_lower_first(_clause(right))} as well.")
    else:
        reply = (f"{_sentence(_clause(left))}, "
                 f"whereas {_lower_first(_clause(right))}.")
    return statements, reply, {"wants": "a comparison", "answerable": True,
                               "agree": left.value == right.value,
                               "_answers": [answer_token(left),
                                            answer_token(right)]}


def _check_claim(facts, twist, rng, spare_family=None):
    pivot = facts[0]
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    if twist == "false_premise":
        claim = _negate(pivot)
        reply = (f"That isn't right. {_sentence(_clause(pivot))}."
                 + ("" if len(facts) == 1
                    else " " + " ".join(statements[1:])))
        ask = {"wants": "a check on a claim", "claim": _sentence(claim) + ".",
               "answerable": True, "claim_is_true": False, "_verdict": "no"}
    else:
        claim = _clause(pivot)
        reply = (f"That's right. {_sentence(_clause(pivot))}."
                 + ("" if len(facts) == 1
                    else " " + " ".join(statements[1:])))
        ask = {"wants": "a check on a claim", "claim": _sentence(claim) + ".",
               "answerable": True, "claim_is_true": True, "_verdict": "yes"}
    ask["_answers"] = [answer_token(f) for f in facts]
    return statements, reply, ask


#: The constraints a `decide` ask can state, per fact kind. Each is a predicate
#: over the fact's own value, so the verdict is computed rather than judged.
def _decide(facts, twist, rng, spare_family=None):
    pivot = facts[0]
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    if pivot.kind == "yesno":
        wants_yes = rng.random() < 0.5
        holds = (pivot.value == "yes") == wants_yes
        # `_polarised` and not `_clause`/`_negate`: the constraint states the
        # property in the polarity the *person* wants, which is independent of
        # the polarity the sheet happens to state it in. Lower-cased because it
        # lands mid-sentence — the sheet capitalises a molecule-level fact, and
        # the turn came out as "I only want it if It is not active in ...".
        constraint = (f"I only want it if "
                      f"{_lower_first(_polarised(pivot, wants_yes))}")
    elif pivot.kind == "count":
        try:
            value = int(pivot.value)
        except (TypeError, ValueError):
            value = 0
        # A threshold of zero is not a constraint anyone states either, and it is
        # worse than a negative one: "at least 0" holds for every molecule, so
        # the question cannot be answered wrongly and the answer is always yes.
        # `max(0, value + rng.choice((-1, 0, 1)))` produced one on any count of
        # zero or one, which was 25 of 156 `decide` rows and every one of them a
        # guaranteed "yes" — visible as a verdict balance of 100 to 56.
        threshold = max(1, value + rng.choice((-1, 0, 1)))
        holds = value >= threshold
        constraint = (f"I need {_count_subject(pivot)} to be at least "
                      f"{threshold}")
    else:
        # No predicate exists over a SMILES or a caption, so there is no verdict
        # to compute. The sampler will not draw this pivot for `decide`; raising
        # rather than returning a constant "yes" is what keeps the two in step,
        # since `sample_intent` drops a task whose render refuses.
        raise ValueError(f"decide has no constraint for a {pivot.kind} fact")
    verdict = "Yes, that one qualifies." if holds else "No, that one doesn't qualify."
    ask = {"wants": "a decision", "constraint": constraint, "answerable": True,
           "verdict": holds, "_verdict": "yes" if holds else "no",
           "_answers": [answer_token(f) for f in facts]}
    if twist == "false_premise":
        # The person states the value wrongly *and* asks for the decision. The
        # reply corrects the premise first and then decides on the true value —
        # which is the whole reason the twist is worth having on this task.
        # Without this branch the twist was declared and never rendered, which is
        # the same silent no-op `fill_record` carried.
        ask["claim"] = _sentence(_negate(pivot)) + "."
        ask["claim_is_true"] = False
        reply = f"That isn't right. {' '.join(statements)} {verdict}"
    else:
        reply = f"{verdict} {' '.join(statements)}"
    return statements, reply, ask


def _fill_record(facts, twist, rng, spare_family=None):
    skeleton, payload = {}, {}
    for i, fact in enumerate(facts):
        key = _record_key(fact, i)
        # Last resort, so that a qualifier I have not thought of cannot silently
        # merge two statements into one field.
        while key in skeleton:
            key = f"{key}_{i}"
        skeleton[key] = _kind_word(fact)
        payload[key] = _record_value(fact)
    statements = [f"{_sentence(_clause(f))}." for f in facts]

    if twist == "unanswerable":
        # The record asks for a field this molecule's sheet has no fact for. The
        # honest fill is a null and a sentence saying why, and the null is not a
        # claim, so it does not join the statements.
        key = spare_family.replace("/", "_")
        skeleton[key] = _kind_word_for_family(spare_family)
        payload[key] = None
        words = FAMILY_WORDS.get(spare_family, "that field")
        reply = (json.dumps(payload, sort_keys=True)
                 + f"\n\n{key} is null: I don't have {words} for this structure.")
        return statements, reply, {"wants": "a filled record",
                                   "answerable": False, "missing": key,
                                   "_skeleton": skeleton,
                                   "_answers": [answer_token(f)
                                                for f in facts]}

    return (statements, json.dumps(payload, sort_keys=True),
            {"wants": "a filled record", "answerable": True,
             "_skeleton": skeleton,
             "_answers": [answer_token(f) for f in facts]})


def _kind_word_for_family(family: str) -> str:
    if family in ("ring_count", "ring_size", "fg_count", "stereo_potential",
                  "stereo_assigned"):
        return "integer"
    if family in ("aromatic_ring",):
        return "boolean"
    return _OFF_SHEET_KINDS.get(family, "string")


_ASSAY_RE = re.compile(r"\bin the ([A-Za-z0-9 _-]+?) assay", re.IGNORECASE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _record_key(fact, index: int) -> str:
    """A field name that distinguishes this fact from the others in the record.

    The atom index does it where there is one. Where there is not, the family
    alone does not: a molecule has an `fg_count` per functional group and a
    `tier_b/tox21` per assay, so three assay facts all wanted the key
    `tier_b_tox21`, two of the three values were lost to the dict, and the
    record then failed verification for statements it had never had room for.
    """
    atoms = "_".join(str(a) for a in fact.atoms)
    base = fact.family.replace("/", "_")
    if atoms:
        return f"{base}_atom_{atoms}"
    assay = _ASSAY_RE.search(fact.text or "")
    qualifier = assay.group(1) if assay else _group_of(fact)
    return f"{base}_{_slug(qualifier)}" if _slug(qualifier) else base


def _kind_word(fact) -> str:
    return {"count": "integer", "yesno": "boolean",
            "smiles": "string", "text": "string"}.get(fact.kind, "string")


def _record_value(fact):
    if fact.kind == "count":
        try:
            return int(fact.value)
        except (TypeError, ValueError):
            return fact.value
    if fact.kind == "yesno":
        return fact.value == "yes"
    return fact.value


def _explain(facts, twist, rng, spare_family=None):
    pivot = facts[0]
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    gloss = GLOSS.get(pivot.family)
    reply = " ".join(statements)
    ask = {"wants": "an explanation", "answerable": True,
           "_answers": [answer_token(f) for f in facts]}
    if gloss:
        # The gloss is the whole difference between `explain` and `report`, and
        # it is *not* a statement: it asserts nothing about this molecule, so a
        # check that enumerates statements cannot see it go missing. Nothing
        # named it and the writer dropped it from 614 of the 670 v6 rows that
        # rendered one — 92% — which left `explain` indistinguishable from
        # `report` and the task axis with eight names and seven behaviours.
        #
        # So it travels in the ask, for the same reason and by the same route as
        # `decide`'s verdict: named in the voice prompt, checked in the accept
        # pass, excused to the judge as something the statements do not license
        # but the reply still owes.
        reply += f" In general, {gloss}."
        ask["_gloss"] = f"In general, {gloss}."
    return statements, reply, ask


def _summarise(facts, twist, rng, spare_family=None):
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    reply = " ".join(statements)
    return statements, reply, {"wants": "a short summary", "answerable": True,
                               "_answers": [answer_token(f) for f in facts]}


def _predicate_of(fact) -> str:
    """A fact's sentence with its atom reference removed — "is in a ring".

    Triage groups atoms that share an answer, and a grouped sentence has to be
    built from the sheet's own predicate rather than from a phrasing invented
    here. If the atom reference cannot be found the fact is not groupable and the
    caller falls back to stating it on its own.
    """
    text = _clause(fact)
    ref = _atom_ref(fact)
    if not ref or not text.lower().startswith(ref.lower()):
        return ""
    return text[len(ref):].strip()


def _grouped(group) -> str:
    """One sentence covering several atoms that share a predicate."""
    refs = [_atom_ref(f) for f in group]
    predicate = _predicate_of(group[0])
    if len(refs) == 1:
        return f"{_sentence(refs[0])} {predicate}."
    joined = ", ".join(refs[:-1]) + f" and {refs[-1]}"
    plural = re.sub(r"^is\b", "are", predicate)
    return f"{_sentence(joined)} {plural}."


def _triage(facts, twist, rng, spare_family=None):
    holds = [f for f in facts if f.value == "yes"]
    misses = [f for f in facts if f.value != "yes"]
    groupable = all(_predicate_of(f) for f in facts) and len(
        {_predicate_of(f) for f in holds}) <= 1 and len(
        {_predicate_of(f) for f in misses}) <= 1
    if not groupable:
        statements = [f"{_sentence(_clause(f))}." for f in facts]
        return statements, " ".join(statements), {
            "wants": "which ones qualify", "answerable": True,
            "_answers": [answer_token(f) for f in facts]}

    groups = [group for group in (holds, misses) if group]
    statements = [_grouped(group) for group in groups]
    return statements, " ".join(statements), {
        "wants": "which ones qualify", "answerable": True,
        "_answers": [answer_token(group[0]) for group in groups]}


#: task -> (renderer, how many facts it needs, which twists it accepts)
TASKS = {
    "report":      (_report,      (1, 3), ("none", "unanswerable",
                                           "needs_clarification")),
    # No clarification on `compare`: its two facts are about two atoms by
    # construction, so "which atom do you mean" has no answer that keeps the
    # comparison intact.
    "compare":     (_compare,     (2, 2), ("none",)),
    "check_claim": (_check_claim, (1, 2), ("none", "false_premise")),
    "decide":      (_decide,      (1, 2), ("none", "false_premise")),
    "fill_record": (_fill_record, (1, 3), ("none", "unanswerable")),
    "explain":     (_explain,     (1, 2), ("none",)),
    "summarise":   (_summarise,   (2, 3), ("none",)),
    "triage":      (_triage,      (2, 4), ("none",)),
}

#: Families that cannot be the pivot of a `check_claim` or a `decide`. The pivot
#: goes into the person's turn verbatim, as a claim to check or a constraint to
#: meet, so it has to be something a person could plausibly assert and something
#: the reply can settle in a few words. The canonical SMILES is neither: "I
#: believe its canonical SMILES is COc1ccccc1N1C(=O)C2C3c4ccccc4..." writes the
#: whole molecule into the question and the reply copies it straight back, so the
#: row is answerable without reading the graph at all. 88 rows of the final build
#: were this. Refusing it here rather than in the sampler is deliberate —
#: `sample_intent` catches the ValueError and draws a different task, so the
#: molecule still yields an intent.
UNPIVOTABLE = frozenset(("smiles",))


def render(facts, task: str, twist: str, rng, spare_family: str = None) -> Render:
    """The reply to an intent, as statements plus a plain-register answer.

    `facts` are the sheet facts the intent declared, already drawn to the task's
    arity by the sampler. `spare_family` is only read for an `unanswerable` ask:
    it is a family this molecule's sheet does not carry.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}")
    fn, (low, high), allowed = TASKS[task]
    if twist not in allowed:
        raise ValueError(f"task {task!r} does not take twist {twist!r}")
    if not low <= len(facts) <= high:
        raise ValueError(f"task {task!r} wants {low}-{high} facts, "
                         f"got {len(facts)}")
    if task in ("check_claim", "decide") and facts[0].family in UNPIVOTABLE:
        raise ValueError(f"task {task!r} cannot pivot on {facts[0].family!r}")

    statements, reply, ask = fn(facts, twist, rng, spare_family)
    skeleton = ask.pop("_skeleton", None)
    answers = ask.pop("_answers", None)
    verdict = ask.pop("_verdict", None)
    gloss = ask.pop("_gloss", None)

    if twist == "needs_clarification":
        # The person's first turn is underspecified; the assistant asks which
        # atom; the person names it. The clarifying question is rendered, not
        # written, so the thing asked for is exactly the thing the facts settle —
        # which only works when the facts are all about one atom. `can_clarify`
        # is what the sampler checks before drawing this twist.
        anchor = _one_atom(facts)
        if not anchor:
            raise ValueError("needs_clarification wants one atom across its facts")
        turns = [("person", None),            # the writer fills this
                 ("assistant", "Which atom do you mean?"),
                 ("person", f"{_sentence(anchor)}."),
                 ("assistant", reply)]
        ask = dict(ask, underspecified=True, anchor=anchor)
    else:
        turns = [("person", None), ("assistant", reply)]

    anchors = sorted({_atom_ref(f) for f in facts if _atom_ref(f)}
                     | {_group_of(f) for f in facts if _group_of(f)})
    ask = dict(ask, task=task, twist=twist, anchors=anchors,
               asks=[ask_phrase(f) for f in facts])
    if twist == "unanswerable" and spare_family:
        # The person is asking for the family the sheet *lacks*, and the facts
        # that survive into the reply are not what they asked about. Leaving
        # their anchors in told the writer to name atoms the question is not
        # about, and then refused the question for not naming them.
        ask["asks"] = [FAMILY_WORDS.get(spare_family, "that property")]
        ask["missing_family"] = spare_family
        ask["anchors"] = []
    return Render(statements, reply, turns, ask, skeleton,
                  tuple(f.kind for f in facts), answers, verdict, gloss)


def can_clarify(facts) -> bool:
    """Whether `needs_clarification` is renderable for this draw."""
    return bool(_one_atom(facts))
