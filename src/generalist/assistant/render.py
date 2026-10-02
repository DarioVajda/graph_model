"""The answer's content, rendered from the fact sheet — never written by a model.

§9.4's rule in its strong form. Earlier builds handed a model a fact sheet and
asked it to compose a reply; three rounds of filters then tried to establish
whether what came back was true. This module composes the reply, in a plain
register, out of statements the domain computed, and the model's whole remaining
job is to change that reply's voice. Invented chemistry stops being a category
that can occur, because no chemistry is generated.

Two things come out of a render and they have different jobs:

* ``statements`` is the list of claims the reply makes. It is what verification
  checks survived the re-voicing, and nothing else in the pipeline may add to it.
* ``reply`` is those statements welded into a plain-register answer shaped by the
  task. It is what the writer is handed.

Everything here is deterministic given (facts, task, twist, domain). Nothing in
this module imports a model, and nothing in it is allowed to say anything the fact
sheet does not license — which is why the words it may add come from the domain's
closed, hand-written vocabulary (`Domain.gloss`, `family_words`, `ask_phrases`,
`count_subjects`) rather than from anything generated. What is left here is the
English the tasks are made of, and it is the same for every domain.
"""

import json
import re

from .domain import get_domain
from .facts import article, clause as _clause


#: The twists, and which tasks each one is available on. A twist changes what the
#: reply has to *do*, which is the axis the earlier builds had no room for.
TWISTS = ("none", "false_premise", "unanswerable", "needs_clarification")


def _count_subject(fact, domain) -> str:
    """The counted thing, named, for a constraint stated mid-sentence.

    Falls back to "that number" wherever the name cannot be filled in, which is
    the old wording and no worse than it: an unnamed number is ambiguous beside
    a second count, never wrong.
    """
    template = domain.count_subjects.get(f"{fact.family}/{fact.kind}") \
        or domain.count_subjects.get(fact.family)
    if template is None:
        return "that number"
    anchor, qualifier = _anchor_ref(fact, domain), domain.qualifier_of(fact)
    if ("{anchor}" in template and not anchor) or ("{qualifiers}" in template
                                                   and not qualifier):
        return "that number"
    return template.format(anchor=anchor,
                           qualifiers=_plural(qualifier) if qualifier else "")


def _plural(word: str) -> str:
    if word.endswith("s"):
        return word
    if re.search(r"(?:ch|sh|x|z)$", word):
        return word + "es"
    if re.search(r"[^aeiou]y$", word):
        return word[:-1] + "ies"
    return word + "s"


_QUALIFIER_SLOTS = ("{qualifier}", "{a_qualifier}", "{qualifiers}")


def ask_phrase(fact, domain=None) -> str:
    """What this fact answers, phrased as the thing a person would ask for.

    Built from the sheet's own wording with the polarity removed, so it names the
    same part and the same qualifier as the fact and commits to nothing about the
    value. A family with no entry falls back to the domain's `ask_fallback`, and
    then to the sentence with its negation and its number stripped, which is
    coarse but still says nothing.
    """
    domain = get_domain(domain)
    anchor = _anchor_ref(fact, domain)
    qualifier = domain.qualifier_of(fact)
    template = domain.ask_phrases.get(f"{fact.family}/{fact.kind}") \
        or domain.ask_phrases.get(fact.family)
    if template is None:
        fallback = domain.ask_fallback(fact, _clause(fact))
        if fallback is not None:
            return fallback
        stripped = re.sub(r"\b(?:not|no)\b", "", _clause(fact))
        stripped = re.sub(r"\b\d+\b", "", stripped)
        return re.sub(r"\s{2,}", " ", stripped).strip()
    if "{anchor}" in template and not anchor:
        return domain.anchorless_ask or template
    unnamed = f"that {domain.anchor_noun}"
    if not qualifier and any(slot in template for slot in _QUALIFIER_SLOTS):
        noun = domain.qualifier_noun
        return template.format(anchor=anchor or unnamed,
                               qualifier=f"that {noun}",
                               a_qualifier=f"that {noun}",
                               qualifiers=f"of those {_plural(noun)}")
    return template.format(
        anchor=anchor or unnamed, qualifier=qualifier,
        qualifiers=_plural(qualifier) if qualifier else "",
        a_qualifier=f"{article(qualifier)} {qualifier}" if qualifier else "")


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


def _lower_first(text: str) -> str:
    """A clause put mid-sentence. Atom facts already read lower-case; the
    molecule-level ones start "It", which reads wrong after "if"."""
    return text[0].lower() + text[1:] if text else text


def _anchor_ref(fact, domain) -> str:
    """The part a fact is about, as it is written on the sheet.

    Searched anywhere in the sentence rather than anchored at the front: a
    `ring_size` fact reads "The smallest ring containing atom 7 (C) has 6 atoms",
    and anchoring found nothing there, which is how a clarification turn came to
    ask "Which atom do you mean?" and be answered "The atom."
    """
    return domain.anchor_ref(fact.text)


def _one_anchor(facts, domain) -> str:
    """The single part every fact is about, or "" if they differ or there is none."""
    refs = {_anchor_ref(f, domain).lower() for f in facts}
    if len(refs) != 1:
        return ""
    ref = refs.pop()
    return _anchor_ref(facts[0], domain) if ref else ""


def _negate(fact, domain) -> str:
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
            qualifier = domain.qualifier_of(fact)
            word = article(qualifier) if qualifier else "a"
            return re.sub(r"\bcontains no\b", f"contains {word}", text,
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


def _polarised(fact, positive: bool, domain) -> str:
    """The fact's predication in the asked-for polarity, whatever its value.

    `_clause` renders a fact as the sheet states it, so it already carries the
    *value's* polarity, and `_negate` flips that. Neither is "the positive form"
    on its own, and reading them as if they were is what made a `decide`
    constraint contradict the statement it was decided against: on a fact whose
    value is "no", "I only want it if X" came out as "I only want it if not X"
    while the verdict had been computed for "X", and the rendered reply then read
    "Yes, that one qualifies. Atom 9 (C) is not in an aromatic ring."
    """
    return (_clause(fact) if (fact.value == "yes") == positive
            else _negate(fact, domain))


# --------------------------------------------------------------------------
# the tasks
# --------------------------------------------------------------------------
#
# Every task renderer takes (facts, twist, rng, spare_family, domain) and returns
# (statements, reply, ask). `render` resolves the domain before calling one.

def _report(facts, twist, rng, spare_family, domain):
    if twist == "unanswerable":
        words = domain.family_words.get(spare_family, "that property")
        anchor = facts[0] if facts else None
        known = _clause(anchor) if anchor is not None else ""
        statements = [f"{_sentence(known)}." ] if known else []
        reply = (f"I can't tell you {words} — that isn't something I have for "
                 f"{domain.subject_phrase}.")
        if known:
            reply += f" What I do have: {known}."
        return statements, reply, {"wants": words, "answerable": False,
                                   "_answers": [answer_token(anchor)]
                                   if known else []}

    statements = [f"{_sentence(_clause(f))}." for f in facts]
    reply = " ".join(statements)
    return statements, reply, {"wants": "the value", "answerable": True,
                               "_answers": [answer_token(f) for f in facts]}


def _compare(facts, twist, rng, spare_family, domain):
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


def _check_claim(facts, twist, rng, spare_family, domain):
    pivot = facts[0]
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    if twist == "false_premise":
        claim = _negate(pivot, domain)
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
def _decide(facts, twist, rng, spare_family, domain):
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
                      f"{_lower_first(_polarised(pivot, wants_yes, domain))}")
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
        constraint = (f"I need {_count_subject(pivot, domain)} to be at least "
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
        ask["claim"] = _sentence(_negate(pivot, domain)) + "."
        ask["claim_is_true"] = False
        reply = f"That isn't right. {' '.join(statements)} {verdict}"
    else:
        reply = f"{verdict} {' '.join(statements)}"
    return statements, reply, ask


def _fill_record(facts, twist, rng, spare_family, domain):
    skeleton, payload = {}, {}
    for i, fact in enumerate(facts):
        key = _record_key(fact, i, domain)
        # Last resort, so that a qualifier I have not thought of cannot silently
        # merge two statements into one field.
        while key in skeleton:
            key = f"{key}_{i}"
        skeleton[key] = _kind_word(fact)
        payload[key] = _record_value(fact)
    statements = [f"{_sentence(_clause(f))}." for f in facts]

    if twist == "unanswerable":
        # The record asks for a field this subject's sheet has no fact for. The
        # honest fill is a null and a sentence saying why, and the null is not a
        # claim, so it does not join the statements.
        key = spare_family.replace("/", "_")
        skeleton[key] = domain.field_kinds.get(spare_family, "string")
        payload[key] = None
        words = domain.family_words.get(spare_family, "that field")
        reply = (json.dumps(payload, sort_keys=True)
                 + f"\n\n{key} is null: I don't have {words} for "
                   f"{domain.subject_phrase}.")
        return statements, reply, {"wants": "a filled record",
                                   "answerable": False, "missing": key,
                                   "_skeleton": skeleton,
                                   "_answers": [answer_token(f)
                                                for f in facts]}

    return (statements, json.dumps(payload, sort_keys=True),
            {"wants": "a filled record", "answerable": True,
             "_skeleton": skeleton,
             "_answers": [answer_token(f) for f in facts]})


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _record_key(fact, index: int, domain) -> str:
    """A field name that distinguishes this fact from the others in the record.

    The part's index does it where there is one. Where there is not, the family
    alone may not — a molecule has an `fg_count` per functional group — and the
    domain's `record_qualifier` says what does.
    """
    anchors = "_".join(str(a) for a in fact.atoms)
    base = fact.family.replace("/", "_")
    if anchors:
        return f"{base}_{domain.anchor_noun}_{anchors}"
    qualifier = domain.record_qualifier(fact)
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


def _explain(facts, twist, rng, spare_family, domain):
    pivot = facts[0]
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    gloss = domain.gloss.get(pivot.family)
    reply = " ".join(statements)
    ask = {"wants": "an explanation", "answerable": True,
           "_answers": [answer_token(f) for f in facts]}
    if gloss:
        # The gloss is the whole difference between `explain` and `report`, and
        # it is *not* a statement: it asserts nothing about this subject, so a
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


def _summarise(facts, twist, rng, spare_family, domain):
    statements = [f"{_sentence(_clause(f))}." for f in facts]
    reply = " ".join(statements)
    return statements, reply, {"wants": "a short summary", "answerable": True,
                               "_answers": [answer_token(f) for f in facts]}


def _predicate_of(fact, domain) -> str:
    """A fact's sentence with its part reference removed — "is in a ring".

    Triage groups parts that share an answer, and a grouped sentence has to be
    built from the sheet's own predicate rather than from a phrasing invented
    here. If the reference cannot be found the fact is not groupable and the
    caller falls back to stating it on its own.
    """
    text = _clause(fact)
    ref = _anchor_ref(fact, domain)
    if not ref or not text.lower().startswith(ref.lower()):
        return ""
    return text[len(ref):].strip()


def _grouped(group, domain) -> str:
    """One sentence covering several parts that share a predicate."""
    refs = [_anchor_ref(f, domain) for f in group]
    predicate = _predicate_of(group[0], domain)
    if len(refs) == 1:
        return f"{_sentence(refs[0])} {predicate}."
    joined = ", ".join(refs[:-1]) + f" and {refs[-1]}"
    plural = re.sub(r"^is\b", "are", predicate)
    return f"{_sentence(joined)} {plural}."


def _triage(facts, twist, rng, spare_family, domain):
    holds = [f for f in facts if f.value == "yes"]
    misses = [f for f in facts if f.value != "yes"]
    groupable = all(_predicate_of(f, domain) for f in facts) and len(
        {_predicate_of(f, domain) for f in holds}) <= 1 and len(
        {_predicate_of(f, domain) for f in misses}) <= 1
    if not groupable:
        statements = [f"{_sentence(_clause(f))}." for f in facts]
        return statements, " ".join(statements), {
            "wants": "which ones qualify", "answerable": True,
            "_answers": [answer_token(f) for f in facts]}

    groups = [group for group in (holds, misses) if group]
    statements = [_grouped(group, domain) for group in groups]
    return statements, " ".join(statements), {
        "wants": "which ones qualify", "answerable": True,
        "_answers": [answer_token(group[0]) for group in groups]}


#: task -> (renderer, how many facts it needs, which twists it accepts)
TASKS = {
    "report":      (_report,      (1, 3), ("none", "unanswerable",
                                           "needs_clarification")),
    # No clarification on `compare`: its two facts are about two parts by
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


def render(facts, task: str, twist: str, rng, spare_family: str = None,
           domain=None) -> Render:
    """The reply to an intent, as statements plus a plain-register answer.

    `facts` are the sheet facts the intent declared, already drawn to the task's
    arity by the sampler. `spare_family` is only read for an `unanswerable` ask:
    it is a family this subject's sheet does not carry.

    A pivot in the domain's `unpivotable` families is refused for `check_claim`
    and `decide`, because the pivot goes into the person's turn verbatim, as a
    claim to check or a constraint to meet (see `MoleculeDomain.unpivotable`).
    `sample_intent` catches the ValueError and draws a different task, so the
    subject still yields an intent.
    """
    domain = get_domain(domain)
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}")
    fn, (low, high), allowed = TASKS[task]
    if twist not in allowed:
        raise ValueError(f"task {task!r} does not take twist {twist!r}")
    if not low <= len(facts) <= high:
        raise ValueError(f"task {task!r} wants {low}-{high} facts, "
                         f"got {len(facts)}")
    if task in ("check_claim", "decide") and facts[0].family in domain.unpivotable:
        raise ValueError(f"task {task!r} cannot pivot on {facts[0].family!r}")

    statements, reply, ask = fn(facts, twist, rng, spare_family, domain)
    skeleton = ask.pop("_skeleton", None)
    answers = ask.pop("_answers", None)
    verdict = ask.pop("_verdict", None)
    gloss = ask.pop("_gloss", None)

    if twist == "needs_clarification":
        # The person's first turn is underspecified; the assistant asks which
        # part; the person names it. The clarifying question is rendered, not
        # written, so the thing asked for is exactly the thing the facts settle —
        # which only works when the facts are all about one part. `can_clarify`
        # is what the sampler checks before drawing this twist.
        anchor = _one_anchor(facts, domain)
        if not anchor:
            raise ValueError("needs_clarification wants one "
                             f"{domain.anchor_noun} across its facts")
        turns = [("person", None),            # the writer fills this
                 ("assistant", f"Which {domain.anchor_noun} do you mean?"),
                 ("person", f"{_sentence(anchor)}."),
                 ("assistant", reply)]
        ask = dict(ask, underspecified=True, anchor=anchor)
    else:
        turns = [("person", None), ("assistant", reply)]

    anchors = sorted({_anchor_ref(f, domain) for f in facts
                      if _anchor_ref(f, domain)}
                     | {domain.qualifier_of(f) for f in facts
                        if domain.qualifier_of(f)})
    ask = dict(ask, task=task, twist=twist, anchors=anchors,
               asks=[ask_phrase(f, domain) for f in facts])
    if twist == "unanswerable" and spare_family:
        # The person is asking for the family the sheet *lacks*, and the facts
        # that survive into the reply are not what they asked about. Leaving
        # their anchors in told the writer to name parts the question is not
        # about, and then refused the question for not naming them.
        ask["asks"] = [domain.family_words.get(spare_family, "that property")]
        ask["missing_family"] = spare_family
        ask["anchors"] = []
    return Render(statements, reply, turns, ask, skeleton,
                  tuple(f.kind for f in facts), answers, verdict, gloss)


def can_clarify(facts, domain=None) -> bool:
    """Whether `needs_clarification` is renderable for this draw."""
    return bool(_one_anchor(facts, get_domain(domain)))
