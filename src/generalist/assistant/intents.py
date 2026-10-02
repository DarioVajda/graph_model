"""The intent: what an example is *about*, declared in data before anything runs.

§9.4's redesign. Both halves of an example descend from one declaration, so they
correspond by construction rather than by inspection — the question is free prose
and the correspondence is structural, which is the opposite arrangement to a
template bank, where the question is constrained and the correspondence is hoped
for.

An intent names six things:

    subject    which structure, and its fact sheet (a molecule and its RDKit
               facts, for the molecule domain)
    task       what the person wants *done* — the axis earlier builds lacked
    twist      none, or a false premise, an unanswerable ask, an underspecified one
    facts      which sheet facts the reply rests on
    situation  who is asking and what they are in the middle of
    style      register, length and format, drawn so as to be satisfiable

Two rules are enforced here rather than checked later, because a filter that
catches them is a filter discovering a build bug:

* **The pivot rule.** A multi-fact draw shares a family or a part with its
  pivot. Asked for several unrelated facts in one answer a writer welds them with
  a false connective ("... Therefore, there are 2 ethers."), and neither that nor
  a bare list is a sentence worth learning.
* **Format satisfiability.** A format is only drawn where the rendered reply can
  satisfy it — "a numbered list" needs two or more statements, "one word" needs
  exactly one and a terse kind. Earlier builds drew the two independently, and an
  impossible brief was the largest remaining defect class in the last hand sample.

Everything a domain decides — how often the pivot is about the whole subject,
which families an unanswerable ask may name, which facts restate each other, who
the people are — comes from the `Domain` passed as ``domain`` (molecules when
nothing is passed). The serialised keys `molecule_id` and `smiles` are the
subject's id and its canonical string, kept under their first names because
every batch built so far carries them.
"""

import json

from .domain import get_domain
from .facts import Fact
from .render import TASKS, can_clarify, render

#: How often each task is drawn. `report` stays the plurality because it is the
#: shape the trunk was trained on and the set is a re-skin, not a new capability;
#: the rest are what turn a lookup set into an instruction set.
TASK_WEIGHTS = {
    "report": 0.26,
    "check_claim": 0.14,
    "compare": 0.12,
    "decide": 0.12,
    "fill_record": 0.10,
    "explain": 0.10,
    "summarise": 0.08,
    "triage": 0.08,
}

#: How often a twist is drawn, given a task that accepts it. A quarter of the set
#: carries one. Below that they are noise the fork cannot learn from; far above
#: it and the set stops being about molecules.
TWIST_RATE = 0.25

REGISTERS = ("plain", "technical", "casual", "formal", "terse professional")
LENGTHS = ("one sentence", "two or three sentences", "as short as possible")

#: Tasks whose ask has a single scalar answer, and so can be answered in one
#: word. `triage` asks which of several qualify, `compare` asks for two things at
#: once, and `summarise`, `explain` and `fill_record` all ask for a shape.
#:
#: **`decide` is not on this list, and the reason generalises.** A `decide` ask
#: always poses two questions — what the value is, and whether it meets the
#: constraint — and those have two different answers. One word can carry only one
#: of them, and the render gives the verdict, so the word is the answer to the
#: question that was not asked first. 144 rows of the final build read "whether
#: atom 6 (C) is part of a halogen" → "Yes" over a statement saying it is *not*:
#: the verdict's polarity is anti-correlated with the value's whenever the
#: constraint runs against it, which makes the single word maximally misleading.
#: A narrower guard that excluded only `decide` over a false premise caught 92 of
#: these and left the rest; the number of questions is the rule, not the twist.
#:
#: `check_claim` stays. It poses one question and one word answers it — "No",
#: "Three", "Six" — the correction included.
ONE_WORD_TASKS = frozenset(("report", "check_claim"))


#: Formats, with the condition each one needs from the rendered reply. The
#: condition is checked against the render, so an unsatisfiable brief cannot be
#: drawn — rule 4 of §9.4, made mechanical.
FORMATS = {
    # A render that declared a skeleton *is* a record: `_fill_record` writes its
    # reply as JSON, so a prose brief over one asked for a shape the render had
    # already committed against, and two of the three prose `fill_record` rows in
    # the third smoke build answered "Does this contain a nitrile?" with
    # `{"fg_presence_nitrile": false}`. Rule 4 again — the format axis and the
    # shape of the reply cannot both be free.
    "prose": lambda r: r.skeleton is None,
    # One word is only answerable where the whole content *is* one word: a single
    # yes/no or count statement, and an ask that has an answer at all. A refusal
    # cannot be one word, and neither can a sentence. Nor can a task whose ask is
    # plural or open — "which of these qualify" answered in one word is a
    # question and an answer that do not fit each other, which is the defect the
    # format axis keeps producing.
    "one word": lambda r: (len(r.statements) == 1 and r.ask.get("answerable")
                           and r.kinds[:1] in (("yesno",), ("count",))
                           and r.ask.get("task") in ONE_WORD_TASKS),
    "a numbered list": lambda r: len(r.statements) >= 2 and not r.skeleton,
    "a bulleted list": lambda r: len(r.statements) >= 2 and not r.skeleton,
    "JSON matching the given schema": lambda r: r.skeleton is not None,
    "lead with the answer, then the detail": lambda r: (len(r.statements) >= 2
                                                        and not r.skeleton),
}

#: Formats that answer by *position*: the n-th item answers the n-th statement,
#: and it answers with a value rather than a sentence. Checking one of these as
#: prose was a third of the first smoke build's rejections and nearly all of them
#: were correct rows — "1. No\n2. No" states both statements it was given.
POSITIONAL_FORMATS = frozenset(("a numbered list", "a bulleted list"))

#: Formats under which a statement is preserved by its *value* rather than by its
#: sentence. §9.4 rule 4: a format that suppresses prose and a check that requires
#: prose cannot both be free variables, so the render declares which it is.
TERSE_FORMATS = frozenset(("one word", "JSON matching the given schema"))

#: Formats with no sentences to count, so no sentence count may be asked of them.
STRUCTURED_FORMATS = frozenset(("one word", "JSON matching the given schema",
                                "a numbered list", "a bulleted list"))


def load_situations(path: str = None, domain=None) -> list:
    """The people who ask: `path`, or the domain's `situations_path`."""
    with open(path or get_domain(domain).situations_path) as handle:
        return json.load(handle)["situations"]


class Intent:
    """One declared example. Serialises to a batch row and back."""

    __slots__ = ("molecule_id", "smiles", "task", "twist", "facts", "situation",
                 "style", "spare_family", "role")

    def __init__(self, molecule_id, smiles, task, twist, facts, situation,
                 style, spare_family=None, role="train"):
        self.molecule_id = molecule_id
        self.smiles = smiles
        self.task = task
        self.twist = twist
        self.facts = list(facts)
        self.situation = situation
        self.style = style
        self.spare_family = spare_family
        self.role = role

    def to_json(self) -> dict:
        return {"molecule_id": self.molecule_id, "smiles": self.smiles,
                "task": self.task, "twist": self.twist,
                "facts": [f.to_json() for f in self.facts],
                "situation": self.situation, "style": self.style,
                "spare_family": self.spare_family, "role": self.role}

    @classmethod
    def from_json(cls, payload: dict) -> "Intent":
        return cls(payload["molecule_id"], payload["smiles"], payload["task"],
                   payload["twist"],
                   [Fact.from_json(f) for f in payload["facts"]],
                   payload["situation"], payload["style"],
                   payload.get("spare_family"), payload.get("role", "train"))

    def cell(self) -> str:
        """The coverage cell this intent falls in, for the build's table."""
        return f"{self.task}/{self.twist}/{self.style['format']}"


def _weighted(rng, weights: dict):
    total = sum(weights.values())
    draw = rng.random() * total
    running = 0.0
    for key, weight in weights.items():
        running += weight
        if draw <= running:
            return key
    return next(iter(weights))


def _pivot_neighbours(sheet, pivot, domain=None) -> list:
    """Sheet facts that may share an answer with `pivot`: same family, or a
    part in common. The rule earlier builds learned the hard way. A fact the
    domain says restates the pivot (`Domain.restates`) is not a neighbour."""
    domain = get_domain(domain)
    out = []
    for fact in sheet:
        if fact is pivot:
            continue
        if domain.restates(fact, pivot):
            continue
        if fact.family == pivot.family:
            out.append(fact)
        elif pivot.atoms and set(fact.atoms) & set(pivot.atoms):
            out.append(fact)
    return out


def _draw_facts(sheet, task, rng, domain=None):
    """A pivot and, where the task wants more, neighbours that share with it."""
    domain = get_domain(domain)
    low, high = TASKS[task][1]
    if not sheet:
        return []
    pool = list(sheet)
    rng.shuffle(pool)

    if task == "compare":
        # Two facts of one family about different atoms, or nothing.
        by_family = {}
        for fact in pool:
            by_family.setdefault(fact.family, []).append(fact)
        for family, group in by_family.items():
            distinct = {tuple(f.atoms): f for f in group if f.atoms}
            if len(distinct) >= 2:
                picked = list(distinct.values())[:2]
                return picked
        return []

    if task == "triage":
        by_family = {}
        for fact in pool:
            if fact.kind == "yesno" and fact.atoms:
                by_family.setdefault(fact.family, []).append(fact)
        for family, group in by_family.items():
            distinct = list({tuple(f.atoms): f for f in group}.values())
            if len(distinct) >= low:
                return distinct[:min(high, len(distinct))]
        return []

    # **The pivot's scope is drawn rather than inherited from the sheet.** A
    # sheet carries several facts per sampled atom against a handful about the
    # whole molecule, so it runs about 81 % atom-level, and a pivot taken off
    # the shuffled pool inherits that: 74.9 % of the first set's rows were
    # atom-level only against 20.1 % molecule-level. The first leg's case study
    # then answered every atom-level question correctly and most molecule-level
    # ones wrongly, which is the same split read twice. `compare` and `triage`
    # are exempt because they are defined over parts and return above. The
    # share is the domain's `whole_subject_pivot`.
    whole = [f for f in pool if not f.atoms]
    parts = [f for f in pool if f.atoms]
    scoped = pool
    if whole and parts:
        scoped = whole if rng.random() < domain.whole_subject_pivot else parts

    pivot = scoped[0]
    if task == "decide":
        # A `decide` verdict is a predicate over the pivot's value, and the only
        # values there are predicates for are a yes/no and a count. Drawn on a
        # SMILES or a caption the constraint degenerates to "I need that on
        # file" and the verdict is yes every time, which is a cell with one
        # answer in it.
        scalar = [f for f in scoped if f.kind in ("yesno", "count")]
        if not scalar:
            # The drawn scope holds no predicate to build a constraint over, so
            # take one from the other rather than dropping the intent: a scope
            # balance is worth less than the cell it would empty.
            scalar = [f for f in pool if f.kind in ("yesno", "count")]
        if not scalar:
            return []
        pivot = scalar[0]
    want = rng.randint(low, high)
    if want == 1:
        return [pivot]
    neighbours = _pivot_neighbours(sheet, pivot, domain)
    rng.shuffle(neighbours)
    # Filtering against the pivot is not enough: two neighbours can restate each
    # other without either restating the pivot.
    drawn = [pivot]
    for fact in neighbours:
        if len(drawn) >= want:
            break
        if any(domain.restates(fact, chosen) for chosen in drawn):
            continue
        drawn.append(fact)
    return drawn


def _draw_twist(task, sheet, facts, rng, domain=None):
    """A twist the render can actually carry, or `none`.

    Feasibility is settled here rather than by catching the renderer's
    complaint, because falling back on the exception drops the whole task for
    that subject instead of just the twist.
    """
    domain = get_domain(domain)
    allowed = [t for t in TASKS[task][2] if t != "none"]
    if not allowed or rng.random() >= TWIST_RATE:
        return "none", None
    rng.shuffle(allowed)
    carried = {f.family for f in sheet}
    for twist in allowed:
        if twist == "needs_clarification":
            if can_clarify(facts, domain):
                return twist, None
        elif twist == "unanswerable":
            # Both pools: the sheet families this subject happens to lack, and
            # the properties no sheet ever carries. Drawing from the first alone
            # gave `caption` 31 times in 32, because a sheet that lacks anything
            # else is rare — so every unanswerable turn was the same sentence.
            spare = [f for f in domain.unanswerable_families if f not in carried]
            spare += list(domain.off_sheet_families)
            if spare:
                return twist, spare[rng.randrange(len(spare))]
        else:
            return twist, None
    return "none", None


def _draw_style(rendered, rng) -> dict:
    """Register, length and format — drawn so the three cannot contradict.

    Length is a sentence count, and a sentence count means nothing under a
    format that has no sentences: "one sentence, as a numbered list" and "two or
    three sentences, as JSON" are briefs no reply can satisfy, which is rule 4
    again on a different axis. Under those formats the length is fixed to the
    one thing that still makes sense.
    """
    usable = [name for name, ok in FORMATS.items() if ok(rendered)]
    fmt = usable[rng.randrange(len(usable))]
    length = ("as short as possible" if fmt in STRUCTURED_FORMATS
              else LENGTHS[rng.randrange(len(LENGTHS))])
    return {"register": REGISTERS[rng.randrange(len(REGISTERS))],
            "length": length, "format": fmt}


def sample_intent(molecule_id, smiles, sheet, situations, rng, role="train",
                  domain=None):
    """One intent over one subject, or None if the sheet cannot support any.

    Returns `(intent, render)` so the caller never re-renders: the render is what
    decides which formats are drawable, so the two are produced together.
    """
    domain = get_domain(domain)
    tasks = dict(TASK_WEIGHTS)
    for _ in range(len(tasks)):
        if not tasks:
            return None
        task = _weighted(rng, tasks)
        facts = _draw_facts(sheet, task, rng, domain)
        low, high = TASKS[task][1]
        if not low <= len(facts) <= high:
            tasks.pop(task, None)         # this sheet cannot support the task
            continue
        twist, spare = _draw_twist(task, sheet, facts, rng, domain)
        try:
            rendered = render(facts, task, twist, rng, spare, domain)
        except (ValueError, KeyError):
            tasks.pop(task, None)
            continue
        situation = situations[rng.randrange(len(situations))]
        style = _draw_style(rendered, rng)
        intent = Intent(molecule_id, smiles, task, twist, facts, situation,
                        style, spare, role)
        return intent, rendered
    return None
