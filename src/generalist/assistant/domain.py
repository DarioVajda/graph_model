"""What a domain supplies to the assistant pipeline, and the registry that finds it.

The pipeline — intent, render, the two writer passes, the judge, accept, compose —
is the same whatever the subject is. What changes between subjects is a closed
list of things, and `Domain` is that list:

* **how a fact names a part of its subject** — `anchor_re` finds "atom 14 (C)"
  in a molecule fact's sentence, `anchor_noun` is what a clarifying question
  calls it ("Which atom do you mean?"), and `qualifier_of` pulls out the other
  thing a fact can be about (a functional group, for a molecule);
* **the vocabulary the renderer may use** — `ask_phrases`, `count_subjects`,
  `family_words`, `gloss`, `field_kinds`. All of it closed and hand written,
  because the renderer is the only place a claim enters an answer;
* **the draw** — which families an `unanswerable` ask may name, which may not
  pivot a claim, how often the pivot is about the whole subject, and which pairs
  of facts are one claim in two wordings;
* **the people** — `situations_path`, a JSON file of personas and contexts;
* **the prompts** — the system prompts of the ask, voice and judge passes;
* **the checks only a domain can make** — `turn_checks`, `mentions_held_out`,
  and `states_any` for the few-shot copy rule;
* **the data** — `source` loads the subject pool and builds each fact sheet, and
  `example_builder` turns a composed row into the graph a model trains on.

A domain is a subclass that fills these in and calls `register`. `get_domain`
resolves a name, an instance, or None — the last meaning molecules, which was the
only domain when every function here took its vocabulary as a module constant,
and which is still what every caller that passes nothing gets.
"""

from __future__ import annotations

import importlib


class Unbuildable(Exception):
    """A row `example_builder`'s ``build`` cannot turn into a graph. The message
    is the reason, and goes into the report as it is."""


class Domain:
    """The domain contract. Attributes are data; methods are hooks.

    The defaults describe a domain whose facts name no parts and carry no
    qualifiers, so a new domain only overrides what it actually has.
    """

    name = ""

    #: What the subject and its parts are called in rendered text. `subject_phrase`
    #: closes a decline ("that isn't something I have for this structure").
    subject_phrase = "this subject"
    anchor_noun = "part"
    #: What an ask calls a qualifier it cannot name ("whether it contains that
    #: group"). See `qualifier_of`.
    qualifier_noun = "thing"

    #: One reference to a part, as a fact sentence writes it. `group(0)` is the
    #: reference. None means facts never name a part.
    anchor_re = None
    #: The same reference as a JSON key spells it (`ring_size_atom_5`), so a
    #: rendered schema in the person's turn is not read as an answer.
    anchor_key_re = None
    #: The index inside a reference, as `group(1)`. A reply may name a part by
    #: index alone ("C14"), so the index is what has to survive.
    anchor_index_re = None

    #: Words too common in this domain's sentences to carry a statement's
    #: identity, on top of the English stop words the accept pass already has.
    stop_words = frozenset()

    gloss: dict = {}
    family_words: dict = {}
    ask_phrases: dict = {}
    count_subjects: dict = {}
    #: family -> the JSON type its field takes in a `fill_record` schema.
    field_kinds: dict = {}
    #: The ask phrase used when a template wants a part and the fact names none.
    anchorless_ask = None

    #: Families a sheet *may* carry but this subject's happens not to.
    unanswerable_families = ()
    #: Properties no sheet ever carries.
    off_sheet_families = ()
    #: Families that cannot be the claim of a `check_claim` or a `decide`.
    unpivotable = frozenset()
    #: How often a pivot is a fact about the whole subject rather than a part.
    whole_subject_pivot = 0.5

    situations_path = None

    shot_pointers = ("Worked examples are attached — see those examples.",)
    shot_label = "a different example"

    ask_system = ""
    voice_system = ""
    judge_system = ""

    #: ``(reason, check)`` pairs run over the person's turn in the accept pass,
    #: in order. ``check(turn, intent, licensed)`` returns None to pass, or the
    #: detail string for the rejection log. ``licensed`` is the text the render
    #: itself put in play — its claim, statements and reply.
    turn_checks = ()

    # ── hooks ────────────────────────────────────────────────────────────────

    def anchor_ref(self, text: str) -> str:
        """The first part reference in ``text``, or ""."""
        if self.anchor_re is None:
            return ""
        match = self.anchor_re.search(text or "")
        return match.group(0) if match else ""

    def is_anchor_ref(self, text: str) -> bool:
        """Is ``text`` a part reference, as opposed to a qualifier?"""
        return self.anchor_re is not None and bool(self.anchor_re.match(text or ""))

    def anchor_index(self, text: str):
        """The index of the first part reference in ``text``, or None."""
        if self.anchor_index_re is None:
            return None
        match = self.anchor_index_re.search(text or "")
        return match.group(1) if match else None

    def strip_anchors(self, text: str) -> str:
        """``text`` with every part reference blanked, so an index is not read
        as a value."""
        if self.anchor_re is not None:
            text = self.anchor_re.sub(" ", text)
        return text

    def qualifier_of(self, fact) -> str:
        """The named thing a fact is about besides a part, or ""."""
        return ""

    def record_qualifier(self, fact) -> str:
        """What tells this fact's `fill_record` key apart from its family's
        other facts, where it names no part."""
        return self.qualifier_of(fact)

    def ask_fallback(self, fact, clause: str):
        """An ask phrase for a family `ask_phrases` has no entry for, or None to
        take the generic fallback (the sentence with its polarity stripped)."""
        return None

    def restates(self, one, other) -> bool:
        """Do these two facts make the same claim in two wordings?"""
        return False

    def mentions_held_out(self, text: str) -> bool:
        """Does this text talk about something the evaluation holds out?"""
        return False

    def states_any(self, answer: str, facts) -> bool:
        """Does ``answer`` state any of ``facts``? The few-shot copy rule."""
        return False

    def legacy_verify(self, answer: str, facts, brief) -> dict:
        """The verifier for rows built before rendered statements existed."""
        raise NotImplementedError(f"{self.name}: no legacy verifier")

    def source(self, config):
        """The subject pool for `pipeline.build`. See `molecules.pool`."""
        raise NotImplementedError(f"{self.name}: no subject source")

    def example_builder(self, config):
        """``(build, tokenizer_name)`` for `pipeline.graphs`, where
        ``build(row)`` returns ``(graph, n_shots)`` — the graph a composed row
        trains on, and how many demonstrations made it in — or raises
        `Unbuildable`."""
        raise NotImplementedError(f"{self.name}: no example builder")


_REGISTRY: dict = {}

#: Domains importable by name before anything has registered them, as modules
#: relative to this package. Importing one registers it. A domain kept outside
#: this package registers itself when its own module is imported.
_MODULES = {"molecules": ".molecules"}

DEFAULT = "molecules"


def register(domain: Domain) -> Domain:
    _REGISTRY[domain.name] = domain
    return domain


def get_domain(domain=None) -> Domain:
    """A `Domain` from a name, an instance, or None (molecules)."""
    if isinstance(domain, Domain):
        return domain
    name = domain or DEFAULT
    if name not in _REGISTRY and name in _MODULES:
        importlib.import_module(_MODULES[name], package=__package__)
    if name not in _REGISTRY:
        raise KeyError(f"unknown assistant domain {name!r}; "
                       f"known: {domain_names()}")
    return _REGISTRY[name]


def domain_names() -> list:
    return sorted(set(_REGISTRY) | set(_MODULES))
