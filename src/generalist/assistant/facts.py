"""The unit everything in the assistant pipeline is built from: one computed fact.

A domain's fact sheet is a list of `Fact`s, and every claim a rendered reply makes
is one of their sentences. Nothing here knows which domain the facts came from.
"""

from __future__ import annotations

import re


class Fact:
    """One statement about a subject, and the value that proves it was used.

    ``value`` is the canonical form a check looks for in an answer; ``text`` is
    the same fact in a sentence, which is what a writer is handed. ``family`` is
    what the fact is about (for a molecule, the Tier-A family, the corpus, or one
    of ``smiles`` / ``caption``), and it is what correctness is broken down by.

    ``atoms`` are the indices of the parts of the subject the fact names — atoms,
    for a molecule — written the way the fact's sentence writes them. Empty means
    the fact is about the whole subject. The name is historical and kept because
    it is the serialised key of every batch built so far.
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


def as_facts(facts) -> list:
    """`Fact`s, whether they arrived as objects or as JSON read back from a batch."""
    return [f if isinstance(f, Fact) else Fact.from_json(f) for f in facts]


def fact_field(fact, name: str, default=""):
    """Facts arrive as objects when written and as dicts when read back from
    JSONL. Both are read the same way here."""
    if isinstance(fact, Fact):
        return getattr(fact, name, default)
    return fact.get(name, default)


def clause(fact) -> str:
    """A fact's sentence with its full stop removed, so it can be joined."""
    return (fact.text or "").strip().rstrip(".")


def article(word: str) -> str:
    """"a amide" and "a ether" are what a writer copies verbatim into the set,
    and both appear in v1. The names a sheet uses are a closed list, so the vowel
    test is enough — none of them is a "european"."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def four_grams(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
