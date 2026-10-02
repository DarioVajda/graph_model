"""The molecule domain: small molecules, their RDKit fact sheets, and `mol/assistant`.

Everything here is an assignment or a one-line delegation; the substance lives in
the neighbouring modules —

    vocabulary.py   what the renderer may say about a molecule
    sheet.py        the fact sheet, the canonical SMILES, the held-out language
    verify.py       the verifier rows built before rendered statements use
    prompts.py      the ask, voice and judge system prompts
    checks.py       the turn checks only a molecule needs
    pool.py         the §3 pool `pipeline/build.py` draws from
    graphs.py       a composed row as the graph `mol/assistant` trains on
"""

from __future__ import annotations

import os
import re

from ..domain import Domain, register
from . import vocabulary as V
from .checks import TURN_CHECKS
from .prompts import ASK_SYSTEM, JUDGE_SYSTEM, VOICE_SYSTEM


_HERE = os.path.dirname(os.path.abspath(__file__))


class MoleculeDomain(Domain):
    name = "molecules"

    subject_phrase = "this structure"
    anchor_noun = "atom"
    qualifier_noun = "group"

    anchor_re = re.compile(r"atom \d+(?:\s*\([a-z]{1,2}\))?", re.IGNORECASE)
    anchor_key_re = re.compile(r"atom[_ ]\d+", re.IGNORECASE)
    anchor_index_re = re.compile(r"\batom\s+(\d+)", re.IGNORECASE)

    stop_words = frozenset(("atom", "atoms", "molecule", "compound", "structure",
                            "smallest"))

    gloss = V.GLOSS
    family_words = V.FAMILY_WORDS
    ask_phrases = V.ASK_PHRASES
    count_subjects = V.COUNT_SUBJECTS
    field_kinds = V.FIELD_KINDS
    anchorless_ask = "how many rings it has"

    unanswerable_families = V.UNANSWERABLE_FAMILIES
    off_sheet_families = V.OFF_SHEET_FAMILIES

    #: The canonical SMILES cannot be the pivot of a `check_claim` or a `decide`.
    #: The pivot goes into the person's turn verbatim, as a claim to check or a
    #: constraint to meet, so it has to be something a person could plausibly
    #: assert and something the reply can settle in a few words. The SMILES is
    #: neither: "I believe its canonical SMILES is COc1ccccc1N1C(=O)C2C3c4ccccc4..."
    #: writes the whole molecule into the question and the reply copies it
    #: straight back, so the row is answerable without reading the graph at all.
    #: 88 rows of the final build were this. Refusing it in the renderer rather
    #: than in the sampler is deliberate — `sample_intent` catches the ValueError
    #: and draws a different task, so the molecule still yields an intent.
    unpivotable = frozenset(("smiles",))

    #: How often an intent's pivot is a fact about the whole molecule rather than
    #: about a named atom. It is a statement about what the set teaches rather
    #: than a correction to it: "how many rings does this have" and "is atom 12
    #: aromatic" are different questions, and a set that is three-quarters the
    #: second teaches the second — which is what the first leg's case study
    #: found, every atom-level question answered and most molecule-level ones
    #: missed.
    #:
    #: **Not the same number as the split it produces.** `compare` and `triage`
    #: are defined over atoms and draw before this applies, so they hold the
    #: atom-level share up from underneath. Measured over the 9,491-molecule
    #: pool: 0.00 gives 97.4 % atom-level, 0.42 gives 65.6 / 30.7, and **0.55
    #: gives 56.1 / 39.8** with 4.1 % carrying both, which is the 55 / 40 this is
    #: set for. The build itself came out at 55.9 / 40.3 / 3.8 over 11,900
    #: intents, so the simulation is worth trusting. Re-measure rather than
    #: re-derive if the task weights move.
    whole_subject_pivot = 0.55

    situations_path = os.path.join(_HERE, "situations.json")

    shot_pointers = V.SHOT_POINTERS
    shot_label = "a different molecule"

    ask_system = ASK_SYSTEM
    voice_system = VOICE_SYSTEM
    judge_system = JUDGE_SYSTEM

    turn_checks = TURN_CHECKS

    def qualifier_of(self, fact) -> str:
        return V.group_of(fact)

    def record_qualifier(self, fact) -> str:
        return V.record_qualifier(fact)

    def ask_fallback(self, fact, clause: str):
        return V.ask_fallback(fact, clause)

    def restates(self, one, other) -> bool:
        """Do these two facts make the same claim in two wordings?

        One case, and it is worth the function because it reads as a defect: a
        `ring_size` of 0 and a `ring_membership` of "no" about one atom are the
        same sentence twice — "Atom 1 (Sn) is in no ring. Atom 1 (Sn) is not in a
        ring." A nonzero `ring_size` beside a `ring_membership` of "yes" is not
        redundant, because the size is more than the membership.
        """
        pair = {one.family, other.family}
        if pair != {"ring_size", "ring_membership"}:
            return False
        if set(one.atoms) != set(other.atoms):
            return False
        size = one if one.family == "ring_size" else other
        try:
            return int(size.value) == 0
        except (TypeError, ValueError):
            return False

    def mentions_held_out(self, text: str) -> bool:
        from .sheet import mentions_held_out
        return mentions_held_out(text)

    def states_any(self, answer: str, facts) -> bool:
        from .verify import facts_contained
        return any(facts_contained(answer, facts).values())

    def legacy_verify(self, answer: str, facts, brief) -> dict:
        from .verify import verify
        return verify(answer, facts, brief)

    def source(self, config):
        from .pool import MoleculePool
        return MoleculePool(config)

    def example_builder(self, config):
        from .graphs import example_builder
        return example_builder(config)


MOLECULES = register(MoleculeDomain())
