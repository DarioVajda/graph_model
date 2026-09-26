"""The assistant set's facts, briefs and verifier (`MOLECULE_GENERALIST.md` §9.4).

The verifier is the task's ``verify`` and the correctness metric at once, so its
two failure directions matter equally and are tested as a pair: an answer that
states a fact in a register nobody anticipated must pass, and an answer that
quotes a *different* value must fail. A verifier that only rejects is a set that
reads like a form; one that only accepts is a correctness number that means
nothing.
"""

from __future__ import annotations

import collections
import random

import pytest
from rdkit import Chem

from src.generalist.assistant import (
    Fact, LEAKY_FORMS, SHOT_COUNTS, SHOT_FRACTION, _subject_of,
    answer_invents_atom,
    brief_key, claims_connectivity, demo_leaks_target, draw_brief,
    question_leaks, question_widens_scope,
    draw_shot_count, fact_polarity, fact_sheet, facts_contained, format_met,
    endpoint_words, four_grams, jaccard, json_key_misnames, mentions_held_out,
    mentions_the_sheet, opener_contradicts, question_changes_subject,
    question_mismatches_facts, question_text, select_facts,
    shot_candidates, shot_molecules, shot_text, ungrounded_claims,
    unsupported_claims, verify,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def _families(facts):
    return {f.family for f in facts}


# ─────────────────────────────────────────────────────────────────────────────
# Fact sheets
# ─────────────────────────────────────────────────────────────────────────────

class TestFactSheet:

    def test_it_states_the_counts_rdkit_computes(self):
        facts = fact_sheet(_mol(ASPIRIN), rng=random.Random(0))
        by_family = {}
        for fact in facts:
            by_family.setdefault(fact.family, []).append(fact)
        assert by_family["ring_count"][0].value == "1"
        assert by_family["stereo_potential"][0].value == "0"
        # Aspirin's ring is aromatic, and it carries a carboxylic acid and an ester
        # oxygen — the group facts are counts, so they are checkable numbers.
        assert any(f.value == "1" for f in by_family["aromatic_ring"])

    def test_the_held_out_families_are_never_in_a_sheet(self):
        """§4 lives or dies here: a fact sheet that quoted `longest_chain` would
        spend the held-out measurement through the back door."""
        for smiles in (ASPIRIN, CAFFEINE):
            facts = fact_sheet(_mol(smiles), rng=random.Random(1))
            assert not (_families(facts) & {"bond_path", "longest_chain"})
            assert not any(mentions_held_out(f.text) for f in facts)

    def test_tier_b_labels_are_stated_with_the_endpoint_in_words(self):
        facts = fact_sheet(_mol(ASPIRIN), rng=random.Random(0),
                           tier_b=[("bbbp", "p_np", True),
                                   ("bace", "Class", False)])
        stated = {f.family: f for f in facts if f.family.startswith("tier_b/")}
        assert stated["tier_b/bbbp"].value == "yes"
        assert "blood-brain barrier" in stated["tier_b/bbbp"].text
        assert stated["tier_b/bace"].value == "no"

    def test_the_smiles_fact_is_the_stereo_free_canonical_form(self):
        facts = fact_sheet(_mol("C[C@H](N)C(=O)O"), rng=random.Random(0))
        smiles = [f for f in facts if f.family == "smiles"][0]
        assert "@" not in smiles.value
        assert Chem.CanonSmiles(smiles.value) == smiles.value

    def test_fg_presence_carries_both_polarities(self):
        """The defect that cost the first assistant leg its grounding: emitting
        `fg_presence` for absent groups only made the family's value a constant,
        the set stated "it contains no X" 1,180 times and "it contains an X"
        never, and the model learned to answer no to every group question while
        scoring 0.994 on the same question in its own validator."""
        seen = set()
        for i, smiles in enumerate((ASPIRIN, CAFFEINE, "CCOCC", "N#CCNCC#N")):
            for fact in fact_sheet(_mol(smiles), rng=random.Random(i)):
                if fact.family == "fg_presence":
                    seen.add(fact.value)
        assert seen == {"yes", "no"}

    def test_a_positive_fg_presence_states_the_group_it_found(self):
        """The article is load-bearing — the sentence goes to the writer, and
        into a false premise verbatim through `render._negate`."""
        facts = [f for f in fact_sheet(_mol("CCOCC"), rng=random.Random(0))
                 if f.family == "fg_presence" and f.value == "yes"]
        assert facts, "diethyl ether contains an ether"
        fact = facts[0]
        assert fact.text == "It contains an ether."
        # Every downstream reader has to find the group, not the clause.
        assert _subject_of(fact) == "ether"

    def test_a_positive_fg_presence_does_not_license_the_wide_negative(self):
        """`ungrounded_claims` exempts a group from the widening check when the
        sheet holds a molecule-level fact about it. Only the *negative* form is
        that licence: matching on the family alone would wave through "contains
        no ether" for a molecule whose sheet says it has one."""
        facts = [Fact("fg_presence", "It contains an ether.", "yes", "yesno"),
                 Fact("fg_atom_membership", "atom 3 (C) is not part of an "
                      "ether.", "no", "yesno", atoms=[3])]
        assert ungrounded_claims("The molecule contains no ether.", facts)

    def test_the_atom_level_group_is_drawn_not_taken_in_dict_order(self):
        """`for name in present: ... break` took `present[0]`, so which group an
        atom-level question asked about was a function of `FUNCTIONAL_GROUPS`'
        declaration order rather than of the molecule: 53 % of the first set's
        3,889 such facts were about hydroxyl or ether, and sulfonamide got
        0.8 %."""
        paracetamol = "CC(=O)Nc1ccc(O)cc1"       # an amide and a hydroxyl
        asked = set()
        for seed in range(30):
            for fact in fact_sheet(_mol(paracetamol), rng=random.Random(seed)):
                if fact.family == "fg_atom_membership":
                    asked.add(_subject_of(fact))
        assert len(asked) > 1, f"only ever asked about {asked}"

    def test_atom_level_group_membership_is_not_almost_always_no(self):
        """The same line held the family at a yes-rate of 0.149, because a
        randomly drawn atom is rarely inside one particular group."""
        values = collections.Counter()
        for seed in range(40):
            for fact in fact_sheet(_mol("CC(=O)Nc1ccc(O)cc1"),
                                   rng=random.Random(seed)):
                if fact.family == "fg_atom_membership":
                    values[fact.value] += 1
        assert values["yes"] and values["no"]

    def test_atom_facts_number_atoms_the_way_the_questions_do(self):
        """Tier-A asks about "atom 14", 1-based. A sheet that numbered from zero
        would name a different atom than the trunk was trained on."""
        mol = _mol(CAFFEINE)
        facts = fact_sheet(mol, rng=random.Random(3))
        for fact in facts:
            for atom in fact.atoms:
                assert 1 <= atom <= mol.GetNumAtoms()
                assert f"atom {atom}" in fact.text


# ─────────────────────────────────────────────────────────────────────────────
# Briefs
# ─────────────────────────────────────────────────────────────────────────────

class TestBriefs:

    def test_briefs_vary_and_are_identified_by_their_axes(self):
        rng = random.Random(0)
        keys = {brief_key(draw_brief(rng)) for _ in range(200)}
        assert len(keys) > 50, "the brief space collapsed to a handful of points"

    def test_a_caption_is_never_combined_with_another_fact(self):
        """A caption is a paragraph. An example that has to contain it *and* a
        ring count has an answer that is a list, which is not what the set is."""
        facts = fact_sheet(_mol(ASPIRIN), rng=random.Random(0),
                           caption="The molecule is a member of benzoic acids.")
        rng = random.Random(0)
        for _ in range(200):
            chosen = select_facts(facts, {"n_facts": 3}, rng)
            if any(f.kind in ("text", "smiles") for f in chosen):
                assert len(chosen) == 1


# ─────────────────────────────────────────────────────────────────────────────
# The verifier
# ─────────────────────────────────────────────────────────────────────────────

class TestVerify:

    @pytest.mark.parametrize("answer", [
        "It has 3 rings.",
        "Three rings.",
        "The ring count is 3, which is what the fused system implies.",
    ])
    def test_a_count_passes_as_a_digit_or_a_word(self, answer):
        fact = Fact("ring_count", "It has 3 ring(s).", 3, "count")
        assert verify(answer, [fact])["passed"]

    @pytest.mark.parametrize("answer", [
        "It has 4 rings.", "Two rings.", "It has no rings at all.",
    ])
    def test_a_wrong_count_fails(self, answer):
        fact = Fact("ring_count", "It has 3 ring(s).", 3, "count")
        assert not verify(answer, [fact])["passed"]

    def test_a_yes_fact_passes_when_stated_without_the_word_yes(self):
        fact = Fact("fg_atom_membership", "atom 7 (O) is part of a hydroxyl group.",
                    "yes", "yesno", atoms=[7])
        assert verify("Atom 7 belongs to a hydroxyl group.", [fact])["passed"]
        assert verify("Yes.", [fact])["passed"]

    def test_a_negated_answer_fails_a_yes_fact_and_passes_a_no_one(self):
        yes = Fact("fg_atom_membership", "atom 7 (O) is part of a hydroxyl group.",
                   "yes", "yesno", atoms=[7])
        no = Fact("fg_atom_membership", "atom 7 (O) is not part of a hydroxyl group.",
                  "no", "yesno", atoms=[7])
        assert not verify("Atom 7 is not part of a hydroxyl group.", [yes])["passed"]
        assert verify("Atom 7 is not part of a hydroxyl group.", [no])["passed"]

    def test_a_quoted_smiles_must_be_this_molecule(self):
        value = Chem.CanonSmiles(ASPIRIN)
        fact = Fact("smiles", f"Its canonical SMILES is {value}.", value, "smiles")
        assert verify(f"The SMILES is {value}", [fact])["passed"]
        # A different valid spelling of the same molecule is still right.
        assert verify("It is O=C(C)Oc1ccccc1C(=O)O", [fact])["passed"]
        assert not verify(f"The SMILES is {Chem.CanonSmiles(CAFFEINE)}", [fact])["passed"]

    def test_every_fact_has_to_be_stated_not_just_one(self):
        facts = [Fact("ring_count", "It has 3 ring(s).", 3, "count"),
                 Fact("fg_count", "It contains 2 ether(s).", 2, "count")]
        assert not verify("It has 3 rings.", facts)["passed"]
        assert verify("It has 3 rings and 2 ethers.", facts)["passed"]

    def test_the_per_fact_breakdown_is_returned(self):
        """§9.4 reports correctness by fact family, which a boolean cannot carry."""
        facts = [Fact("ring_count", "It has 3 ring(s).", 3, "count"),
                 Fact("fg_count", "It contains 2 ether(s).", 2, "count")]
        result = verify("It has 3 rings.", facts)
        assert [f["contained"] for f in result["facts"]] == [True, False]
        assert [f["family"] for f in result["facts"]] == ["ring_count", "fg_count"]


class TestContainmentReadsWhatTheBriefsAsk:
    """Four ways the verifier refused an answer that stated its facts.

    All four were found the same way, by joining a rejection log back to the
    sheets it was written from: together they are ~43 % of `facts_missing`, the
    largest rejection reason on the v3 write.
    """

    def test_a_zero_ring_size_passes_in_the_sheets_own_words(self):
        """The sheet's sentence for this fact has no digit in it."""
        fact = Fact("ring_size", "atom 1 (C) is in no ring.", 0, "count", atoms=[1])
        assert verify("Atom 1 (C) is in no ring.", [fact])["passed"]
        assert verify("Atom 1 is not in any ring.", [fact])["passed"]
        assert verify("Atom 1 is not part of a ring.", [fact])["passed"]

    def test_the_no_ring_wording_does_not_pass_a_nonzero_size(self):
        fact = Fact("ring_size", "The smallest ring containing atom 1 (C) has "
                    "6 atoms.", 6, "count", atoms=[1])
        assert not verify("Atom 1 is in no ring.", [fact])["passed"]

    def test_two_properties_of_one_atom_are_not_confusable(self):
        """`kind` grouping made these two demand an atom the other already fixes."""
        facts = [Fact("ring_membership", "atom 17 (C) is in a ring.", "yes",
                      "yesno", atoms=[17]),
                 Fact("aromatic_ring", "atom 17 (C) is in an aromatic ring.",
                      "yes", "yesno", atoms=[17])]
        assert verify("Yes. It is in an aromatic ring.", facts)["passed"]

    def test_one_family_over_two_atoms_still_has_to_name_them(self):
        """The case the ambiguity rule exists for, and it still bites."""
        facts = [Fact("ring_size", "atom 2 (C) is in no ring.", 0, "count",
                      atoms=[2]),
                 Fact("ring_size", "atom 22 (N) is in no ring.", 0, "count",
                      atoms=[22])]
        assert not verify("It is in no ring.", facts)["passed"]
        assert verify("Neither atom 2 nor atom 22 is in any ring.", facts)["passed"]

    @pytest.mark.parametrize("answer", ["None", "Absent", "Inactive", "None."])
    def test_the_words_a_one_word_brief_produces_settle_polarity(self, answer):
        fact = Fact("tier_b/sider", "It is not active in the SIDER assay.", "no",
                    "yesno")
        assert verify(answer, [fact])["passed"]

    @pytest.mark.parametrize("answer", ["Present", "Active"])
    def test_the_same_words_settle_a_yes_the_other_way(self, answer):
        fact = Fact("tier_b/sider", "It is active in the SIDER assay.", "yes",
                    "yesno")
        assert verify(answer, [fact])["passed"]

    def test_a_json_answer_is_read_as_its_values(self):
        """The format axis asks for this shape; the key is the question restated."""
        fact = Fact("ring_membership", "atom 8 (C) is in a ring.", "yes", "yesno",
                    atoms=[8])
        assert verify('{"atom_in_ring": true}', [fact])["passed"]
        assert verify('{"answer": true}', [fact])["passed"]

    def test_a_json_answer_with_the_wrong_polarity_still_fails(self):
        fact = Fact("ring_membership", "atom 8 (C) is in a ring.", "yes", "yesno",
                    atoms=[8])
        assert not verify('{"atom_in_ring": false}', [fact])["passed"]

    def test_a_structured_answer_needs_a_value_for_each_fact(self):
        """The brief asks for JSON with the fact as the value, so one value is
        one fact: `{"nr_er_lbd": false}` drawn from two assays states one."""
        facts = [Fact("tier_b/tox21", "It is not active in the NR-ER-LBD assay.",
                      "no", "yesno"),
                 Fact("tier_b/tox21", "It is not active in the NR-AhR assay.",
                      "no", "yesno")]
        assert not verify('{"nr_er_lbd": false}', facts)["passed"]
        assert verify('{"nr_er_lbd": false, "nr_ahr": false}', facts)["passed"]

    def test_prose_keeps_the_latitude_the_count_takes_from_json(self):
        """A bare "No." answering two negatives is a register the briefs ask for."""
        facts = [Fact("tier_b/tox21", "It is not active in the NR-ER-LBD assay.",
                      "no", "yesno"),
                 Fact("tier_b/tox21", "It is not active in the NR-AhR assay.",
                      "no", "yesno")]
        assert verify("No.", facts)["passed"]

    def test_a_json_value_that_is_a_sentence_falls_through_to_prose(self):
        fact = Fact("fg_atom_membership", "atom 7 (O) is part of a hydroxyl group.",
                    "yes", "yesno", atoms=[7])
        assert verify('{"answer": "Atom 7 belongs to a hydroxyl group."}',
                      [fact])["passed"]

    def test_prose_that_merely_opens_with_a_brace_is_not_structured(self):
        fact = Fact("ring_membership", "atom 8 (C) is in a ring.", "yes", "yesno",
                    atoms=[8])
        assert verify("{atom 8} is in a ring.", [fact])["passed"]

    #: Two facts of opposite polarity about one atom, which the sampler draws
    #: constantly and the writer answers in one sentence.
    MIXED = [Fact("ring_size", "The smallest ring containing atom 24 (C) has "
                  "6 atoms.", 6, "count", atoms=[24]),
             Fact("aromatic_ring", "atom 24 (C) is not in an aromatic ring.",
                  "no", "yesno", atoms=[24]),
             Fact("ring_membership", "atom 24 (C) is in a ring.", "yes",
                  "yesno", atoms=[24])]

    def test_one_negation_does_not_negate_the_whole_sentence(self):
        assert verify("Atom 24 (C) is in a ring, the smallest ring containing "
                      "it has 6 atoms, and it is not in an aromatic ring.",
                      self.MIXED)["passed"]

    def test_an_atom_keeps_the_subject_until_another_is_named(self):
        assert verify("Atom 24 (C) is in a ring. The smallest ring containing "
                      "it has 6 atoms. It is not in an aromatic ring.",
                      self.MIXED)["passed"]

    def test_two_atoms_contrasted_in_one_sentence(self):
        facts = [Fact("ring_membership", "atom 19 (C) is in a ring.", "yes",
                      "yesno", atoms=[19]),
                 Fact("ring_membership", "atom 3 (C) is not in a ring.", "no",
                      "yesno", atoms=[3])]
        assert verify("Atom 19 (C) is in a ring, whereas atom 3 (C) is not in "
                      "a ring.", facts)["passed"]

    def test_a_compound_subject_is_not_two_clauses(self):
        facts = [Fact("ring_membership", "atom 5 (O) is in a ring.", "yes",
                      "yesno", atoms=[5]),
                 Fact("ring_membership", "atom 21 (C) is in a ring.", "yes",
                      "yesno", atoms=[21])]
        assert verify("Atom 5 (O) and atom 21 (C) are in a ring.",
                      facts)["passed"]

    def test_naming_the_atom_is_not_stating_the_fact(self):
        fact = Fact("ring_membership", "atom 14 (C) is in a ring.", "yes",
                    "yesno", atoms=[14])
        assert not verify("Atom 14 (C) is a carbon.", [fact])["passed"]

    def test_a_contradiction_still_fails(self):
        fact = Fact("ring_membership", "atom 14 (C) is in a ring.", "yes",
                    "yesno", atoms=[14])
        assert not verify("Atom 14 (C) is not in a ring.", [fact])["passed"]


class TestFormatAndFilters:

    @pytest.mark.parametrize("instruction, answer, ok", [
        ("answer as JSON with the fact as the value", '{"rings": 3}', True),
        ("answer as JSON with the fact as the value", "3 rings", False),
        ("answer in one word", "3", True),
        ("answer in one word", "It has 3 rings", False),
        ("answer as a numbered list", "1. rings: 3\n2. ethers: 2", True),
        ("answer as a numbered list", "rings: 3", False),
        ("start the answer with the number", "3 rings, all aromatic", True),
        ("start the answer with the number", "It has 3 rings", False),
        ("", "anything at all", True),
    ])
    def test_the_briefs_format_instruction_is_checked(self, instruction, answer, ok):
        assert format_met(answer, {"format": instruction}) is ok

    @pytest.mark.parametrize("text", [
        "What is the longest chain of carbons?",
        "How many bonds apart are atoms 3 and 9?",
        "Was it approved by the FDA?",
        "Is this molecule toxic in clinical trials?",
    ])
    def test_held_out_language_is_refused(self, text):
        assert mentions_held_out(text)

    def test_ordinary_molecule_questions_are_not_refused(self):
        for text in ("How many rings does this molecule have?",
                     "Is atom 4 aromatic?",
                     "Does it contain a sulfonamide?"):
            assert not mentions_held_out(text)

    def test_near_duplicates_are_close_in_four_gram_jaccard(self):
        a = four_grams("how many rings does this molecule have in total")
        b = four_grams("how many rings does this molecule have overall")
        c = four_grams("is atom four part of an aromatic ring system")
        assert jaccard(a, b) > 0.4
        assert jaccard(a, c) == 0.0


class TestUnsupportedClaims:
    """The half of verification that catches what a writer *added*."""

    SHEET = [Fact("fg_count", "It contains 1 amide(s).", 1, "count"),
             Fact("ring_count", "It has 2 ring(s).", 2, "count"),
             Fact("fg_presence", "It contains no nitrile.", "no", "yesno")]

    def test_a_group_on_the_sheet_may_be_stated(self):
        assert unsupported_claims("It has one amide.", self.SHEET) == []

    def test_a_group_the_sheet_denies_may_still_be_named(self):
        # "It contains no nitrile" is itself a fact, so naming the nitrile to
        # deny it is the sheet's own language, not an addition.
        assert unsupported_claims("There is no nitrile here.", self.SHEET) == []

    def test_a_group_the_sheet_never_mentions_is_refused(self):
        assert unsupported_claims("It has a ketone and an amide.",
                                  self.SHEET) == ["ketone"]

    def test_a_named_ring_system_is_always_refused(self):
        found = unsupported_claims(
            "The molecule has two benzene rings joined by an ester.", self.SHEET)
        assert set(found) == {"benzene", "ester"}

    def test_the_real_failure_from_the_writer_smoke(self):
        # Handed one SMILES fact, a 12B writer described rings and groups it had
        # inferred. Every clause here is invention.
        answer = ("The molecule has two benzene rings linked through a central "
                  "ethylene bridge, with each ring carrying a hydroxyl group.")
        assert unsupported_claims(answer, [Fact("smiles", "Its canonical "
                                                "SMILES is CCO.", "CCO",
                                                "smiles")])

    def test_plain_structural_language_is_left_alone(self):
        assert unsupported_claims(
            "It has two rings, one of them aromatic, and atom 4 is in it.",
            self.SHEET) == []


class TestBriefsAreSatisfiable:
    """A brief the writer cannot satisfy spends a pass to produce a rejection."""

    def test_a_terse_format_never_asks_for_more_than_one_fact(self):
        rng = random.Random(0)
        for _ in range(500):
            brief = draw_brief(rng)
            if brief["format"] in ("answer in one word",
                                   "start the answer with the number"):
                assert brief["n_facts"] == 1

    def test_start_with_the_number_picks_a_fact_that_has_one(self):
        facts = [Fact("ring_membership", "atom 3 (C) is in a ring.", "yes",
                      "yesno", atoms=[3]),
                 Fact("ring_count", "It has 2 ring(s).", 2, "count")]
        brief = {"format": "start the answer with the number", "n_facts": 1}
        chosen = select_facts(facts, brief, random.Random(0))
        assert [f.kind for f in chosen] == ["count"]

    def test_a_one_word_brief_never_draws_a_caption(self):
        facts = [Fact("caption", "The molecule is a long paragraph …", "", "text"),
                 Fact("ring_count", "It has 2 ring(s).", 2, "count")]
        brief = {"format": "answer in one word", "n_facts": 1}
        for seed in range(50):
            chosen = select_facts(facts, brief, random.Random(seed))
            assert all(f.kind != "text" for f in chosen)

    #: One functional-group count, and two atom-level facts about two different
    #: atoms from a third family — the shape that produced "There are 2 ethers.
    #: Atom 20 (O) is not in an aromatic ring … Therefore, there are 2 ethers."
    MIXED = [Fact("fg_count", "It contains 2 ether(s).", 2, "count"),
             Fact("aromatic_ring", "atom 20 (O) is not in an aromatic ring.",
                  "no", "yesno", atoms=[20]),
             Fact("aromatic_ring", "atom 10 (C) is not in an aromatic ring.",
                  "no", "yesno", atoms=[10]),
             Fact("ring_membership", "atom 20 (O) is in a ring.", "yes",
                  "yesno", atoms=[20])]

    def test_a_multi_fact_draw_shares_a_family_or_an_atom(self):
        """Unrelated facts in one answer invite a false connective joining them,
        so a draw that cannot be coherent is shortened instead."""
        for seed in range(100):
            chosen = select_facts(self.MIXED, {"n_facts": 3}, random.Random(seed))
            pivot = chosen[0]
            for other in chosen[1:]:
                assert (other.family == pivot.family
                        or set(other.atoms) & set(pivot.atoms)), \
                    f"{pivot!r} drawn with {other!r}"

    def test_a_pivot_with_no_kin_is_written_from_alone(self):
        """`fg_count` is the only fact of its family here and is scoped to no
        atom, so the coherent draw is one fact, not three."""
        facts = [self.MIXED[0], self.MIXED[1], self.MIXED[2]]
        lone = [select_facts(facts, {"n_facts": 3}, random.Random(seed))
                for seed in range(100)]
        assert any(len(c) == 1 and c[0].family == "fg_count" for c in lone)
        assert all(len(c) > 1 or c[0].family == "fg_count" or len(facts) == 1
                   for c in lone)

    def test_the_coherent_draw_still_reaches_the_asked_for_size(self):
        facts = [Fact("aromatic_ring", f"atom {i} (C) is in an aromatic ring.",
                      "yes", "yesno", atoms=[i]) for i in range(1, 6)]
        chosen = select_facts(facts, {"n_facts": 3}, random.Random(0))
        assert len(chosen) == 3


class TestEndpointClauses:
    """A negative label has to read as English, not as a prefixed positive."""

    def test_a_negative_assay_label_is_a_sentence(self):
        assert endpoint_words("tox21", "SR-MMP", negative=True) == \
            "is not active in the SR MMP assay"

    def test_a_negative_headline_endpoint_conjugates(self):
        for corpus, endpoint in (("bace", "Class"), ("bbbp", "p_np"),
                                 ("hiv", "HIV_active")):
            clause = endpoint_words(corpus, endpoint, negative=True)
            assert not clause.startswith("does not is")
            assert f"It {clause}." != f"It {endpoint_words(corpus, endpoint)}."

    def test_the_sheet_states_a_negative_label_that_way(self):
        facts = fact_sheet(_mol(ASPIRIN), rng=random.Random(0),
                           tier_b=[("tox21", "SR-MMP", False)])
        stated = [f.text for f in facts if f.family == "tier_b/tox21"]
        assert stated == ["It is not active in the SR MMP assay."]


class TestSheetTalk:
    """An answer that reasons from the list is teaching a move it cannot make."""

    @pytest.mark.parametrize("answer", [
        "No, because fact 1 states that atom 7 is not in a ring.",
        "According to the facts provided, atom 5 is not aromatic.",
        "From the given information, it has two rings.",
        "As stated above, there is no nitrile.",
        "The fact sheet says it contains one ether.",
    ])
    def test_talking_about_the_list_is_refused(self, answer):
        assert mentions_the_sheet(answer)

    @pytest.mark.parametrize("answer", [
        "It has two rings, one of which is aromatic.",
        "No — atom 5 is not in an aromatic ring.",
        "The smallest ring containing atom 9 has 6 atoms.",
    ])
    def test_an_answer_about_the_molecule_is_not(self, answer):
        assert not mentions_the_sheet(answer)


def test_the_leading_number_must_be_the_facts_number():
    brief = {"format": "start the answer with the number"}
    facts = [Fact("ring_size", "The smallest ring containing atom 16 (C) has 0 "
                  "atoms (0 means it is in no ring).", 0, "count")]
    assert format_met("0, because atom 16 is in no ring.", brief, facts)
    # A list marker is a leading digit and answers nothing.
    assert not format_met("1, atom 16 is not in a ring. 2, the smallest ring "
                          "has 0 atoms.", brief, facts)


def test_a_numbered_list_need_not_use_newlines():
    brief = {"format": "answer as a numbered list"}
    assert format_met("1. It has two rings.  2. One of them is aromatic.", brief)
    assert not format_met("It has two rings and one is aromatic.", brief)


class TestUngroundedClaims:
    """Every case here is a row the containment verifier accepted in v1, and the
    hand review of that sample rejected. They share a shape: the fact is stated
    correctly and the sentence around it asserts something else."""

    def test_a_law_of_chemistry_is_not_a_fact_about_this_molecule(self):
        facts = [Fact("fg_count", "It contains 1 ether(s).", 1, "count")]
        found = ungrounded_claims(
            "1, because there is 1 ether and all ethers contain exactly 2 "
            "oxygen atoms.", facts)
        assert found and found[0].startswith("universal:")

    def test_an_atom_index_is_not_an_atomic_number(self):
        facts = [Fact("aromatic_ring", "atom 17 (Cl) is not in an aromatic "
                      "ring.", "no", "yesno", atoms=[17])]
        assert ungrounded_claims(
            "No. The atom is chlorine, which has the atomic number 17.", facts)

    def test_a_per_atom_negative_does_not_clear_the_molecule(self):
        facts = [Fact("fg_atom_membership", "atom 14 (C) is not part of a "
                      "ether.", "no", "yesno", atoms=[14])]
        found = ungrounded_claims(
            "No, this molecule is not an ether because it contains no ether "
            "group.", facts)
        assert found == ["molecule-wide: ether"]

    def test_the_wide_form_is_fine_when_a_wide_fact_licenses_it(self):
        facts = [Fact("fg_presence", "It contains no ether.", "no", "yesno"),
                 Fact("fg_atom_membership", "atom 14 (C) is not part of a "
                      "ether.", "no", "yesno", atoms=[14])]
        assert not ungrounded_claims("It contains no ether.", facts)

    def test_an_ordinary_grounded_answer_passes(self):
        facts = [Fact("ring_count", "It has 1 ring(s).", 1, "count")]
        assert not ungrounded_claims(
            "It has one ring, and that ring is aromatic.", facts)


class TestOpenerContradicts:
    YESNO = [Fact("fg_atom_membership", "atom 8 (N) is not part of a hydroxyl "
                  "group.", "no", "yesno", atoms=[8])]

    def test_yes_followed_by_a_negation_is_caught(self):
        assert opener_contradicts(
            "Yes, atom 8, which is a nitrogen atom, is not part of a hydroxyl "
            "group.",
            "Does this nitrogen atom belong to a hydroxyl group?", self.YESNO)

    def test_no_followed_by_a_negation_agrees_with_itself(self):
        assert not opener_contradicts(
            "No, atom 8 is not part of a hydroxyl group.",
            "Does this nitrogen atom belong to a hydroxyl group?", self.YESNO)

    def test_a_negative_question_may_be_confirmed_with_yes(self):
        assert not opener_contradicts(
            "Yes, atom 8 is not part of a hydroxyl group.",
            "Is it true that atom 8 is not part of a hydroxyl group?",
            self.YESNO)

    def test_an_answer_that_does_not_open_with_yes_or_no_is_untouched(self):
        assert not opener_contradicts(
            "Atom 8 is not part of a hydroxyl group.", "Where is atom 8?",
            self.YESNO)

    def test_a_bare_no_is_not_a_contradiction(self):
        """The most ordinary correct answer in the set. With nothing after the
        opener there is no clause for it to contradict, and the check has to
        abstain rather than read an empty rest as agreement."""
        for answer in ("No.", "No", "no.", "No. "):
            assert not opener_contradicts(
                answer, "Is atom 22 (C) part of a carboxylic acid?", self.YESNO)

    def test_a_bare_no_followed_by_a_restatement_is_not_a_contradiction(self):
        assert not opener_contradicts(
            "No. 1. It is not part of a hydroxyl group.",
            "Is atom 8 part of a hydroxyl group?", self.YESNO)

    def test_yes_is_not_read_as_polarity_outside_a_yes_no_question(self):
        """"Yes" opening an answer to "what can be inferred?" is a discourse
        marker; the fact after it is stated correctly and the row is correct."""
        assert not opener_contradicts(
            "Yes, it can be inferred that the molecule lacks any amide group.",
            "If a molecule is devoid of amide groups, what can be inferred "
            "about its chemical composition?", self.YESNO)

    def test_the_original_defect_still_fires_under_a_polar_question(self):
        for question in ("Is atom 8 part of a hydroxyl group?",
                         "Does atom 8 belong to a hydroxyl group?",
                         "Atom 8 is in a hydroxyl group, correct?"):
            assert opener_contradicts(
                "Yes, atom 8 is not part of a hydroxyl group.", question,
                self.YESNO), question


class TestQuestionLeaks:
    YES = [Fact("ring_membership", "atom 3 (C) is in a ring.", "yes", "yesno",
                atoms=[3])]
    NO = [Fact("ring_membership", "atom 1 (C) is not in a ring.", "no",
               "yesno", atoms=[1])]
    COUNT = [Fact("ring_count", "It has 3 ring(s).", 3, "count")]

    def test_a_neutral_yes_no_question_is_not_a_leak(self):
        """The plainest question in the set. It commits to nothing, and only
        looks like an assertion because containment cannot tell the two apart."""
        for question in ("Is atom 3 (C) in a ring?",
                         "Is atom 3 (C) in a ring?",
                         "Does atom 3 (C) sit in a ring?"):
            assert not question_leaks(question, self.YES), question

    def test_a_presupposing_question_still_leaks(self):
        assert question_leaks("Why is atom 3 (C) in a ring?", self.YES)
        assert question_leaks("Can you explain why atom 3 (C) is in a ring?",
                              self.YES)

    def test_a_negative_fact_gets_no_exemption(self):
        """A question only reaches the negative by putting it there."""
        assert question_leaks("Is atom 1 (C) not in a ring?", self.NO)
        assert question_leaks("Why is atom 1 (C) not in a ring?", self.NO)

    def test_a_declarative_sentence_beside_the_question_still_leaks(self):
        assert question_leaks(
            "My friend found a molecule where atom 3 (C) is in a ring. "
            "Should I tell them?", self.YES)

    def test_a_preamble_does_not_cost_the_question_its_polarity(self):
        """Three brief shapes put the interrogative second — the scenario, the
        stated reason for asking, and every voice that sets a scene first. The
        exemption used to be anchored at the start of the whole string, so it
        never once applied to them: on a 400-row arm it called 0 of 23 leak
        rejections polar, and every plain question about a `yes` fact was
        refused for naming its own subject."""
        for question in (
                "To map the structure, is atom 3 (C) in a ring?",
                "I am reviewing this molecule for the log. Is atom 3 (C) in a ring?",
                "Examining the cyclic structure, does atom 3 (C) sit in a ring?",
                "I'm studying this structure; is atom 3 (C) in a ring?"):
            assert not question_leaks(question, self.YES), question

    def test_an_instruction_is_a_yes_no_question(self):
        """The "an instruction" shape asks for exactly this, and it is as
        neutral as the interrogative it paraphrases."""
        for question in ("Determine if atom 3 (C) is in a ring.",
                         "Please determine whether atom 3 (C) is in a ring.",
                         "Check if atom 3 (C) is in a ring.",
                         "Tell me whether atom 3 (C) is in a ring."):
            assert not question_leaks(question, self.YES), question

    def test_naming_the_atom_before_asking_is_not_asserting(self):
        """The atom introduced in one clause and asked about in the next.
        `_subject_of` an atom-scoped fact is the atom reference alone, so the
        naming clause "contains" the fact by every containment test — but it
        claims nothing, and the clause that does the asking is the polar one."""
        for question in ("I'm looking at atom 3. Is it in a ring?",
                         "Consider atom 3 (C). Is it cyclic?",
                         "Examine atom 3. Confirm if it sits in a ring."):
            assert not question_leaks(question, self.YES), question

    def test_an_assertion_is_still_an_assertion_after_a_preamble(self):
        """The widening must not reach a clause that states the fact outright."""
        assert question_leaks(
            "I checked the structure and atom 3 (C) is in a ring. "
            "Is that worth noting?", self.YES)

    def test_a_count_stated_in_the_question_leaks(self):
        assert question_leaks("She notices it has 3 rings. What does that "
                              "mean?", self.COUNT)

    def test_a_question_that_states_nothing_is_fine(self):
        assert not question_leaks("How many rings does this molecule have?",
                                  self.COUNT)

    def test_the_two_leaky_forms_are_exempt_by_design(self):
        for form in LEAKY_FORMS:
            assert not question_leaks("Why is atom 1 (C) not in a ring?",
                                      self.NO, form)


class TestQuestionMismatchesFacts:
    COUNT = [Fact("ring_size", "The smallest ring containing atom 20 (C) has "
                  "6 atoms.", 6, "count", atoms=[20])]
    YESNO = [Fact("aromatic_ring", "atom 24 (C) is in an aromatic ring.",
                  "yes", "yesno", atoms=[24])]

    def test_a_count_question_written_from_a_yes_no_fact_is_caught(self):
        assert question_mismatches_facts(
            "How many atoms are in an aromatic ring containing atom 24?",
            "yes", self.YESNO)

    def test_a_count_question_with_a_count_fact_is_fine(self):
        assert not question_mismatches_facts(
            "How many atoms are in the smallest ring containing atom 20 (C)?",
            "6, because it sits in a six-membered ring.", self.COUNT)

    def test_a_question_about_something_nobody_computed_is_caught(self):
        """The drawn fact is a ring size for atom 20 and the example never
        mentions atom 20 — the number is borrowed, and the claim about bromine
        came from the writer."""
        reason = question_mismatches_facts(
            "Is bromine present in the molecule?",
            "0, because bromine is not present in the molecule.", self.COUNT)
        assert "atom" in reason and "20" in reason

    def test_naming_the_atom_in_the_answer_alone_is_enough(self):
        """The brief may ask a question that does not name the atom, so long as
        the example is about it — "answer only" briefs do exactly that."""
        assert not question_mismatches_facts(
            "What is the smallest ring here?",
            "Atom 20 (C) sits in a ring of 6 atoms.", self.COUNT)

    def test_a_molecule_level_fact_is_not_asked_to_name_an_atom(self):
        facts = [Fact("ring_count", "It has 3 ring(s).", 3, "count")]
        assert not question_mismatches_facts("How many rings?", "3", facts)

    def test_no_facts_is_not_a_mismatch(self):
        assert not question_mismatches_facts("Anything?", "Yes", [])


class TestQuestionChangesSubject:
    RING_COUNT = [Fact("ring_count", "It has 2 ring(s).", 2, "count")]
    STEREO = [Fact("stereo_assigned", "It has 1 assigned stereocentre(s).", 1,
                   "count")]

    def test_a_chirality_question_answered_from_a_ring_count_is_caught(self):
        """The v2 review's worst surviving row: "2, because it has two rings",
        written for "what is the maximum number of chiral centers". Every other
        check passes — it is a count question with a count fact, and the value
        is contained."""
        assert question_changes_subject(
            "What is the maximum number of chiral centers in this molecule?",
            self.RING_COUNT) == "stereochemistry"

    def test_the_same_question_with_a_stereo_fact_is_fine(self):
        assert not question_changes_subject(
            "What is the maximum number of chiral centers in this molecule?",
            self.STEREO)

    def test_a_ring_question_with_a_ring_fact_is_fine(self):
        assert not question_changes_subject("How many rings does it have?",
                                            self.RING_COUNT)

    def test_an_aromaticity_question_is_answered_by_the_aromatic_family_only(self):
        assert question_changes_subject("How many aromatic rings are there?",
                                        self.RING_COUNT) == "aromaticity"
        aromatic = [Fact("aromatic_ring", "2 of its rings are aromatic.", 2,
                         "count")]
        assert not question_changes_subject("How many aromatic rings?", aromatic)

    def test_a_functional_group_question_needs_a_functional_group_fact(self):
        assert question_changes_subject("How many ethers are present?",
                                        self.RING_COUNT) == "functional group"

    def test_a_question_raising_no_named_subject_is_left_alone(self):
        """Most questions name their subject in words the sheet never uses, and
        this check is not entitled to an opinion about those."""
        assert not question_changes_subject("What can you tell me about this?",
                                            self.RING_COUNT)

    def test_an_element_census_has_no_family_and_is_always_refused(self):
        """"Does this molecule have three carbons?" answered "Yes … the SMILES
        starts with C, which represents a carbon atom", for a molecule with
        twenty. Nothing here counts elements, so nothing can answer it."""
        smiles = [Fact("smiles", "Its canonical SMILES is CCO.", "CCO",
                       "smiles")]
        assert question_changes_subject("Does it have three carbon atoms?",
                                        smiles) == "atom census"
        assert question_changes_subject("What is the molecular weight?",
                                        self.RING_COUNT) == "atom census"
        assert question_changes_subject("How many halogen atoms does it "
                                        "contain?", self.RING_COUNT)

    def test_a_ring_size_question_is_not_an_element_census(self):
        """"How many atoms are in the smallest ring containing atom 24?" is a
        ring_size question and one of the commonest shapes in the set — a bare
        "how many atoms" cannot be the trigger."""
        ring_size = [Fact("ring_size", "The smallest ring containing atom 24 "
                          "(C) has 6 atoms.", 6, "count", atoms=[24])]
        assert not question_changes_subject(
            "How many atoms are in the smallest ring containing atom 24?",
            ring_size)
        stereo = [Fact("stereo_potential", "2 atom(s) could be stereocentres.",
                       2, "count")]
        assert not question_changes_subject(
            "What is the number of atoms that could be stereocenters?", stereo)
        assert not question_changes_subject(
            "How many atoms in this molecule can be stereocenters?", stereo)

    def test_a_smiles_fact_answers_about_smiles_and_nothing_else(self):
        smiles = [Fact("smiles", "Its canonical SMILES is CCO.", "CCO",
                       "smiles")]
        assert not question_changes_subject("What is the SMILES string?", smiles)
        assert question_changes_subject("How many rings does it have?",
                                        smiles) == "rings"

    def test_a_caption_may_raise_anything(self):
        facts = [Fact("caption", "A steroid with two fused rings.",
                      "A steroid with two fused rings.", "text")]
        assert not question_changes_subject("Describe its stereochemistry.",
                                            facts)


class TestAnswerInventsAtom:
    ATOM_15 = [Fact("aromatic_ring", "atom 15 (C) is in an aromatic ring.",
                    "yes", "yesno", atoms=[15])]

    def test_a_second_atom_the_sheet_never_mentioned_is_caught(self):
        """From the v2 review. Sentence one states the fact and verifies; the
        list item after it is invented, and containment cannot see it."""
        assert answer_invents_atom(
            "1. Atom 15 (C) is in an aromatic ring.\n"
            "2. Atom 16 (C) is in an aromatic ring.", self.ATOM_15) == "16"

    def test_naming_only_the_facts_own_atom_is_fine(self):
        assert not answer_invents_atom("Yes — atom 15 (C) is aromatic.",
                                       self.ATOM_15)

    def test_a_molecule_level_fact_licenses_no_atom_at_all(self):
        facts = [Fact("ring_count", "It has 2 ring(s).", 2, "count")]
        assert answer_invents_atom("2, counting the ring at atom 7.",
                                   facts) == "7"

    def test_every_invented_atom_is_reported(self):
        assert answer_invents_atom("Atoms 16, 17 and 15 are aromatic.",
                                   self.ATOM_15) == "16,17"

    def test_a_plain_number_is_not_an_atom_reference(self):
        assert not answer_invents_atom("There are 6 of them.", self.ATOM_15)

    def test_a_smiles_answer_is_exempt(self):
        facts = [Fact("smiles", "Its canonical SMILES is CCO.", "CCO", "smiles")]
        assert not answer_invents_atom("CCO, which has atom 3 as oxygen.", facts)


class TestQuestionWidensScope:
    ATOM = [Fact("fg_atom_membership", "atom 15 (C) is not part of an ether.",
                 "no", "yesno", atoms=[15])]

    def test_a_molecule_question_from_an_atom_fact_is_caught(self):
        """"No, because atom 15 (C) is not part of an ether" is true of atom 15
        and says nothing about the other forty."""
        assert question_widens_scope("Is this molecule an ether?", self.ATOM)

    def test_any_is_a_quantifier_over_the_molecule(self):
        assert question_widens_scope("Does it have any primary amines?",
                                     self.ATOM)

    def test_a_question_that_names_its_atom_is_exempt(self):
        assert not question_widens_scope(
            "In this molecule, is atom 15 part of an ether?", self.ATOM)

    def test_a_molecule_level_fact_is_not_this_checks_business(self):
        facts = [Fact("fg_presence", "It contains no ether.", "no", "yesno")]
        assert not question_widens_scope("Is this molecule an ether?", facts)

    def test_a_mixed_draw_is_left_alone(self):
        facts = self.ATOM + [Fact("ring_count", "It has 2 ring(s).", 2, "count")]
        assert not question_widens_scope("Is this molecule an ether?", facts)

    def test_one_witness_settles_an_existential_question(self):
        """"Are any atoms in an aromatic ring?" answered "Yes. Atom 8 is in an
        aromatic ring" is sound — the first version of this check refused it."""
        yes = [Fact("aromatic_ring", "atom 8 (C) is in an aromatic ring.",
                    "yes", "yesno", atoms=[8])]
        assert not question_widens_scope("Are any atoms in an aromatic ring?",
                                         yes)

    def test_no_number_of_atoms_settles_it_in_the_negative(self):
        """The asymmetry: one atom that is not an ether leaves the rest
        unexamined, so the existential exemption is for positives only."""
        assert question_widens_scope("Does this molecule have any ethers?",
                                     self.ATOM)


class TestClaimsConnectivity:
    RING = [Fact("ring_membership", "atom 13 (C) is in a ring.", "yes",
                 "yesno", atoms=[13])]

    def test_a_bond_justification_is_refused(self):
        """No family in the sheet holds connectivity, so the "because" here is
        reasoning from something nothing computed."""
        assert claims_connectivity(
            "Yes, atom 13 is aromatic. This is because the atom is bonded to "
            "four atoms.", self.RING) == "bonded"

    def test_connected_to_and_attached_to_are_the_same_claim(self):
        assert claims_connectivity("1, because it has one oxygen connected to "
                                   "two carbons.", self.RING)
        assert claims_connectivity("The hydroxyl is attached to a carbon.",
                                   self.RING)

    def test_an_answer_that_states_the_fact_is_fine(self):
        assert not claims_connectivity("Yes, atom 13 (C) is in a ring.",
                                       self.RING)

    def test_a_generic_definition_is_not_a_claim_about_this_molecule(self):
        """A third of the first version's catches were textbook definitions."""
        assert not claims_connectivity(
            "1 primary amine, because a primary amine is a nitrogen atom "
            "bonded to one carbon atom and two hydrogen atoms.", self.RING)
        assert not claims_connectivity(
            "Amides are characterized by a carbonyl group bonded to a "
            "nitrogen.", self.RING)

    def test_a_definition_does_not_excuse_the_claim_beside_it(self):
        assert claims_connectivity(
            "An amide is a nitrogen with a double bond to a carbon. We see "
            "that atom 11 is not bonded to a nitrogen.", self.RING)

    def test_the_sheets_own_containing_is_not_connectivity(self):
        facts = [Fact("ring_size", "The smallest ring containing atom 9 (O) "
                      "has 6 atoms.", 6, "count", atoms=[9])]
        assert not claims_connectivity(
            "The smallest ring containing atom 9 (O) has 6 atoms.", facts)

    def test_a_smiles_answer_is_exempt(self):
        """A SMILES string *is* connectivity, in the one notation this pipeline
        computes."""
        facts = [Fact("smiles", "Its canonical SMILES is CCO.", "CCO", "smiles")]
        assert not claims_connectivity(
            "CCO — the oxygen is attached to a carbon.", facts)


class TestJsonKeyMisnames:
    BRIEF = {"format": "answer as JSON with the fact as the value"}
    RING_SIZE = [Fact("ring_size", "atom 25 (C) is in no ring.", 0, "count",
                      atoms=[25])]

    def test_a_key_naming_another_family_is_a_mislabel(self):
        assert json_key_misnames('{"ring_count": 0}', self.BRIEF,
                                 self.RING_SIZE) == "ring_count"

    def test_the_example_s_own_family_is_fine(self):
        assert not json_key_misnames('{"ring_size": 0}', self.BRIEF,
                                     self.RING_SIZE)

    def test_a_free_form_key_is_fine(self):
        assert not json_key_misnames('{"smallest_ring": 0}', self.BRIEF,
                                     self.RING_SIZE)

    def test_a_non_json_brief_is_not_policed(self):
        assert not json_key_misnames('{"ring_count": 0}', {"format": ""},
                                     self.RING_SIZE)


def test_the_ring_size_gloss_appears_only_where_it_applies():
    """The "(0 means it is in no ring)" parenthetical, carried on a nonzero size,
    is what produced "has 6 atoms. Since it is in no ring, the ring size is 0"."""
    facts = fact_sheet(Chem.MolFromSmiles(CAFFEINE), rng=random.Random(0))
    sizes = [f for f in facts if f.family == "ring_size"]
    assert sizes, "caffeine has ring atoms"
    for fact in sizes:
        assert "0 means" not in fact.text
        if str(fact.value) == "0":
            assert "in no ring" in fact.text
        else:
            assert "in no ring" not in fact.text


def test_an_atom_scoped_fact_must_name_its_atom():
    """Two atoms with the same ring size were both satisfied by one sentence
    about one of them, because the number looked for is "0" either way."""
    facts = [Fact("ring_size", "atom 22 (F) is in no ring.", 0, "count",
                  atoms=[22]),
             Fact("ring_size", "atom 2 (C) is in no ring.", 0, "count",
                  atoms=[2])]
    contained = facts_contained("Atom 2 is in no ring: it has 0 ring atoms.",
                                facts)
    assert contained == {0: False, 1: True}
    # A lone atom-scoped fact still passes in any register, "Yes" included.
    assert facts_contained("Yes.", [Fact("ring_membership", "atom 9 (C) is in "
                                         "a ring.", "yes", "yesno", atoms=[9])])


def test_group_names_take_the_right_article():
    facts = fact_sheet(Chem.MolFromSmiles(ASPIRIN), rng=random.Random(0))
    for fact in facts:
        assert " a amide" not in fact.text and " a ether" not in fact.text
        assert " a ester" not in fact.text and " a alcohol" not in fact.text


def test_sider_columns_are_not_called_assays():
    """SIDER's columns are MedDRA system-organ classes. "Active in the
    Investigations assay" states something nobody measured."""
    assert "assay" not in endpoint_words("sider", "Investigations")
    assert "assay" not in endpoint_words("sider", "Investigations", negative=True)
    assert "assay" in endpoint_words("tox21", "NR-AR")


def test_facts_contained_reads_every_fact_independently():
    facts = [Fact("ring_count", "It has 3 ring(s).", 3, "count"),
             Fact("stereo_assigned", "2 stereocenter(s) have a defined "
                  "configuration.", 2, "count")]
    contained = facts_contained("Three rings; two defined stereocenters.", facts)
    assert contained == {0: True, 1: True}


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot demonstrations
# ─────────────────────────────────────────────────────────────────────────────

def _row(id_, key, question, answer, facts, fmt=""):
    return {"id": id_, "key": key, "question": question, "answer": answer,
            "facts": [f.to_json() for f in facts], "brief": {"format": fmt}}


class TestShotDraw:
    def test_the_share_with_demonstrations_lands_near_the_constant(self):
        rng = random.Random(0)
        counts = [draw_shot_count(rng) for _ in range(20_000)]
        share = sum(1 for c in counts if c) / len(counts)
        assert abs(share - SHOT_FRACTION) < 0.02
        assert set(c for c in counts if c) <= set(SHOT_COUNTS)
        assert max(counts) <= 4

    def test_the_fraction_can_be_overridden(self):
        rng = random.Random(0)
        counts = [draw_shot_count(rng, 0.0) for _ in range(200)]
        assert set(counts) == {0}


class TestDemoLeaksTarget:
    def test_a_demonstration_answering_the_targets_own_value_is_refused(self):
        target = [Fact("ring_count", "It has 3 ring(s).", 3, "count")]
        assert demo_leaks_target("3, because it has three rings.", target)
        assert demo_leaks_target("Three.", target)

    def test_a_demonstration_with_a_different_value_is_fine(self):
        target = [Fact("ring_count", "It has 3 ring(s).", 3, "count")]
        assert not demo_leaks_target("5, because it has five rings.", target)

    def test_a_target_with_no_facts_cannot_be_leaked(self):
        assert not demo_leaks_target("anything at all", [])


class TestShotCandidates:
    def _pool(self):
        ring3 = [Fact("ring_count", "It has 3 ring(s).", 3, "count")]
        ring5 = [Fact("ring_count", "It has 5 ring(s).", 5, "count")]
        stereo = [Fact("stereo_assigned", "1 stereocenter(s) have a defined "
                       "configuration.", 1, "count")]
        return [
            _row("train-1", "CCO", "How many rings?", "5", ring5),
            _row("train-2", "CCC", "Rings?", "1 stereocenter.", stereo),
            _row("train-3", "CCN", "How many rings?", "5", ring5,
                 fmt="answer in one word"),
        ]

    def _target(self):
        return _row("train-9", "c1ccccc1", "Ring count?", "3",
                    [Fact("ring_count", "It has 3 ring(s).", 3, "count")],
                    fmt="answer in one word")

    def test_the_target_and_its_own_molecule_are_never_candidates(self):
        target = self._target()
        pool = self._pool() + [target,
                               _row("train-8", "c1ccccc1", "Other wording?",
                                    "9", [])]
        ids = {row["id"] for row in shot_candidates(target, pool)}
        assert "train-9" not in ids           # itself
        assert "train-8" not in ids           # the same molecule, another row

    def test_a_demonstration_stating_the_targets_fact_is_dropped(self):
        target = self._target()
        pool = self._pool() + [_row("train-7", "CCCC", "Rings?", "3",
                                    [Fact("ring_count", "It has 3 ring(s).",
                                          3, "count")])]
        assert "train-7" not in {r["id"] for r in shot_candidates(target, pool)}

    def test_the_matching_format_is_preferred_over_the_matching_family(self):
        ordered = shot_candidates(self._target(), self._pool())
        # train-3 matches both format and family; train-1 only the family.
        assert ordered[0]["id"] == "train-3"
        assert ordered[1]["id"] == "train-1"
        assert ordered[-1]["id"] == "train-2"


class TestFactPolarity:
    def test_a_yes_no_row_reports_its_value(self):
        yes = [Fact("ring_membership", "atom 2 (C) is in a ring.", "yes",
                    "yesno", atoms=[2])]
        no = [Fact("fg_presence", "It contains no ketone.", "no", "yesno")]
        assert fact_polarity(yes) == "yes"
        assert fact_polarity(no) == "no"

    def test_a_count_row_has_no_polarity(self):
        assert fact_polarity([Fact("ring_count", "It has 3 ring(s).", 3,
                                   "count")]) == ""

    def test_facts_that_disagree_report_no_polarity(self):
        """A row that is both a yes and a no cannot balance anything, so it is
        excluded rather than counted as whichever fact came first."""
        mixed = [Fact("ring_membership", "atom 2 (C) is in a ring.", "yes",
                      "yesno", atoms=[2]),
                 Fact("fg_presence", "It contains no ketone.", "no", "yesno")]
        assert fact_polarity(mixed) == ""


class TestComposedText:
    def test_the_pointer_precedes_the_question_and_leaves_it_intact(self):
        row = {"question": "How many rings?", "pointer": "See the examples."}
        assert question_text(row) == "See the examples.\nHow many rings?"

    def test_a_row_without_shots_reads_exactly_as_it_was_accepted(self):
        row = {"question": "How many rings?", "pointer": ""}
        assert question_text(row) == "How many rings?"

    def test_a_single_turn_row_is_not_framed_as_a_transcript(self):
        row = {"question": "How many rings?", "pointer": "",
               "turns": [{"role": "person", "text": None},
                         {"role": "assistant", "text": "It has 3 ring(s)."}]}
        assert question_text(row) == "How many rings?"

    def test_a_clarified_row_asks_the_whole_exchange(self):
        # The opening turn is underspecified on purpose. Asked on its own it has
        # no answer, and the answer beside it names an atom it never mentioned.
        row = {"question": "Is that atom part of a ring? Please answer in one "
                           "word.",
               "pointer": "",
               "turns": [{"role": "person", "text": None},
                         {"role": "assistant", "text": "Which atom do you mean?"},
                         {"role": "person", "text": "Atom 2 (C)."},
                         {"role": "assistant", "text": "Atom 2 (C) is not in a "
                                                       "ring."}]}
        assert question_text(row) == (
            "You: Is that atom part of a ring? Please answer in one word.\n"
            "Assistant: Which atom do you mean?\n"
            "You: Atom 2 (C).")

    def test_the_pointer_still_leads_a_clarified_row(self):
        row = {"question": "Is that atom in a ring?", "pointer": "See below.",
               "turns": [{"role": "person", "text": None},
                         {"role": "assistant", "text": "Which atom do you mean?"},
                         {"role": "person", "text": "Atom 2 (C)."},
                         {"role": "assistant", "text": "It is."}]}
        assert question_text(row).startswith("See below.\nYou: Is that atom")

    def test_a_demonstration_is_numbered_and_marked_as_another_molecule(self):
        text = shot_text("How many rings?", "3", 0)
        assert text.startswith("Example 1")
        assert "different molecule" in text
        assert "Q: How many rings?" in text and "A: 3" in text

    def test_demonstration_molecules_are_rebuilt_from_the_partition_key(self):
        row = {"shots": [{"key": ASPIRIN, "question": "Rings?", "answer": "1"},
                         {"key": "not a smiles", "question": "q", "answer": "a"}]}
        shots = shot_molecules(row)
        # The unparseable key drops its shot and keeps the row.
        assert len(shots) == 1
        mol, question, answer = shots[0]
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))
        assert (question, answer) == ("Rings?", "1")
