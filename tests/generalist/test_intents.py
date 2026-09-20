"""The renderer and the intent sampler — §9.4's rendered-answer design.

The renderer is the only place a claim about a molecule can enter an answer, so
what is tested here is mostly that it says exactly what it was given and nothing
else, and that the sampler cannot declare an intent the renderer would have to
invent something to satisfy.
"""

import json
import random
import re

import pytest

from src.generalist.assistant import Fact
from src.generalist.intents import (
    FORMATS, ONE_WORD_TASKS, STRUCTURED_FORMATS, TASK_WEIGHTS, Intent,
    load_situations, sample_intent, _draw_facts, _pivot_neighbours,
)
from src.generalist.render import (FAMILY_WORDS, GLOSS, OFF_SHEET_FAMILIES,
                                   TASKS, ask_phrase, render)


def ring(atom=14, yes=True):
    text = (f"atom {atom} (C) is in a ring." if yes
            else f"atom {atom} (C) is not in a ring.")
    return Fact("ring_membership", text, "yes" if yes else "no", "yesno",
                atoms=[atom])


def size(atom=14, value=6):
    return Fact("ring_size", f"The smallest ring containing atom {atom} (C) has "
                f"{value} atoms.", value, "count", atoms=[atom])


def count(value=3):
    return Fact("ring_count", f"It has {value} ring(s).", value, "count")


def no_ring(atom=14):
    """A `ring_size` of 0, which the sheet spells without the digit."""
    return Fact("ring_size", f"atom {atom} (C) is in no ring.", 0, "count",
                atoms=[atom])


def smiles(text="CCO"):
    return Fact("smiles", f"Its canonical SMILES is {text}.", text, "text")


class TestRenderSaysOnlyWhatItWasGiven:

    def test_report_states_every_fact(self):
        facts = [ring(14), size(14)]
        out = render(facts, "report", "none", random.Random(0))
        assert len(out.statements) == 2
        for fact in facts:
            assert any(fact.text.rstrip(".").lower() in s.lower()
                       for s in out.statements)

    def test_the_reply_is_built_only_from_the_statements(self):
        out = render([ring(14)], "report", "none", random.Random(0))
        assert out.reply.strip() == out.statements[0]

    def test_compare_contrasts_two_atoms(self):
        out = render([ring(19, True), ring(3, False)], "compare", "none",
                     random.Random(0))
        assert "whereas" in out.reply
        assert len(out.statements) == 2

    def test_compare_does_not_contrast_two_facts_that_agree(self):
        # "Atom 5 (C) is in a ring, whereas atom 15 (C) is in a ring" is a
        # contrast between two identical claims — a connective nothing licenses.
        out = render([ring(5, True), ring(15, True)], "compare", "none",
                     random.Random(0))
        assert "whereas" not in out.reply
        assert "as well" in out.reply
        assert out.ask["agree"] is True

    def test_explain_adds_only_a_closed_gloss(self):
        out = render([ring(14)], "explain", "none", random.Random(0))
        assert GLOSS["ring_membership"] in out.reply
        # the gloss is not a claim about this molecule, so it is not a statement
        assert len(out.statements) == 1

    def test_fill_record_declares_the_schema_it_fills(self):
        out = render([ring(14), size(14)], "fill_record", "none",
                     random.Random(0))
        payload = json.loads(out.reply)
        assert set(payload) == set(out.skeleton)
        assert payload[[k for k in payload if "membership" in k][0]] is True

    def test_triage_groups_the_atoms_that_share_an_answer(self):
        out = render([ring(3, True), ring(7, False), ring(9, True)], "triage",
                     "none", random.Random(0))
        lowered = out.reply.lower()
        assert "atom 3 (c) and atom 9 (c) are in a ring" in lowered
        assert "atom 7 (c) is not in a ring" in lowered
        # one sentence per polarity, and no fact restated afterwards
        assert len(out.statements) == 2
        assert out.reply == " ".join(out.statements)

    def test_triage_states_them_singly_when_they_do_not_group(self):
        mixed = [ring(3, True), size(7, 5)]
        out = render(mixed, "triage", "none", random.Random(0))
        assert len(out.statements) == 2


class TestTheAskCommitsToNothing:

    def test_a_polar_fact_becomes_a_neutral_ask(self):
        assert ask_phrase(ring(14, True)) == "whether atom 14 (C) is in a ring"
        assert ask_phrase(ring(14, False)) == "whether atom 14 (C) is in a ring"

    def test_the_ask_names_the_atom_and_the_group(self):
        fact = Fact("fg_atom_membership",
                    "atom 7 (O) is not part of a hydroxyl group.", "no",
                    "yesno", atoms=[7])
        phrase = ask_phrase(fact)
        assert phrase == "whether atom 7 (O) is part of a hydroxyl"
        assert " not " not in phrase

    def test_a_group_ask_carries_its_article(self):
        # `_group_of` strips the article off the sheet's wording, and "whether it
        # contains nitrile" is not a sentence anyone types
        fact = Fact("fg_presence", "It contains no nitrile.", "no", "yesno")
        assert ask_phrase(fact) == "whether it contains a nitrile"
        ether = Fact("fg_presence", "It contains an ether.", "yes", "yesno")
        assert ask_phrase(ether) == "whether it contains an ether"

    def test_an_assay_ask_drops_the_polarity(self):
        fact = Fact("tier_b/tox21", "It is not active in the SR ATAD5 assay.",
                    "no", "yesno")
        assert ask_phrase(fact) == "whether it is active in the SR ATAD5 assay"

    def test_a_group_count_ask_pluralises_rather_than_appending(self):
        # "It contains 2 hydroxyl group(s)." asked as "how many hydroxyl group
        # groups it contains" is what pasting "groups" onto the sheet's wording
        # gives you
        fact = Fact("fg_count", "It contains 2 hydroxyl group(s).", 2, "count")
        assert ask_phrase(fact) == "how many hydroxyls it contains"
        halogen = Fact("fg_count", "It contains 3 halogen(s).", 3, "count")
        assert ask_phrase(halogen) == "how many halogens it contains"

    def test_an_aromatic_count_asks_about_aromaticity(self):
        # this family covers both a per-atom yes/no and a whole-molecule count,
        # and the count shape has no atom to fall back on
        fact = Fact("aromatic_ring", "2 of its rings are aromatic.", 2, "count")
        assert ask_phrase(fact) == "how many of its rings are aromatic"
        atom = Fact("aromatic_ring", "atom 2 (C) is in an aromatic ring.", "yes",
                    "yesno", atoms=[2])
        assert ask_phrase(atom) == "whether atom 2 (C) is in an aromatic ring"

    def test_a_count_ask_carries_no_number(self):
        assert not re.search(r"\d", ask_phrase(count(3)))
        assert not re.search(r"\b6\b", ask_phrase(size(7, 6)))

    def test_the_render_publishes_asks_and_anchors(self):
        out = render([ring(14), size(14)], "report", "none", random.Random(0))
        assert out.ask["anchors"] == ["atom 14 (C)"]
        assert len(out.ask["asks"]) == 2
        for phrase in out.ask["asks"]:
            assert "6" not in phrase

    def test_an_unanswerable_ask_asks_for_the_missing_family(self):
        out = render([ring(14)], "report", "unanswerable", random.Random(0),
                     spare_family="ring_count")
        assert out.ask["missing_family"] == "ring_count"
        assert out.ask["asks"] == ["how many rings it has"]


class TestTwists:

    def test_a_false_premise_contradicts_the_fact_it_is_built_from(self):
        fact = ring(14, True)
        out = render([fact], "check_claim", "false_premise", random.Random(0))
        assert out.ask["claim_is_true"] is False
        assert "is not in a ring" in out.ask["claim"].lower()
        # and the reply corrects it with the sheet's own wording
        assert "isn't right" in out.reply.lower()
        assert out.statements == ["Atom 14 (C) is in a ring."]

    def test_a_false_premise_on_a_count_moves_the_number(self):
        out = render([count(3)], "check_claim", "false_premise",
                     random.Random(0))
        assert "4" in out.ask["claim"]
        assert out.statements == ["It has 3 ring(s)."]

    def test_a_false_premise_on_a_verb_carried_polarity_is_not_a_double_negative(self):
        # "It shows no activity against HIV replication." hides its polarity in
        # the verb, so the flip has to reach the verb. Wrapping it instead gave
        # the person "I believe this molecule does not show no activity", and
        # the writer answered that with "Correct", which means nothing.
        fact = Fact("tier_b/hiv", "It shows no activity against HIV "
                    "replication.", "no", "yesno")
        out = render([fact], "check_claim", "false_premise", random.Random(0))
        claim = out.ask["claim"].lower()
        assert "shows activity against hiv replication" in claim
        assert "not the case" not in claim
        assert "no activity" not in claim
        # and the ask itself commits to neither polarity
        assert "no activity" not in ask_phrase(fact)

    def test_a_true_claim_is_confirmed(self):
        out = render([ring(14)], "check_claim", "none", random.Random(0))
        assert out.ask["claim_is_true"] is True
        assert "that's right" in out.reply.lower()

    def test_an_unanswerable_ask_refuses_and_claims_nothing_about_it(self):
        out = render([ring(14)], "report", "unanswerable", random.Random(0),
                     spare_family="ring_count")
        assert out.ask["answerable"] is False
        assert "can't tell you" in out.reply.lower()
        # the refusal is not a claim, so the only statement is the fact it does
        # carry — nothing asserts a ring count
        assert out.statements == ["Atom 14 (C) is in a ring."]
        assert "ring(s)" not in out.reply

    def test_every_off_sheet_family_has_words_and_renders(self):
        # The pool exists to give the unanswerable twist more than one sentence.
        # A family with no entry in FAMILY_WORDS falls back to "that property"
        # and puts the twist straight back where it was.
        rng = random.Random(0)
        for family in OFF_SHEET_FAMILIES:
            assert family in FAMILY_WORDS
            out = render([count(2)], "report", "unanswerable", rng,
                         spare_family=family)
            assert out.ask["answerable"] is False
            assert FAMILY_WORDS[family] in out.reply
            # the refusal states the fact it does have and claims nothing else
            assert out.statements == ["It has 2 ring(s)."]

    def test_an_unanswerable_ask_is_not_always_the_same_ask(self):
        sheet = [ring(14), size(14), count(2)]
        situations = load_situations()
        rng = random.Random(0)
        asks = set()
        for _ in range(400):
            drawn = sample_intent("m", "CCO", sheet, situations, rng)
            if drawn and drawn[0].twist == "unanswerable":
                asks.add(tuple(drawn[1].ask["asks"]))
        assert len(asks) >= 6, f"unanswerable collapsed to {len(asks)} ask(s)"

    def test_a_clarification_is_three_turns_before_the_answer(self):
        out = render([ring(14)], "report", "needs_clarification",
                     random.Random(0))
        roles = [role for role, _ in out.turns]
        assert roles == ["person", "assistant", "person", "assistant"]
        assert out.turns[1][1].endswith("?")
        assert out.turns[-1][1] == out.reply

    def test_the_clarified_atom_is_found_mid_sentence(self):
        # a ring_size fact does not open with its atom, and anchoring the search
        # at the front asked "which atom do you mean" and answered "The atom."
        out = render([size(7, 6)], "report", "needs_clarification",
                     random.Random(0))
        assert out.ask["anchor"].lower().startswith("atom 7")
        assert out.turns[2][1] == "Atom 7 (C)."

    def test_clarification_needs_one_atom_across_its_facts(self):
        from src.generalist.render import can_clarify
        assert can_clarify([ring(14), size(14)])
        assert not can_clarify([ring(14), ring(3)])
        assert not can_clarify([count(2)])
        with pytest.raises(ValueError):
            render([ring(14), ring(3)], "report", "needs_clarification",
                   random.Random(0))

    def test_compare_does_not_take_a_clarification(self):
        assert "needs_clarification" not in TASKS["compare"][2]

    def test_an_unanswerable_record_field_is_null_and_says_why(self):
        out = render([count(2)], "fill_record", "unanswerable",
                     random.Random(0), spare_family="stereo_potential")
        head = out.reply.split("\n\n")[0]
        payload = json.loads(head)
        assert payload["stereo_potential"] is None
        assert "stereo_potential" in out.skeleton
        assert "don't have" in out.reply
        assert out.ask["answerable"] is False
        # the null is not a claim, so the only statement is the fact it has
        assert out.statements == ["It has 2 ring(s)."]

    def test_decide_renders_the_false_premise_it_accepts(self):
        # a twist a task accepts and never renders is a declared cell that is
        # silently empty, which is the failure mode the coverage report exists
        # for and the one `fill_record` already had
        out = render([ring(14, True)], "decide", "false_premise",
                     random.Random(0))
        assert out.ask["claim_is_true"] is False
        assert "is not in a ring" in out.ask["claim"].lower()
        assert "isn't right" in out.reply.lower()
        assert out.ask["constraint"]
        assert out.statements == ["Atom 14 (C) is in a ring."]

    def test_a_decide_verdict_agrees_with_the_constraint_it_states(self):
        # The bug this pins: the constraint was built as `_clause(pivot)` when
        # the person wanted the property to hold, but `_clause` already carries
        # the *value's* polarity. On a fact whose value is "no" that renders the
        # constraint negated while the verdict was computed for the positive,
        # and the reply then contradicts its own statement — "Yes, that one
        # qualifies. Atom 9 (C) is not in an aromatic ring."
        rng = random.Random(0)
        seen = set()
        for _ in range(200):
            for holds_in_fact in (True, False):
                fact = ring(9, holds_in_fact)
                out = render([fact], "decide", "none", rng)
                constraint = out.ask["constraint"]
                wants_yes = " not " not in constraint
                assert out.ask["verdict"] is (holds_in_fact == wants_yes)
                assert out.verdict == ("yes" if out.ask["verdict"] else "no")
                # and the rendered reply says the same thing twice over
                qualifies = "that one qualifies" in out.reply and \
                    "doesn't qualify" not in out.reply
                assert qualifies is out.ask["verdict"]
                seen.add((holds_in_fact, wants_yes))
        assert len(seen) == 4, "all four polarity pairings should be drawn"

    def test_a_negated_group_constraint_keeps_its_article(self):
        # The constraint goes into the person's turn verbatim, so "I only want
        # it if It contains primary amine" is a defect the reader sees.
        fg = Fact("fg_presence", "It contains no primary amine.", "no", "yesno")
        rng = random.Random(3)
        seen = set()
        for _ in range(40):
            constraint = render([fg], "decide", "none", rng).ask["constraint"]
            assert "if It " not in constraint
            assert "contains primary" not in constraint
            seen.add("contains a primary amine" in constraint)
        assert seen == {True, False}, "both polarities should be drawn"

    def test_a_draw_does_not_state_one_fact_twice(self):
        # ring_size 0 and ring_membership "no" about one atom are the same
        # sentence: "Atom 1 (Sn) is in no ring. Atom 1 (Sn) is not in a ring."
        size = Fact("ring_size", "Atom 1 (Sn) is in no ring.", 0, "count",
                    atoms=[1])
        member = Fact("ring_membership", "atom 1 (Sn) is not in a ring.", "no",
                      "yesno", atoms=[1])
        for _ in range(20):
            drawn = _draw_facts([size, member], "explain", random.Random(0))
            assert len(drawn) == 1
        # a nonzero size beside a "yes" membership is not redundant, though
        size6 = Fact("ring_size", "The smallest ring containing atom 1 (C) has "
                     "6 atoms.", 6, "count", atoms=[1])
        yes = Fact("ring_membership", "atom 1 (C) is in a ring.", "yes",
                   "yesno", atoms=[1])
        rng = random.Random(0)
        assert any(len(_draw_facts([size6, yes], "explain", rng)) == 2
                   for _ in range(20))

    def test_a_record_gives_every_fact_its_own_field(self):
        # Three assay facts all keyed `tier_b_tox21` left one field in the JSON
        # and two statements with nowhere to be stated.
        tox = [Fact("tier_b/tox21", f"It is not active in the {assay} assay.",
                    "no", "yesno")
               for assay in ("NR AR LBD", "SR p53", "NR AR")]
        out = render(tox, "fill_record", "none", random.Random(0))
        assert len(json.loads(out.reply)) == len(out.statements) == 3
        assert len(out.skeleton) == 3

        groups = [Fact("fg_count", "It contains 2 hydroxyl group(s).", 2,
                       "count"),
                  Fact("fg_count", "It contains 1 amide(s).", 1, "count")]
        out = render(groups, "fill_record", "none", random.Random(0))
        payload = json.loads(out.reply)
        assert payload == {"fg_count_hydroxyl": 2, "fg_count_amide": 1}

    def test_decide_refuses_a_pivot_it_has_no_predicate_for(self):
        caption = Fact("caption", "The molecule is an alkaloid.",
                       "The molecule is an alkaloid.", "text")
        with pytest.raises(ValueError):
            render([caption], "decide", "none", random.Random(0))
        # and the sampler never hands it one: a sheet of nothing but captions
        # supports no `decide` at all
        assert _draw_facts([caption], "decide", random.Random(0)) == []

    def test_a_count_constraint_is_never_negative(self):
        rng = random.Random(0)
        for _ in range(60):
            out = render([count(0)], "decide", "none", rng)
            assert "-" not in out.ask["constraint"]
            # Nor a threshold every molecule meets, which makes the decision
            # unanswerable-by-being-trivial rather than unanswerable.
            assert "at least 0" not in out.ask["constraint"]

    def test_a_task_refuses_a_twist_it_does_not_take(self):
        with pytest.raises(ValueError):
            render([ring(14)], "summarise", "false_premise", random.Random(0))

    def test_a_task_refuses_the_wrong_number_of_facts(self):
        with pytest.raises(ValueError):
            render([ring(14)], "compare", "none", random.Random(0))


class TestAClauseReadsAsOneSentence:
    """Clauses are joined mid-sentence, and the sheet capitalises some of them."""

    def test_a_comparison_does_not_capitalise_its_second_clause(self):
        # "Atom 1 (C) is in no ring, whereas The smallest ring containing atom
        # 19 (C) has 6 atoms." Both connectives take the same clause, so both
        # are checked: the agreeing pair joins with "and ... as well".
        for left in (ring(1, False), ring(1, True)):
            out = render([left, size(19)], "compare", "none", random.Random(0))
            assert ", whereas The" not in out.reply
            assert ", and The" not in out.reply

    def test_a_count_constraint_names_the_number_it_is_about(self):
        rng = random.Random(0)
        groups = Fact("fg_count", "It contains 4 ether(s).", 4, "count")
        assert "the number of ethers" in render(
            [groups, count(2)], "decide", "none", rng).ask["constraint"]
        assert "the number of rings" in render(
            [count(2)], "decide", "none", rng).ask["constraint"]
        assert "the size of the smallest ring containing atom 14 (C)" in render(
            [size(14)], "decide", "none", rng).ask["constraint"]

    def test_a_constraint_without_a_name_still_states_a_number(self):
        # The fallback is the old wording. An unnamed number is ambiguous beside
        # a second count; it is never wrong.
        odd = Fact("unheard_of", "It has 3 widget(s).", 3, "count")
        out = render([odd], "decide", "none", random.Random(0))
        assert "that number" in out.ask["constraint"]


class TestTheReplyOwesMoreThanItsStatements:
    """Two things a reply must carry that no statement licenses."""

    def test_an_explain_render_declares_its_gloss(self):
        # In `ask` it would be invisible to the accept pass and the composed
        # set's re-check, both of which read the row and not the ask.
        out = render([ring(14)], "explain", "none", random.Random(0))
        assert out.gloss and out.gloss.startswith("In general,")
        assert out.gloss in out.reply
        assert out.to_json()["gloss"] == out.gloss
        # It is emphatically not a statement: it says nothing about *this*
        # molecule, and counting it as one would put it in a numbered list's
        # positional answers.
        assert all("In general" not in s for s in out.statements)

    def test_a_task_with_no_gloss_declares_none(self):
        out = render([ring(14)], "report", "none", random.Random(0))
        assert out.gloss is None
        assert "gloss" not in out.to_json()

    def test_a_decision_is_never_one_word(self):
        # Two questions — what the value is, and whether it meets the constraint
        # — with two different answers. One word gives the verdict, so the word
        # answers the question that was not asked first, and where the constraint
        # runs against the value it reads as its opposite.
        for twist in ("none", "false_premise"):
            for facts in ([ring(9)], [count(2)]):
                out = render(facts, "decide", twist, random.Random(0))
                assert not FORMATS["one word"](out)

    def test_the_one_word_verdict_can_contradict_the_value_asked(self):
        # Why the guard is on the task and not on the twist. Nothing here is a
        # false premise, and the verdict is still the opposite of the answer to
        # the question the ask puts first.
        found = False
        for seed in range(40):
            out = render([ring(9, False)], "decide", "none", random.Random(seed))
            if out.verdict and out.answers[0] != out.verdict:
                found = True
                assert not FORMATS["one word"](out)
        assert found, "no draw put the verdict against the value"

    def test_a_check_over_a_false_premise_still_may_be(self):
        # One question, and one word answers it: "No", "Three".
        out = render([ring(9)], "check_claim", "false_premise", random.Random(0))
        assert FORMATS["one word"](out)


class TestAFalsePremiseIsFalse:
    """The claim a `false_premise` ask puts in the person's mouth is wrong.

    Obvious, and it was not true: `_negate` on a count swaps the fact's own digit
    into its own sentence, and a `ring_size` of 0 is spelled "atom 14 (C) is in
    no ring" with no digit anywhere in it. The substitution was a silent no-op,
    so 39 rows of the final build asserted something true and were told it was
    wrong.
    """

    def test_a_count_whose_sentence_hides_its_digit_is_still_contradicted(self):
        fact = no_ring(14)
        out = render([fact], "check_claim", "false_premise", random.Random(0))
        claim = out.ask["claim"].lower().rstrip(". ")
        assert claim != fact.text.lower().rstrip(". ")
        assert "not the case" in claim

    def test_no_false_premise_render_claims_one_of_its_own_statements(self):
        def normalised(text):
            return " ".join(text.lower().replace(".", " ").split())

        for facts in ([no_ring(14)], [no_ring(14), ring(14, False)],
                      [size(14)], [count(0)], [count(3)], [ring(9)],
                      [ring(9, False)]):
            for task in ("check_claim", "decide"):
                out = render(facts, task, "false_premise", random.Random(0))
                claim = normalised(out.ask.get("claim") or "")
                if not claim:
                    continue
                assert all(claim != normalised(s) for s in out.statements), \
                    f"{task} {facts[0].text}"


class TestTheClaimIsNotTheWholeMolecule:
    """A `check_claim` or `decide` pivot has to be something a person asserts.

    The canonical SMILES is not. "I believe its canonical SMILES is
    COc1ccccc1N1C(=O)..." writes the molecule into the question and the reply
    copies it back, so the row is answerable without the graph — 88 rows of the
    final build. `render` refuses the draw and `sample_intent` takes another
    task.
    """

    def test_render_refuses_a_smiles_pivot(self):
        for task in ("check_claim", "decide"):
            for twist in ("none", "false_premise"):
                with pytest.raises(ValueError, match="pivot"):
                    render([smiles()], task, twist, random.Random(0))

    def test_other_tasks_may_still_state_it(self):
        out = render([smiles()], "report", "none", random.Random(0))
        assert "CCO" in out.reply

    def test_the_sampler_draws_a_different_task_rather_than_failing(self):
        sheet = [smiles("CCO"), ring(14), size(14)]
        situations = load_situations()
        rng = random.Random(1)
        for _ in range(200):
            out = sample_intent("m", "CCO", sheet, situations, rng)
            if out is None:
                continue
            intent, _ = out
            if intent.task in ("check_claim", "decide"):
                assert intent.facts[0].family != "smiles"


class TestTheSamplerCannotDeclareTheImpossible:

    SHEET = [ring(14), size(14), ring(3, False), count(2),
             Fact("fg_presence", "It contains no nitrile.", "no", "yesno"),
             Fact("fg_atom_membership", "atom 14 (C) is not part of an ether.",
                  "no", "yesno", atoms=[14])]

    def test_every_drawn_format_is_satisfiable_by_its_own_render(self):
        situations = load_situations()
        rng = random.Random(7)
        drawn = 0
        for _ in range(400):
            out = sample_intent("m", "CCO", self.SHEET, situations, rng)
            if out is None:
                continue
            intent, rendered = out
            drawn += 1
            assert FORMATS[intent.style["format"]](rendered), intent.cell()
        assert drawn > 350

    def test_a_multi_fact_draw_shares_a_family_or_an_atom(self):
        rng = random.Random(3)
        for _ in range(300):
            task = "report"
            facts = _draw_facts(self.SHEET, task, rng)
            if len(facts) < 2:
                continue
            pivot = facts[0]
            for other in facts[1:]:
                assert (other.family == pivot.family
                        or (pivot.atoms and set(other.atoms) & set(pivot.atoms)))

    def test_one_word_is_never_drawn_for_a_refusal_or_a_sentence(self):
        situations = load_situations()
        rng = random.Random(23)
        for _ in range(800):
            out = sample_intent("m", "CCO", self.SHEET, situations, rng)
            if out is None:
                continue
            intent, rendered = out
            if intent.style["format"] != "one word":
                continue
            assert rendered.ask["answerable"] is True
            assert len(rendered.statements) == 1
            assert intent.facts[0].kind in ("yesno", "count")
            # and not for a task whose ask is plural — "which of these qualify"
            # has no one-word answer even when the facts happen to group into one
            # statement
            assert intent.task in ONE_WORD_TASKS

    def test_no_sentence_count_is_asked_of_a_format_with_no_sentences(self):
        situations = load_situations()
        rng = random.Random(31)
        for _ in range(600):
            out = sample_intent("m", "CCO", self.SHEET, situations, rng)
            if out is None:
                continue
            intent, _ = out
            if intent.style["format"] in STRUCTURED_FORMATS:
                assert intent.style["length"] == "as short as possible"

    def test_an_unanswerable_family_is_one_the_sheet_lacks(self):
        situations = load_situations()
        rng = random.Random(11)
        seen = 0
        for _ in range(600):
            out = sample_intent("m", "CCO", self.SHEET, situations, rng)
            if out is None:
                continue
            intent, _ = out
            if intent.twist != "unanswerable":
                continue
            seen += 1
            assert intent.spare_family not in {f.family for f in self.SHEET}
        assert seen > 0

    def test_a_record_is_never_asked_for_in_prose(self):
        situations = load_situations()
        rng = random.Random(17)
        seen = 0
        for _ in range(800):
            out = sample_intent("m", "CCO", self.SHEET, situations, rng)
            if out is None:
                continue
            intent, rendered = out
            if rendered.skeleton is None:
                continue
            seen += 1
            # `_fill_record` writes its reply as JSON, so the only brief that
            # agrees with it is the JSON one.
            assert intent.style["format"] == "JSON matching the given schema"
        assert seen > 0

    def test_a_sheet_too_thin_for_a_task_does_not_hang(self):
        situations = load_situations()
        rng = random.Random(5)
        thin = [count(1)]
        for _ in range(200):
            out = sample_intent("m", "C", thin, situations, rng)
            if out is None:
                continue
            intent, rendered = out
            low, high = TASKS[intent.task][1]
            assert low <= len(intent.facts) <= high

    def test_an_intent_round_trips(self):
        situations = load_situations()
        rng = random.Random(1)
        intent, _ = sample_intent("m", "CCO", self.SHEET, situations, rng)
        again = Intent.from_json(json.loads(json.dumps(intent.to_json())))
        assert again.to_json() == intent.to_json()

    def test_sampling_is_deterministic_in_its_seed(self):
        situations = load_situations()
        first = sample_intent("m", "CCO", self.SHEET, situations,
                              random.Random(42))[0].to_json()
        second = sample_intent("m", "CCO", self.SHEET, situations,
                               random.Random(42))[0].to_json()
        assert first == second


class TestTheBanksAreReadable:

    def test_every_situation_has_a_persona_and_a_context(self):
        for situation in load_situations():
            assert situation["persona"] and situation["context"]
            assert situation["id"]

    def test_situation_ids_are_unique(self):
        ids = [s["id"] for s in load_situations()]
        assert len(ids) == len(set(ids))

    def test_every_task_has_a_weight_and_every_weight_a_task(self):
        assert set(TASK_WEIGHTS) == set(TASKS)

    def test_every_family_a_gloss_names_is_one_the_sheet_uses(self):
        # a gloss for a family nothing renders is a line nobody reads
        assert "bond_path" not in GLOSS and "longest_chain" not in GLOSS
