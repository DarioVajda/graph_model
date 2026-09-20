"""The accept pass's three checks — §9.4's verification after the redesign.

The thing worth testing here is not that the checks catch defects; it is that
they do not catch anything else. Five times now a filter in this section has been
quietly discarding correct rows, and every time the symptom was a yield that
looked plausible, so every check below has at least one test of a *correct* row
it must not reject. Most of these cases are transcribed from the first smoke
build's rejection log, where ten of twelve rows read by hand were correct.
"""

from src.generalist.tools.intent_accept import (
    _anchor_named, _leaked_statements, asks_for_a_joint_ring, carries_token,
    dropped_statements, declines, format_met, invents_a_structure,
    pivot_is_unassertable, premise_is_not_false, read_order,
    statement_survives, states_the_gloss, states_the_verdict,
)


def rendered(statements, answers, verdict=None, skeleton=None):
    out = {"statements": statements, "answers": answers}
    if verdict is not None:
        out["verdict"] = verdict
    if skeleton is not None:
        out["skeleton"] = skeleton
    return out


#: Transcribed from the second smoke build's rejection log, where six of eight
#: rows read by hand were correct and the other two were renderer bugs rather
#: than writer bugs.
class TestTheSecondSmokeBuildsRejections:

    def test_a_colloquial_yes_or_no_is_still_an_answer(self):
        # "1. Nope\n2. Nope\n3. None" answered all three asks it was given
        render = rendered(["Atom 8 (C) is not part of a primary amine.",
                           "Atom 8 (C) is not in an aromatic ring.",
                           "Atom 8 (C) is in no ring."], ["no", "no", "0"])
        assert not dropped_statements(
            render, "1. Nope\n2. Nope\n3. None", "a numbered list")
        assert carries_token("Yep", "yes") and carries_token("Nope", "no")

    def test_a_coordination_is_not_three_clauses(self):
        # The comma in "the NR AR LBD, SR p53, or NR AR assays" is not a clause
        # boundary. Split there, no fragment carried enough of any statement's
        # content words to be a candidate, and all three were called dropped.
        render = rendered(["It is not active in the NR AR LBD assay.",
                           "It is not active in the SR p53 assay.",
                           "It is not active in the NR AR assay."],
                          ["no", "no", "no"])
        assert not dropped_statements(
            render, "No, it's not active in any of them. It isn't active in "
            "the NR AR LBD, SR p53, or NR AR assays.", "prose")

    def test_a_comma_that_does_separate_two_clauses_still_splits(self):
        # The counterpart the rule above must not give away: a comma splice
        # between two predications keeps its boundary, so one "not" cannot
        # negate the fact on the other side of it.
        render = rendered(["Atom 5 (C) is in an aromatic ring.",
                           "Atom 9 (O) is not in an aromatic ring."],
                          ["yes", "no"])
        assert not dropped_statements(
            render, "Atom 5 (C) is in an aromatic ring, atom 9 (O) is not.",
            "prose")
        assert dropped_statements(
            render, "Atom 5 (C) is not in an aromatic ring, atom 9 (O) is not.",
            "prose")

    def test_a_description_is_preserved_across_a_reordering(self):
        # A caption has no answer token, so there is no polarity to mis-read and
        # no reason to narrow to a clause — and no clause of a re-voiced
        # paragraph carries 70% of the paragraph.
        statement = ("The molecule is a monocarboxylic acid that is acetic acid "
                     "in which one of the methyl hydrogens is replaced by a "
                     "3-(4-chlorophenyl)-1-phenylpyrazol-4-yl group. It has a "
                     "role as a non-steroidal anti-inflammatory drug and an "
                     "antineoplastic agent. It is a member of pyrazoles.")
        reply = ("This monocarboxylic acid is acetic acid with one methyl "
                 "hydrogen replaced by a 3-(4-chlorophenyl)-1-phenylpyrazol-4-yl "
                 "group. It is a member of pyrazoles. It acts as a non-steroidal "
                 "anti-inflammatory drug and an antineoplastic agent.")
        assert not dropped_statements(rendered([statement], [None]), reply,
                                      "prose")
        assert dropped_statements(rendered([statement], [None]),
                                  "It is a member of pyrazoles.", "prose")


#: The third smoke build's rejection log, where all seven `statement_dropped`
#: rows read by hand were correct. Three causes: a conjunction inside a
#: coordination read as a clause boundary, a bare "no." carried into the clause
#: after it, and a list whose items are fewer than its statements.
class TestTheThirdSmokeBuildsRejections:

    def test_a_conjunction_inside_a_coordination_is_not_a_boundary(self):
        render = rendered(["It is not active in the SR ARE assay.",
                           "It is not active in the SR ATAD5 assay.",
                           "It is not active in the NR AhR assay."],
                          ["no", "no", "no"])
        assert not dropped_statements(
            render, "The molecule is inactive in the SR ARE and SR ATAD5 "
            "assays. Additionally, it shows no activity in the NR AhR assay.",
            "prose")

    def test_a_long_coordinated_noun_phrase_stays_one_clause(self):
        statement = ("It has no reported side effects in the Musculoskeletal "
                     "and connective tissue disorders class.")
        render = rendered([statement, "It has no reported side effects in the "
                           "Endocrine disorders class."], ["no", "no"])
        assert not dropped_statements(
            render, "No reported side effects in the Musculoskeletal and "
            "connective tissue disorders or Endocrine disorders classes.",
            "prose")

    def test_a_leading_verdict_does_not_negate_the_clause_after_it(self):
        # "Yes and no." is the lead-with-the-answer opener. Merged into the
        # sentence that follows it, its "no" negated a positive fact.
        render = rendered(["Atom 9 (C) and atom 20 (C) are in an aromatic ring.",
                           "Atom 13 (O) is not in an aromatic ring."],
                          ["yes", "no"])
        assert not dropped_statements(
            render, "Yes and no. Atom 9 (C) and 20 (C) are aromatic; atom 13 "
            "(O) isn't.", "prose")

        render = rendered(["Atom 6 (C) is in a ring.",
                           "Atom 3 (C) is not in a ring."], ["yes", "no"])
        assert not dropped_statements(
            render, "Yes and no. Atom 6 (C) is in a ring, but atom 3 (C) "
            "isn't.", "prose")

    def test_an_elided_predicate_is_not_merged_into_the_next_sentence(self):
        # "atom 8 (C) is." is complete; folding it into the sentence after it
        # borrowed that sentence's "not".
        render = rendered(["Atom 1 (C) is not in an aromatic ring.",
                           "Atom 8 (C) is in an aromatic ring.",
                           "Atom 1 (C) is in no ring."], ["no", "yes", "0"])
        assert not dropped_statements(
            render, "Atom 1 (C) is not aromatic, while atom 8 (C) is. Atom 1 "
            "(C) is not contained within any ring.", "prose")

    def test_the_format_request_is_not_an_answer(self):
        # "Please answer in one word" leaked the count 1 through "one"; a
        # rendered JSON schema leaked both the atom and the size through the
        # key `ring_size_atom_5`.
        render = rendered(["1 of its rings is aromatic."], ["1"])
        assert not _leaked_statements(
            render, "How many of the rings in this molecule are aromatic? "
            "Please answer in one word.", "report", "one word")

        render = rendered(["The smallest ring containing atom 5 (C) has 5 "
                           "atoms."], ["5"])
        assert not _leaked_statements(
            render, "What is the size of the smallest ring containing atom 5 "
            '(C)? Please provide the answer as JSON matching the schema: '
            '{"ring_size_atom_5": "integer"}.', "fill_record",
            "JSON matching the given schema")

        # a turn that really does state the count is still refused
        assert _leaked_statements(
            render, "The smallest ring here has 5 atoms, right?", "report", "")

    def test_a_structure_the_render_states_is_not_an_invention(self):
        smiles = "CNC(=O)CNC(=O)CCC(=O)OC"
        turn = f"I believe the canonical SMILES is {smiles}; can you check?"
        assert invents_a_structure(turn)
        assert not invents_a_structure(turn, f"Its canonical SMILES is {smiles}.")

    def test_a_list_shorter_than_its_statements_is_read_as_prose(self):
        # The question bundled two asks into one, so the writer answered them
        # in one item. Nothing was dropped.
        render = rendered(["Atom 12 (N) is not part of an ether.",
                           "Atom 12 (N) is in a ring.",
                           "The smallest ring containing atom 12 (N) has 6 "
                           "atoms."], ["no", "yes", "6"])
        assert not dropped_statements(render, "1. No\n2. Yes, 6",
                                      "a numbered list")
        # but a list that really has dropped the count still fails
        assert dropped_statements(render, "1. No\n2. Yes", "a numbered list")


class TestCarriesToken:

    def test_a_count_survives_spelled_out(self):
        assert carries_token("Two of the molecule's rings are aromatic.", "2")
        assert not carries_token("Three of its rings are aromatic.", "2")

    def test_an_atom_index_is_not_a_count(self):
        assert not carries_token("Atom 2 (C) is in a ring.", "2")

    def test_a_number_inside_a_name_is_not_a_count(self):
        # "SR ATAD5" made the first build read the assay's name as the answer
        assert not carries_token("Inactive in the SR ATAD5 assay.", "5")

    def test_a_yes_survives_as_a_restatement(self):
        assert carries_token("Yes", "yes")
        assert carries_token("Atom 14 (C) is in a ring.", "yes")
        assert not carries_token("Atom 14 (C) is not in a ring.", "yes")

    def test_a_no_survives_in_its_many_spellings(self):
        for text in ("No", "Not in a ring", "Atom 8 (C) isn't.", "Inactive",
                     '{"ring_atom_8": false}'):
            assert carries_token(text, "no"), text
        assert not carries_token("Yes, it is.", "no")


class TestDroppedStatements:

    def test_a_positional_list_answers_by_position(self):
        # "1. No\n2. No" states both statements; read as prose it states neither
        render = rendered(["Atom 11 (C) is not part of an ether.",
                           "Atom 1 (C) is not part of an ether."],
                          ["no", "no"])
        assert not dropped_statements(render, "1. No\n2. No", "a numbered list")

    def test_a_positional_list_with_an_ellipsis_still_answers(self):
        render = rendered(["Atom 8 (C) is not in a ring.",
                           "Atom 13 (C) is in a ring."], ["no", "yes"])
        assert not dropped_statements(
            render, "1. Atom 8 (C) isn't.\n2. Atom 13 (C) is.",
            "a numbered list")

    def test_a_positional_list_that_drops_an_item_is_caught(self):
        render = rendered(["Atom 8 (C) is not in a ring.",
                           "Atom 13 (C) is in a ring."], ["no", "yes"])
        assert dropped_statements(render, "1. Atom 8 (C) isn't.",
                                  "a numbered list")

    def test_a_positional_list_that_flips_a_polarity_is_caught(self):
        render = rendered(["Atom 8 (C) is not in a ring.",
                           "Atom 13 (C) is in a ring."], ["no", "yes"])
        assert dropped_statements(render, "1. Yes\n2. Yes", "a numbered list")

    def test_one_word_carries_the_verdict_not_the_facts(self):
        # "I only want it if atom 14 (O) is not in an aromatic ring" — it isn't,
        # so the one-word answer is "Yes" and the statement behind it is negative
        render = rendered(["Atom 14 (O) is not in an aromatic ring."], ["no"],
                          verdict="yes")
        assert not dropped_statements(render, "Yes", "one word")
        assert dropped_statements(render, "No", "one word")

    def test_a_terse_reply_carries_the_value(self):
        render = rendered(["It has 4 ring(s)."], ["4"])
        assert not dropped_statements(render, "4", "one word")
        assert dropped_statements(render, "3", "one word")

    def test_a_json_record_preserves_its_booleans(self):
        render = rendered(["Atom 14 (C) is in a ring."], ["yes"],
                          skeleton={"ring_atom_14": "boolean"})
        fmt = "JSON matching the given schema"
        assert not dropped_statements(render, '{"ring_atom_14": true}', fmt)
        assert dropped_statements(render, '{"ring_atom_14": false}', fmt)

    def test_prose_survives_a_rewording(self):
        render = rendered(["It has 3 ring(s)."], ["3"])
        assert not dropped_statements(
            render, "The molecule meets the specified constraint. It contains "
            "3 rings.", "prose")

    def test_prose_survives_a_plural(self):
        render = rendered(["2 of its rings are aromatic."], ["2"])
        assert not dropped_statements(
            render, "Two of the molecule's rings exhibit aromaticity.", "prose")

    def test_prose_reads_polarity_on_the_clause_that_carries_it(self):
        # one "not" in a compound answer must not negate the other statement
        render = rendered(["Atom 5 (C) is in a ring.",
                           "Atom 9 (C) is not in a ring."], ["yes", "no"])
        assert not dropped_statements(
            render, "Atom 5 (C) is in a ring, whereas atom 9 (C) is not.",
            "prose")

    def test_a_zero_count_survives_as_an_absence(self):
        # the sheet writes a `ring_size` of 0 as "is in no ring", so a reply that
        # says "None" has stated the number
        render = rendered(["Atom 1 (N) is in no ring.",
                           "The smallest ring containing atom 17 (C) has 6 "
                           "atoms."], ["0", "6"])
        assert not dropped_statements(
            render, "None and 6. Atom 1 (N) is in no ring; the smallest ring "
            "containing atom 17 (C) has 6 atoms.", "prose")

    def test_an_elided_second_clause_still_answers(self):
        render = rendered(["Atom 10 (C) is in an aromatic ring.",
                           "Atom 17 (C) is not in an aromatic ring."],
                          ["yes", "no"])
        assert not dropped_statements(
            render, "Only atom 10 (C) is in an aromatic ring, while atom 17 (C) "
            "is not.", "prose")

    def test_a_compound_subject_is_not_split_off_its_predicate(self):
        render = rendered(["Atom 26 (C) is in an aromatic ring.",
                           "Atom 21 (O) and atom 3 (C) are not in an aromatic "
                           "ring."], ["yes", "no"])
        assert not dropped_statements(
            render, "Atom 26 (C) is in an aromatic ring, whereas atom 21 (O) "
            "and atom 3 (C) are not.", "prose")

    def test_a_pronoun_still_carries_its_count(self):
        render = rendered(["Atom 17 (C) is not in an aromatic ring.",
                           "The smallest ring containing atom 17 (C) has 5 "
                           "atoms."], ["no", "5"])
        assert not dropped_statements(
            render, "Atom 17 (C) is not in an aromatic ring, and the smallest "
            "ring containing it consists of 5 atoms.", "prose")

    def test_a_short_prose_reply_is_a_style_miss_not_a_drop(self):
        render = rendered(["It shows no activity against HIV replication."],
                          ["no"])
        assert not dropped_statements(render, "No activity.", "prose")

    def test_a_false_premise_may_be_answered_with_the_true_value(self):
        render = rendered(["The smallest ring containing atom 8 (C) has 4 "
                           "atoms."], ["4"], verdict="no")
        assert not dropped_statements(render, "Four", "one word")
        assert not dropped_statements(render, "No", "one word")
        assert dropped_statements(render, "Five", "one word")

    def test_a_reordered_list_still_answers(self):
        # the writer sometimes reorders the asks; a reply that answers the
        # question it was actually asked has not dropped anything
        render = rendered(["The smallest ring containing atom 12 (C) has 5 "
                           "atoms.",
                           "Atom 12 (C) is not part of an amide.",
                           "Atom 12 (C) is in a ring."], ["5", "no", "yes"])
        assert not dropped_statements(render, "1. Yes\n2. 5\n3. No",
                                      "a numbered list")

    def test_prose_catches_a_genuinely_dropped_statement(self):
        render = rendered(["It contains 4 ether(s).",
                           "It contains 3 hydroxyl group(s)."], ["4", "3"])
        assert dropped_statements(render, "It contains 4 ethers.", "prose")


class TestTheLeakCheckOnlyCatchesLeaks:

    def test_the_ordinary_question_is_not_a_leak(self):
        render = rendered(["Atom 14 (C) is in a ring."], ["yes"])
        assert not _leaked_statements(render, "Is atom 14 (C) in a ring?",
                                      "report")

    def test_a_list_format_question_is_not_a_leak(self):
        # every one of the first build's leak rejections was this shape
        render = rendered(["Atom 5 (C) and atom 9 (C) are in an aromatic ring."],
                          ["yes"])
        assert not _leaked_statements(
            render,
            "Please tell me if the following are in an aromatic ring, provided "
            "as a numbered list:\n1. atom 5 (C)\n2. atom 4 (N)\n3. atom 9 (C)",
            "triage")

    def test_a_question_that_names_the_count_is_a_leak(self):
        render = rendered(["It has 3 ring(s)."], ["3"])
        assert _leaked_statements(render, "Does this molecule have 3 rings?",
                                  "report")

    def test_an_atom_index_matching_the_count_is_not_a_leak(self):
        render = rendered(["It has 3 ring(s)."], ["3"])
        assert not _leaked_statements(
            render, "How many rings does this have? I'm looking at atom 3 (C).",
            "report")

    def test_check_claim_states_its_claim_by_design(self):
        render = rendered(["It has 3 ring(s)."], ["3"])
        assert not _leaked_statements(render, "I have 3 rings down; right?",
                                      "check_claim")


class TestFormatMet:

    def test_one_word_is_one_word(self):
        assert format_met("4", "one word", None)
        assert format_met("Yes.", "one word", None)
        assert not format_met("Yes, it is.", "one word", None)

    def test_json_must_carry_the_declared_keys(self):
        skeleton = {"ring_atom_14": "boolean"}
        assert format_met('{"ring_atom_14": true}',
                          "JSON matching the given schema", skeleton)
        assert not format_met('{"ring_count": 3}',
                              "JSON matching the given schema", skeleton)
        assert not format_met("no json here",
                              "JSON matching the given schema", skeleton)

    def test_a_numbered_list_needs_two_items(self):
        assert format_met("1. first 2. second", "a numbered list", None)
        assert not format_met("1. only one", "a numbered list", None)

    def test_a_bulleted_list_needs_two_bullets(self):
        assert format_met("- first\n- second", "a bulleted list", None)
        assert not format_met("- only one", "a bulleted list", None)

    def test_prose_is_never_refused(self):
        assert format_met("anything at all", "prose", None)
        assert format_met("anything at all",
                          "lead with the answer, then the detail", None)


class TestInventsAStructure:

    def test_a_quoted_smiles_is_caught(self):
        assert invents_a_structure(
            "How many rings are in the molecule represented by the SMILES "
            "string CC1=CC=C(C=C1)C2=CC=NC3=C2C=CC=C3?")

    def test_an_ordinary_question_is_not(self):
        assert not invents_a_structure(
            "Is atom 14 (C) in a ring, and how big is that ring?")

    def test_a_chemical_name_is_not_a_structure(self):
        # a caption-derived question carries IUPAC-ish names, and they are not
        # SMILES however many brackets and digits they have
        assert not invents_a_structure(
            "Can you describe this compound? I have it down as a "
            "3-(4-chlorophenyl)-1-phenylpyrazol-4-yl derivative.")

    def test_a_formula_is_not_a_structure(self):
        assert not invents_a_structure("Is the formula C6H12O6 right?")


class TestAnchorNamed:

    def test_an_atom_is_named_by_its_index(self):
        # a person types "C14" or "position 14" as readily as "atom 14 (C)"
        assert _anchor_named("atom 14 (C)", "is C14 in a ring?")
        assert _anchor_named("atom 14 (C)", "what about position 14?")
        assert not _anchor_named("atom 14 (C)", "is atom 41 in a ring?")

    def test_a_group_is_named_by_its_word(self):
        assert _anchor_named("hydroxyl", "does it have a hydroxyl anywhere?")
        assert not _anchor_named("hydroxyl", "does it have an amide?")


#: Transcribed from the hundred rows read by hand off the third smoke build's
#: calibration sheet. Every one of these was *accepted*: the statement check
#: counts statements, and a `decide` verdict is not one of them.
class TestTheVerdictIsNotAStatement:

    def test_a_list_that_states_both_facts_and_never_decides_is_a_drop(self):
        out = rendered(["Atom 19 (C) is in a ring.", "Atom 25 (C) is in a ring."],
                       ["yes", "yes"], verdict="no")
        reply = "* Atom 19 (C) is in a ring.\n* Atom 25 (C) is in a ring."
        assert dropped_statements(out, reply, "a bulleted list") == []
        assert not states_the_verdict(out, reply, "a bulleted list")

    def test_an_abbreviated_list_that_never_decides_is_a_drop(self):
        out = rendered(["Atom 17 (C) is not part of an ether.",
                        "Atom 17 (C) is in a ring."], ["no", "yes"],
                       verdict="yes")
        assert not states_the_verdict(out, "1. Not part of ether.\n2. In ring.",
                                      "a numbered list")

    def test_a_list_with_a_spare_item_may_decide_in_it(self):
        out = rendered(["Atom 17 (C) is not part of an ether.",
                        "Atom 17 (C) is in a ring."], ["no", "yes"],
                       verdict="yes")
        assert states_the_verdict(
            out, "1. Yes\n2. Not part of ether.\n3. In ring.", "a numbered list")

    def test_a_bare_yes_in_a_list_of_answers_is_not_the_verdict(self):
        # The first item answers the first *statement*. Reading its polarity as
        # the decision would pass a reply that never decided and fail one that
        # decided the other way.
        out = rendered(["Atom 19 (C) is in a ring."], ["yes"], verdict="no")
        assert not states_the_verdict(out, "1. Yes", "a numbered list")

    def test_prose_decides_in_words_that_name_the_constraint(self):
        out = rendered(["Atom 24 (C) is not in a ring."], ["no"], verdict="yes")
        # The negation belongs to the fact, not to the decision, and the clause
        # that names the decision carries none of it.
        assert states_the_verdict(
            out, "Atom 24 (C) is not located within a ring structure. "
                 "Consequently, the molecule satisfies the specified constraint.",
            "prose")

    def test_prose_may_decide_in_a_bare_word(self):
        out = rendered(["It contains 1 amide(s).", "It contains 1 halogen(s)."],
                       ["1", "1"], verdict="yes")
        assert states_the_verdict(out, "1 amide; 1 halogen. Qualifies.", "prose")
        assert states_the_verdict(out, "Yes. 1 amide and 1 halogen.", "prose")

    def test_a_decision_the_other_way_round_does_not_pass(self):
        out = rendered(["Atom 16 (C) is not in a ring."], ["no"], verdict="yes")
        assert not states_the_verdict(
            out, "No, that one doesn't qualify. Atom 16 (C) is not in a ring.",
            "prose")

    def test_a_verdict_on_its_own_line_below_the_list_counts(self):
        # "1. 6\n2. 6\nYes." — the deciding line carries no list marker, so
        # splitting on markers found two items and no room for a verdict.
        out = rendered(["The smallest ring containing atom 4 (C) has 6 atoms.",
                        "The smallest ring containing atom 9 (C) has 6 atoms."],
                       ["6", "6"], verdict="yes")
        assert states_the_verdict(out, "1. 6\n2. 6\nYes.", "a numbered list")

    def test_a_named_decision_is_read_after_the_bare_one_that_precedes_it(self):
        # The clause naming the constraint also carries the fact's negation, so
        # reading the polarity there answers the opposite of what the reply says.
        out = rendered(["Atom 11 (C) is not part of an ether.",
                        "Atom 11 (C) is in a ring."], ["no", "yes"],
                       verdict="yes")
        assert states_the_verdict(
            out, "Yes, it meets the constraint because atom 11 (C) is not part "
                 "of an ether, although it is in a ring.", "prose")

    def test_a_list_item_that_opens_with_no_is_not_the_decision(self):
        # "No sulfonamide." begins with "No" and answers a statement. Read as
        # the decision it inverted a correct reply whose verdict was the last
        # line.
        out = rendered(["It contains no sulfonamide.",
                        "It contains no hydroxyl group."], ["no", "no"],
                       verdict="yes")
        assert states_the_verdict(
            out, "1. No sulfonamide.\n2. No hydroxyl group.\n3. Yes.",
            "a numbered list")

    def test_a_decision_split_from_its_answer_by_a_colon(self):
        out = rendered(["0 of its rings are aromatic.",
                        "Atom 20 (C) is not in an aromatic ring."],
                       ["0", "no"], verdict="no")
        assert states_the_verdict(
            out, "* 0 aromatic rings\n* Atom 20 (C): not aromatic\n"
                 "* Constraint met: No", "a bulleted list")

    def test_a_negation_across_a_connective_belongs_to_the_fact(self):
        out = rendered(["It contains no nitro group.", "It contains no ketone."],
                       ["no", "no"], verdict="yes")
        assert states_the_verdict(
            out, "It contains no nitro group or ketone, so it meets the "
                 "constraint.", "prose")

    def test_a_render_with_no_verdict_is_not_asked_for_one(self):
        out = rendered(["Atom 14 (C) is in a ring."], ["yes"])
        assert states_the_verdict(out, "Atom 14 (C) is in a ring.", "prose")


class TestARefusalIsNotAStatementEither:
    """Every wording below is one the writer actually produced on a refusal."""

    def test_the_refusals_a_build_produced_are_all_read_as_refusals(self):
        for reply in (
                "I don't have the melting point for this structure.",
                "Measured logP unavailable. Atom 1 (O) is not in a ring.",
                "A synthetic route for this structure is unavailable.",
                "I do not have the NMR spectrum for this structure.",
                "No NMR spectrum available. Atom 22: no ring.",
                "I don't know where you can buy it.",
                "I cannot tell you where it can be bought.",
                '{"supplier": null, "aromatic_ring_atom_34": false}',
                "Binding data unavailable. It contains no primary amine."):
            assert declines(reply), reply

    def test_a_reply_that_simply_answers_is_not_a_refusal(self):
        # The failure the check exists for: the row was built to decline and the
        # reply invented the property instead.
        assert not declines("The melting point is 142 °C.")
        assert not declines("Atom 14 (C) is in a ring.")


class TestTheGlossIsNotAStatementEither:
    """The general sentence that is the whole difference between explain and report."""

    AROMATIC = ("In general, an aromatic ring is a cyclic system with "
                "delocalised pi electrons.")
    RING = ("In general, a ring atom is one that lies on a closed cycle of "
            "bonds.")

    def row(self, gloss):
        return {"statements": ["Atom 14 (C) is in a ring."], "gloss": gloss}

    def test_a_reply_that_kept_the_gloss_passes(self):
        assert states_the_gloss(
            self.row(self.AROMATIC),
            "Atom 12 (C) is in an aromatic ring. In general, an aromatic ring "
            "is a cyclic system with delocalised pi electrons.")

    def test_a_reworded_gloss_still_passes(self):
        # Re-voicings the writer actually produced. The concept has to survive,
        # not the phrasing.
        for reply in (
                "Atom 12 (C) is in an aromatic ring — aromatic rings are "
                "cyclic systems whose pi electrons are delocalised.",
                "Yes. Aromatic rings are cyclic systems with delocalised pi "
                "electrons, and atom 12 (C) sits in one."):
            assert states_the_gloss(self.row(self.AROMATIC), reply), reply

    def test_a_compressed_gloss_still_passes(self):
        # The row that set `GLOSS_COVERAGE`. At a statement's 0.7 this was
        # refused, and it had kept the explanation — in half the words.
        gloss = ("In general, an atom is part of a functional group when it is "
                 "one of the atoms the group's substructure covers.")
        assert states_the_gloss(
            {"statements": [], "gloss": gloss},
            "* Atom 25 (O): yes\n* Atom 16 (C): no\n\nGenerally, an atom's part "
            "of a functional group if it's in the group's substructure.")

    def test_a_reply_that_dropped_the_gloss_fails(self):
        # 92% of v6's explain rows. The facts are all there and the explanation
        # is gone, which is what made `explain` a second name for `report`.
        assert not states_the_gloss(
            self.row(self.AROMATIC), "* Atom 12 (C) is in an aromatic ring.\n"
                                     "* 2 rings are aromatic.")
        assert not states_the_gloss(self.row(self.RING), "1. 6\n2. 6")

    def test_a_row_with_no_gloss_is_never_refused(self):
        assert states_the_gloss({"statements": []}, "Three")
        assert states_the_gloss(self.row(None), "Three")


class TestTheTestSplitReadsFirst:
    """Which split pays for a near-duplicate is decided by the reading order."""

    def test_test_batches_come_before_every_train_batch(self):
        paths = ["/b/b-train-batch-0000.json", "/b/test-batch-0044.json",
                 "/b/train-batch-0000.json", "/b/test-batch-0000.json"]
        out = read_order(paths)
        assert out[:2] == ["/b/test-batch-0000.json", "/b/test-batch-0044.json"]
        assert out[2:] == ["/b/b-train-batch-0000.json",
                           "/b/train-batch-0000.json"]

    def test_a_top_up_prefix_does_not_jump_the_queue(self):
        # The whole point. `b-train-*` sorts ahead of `test-*` alphabetically,
        # which is how the test split lost 94 rows to a top-up build.
        paths = sorted(["/b/b-train-batch-0000.json", "/b/test-batch-0000.json"])
        assert paths[0].endswith("b-train-batch-0000.json")
        assert read_order(paths)[0].endswith("test-batch-0000.json")

    def test_the_order_within_a_split_is_still_deterministic(self):
        paths = ["/b/train-batch-0002.json", "/b/train-batch-0000.json",
                 "/b/train-batch-0001.json"]
        assert read_order(paths) == sorted(paths)
        assert read_order(list(reversed(paths))) == sorted(paths)


class TestAPremiseThatIsNotFalse:
    """A `false_premise` row whose premise is true — 39 rows of the final build.

    The reply is computed from the statements, so it corrects a belief that
    needed no correcting: the person says "atom 24 (O) is in no ring", which is
    right, and is told it is not. Every other defect this pipeline has produced
    was an omission; this one asserts.
    """

    def row(self, claim, statements, claim_is_true=False):
        return {"statements": statements,
                "ask": {"claim": claim, "claim_is_true": claim_is_true}}

    def test_a_claim_that_is_one_of_the_statements_is_caught(self):
        assert premise_is_not_false(self.row(
            "Atom 24 (O) is in no ring.",
            ["Atom 24 (O) is in no ring.",
             "Atom 24 (O) is not part of an ether."]))

    def test_capitalisation_and_the_full_stop_do_not_hide_it(self):
        # The ask capitalises and punctuates its claim; the statement list is
        # built from the same `_clause` call. Only those differ when it fires.
        assert premise_is_not_false(self.row(
            "atom 8 (C) is in no ring", ["Atom 8 (C) is in no ring."]))

    def test_a_properly_negated_claim_passes(self):
        assert not premise_is_not_false(self.row(
            "Atom 24 (O) is in a ring.", ["Atom 24 (O) is in no ring."]))
        assert not premise_is_not_false(self.row(
            "The smallest ring containing atom 32 (C) has 16 atoms.",
            ["The smallest ring containing atom 32 (C) has 15 atoms."]))

    def test_a_true_claim_is_not_this(self):
        # `check_claim` with no twist states the truth on purpose, and the reply
        # confirms it. Only a claim the render *declared* false is a defect.
        assert not premise_is_not_false(self.row(
            "Atom 14 (C) is in a ring.", ["Atom 14 (C) is in a ring."],
            claim_is_true=True))

    def test_an_ask_with_no_claim_is_not_this(self):
        assert not premise_is_not_false(
            {"statements": ["Atom 14 (C) is in a ring."], "ask": {}})


class TestAPivotTheAnswerCanBeCopiedFrom:

    def test_a_smiles_claim_is_caught(self):
        for task in ("check_claim", "decide"):
            assert pivot_is_unassertable(
                {"task": task, "facts": [{"family": "smiles"}]})

    def test_a_smiles_stated_by_another_task_is_not(self):
        # `report` may answer "what is its canonical SMILES" — there the
        # structure is the answer, not the question.
        assert not pivot_is_unassertable(
            {"task": "report", "facts": [{"family": "smiles"}]})

    def test_an_ordinary_pivot_passes(self):
        assert not pivot_is_unassertable(
            {"task": "check_claim", "facts": [{"family": "ring_membership"},
                                              {"family": "smiles"}]})

    def test_an_intent_with_no_facts_is_not_this(self):
        assert not pivot_is_unassertable({"task": "check_claim", "facts": []})


#: Two `ring_size` facts, the draw that produces the collapsible ask.
TWO_RINGS = {"facts": [{"family": "ring_size"}, {"family": "ring_size"}]}


class TestAnAskForARingNobodyComputed:

    def test_three_atoms_in_one_clause_are_caught(self):
        # Transcribed from train-04923 of the final build.
        assert asks_for_a_joint_ring(
            "What is the size of the smallest ring containing atom 14 (C), "
            "atom 15 (N), and atom 10 (O)? Please lead with the answer, then "
            "the detail.",
            {"facts": [{"family": "ring_size"}] * 3})

    def test_two_atoms_in_one_clause_are_caught(self):
        assert asks_for_a_joint_ring(
            "What is the size of the smallest ring containing atom 3 (N), "
            "atom 8 (C)?", TWO_RINGS)

    def test_the_repeated_clause_is_the_correct_spelling(self):
        # 821 rows of the final build ask it this way and mean two rings.
        assert not asks_for_a_joint_ring(
            "What is the size of the smallest ring containing atom 23 (O) and "
            "the size of the smallest ring containing atom 15 (C)? Please "
            "provide the answer as a bulleted list.", TWO_RINGS)

    def test_one_atom_is_not_this(self):
        assert not asks_for_a_joint_ring(
            "What is the size of the smallest ring containing atom 7 (C)?",
            {"facts": [{"family": "ring_size"}]})

    def test_one_ring_fact_beside_others_is_not_this(self):
        # The clause names two atoms but only one ring size was computed, so the
        # second atom belongs to a different question and the reply can answer.
        assert not asks_for_a_joint_ring(
            "What is the size of the smallest ring containing atom 3 (N), "
            "atom 8 (C)?",
            {"facts": [{"family": "ring_size"}, {"family": "aromatic_ring"}]})

    def test_a_turn_with_no_ring_clause_passes(self):
        assert not asks_for_a_joint_ring(
            "Is atom 3 (N) in a ring, and is atom 8 (C) in a ring?", TWO_RINGS)


class TestStatementSurvives:

    def test_a_reworded_statement_survives(self):
        assert statement_survives(
            "Atom 14 (C) is in a ring.",
            "Yes — atom 14, the carbon, sits on a ring.", terse=False,
            token="yes")

    def test_a_dropped_statement_does_not(self):
        assert not statement_survives(
            "It contains 3 hydroxyl group(s).", "It contains 4 ether(s).",
            terse=False, token="3")
