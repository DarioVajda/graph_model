"""§9.4 step 12: the hand-written case set and the tool that puts it to a model.

Nothing here is scored, so what is testable is the part that would be silently
wrong rather than visibly wrong: the atom indices (1-based in the file, 0-based
in the graph — an off-by-one points the prompt node at the neighbouring atom and
nothing complains), the leading space on the stored answer, and the file itself
staying parseable and in range as rows are added to it.
"""

import json

import pytest

from src.generalist.tools import assistant_case_study as CS

rdkit = pytest.importorskip("rdkit")


@pytest.fixture(scope="module")
def cases():
    return CS.load_cases(CS.CASES)


def test_the_comment_row_is_not_a_case(cases):
    assert all("_comment" not in row for row in cases)
    assert len(cases) == 30


def test_every_case_has_the_fields_the_report_prints(cases):
    for row in cases:
        for key in ("id", "smiles", "question", "probe", "expect"):
            assert row.get(key), f"{row.get('id')}: missing {key}"
    assert len({row["id"] for row in cases}) == len(cases)


def test_every_smiles_parses_and_every_atom_is_in_range(cases):
    """Also the only check that `atoms` is 1-based: index 0 would be atom -1."""
    from rdkit import Chem

    for row in cases:
        mol = Chem.MolFromSmiles(row["smiles"])
        assert mol is not None, row["id"]
        for atom in row.get("atoms") or []:
            assert 1 <= atom <= mol.GetNumAtoms(), f"{row['id']}: atom {atom}"


def test_draws_convert_the_atom_indices_to_zero_based():
    draws = CS.draws_for([{"id": "x", "smiles": "CCO", "atoms": [1, 3],
                           "question": "q", "probe": "p", "expect": "e"}])
    (_mol, question, answer, named, key, meta) = draws[0]
    assert named == [0, 2]
    assert question == "q"
    assert key == "x"
    assert meta["atoms"] == [1, 3]
    assert meta["shots"] == []


def test_the_stored_answer_keeps_its_leading_space():
    """`_materialise` stores ``answer[1:]`` for a text answer — same trap as the draw."""
    draws = CS.draws_for([{"id": "x", "smiles": "CCO", "atoms": [],
                           "question": "q", "expect": "3"}])
    assert draws[0][2] == " 3"
    assert draws[0][2][1:] == "3"


def test_a_molecule_level_row_wires_no_named_atom():
    draws = CS.draws_for([{"id": "x", "smiles": "CCO", "question": "q",
                           "expect": "e"}])
    assert draws[0][3] == []


def test_an_atom_past_the_end_is_refused():
    with pytest.raises(ValueError, match="outside a molecule"):
        CS.draws_for([{"id": "x", "smiles": "CCO", "atoms": [9],
                       "question": "q", "expect": "e"}])


def test_an_unparseable_smiles_is_refused():
    with pytest.raises(ValueError, match="unparseable"):
        CS.draws_for([{"id": "x", "smiles": "not-a-molecule", "question": "q",
                       "expect": "e"}])


def test_the_report_carries_every_checkpoint_for_every_case():
    rows = [{"id": "cs-01", "smiles": "CCO", "atoms": [], "question": "q?",
             "probe": "a probe", "expect": "an expectation",
             "replies": {"control": "no", "assistant": ""}}]
    text = CS._markdown(rows, ["control", "assistant"],
                        [("control", "/a"), ("assistant", "/b")])
    assert "## cs-01 — a probe" in text
    assert "**control:** no" in text
    # An empty reply is a result, and a report that dropped it would read as a
    # missing row rather than as the refusal to answer that it is.
    assert "**assistant:** (empty)" in text
