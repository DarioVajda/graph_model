"""
Coverage accounting for Tier C (`molecules/chebi.py`).

`excluded_cids` decides the denominator of every published caption number: the
molecules it names are charged as misses by `chebi_lit_metrics.py`, and one it
fails to name is a molecule that silently leaves the benchmark. `PLAN.md` §9's
rule applies directly — the quantity is *read* by the rescorer and must therefore
be asserted on somewhere.

What is pinned is that the exclusion list is the exact complement of what the
build keeps, because that is the property the accounting depends on: `load_chebi`
keeps a molecule and `excluded_cids` names it, never both and never neither. The
cases are a six-line fake benchmark file, so this is not a `slow` test.
"""

import pytest

from src.experiments.molecules import chebi


#: (cid, smiles, description). Chosen so each filter fires exactly once.
ROWS = [
    ("1", "CCO", "The molecule is ethanol."),                      # kept
    ("2", "Cc1ccccc1", "The molecule is toluene."),                # kept
    ("3", "CCO.CCO", "The molecule is a two-fragment thing."),     # disconnected
    ("4", "C" * 40, "The molecule is a long alkane."),             # over a cap of 20
    ("5", "not-a-smiles", "The molecule is unparseable."),         # parse failure
    ("6", "CCC", ""),                                              # empty description
]


@pytest.fixture
def chebi_dir(tmp_path):
    path = tmp_path / "chebi20"
    path.mkdir()
    with open(path / "test.txt", "w", encoding="utf-8") as fh:
        fh.write("CID\tSMILES\tdescription\n")
        for cid, smiles, text in ROWS:
            fh.write(f"{cid}\t{smiles}\t{text}\n")
    # `load_chebi` reads whichever splits it is asked for; the other two exist so
    # a default call does not trip over a missing file.
    for name in ("train.txt", "validation.txt"):
        with open(path / name, "w", encoding="utf-8") as fh:
            fh.write("CID\tSMILES\tdescription\n")
            fh.write("900\tCCO\tThe molecule is ethanol.\n")
    return str(path)


def test_each_filter_names_its_molecule(chebi_dir):
    excluded = chebi.excluded_cids("test", heavy_atom_cap=20,
                                   allow_disconnected=False, chebi_dir=chebi_dir)

    assert {e["cid"]: e["reason"] for e in excluded} == {
        "3": "disconnected", "4": "heavy_atom_cap",
        "5": "parse", "6": "empty_description"}


def test_admitting_disconnected_molecules_removes_only_that_reason(chebi_dir):
    excluded = chebi.excluded_cids("test", heavy_atom_cap=20,
                                   allow_disconnected=True, chebi_dir=chebi_dir)
    reasons = {e["cid"]: e["reason"] for e in excluded}

    assert "3" not in reasons, "a disconnected molecule is admitted when allowed"
    assert reasons == {"4": "heavy_atom_cap", "5": "parse",
                       "6": "empty_description"}


def test_raising_the_cap_admits_the_large_molecule(chebi_dir):
    excluded = chebi.excluded_cids("test", heavy_atom_cap=128,
                                   allow_disconnected=True, chebi_dir=chebi_dir)

    assert {e["cid"] for e in excluded} == {"5", "6"}


def test_the_exclusion_list_is_the_complement_of_what_is_kept(chebi_dir):
    """The load-time filters and the report's filters must be one function.

    This is the assertion the accounting actually rests on. If `load_chebi` grew a
    filter `excluded_cids` did not, the difference would be molecules absent from
    both the scored rows and the excluded list — silently shrinking the
    denominator, which is the defect the whole coverage pass exists to prevent.
    """
    kept, _stats = chebi.load_chebi(heavy_atom_cap=20, allow_disconnected=False,
                                    chebi_dir=chebi_dir, splits=("test",))
    kept_cids = {r["cid"] for r in kept["test"]}
    excluded = {e["cid"] for e in chebi.excluded_cids(
        "test", heavy_atom_cap=20, allow_disconnected=False, chebi_dir=chebi_dir)}

    assert kept_cids & excluded == set(), "a molecule cannot be both"
    assert kept_cids | excluded == {cid for cid, _, _ in ROWS}, \
        "every benchmark row is either scored or charged as a miss"


def test_examples_carry_the_question_and_a_leading_space(chebi_dir):
    """The answer boundary is a space, and `dataset.py` supervises from it."""
    splits, stats = chebi.build_chebi_examples(
        heavy_atom_cap=128, allow_disconnected=True, chebi_dir=chebi_dir)

    mol, question, answer = splits["test"][0]
    assert question == chebi.CHEBI_QUESTION
    assert answer.startswith(" "), "the supervised span begins after the space"
    assert answer.strip() == "The molecule is ethanol."
    assert stats["split_sizes"]["test"] == 4


def test_reference_captions_cover_every_described_row(chebi_dir):
    """The rescorer charges an excluded molecule against its real caption."""
    captions = chebi.reference_captions("test", chebi_dir=chebi_dir)

    assert captions["4"] == "The molecule is a long alkane."
    assert "6" not in captions, "a row with no description has no reference"
