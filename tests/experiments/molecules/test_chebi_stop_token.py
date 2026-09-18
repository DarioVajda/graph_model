"""
Tier C's stop token, and the assert that reproduces the defect it prevents.

`PLAN.md` §9's standing rule is that every instrument gets a test which fails
when the instrument is wrong, **including one that reproduces the defect**. The
defect here is real and was reintroduced on 2026-09-18: Tier C was ported into
this package without carrying `add_eos` across, because Tiers A and B answer in
one teacher-forced token and correctly leave it off. Six cells trained for two
and a half hours before a selection curve that read 0.176 where 0.44 was expected
prompted someone to look at the tokens rather than believe the story.

`generalist/MOLECULE_GENERALIST.md` §8.5 is the same defect's first outing: a
graph arm writing the exactly-correct canonical SMILES 46.5 % of the time, as a
*prefix* of a runaway generation, reported as `exact_match` 0.0000 for two months.

So this pins two things: that the checker rejects a build whose generative answers
do not end in the stop token, and that it accepts one that does.
"""

import pytest

from src.experiments.molecules import dataset as ds_mod


class _Tok:
    """Minimal stand-in: the checker only reads `eos_token_id`."""

    def __init__(self, eos=128001):
        self.eos_token_id = eos


class _Rows(list):
    """A list of rows shaped like `TextGraphDataset.__getitem__` returns."""


def _rows(tails, prompt_node=1):
    return _Rows({"prompt_node": prompt_node,
                  "input_ids": [[9, 9, 9], [5, 6, tail]]}
                 for tail in tails)


def test_a_build_whose_answers_end_in_eos_is_accepted():
    rows = _rows([128001] * 8)

    assert ds_mod.verify_generative_stop_token(rows, _Tok(), sample=8) is True


def test_a_build_without_a_stop_token_is_REFUSED():
    """The defect, reproduced: the caption's last token is the full stop, not EOS.

    `13` is the id of ``.`` — the real tail of a ChEBI caption on the build that
    was thrown away, which ended ``'…icosatrienoic acid.'`` and nothing further.
    """
    rows = _rows([13] * 8)

    with pytest.raises(AssertionError, match="does not end with the stop token"):
        ds_mod.verify_generative_stop_token(rows, _Tok(), sample=8)


def test_one_bad_row_among_good_ones_is_enough_to_refuse():
    """Sampling must not let a partial failure through as a pass."""
    rows = _rows([128001] * 40 + [13] + [128001] * 40)

    with pytest.raises(AssertionError, match="does not end with the stop token"):
        ds_mod.verify_generative_stop_token(rows, _Tok(), sample=81)


def test_a_tokenizer_with_no_eos_is_refused():
    rows = _rows([128001] * 4)

    with pytest.raises(AssertionError, match="eos_token_id"):
        ds_mod.verify_generative_stop_token(rows, _Tok(eos=None), sample=4)


def test_the_dataset_path_distinguishes_a_build_with_a_stop_token():
    """A defective artifact must not be silently reusable.

    The fix changes what is supervised, so it has to change the path — otherwise
    a rebuild resolves to the cached file and the run trains on the old data
    while the config claims otherwise. This is the same discipline `molsplit`
    carries for the Tier-A split defect.
    """
    from src.experiments.molecules.config import RunConfig

    cfg = RunConfig(task="chebi20", arm="graph", encoding="rich_levi",
                    model_name="meta-llama/Llama-3.2-1B")

    assert "_eos_" in ds_mod.dataset_path(cfg), \
        "a Tier-C artifact must be distinguishable from one built without a stop token"
