"""Tier A's test split is NOT molecule-disjoint. This test documents that (PLAN.md §3.2.10).

**This pins a known defect, not a desired property.** `generate_examples` draws molecules
with replacement from the bace+bbbp pool and `prepare_dataset` slices the result
positionally, so a molecule can appear in train and again in test. For a molecule-level
family the example is a deterministic function of the molecule, so those are exact
duplicates and memorising them answers the test item -- measured accuracy on that subset
is 1.0000 in almost every run.

It is pinned because §3.2.10 quotes these rates and because the defect is invisible in any
single result: every arm is affected equally, so nothing looks anomalous. It only surfaced
when gate A2's control, which PREDICTED chance, came back at 0.936.

**When the generator is fixed** to split by molecule first (the discipline `tier_b.py`
already follows), this test should FAIL. That is the point: the fix must be a deliberate,
visible change, not something that quietly alters what every Tier-A number means. Invert it
then -- assert disjointness -- and re-derive §3.2.4-§3.2.9 rather than editing this file to
match new output.
"""

import pytest

from src.experiments.molecules.duplicate_analysis import TRAIN, VAL, replay

# Measured 2026-08-31 at data_seed 0. Loose bounds: the point is the magnitude, not
# a digit that would make the test brittle against an RDKit version bump.
MOLECULE_LEVEL = ["longest_chain", "stereo_assigned", "stereo_potential", "ring_count"]
ATOM_LEVEL = ["ring_membership", "ring_size"]


@pytest.mark.parametrize("task", MOLECULE_LEVEL)
def test_molecule_level_families_have_a_large_exact_duplicate_rate(task):
    """~72-74%: the test item is identical to one the model trained on."""
    _train_mols, train_exact, test_items = replay(task)
    exact = sum((s, q) in train_exact for s, q, _a in test_items) / len(test_items)
    assert 0.65 < exact < 0.80, (
        f"{task}: exact train/test duplicate rate is {exact:.1%}. If this dropped, the "
        "generator may have been fixed -- see this module's docstring before editing.")


@pytest.mark.parametrize("task", ATOM_LEVEL)
def test_atom_level_families_leak_the_molecule_but_rarely_the_example(task):
    """The named atom varies, so exact duplicates are ~6% while the MOLECULE is still
    seen ~70% of the time -- structural familiarity, not a memorised answer."""
    train_mols, train_exact, test_items = replay(task)
    exact = sum((s, q) in train_exact for s, q, _a in test_items) / len(test_items)
    seen = sum(s in train_mols for s, _q, _a in test_items) / len(test_items)
    assert exact < 0.15, f"{task}: exact duplicates {exact:.1%}"
    assert seen > 0.60, f"{task}: molecules seen in train {seen:.1%}"


def test_the_replay_reproduces_the_configured_split_sizes():
    """Guards the index arithmetic `duplicate_analysis` maps per-example rows through."""
    _train_mols, _train_exact, test_items = replay("longest_chain")
    assert len(test_items) == 1000
    assert (TRAIN, VAL) == (4000, 500)


def test_replayed_answers_match_the_cached_dataset_distribution():
    """The replay must describe the data the runs actually USED, or §3.2.10 re-scores a
    dataset nobody trained on. Compared against the `.gtds` meta sidecar.

    `pool` is passed explicitly: `RunConfig.pool` defaults to five corpora while every
    §3.2 sweep sets `bace,bbbp`, and replaying against the wrong pool draws different
    molecules and fails silently.
    """
    import json
    import os
    from collections import Counter

    from src.experiments.molecules.config import RunConfig
    from src.experiments.molecules.dataset import dataset_path
    from src.experiments.molecules.duplicate_analysis import SWEEP_POOL

    cfg = RunConfig(task="longest_chain", arm="graph", pool=SWEEP_POOL)
    meta_path = dataset_path(cfg) + ".meta.json"
    if not os.path.exists(meta_path):
        pytest.skip("cached longest_chain dataset not present")

    from src.experiments.molecules.duplicate_analysis import replay_stream

    recorded = json.load(open(meta_path))["answers"]
    replayed = dict(Counter(a for _s, _q, a in replay_stream("longest_chain")))

    assert replayed == recorded, (
        "the replayed generation does not reproduce the cached dataset's answer "
        "distribution, so §3.2.10's re-scoring describes a different dataset")
