"""Tier-A splits must be molecule-disjoint.

This is the test that did not exist. Tier-A generation had no coverage at all, which
is how the split defect survived a whole campaign: `generate_examples` drew molecules
with replacement from one pool and `prepare_dataset` sliced the result positionally,
so ~70% of a test split was also in train and, for a family whose answer is a function
of the molecule alone, the test item was an exact duplicate of a training item. Nothing
looked anomalous because every arm was affected equally.

The property is now the one `tier_b.py` has always had: split the MOLECULES first,
generate examples inside each split.
"""

import random

import pytest
from rdkit import Chem

from src.experiments.molecules import dataset as ds_mod
from src.experiments.molecules.config import RunConfig
from src.experiments.molecules.data import murcko_scaffold
from src.experiments.molecules.dataset import (
    SINGLE_EXAMPLE_TASKS,
    generate_examples,
    split_molecule_pool,
)

POOL = ("bace",)


def _cfg(task, **kw):
    kw.setdefault("train_size", 40)
    kw.setdefault("val_size", 10)
    kw.setdefault("test_size", 20)
    return RunConfig(task=task, arm="flat", pool=POOL, **kw)


def _smiles(mols):
    return {Chem.MolToSmiles(m, canonical=True) for m in mols}


# -- the pool partition --------------------------------------------------------

def test_the_three_pools_share_no_molecule():
    pools = split_molecule_pool(_cfg("longest_chain"))
    train, val, test = (_smiles(pools[k]) for k in ("train", "val", "test"))
    assert not train & test
    assert not train & val
    assert not val & test


def test_no_scaffold_spans_two_pools():
    """Scaffold, not merely molecule, disjointness -- the test set is structurally
    novel, which is the property a structural-reasoning claim needs."""
    pools = split_molecule_pool(_cfg("longest_chain"))
    seen = {}
    for name, mols in pools.items():
        for mol in mols:
            seen.setdefault(murcko_scaffold(Chem.MolToSmiles(mol, canonical=True)),
                            set()).add(name)
    straddling = {s: n for s, n in seen.items() if len(n) > 1}
    assert not straddling, f"{len(straddling)} scaffolds span pools"


def test_every_pool_molecule_comes_from_the_source_pool():
    cfg = _cfg("longest_chain")
    pools = split_molecule_pool(cfg)
    union = _smiles(pools["train"]) | _smiles(pools["val"]) | _smiles(pools["test"])
    assert union <= _smiles(ds_mod._molecule_pool(cfg))


def test_pool_fractions_follow_the_requested_example_counts():
    """So "a single-example family needs a pool as large as the examples requested"
    is the entire sizing rule."""
    cfg = _cfg("longest_chain", train_size=800, val_size=100, test_size=100)
    pools = split_molecule_pool(cfg)
    total = sum(len(v) for v in pools.values())
    assert 0.75 < len(pools["train"]) / total < 0.85
    assert 0.05 < len(pools["test"]) / total < 0.15


# -- generation stays inside its split ----------------------------------------

def _molecules_used(cfg, molecules, monkeypatch, n):
    """Run the real generator, recording which molecules it consumed."""
    used = []

    def _record(mol, question, answer, cfg_):
        used.append(Chem.MolToSmiles(mol, canonical=True))
        return {"text": question}

    monkeypatch.setattr(ds_mod, "build_flat_example", _record)
    generate_examples(cfg, n, random.Random(0), molecules, split="t")
    return used


@pytest.mark.parametrize("task", ["longest_chain", "ring_membership"])
def test_a_test_example_never_uses_a_training_molecule(task, monkeypatch):
    """THE regression test. Before the fix this failed for every family."""
    cfg = _cfg(task)
    pools = split_molecule_pool(cfg)
    train_used = set(_molecules_used(cfg, pools["train"], monkeypatch, cfg.train_size))
    test_used = set(_molecules_used(cfg, pools["test"], monkeypatch, cfg.test_size))
    assert train_used and test_used
    assert not (train_used & test_used)


def test_the_disjointness_check_can_actually_fail(monkeypatch):
    """A guard that cannot fail is decoration.

    Reproduces the OLD behaviour -- both splits drawn from one undivided pool -- and
    requires the overlap to appear. If this ever stops overlapping, the test above is
    passing for some reason other than the partition and proves nothing.

    Sized like the real sweeps (800 + 400 draws from bace's 1513 molecules, expected
    overlap ~210) rather than like the other tests here: at 40 + 20 draws the expected
    overlap is 0.5 and the demonstration would be a coin flip.
    """
    cfg = _cfg("longest_chain", train_size=800, val_size=100, test_size=400)
    whole = ds_mod._molecule_pool(cfg)
    train_used = set(_molecules_used(cfg, whole, monkeypatch, cfg.train_size))
    # A fresh rng, as the old code effectively had when it kept drawing from one pool.
    used = []

    def _record(mol, question, answer, cfg_):
        used.append(Chem.MolToSmiles(mol, canonical=True))
        return {"text": question}

    monkeypatch.setattr(ds_mod, "build_flat_example", _record)
    generate_examples(cfg, cfg.test_size, random.Random(1), whole, split="test")
    assert train_used & set(used), (
        "drawing both splits from one pool produced no overlap, so the disjointness "
        "assertions above are not testing what they claim")


def test_a_single_example_family_never_repeats_a_molecule(monkeypatch):
    """`longest_chain` asks one fixed question, so a repeated molecule is a
    byte-identical duplicate example."""
    cfg = _cfg("longest_chain")
    pools = split_molecule_pool(cfg)
    used = _molecules_used(cfg, pools["test"], monkeypatch, cfg.test_size)
    assert len(used) == len(set(used)) == cfg.test_size


def test_asking_a_single_example_family_for_more_than_the_pool_raises():
    """Silently duplicating would recreate the defect inside one split."""
    cfg = _cfg("longest_chain")
    pools = split_molecule_pool(cfg)
    with pytest.raises(ValueError, match="one example per molecule"):
        generate_examples(cfg, len(pools["test"]) + 50, random.Random(0),
                          pools["test"], split="test")


def test_a_multi_example_family_may_reuse_a_molecule(monkeypatch):
    """`ring_membership` names a different atom each time, so a second pass over the
    same molecules yields genuinely different examples -- allowed, and needed."""
    cfg = _cfg("ring_membership")
    pools = split_molecule_pool(cfg)
    n = len(pools["test"]) + 25
    used = _molecules_used(cfg, pools["test"], monkeypatch, n)
    assert len(used) == n
    assert len(set(used)) < n, "expected a second pass to reuse molecules"


def test_a_repeated_molecule_question_pair_is_counted_not_hidden(monkeypatch):
    """A multi-example family CAN re-emit a (molecule, question) pair on a later
    pass, because the named atom is drawn at random. That costs effective sample
    size, so it has to be visible in the record rather than inferred later.

    Measured on the real sweeps: at 4000/500/1000 over bace+bbbp+tox21+lipo,
    `fg_atom_membership` -- the lowest-yield family, 1648 usable test molecules --
    repeats 8 of 1000 test examples. Every other family repeats none.
    """
    cfg = _cfg("ring_membership")
    pools = split_molecule_pool(cfg)
    _, stats = generate_examples(
        cfg, len(pools["test"]) * 3, random.Random(0), pools["test"], split="test")
    assert "repeats" in stats
    assert stats["repeats"] > 0, (
        "three passes over one pool re-drew no (molecule, question) pair, so the "
        "counter is not wired to anything")


def test_both_arms_see_the_same_molecules_questions_and_answers(monkeypatch):
    """The graph/flat contrast is only a control if the two arms differ in
    REPRESENTATION and nothing else. `generate_examples` consumes the rng
    identically in both arms -- the arm only selects which builder runs -- so this
    pins that property rather than trusting it.
    """
    seen = {}

    def _recorder(bucket):
        def _record(mol, question, answer, *rest):
            seen.setdefault(bucket, []).append(
                (Chem.MolToSmiles(mol, canonical=True), question, answer))
            return {"text": question}
        return _record

    monkeypatch.setattr(ds_mod, "build_flat_example", _recorder("flat"))
    monkeypatch.setattr(ds_mod, "build_graph_example", _recorder("graph"))

    for arm in ("flat", "graph"):
        cfg = RunConfig(task="ring_membership", arm=arm, pool=POOL,
                        train_size=40, val_size=10, test_size=20)
        pools = split_molecule_pool(cfg)
        generate_examples(cfg, 40, random.Random(0), pools["train"], split="train")

    assert seen["flat"] and seen["flat"] == seen["graph"]


def test_single_example_task_list_matches_the_generators():
    """A family wrongly listed here would be capped for no reason; one wrongly
    omitted would be allowed to emit duplicates."""
    assert SINGLE_EXAMPLE_TASKS == {"longest_chain", "ring_count",
                                    "stereo_potential", "stereo_assigned"}


# -- determinism ---------------------------------------------------------------

def test_the_partition_is_deterministic():
    a = _smiles(split_molecule_pool(_cfg("longest_chain"))["test"])
    b = _smiles(split_molecule_pool(_cfg("longest_chain"))["test"])
    assert a == b


def test_generation_is_deterministic_under_data_seed(monkeypatch):
    cfg = _cfg("ring_membership")
    pools = split_molecule_pool(cfg)
    first = _molecules_used(cfg, pools["train"], monkeypatch, 30)
    second = _molecules_used(cfg, pools["train"], monkeypatch, 30)
    assert first == second


# -- the cache key -------------------------------------------------------------

def test_the_artifact_path_marks_the_molecule_split():
    """Artifacts built before the fix must never be loaded by the fixed code: their
    paths lack the tag, so they cannot match."""
    assert "molsplit" in ds_mod.dataset_path(_cfg("longest_chain"))
