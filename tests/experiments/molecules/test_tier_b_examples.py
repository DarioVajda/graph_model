"""Tier-B example construction: label alignment, split integrity, ordering.

The molecules analogue of `relbench/test_evaluate_crosscheck.py`. Relbench can cross-check
its metric against `task.evaluate` over the task table; molecules has no external oracle --
the labels come from our own CSVs -- so the guard has to be on the *construction* instead.

The failure this file exists to catch: `load_data` slices the cached dataset positionally,
`ds[:train_end], ds[train_end:val_end], ds[val_end:total]`, using sizes read back from the
meta sidecar (`dataset.py:280`). If the ordering in `prepare_tier_b_graphs` and the recorded
sizes ever disagree, the test metric is computed over training molecules and **nothing
raises** -- it just reports a suspiciously good scaffold-split AUROC. That is the same class
of silent error PLAN.md §8 built the round-trip test for.
"""

import math
from types import SimpleNamespace

import pytest

from src.experiments.molecules import dataset as ds_mod
from src.experiments.molecules import tier_b as tb
from src.experiments.molecules.config import RunConfig
from src.experiments.molecules.data import murcko_scaffold

# Declared in PLAN.md §1 Tier B, measured at M0. Pinned so a change to
# `scaffold_split` cannot silently move the boundaries every published number sits on.
DECLARED = {
    "bace": {"sizes": {"train": 1210, "val": 151, "test": 152}, "pos_rate": 0.4567},
    "bbbp": {"sizes": {"train": 1631, "val": 204, "test": 204}, "pos_rate": 0.7651},
}


# -- synthetic corpora: the construction logic, without the CSVs ---------------

class _Mol:
    """Stands in for an RDKit Mol; `build_tier_b_examples` only carries it through."""

    def __init__(self, tag):
        self.tag = tag


def _fake_corpus(monkeypatch, records, task_cols, split_indices):
    """Point `build_tier_b_examples` at a corpus we control, keeping its real logic."""
    spec = SimpleNamespace(smiles_col="smiles", task_cols=tuple(task_cols))
    monkeypatch.setattr(tb, "load_tier_b",
                        lambda task: (records, spec, {"parse": 0, "unsupported_bond": 0}))
    monkeypatch.setattr(tb, "scaffold_split", lambda smiles_list: split_indices)


def test_the_answer_matches_the_label_for_every_example(monkeypatch):
    """A misaligned zip here would be invisible: the AUROC would just be ~0.5."""
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"y": i % 2}}
        for i in range(6)
    ]
    _fake_corpus(monkeypatch, records, ["y"], ([0, 1, 2, 3], [4], [5]))
    splits, _ = tb.build_tier_b_examples("bace")

    seen = {}
    for items in splits.values():
        for mol, _question, answer in items:
            seen[mol.tag] = answer
    assert seen == {0: " No", 1: " Yes", 2: " No", 3: " Yes", 4: " No", 5: " Yes"}


def test_missing_labels_are_skipped_not_imputed(monkeypatch):
    """Imputing NaN as 0 would invent negatives; Tox21 alone has 16012 absent labels."""
    records = [
        {"smiles": "C0", "mol": _Mol(0), "targets": {"a": 1, "b": float("nan")}},
        {"smiles": "C1", "mol": _Mol(1), "targets": {"a": None, "b": 0}},
    ]
    _fake_corpus(monkeypatch, records, ["a", "b"], ([0, 1], [], []))
    splits, stats = tb.build_tier_b_examples("tox21")

    assert stats["unlabelled"] == 2
    assert len(splits["train"]) == 2
    assert stats["positives"] == 1 and stats["negatives"] == 1


def test_every_endpoint_of_a_molecule_lands_in_one_split(monkeypatch):
    """Otherwise the same structure appears in train and test under a different
    question, and the scaffold guarantee is void (tier_b.py docstring, point 3)."""
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"a": 1, "b": 0, "c": 1}}
        for i in range(4)
    ]
    _fake_corpus(monkeypatch, records, ["a", "b", "c"], ([0, 1], [2], [3]))
    splits, _ = tb.build_tier_b_examples("tox21")

    where = {}
    for name, items in splits.items():
        for mol, _q, _a in items:
            where.setdefault(mol.tag, set()).add(name)
    assert all(len(names) == 1 for names in where.values()), where
    assert {k: len(v) for k, v in splits.items()} == {"train": 6, "val": 3, "test": 3}


def test_the_endpoint_is_named_in_the_question(monkeypatch):
    """12 Tox21 endpoints are one model only because the question distinguishes them.
    Identical questions would make the task unanswerable and the labels contradictory."""
    records = [{"smiles": "C0", "mol": _Mol(0), "targets": {"NR-AR": 1, "SR-MMP": 0}}]
    _fake_corpus(monkeypatch, records, ["NR-AR", "SR-MMP"], ([0], [], []))
    splits, _ = tb.build_tier_b_examples("tox21")

    questions = [q for _m, q, _a in splits["train"]]
    assert len(set(questions)) == 2
    assert any("NR-AR" in q for q in questions), questions


def test_assay_codes_are_not_mangled_into_prose():
    """`_readable` deliberately keeps 'NR-AR' rather than 'nr ar' -- the code is what a
    chemistry-pretrained model has actually seen."""
    assert tb._readable("NR-AR") == "NR-AR"
    assert tb._readable("FDA_APPROVED") == "FDA APPROVED"


def test_regression_sets_are_refused_rather_than_binarised():
    for task in tb.REGRESSION_TASKS:
        with pytest.raises(NotImplementedError, match="regression"):
            tb.build_tier_b_examples(task)


def test_label_parsing_handles_the_csv_spellings():
    assert tb._label("1") is True and tb._label(1.0) is True
    assert tb._label("0") is False and tb._label(0) is False
    assert tb._label(float("nan")) is None
    assert tb._label(None) is None
    assert tb._label("") is None


# -- the ordering contract: the silent-failure guard --------------------------

def test_prepare_orders_train_then_val_then_test(monkeypatch):
    """`load_data` slices positionally by the recorded sizes. If the emitted order and
    the recorded sizes ever disagree, the test metric is computed over training
    molecules and nothing raises."""
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"y": i % 2}} for i in range(10)
    ]
    _fake_corpus(monkeypatch, records, ["y"], (list(range(6)), [6, 7], [8, 9]))
    # Keep the items themselves rather than featurising them: the invariant is order.
    monkeypatch.setattr(ds_mod, "_build_split_graphs", lambda items, cfg: list(items))

    cfg = RunConfig(task="bace", arm="graph")
    ordered, _stats, sizes = ds_mod.prepare_tier_b_graphs(cfg)

    assert sizes == {"train": 6, "val": 2, "test": 2}
    train_end, val_end = sizes["train"], sizes["train"] + sizes["val"]
    assert [m.tag for m, _q, _a in ordered[:train_end]] == list(range(6))
    assert [m.tag for m, _q, _a in ordered[train_end:val_end]] == [6, 7]
    assert [m.tag for m, _q, _a in ordered[val_end:]] == [8, 9]


def test_caps_subsample_randomly_rather_than_slicing(monkeypatch):
    """`scaffold_split` emits groups largest-first, so a slice would keep the most
    common scaffolds and quietly change the task (dataset.py:154 docstring)."""
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"y": i % 2}} for i in range(40)
    ]
    _fake_corpus(monkeypatch, records, ["y"],
                 (list(range(30)), list(range(30, 35)), list(range(35, 40))))
    monkeypatch.setattr(ds_mod, "_build_split_graphs", lambda items, cfg: list(items))

    cfg = RunConfig(task="bace", arm="graph", max_train_examples=10, max_eval_examples=3)
    ordered, _stats, sizes = ds_mod.prepare_tier_b_graphs(cfg)

    assert sizes == {"train": 10, "val": 3, "test": 3}
    kept = [m.tag for m, _q, _a in ordered[:10]]
    assert kept != list(range(10)), "a cap that returns the first N rows is a slice"
    assert len(set(kept)) == 10


def test_stats_carry_the_answer_distribution(monkeypatch):
    """REGRESSION (010, 2026-08-31). `build_tier_b_examples` produced no `answers`
    key, so `load_or_create_dataset` raised `KeyError: 'answers'` on a *print* and
    failed every Tier-B run -- after the artifact had already been written.

    The quieter half mattered more: `_answer_stats` reads the same key with `.get`,
    so had the print not crashed, every Tier-B run would have recorded
    `base_rate: null` -- the field PLAN.md §3.2.4.1 made mandatory precisely because
    a score without its floor is uninterpretable.
    """
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"y": 1 if i < 7 else 0}}
        for i in range(10)
    ]
    _fake_corpus(monkeypatch, records, ["y"], (list(range(6)), [6, 7], [8, 9]))
    monkeypatch.setattr(ds_mod, "_build_split_graphs", lambda items, cfg: list(items))

    _graphs, stats, _sizes = ds_mod.prepare_tier_b_graphs(RunConfig(task="bace"))

    assert stats["answers"] == {" Yes": 7, " No": 3}
    assert stats["answers_by_split"]["test"] == {" No": 2}
    assert sum(sum(v.values()) for v in stats["answers_by_split"].values()) == 10


def test_the_answer_distribution_reflects_a_cap_rather_than_the_full_corpus(monkeypatch):
    """A cap changes what is in the artifact, so it must change the recorded floor."""
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"y": i % 2}} for i in range(40)
    ]
    _fake_corpus(monkeypatch, records, ["y"],
                 (list(range(30)), list(range(30, 35)), list(range(35, 40))))
    monkeypatch.setattr(ds_mod, "_build_split_graphs", lambda items, cfg: list(items))

    cfg = RunConfig(task="bace", max_train_examples=10, max_eval_examples=3)
    _graphs, stats, sizes = ds_mod.prepare_tier_b_graphs(cfg)

    for name in ("train", "val", "test"):
        assert sum(stats["answers_by_split"][name].values()) == sizes[name]
    assert sum(stats["answers"].values()) == sum(sizes.values()) == 16


def test_base_rate_is_recorded_for_tier_b_and_taken_from_the_test_split():
    """The headline is a test number, so its floor is the test split's majority rate.
    On BBBP the corpus-wide rate is 0.765 and the test rate is 0.524 -- quoting the
    former against a test accuracy compares a score to a different distribution."""
    from src.experiments.molecules.train import _answer_stats

    stats = {"answers": {" Yes": 80, " No": 20},
             "answers_by_split": {"train": {" Yes": 78, " No": 2},
                                  "val": {" Yes": 1, " No": 9},
                                  "test": {" Yes": 1, " No": 9}}}
    out = _answer_stats(stats)
    assert out["base_rate"] == pytest.approx(0.9)          # the TEST floor, not 0.8
    assert out["base_rate_source"] == "test_split"
    assert out["n_classes"] == 2


def test_records_without_a_per_split_breakdown_fall_back_to_the_corpus_rate():
    """Covers artifacts built before either tier recorded `answers_by_split` -- they
    must still yield a base_rate rather than a null, and must say which one it is."""
    from src.experiments.molecules.train import _answer_stats

    out = _answer_stats({"answers": {" 0": 760, " 1": 240}})
    assert out["base_rate"] == pytest.approx(0.76)
    assert out["base_rate_source"] == "all_examples"


def test_missing_stats_still_yield_a_record_rather_than_raising():
    from src.experiments.molecules.train import _answer_stats

    for stats in ({}, None, {"answers": {}}):
        assert _answer_stats(stats)["base_rate"] is None


def test_the_cap_is_deterministic_under_data_seed(monkeypatch):
    records = [
        {"smiles": f"C{i}", "mol": _Mol(i), "targets": {"y": i % 2}} for i in range(40)
    ]
    _fake_corpus(monkeypatch, records, ["y"],
                 (list(range(30)), list(range(30, 35)), list(range(35, 40))))
    monkeypatch.setattr(ds_mod, "_build_split_graphs", lambda items, cfg: list(items))

    def kept(seed):
        cfg = RunConfig(task="bace", arm="graph", max_train_examples=10,
                        max_eval_examples=3, data_seed=seed)
        return [m.tag for m, _q, _a in ds_mod.prepare_tier_b_graphs(cfg)[0][:10]]

    assert kept(0) == kept(0)
    assert kept(0) != kept(1)


# -- the real corpora ---------------------------------------------------------

@pytest.mark.parametrize("task", sorted(DECLARED))
def test_declared_split_sizes_and_base_rate_still_hold(task):
    """Every Tier-B number we publish sits on these boundaries (PLAN.md §1)."""
    splits, stats = tb.build_tier_b_examples(task)
    assert stats["split_sizes"] == DECLARED[task]["sizes"]
    n = stats["positives"] + stats["negatives"]
    assert math.isclose(stats["positives"] / n, DECLARED[task]["pos_rate"], abs_tol=5e-4)


@pytest.mark.parametrize("task", sorted(DECLARED))
def test_no_scaffold_spans_two_splits(task):
    """The whole point of a scaffold split: the test set is structurally novel. A leak
    here inflates the headline in the direction nobody checks."""
    splits, _ = tb.build_tier_b_examples(task)
    from rdkit import Chem

    seen = {}
    for name, items in splits.items():
        for mol, _q, _a in items:
            scaffold = murcko_scaffold(Chem.MolToSmiles(mol, canonical=True))
            seen.setdefault(scaffold, set()).add(name)
    straddling = {s: names for s, names in seen.items() if len(names) > 1}
    assert not straddling, f"{len(straddling)} scaffolds span splits, e.g. {list(straddling)[:3]}"


def test_bace_answers_match_the_source_csv():
    """End-to-end against the real file: the answer token carries the CSV's own label."""
    import pandas as pd
    from rdkit import Chem

    from src.experiments.molecules.data import RAW_DIR, TIER_B

    spec = TIER_B["bace"]
    df = pd.read_csv(f"{RAW_DIR}/{spec.filename}")
    truth = {}
    for row in df.to_dict("records"):
        mol = Chem.MolFromSmiles(row[spec.smiles_col])
        if mol is None:
            continue
        truth[Chem.MolToSmiles(Chem.RemoveAllHs(mol), canonical=True)] = int(row["Class"])

    splits, _ = tb.build_tier_b_examples("bace")
    checked = 0
    for items in splits.values():
        for mol, _q, answer in items:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            assert answer == (" Yes" if truth[smiles] else " No"), smiles
            checked += 1
    assert checked == 1513
