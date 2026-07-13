"""
E2/E3 (TODO_cwq): multi-dataset config surface + per-dataset evaluation.

Covers: dataset-resolvable knob resolution (scalar vs mapping), single-dataset
views, cache-key stability for WebQSP regardless of CWQ settings, dataset-role
validation, the --dataset CLI alias, per-dataset metric prefixing + the
eval_sel_f1 selection metric (mean vs pinned), concat train loading, the
seeded gen-eval subsample, and the strict-metrics opt-in default.
"""

import pytest
import torch

from sweep.execute import render_flags
from src.experiments.kgqa.__main__ import build_parser, config_from_args
from src.experiments.kgqa.config import RunConfig
from src.experiments.kgqa.evaluate import PerDatasetEvalMixin, eval_indices
from src.experiments.kgqa.flat_train import FlatVersionedDataset, flat_generative_eval
from src.experiments.kgqa.process_dataset import role_splits


# ── E2.2: dataset-resolvable knobs ───────────────────────────────────────────
def test_scalar_knob_applies_to_every_dataset():
    cfg = RunConfig(train_datasets=("webqsp", "cwq"), eval_datasets=("webqsp", "cwq"))
    assert cfg.resolved_max_nodes("webqsp") == cfg.resolved_max_nodes("cwq") == 512


def test_mapping_knob_resolves_per_dataset():
    cfg = RunConfig(max_nodes={"webqsp": 512, "cwq": 1024},
                    train_datasets=("webqsp", "cwq"), eval_datasets=("cwq",))
    assert cfg.resolved_max_nodes("webqsp") == 512
    assert cfg.resolved_max_nodes("cwq") == 1024


def test_mapping_missing_referenced_dataset_rejected():
    cfg = RunConfig(max_nodes={"webqsp": 512},
                    train_datasets=("webqsp", "cwq"), eval_datasets=("cwq",))
    with pytest.raises(ValueError, match="no entry for dataset"):
        cfg.validate()


def test_for_dataset_view_collapses_to_scalars():
    cfg = RunConfig(max_nodes={"webqsp": 512, "cwq": 1024},
                    versions={"webqsp": 8, "cwq": 1},
                    train_datasets=("webqsp", "cwq"), eval_datasets=("webqsp", "cwq"))
    view = cfg.for_dataset("cwq")
    assert view.max_nodes == 1024 and view.versions == 1 and view.n_max == 20
    assert view.train_datasets == ("cwq",) and view.eval_datasets == ("cwq",)


def test_webqsp_cache_key_stable_under_cwq_settings():
    """WebQSP keys must be byte-identical whatever the CWQ knobs say."""
    single = RunConfig()
    mixed = RunConfig(max_nodes={"webqsp": 512, "cwq": 2048},
                      versions={"webqsp": 8, "cwq": 1},
                      train_datasets=("webqsp", "cwq"), eval_datasets=("webqsp", "cwq"))
    assert mixed.data_config_key("webqsp") == single.data_config_key("webqsp")
    assert "_cap2048_" in mixed.data_config_key("cwq")
    assert "_ver1_" in mixed.data_config_key("cwq")


# ── E2.1: dataset roles + validation + CLI alias ─────────────────────────────
@pytest.mark.parametrize("bad", [
    {"train_datasets": ()},
    {"eval_datasets": ("freebase",)},
    {"train_datasets": ("webqsp", "webqsp")},
    {"selection_dataset": "cwq"},                      # not in eval_datasets
    {"max_nodes": {"freebase": 512}},                  # key outside DATASETS
])
def test_validate_rejects_bad_dataset_roles(bad):
    with pytest.raises(ValueError):
        RunConfig(**bad).validate()


def test_selection_dataset_must_be_evaluated():
    cfg = RunConfig(eval_datasets=("webqsp", "cwq"), train_datasets=("cwq",),
                    selection_dataset="cwq", versions=1)
    assert cfg.validate() is cfg


def test_dataset_alias_sets_both_roles():
    cfg = config_from_args(build_parser().parse_args(
        ["--dataset", "cwq", "--versions", "1"]))
    assert cfg.train_datasets == ("cwq",) and cfg.eval_datasets == ("cwq",)


def test_dataset_alias_conflicts_with_explicit_roles():
    # (detectable only when the explicit flag differs from its default)
    args = build_parser().parse_args(
        ["--dataset", "cwq", "--train-datasets", "webqsp,cwq"])
    with pytest.raises(SystemExit):
        config_from_args(args)


def test_per_dataset_flag_roundtrip():
    params = {"max_nodes": {"webqsp": 512, "cwq": 1024}, "versions": 1,
              "train_datasets": ["webqsp", "cwq"], "eval_datasets": ["cwq"]}
    args = build_parser().parse_args(render_flags(params))
    assert args.max_nodes == {"webqsp": 512, "cwq": 1024}
    assert args.versions == 1
    assert args.train_datasets == ["webqsp", "cwq"]


def test_role_splits_builds_only_what_the_role_needs():
    cfg = RunConfig(train_datasets=("cwq",), eval_datasets=("webqsp", "cwq"), versions=1)
    assert role_splits(cfg, "cwq") == ("train", "dev", "test")
    assert role_splits(cfg, "webqsp") == ("dev", "test")


# ── E3.1: per-dataset eval fan-out + selection metric ────────────────────────
class _Base:
    """Stands in for transformers.Trainer.evaluate (the teacher-forced part)."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        return {f"{metric_key_prefix}_loss": 0.5}


class _Dummy(PerDatasetEvalMixin, _Base):
    def __init__(self, f1s, selection_dataset=None):
        self._f1s = f1s                                  # {name: f1 to report}
        self.eval_dataset = {name: object() for name in f1s}
        self._selection_dataset = selection_dataset
        self.logged = []

    def log(self, d):
        self.logged.append(d)

    def _generative_metrics(self, handle, prefix):
        name = handle if isinstance(handle, str) else \
            next(n for n, ds in self.eval_dataset.items() if ds is handle)
        return {f"{prefix}_f1": self._f1s[name]}


def test_metrics_are_dataset_prefixed_and_selection_is_mean():
    t = _Dummy({"webqsp": 0.7, "cwq": 0.5})
    metrics = t.evaluate()                               # in-training path (by name)
    assert metrics["eval_webqsp_f1"] == 0.7
    assert metrics["eval_cwq_f1"] == 0.5
    assert metrics["eval_webqsp_loss"] == 0.5            # HF metrics prefixed too
    assert metrics["eval_sel_f1"] == pytest.approx(0.6)


def test_selection_dataset_overrides_mean():
    t = _Dummy({"webqsp": 0.7, "cwq": 0.5}, selection_dataset="cwq")
    assert t.evaluate()["eval_sel_f1"] == 0.5


def test_single_dataset_run_is_still_prefixed():
    t = _Dummy({"webqsp": 0.7})
    metrics = t.evaluate()
    assert metrics["eval_webqsp_f1"] == 0.7
    assert metrics["eval_sel_f1"] == 0.7
    assert "eval_f1" not in metrics


def test_explicit_dict_scores_under_given_prefix():
    t = _Dummy({"webqsp": 0.7, "cwq": 0.5})
    metrics = t.evaluate(eval_dataset=dict(t.eval_dataset), metric_key_prefix="test")
    assert metrics["test_cwq_f1"] == 0.5
    assert metrics["test_sel_f1"] == pytest.approx(0.6)


# ── E2.3: concat train loading (versions honored per dataset) ────────────────
def test_concat_train_respects_per_dataset_versions():
    from torch.utils.data import ConcatDataset
    row = {"input_ids": [1, 2], "labels": [-100, 2]}
    webqsp = FlatVersionedDataset([dict(row)] * 8, versions=8)    # 1 question
    cwq = FlatVersionedDataset([dict(row)] * 3, versions=1)       # 3 questions
    concat = ConcatDataset([webqsp, cwq])
    assert len(webqsp) == 1 and len(cwq) == 3 and len(concat) == 4
    assert concat[0]["input_ids"] == [1, 2]


# ── E3.3: seeded fixed subsample ─────────────────────────────────────────────
def test_eval_indices_full_when_uncapped():
    assert list(eval_indices(5, None)) == [0, 1, 2, 3, 4]
    assert list(eval_indices(5, 10)) == [0, 1, 2, 3, 4]


def test_eval_indices_capped_is_fixed_sorted_subsample():
    a = list(eval_indices(3519, 128))
    b = list(eval_indices(3519, 128))
    assert a == b == sorted(set(a))                       # deterministic, no dups
    assert len(a) == 128
    assert a != list(range(128))                          # not first-n


# ── E3.2: strict metrics are opt-in ──────────────────────────────────────────
class _StubModel:
    training = False

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids=None, attention_mask=None, **kw):
        return torch.cat([input_ids, torch.tensor([[9, 9]])], dim=1)


class _StubTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "foo"


def _rows():
    return [{"input_ids": [1, 2, 3], "prefix_len": 2, "gold_answers": ["foo"]}]


def test_flat_generative_eval_strict_off_by_default():
    out = flat_generative_eval(_StubModel(), _rows(), _StubTokenizer(),
                               device="cpu", prefix="eval_webqsp")
    assert out["eval_webqsp_f1"] == 1.0
    assert not any(k.endswith("_strict") for k in out)


def test_flat_generative_eval_strict_opt_in():
    out = flat_generative_eval(_StubModel(), _rows(), _StubTokenizer(),
                               device="cpu", prefix="eval_webqsp", include_strict=True)
    assert out["eval_webqsp_f1_strict"] == 1.0
