"""_load_best_model must restore bias_parameters.pt, not just the adapter.

HF's Trainer._load_best_model only reloads checkpoint formats it wrote itself —
for a PEFT model it calls load_adapter and stops. Our trainers store the
trainable graph-bias tensors separately (save_bias_parameters ->
bias_parameters.pt), so without the override the "best" model after training is
the best-step adapter paired with END-of-training bias weights, and the final
reported evaluate() silently scores a model that never existed (the recorded
metric is understated by a run-dependent amount; found 2026-07-17, measured in
src/experiments/kgqa/results/reeval_bias_bug).

Style mirrors test_graph_trainer_eval_cache.py: bare trainers (no model/args
construction) with the HF super method spied out, plus a real save->load
roundtrip through src.models.io on a tiny nn.Module.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from src.models.io import save_bias_parameters, load_bias_parameters
from src.utils.text_graph_trainer import GraphTrainer
from src.utils.text_graph_trainer_v2 import GraphTrainerV2

TRAINERS = [GraphTrainer, GraphTrainerV2]

ACTIVE = ["graph_bias"]


class TinyModel(torch.nn.Module):
    """One "backbone" param and one graph-bias param (name matches ACTIVE)."""

    def __init__(self, bias_val=0.0):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.graph_bias_proj = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.graph_bias_proj.weight.fill_(bias_val)


def _bare(cls, model, active_params, best_ckpt):
    tr = cls.__new__(cls)
    tr.active_params = active_params
    tr.model = model
    tr.state = SimpleNamespace(best_model_checkpoint=best_ckpt)
    return tr


def _spy_super():
    calls = []
    return mock.patch("transformers.Trainer._load_best_model",
                      lambda self: calls.append(True)), calls


@pytest.mark.parametrize("cls", TRAINERS)
def test_bias_restored_from_best_checkpoint(cls, tmp_path):
    """End-of-training bias values must be overwritten by the checkpoint's."""
    save_bias_parameters(TinyModel(bias_val=1.0), str(tmp_path), params=ACTIVE)

    model = TinyModel(bias_val=9.0)  # drifted end-of-training state
    backbone_before = model.backbone.weight.clone()
    tr = _bare(cls, model, ACTIVE, str(tmp_path))
    patch, calls = _spy_super()
    with patch:
        tr._load_best_model()

    assert calls, "HF's _load_best_model (adapter reload) must still run"
    assert torch.all(model.graph_bias_proj.weight == 1.0)          # restored
    assert torch.equal(model.backbone.weight, backbone_before)     # untouched


@pytest.mark.parametrize("cls", TRAINERS)
def test_none_active_params_skips_bias_load(cls, tmp_path):
    """active_params=None runs use pure-HF checkpoints — no bias file expected."""
    model = TinyModel(bias_val=9.0)
    tr = _bare(cls, model, None, str(tmp_path))  # no bias_parameters.pt here
    patch, calls = _spy_super()
    with patch:
        tr._load_best_model()
    assert calls
    assert torch.all(model.graph_bias_proj.weight == 9.0)  # left alone


@pytest.mark.parametrize("cls", TRAINERS)
def test_missing_bias_file_is_loud(cls, tmp_path):
    """A bias-arm best checkpoint without bias_parameters.pt must not pass silently."""
    tr = _bare(cls, TinyModel(), ACTIVE, str(tmp_path / "empty_ckpt"))
    (tmp_path / "empty_ckpt").mkdir()
    patch, _ = _spy_super()
    with patch, pytest.raises(FileNotFoundError):
        tr._load_best_model()


@pytest.mark.parametrize("cls", TRAINERS)
def test_no_best_checkpoint_is_safe(cls):
    """No eval ever ran -> best_model_checkpoint is None -> nothing to restore."""
    tr = _bare(cls, TinyModel(), ACTIVE, None)
    patch, calls = _spy_super()
    with patch:
        tr._load_best_model()
    assert calls


@pytest.mark.parametrize("cls", TRAINERS)
def test_empty_active_params_flat_arm_roundtrip(cls, tmp_path):
    """Flat arms (active_params=[]) save an empty bias file; reload is a no-op."""
    save_bias_parameters(TinyModel(), str(tmp_path), params=[])
    model = TinyModel(bias_val=9.0)
    tr = _bare(cls, model, [], str(tmp_path))
    patch, _ = _spy_super()
    with patch:
        tr._load_best_model()
    assert torch.all(model.graph_bias_proj.weight == 9.0)  # nothing saved, nothing loaded


def test_mismatched_names_raise(tmp_path):
    """Saved-vs-model name mismatch must raise, not silently restore nothing."""
    save_bias_parameters(TinyModel(bias_val=1.0), str(tmp_path), params=ACTIVE)

    class Renamed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.other_proj = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="no matching model parameter"):
        load_bias_parameters(Renamed(), str(tmp_path))
