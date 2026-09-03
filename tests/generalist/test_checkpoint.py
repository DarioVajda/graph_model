"""T4 (checkpoint half) — a checkpoint is whole, or it is not resumed from.

The tests build a checkpoint directory shaped like the real one: a PEFT adapter,
a ``bias_parameters.pt`` written by the same ``save_bias_parameters`` the trainer
uses, stand-ins for HF's ``optimizer.pt`` / ``scheduler.pt``, and then
``finalize`` on top. Everything else here is about the two failure modes that
have actually cost runs in this project: a checkpoint written by a job that was
killed part-way, and an adapter paired with the wrong graph-bias tensors
(2026-07-17).

The model is a 2-layer, 64-hidden Llama with a LoRA adapter and one parameter
named ``graph_bias_*``. It is not a GTLM — nothing here touches attention — but
it has the two things a checkpoint has to keep together.
"""

import json
import os

import pytest
import torch

from src.generalist.checkpoint import (
    CheckpointError, bias_norm, discontinuities, finalize, is_complete, is_pinned,
    latest, list_checkpoints, pin, restore_extras, rotate, verify,
)
from src.generalist.schedule import Schedule
from src.models.io import save_bias_parameters
from tests.helpers.tiny_model import BASE_CONFIG

ACTIVE_PARAMS = ["graph_bias"]


def _tiny_peft_model(seed=0):
    """A tiny causal LM with both halves of the checkpoint: a LoRA adapter and a
    graph-bias parameter that lives outside it."""
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    model = LlamaForCausalLM(LlamaConfig(**BASE_CONFIG))
    model.register_parameter(
        "graph_bias_weights", torch.nn.Parameter(torch.randn(4, 8)))
    return get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))


def _write_hf_shaped_checkpoint(ckpt_dir, model):
    """What the trainer's ``super()._save_checkpoint`` + ``save_model`` leave
    behind, before ``finalize`` adds the harness's files."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)                      # adapter_* files
    save_bias_parameters(model, ckpt_dir, ACTIVE_PARAMS)  # bias_parameters.pt
    torch.save({"state": {}}, os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save({"last_epoch": 7}, os.path.join(ckpt_dir, "scheduler.pt"))
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as fh:
        json.dump({"global_step": 7}, fh)


def _state(step=7, **over):
    base = {"step": step, "mixture_hash": "mix-a", "tokens_per_step": 4096,
            "lr": 1e-4, "bias_lr": 1e-2, "hardware": "NVIDIA H100",
            "examples_per_task": {"mol/bace": 12}, "schema_version": 3}
    base.update(over)
    return base


@pytest.fixture(scope="module")
def model():
    return _tiny_peft_model()


def _make_checkpoint(run_dir, step, model=None, complete=True, schedule=None,
                     state=None):
    ckpt = os.path.join(run_dir, f"checkpoint-{step}")
    os.makedirs(ckpt, exist_ok=True)
    if model is not None:
        _write_hf_shaped_checkpoint(ckpt, model)
    if not complete:
        # A job killed mid-write: HF's files are there, ours are not.
        return ckpt
    finalize(ckpt,
             model=model, active_params=ACTIVE_PARAMS if model is not None else None,
             schedule=schedule or Schedule.training(warmup_steps=4),
             sampler_state={"cursors": {"mol/bace": 12}, "pass_ids": {"mol/bace": 0}},
             state=state or _state(step=step))
    return ckpt


# ── writing and verifying ───────────────────────────────────────────────────

class TestFinalizeAndVerify:

    def test_finalize_writes_the_harness_files_and_the_marker_last(self, model, tmp_path):
        ckpt = _make_checkpoint(str(tmp_path), 7, model=model)
        for name in ("schedule.json", "sampler.json", "state.json", "COMPLETE",
                     "bias_parameters.pt", "adapter_config.json"):
            assert os.path.exists(os.path.join(ckpt, name)), name
        assert is_complete(ckpt)

        state = json.load(open(os.path.join(ckpt, "state.json")))
        assert state["step"] == 7
        assert state["schema_version"] == 3          # the trainer's, not overwritten
        assert state["bias_norm"] == pytest.approx(bias_norm(model, ACTIVE_PARAMS))
        assert state["written_at"]
        assert "state.json" in state["files"]
        assert "bias_parameters.pt" in state["files"]
        assert "COMPLETE" not in state["files"]      # the marker is not its own content

    def test_verify_passes_on_the_model_it_was_written_from(self, model, tmp_path):
        ckpt = _make_checkpoint(str(tmp_path), 7, model=model)
        state = verify(ckpt, model=model, active_params=ACTIVE_PARAMS)
        assert state["step"] == 7
        # active_params falls back to what state.json recorded.
        assert verify(ckpt, model=model)["step"] == 7

    def test_verify_rejects_perturbed_bias_tensors(self, tmp_path):
        """The 2026-07-17 failure: the adapter is fine, the bias tensors are not
        the ones it was trained with, and nothing crashes on its own."""
        m = _tiny_peft_model(seed=1)
        ckpt = _make_checkpoint(str(tmp_path), 7, model=m)
        assert verify(ckpt, model=m, active_params=ACTIVE_PARAMS)

        # Re-write bias_parameters.pt from a perturbed model without re-writing
        # state.json — exactly what a mispaired save/copy produces.
        with torch.no_grad():
            for name, p in m.named_parameters():
                if "graph_bias" in name:
                    p.add_(0.01)
        save_bias_parameters(m, ckpt, ACTIVE_PARAMS)

        with pytest.raises(CheckpointError, match="bias norm"):
            verify(ckpt, model=m, active_params=ACTIVE_PARAMS)

    def test_verify_accepts_a_checkpoint_with_no_bias_channel_at_all(
            self, model, tmp_path):
        """The flat arm, where the fingerprint is absent on both sides.

        `active_params` names the bias group whatever the arm is, but on a
        single-node graph there is no bias module holding those parameters, so
        `bias_norm` finds none and the checkpoint recorded none. Refusing that
        pairing made the flat cross-check unforkable: the run trained its 1510
        steps and then could not be annealed, which is the leg the reportable
        number comes from. Both sides absent is agreement that there is nothing
        to pair; one side absent is still an error.
        """
        flat_params = ("a_module_this_model_does_not_have",)
        ckpt = os.path.join(str(tmp_path), "flat", "checkpoint-7")
        os.makedirs(ckpt, exist_ok=True)
        _write_hf_shaped_checkpoint(ckpt, model)
        finalize(ckpt, model=model, active_params=flat_params,
                 schedule=Schedule.training(warmup_steps=4),
                 sampler_state={"cursors": {}, "pass_ids": {}},
                 state=_state(step=7))
        assert json.load(open(os.path.join(ckpt, "state.json")))["bias_norm"] is None
        assert verify(ckpt, model=model, active_params=flat_params)["step"] == 7

        # And the asymmetric case is untouched: a fingerprint with nothing to
        # check it against is still the failure this guard is for.
        path = os.path.join(ckpt, "state.json")
        state = json.load(open(path))
        state["bias_norm"] = 1.25
        with open(path, "w") as fh:
            json.dump(state, fh)
        with pytest.raises(CheckpointError, match="fingerprint"):
            verify(ckpt, model=model, active_params=flat_params)

    def test_verify_refuses_a_directory_without_complete(self, model, tmp_path):
        ckpt = _make_checkpoint(str(tmp_path), 7, model=model)
        os.remove(os.path.join(ckpt, "COMPLETE"))
        assert not is_complete(ckpt)
        with pytest.raises(CheckpointError, match="COMPLETE"):
            verify(ckpt, model=model, active_params=ACTIVE_PARAMS)

    def test_verify_refuses_a_missing_listed_file(self, model, tmp_path):
        ckpt = _make_checkpoint(str(tmp_path), 7, model=model)
        os.remove(os.path.join(ckpt, "optimizer.pt"))
        with pytest.raises(CheckpointError, match="optimizer.pt"):
            verify(ckpt)

    def test_verify_refuses_a_missing_bias_file(self, model, tmp_path):
        ckpt = _make_checkpoint(str(tmp_path), 7, model=model)
        os.remove(os.path.join(ckpt, "bias_parameters.pt"))
        # Take it out of the file list too, so the check under test is the one
        # that fires rather than the generic missing-file check.
        path = os.path.join(ckpt, "state.json")
        state = json.load(open(path))
        state["files"] = [f for f in state["files"] if f != "bias_parameters.pt"]
        with open(path, "w") as fh:
            json.dump(state, fh)
        with pytest.raises(CheckpointError, match="bias_parameters.pt"):
            verify(ckpt, active_params=ACTIVE_PARAMS)

    def test_bias_norm_is_none_without_an_arm(self, model):
        assert bias_norm(model, []) is None
        assert bias_norm(None, ACTIVE_PARAMS) is None
        assert bias_norm(model, ["no_such_parameter"]) is None


# ── restore ─────────────────────────────────────────────────────────────────

class TestRestore:

    def test_restore_extras_round_trips_schedule_and_sampler(self, tmp_path):
        schedule = Schedule.training(warmup_steps=10)
        schedule.append_rewarm(at_step=100, rewarm_steps=20, from_factor=0.3)
        ckpt = _make_checkpoint(str(tmp_path), 120, schedule=schedule)

        restored, sampler_state, state = restore_extras(ckpt)
        assert restored.segments == schedule.segments
        assert restored.factor(110) == pytest.approx(schedule.factor(110))
        assert restored.position(125) == schedule.position(125)
        assert sampler_state == {"cursors": {"mol/bace": 12},
                                 "pass_ids": {"mol/bace": 0}}
        assert state["step"] == 120


# ── enumeration, pinning, rotation ──────────────────────────────────────────

class TestRunDirectory:

    def test_latest_is_the_highest_complete_step(self, tmp_path):
        run = str(tmp_path)
        for step in (100, 200, 1000):
            _make_checkpoint(run, step)
        _make_checkpoint(run, 1200, complete=False)   # killed mid-write

        # By step number, not lexically and not by mtime.
        assert [os.path.basename(p) for p in list_checkpoints(run)] == [
            "checkpoint-100", "checkpoint-200", "checkpoint-1000", "checkpoint-1200"]
        assert os.path.basename(latest(run)) == "checkpoint-1000"

    def test_latest_is_none_on_an_empty_run(self, tmp_path):
        assert latest(str(tmp_path)) is None
        assert latest(os.path.join(str(tmp_path), "nope")) is None
        assert list_checkpoints(str(tmp_path)) == []

    def test_rotate_keeps_recent_pinned_and_incomplete(self, tmp_path):
        run = str(tmp_path)
        for step in (100, 200, 300, 400):
            _make_checkpoint(run, step)
        _make_checkpoint(run, 500, complete=False)
        pin(os.path.join(run, "checkpoint-100"), reason="anneal fork")
        assert is_pinned(os.path.join(run, "checkpoint-100"))

        result = rotate(run, keep=2)
        assert [os.path.basename(p) for p in result["deleted"]] == ["checkpoint-200"]
        assert [os.path.basename(p) for p in result["pinned"]] == ["checkpoint-100"]
        assert [os.path.basename(p) for p in result["incomplete"]] == ["checkpoint-500"]
        assert sorted(os.path.basename(p) for p in result["kept"]) == [
            "checkpoint-100", "checkpoint-300", "checkpoint-400"]

        left = {os.path.basename(p) for p in list_checkpoints(run)}
        assert left == {"checkpoint-100", "checkpoint-300", "checkpoint-400",
                        "checkpoint-500"}

    def test_rotate_with_a_generous_keep_deletes_nothing(self, tmp_path):
        run = str(tmp_path)
        for step in (10, 20):
            _make_checkpoint(run, step)
        assert rotate(run, keep=5)["deleted"] == []
        assert len(list_checkpoints(run)) == 2


# ── discontinuities ─────────────────────────────────────────────────────────

class TestDiscontinuities:

    def test_nothing_changed(self):
        assert discontinuities(_state(), _state()) == []

    def test_names_exactly_the_changed_keys(self):
        prev = _state()
        assert discontinuities(prev, _state(mixture_hash="mix-b")) == ["mixture_hash"]
        assert discontinuities(prev, _state(lr=3e-4)) == ["lr"]
        assert discontinuities(prev, _state(bias_lr=5e-3, tokens_per_step=8192)) == [
            "tokens_per_step", "bias_lr"]
        assert discontinuities(prev, _state(hardware="NVIDIA B200")) == ["hardware"]

    def test_unwatched_fields_are_not_discontinuities(self):
        # A different step, or more examples seen, is what a resume *is*.
        assert discontinuities(_state(step=7), _state(step=8)) == []
        assert discontinuities(_state(), _state(examples_per_task={"mol/bace": 99})) == []

    def test_a_key_present_on_one_side_only_counts_as_changed(self):
        prev = dict(_state())
        prev.pop("hardware")
        assert discontinuities(prev, _state()) == ["hardware"]
        assert discontinuities({}, {}) == []
