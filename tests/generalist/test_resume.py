"""T4 (trainer half) — a resume continues the run, or it refuses to start.

Six claims, each of which has a way of being false that produces *numbers*
rather than a crash:

* **A resume is the same run.** Three steps, a checkpoint, three more steps must
  draw the same examples in the same order, produce the same loss at each step,
  and end at the same parameters as six steps in one go. Comparing only the
  endpoint would miss a resume that saw the right examples in the wrong order,
  so the batch *sequence* is compared too.
* **An incomplete checkpoint is refused**, and a ``bias_parameters.pt`` that does
  not belong to its ``state.json`` aborts before a single step. That second one
  is the 2026-07-17 bug: an adapter reloaded beside the wrong bias tensors
  trains and evaluates perfectly happily at silently wrong numbers.
* **The normalisation is applied exactly once.** The same two optimizer steps at
  ``gradient_accumulation_steps`` 1, 2 and 4 must reach the same parameters.
  ``MixtureLoss`` normalises by the whole step's example count and HF's
  ``training_step`` may divide by the accumulation count on top of it; this is
  the assertion that says only one of the two survives, and it is run under both
  branches of HF's condition rather than under the one this repo's models happen
  to take.
* **The LR that is applied is the schedule's**, read off ``optimizer.param_groups``
  during training rather than off the schedule object.
* **A discontinuity appends a re-warm** and a clean resume does not.
* **The bias channel is trained and restored** — it moves off its zero init, the
  fingerprint in ``state.json`` matches the live norm, and the resumed model
  carries the saved values.

Everything runs on CPU against the tiny three-task run in ``tiny_run.py``.
"""

import json
import os
import shutil

import pytest
import torch

from src.generalist import checkpoint as ckpt_mod
from src.generalist.checkpoint import CheckpointError
from src.generalist.trainer import TrainerError, align_to_accumulation
from src.models.io import save_bias_parameters
from tests.generalist import tiny_run
from tests.generalist.tiny_run import (
    ACTIVE_PARAMS, LearningRateProbe, bias_tensors, build_model, build_trainer,
    trainable_tensors,
)

#: bf16 keeps ~8 mantissa bits, so 2**-8 is the relative bound the design asks a
#: resume to hold to. The comparison itself runs in fp32 on CPU, where the two
#: runs are in practice identical — the loose bound is the contract, the tight
#: assertions below are what actually holds.
BF16_RTOL = 2.0 ** -8
BF16_ATOL = 1e-6


def _assert_params_close(a: dict, b: dict, rtol=BF16_RTOL, atol=BF16_ATOL):
    assert set(a) == set(b), f"different parameter sets: {set(a) ^ set(b)}"
    for name in sorted(a):
        torch.testing.assert_close(a[name], b[name], rtol=rtol, atol=atol,
                                   msg=lambda m, n=name: f"{n}: {m}")


def _moved(before: dict, after: dict, threshold: float = 1e-5) -> float:
    """Largest absolute parameter change — a precondition, not an assertion.

    Every "the two runs agree" test below would pass on two runs that both did
    nothing, so each one first establishes that something happened.
    """
    return max(float((after[n] - before[n]).abs().max()) for n in before)


# ─────────────────────────────────────────────────────────────────────────────
# The regrouping helper
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignToAccumulation:
    """The sampler sizes micro-batches from a token budget; HF wants a fixed
    count. The reshape between them must conserve examples exactly."""

    def _keys(self, groups):
        return sorted(item for g in groups for item in g)

    def test_merges_down_to_the_target(self):
        batches = [[1, 2], [3, 4], [5], [6, 7, 8]]
        out = align_to_accumulation(batches, 2)
        assert len(out) == 2
        assert self._keys(out) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_splits_up_to_the_target(self):
        out = align_to_accumulation([[1, 2, 3, 4, 5, 6]], 3)
        assert len(out) == 3
        assert self._keys(out) == [1, 2, 3, 4, 5, 6]

    def test_a_matching_count_is_left_alone(self):
        batches = [[1, 2], [3]]
        assert align_to_accumulation(batches, 2) == batches

    def test_too_few_examples_is_an_error_not_an_empty_batch(self):
        with pytest.raises(TrainerError, match="micro-batches"):
            align_to_accumulation([[1], [2]], 4)


# ─────────────────────────────────────────────────────────────────────────────
# Bit-exact resume
# ─────────────────────────────────────────────────────────────────────────────

def _flat_keys(trainer):
    """Every (task, row, step) the trainer was handed, micro-batches flattened."""
    return [key for batch in trainer.batch_keys for key in batch]


class TestResumeIsTheSameRun:

    @pytest.fixture(scope="class")
    def runs(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("bitexact")

        # (a) six steps in one go.
        whole, whole_model, _s, _sc = build_trainer(
            str(root / "whole"), max_steps=6, seed=0)
        whole_init = trainable_tensors(whole_model)
        whole.train()

        # (b) three steps, checkpoint, then a fresh trainer resumes.
        first, first_model, _s, _sc = build_trainer(
            str(root / "split"), max_steps=3, save_steps=3, seed=0)
        _assert_params_close(trainable_tensors(first_model), whole_init,
                             rtol=0, atol=0)     # identical starting point
        first.train()
        ckpt = os.path.join(str(root / "split"), "checkpoint-3")
        assert ckpt_mod.is_complete(ckpt)

        second, second_model, _s, _sc = build_trainer(
            str(root / "split"), max_steps=6, save_steps=3, seed=0)
        second.resume(ckpt)

        return {
            "whole": whole, "whole_model": whole_model,
            "first": first, "second": second, "second_model": second_model,
            "init": whole_init, "ckpt": ckpt, "root": str(root),
        }

    def test_the_same_examples_in_the_same_order(self, runs):
        assert _flat_keys(runs["first"]) + _flat_keys(runs["second"]) == \
            _flat_keys(runs["whole"])
        # And the run did draw something: 6 steps x 8 examples per step.
        assert len(_flat_keys(runs["whole"])) == 48
        # Steps are contiguous 0..5 and the resumed half starts where the first
        # stopped, which is the D4.1 promise being checked.
        assert sorted({k[2] for k in _flat_keys(runs["first"])}) == [0, 1, 2]
        assert sorted({k[2] for k in _flat_keys(runs["second"])}) == [3, 4, 5]

    def test_the_same_loss_at_every_step(self, runs):
        joined = runs["first"].step_losses + runs["second"].step_losses
        reference = runs["whole"].step_losses
        assert len(joined) == len(reference)
        for i, (a, b) in enumerate(zip(joined, reference)):
            assert a == pytest.approx(b, rel=BF16_RTOL, abs=BF16_ATOL), \
                f"micro-batch {i}: resumed {a} vs uninterrupted {b}"

    def test_the_same_parameters_at_the_end(self, runs):
        whole = trainable_tensors(runs["whole_model"])
        resumed = trainable_tensors(runs["second_model"])
        assert _moved(runs["init"], whole) > 1e-5, "the reference run did not train"
        _assert_params_close(resumed, whole)

    def test_per_task_accounting_survives_the_boundary(self, runs):
        whole = runs["whole"].examples_per_task
        resumed = runs["second"].examples_per_task
        assert sum(whole.values()) == 48
        assert whole == resumed
        # Every task in the mixture actually trained (the --magnetic-groups class
        # of bug: a task in the config and absent from the gradient).
        assert all(v > 0 for v in whole.values())
        assert len(runs["whole"].task_loss_history) == 6


# ─────────────────────────────────────────────────────────────────────────────
# Refusals
# ─────────────────────────────────────────────────────────────────────────────

class TestARefusedResumeCostsNoStep:

    @pytest.fixture(scope="class")
    def source_checkpoint(self, tmp_path_factory):
        """One two-step run whose checkpoint the refusal tests copy and damage."""
        root = tmp_path_factory.mktemp("refusals")
        trainer, model, _s, _sc = build_trainer(
            str(root / "run"), max_steps=2, save_steps=2, seed=0)
        trainer.train()
        return os.path.join(str(root / "run"), "checkpoint-2")

    def _copy(self, source, tmp_path):
        target = os.path.join(str(tmp_path), "run", "checkpoint-2")
        shutil.copytree(source, target)
        return target

    def test_a_directory_without_complete_is_refused(self, source_checkpoint, tmp_path):
        ckpt = self._copy(source_checkpoint, tmp_path)
        os.remove(os.path.join(ckpt, ckpt_mod.COMPLETE_MARKER))

        trainer, _m, _s, _sc = build_trainer(str(tmp_path / "run"), max_steps=4,
                                             seed=0)
        with pytest.raises(CheckpointError, match="COMPLETE"):
            trainer.prepare_resume(ckpt)
        assert trainer.state.global_step == 0
        assert trainer.batch_keys == []

    def test_latest_ignores_an_incomplete_directory(self, source_checkpoint, tmp_path):
        ckpt = self._copy(source_checkpoint, tmp_path)
        os.remove(os.path.join(ckpt, ckpt_mod.COMPLETE_MARKER))
        trainer, _m, _s, _sc = build_trainer(str(tmp_path / "run"), max_steps=4,
                                             seed=0)
        with pytest.raises(CheckpointError, match="no complete checkpoint"):
            trainer.prepare_resume("latest")

    def test_a_tampered_bias_norm_aborts_before_a_step(self, source_checkpoint,
                                                       tmp_path):
        """The adapter is intact, ``state.json`` is intact, and the bias tensors
        are somebody else's. Nothing raises on its own — the fingerprint has to."""
        ckpt = self._copy(source_checkpoint, tmp_path)
        other = build_model(seed=7)
        with torch.no_grad():
            for name, p in other.named_parameters():
                if any(a in name for a in ACTIVE_PARAMS):
                    p.add_(0.5)
        save_bias_parameters(other, ckpt, ACTIVE_PARAMS)

        trainer, _m, _s, _sc = build_trainer(str(tmp_path / "run"), max_steps=4,
                                             seed=0)
        with pytest.raises(CheckpointError, match="bias norm"):
            trainer.prepare_resume(ckpt)
        assert trainer.state.global_step == 0
        assert trainer.batch_keys == []

    def test_a_schema_version_change_is_refused(self, source_checkpoint, tmp_path):
        ckpt = self._copy(source_checkpoint, tmp_path)
        path = os.path.join(ckpt, ckpt_mod.STATE_FILE)
        with open(path) as fh:
            state = json.load(fh)
        state["schema_version"] = "not-this-one"
        with open(path, "w") as fh:
            json.dump(state, fh)

        trainer, _m, _s, _sc = build_trainer(str(tmp_path / "run"), max_steps=4,
                                             seed=0)
        with pytest.raises(CheckpointError, match="schema version"):
            trainer.prepare_resume(ckpt)

    def test_an_architecture_change_is_refused(self, source_checkpoint, tmp_path):
        ckpt = self._copy(source_checkpoint, tmp_path)
        path = os.path.join(ckpt, ckpt_mod.STATE_FILE)
        with open(path) as fh:
            state = json.load(fh)
        state["architecture_hash"] = "0" * 64
        with open(path, "w") as fh:
            json.dump(state, fh)

        trainer, _m, _s, _sc = build_trainer(str(tmp_path / "run"), max_steps=4,
                                             seed=0)
        with pytest.raises(CheckpointError, match="architecture hash"):
            trainer.prepare_resume(ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# The one that matters: exactly one normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalisationIsInvariantToAccumulation:
    """Two optimizer steps, chopped three ways, must land in the same place.

    ``MixtureLoss`` divides by the whole step's example count (8 here, whatever
    the accumulation) and HF's ``training_step`` divides the returned loss by
    ``gradient_accumulation_steps`` when the model's forward takes no ``**kwargs``.
    If both applied, accumulation 4 would take a step a quarter the size of
    accumulation 1 — a change that shows up as "the run is slower to converge",
    never as an error.
    """

    STEPS = 2

    def _run(self, tmp_path, accumulation, force_hf_division=False):
        model = build_model(seed=0)
        init = trainable_tensors(model)
        trainer, model, _s, _sc = build_trainer(
            os.path.join(str(tmp_path), f"acc{accumulation}"
                                        f"{'-forced' if force_hf_division else ''}"),
            model=model, accumulation_steps=accumulation, max_steps=self.STEPS,
            seed=0)
        if force_hf_division:
            # Take HF's other branch on purpose. Every model in this repo has
            # `**kwargs` in its forward, so `model_accepts_loss_kwargs` is True
            # and the division is dead code today; a backbone without it would
            # turn the branch on with no other symptom.
            trainer.model_accepts_loss_kwargs = False
        assert trainer._hf_accumulation_scale() == (
            float(accumulation) if force_hf_division else 1.0)
        trainer.train()
        return trainer, init, trainable_tensors(model)

    @pytest.fixture(scope="class")
    def baseline(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("accum1")
        return self._run(root, 1)

    @pytest.mark.parametrize("accumulation", [2, 4])
    def test_the_same_update_at_any_accumulation(self, baseline, tmp_path,
                                                 accumulation):
        base_trainer, init, base_final = baseline
        trainer, _init, final = self._run(tmp_path, accumulation)

        assert _moved(init, base_final) > 1e-5, "the baseline run did not train"
        # The same examples, however they were grouped into micro-batches.
        assert sorted(_flat_keys(trainer)) == sorted(_flat_keys(base_trainer))
        assert len(trainer.batch_keys) == accumulation * self.STEPS
        _assert_params_close(final, base_final, rtol=1e-4, atol=1e-7)

    def test_the_same_update_when_hf_divides(self, baseline, tmp_path):
        """The compensation, not the branch, is what makes it come out right."""
        _base_trainer, _init, base_final = baseline
        _t, _i, final = self._run(tmp_path, 2, force_hf_division=True)
        _assert_params_close(final, base_final, rtol=1e-4, atol=1e-7)

    def test_a_step_is_the_whole_step_however_it_is_split(self, baseline, tmp_path):
        """The per-micro-batch losses of a step sum to the step's loss.

        The loss the trainer hands back is already the micro-batch's *share* of
        the step, so the sum over a step is accumulation-invariant even though
        the individual terms are not.
        """
        base_trainer, _init, _final = baseline
        trainer, _i, _f = self._run(tmp_path, 4)
        for step in range(self.STEPS):
            base = base_trainer.step_losses[step]
            chunk = trainer.step_losses[step * 4:(step + 1) * 4]
            assert sum(chunk) == pytest.approx(base, rel=1e-4, abs=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# The applied learning rate
# ─────────────────────────────────────────────────────────────────────────────

class TestAppliedLearningRate:

    def test_param_groups_follow_the_schedule(self, tmp_path):
        probe = LearningRateProbe()
        lr, bias_lr = 1e-2, 5e-2
        trainer, _m, _s, schedule = build_trainer(
            str(tmp_path / "lr"), max_steps=6, warmup_steps=5, lr_min_factor=0.25,
            lr=lr, bias_lr=bias_lr, seed=0, callbacks=[probe])
        trainer.train()

        for step in (0, 1, 3, 5):
            seen_lora, seen_bias = probe.lrs_at(step)
            assert seen_lora == pytest.approx(schedule.factor(step) * lr, rel=1e-9)
            # The bias group's base LR is `bias_lr`, not `lr`: `make_lr_scheduler`
            # gives each group a lambda over its OWN base, which is the whole
            # reason a segment carries a factor rather than a learning rate.
            assert seen_bias == pytest.approx(
                schedule.bias_factor(step) * bias_lr, rel=1e-9)

        # The warmup is not flat, or the assertion above would hold for any
        # constant schedule.
        assert probe.lrs_at(0)[0] < probe.lrs_at(3)[0] < probe.lrs_at(5)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Discontinuities
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscontinuityAppendsARewarm:

    @pytest.fixture(scope="class")
    def parent(self, tmp_path_factory):
        """A parent that has reached its stable phase (warmup 1, stopped at 2).

        The re-warm is appended to the schedule the *checkpoint* carries, not to
        the resuming run's, so it is the parent's warmup length that decides
        whether step 2 is past it.
        """
        root = tmp_path_factory.mktemp("rewarm")
        trainer, _m, _s, _sc = build_trainer(str(root / "run"), max_steps=2,
                                             warmup_steps=1, save_steps=2, seed=0)
        trainer.train()
        return os.path.join(str(root / "run"), "checkpoint-2")

    @pytest.fixture(scope="class")
    def parent_mid_warmup(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("rewarm_warm")
        trainer, _m, _s, _sc = build_trainer(str(root / "run"), max_steps=2,
                                             warmup_steps=5, save_steps=2, seed=0)
        trainer.train()
        return os.path.join(str(root / "run"), "checkpoint-2")

    @pytest.fixture(scope="class")
    def parent_no_warmup(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("rewarm_none")
        trainer, _m, _s, _sc = build_trainer(str(root / "run"), max_steps=2,
                                             warmup_steps=0, save_steps=2, seed=0)
        trainer.train()
        return os.path.join(str(root / "run"), "checkpoint-2")

    def test_a_clean_resume_does_not_rewarm(self, parent, tmp_path):
        entries = []
        trainer, _m, _s, schedule = build_trainer(
            str(tmp_path / "run"), max_steps=4, seed=0,
            lineage_hook=entries.append)
        trainer.prepare_resume(parent)

        assert [s.kind for s in trainer.schedule.segments] == ["warmup", "stable"]
        assert entries and entries[-1]["causes"] == []
        assert entries[-1]["parent_step"] == 2

    def test_a_changed_learning_rate_rewarms_and_is_recorded(self, parent, tmp_path):
        entries = []
        trainer, _m, _s, _sc = build_trainer(
            str(tmp_path / "run"), max_steps=4, seed=0, lr=3e-4,
            lineage_hook=entries.append)
        trainer.prepare_resume(parent)

        kinds = [s.kind for s in trainer.schedule.segments]
        assert kinds == ["warmup", "stable", "rewarm", "stable"]
        rewarm = trainer.schedule.segments[2]
        assert rewarm.start == 2
        assert rewarm.factor_start == pytest.approx(trainer.rewarm_from)
        assert rewarm.factor_end == pytest.approx(1.0)
        assert entries[-1]["causes"] == ["lr"]
        assert entries[-1]["rewarm_steps"] == rewarm.steps

    def test_a_changed_mixture_rewarms(self, parent, tmp_path):
        entries = []
        trainer, _m, _s, _sc = build_trainer(
            str(tmp_path / "run"), max_steps=4, seed=0,
            weights={"t/alpha": 3.0, "t/beta": 1.0, "t/gamma": 1.0},
            lineage_hook=entries.append)
        trainer.prepare_resume(parent)

        assert "rewarm" in [s.kind for s in trainer.schedule.segments]
        assert entries[-1]["causes"] == ["mixture_hash"]

    def test_the_rewarm_length_must_come_from_somewhere(self, parent_no_warmup,
                                                        tmp_path):
        """No warmup segment and no configured length: a re-warm cannot be
        guessed, so the resume says so rather than picking a number."""
        trainer, _m, _s, _sc = build_trainer(
            str(tmp_path / "run"), max_steps=4, seed=0, lr=3e-4, warmup_steps=0)
        with pytest.raises(TrainerError, match="rewarm_steps"):
            trainer.prepare_resume(parent_no_warmup)

    def test_an_explicit_rewarm_length_is_used(self, parent_no_warmup, tmp_path):
        trainer, _m, _s, _sc = build_trainer(
            str(tmp_path / "run"), max_steps=4, seed=0, lr=3e-4, warmup_steps=0,
            rewarm_steps=17)
        trainer.prepare_resume(parent_no_warmup)
        assert [s.kind for s in trainer.schedule.segments] == [
            "stable", "rewarm", "stable"]
        assert trainer.schedule.segments[1].steps == 17

    def test_a_discontinuity_inside_the_warmup_records_but_does_not_rewarm(
            self, parent_mid_warmup, tmp_path):
        """Step 2 of a 5-step warmup is already climbing from a low LR; a second
        warmup on top of it would be strictly worse than the one running."""
        entries = []
        trainer, _m, _s, _sc = build_trainer(
            str(tmp_path / "run"), max_steps=4, seed=0, lr=3e-4,
            lineage_hook=entries.append)
        trainer.prepare_resume(parent_mid_warmup)

        assert [s.kind for s in trainer.schedule.segments] == ["warmup", "stable"]
        assert entries[-1]["causes"] == ["lr"]
        assert entries[-1]["rewarm_steps"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# The bias channel
# ─────────────────────────────────────────────────────────────────────────────

class TestBiasParametersAreTrainedAndRestored:
    """The 2026-07-17 regression, from both ends: the bias has to move, and it
    has to come back."""

    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("bias")
        model = build_model(seed=0)
        at_init = bias_tensors(model)
        trainer, model, _s, _sc = build_trainer(
            str(root / "run"), model=model, max_steps=3, save_steps=3, seed=0)
        trainer.train()
        return {"init": at_init, "final": bias_tensors(model),
                "ckpt": os.path.join(str(root / "run"), "checkpoint-3")}

    def test_the_bias_moves_off_its_init(self, trained):
        assert trained["init"], "the model has no graph-bias parameters at all"
        assert all(float(t.abs().max()) == 0.0 for t in trained["init"].values())
        assert _moved(trained["init"], trained["final"]) > 1e-6

    def test_the_fingerprint_matches_the_live_norm(self, trained):
        state = ckpt_mod.read_state(trained["ckpt"])
        expected = torch.sqrt(sum((t.double() ** 2).sum()
                                  for t in trained["final"].values()))
        assert state["bias_norm"] == pytest.approx(float(expected), rel=1e-9)
        assert state["bias_norm"] > 0.0

    def test_a_resume_restores_the_saved_values(self, trained, tmp_path):
        fresh = build_model(seed=3)          # a different init on purpose
        assert not all(
            torch.allclose(fresh_t, trained["final"][name])
            for name, fresh_t in bias_tensors(fresh).items())

        trainer, model, _s, _sc = build_trainer(
            str(tmp_path / "run"), model=fresh, max_steps=6, seed=0)
        trainer.prepare_resume(trained["ckpt"])
        _assert_params_close(bias_tensors(model), trained["final"], rtol=0, atol=0)


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────────────────────

class TestConstructionRefusals:

    def test_load_best_model_at_end_is_refused(self, tmp_path):
        with pytest.raises(TrainerError, match="load_best_model_at_end"):
            build_trainer(str(tmp_path / "run"), max_steps=2, seed=0,
                          args_overrides={"load_best_model_at_end": True})

    def test_an_accumulation_mismatch_is_refused(self, tmp_path):
        registry, mixture = tiny_run.build_mixture()
        sampler = tiny_run.build_sampler(mixture, accumulation_steps=2)
        with pytest.raises(TrainerError, match="accumulation"):
            build_trainer(str(tmp_path / "run"), max_steps=2, seed=0,
                          accumulation_steps=1, sampler=sampler,
                          mixture=mixture, registry=registry)
