"""T5 — the horizon-free WSD schedule (DESIGN.md D5.2).

What has to hold, and why each of these is a test rather than an assumption:

* the factor at every segment boundary, because an off-by-one at a boundary is
  invisible in a loss curve and shows up only as a run that trained at the wrong
  LR for a few hundred steps;
* truncation, because appending a re-warm or a decay rewrites the segment the run
  is currently in and a wrong truncation would move the schedule under a resume;
* the bias ratio, because it is the one multiplier that touches only half the
  parameter groups;
* and the LambdaLR round-trip, because HF resumes a scheduler by
  ``load_state_dict`` into lambdas that this module rebuilt from JSON — if
  anything other than ``last_epoch`` had to survive that, every resume would
  restart the schedule silently.
"""

import copy

import pytest
import torch

from src.generalist.schedule import (
    Schedule, ScheduleError, Segment, make_lr_scheduler,
)


def _two_group_optimizer(lr=1e-3, bias_lr=1e-2):
    """The shape ``GraphTrainerV2.create_optimizer`` produces: one LoRA group and
    one group tagged ``is_bias``."""
    lora = torch.nn.Parameter(torch.zeros(2))
    bias = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD([
        {"params": [lora], "lr": lr, "is_bias": False},
        {"params": [bias], "lr": bias_lr, "is_bias": True},
    ])


def _scheduler(optimizer, schedule):
    """``make_lr_scheduler`` plus one no-op optimizer step (no grads, so nothing
    moves), which puts the two in the order training uses them — optimizer then
    scheduler — and keeps torch from warning about it on every step below."""
    lr_sched = make_lr_scheduler(optimizer, schedule)
    optimizer.step()
    return lr_sched


# ── segment boundaries ──────────────────────────────────────────────────────

class TestFactors:

    def test_hand_built_schedule_at_every_boundary(self):
        sched = Schedule([
            Segment("warmup", 0, 10, 0.0, 1.0),
            Segment("stable", 10, 20, 1.0, 1.0),
            Segment("rewarm", 30, 10, 0.5, 1.0),
            Segment("decay", 40, 10, 1.0, 0.1, shape="linear"),
        ])
        assert sched.factor(0) == pytest.approx(0.0)
        assert sched.factor(5) == pytest.approx(0.5)
        assert sched.factor(10) == pytest.approx(1.0)     # first stable step
        assert sched.factor(29) == pytest.approx(1.0)
        assert sched.factor(30) == pytest.approx(0.5)     # re-warm starts here
        assert sched.factor(35) == pytest.approx(0.75)
        assert sched.factor(40) == pytest.approx(1.0)     # decay starts at full LR
        assert sched.factor(45) == pytest.approx(0.55)
        assert sched.factor(49) == pytest.approx(0.19)
        # Past the end of a closed schedule the LR stays at the floor.
        assert sched.factor(50) == pytest.approx(0.1)
        assert sched.factor(500) == pytest.approx(0.1)
        assert sched.end_step() == 50

    def test_training_schedule_is_open_and_warms_from_zero(self):
        sched = Schedule.training(warmup_steps=4)
        assert sched.factor(0) == pytest.approx(0.0)
        assert sched.factor(1) == pytest.approx(0.25)
        assert sched.factor(4) == pytest.approx(1.0)
        assert sched.factor(10_000) == pytest.approx(1.0)
        assert sched.end_step() is None
        assert sched.is_open

    def test_zero_warmup_starts_at_full_lr(self):
        sched = Schedule.training(warmup_steps=0)
        assert [s.kind for s in sched.segments] == ["stable"]
        assert sched.factor(0) == pytest.approx(1.0)

    def test_lr_min_factor_is_where_the_warmup_starts(self):
        sched = Schedule.training(warmup_steps=10, lr_min_factor=0.1)
        assert sched.factor(0) == pytest.approx(0.1)
        assert sched.factor(5) == pytest.approx(0.55)
        assert sched.factor(10) == pytest.approx(1.0)

    def test_cosine_decay_endpoints_and_midpoint(self):
        sched = Schedule.training(warmup_steps=0)
        sched.append_decay(at_step=0, decay_steps=100, min_factor=0.1, shape="cosine")
        assert sched.factor(0) == pytest.approx(1.0)
        assert sched.factor(50) == pytest.approx(0.55)      # cosine is symmetric
        assert sched.factor(100) == pytest.approx(0.1)
        # Cosine decays more slowly than linear early on and faster late.
        assert sched.factor(25) > 0.1 + 0.9 * 0.75
        assert sched.factor(75) < 0.1 + 0.9 * 0.25

    def test_negative_step_is_refused(self):
        with pytest.raises(ScheduleError):
            Schedule.training(4).factor(-1)


# ── appending ───────────────────────────────────────────────────────────────

class TestAppend:

    def test_rewarm_truncates_the_open_stable(self):
        sched = Schedule.training(warmup_steps=10)
        sched.append_rewarm(at_step=100, rewarm_steps=20, from_factor=0.3)
        kinds = [s.kind for s in sched.segments]
        assert kinds == ["warmup", "stable", "rewarm", "stable"]
        assert sched.segments[1].start == 10 and sched.segments[1].steps == 90
        assert sched.segments[2].start == 100 and sched.segments[2].steps == 20
        assert sched.segments[3].start == 120 and sched.segments[3].is_open
        assert sched.factor(99) == pytest.approx(1.0)
        assert sched.factor(100) == pytest.approx(0.3)
        assert sched.factor(110) == pytest.approx(0.65)
        assert sched.factor(120) == pytest.approx(1.0)
        assert sched.end_step() is None       # still horizon-free

    def test_rewarm_at_the_stable_start_drops_the_empty_segment(self):
        sched = Schedule.training(warmup_steps=10)
        sched.append_rewarm(at_step=10, rewarm_steps=5, from_factor=0.0)
        assert [s.kind for s in sched.segments] == ["warmup", "rewarm", "stable"]
        assert sched.factor(10) == pytest.approx(0.0)
        assert sched.factor(15) == pytest.approx(1.0)

    def test_decay_closes_the_schedule(self):
        sched = Schedule.training(warmup_steps=10)
        sched.append_decay(at_step=200, decay_steps=20, min_factor=0.1, shape="linear")
        assert [s.kind for s in sched.segments] == ["warmup", "stable", "decay"]
        assert sched.end_step() == 220
        assert not sched.is_open
        assert sched.factor(210) == pytest.approx(0.55)
        assert sched.factor(220) == pytest.approx(0.1)
        with pytest.raises(ScheduleError):
            sched.append_rewarm(at_step=230, rewarm_steps=5, from_factor=0.5)
        with pytest.raises(ScheduleError):
            sched.append_decay(at_step=230, decay_steps=5, min_factor=0.1)

    def test_appending_before_the_open_stable_is_refused(self):
        sched = Schedule.training(warmup_steps=10)
        # Step 5 is inside the warmup, which is already fixed history.
        with pytest.raises(ScheduleError):
            sched.append_rewarm(at_step=5, rewarm_steps=5, from_factor=0.5)
        with pytest.raises(ScheduleError):
            sched.append_decay(at_step=5, decay_steps=5, min_factor=0.1)
        # And a second append before the first one's segments is refused too.
        sched.append_rewarm(at_step=100, rewarm_steps=10, from_factor=0.5)
        with pytest.raises(ScheduleError):
            sched.append_rewarm(at_step=105, rewarm_steps=10, from_factor=0.5)
        sched.append_rewarm(at_step=110, rewarm_steps=10, from_factor=0.5)
        assert [s.kind for s in sched.segments] == [
            "warmup", "stable", "rewarm", "rewarm", "stable"]

    def test_non_positive_lengths_are_refused(self):
        with pytest.raises(ScheduleError):
            Schedule.training(4).append_rewarm(at_step=10, rewarm_steps=0, from_factor=0.5)
        with pytest.raises(ScheduleError):
            Schedule.training(4).append_decay(at_step=10, decay_steps=0, min_factor=0.1)
        with pytest.raises(ScheduleError):
            Schedule.training(4).append_decay(at_step=10, decay_steps=5,
                                              min_factor=0.1, shape="quadratic")

    def test_open_segment_must_be_last_and_stable(self):
        with pytest.raises(ScheduleError):
            Schedule([Segment("stable", 0, None, 1.0, 1.0),
                      Segment("decay", 10, 5, 1.0, 0.1)])
        with pytest.raises(ScheduleError):
            Segment("decay", 0, None, 1.0, 0.1)
        with pytest.raises(ScheduleError):   # not contiguous
            Schedule([Segment("warmup", 0, 10, 0.0, 1.0),
                      Segment("stable", 11, None, 1.0, 1.0)])


# ── the bias ratio hook ─────────────────────────────────────────────────────

class TestRatio:

    def test_default_ratio_is_a_no_op(self):
        sched = Schedule.training(warmup_steps=4)
        sched.append_decay(at_step=100, decay_steps=10, min_factor=0.1, shape="linear")
        for step in (0, 2, 4, 50, 100, 105, 110, 200):
            assert sched.bias_factor(step) == pytest.approx(sched.factor(step))

    def test_ratio_interpolates_within_a_segment(self):
        # A decay that also walks the bias:LoRA ratio from 10 down to 1 — the
        # trunk's "decay the ratio toward 1" hook, with no new mechanism.
        sched = Schedule([
            Segment("stable", 0, 10, 1.0, 1.0, ratio_start=10.0, ratio_end=10.0),
            Segment("decay", 10, 10, 1.0, 0.5, ratio_start=10.0, ratio_end=1.0,
                    shape="linear"),
        ])
        assert sched.bias_factor(0) == pytest.approx(10.0)
        assert sched.bias_factor(10) == pytest.approx(10.0)
        assert sched.bias_factor(15) == pytest.approx(0.75 * 5.5)
        assert sched.bias_factor(20) == pytest.approx(0.5 * 1.0)
        assert sched.bias_factor(999) == pytest.approx(0.5 * 1.0)
        # The LoRA factor is untouched by the ratio.
        assert sched.factor(15) == pytest.approx(0.75)

    def test_rewarm_carries_the_ratio_it_found(self):
        sched = Schedule([Segment("stable", 0, None, 1.0, 1.0,
                                  ratio_start=4.0, ratio_end=4.0)])
        sched.append_rewarm(at_step=50, rewarm_steps=10, from_factor=0.5, ratio_end=2.0)
        assert sched.bias_factor(50) == pytest.approx(0.5 * 4.0)
        assert sched.bias_factor(60) == pytest.approx(1.0 * 2.0)
        assert sched.bias_factor(1000) == pytest.approx(1.0 * 2.0)


# ── serialisation ───────────────────────────────────────────────────────────

class TestSerialisation:

    def test_position_and_factors_survive_a_round_trip(self):
        sched = Schedule.training(warmup_steps=10)
        sched.append_rewarm(at_step=100, rewarm_steps=20, from_factor=0.3)
        sched.append_decay(at_step=200, decay_steps=30, min_factor=0.05)

        import json
        restored = Schedule.from_json(json.loads(json.dumps(sched.to_json())))
        assert restored.segments == sched.segments
        for step in (0, 5, 10, 99, 100, 110, 120, 199, 200, 215, 230, 400):
            assert restored.position(step) == sched.position(step)
            assert restored.factor(step) == pytest.approx(sched.factor(step))
            assert restored.bias_factor(step) == pytest.approx(sched.bias_factor(step))
        assert restored.end_step() == sched.end_step()

    def test_position_is_segment_relative(self):
        sched = Schedule.training(warmup_steps=10)
        sched.append_rewarm(at_step=100, rewarm_steps=20, from_factor=0.3)
        assert sched.position(0) == (0, 0)
        assert sched.position(9) == (0, 9)
        assert sched.position(10) == (1, 0)
        assert sched.position(100) == (2, 0)
        assert sched.position(125) == (3, 5)

    def test_from_json_accepts_a_string_and_refuses_a_future_version(self):
        import json
        sched = Schedule.training(warmup_steps=3)
        assert Schedule.from_json(json.dumps(sched.to_json())).segments == sched.segments
        payload = sched.to_json()
        payload["version"] = 99
        with pytest.raises(ScheduleError):
            Schedule.from_json(payload)


# ── the LambdaLR that HF is handed ──────────────────────────────────────────

class TestLambdaLR:

    def test_both_groups_scale_their_own_base_lr(self):
        sched = Schedule.training(warmup_steps=4)
        sched.append_decay(at_step=10, decay_steps=4, min_factor=0.25, shape="linear")
        opt = _two_group_optimizer(lr=1e-3, bias_lr=1e-2)
        lr_sched = _scheduler(opt, sched)

        seen = []
        for step in range(16):
            seen.append((opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))
            lr_sched.step()

        for step, (lora_lr, bias_lr) in enumerate(seen):
            assert lora_lr == pytest.approx(1e-3 * sched.factor(step))
            assert bias_lr == pytest.approx(1e-2 * sched.bias_factor(step))
        assert seen[0] == pytest.approx((0.0, 0.0))
        assert seen[4] == pytest.approx((1e-3, 1e-2))
        assert seen[14] == pytest.approx((0.25e-3, 0.25e-2))

    def test_ratio_only_moves_the_bias_group(self):
        sched = Schedule([Segment("stable", 0, 10, 1.0, 1.0,
                                  ratio_start=1.0, ratio_end=0.5, shape="linear")])
        opt = _two_group_optimizer(lr=1e-3, bias_lr=1e-2)
        lr_sched = _scheduler(opt, sched)
        for _ in range(5):
            lr_sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
        assert opt.param_groups[1]["lr"] == pytest.approx(1e-2 * 0.75)

    def test_resume_needs_only_last_epoch(self):
        """A scheduler rebuilt from schedule.json + scheduler.pt continues where
        the killed process left off — which is the whole resume contract, since
        the lambdas are never serialised."""
        import json

        sched = Schedule.training(warmup_steps=6)
        opt = _two_group_optimizer()
        lr_sched = _scheduler(opt, sched)
        for _ in range(3):
            lr_sched.step()
        saved = copy.deepcopy(lr_sched.state_dict())
        # Nothing schedule-shaped is in there: the lambdas serialise as None.
        assert saved["lr_lambdas"] == [None, None]

        # Rebuild the way a resume does: segments from JSON, lambdas fresh.
        restored_schedule = Schedule.from_json(json.loads(json.dumps(sched.to_json())))
        opt2 = _two_group_optimizer()
        lr_sched2 = _scheduler(opt2, restored_schedule)
        lr_sched2.load_state_dict(saved)
        assert lr_sched2.last_epoch == lr_sched.last_epoch

        for _ in range(5):
            lr_sched.step()
            lr_sched2.step()
            assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
            assert opt2.param_groups[1]["lr"] == pytest.approx(opt.param_groups[1]["lr"])

        # And `last_epoch` alone carries it: a fresh scheduler given nothing but
        # that number lands on the same LRs.
        opt3 = _two_group_optimizer()
        lr_sched3 = _scheduler(opt3, Schedule.from_json(sched.to_json()))
        lr_sched3.load_state_dict({**lr_sched3.state_dict(),
                                   "last_epoch": saved["last_epoch"]})
        for _ in range(5):
            lr_sched3.step()
        assert opt3.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
        assert opt3.param_groups[1]["lr"] == pytest.approx(opt.param_groups[1]["lr"])
