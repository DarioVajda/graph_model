"""
The step count the KGQA warmup is laid out against, under torchrun.

Same defect as the molecules Tier-C sweeps (see
`tests/experiments/molecules/test_schedule_steps.py`): `steps_per_epoch` divided
by `batch_size` and `accumulation_steps` but not by `world_size`. One optimizer
step consumes the product of all three.

KGQA gets off more lightly than molecules did, because it runs
`cosine_with_min_lr` and HF derives the decay curve from its own step count —
only `warmup_steps = total_steps // 10` reads the prediction. But "more lightly"
is not "harmlessly": the CWQ headline is 23,440 steps over four ranks, and the
uncorrected count would put 9,376 of them in warmup, so the run would reach peak
LR two-fifths of the way through and decay from there. At eight ranks the same
blindness buries four-fifths of the run in warmup.

Every KGQA run before 2026-09-19 is single-rank, so the fix moves no number on
record — the first test pins that.
"""

import pytest

from src.experiments.kgqa.train import steps_per_epoch_for

# 022/042's real shape: 23,441 CWQ training questions at cap1024/ver1.
# 022 ran one rank at batch 2 x accumulation 4; 042 runs four at 1 x 2.
N_EXAMPLES = 23441
SINGLE_RANK = dict(batch_size=2, accumulation_steps=4)
FOUR_RANK = dict(batch_size=1, accumulation_steps=2)
EIGHT_RANK = dict(batch_size=1, accumulation_steps=1)


def test_the_single_rank_history_is_untouched():
    assert steps_per_epoch_for(N_EXAMPLES, **SINGLE_RANK, world_size=1) == 2930


def test_every_shape_with_an_effective_batch_of_eight_runs_the_same_steps():
    """The premise of the DDP port: same effective batch, same schedule.

    1 x 2 x 4, 4 x 1 x 2 and 8 x 1 x 1 all consume 8 examples per optimizer
    step, so the ported recipe has to produce the identical step count —
    otherwise it is a different run wearing 022's config.
    """
    single = steps_per_epoch_for(N_EXAMPLES, **SINGLE_RANK, world_size=1)
    four = steps_per_epoch_for(N_EXAMPLES, **FOUR_RANK, world_size=4)
    eight = steps_per_epoch_for(N_EXAMPLES, **EIGHT_RANK, world_size=8)

    assert four == eight == single == 2930


def test_the_world_size_defaults_to_the_torchrun_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert steps_per_epoch_for(N_EXAMPLES, **EIGHT_RANK) == 2930


def test_an_absent_world_size_means_one_rank(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    assert steps_per_epoch_for(N_EXAMPLES, **EIGHT_RANK) == 23441


@pytest.mark.parametrize("shape,world,blind_warmup,fraction", [
    (FOUR_RANK, 4, 9376, 0.40),
    (EIGHT_RANK, 8, 18752, 0.80),
])
def test_THE_DEFECT_a_world_size_blind_count_over_long_warmup(
        shape, world, blind_warmup, fraction):
    """Reproduced: lay the 8-epoch headline's warmup out against the blind count."""
    blind = steps_per_epoch_for(N_EXAMPLES, **shape, world_size=1)
    warmup = (blind * 8) // 10

    actual_total = steps_per_epoch_for(N_EXAMPLES, **shape, world_size=world) * 8

    assert actual_total == 23440
    assert warmup == blind_warmup
    assert warmup / actual_total == pytest.approx(fraction, abs=0.01)


@pytest.mark.parametrize("shape,world", [(FOUR_RANK, 4), (EIGHT_RANK, 8)])
def test_the_corrected_count_reproduces_022s_warmup_exactly(shape, world):
    right = steps_per_epoch_for(N_EXAMPLES, **shape, world_size=world)

    assert (right * 8) // 10 == 2344
