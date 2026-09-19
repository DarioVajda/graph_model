"""
The step count a schedule is laid out against, and the anneal it silently skips.

`PLAN.md` §9's standing rule: every instrument gets a test that fails when the
instrument is wrong, **including one that reproduces the defect**.

The defect, found 2026-09-18 while checking the batch arithmetic for a four-card
cell: `steps_per_epoch` divided by `batch_size` and `accumulation_steps` but not
by `world_size`. One optimizer step consumes the product of all three, so at two
ranks the Trainer runs half the predicted steps. Nothing about the run announces
this — the loss curve is fine and the checkpoint is real — but a WSD schedule
sized to the prediction ends mid-stable and never reaches its decay segment.

041 ran nine epochs of ChEBI-20 that way and reported a model still sitting at
peak LR. It cost a whole sweep and, worse, it was read as evidence in a schedule
comparison: cosine beat "WSD" by 0.023 BLEU-2, which is the sort of margin an
anneal buys on its own.
"""

import pytest

from src.experiments.molecules.train import _schedule_args, steps_per_epoch_for


class _Cfg:
    """Minimal stand-in: `_schedule_args` reads only these."""

    def __init__(self, lr_schedule="wsd", lr=1e-4, wsd_decay_fraction=0.1):
        self.lr_schedule = lr_schedule
        self.lr = lr
        self.wsd_decay_fraction = wsd_decay_fraction


# 041's real shape: 26,071 ChEBI training molecules, batch 4, accumulation 8.
N_EXAMPLES, BATCH, ACCUM = 26071, 4, 8


def test_two_ranks_halve_the_steps_in_an_epoch():
    single = steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM, world_size=1)
    double = steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM, world_size=2)

    assert single == 814
    assert double == 407, "one step consumes batch * accum * world_size examples"


def test_the_world_size_defaults_to_the_torchrun_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM) == 203


def test_an_absent_world_size_means_one_rank(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    assert steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM) == 814


def test_THE_DEFECT_a_world_size_blind_count_strands_wsd_before_its_decay():
    """Reproduced: lay 041's schedule out against the single-rank count.

    The segments are computed for 7,326 steps, the run executes 3,663, and the
    decay starts at 6,593 — 2,930 steps after the run is over.
    """
    wrong = steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM, world_size=1)
    args = _schedule_args(_Cfg(), wrong, wrong * 9)
    kwargs = args["lr_scheduler_kwargs"]

    decay_begins = args["warmup_steps"] + kwargs["num_stable_steps"]
    actual_last_step = steps_per_epoch_for(
        N_EXAMPLES, BATCH, ACCUM, world_size=2) * 9

    assert actual_last_step == 3663
    assert decay_begins == 6593
    assert decay_begins > actual_last_step, (
        "the run would end at peak LR, having never entered the decay segment")


def test_the_corrected_count_puts_the_decay_inside_the_run():
    right = steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM, world_size=2)
    total = right * 9
    args = _schedule_args(_Cfg(), right, total)
    kwargs = args["lr_scheduler_kwargs"]

    decay_begins = args["warmup_steps"] + kwargs["num_stable_steps"]

    assert decay_begins < total, "the anneal has to happen before the run ends"
    assert kwargs["num_decay_steps"] == pytest.approx(total * 0.1, abs=1)


def test_wsd_segments_sum_to_the_run_length():
    """HF runs the remainder at the floor if they do not, which is silent."""
    right = steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM, world_size=2)
    total = right * 9
    args = _schedule_args(_Cfg(), right, total)
    kwargs = args["lr_scheduler_kwargs"]

    assert (args["warmup_steps"] + kwargs["num_stable_steps"]
            + kwargs["num_decay_steps"]) == total


def test_cosine_warmup_is_one_real_epoch_not_two():
    """The same bug's milder form: 042 warmed up for two epochs, not one.

    Cosine survives a wrong `steps_per_epoch` because HF derives the decay curve
    from its own step count, so only the warmup length is affected. That made it
    a deviation rather than a defect — but it is still not the recipe the config
    describes.
    """
    right = steps_per_epoch_for(N_EXAMPLES, BATCH, ACCUM, world_size=2)
    args = _schedule_args(_Cfg(lr_schedule="cosine", lr=2e-4), right, right * 12)

    assert args["warmup_steps"] == 407
    assert args["lr_scheduler_kwargs"]["min_lr"] == pytest.approx(2e-5)
