"""T4 (readout half) — the per-task loss closure `grad_share` backwards through.

`measure_grad_share` is tested on a stub in ``test_mixture.py`` and the validator
is tested against an injected closure in ``test_validators.py``. Neither of those
touches the one piece that has to be right for the readout to mean anything on a
real run: the closure the *trainer* hands over. It is the join between the two,
and the failure it guards against is silent — a closure that returned a detached
tensor, or one that normalised over the micro-batch instead of the step, still
produces a plausible table of shares.

Four claims:

* **The closure is differentiable and reaches the trainable parameters.** A
  detached scalar would make ``torch.autograd.grad`` raise; a scalar attached to a
  graph that misses the LoRA and bias tensors would come back all-``None`` and be
  reported as a zero share.
* **The tasks partition the step.** Summing the closure over every task and every
  micro-batch reproduces the loss the mixture computes for that step — same rows,
  same divisor. A closure that dropped rows, measured only the last micro-batch,
  or divided by its own row count would fail here and nowhere else.
* **A task with no rows returns ``None``**, and so does a task the sampler has
  never heard of. ``measure_grad_share`` skips those; returning a zero instead
  would report "this task contributes nothing to the gradient" for a task that
  simply was not sampled.
* **The wiring installs it.** ``validator_config`` carries the key when there is a
  trainer and leaves it out when there is not.

CPU, against the tiny three-task run in ``tiny_run.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from src.generalist.evaluate import builtin
from src.generalist.trainer import TrainerError
from src.generalist.wiring import validator_config
from tests.generalist.tiny_run import TASKS, build_trainer


@pytest.fixture
def trained(tmp_path):
    """A trainer that has run two optimizer steps, so it holds a whole step.

    Accumulation is 3 rather than 1: the readout is over the step's micro-batches
    and a run whose step *is* one micro-batch cannot tell the two apart.
    """
    trainer, model, sampler, _schedule = build_trainer(
        str(tmp_path), max_steps=2, accumulation_steps=3)
    trainer.train()
    return trainer


def _summed_loss(trainer):
    """What ``MixtureLoss`` makes of the stored step, micro-batch by micro-batch."""
    total = None
    for batch in trainer._step_batches:
        token_losses, mask, _ = trainer._token_losses(
            trainer.model, batch["inputs"], batch["labels"])
        loss, _per_task = trainer.mixture_loss(token_losses, mask,
                                               batch["task_ids"],
                                               batch["examples_in_step"])
        total = loss if total is None else total + loss
    return total


def test_the_closure_is_differentiable_to_the_trainable_parameters(trained):
    fn = trained.per_task_loss_fn()
    params = [p for p in trained.model.parameters() if p.requires_grad]
    assert params, "the tiny model should have LoRA and bias parameters"

    losses = {task: fn(task) for task in TASKS}
    present = {task: value for task, value in losses.items() if value is not None}
    assert present, "a step of eight examples over three tasks holds some"

    task, stream = next(iter(present.items()))
    loss = next(iter(stream))
    assert loss.requires_grad, f"{task}: the closure returned a detached scalar"
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    moved = max(float(g.abs().max()) for g in grads if g is not None)
    assert moved > 0.0, "the gradient reached no trainable parameter"


def test_the_tasks_partition_the_whole_step(trained):
    """Every row of every micro-batch, once, under the step's own divisor."""
    fn = trained.per_task_loss_fn()
    assert len(trained._step_batches) > 1, (
        "this run should accumulate, or the step and the micro-batch are the "
        "same thing and the test proves nothing")
    total = None
    for task in TASKS:
        stream = fn(task)
        if stream is None:
            continue
        for part in stream:
            total = part if total is None else total + part

    torch.testing.assert_close(total, _summed_loss(trained),
                               rtol=1e-5, atol=1e-6)


def test_a_task_the_step_did_not_sample_returns_none(tmp_path):
    """A task at a millionth of the weight draws no examples in eight.

    A zero weight is refused outright — `registry.resolve` will not resolve a
    mixture containing a task it would never sample — so "absent from this step"
    is produced the way a real run produces it: a share small enough that the
    step's counts round it away.
    """
    from tests.generalist.tiny_run import build_mixture

    absent = TASKS[-1]
    weights = {name: (1e-6 if name == absent else 1.0) for name in TASKS}
    registry, mixture = build_mixture(weights=weights)
    trainer, _model, _sampler, _schedule = build_trainer(
        str(tmp_path), max_steps=2, accumulation_steps=3,
        registry=registry, mixture=mixture)
    trainer.train()

    fn = trainer.per_task_loss_fn()
    assert fn(absent) is None
    assert [task for task in TASKS if fn(task) is not None] == list(TASKS[:-1])


def test_an_unknown_task_returns_none_rather_than_raising(trained):
    assert trained.per_task_loss_fn()("t/nonexistent") is None


def test_the_counts_describe_the_step_that_was_measured(trained):
    """A share measured on four examples and one on forty are different claims."""
    counts = trained.per_task_batch_counts()
    rows = sum(int(b["task_ids"].shape[0]) for b in trained._step_batches)
    assert set(counts) <= set(TASKS)
    assert sum(counts.values()) == rows
    # The step's own example count, which is what makes the reported share a
    # share of the step rather than of whatever the last micro-batch held.
    assert rows == trained._step_batches[0]["examples_in_step"]

    fn = trained.per_task_loss_fn()
    for task in TASKS:
        assert (counts.get(task, 0) > 0) == (fn(task) is not None)


def test_the_counts_are_empty_before_any_micro_batch_has_run(tmp_path):
    trainer, _model, _sampler, _schedule = build_trainer(str(tmp_path), max_steps=2)
    assert trainer.per_task_batch_counts() == {}


def test_the_closure_refuses_before_any_micro_batch_has_run(tmp_path):
    trainer, _model, _sampler, _schedule = build_trainer(str(tmp_path), max_steps=2)
    with pytest.raises(TrainerError, match="grad_share fired before"):
        trainer.per_task_loss_fn()(TASKS[0])


def test_only_the_step_in_progress_is_kept(trained):
    """The previous step's micro-batches are dropped, not accumulated.

    Two steps ran; if the batches piled up, the readout would attribute a
    gradient to examples the model has already moved past, and the memory the
    docstring calls "a few hundred kilobytes" would grow with the run.
    """
    assert len(trained._step_batches) == trained.sampler.accumulation_steps
    assert trained._step_batches_step == 1          # zero-based: the second step


def test_each_forward_is_made_only_when_it_is_about_to_be_used(trained, monkeypatch):
    """One forward per (task, micro-batch) pair, produced lazily.

    ``measure_grad_share`` backwards without retaining a graph, because a compiled
    backward with donated buffers refuses ``retain_graph=True`` — so a closure
    that built the step's graphs eagerly and summed them would work in this CPU
    test and fail at step 50 of a real run, which is exactly how it was found.
    The count is the assertion that they are built one at a time: asking for the
    generator must run no forward at all.
    """
    calls = []
    original = trained._token_losses

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(trained, "_token_losses", counted)
    fn = trained.per_task_loss_fn()
    streams = {task: fn(task) for task in TASKS}
    assert calls == [], "the generator ran a forward before it was consumed"

    expected = 0
    for task, stream in streams.items():
        if stream is None:
            continue
        for _loss in stream:
            expected += 1
    assert len(calls) == expected
    assert expected == sum(
        1 for task in TASKS for b in trained._step_batches
        if bool((b["task_ids"] == trained.sampler.task_ids[task]).any()))

    assert trained.per_task_loss_fn()("t/nonexistent") is None
    assert len(calls) == expected


def test_the_readout_backwards_without_retaining_a_graph(trained):
    """What `measure_grad_share` actually does, against the real closure.

    The failure this pins down is not visible in a single ``autograd.grad``
    call: it is the *second* task's backward that raises, once on a shared graph
    that the first backward has already freed.
    """
    from src.generalist.mixture import measure_grad_share

    shares = measure_grad_share(trained.model, trained.per_task_loss_fn(),
                                tasks=list(TASKS))
    assert shares, "no task produced a gradient"
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Wiring
# ─────────────────────────────────────────────────────────────────────────────

def _stub_run(trainer):
    config = SimpleNamespace(max_spd=8, to_dict=lambda: {})
    return SimpleNamespace(config=config, trainer=trainer)


def test_validator_config_installs_the_closure_when_there_is_a_trainer(trained):
    config = validator_config(_stub_run(trained))
    assert callable(config[builtin.GRAD_SHARE_LOSS_FN])


def test_validator_config_leaves_the_key_out_without_a_trainer():
    # `validate` mode resolves a whole config with no model and no trainer. The
    # validator's own refusal is then the right behaviour; a key holding `None`
    # would read as an installed closure.
    assert builtin.GRAD_SHARE_LOSS_FN not in validator_config(_stub_run(None))
