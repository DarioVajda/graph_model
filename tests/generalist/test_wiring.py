"""T9 (wiring half) — a measurement never loses a run.

`run_validators` catches what a *validator* raises and reports it as a status.
Everything around it — building the context, logging the metrics, printing the
statuses — sits outside that guard, and this is the file that says the guard has
to extend over the whole firing.

The bug this pins down is the reason it exists: the line that reported a
validator error read a field name that :class:`ValidatorStatus` does not have,
so the first validator failure of the first smoke run raised an
``AttributeError`` inside ``on_step_end`` and killed the training run — before
the checkpoint that step would have written. Every validator had behaved
correctly; the reporting killed the run.

No model and no GPU: the callback is exercised against a stub evaluation, which
is the only way to make the failure the test is about the *only* thing happening.
"""

from types import SimpleNamespace

import pytest

from src.generalist import wiring
from src.generalist.evaluate import EvalRun, ValidatorStatus


def _run(logged):
    """A `Run`-shaped stub: the callback reads `config` and `trainer` only."""
    trainer = SimpleNamespace(log=lambda metrics: logged.append(("log", metrics)))
    config = SimpleNamespace(milestone_steps=0)
    return SimpleNamespace(config=config, trainer=trainer)


def _result(statuses, metrics=None):
    return EvalRun(step=50, event="step", metrics=metrics or {},
                   statuses=tuple(statuses), versions=())


ERROR_STATUS = ValidatorStatus(
    name="grad_share", protocol_version="1", state="error",
    message="a compiled backward refuses retain_graph=True")


def test_a_validator_error_is_reported_and_training_continues(monkeypatch):
    lines = []
    logged = []
    run = _run(logged)
    monkeypatch.setattr(wiring, "run_evaluation",
                        lambda *a, **k: _result([ERROR_STATUS], {"bias_norm/l2": 1.0}))

    callback = wiring.validator_callback(run, log=lines.append)
    control = SimpleNamespace()
    assert callback.on_step_end(None, SimpleNamespace(global_step=50), control) is control

    assert logged == [("log", {"bias_norm/l2": 1.0})], (
        "the validators that did run must still be logged")
    assert any("grad_share" in line and ERROR_STATUS.message in line
               for line in lines), lines


def test_an_evaluation_that_fails_outside_a_validator_does_not_stop_training(
        monkeypatch, capsys):
    """The exact shape of the smoke run's step-50 death."""
    lines = []
    run = _run([])

    def boom(*args, **kwargs):
        raise AttributeError("'ValidatorStatus' object has no attribute 'detail'")

    monkeypatch.setattr(wiring, "run_evaluation", boom)
    callback = wiring.validator_callback(run, log=lines.append)
    control = SimpleNamespace()
    assert callback.on_step_end(None, SimpleNamespace(global_step=50), control) is control
    assert any("failed outside a validator" in line for line in lines), lines
    # The traceback goes to stderr rather than into the log callable, so a run
    # that survives one of these still says exactly what happened.
    assert "AttributeError" in capsys.readouterr().err


def test_the_end_of_training_is_reported_the_same_way(monkeypatch):
    lines = []
    run = _run([])
    monkeypatch.setattr(wiring, "run_evaluation",
                        lambda *a, **k: _result([ERROR_STATUS]))
    callback = wiring.validator_callback(run, log=lines.append)
    control = SimpleNamespace()
    assert callback.on_train_end(None, SimpleNamespace(global_step=200),
                                 control) is control
    assert any("grad_share" in line for line in lines), lines


def test_step_zero_fires_nothing(monkeypatch):
    """`on_step_end` at step 0 would measure a model that has not trained."""
    fired = []
    monkeypatch.setattr(wiring, "run_evaluation",
                        lambda *a, **k: fired.append(1) or _result([]))
    callback = wiring.validator_callback(_run([]), log=lambda _line: None)
    callback.on_step_end(None, SimpleNamespace(global_step=0), SimpleNamespace())
    assert fired == []


def test_the_milestone_set_fires_on_its_own_cadence(monkeypatch):
    events = []
    run = _run([])
    run.config.milestone_steps = 100
    monkeypatch.setattr(wiring, "run_evaluation",
                        lambda _run, _step, event="step", **k:
                        events.append(event) or _result([]))
    callback = wiring.validator_callback(run, log=lambda _line: None)
    for step in (50, 100):
        callback.on_step_end(None, SimpleNamespace(global_step=step),
                             SimpleNamespace())
    assert events == ["step", "step", "milestone"]


# ─────────────────────────────────────────────────────────────────────────────
# get_source
# ─────────────────────────────────────────────────────────────────────────────

def _registry():
    from src.generalist.registry import Registry, TaskSpec

    return Registry([
        TaskSpec(name="mol/bace", domain="molecules", adapter="molecules",
                 kind="corpus", passes=3, train_size=1178, mean_tokens=100.0),
        TaskSpec(name="mol/g2s", domain="molecules", adapter="molecules",
                 kind="generator", cap_per_pass=500, train_size=500,
                 mean_tokens=100.0),
        TaskSpec(name="mol/longest_chain", domain="molecules", adapter="molecules",
                 kind="generator", held_out=True, cap_per_pass=200,
                 train_size=200, mean_tokens=100.0),
    ])


def test_only_a_generators_passes_are_read_off_disk(monkeypatch):
    """The failure that ended the first smoke run, 122 steps in.

    A corpus is one built artifact that the sampler re-permutes per pass (D4.2);
    only a generator draws fresh examples, and only its passes are materialised.
    Asking the adapter for pass 2 of a corpus asks for a file no ``data_prep``
    ever writes, and the run dies at the first pass boundary — after the budget
    resolved cleanly and `validate` printed the three passes it would take.
    """
    from src.generalist.adapters import molecules

    asked = []
    monkeypatch.setattr(molecules, "load",
                        lambda task, split, arm, pass_id=0, config=None:
                        asked.append((task, split, pass_id)) or "src")

    config = SimpleNamespace(arm="graph")
    get_source = wiring.make_get_source(config, _registry(), adapter_config=None)
    for pass_id in (0, 1, 2):
        get_source("mol/bace", pass_id)
        get_source("mol/g2s", pass_id)

    assert [p for task, _split, p in asked if task == "mol/bace"] == [0, 0, 0]
    assert [p for task, _split, p in asked if task == "mol/g2s"] == [0, 1, 2]
    assert {split for _task, split, _p in asked} == {"train"}


def test_a_held_out_task_is_loaded_off_its_held_out_split(monkeypatch):
    """An `adapt` fork trains the one task that has no train split."""
    from src.generalist.adapters import molecules

    asked = []
    monkeypatch.setattr(molecules, "load",
                        lambda task, split, arm, pass_id=0, config=None:
                        asked.append((task, split, pass_id)) or "src")

    get_source = wiring.make_get_source(SimpleNamespace(arm="graph"), _registry(),
                                        adapter_config=None)
    get_source("mol/longest_chain", 0)
    assert asked == [("mol/longest_chain", "held_out", 0)]


def test_a_held_out_generator_still_reads_its_one_built_pass(monkeypatch):
    """`build` writes extra passes for a generator's *train* split only.

    A held-out task has no train split — `splits_for` gives it ``held_out`` and
    nothing else — so `bond_path` is a generator with exactly one artifact on
    disk. The `adapt` fork is the only thing that trains on it, and it exhausts
    those 200 examples in single-figure steps; passing the pass id through would
    then ask for pass 1 of a file nothing will ever build. Same failure as the
    corpus one above, one axis over.
    """
    from src.generalist.adapters import molecules

    asked = []
    monkeypatch.setattr(molecules, "load",
                        lambda task, split, arm, pass_id=0, config=None:
                        asked.append((task, split, pass_id)) or "src")

    get_source = wiring.make_get_source(SimpleNamespace(arm="graph"), _registry(),
                                        adapter_config=None)
    for pass_id in (0, 1, 2, 7):
        get_source("mol/longest_chain", pass_id)
    assert [p for _task, _split, p in asked] == [0, 0, 0, 0]


@pytest.mark.parametrize("has_trainer", (True, False))
def test_validator_config_carries_the_grad_share_closure_only_with_a_trainer(
        has_trainer):
    """`validate` mode builds no trainer, so the key must be absent, not None."""
    from src.generalist.evaluate import builtin

    trainer = (SimpleNamespace(per_task_loss_fn=lambda: (lambda task: None),
                               per_task_batch_counts=dict)
               if has_trainer else None)
    config = SimpleNamespace(max_spd=32, to_dict=lambda: {})
    out = wiring.validator_config(SimpleNamespace(config=config, trainer=trainer))
    assert (builtin.GRAD_SHARE_LOSS_FN in out) is has_trainer
    assert (builtin.GRAD_SHARE_COUNTS_FN in out) is has_trainer


def test_a_forks_validators_run_on_the_device_its_model_is_on(monkeypatch):
    """`fork` mode has no model to ask until the factory has built one.

    So it passes no device, and everything that scores an eval set put its index
    tensors on the CPU while the model answered from CUDA: the anneal fork
    trained its 21 steps and then lost `in_mixture`, `held_out`, `perm_spread`
    and `per_example` to a device mismatch — every measurement an anneal exists
    to produce. The device the leg's run resolved is the one to use.
    """
    seen = {}
    built_run = SimpleNamespace(tokenizer="tok", device="cuda:0",
                                trainer=SimpleNamespace())
    monkeypatch.setattr(wiring, "build_run", lambda *a, **k: built_run)
    monkeypatch.setattr(wiring, "run_evaluation",
                        lambda run, step, **k: seen.update(device=run.device)
                        or _result([]))

    config = SimpleNamespace(seed=0, milestone_steps=0)
    factory, validate = wiring.fork_callables(
        config, registry=None, adapter_config=None, validators=(), eval_sets={})
    leg = SimpleNamespace(name="anneal", output_dir="", mixture=None,
                          schedule=None, seed=0, max_steps=21)
    factory(leg, None)

    trainer = SimpleNamespace(args=SimpleNamespace(output_dir="", max_steps=21))
    validate(SimpleNamespace(trainer=trainer, model=None, validators=(),
                             stage="end", step=221, scratch_dir="",
                             schedule_position=None))
    assert seen["device"] == "cuda:0"
