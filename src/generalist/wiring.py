"""
D8 — the wiring: a ``RunConfig`` becomes a registry, a mixture, a trainer and a
set of evaluations.

``__main__.py`` is a dispatcher and this is what it dispatches to. Everything
here is a function of a :class:`~src.generalist.config.RunConfig`, which is what
makes the six modes six lines each and makes a fork's legs identical to a
training run by construction rather than by care.

Three seams meet in this file, and they meet here because none of the modules
they belong to is allowed to import the others:

* **D6 needs two callables.** ``fork.py`` never builds a model and never imports
  ``evaluate/`` — that is what lets a fork be planned on a login node. So it asks
  for ``trainer_factory(leg, plan)`` and ``validate(request)``, and
  :func:`fork_callables` is both.
* **D7 needs an ``EvalContext``.** A ``ValidationRequest`` carries the trainer
  and which stage of which leg is asking; an ``EvalContext`` carries the model,
  the eval sets and the registry. :func:`validation_hook` is the adapter between
  them, and it is one function.
* **The validator list is built once per run.** ``throughput`` measures wall
  clock *between* firings, so a list rebuilt at each firing would report the time
  since it was constructed. :func:`build_run` takes ``validators=`` for exactly
  this reason: a fork builds the list once and hands the same tuple to every leg.

Nothing here is imported at module scope that a login node cannot afford: torch,
transformers and the adapter all arrive inside the functions that need them, so
``validate`` mode imports this file and resolves a whole config without either.
"""

from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass, field

from .config import ACTIVE_PARAMS, RunConfig
from .registry import Registry, is_held_out, resolve

#: Splits a validator reads without declaring them in a ``splits`` attribute.
#: These mirror the ``option("split", "test")`` defaults in `evaluate/builtin.py`;
#: a validator that takes an explicit ``split`` option overrides them.
DEFAULT_VALIDATOR_SPLIT = {"perm_spread": "test", "per_example": "test"}

#: Splits that can be loaded for an in-mixture task.
MIXTURE_SPLITS = ("val", "test")


class WiringError(RuntimeError):
    """A run that cannot be assembled. The message names what is missing."""


# ─────────────────────────────────────────────────────────────────────────────
# Registry and mixture — the torch-free half
# ─────────────────────────────────────────────────────────────────────────────

def build_registry(config: RunConfig, adapter_config=None):
    """``(registry, adapter_config)`` for this run's arm.

    Every ``mol/`` task is registered, not just the ones in the mixture: the
    ``held_out`` validator picks its tasks up from the registry (D7.3), and a
    fork's ``adapt`` mode resolves a held-out task's spec from it.
    """
    from .adapters import molecules

    adapter_config = adapter_config or config.adapter_config()
    adapter_config.validate()
    registry = Registry()
    molecules.register_molecule_tasks(registry, adapter_config, arm=config.arm)
    return registry, adapter_config


def unbuilt_tasks(registry: Registry, config: RunConfig) -> list:
    """Mixture tasks whose spec has no measured size yet, i.e. no ``data_prep``.

    ``mean_tokens`` and ``train_size`` are properties of the built data (D2), so
    a mixture cannot be resolved before the build. This names the tasks rather
    than letting ``registry.resolve`` raise on whichever one it reached first.
    """
    missing = []
    for entry in config.mixture_entries():
        spec = registry.get(entry["name"])
        if spec.mean_tokens is None or (spec.kind == "corpus"
                                        and spec.train_size is None):
            missing.append(spec.name)
    return missing


def resolve_mixture(config: RunConfig, registry: Registry, *, steps=None,
                    allow_held_out=()):
    """The config's task list as a resolved :class:`Mixture`.

    ``steps`` defaults to ``config.max_steps`` (0 meaning "let the finite
    sources' pass caps set the budget", `MOLECULE_GENERALIST.md` §2).
    """
    steps = steps if steps is not None else (config.max_steps or None)
    return resolve(registry, config.mixture_entries(),
                   tokens_per_step=config.tokens_per_step, steps=steps,
                   min_examples_per=config.min_examples_per,
                   allow_held_out=allow_held_out)


def passes_needed(mixture, registry: Registry) -> dict:
    """``{task: passes the run will consume}``.

    A generator draws ``cap_per_pass`` fresh examples per pass (D4.2), a corpus
    walks ``train_size`` per pass. ``data_prep`` materialises generator passes
    ahead of time — ``load`` never generates — so a run that needs seven passes
    of graph-to-SMILES and has three built stops mid-run with a build error.
    Computing it here is what lets ``validate`` say so first.
    """
    out = {}
    for entry in mixture.entries:
        spec = registry.get(entry.name)
        per_pass = entry.cap_per_pass if spec.kind == "generator" else spec.train_size
        if not per_pass:
            continue
        out[entry.name] = max(1, int(math.ceil(entry.examples / float(per_pass))))
    return out


def generator_passes(config: RunConfig, mixture=None, registry: Registry = None) -> int:
    """How many generator passes ``data_prep`` should build.

    The configured value wins; 0 means "as many as the resolved mixture will
    consume", which needs a resolved mixture and therefore a previous build.
    With neither, one pass — enough to make the *next* resolve possible.
    """
    if config.generator_passes:
        return int(config.generator_passes)
    if mixture is None or registry is None:
        return 1
    needed = [n for name, n in passes_needed(mixture, registry).items()
              if registry.get(name).kind == "generator"]
    return max(needed) if needed else 1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation sets
# ─────────────────────────────────────────────────────────────────────────────

def splits_wanted(validators) -> set:
    """Which splits the configured validators will ask for.

    Read off the validators rather than fixed here, so a config that drops
    ``held_out`` does not pay to load the held-out sets, and one that adds a
    validator scoring ``val`` gets them without this function being edited.
    """
    out = set()
    for validator in validators:
        out |= set(getattr(validator, "splits", ()) or ())
        chosen = (getattr(validator, "options", None) or {}).get("split")
        if chosen:
            out.add(str(chosen))
        elif validator.name in DEFAULT_VALIDATOR_SPLIT:
            out.add(DEFAULT_VALIDATOR_SPLIT[validator.name])
    return out


def build_eval_sets(config: RunConfig, registry: Registry, mixture, splits,
                    adapter_config=None, log=print) -> dict:
    """``{task: {split: TaskSource}}`` for the splits the validators want.

    In-mixture tasks contribute the splits their spec's ``eval_splits`` allows;
    held-out tasks contribute ``held_out`` and are taken from the *registry*, not
    the mixture, because by construction they are not in it.

    A source that was never built is logged and skipped rather than raised. An
    evaluation set is a measurement, and the rule the whole of D7 is built on is
    that a measurement never loses a run that has already cost GPU-hours; the
    validator then simply has nothing to score for that task and says so.
    """
    from .adapters import molecules

    adapter_config = adapter_config or config.adapter_config()
    splits = set(splits or ())
    out: dict = {}

    def add(task, split):
        try:
            source = molecules.load(task, split, config.arm, config=adapter_config)
        except Exception as exc:                                   # noqa: BLE001
            log(f"[eval-sets] {task}/{split}: not loaded ({type(exc).__name__}: "
                f"{exc}); the validators that wanted it will report nothing for it")
            return
        out.setdefault(task, {})[split] = source

    for entry in mixture.entries if mixture is not None else ():
        spec = registry.get(entry.name)
        for split in MIXTURE_SPLITS:
            if split in splits and split in spec.eval_splits:
                add(entry.name, split)

    if "held_out" in splits:
        for spec in registry:
            if is_held_out(spec):
                add(spec.name, "held_out")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The assembled run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    """One assembled run: everything a mode needs, built once and shared."""

    config: RunConfig
    adapter_config: object
    registry: Registry
    mixture: object
    sampler: object
    schedule: object
    model: object
    tokenizer: object
    collator: object
    trainer: object
    validators: tuple
    eval_sets: dict
    device: object
    output_dir: str
    max_steps: int
    seed: int
    lineage: object = None
    extras: dict = field(default_factory=dict)


def make_get_source(config: RunConfig, registry: Registry, adapter_config):
    """``(task, pass_id) -> TaskSource`` for the mixture sampler.

    The split is decided from the registry rather than hard-coded to ``train``:
    an ``adapt`` fork (D6) trains on one held-out task, whose only split is
    ``held_out``, and asking for its train split would be refused by the adapter
    — correctly, and unhelpfully, since that fork is the sanctioned exception.

    **Only a generator's passes are materialised.** D4.2 splits the two cases: a
    generator draws fresh examples per pass, so each pass is its own artifact and
    ``data_prep`` builds as many as the resolved mixture will consume; a corpus is
    one finite set of examples that the sampler re-permutes per pass, so every
    pass of it reads the same artifact. Passing the pass id through for a corpus
    too asks for an artifact nothing ever builds, and the run dies at the first
    pass boundary — 122 steps into the first smoke run, on the pass-2 of a corpus
    that `validate` had printed as needing three.

    **A held-out task is single-artifact whatever its kind.** ``build`` draws
    extra passes only for a generator's ``train`` split, and a held-out task has
    no ``train`` split to draw them for — `splits_for` gives it ``held_out`` and
    nothing else. So ``bond_path`` is a generator with exactly one pass on disk,
    and an `adapt` fork, which is the only thing that trains on it, exhausts
    those 200 examples in nine steps and then asks for pass 1 of an artifact
    nothing will ever build. It is the same failure as the corpus one above, one
    axis over, and the same rule fixes it: the pass id is only a filename for a
    source that has more than one file.
    """
    from .adapters import molecules

    def get_source(task: str, pass_id: int):
        spec = registry.get(task)
        held_out = is_held_out(spec)
        split = "held_out" if held_out else "train"
        built = int(pass_id) if (spec.kind == "generator" and not held_out) else 0
        return molecules.load(task, split, config.arm, pass_id=built,
                              config=adapter_config)

    return get_source


def build_run(config: RunConfig, *, output_dir=None, mixture=None, schedule=None,
              seed=None, max_steps=None, registry=None, adapter_config=None,
              validators=None, eval_sets=None, lineage=None,
              force_final_save: bool = False, fire_validators: bool = True,
              callbacks=(), log=print) -> Run:
    """Model, data, sampler, schedule, trainer — one run, assembled.

    Used by ``train``, ``resume``, ``eval`` and by every leg of a fork, so that
    "the same config" between a trunk and its anneal is a property of the code
    and not of two call sites agreeing.

    ``mixture``, ``schedule``, ``seed``, ``max_steps`` and ``output_dir`` are the
    five things a :class:`~src.generalist.fork.ForkLeg` overrides; passing them
    is how :func:`fork_callables` honours a leg.
    """
    import torch
    from transformers import TrainingArguments, set_seed

    from ..experiments.expressiveness.training.dispatch import (
        build_collator, build_model, select_active_params,
    )
    from .evaluate import build_validators
    from .lineage import Lineage
    from .mixture import MixtureSampler
    from .schedule import Schedule
    from .trainer import GeneralistTrainer

    if registry is None:
        registry, adapter_config = build_registry(config, adapter_config)
    elif adapter_config is None:
        adapter_config = config.adapter_config()
    from .adapters import molecules
    molecules.configure(adapter_config)

    if mixture is None:
        mixture = resolve_mixture(config, registry)
    output_dir = os.path.abspath(output_dir or config.run_dir())
    seed = config.seed if seed is None else int(seed)
    max_steps = int(max_steps if max_steps is not None else mixture.steps)
    lineage = lineage if lineage is not None else Lineage(config.lineage_dir())

    # One rank per process; the sampler slices a step across ranks itself (D4.3),
    # so it has to agree with the world size the TrainingArguments will report.
    # Read from the environment because the sampler is built *before* the
    # TrainingArguments that would otherwise be the source of truth.
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    model, tokenizer = build_model(
        config.impl, config.model_name, config.model_bias_config(),
        config.k_hop, config.k_hop_directed, device, config.flex_compile_mode)
    model = select_active_params(model, active_params=list(ACTIVE_PARAMS),
                                 lora=config.lora_config())
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id)
    collator = build_collator(config.impl, tokenizer, pad_token_id, config.k_hop,
                              config.k_hop_directed, magnetic_m=config.magnetic_m)

    sampler = MixtureSampler(
        mixture, seed=seed, get_source=make_get_source(config, registry,
                                                       adapter_config),
        accumulation_steps=config.accumulation_steps, world_size=world_size)
    schedule = schedule or Schedule.training(warmup_steps=config.warmup_steps)

    if validators is None:
        validators = build_validators(config.validator_specs())
    if eval_sets is None:
        eval_sets = build_eval_sets(config, registry, mixture,
                                    splits_wanted(validators), adapter_config,
                                    log=log)

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        # The dataloader runs with `batch_size=None` (D4.4: the sampler sizes a
        # micro-batch from a token budget), so this is not the batch size — it is
        # only what HF divides its own throughput counters by. One keeps those
        # counters in examples-per-micro-batch units, which is the honest reading.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_strategy="steps", logging_steps=config.logging_steps,
        save_strategy="steps", save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # No HF evaluation loop: D7's validators are the evaluation, and they run
        # on `TaskSource` objects rather than on an `eval_dataset`.
        eval_strategy="no",
        report_to=("wandb" if config.wandb_project else "none"),
        run_name=config.run_name,
        seed=seed, data_seed=seed,
        dataloader_num_workers=0,
        save_safetensors=True,
    )

    trainer_callbacks = list(callbacks)
    trainer = GeneralistTrainer(
        model=model, args=args, train_dataset=None, eval_dataset=None,
        data_collator=collator,
        active_params=list(ACTIVE_PARAMS), bias_lr=config.bias_lr,
        sampler=sampler, schedule=schedule, registry=registry,
        rewarm_steps=config.rewarm_steps,
        config_hash=config.config_hash(),
        lineage_hook=lineage.hook(child=output_dir),
        save_total_limit=config.save_total_limit,
        callbacks=trainer_callbacks,
    )

    run = Run(config=config, adapter_config=adapter_config, registry=registry,
              mixture=mixture, sampler=sampler, schedule=schedule, model=model,
              tokenizer=tokenizer, collator=collator, trainer=trainer,
              validators=tuple(validators), eval_sets=eval_sets, device=device,
              output_dir=output_dir, max_steps=max_steps, seed=seed,
              lineage=lineage)

    if fire_validators and validators:
        trainer.add_callback(validator_callback(run, log=log))
    if force_final_save:
        trainer.add_callback(save_final_step_callback())
    return run


# ─────────────────────────────────────────────────────────────────────────────
# D7 — the EvalContext, and firing validators during training
# ─────────────────────────────────────────────────────────────────────────────

def validator_config(run: Run) -> dict:
    """The open ``ctx.config`` field the built-ins read (D7.1).

    Only documented keys go in. ``grad_share``'s per-task loss closure is one of
    them and comes from the trainer, which is the only object that holds a real
    micro-batch; a run built without a trainer (``validate`` mode) leaves the key
    out, and the validator refuses rather than reporting a number it invented.
    """
    from .evaluate import builtin

    config = {
        builtin.ACTIVE_PARAMS: list(ACTIVE_PARAMS),
        builtin.MAX_SPD: run.config.max_spd,
        # The graph biases are all row-constant on the single-node text graphs
        # `base_exact` uses, so Property 2 holds and the check is meaningful.
        builtin.UNCONDITIONAL_FORWARD: False,
        "run_config": run.config.to_dict(),
    }
    loss_fn = getattr(run.trainer, "per_task_loss_fn", None)
    if loss_fn is not None:
        config[builtin.GRAD_SHARE_LOSS_FN] = loss_fn()
    counts_fn = getattr(run.trainer, "per_task_batch_counts", None)
    if counts_fn is not None:
        config[builtin.GRAD_SHARE_COUNTS_FN] = counts_fn
    return config


def eval_context(run: Run, step: int, *, scratch_dir=None, model=None,
                 schedule_position=None):
    """A :class:`~src.generalist.evaluate.EvalContext` for this run at ``step``."""
    from .evaluate import EvalContext

    trainer = run.trainer
    if schedule_position is None and run.schedule is not None:
        schedule_position = run.schedule.position(int(step))
    return EvalContext(
        step=int(step),
        model=model if model is not None else getattr(trainer, "model", run.model),
        tokenizer=run.tokenizer, registry=run.registry, mixture=run.mixture,
        arm=run.config.arm, schedule_position=schedule_position,
        eval_sets=run.eval_sets, train_sampler=run.sampler,
        base_model_name=run.config.model_name, collator=run.collator,
        device=run.device,
        scratch_dir=scratch_dir or os.path.join(run.output_dir, "eval_scratch"),
        config=validator_config(run),
    )


def run_evaluation(run: Run, step: int, *, event: str = "step", only=None,
                   scratch_dir=None, model=None, schedule_position=None,
                   strict: bool = False):
    """Fire the due validators and return the :class:`EvalRun`."""
    from .evaluate import run_validators

    ctx = eval_context(run, step, scratch_dir=scratch_dir, model=model,
                       schedule_position=schedule_position)
    os.makedirs(ctx.scratch_dir, exist_ok=True)
    return run_validators(ctx, run.validators, event=event, only=only,
                          strict=strict)


def validator_callback(run: Run, log=print):
    """D7's validators, fired from the training loop.

    ``on_step_end`` rather than HF's evaluation hook because there is no HF
    evaluation loop here: the validators score ``TaskSource`` objects, not an
    ``eval_dataset``, and their cadences are their own (``steps:<n>``,
    ``milestone``, ``end``). The metrics go through ``trainer.log`` so they land
    in ``log_history`` beside the loss, which is where the run record reads them
    from.

    Built inside a function, as `fork._periodic_validation` is and for the same
    reason: subclassing ``TrainerCallback`` at module scope would pull
    transformers into ``validate`` mode's import path.
    """
    from transformers import TrainerCallback

    class _ValidatorCallback(TrainerCallback):
        def _fire(self, event: str, step: int) -> None:
            # D7's rule is that a measurement never loses a run that has already
            # cost GPU-hours, and `run_validators` honours it for the validators
            # themselves. Everything *around* them — building the context,
            # logging, this reporting loop — is outside that guard, and a
            # 200-step smoke run is the record of why the guard has to be here
            # too: a typo in the line that reports a validator error killed the
            # run at step 50, before the checkpoint that step would have written.
            try:
                result = run_evaluation(run, step, event=event)
                if result.metrics:
                    run.trainer.log(dict(result.metrics))
                    if event == "end":
                        # `log` alone is invisible here. HF closes the progress
                        # bar in its own `on_train_end`, and its `on_log` writes
                        # through that bar, so an end-event metric reaches
                        # `log_history` and nothing else — which is how the
                        # smoke run computed `perm_spread` and `per_example` and
                        # printed neither. The durable copy is
                        # `log_history.json`; this is the readable one.
                        for key in sorted(result.metrics):
                            log(f"[eval] end {key} = {result.metrics[key]}")
                for status in result.errors():
                    log(f"[eval] {status.name} at step {step}: {status.message}")
            except Exception as exc:                               # noqa: BLE001
                log(f"[eval] the {event} evaluation at step {step} failed "
                    f"outside a validator ({type(exc).__name__}: {exc}); "
                    f"training continues")
                traceback.print_exc()

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step <= 0:
                return control
            self._fire("step", step)
            milestone = int(run.config.milestone_steps or 0)
            if milestone and step % milestone == 0:
                self._fire("milestone", step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self._fire("end", int(state.global_step))
            return control

    return _ValidatorCallback()


def save_final_step_callback():
    """Force a checkpoint at ``max_steps``.

    An anneal fork's *last* step is the reportable model — the LR is exactly
    ``lr_min`` there and nowhere else (D6, and `fork._plan_anneal`'s
    ``decay_steps + 1``). ``save_steps`` is a cadence, so unless the final step
    happens to be a multiple of it the model that gets published would be the one
    from just before the anneal finished, at a visibly higher LR than the
    schedule advertises.
    """
    from transformers import TrainerCallback

    class _SaveFinalStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.max_steps and int(state.global_step) >= int(state.max_steps):
                control.should_save = True
            return control

    return _SaveFinalStep()


# ─────────────────────────────────────────────────────────────────────────────
# D6 — the two callables fork.py asks for
# ─────────────────────────────────────────────────────────────────────────────

def validation_hook(config: RunConfig, *, registry, mixture, validators,
                    eval_sets, tokenizer=None, collator=None, device=None,
                    log=print):
    """``ValidationRequest -> {key: value}``: D6's validator seam, wired to D7.

    The cadence rule is the one thing this has to decide, because a request
    carries a stage and not an event:

    * ``stage="end"`` fires the **whole** set regardless of cadence. D6 says an
      anneal "runs the full validator set at the end", and the ``manual`` event
      is `evaluate/__init__.py`'s name for exactly that.
    * ``stage="periodic"`` respects cadences when the fork asked for everything
      (``ALL_VALIDATORS``), and fires unconditionally when it named a set. An
      ``adapt`` fork evaluates every ``eval_steps`` to find a first crossing, and
      a ``steps:500`` cadence would silently never fire at ``eval_steps: 25``;
      naming the validators is how a fork says "these, now".
    """
    from .fork import ALL_VALIDATORS

    def validate(request) -> dict:
        trainer = request.trainer
        names = tuple(request.validators or ())
        wants_all = (not names) or ALL_VALIDATORS in names
        only = None if wants_all else set(names)
        if request.stage == "end":
            event = "manual"
        else:
            event = "step" if wants_all else "manual"

        run = Run(
            config=config, adapter_config=None, registry=registry,
            mixture=getattr(trainer, "mixture", mixture),
            sampler=getattr(trainer, "sampler", None),
            schedule=getattr(trainer, "schedule", None),
            model=request.model, tokenizer=tokenizer,
            collator=getattr(trainer, "data_collator", collator),
            trainer=trainer, validators=tuple(validators), eval_sets=eval_sets,
            device=device, output_dir=getattr(trainer.args, "output_dir", ""),
            max_steps=int(getattr(trainer.args, "max_steps", 0) or 0),
            seed=config.seed)
        result = run_evaluation(run, int(request.step), event=event, only=only,
                                scratch_dir=request.scratch_dir,
                                schedule_position=request.schedule_position)
        for status in result.errors():
            log(f"[eval] {status.name} at step {request.step}: {status.message}")
        return dict(result.metrics)

    return validate


def fork_callables(config: RunConfig, *, registry, adapter_config, validators,
                   eval_sets, tokenizer=None, collator=None, device=None,
                   lineage=None, log=print):
    """``(trainer_factory, validate)`` — everything ``fork.fork`` needs from D8.

    The factory honours the five fields a leg owns: ``output_dir``, ``schedule``,
    ``mixture``, ``seed`` and ``max_steps``. It forces a save at ``max_steps``
    (see :class:`SaveFinalStep`) and it does *not* install a training-time
    validator callback — a fork's evaluations are driven by ``fork.py``, and two
    sources firing the same validator instances would corrupt ``throughput``'s
    wall-clock reference and double every generative pass.
    """
    built: dict = {}

    def trainer_factory(leg, plan):
        run = build_run(
            config, output_dir=leg.output_dir, mixture=leg.mixture,
            schedule=leg.schedule, seed=leg.seed, max_steps=leg.max_steps,
            registry=registry, adapter_config=adapter_config,
            validators=validators, eval_sets=eval_sets, lineage=lineage,
            force_final_save=True, fire_validators=False, log=log)
        built[leg.name] = run
        return run.trainer

    def _tokenizer():
        for run in built.values():
            return run.tokenizer
        return tokenizer

    def _device():
        """The device the leg's model is actually on.

        `mode_fork` passes none — it has no model until the factory builds one —
        so without this the whole validator set runs against ``device=None``,
        which the scorers read as CPU and the model answers from CUDA. The
        anneal fork trained its 21 steps and then lost every measurement to
        "index is on cpu, different from other tensors on cuda:0", which is the
        one thing an anneal exists to produce. `build_run` resolved the device
        correctly all along; nothing asked it for the answer.
        """
        for run in built.values():
            return run.device
        return device

    def validate(request):
        hook = validation_hook(
            config, registry=registry, mixture=None, validators=validators,
            eval_sets=eval_sets, tokenizer=_tokenizer(), collator=collator,
            device=_device(), log=log)
        return hook(request)

    return trainer_factory, validate
