"""D6 — branching a checkpoint: ``anneal``, ``admit``, ``adapt``.

A fork is the only way this harness produces a *reportable* model, and the only
way it answers "would this new dataset help" or "does the trunk learn a new graph
task faster than base Llama does". All three share one shape, and that shape is
what this module is:

    the parent keeps training; the child is a new run directory that starts from
    a copy of one of the parent's checkpoints, under a config that differs from
    the parent's in a written-down way.

Three properties hold in every mode, and each of them is a thing that goes wrong
quietly if it does not:

* **The parent is pinned and untouched.** ``checkpoint.pin`` exempts it from
  rotation, so the checkpoint a published number was branched from still exists
  months later. Nothing else in the parent's directory is written — the child
  works on its own copy, so a fork can never move the trunk's schedule or its
  sampler cursor.
* **The child is a new run directory** with its own record (``fork.json``, and
  ``result.json`` once it has run) and its own checkpoints. Forking in place
  would make the parent's checkpoint series mean two different things depending
  on when it was read.
* **A lineage entry is written** (D6, `lineage.py`) before the child trains, so a
  fork that dies on the first step still leaves a record of what it was.

**Modes.**

``anneal``
    Appends a ``decay`` segment to the parent's schedule, trains on the parent's
    mixture, and runs the full validator set at the end. This is the reportable
    model for a milestone (`MOLECULE_GENERALIST.md` §7: no best-val selection,
    the annealed checkpoint is the number). A leg that continues the parent's
    corpora usually has to raise their ``passes``: see :func:`_with_passes`.

``admit``
    Adds a candidate task to the mixture at a configured weight, appends a
    ``rewarm``, and trains a fixed budget. `PLAN.md` §5's four-part criterion has
    to be **in the config before the fork runs**, and :func:`plan_fork` refuses a
    fork without it; :func:`check_admission` applies it to whatever metrics come
    back. The four *regression suites* themselves are the trunk's validators and
    land with the trunk — :func:`run_admission_suites` is the named seam and
    raises rather than inventing them.

``adapt``
    Trains on **one held-out task only**, from the parent *and* from base Llama
    under identical configs, evaluating every ``eval_steps`` and recording
    steps-to-target. This is `PLAN.md` §3.3's adaptation-efficiency number, and
    for the molecule generalist it runs over three held-out tasks × two starting
    points × seeds. The two starting points are two *legs* of one fork, built
    from one plan, and :meth:`ForkLeg.fingerprint` is what makes "identical
    config" checkable rather than asserted.

**What this module does not build.** It never constructs a model, and it never
imports ``evaluate/``. A fork needs both, so both arrive as callables:

* ``trainer_factory(leg, plan) -> GeneralistTrainer`` — the caller builds the
  model, collator, sampler and ``TrainingArguments`` for one leg. It is handed
  ``leg.output_dir``, ``leg.schedule``, ``leg.mixture``, ``leg.seed`` and
  ``leg.max_steps`` and is expected to use them.
* ``validate(request: ValidationRequest) -> dict[str, float]`` — runs the
  validators the request names and returns a flat metric dict.

That is D8's wiring, not D6's; keeping it out here is what lets a fork be planned
(and its config checked) on a login node with neither torch nor a GPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from . import checkpoint as ckpt_mod
from .lineage import FORK_MODES, Lineage, append_line, utc_now
from .registry import Registry, is_held_out, resolve
from .schedule import Schedule, ScheduleError

logger = logging.getLogger(__name__)

#: D6's default decay length for an anneal fork: 10 % of the parent's steps so
#: far. DESIGN.md §10 leaves whether it is enough to the smoke run, which is why
#: it is a default and not a constant in the schedule.
DEFAULT_DECAY_FRACTION = 0.10

#: Where an anneal decays to, as a fraction of the run's LR.
#: `MOLECULE_GENERALIST.md` §7: "decays to lr/10".
DEFAULT_MIN_FACTOR = 0.1

DEFAULT_DECAY_SHAPE = "cosine"

#: Passed to a validator hook as the validator list when the fork wants
#: everything the harness has registered (D6: an anneal "runs the full validator
#: set"). Naming them here would duplicate ``evaluate/``'s registry and go stale.
ALL_VALIDATORS = "*"

#: `PLAN.md` §5's four parts, in its order. The names are the criterion's keys.
ADMISSION_PARTS = ("held_out", "text_only", "in_mixture", "candidate")

#: The first three must not regress; the fourth must improve.
ADMISSION_RULES = {"held_out": "no_regression", "text_only": "no_regression",
                   "in_mixture": "no_regression", "candidate": "improves"}

_MISSING = object()


class ForkError(RuntimeError):
    """A fork was asked for that would produce a number nobody could defend."""


def _write_json(path: str, payload) -> None:
    """Write a record, flushed to disk. Fork records outlive the job that ran."""
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


# ─────────────────────────────────────────────────────────────────────────────
# The validator seam
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationRequest:
    """What a fork hands its ``validate`` callable.

    Deliberately not an ``EvalContext``: ``evaluate/`` owns that type and this
    module must not import it. A request carries the *trainer*, from which the
    model, tokenizer, registry, collator, device and step all hang, plus the
    three things a fork knows and a trainer does not — which stage of which mode
    of which leg is asking.
    """

    #: ``"end"`` after a leg finishes; ``"periodic"`` at an ``eval_steps`` mark.
    stage: str
    mode: str
    leg: str
    step: int
    trainer: object
    #: Names to run, or ``(ALL_VALIDATORS,)`` for the whole registered set.
    validators: tuple
    scratch_dir: str
    plan: "ForkPlan"

    @property
    def model(self):
        return getattr(self.trainer, "model", None)

    @property
    def schedule_position(self):
        schedule = getattr(self.trainer, "schedule", None)
        return schedule.position(self.step) if schedule is not None else None


def run_admission_suites(*_args, **_kwargs):
    """The seam `PLAN.md` §5's four regression suites land in, and nothing more.

    They are the trunk's validators (DESIGN.md §9: "the admission regression
    gate — deferred; `admit` mode exists, the four suites are validators
    registered by name"), so they are named in a fork's config and run through
    the caller's ``validate`` hook like any other validator.
    :func:`check_admission` then applies the criterion to the metrics that come
    back. Inventing a suite here would produce a gate that passes for reasons
    nobody chose.
    """
    raise NotImplementedError(
        "the admission regression suites land with the trunk (DESIGN.md §9). "
        "Name them in the fork config's `criterion` as validators, run them "
        "through the `validate` hook, and pass the metrics to check_admission().")


# ─────────────────────────────────────────────────────────────────────────────
# The admission criterion (PLAN.md §5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdmissionVerdict:
    """The four-part criterion applied to one admit fork's metrics.

    ``decided`` is False whenever a part's metric is missing. An undecided gate
    is not a failing gate and is certainly not a passing one: the honest state
    when a suite did not run is "we do not know", and collapsing that into a
    boolean is how a merge happens on the strength of a suite that silently
    never executed.
    """

    decided: bool
    passed: Optional[bool]
    parts: dict
    missing: tuple
    reason: str

    def to_json(self) -> dict:
        return {"decided": self.decided, "passed": self.passed,
                "parts": self.parts, "missing": list(self.missing),
                "reason": self.reason}


def check_criterion(criterion) -> dict:
    """Refuse an admit fork whose criterion is not written down in full.

    `PLAN.md` §5 says the criterion goes into the config *before* the fork runs,
    and §3.4 adds that the seed count and aggregation rule are part of it — a
    test stated against noise bars but not against an *n* is not a test. Both are
    checked here, before a GPU is asked for, because after the run the temptation
    to choose the threshold that gives the answer is real and undocumented.
    """
    if not isinstance(criterion, dict) or not criterion:
        raise ForkError(
            "admit: the fork config needs a `criterion` block. PLAN.md §5's "
            "four-part admission test is written into the config *before* the "
            "fork runs, never chosen after the numbers are in.")
    missing = [p for p in ADMISSION_PARTS if p not in criterion]
    if missing:
        raise ForkError(
            f"admit: criterion is missing {missing}. All four of "
            f"{list(ADMISSION_PARTS)} must be stated (PLAN.md §5): the first three "
            "are 'does not regress', the fourth is 'the new task actually improves'.")
    if not criterion.get("seeds"):
        raise ForkError(
            "admit: criterion needs `seeds` — PLAN.md §3.4: the admission test is "
            "stated against noise bars but without a seed count it is not a test.")
    if not criterion.get("aggregate"):
        raise ForkError(
            "admit: criterion needs `aggregate` (how the seeds combine, e.g. "
            "\"mean\" or \"worst\"); PLAN.md §3.4 fixes it before the first fork.")

    checked = {"seeds": int(criterion["seeds"]),
               "aggregate": str(criterion["aggregate"])}
    for part in ADMISSION_PARTS:
        spec = criterion[part]
        if not isinstance(spec, dict):
            raise ForkError(f"admit: criterion.{part} must be an object, got {spec!r}")
        if not spec.get("metric"):
            raise ForkError(
                f"admit: criterion.{part} needs a `metric` — the validator key the "
                "rule is applied to.")
        rule = spec.get("rule", ADMISSION_RULES[part])
        if rule != ADMISSION_RULES[part]:
            raise ForkError(
                f"admit: criterion.{part} has rule {rule!r}; PLAN.md §5 fixes it at "
                f"{ADMISSION_RULES[part]!r}. A part that means something else is a "
                "different criterion and should be named differently.")
        if spec.get("tolerance") is None:
            raise ForkError(
                f"admit: criterion.{part} needs a `tolerance` — the recorded "
                "seed-noise bar the rule is measured against (PLAN.md §5 quotes "
                "±0.4–1.0 F1 on KGQA).")
        direction = spec.get("direction", "max")
        if direction not in ("max", "min"):
            raise ForkError(
                f"admit: criterion.{part}.direction must be 'max' or 'min', got "
                f"{direction!r}")
        checked[part] = {"metric": str(spec["metric"]), "rule": rule,
                         "tolerance": float(spec["tolerance"]),
                         "direction": direction,
                         "validator": spec.get("validator")}
    return checked


def check_admission(criterion: dict, metrics: dict,
                    baseline: dict = None) -> AdmissionVerdict:
    """Apply a checked criterion to a fork's metrics against the parent's.

    ``no_regression`` holds when the child is no worse than the parent by more
    than the part's tolerance; ``improves`` needs the child to be better than the
    parent by *at least* the tolerance, so a candidate that moves its own task by
    less than the noise bar does not get in on the strength of the sign of a
    difference.
    """
    metrics = metrics or {}
    baseline = baseline or {}
    parts, missing = {}, []
    for part in ADMISSION_PARTS:
        spec = criterion.get(part)
        if not spec:
            missing.append(part)
            continue
        key = spec["metric"]
        got, was = metrics.get(key, _MISSING), baseline.get(key, _MISSING)
        if got is _MISSING or was is _MISSING:
            missing.append(part)
            parts[part] = {"metric": key, "child": None if got is _MISSING else got,
                           "parent": None if was is _MISSING else was,
                           "held": None,
                           "why": "metric absent — the suite did not report it"}
            continue
        sign = 1.0 if spec.get("direction", "max") == "max" else -1.0
        delta = sign * (float(got) - float(was))
        tol = float(spec["tolerance"])
        held = delta >= tol if spec["rule"] == "improves" else delta >= -tol
        parts[part] = {"metric": key, "child": float(got), "parent": float(was),
                       "delta": delta, "tolerance": tol, "rule": spec["rule"],
                       "held": bool(held)}
    if missing:
        return AdmissionVerdict(
            decided=False, passed=None, parts=parts, missing=tuple(missing),
            reason=f"undecided: no metric for {list(missing)}. An admission is a "
                   "merge into the trunk and is never taken on a partial gate.")
    passed = all(p["held"] for p in parts.values())
    failed = [name for name, p in parts.items() if not p["held"]]
    return AdmissionVerdict(
        decided=True, passed=passed, parts=parts, missing=(),
        reason="all four parts hold" if passed else f"failed on {failed}")


# ─────────────────────────────────────────────────────────────────────────────
# The plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ForkLeg:
    """One run the fork launches.

    ``anneal`` and ``admit`` have exactly one leg. ``adapt`` has one per
    (starting point × seed) and its whole claim rests on those legs being
    identical apart from where the weights came from — hence
    :meth:`fingerprint`, which hashes everything a leg *is* except the three
    fields that are allowed to differ.
    """

    name: str
    #: ``"parent"`` (the checkpoint's weights) or ``"base"`` (base Llama).
    start: str
    seed: int
    output_dir: str
    #: The fork's own copy of the parent checkpoint, or ``None`` for a base leg.
    start_checkpoint: Optional[str]
    #: True to continue the parent run (HF resume: optimizer, RNG, sampler and
    #: schedule position all restored). False to take the *weights* only and
    #: start a fresh optimizer and schedule — which is what ``adapt`` measures,
    #: since Adam moments carried over from a different mixture would be a
    #: confound in a steps-to-target number.
    resume: bool
    schedule: Schedule
    mixture: object
    mixture_config: tuple
    max_steps: int
    eval_steps: Optional[int] = None

    def fingerprint(self) -> str:
        """Everything about this leg except where it starts and where it writes.

        Two legs of an ``adapt`` fork must agree on this exactly; if they do not,
        the steps-to-target difference between them is not attributable to the
        starting weights.
        """
        payload = {
            "seed": int(self.seed),
            "max_steps": int(self.max_steps),
            "eval_steps": self.eval_steps,
            "schedule": self.schedule.to_json(),
            "mixture_hash": self.mixture.hash(),
            "mixture_config": [dict(e) for e in self.mixture_config],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         default=str).encode()).hexdigest()

    def to_json(self) -> dict:
        return {"name": self.name, "start": self.start, "seed": int(self.seed),
                "output_dir": self.output_dir,
                "start_checkpoint": self.start_checkpoint, "resume": self.resume,
                "max_steps": int(self.max_steps), "eval_steps": self.eval_steps,
                "schedule": self.schedule.to_json(),
                "mixture_config": [dict(e) for e in self.mixture_config],
                "mixture_hash": self.mixture.hash(),
                "fingerprint": self.fingerprint()}


@dataclass
class ForkPlan:
    """Everything a fork decides before it touches a GPU.

    Separated from :func:`fork` so that `validate` mode can resolve a fork
    config, print what it would do and refuse a bad one on a login node — the
    same reason D8.1 has a `validate` mode at all. Nothing in here imports torch.
    """

    mode: str
    parent_ckpt: str
    parent_step: int
    parent_state: dict
    parent_schedule: Schedule
    run_dir: str
    legs: tuple
    config: dict
    config_diff: dict
    validators: tuple
    criterion: Optional[dict] = None
    target: Optional[dict] = None
    created: str = field(default_factory=utc_now)

    def leg(self, name: str) -> ForkLeg:
        for leg in self.legs:
            if leg.name == name:
                return leg
        raise KeyError(f"{name}: not a leg of this fork "
                       f"({[one.name for one in self.legs]})")

    def to_json(self) -> dict:
        return {
            "mode": self.mode, "created": self.created,
            "parent": self.parent_ckpt, "parent_step": int(self.parent_step),
            "run_dir": self.run_dir,
            "config": self.config, "config_diff": self.config_diff,
            "validators": list(self.validators),
            "criterion": self.criterion, "target": self.target,
            "legs": [leg.to_json() for leg in self.legs],
            "parent_state": {k: self.parent_state.get(k) for k in
                             ("step", "lr", "bias_lr", "tokens_per_step",
                              "mixture_hash", "schema_version", "seed",
                              "architecture_hash", "config_hash")},
        }


def plan_fork(from_ckpt: str, mode: str, config: dict, *,
              registry: Registry = None, parent_mixture=None,
              run_dir: str = None) -> ForkPlan:
    """Resolve a fork config against a parent checkpoint. No side effects.

    Refuses, before anything is written: an unknown mode; a checkpoint without
    ``COMPLETE`` (it is a partial write, and a fork from one would report numbers
    for a model that was never finished); a schedule that can no longer be
    extended; an ``admit`` without its criterion; an ``adapt`` on a task that is
    not held out.
    """
    if mode not in FORK_MODES:
        raise ForkError(f"mode must be one of {FORK_MODES}, got {mode!r}")
    config = dict(config or {})

    from_ckpt = os.path.abspath(from_ckpt)
    try:
        parent_state = ckpt_mod.verify(from_ckpt)
    except ckpt_mod.CheckpointError as exc:
        raise ForkError(f"fork --from {from_ckpt}: {exc}") from exc
    parent_schedule, _sampler_state, _state = ckpt_mod.restore_extras(from_ckpt)

    parent_step = int(parent_state.get("step") or 0)
    parent_run_dir = os.path.dirname(from_ckpt)
    name = config.get("name") or f"{os.path.basename(parent_run_dir)}-{mode}-{parent_step}"
    run_dir = os.path.abspath(
        run_dir or config.get("run_dir")
        or os.path.join(os.path.dirname(parent_run_dir), name))
    if os.path.abspath(run_dir) == os.path.abspath(parent_run_dir):
        raise ForkError(
            f"fork: the child run directory resolves to the parent's ({run_dir}). "
            "A fork writes its own checkpoints and its own record; sharing the "
            "parent's directory would make the parent's series mean two things.")

    tokens_per_step = int(config.get("tokens_per_step")
                          or parent_state.get("tokens_per_step") or 0)
    if tokens_per_step <= 0:
        raise ForkError(
            f"fork: no tokens_per_step — the parent checkpoint records "
            f"{parent_state.get('tokens_per_step')!r} and the config sets none. "
            "It is what turns a step budget into examples (D4.4).")
    seed = int(config.get("seed", parent_state.get("seed") or 0))

    # `anneal` and `admit` train the parent's mixture, and the checkpoint records
    # it entry by entry, so a fork is self-contained from a checkpoint path. An
    # explicit `parent_mixture=` still wins: a caller that has the live Mixture in
    # hand should not have to round-trip it through JSON.
    if parent_mixture is None:
        parent_mixture = parent_state.get("mixture_entries") or None

    builder = {"anneal": _plan_anneal, "admit": _plan_admit,
               "adapt": _plan_adapt}[mode]
    built = builder(config=config, registry=registry, parent_mixture=parent_mixture,
                    parent_state=parent_state, parent_schedule=parent_schedule,
                    parent_step=parent_step, tokens_per_step=tokens_per_step,
                    seed=seed, run_dir=run_dir, from_ckpt=from_ckpt)

    parent_view = {
        "mixture": list(_as_mixture_config(parent_mixture)) if parent_mixture else None,
        "tokens_per_step": int(parent_state.get("tokens_per_step") or 0),
        "seed": parent_state.get("seed"),
        "schedule_segments": len(parent_schedule.segments),
    }
    child_view = {
        "mixture": list(built["legs"][0].mixture_config),
        "tokens_per_step": tokens_per_step,
        "seed": seed,
        "schedule_segments": len(built["legs"][0].schedule.segments),
    }
    child_view.update(built.get("diff_extra", {}))
    config_diff = _diff(parent_view, child_view)

    return ForkPlan(
        mode=mode, parent_ckpt=from_ckpt, parent_step=parent_step,
        parent_state=parent_state, parent_schedule=parent_schedule,
        run_dir=run_dir, legs=tuple(built["legs"]), config=config,
        config_diff=config_diff,
        validators=tuple(config.get("validators") or ()) or (ALL_VALIDATORS,),
        criterion=built.get("criterion"), target=built.get("target"))


def _plan_anneal(*, config, parent_mixture, parent_schedule, parent_step,
                 tokens_per_step, seed, run_dir, from_ckpt, registry,
                 parent_state, **_):
    """Append a ``decay`` to the parent's schedule and train the parent's mixture."""
    if not parent_mixture:
        raise ForkError(
            "anneal: needs the parent's mixture (`parent_mixture=` or the config's "
            "`mixture`). It is not reconstructible from the checkpoint — state.json "
            "records the mixture *hash* and the registry snapshot, not the entry "
            "list with its weight overrides.")
    mixture_config = _with_passes(_as_mixture_config(parent_mixture),
                                  config.get("passes"), mode="anneal")

    decay_steps = int(config.get("decay_steps")
                      or max(1, round(DEFAULT_DECAY_FRACTION * parent_step)))
    if decay_steps < 1:
        raise ForkError(f"anneal: decay_steps must be >= 1, got {decay_steps}")
    min_factor = _min_factor(config, parent_state)
    shape = config.get("decay_shape", DEFAULT_DECAY_SHAPE)

    schedule = _copy_schedule(parent_schedule)
    try:
        schedule.append_decay(at_step=parent_step, decay_steps=decay_steps,
                              min_factor=min_factor, shape=shape)
    except ScheduleError as exc:
        raise ForkError(
            f"anneal: cannot append a decay at step {parent_step} to "
            f"{parent_schedule!r}: {exc}") from exc

    mixture = _resolve(registry, mixture_config, tokens_per_step,
                       steps=decay_steps + 1)
    _check_budget_left(mixture, parent_state, mode="anneal")
    # The decay's endpoint is a step the model must actually take. A segment of
    # `decay_steps` interpolates over [start, start + decay_steps], so the LR is
    # exactly `min_factor` at the *last* of `decay_steps + 1` steps; stopping one
    # earlier would make the reportable model the one from just before the anneal
    # finished, at a visibly higher LR than the schedule advertises.
    legs = [ForkLeg(name="anneal", start="parent", seed=seed,
                    output_dir=os.path.join(run_dir, "anneal"),
                    start_checkpoint=None, resume=True, schedule=schedule,
                    mixture=mixture, mixture_config=mixture_config,
                    max_steps=parent_step + decay_steps + 1,
                    eval_steps=config.get("eval_steps"))]
    return {"legs": legs,
            "diff_extra": {"decay_steps": decay_steps, "min_factor": min_factor,
                           "decay_shape": shape}}


def _plan_admit(*, config, parent_mixture, parent_schedule, parent_step,
                tokens_per_step, seed, run_dir, registry, parent_state, **_):
    """Add the candidate at its weight, re-warm, train a fixed budget."""
    if not parent_mixture:
        raise ForkError(
            "admit: needs the parent's mixture (`parent_mixture=` or the config's "
            "`mixture`) — the candidate is added *to* it, so it has to be known.")
    candidate = config.get("candidate")
    if not isinstance(candidate, dict) or not candidate.get("name"):
        raise ForkError(
            "admit: config needs `candidate` as {\"name\": …, \"weight\": …} — the "
            "task being tested for admission and the share it is tested at.")
    if candidate.get("weight") is None:
        raise ForkError(
            f"admit: candidate {candidate['name']} has no weight. The gate is "
            "measured at a share, and a default share would make the verdict a "
            "property of this code rather than of the config.")

    mixture_config = _as_mixture_config(parent_mixture)
    if any(e["name"] == candidate["name"] for e in mixture_config):
        raise ForkError(
            f"admit: {candidate['name']} is already in the parent's mixture; there "
            "is nothing to admit.")
    mixture_config = mixture_config + ({"name": candidate["name"],
                                        "weight": float(candidate["weight"])},)
    mixture_config = _with_passes(mixture_config, config.get("passes"),
                                  mode="admit")

    budget_steps = config.get("budget_steps")
    if not budget_steps:
        raise ForkError(
            "admit: config needs `budget_steps` — PLAN.md §5 runs the gate over a "
            "*fixed* budget, so it cannot be left to a corpus's pass cap.")
    budget_steps = int(budget_steps)

    rewarm_steps = config.get("rewarm_steps") or _warmup_length(parent_schedule)
    if not rewarm_steps:
        raise ForkError(
            "admit: needs `rewarm_steps` — the mixture changes, so D5.2 calls for a "
            "re-warm, and the parent's schedule has no warmup segment to take a "
            "length from.")
    rewarm_from = float(config.get("rewarm_from", 0.1))

    schedule = _copy_schedule(parent_schedule)
    try:
        schedule.append_rewarm(at_step=parent_step, rewarm_steps=int(rewarm_steps),
                               from_factor=rewarm_from)
    except ScheduleError as exc:
        raise ForkError(
            f"admit: cannot append a re-warm at step {parent_step} to "
            f"{parent_schedule!r}: {exc}") from exc

    criterion = check_criterion(config.get("criterion"))
    mixture = _resolve(registry, mixture_config, tokens_per_step, steps=budget_steps)
    _check_budget_left(mixture, parent_state, mode="admit")
    legs = [ForkLeg(name="admit", start="parent", seed=seed,
                    output_dir=os.path.join(run_dir, "admit"),
                    start_checkpoint=None, resume=True, schedule=schedule,
                    mixture=mixture, mixture_config=mixture_config,
                    max_steps=parent_step + budget_steps,
                    eval_steps=config.get("eval_steps"))]
    return {"legs": legs, "criterion": criterion,
            "diff_extra": {"candidate": candidate["name"],
                           "candidate_weight": float(candidate["weight"]),
                           "rewarm_steps": int(rewarm_steps),
                           "budget_steps": budget_steps}}


def _plan_adapt(*, config, tokens_per_step, seed, run_dir, registry,
                parent_state, **_):
    """One held-out task, from the parent and from base, identical otherwise."""
    task = config.get("task")
    if not task:
        raise ForkError(
            "adapt: config needs `task` — the single held-out task the fork trains "
            "on (D6). A mixture here would not be an adaptation measurement.")
    if registry is None:
        raise ForkError("adapt: needs a registry to resolve the task's spec")
    spec = registry.get(task)
    if not is_held_out(spec) and not config.get("allow_in_mixture_task"):
        raise ForkError(
            f"adapt: {task} is not held out. Steps-to-target from the parent is only "
            "an adaptation number for a task the parent never trained on — on an "
            "in-mixture task it measures how recently the task was sampled. Set "
            "`allow_in_mixture_task: true` if this is a deliberate control.")

    budget_steps = int(config.get("budget_steps") or 0)
    if budget_steps < 1:
        raise ForkError(
            "adapt: config needs `budget_steps` — the fixed budget both starting "
            "points get. Steps-to-target is only comparable under one budget.")
    eval_steps = int(config.get("eval_steps") or 0)
    if eval_steps < 1:
        raise ForkError(
            "adapt: config needs `eval_steps` — the target is crossed *between* "
            "evaluations, so the resolution of the number is this value.")
    target = _check_target(config.get("target"))

    starts = tuple(config.get("starts") or ("parent", "base"))
    for start in starts:
        if start not in ("parent", "base"):
            raise ForkError(
                f"adapt: start must be 'parent' or 'base', got {start!r}")
    seeds = [int(s) for s in (config.get("seeds") or [seed])]

    warmup_steps = int(config.get("warmup_steps",
                                  max(1, min(10, budget_steps // 10))))
    mixture_config = ({"name": task,
                       "weight": float(config.get("weight", 1.0))},)

    legs = []
    for leg_seed in seeds:
        # One mixture and one schedule object per seed, shared by both starts:
        # the two legs are the same run twice over apart from where the weights
        # came from, and building them separately would leave room to drift.
        mixture = _adapt_mixture(registry, task, tokens_per_step, budget_steps,
                                 weight=mixture_config[0]["weight"])
        for start in starts:
            leg_name = f"{start}-s{leg_seed}" if len(seeds) > 1 else start
            legs.append(ForkLeg(
                name=leg_name, start=start, seed=leg_seed,
                output_dir=os.path.join(run_dir, leg_name),
                start_checkpoint=None, resume=False,
                schedule=Schedule.training(warmup_steps=warmup_steps),
                mixture=mixture, mixture_config=mixture_config,
                max_steps=budget_steps, eval_steps=eval_steps))

    by_seed: dict = {}
    for leg in legs:
        by_seed.setdefault(leg.seed, []).append(leg)
    for leg_seed, group in by_seed.items():
        prints = {leg.fingerprint() for leg in group}
        if len(prints) > 1:
            raise ForkError(
                f"adapt: the legs at seed {leg_seed} do not share a config "
                f"({[leg.name for leg in group]}). The whole measurement is the "
                "difference between two runs that differ only in their starting "
                "weights.")
    return {"legs": legs, "target": target,
            "diff_extra": {"task": task, "budget_steps": budget_steps,
                           "eval_steps": eval_steps, "starts": list(starts),
                           "seeds": seeds, "warmup_steps": warmup_steps}}


# ─────────────────────────────────────────────────────────────────────────────
# Plan helpers
# ─────────────────────────────────────────────────────────────────────────────

def _copy_schedule(schedule: Schedule) -> Schedule:
    """A detached copy. Appending to the parent's own object would edit the trunk."""
    return Schedule.from_json(copy.deepcopy(schedule.to_json()))


def _warmup_length(schedule: Schedule):
    for seg in schedule.segments:
        if seg.kind == "warmup" and seg.steps:
            return int(seg.steps)
    return None


def _min_factor(config: dict, parent_state: dict) -> float:
    """Where the anneal decays to, as a *factor* on the run's LR.

    The schedule works in factors so that one curve serves both parameter groups
    (D5.2), but a config is more likely to say `lr_min: 3e-5` than
    `min_factor: 0.1`. Both are accepted; the absolute form is converted against
    the LR the parent recorded, and a conversion with no LR to divide by is an
    error rather than a guess.
    """
    if config.get("min_factor") is not None:
        return float(config["min_factor"])
    if config.get("lr_min") is not None:
        lr = parent_state.get("lr")
        if not lr:
            raise ForkError(
                "anneal: `lr_min` is absolute and the parent checkpoint records no "
                "`lr` to convert it against; set `min_factor` instead.")
        return float(config["lr_min"]) / float(lr)
    return DEFAULT_MIN_FACTOR


def _check_target(target) -> dict:
    if not isinstance(target, dict) or not target.get("metric"):
        raise ForkError(
            "adapt: config needs `target` as {\"metric\": …, \"value\": …, "
            "\"direction\": \"max\"|\"min\"} — steps-to-*what* is the number.")
    if target.get("value") is None:
        raise ForkError("adapt: target needs a `value` to reach")
    direction = target.get("direction", "max")
    if direction not in ("max", "min"):
        raise ForkError(f"adapt: target.direction must be 'max' or 'min', "
                        f"got {direction!r}")
    if _names_test_split(target["metric"]):
        # D7.4: the harness refuses any selection key naming the test split. An
        # adapt target *is* a selection rule — it decides when the run has
        # arrived — so it falls under the same refusal.
        raise ForkError(
            f"adapt: target metric {target['metric']!r} names a test split. "
            "Selection against test is refused (D7.4); target the val split.")
    return {"metric": str(target["metric"]), "value": float(target["value"]),
            "direction": direction}


def _names_test_split(key: str) -> bool:
    """True if ``test`` is a *segment* of a metric key, not a substring of a word.

    ``<validator>/<task>/<metric>`` is the namespacing D7.1 fixes, and splits are
    written into the metric part (``test_roc_auc``). Matching the bare substring
    would also catch a task legitimately called ``fastest_path``.
    """
    return "test" in str(key).replace("/", "_").replace("-", "_").split("_")


def _as_mixture_config(mixture) -> tuple:
    """``[{"name", "weight", …}]`` from a config list or a resolved ``Mixture``."""
    entries = getattr(mixture, "entries", None)
    if entries is not None:
        return tuple({"name": e.name, "weight": float(e.weight)} for e in entries)
    out = []
    for entry in mixture:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ForkError(f"mixture entry must be an object with a name, got {entry!r}")
        out.append(dict(entry))
    return tuple(out)


def _with_passes(mixture_config, overrides, mode: str) -> tuple:
    """Raise the parent's ``passes`` for the fork's own leg.

    A fork that continues the parent's mixture continues the parent's *corpora*
    too, and a corpus is bounded: ``passes x train_size`` is everything it will
    ever hand out, across the trunk and every leg that follows it. The trunk's
    budget rule spends that allowance to the last example — it sizes itself as
    ``min over corpora of available / share`` — so a fork appended to a trunk that
    ran to its own step count starts with nothing left to draw. Extending the
    corpus is the only thing that can change, so ``passes`` is the only thing this
    override may set, and only for tasks the parent already trains.

    The override is a fork-config field rather than something inferred here
    because "how many more epochs of BACE this anneal is allowed" is a statement
    about the experiment, not arithmetic: it belongs in the file that a reader
    compares against the specialist's recipe.
    """
    if not overrides:
        return tuple(mixture_config)
    if not isinstance(overrides, dict):
        raise ForkError(
            f"{mode}: `passes` must be an object mapping task name -> passes, got "
            f"{overrides!r}")
    names = {entry["name"] for entry in mixture_config}
    unknown = sorted(set(overrides) - names)
    if unknown:
        raise ForkError(
            f"{mode}: `passes` names {', '.join(unknown)}, which the fork's mixture "
            f"does not train. It may only raise the allowance of a task already in "
            f"the mixture ({', '.join(sorted(names))}); adding a task is what an "
            "`admit` fork is for.")
    out = []
    for entry in mixture_config:
        entry = dict(entry)
        if entry["name"] in overrides:
            value = int(overrides[entry["name"]])
            if value < 1:
                raise ForkError(f"{mode}: passes for {entry['name']} must be >= 1, "
                                f"got {value}")
            entry["passes"] = value
        out.append(entry)
    return tuple(out)


def _check_budget_left(mixture, parent_state: dict, mode: str) -> None:
    """Refuse a leg whose corpora the parent has already spent.

    Without this the shortfall surfaces deep inside training, as an accumulation
    error from the one short step at the end of the data — "an optimizer step drew
    6 example(s) but gradient_accumulation_steps is 8" — which points at the batch
    shape and says nothing about the cause. The plan is where it is knowable and
    where the fix (:func:`_with_passes`) is a one-line config change, so the
    refusal belongs there and it names the number to set.

    Generators are unbounded by D4.2 — they draw a fresh pass — so only corpora
    (``available is not None``) can run short.
    """
    consumed = parent_state.get("examples_per_task") or {}
    short = []
    for entry in mixture.entries:
        if entry.available is None:
            continue
        spent = int(consumed.get(entry.name, 0))
        left = int(entry.available) - spent
        if left < entry.examples:
            per_pass = int(entry.available) // max(1, int(entry.passes))
            need = spent + entry.examples
            short.append((entry.name, left, entry.examples, entry.passes,
                          -(-need // max(1, per_pass))))
    if not short:
        return
    lines = [f"{name}: {left} example(s) left of a {passes}-pass allowance, "
             f"{want} wanted -- set passes to {need_passes}"
             for name, left, want, passes, need_passes in short]
    raise ForkError(
        f"{mode}: the parent has already spent the corpus it would train on. "
        + "; ".join(lines)
        + ". A corpus hands out `passes x train_size` examples in total and the "
        "trunk's budget rule spends all of them, so a leg after it needs the "
        "allowance raised: add `\"passes\": {\"<task>\": <n>}` to the fork config.")


def _resolve(registry, mixture_config, tokens_per_step, steps):
    if registry is None:
        raise ForkError("fork: needs a registry to resolve the child's mixture")
    return resolve(registry, [dict(e) for e in mixture_config],
                   tokens_per_step=tokens_per_step, steps=int(steps))


def _adapt_mixture(registry: Registry, task: str, tokens_per_step: int, steps: int,
                   weight: float = 1.0):
    """A one-task mixture over a task every other mixture must refuse.

    D2.1 makes ``resolve`` reject a held-out task on sight, and that is right for
    every training mixture — an ``adapt`` fork is the single sanctioned exception
    (D6), safe because the fork is a leaf: it never merges back, and the parent
    never sees the task. ``resolve`` takes that exception by name through
    ``allow_held_out``, so the budget arithmetic is the same one every other
    mixture goes through and the exception is visible in the call.
    """
    return resolve(registry, [{"name": task, "weight": float(weight)}],
                   tokens_per_step=tokens_per_step, steps=int(steps),
                   allow_held_out=(task,))


def _diff(parent_view: dict, child_view: dict) -> dict:
    diff = {}
    for key, new in child_view.items():
        old = parent_view.get(key, _MISSING)
        if old is _MISSING or old != new:
            diff[key] = {"parent": None if old is _MISSING else old, "child": new}
    return diff


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LegResult:
    """One leg's outcome: its metrics over time and its steps-to-target."""

    leg: ForkLeg
    final_step: int = 0
    metrics: dict = field(default_factory=dict)
    #: ``[(step, {metric: value})]`` from the ``eval_steps`` marks.
    history: list = field(default_factory=list)
    steps_to_target: Optional[int] = None
    train_output: object = None

    def to_json(self) -> dict:
        return {"leg": self.leg.name, "start": self.leg.start,
                "seed": int(self.leg.seed), "output_dir": self.leg.output_dir,
                "final_step": int(self.final_step), "metrics": self.metrics,
                "history": [[int(s), m] for s, m in self.history],
                "steps_to_target": self.steps_to_target}


@dataclass
class ForkResult:
    plan: ForkPlan
    legs: dict = field(default_factory=dict)
    verdict: Optional[AdmissionVerdict] = None
    ran: bool = False

    def adaptation_table(self) -> dict:
        """``{leg name: steps to target}`` — `PLAN.md` §3.3's readout.

        ``None`` for a leg that never reached the target inside its budget, which
        is a result and not a missing value: "did not get there in N steps" is
        exactly what an adaptation curve is allowed to say.
        """
        return {name: r.steps_to_target for name, r in self.legs.items()}

    def to_json(self) -> dict:
        return {"mode": self.plan.mode, "run_dir": self.plan.run_dir,
                "parent": self.plan.parent_ckpt,
                "parent_step": int(self.plan.parent_step), "ran": self.ran,
                "finished": utc_now(),
                "legs": {n: r.to_json() for n, r in self.legs.items()},
                "adaptation": self.adaptation_table(),
                "verdict": self.verdict.to_json() if self.verdict else None}


def steps_to_target(history, target: dict):
    """The first evaluated step at which ``target`` is met, or ``None``.

    The *first* crossing, not the best value: adaptation efficiency is how long
    it took to get there, and a run that crosses at step 200 and dips afterwards
    still crossed at 200. The resolution is the fork's ``eval_steps``.
    """
    if not target:
        return None
    key, value = target["metric"], float(target["value"])
    want_max = target.get("direction", "max") == "max"
    for step, metrics in sorted(history, key=lambda pair: pair[0]):
        got = (metrics or {}).get(key)
        if got is None:
            continue
        if (float(got) >= value) if want_max else (float(got) <= value):
            return int(step)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Running a fork
# ─────────────────────────────────────────────────────────────────────────────

def prepare_fork(plan: ForkPlan, *, lineage: Lineage = None,
                 results_dir: str = None, copy_checkpoint: bool = True,
                 runs_jsonl: str = None) -> ForkPlan:
    """Everything on disk that must exist before a leg trains.

    In order, because the order is what makes a crash recoverable: pin the
    parent, then build the child's directories and its copy of the checkpoint,
    then write ``fork.json``, then the lineage entries. A fork that dies after
    this has a complete record of what it was going to do; one that dies before
    it has left the parent exactly as it found it, plus a ``PINNED`` marker,
    which costs nothing but a file.

    Returns the plan with the legs' ``start_checkpoint`` filled in.
    """
    lineage = lineage or Lineage(results_dir or os.path.dirname(plan.run_dir))

    # The parent is pinned first: from here on it is exempt from rotation, so a
    # long fork cannot have the checkpoint it branched from deleted underneath it
    # by the parent's own next save.
    ckpt_mod.pin(plan.parent_ckpt,
                 reason=f"fork {plan.mode} -> {plan.run_dir}")

    os.makedirs(plan.run_dir, exist_ok=True)
    legs = []
    for leg in plan.legs:
        os.makedirs(leg.output_dir, exist_ok=True)
        start_ckpt = None
        if leg.start == "parent":
            if not copy_checkpoint:
                if leg.resume:
                    raise ForkError(
                        f"leg {leg.name!r}: copy_checkpoint=False would resume "
                        "directly out of the parent's directory, and a resume "
                        "restores the schedule from the checkpoint it resumes — so "
                        "the fork's appended segment would have to be written into "
                        "the parent. A fork never writes to its parent.")
                start_ckpt = plan.parent_ckpt
            else:
                start_ckpt = os.path.join(
                    leg.output_dir, os.path.basename(plan.parent_ckpt))
                if not os.path.exists(start_ckpt):
                    shutil.copytree(plan.parent_ckpt, start_ckpt)
                # The child's schedule lives in the child's copy.
                # `prepare_resume` restores the schedule from the checkpoint it
                # resumes, so the appended decay (or re-warm) has to be *in* that
                # file; written anywhere else it is silently overwritten on the
                # first step. The file name is unchanged, so the `files` list
                # state.json carries still verifies.
                _write_json(os.path.join(start_ckpt, ckpt_mod.SCHEDULE_FILE),
                            leg.schedule.to_json())
                ckpt_mod.pin(start_ckpt, reason=f"fork origin ({plan.mode})")
        legs.append(replace(leg, start_checkpoint=start_ckpt))
    plan.legs = tuple(legs)

    _write_json(os.path.join(plan.run_dir, "fork.json"), plan.to_json())

    for leg in plan.legs:
        lineage.record_fork(
            child=leg.output_dir, parent=plan.parent_ckpt,
            parent_step=plan.parent_step, mode=plan.mode,
            config_diff=dict(plan.config_diff, start={"parent": "parent",
                                                      "child": leg.start}),
            schedule=leg.schedule.to_json(),
            note=f"leg {leg.name!r} starts from "
                 f"{'the parent checkpoint' if leg.start == 'parent' else 'base'}")

    if runs_jsonl:
        # D6 gives a fork "its own runs.jsonl line". The sweep runner owns that
        # file's schema, so this writes the minimum that identifies the run and
        # leaves the metrics to whatever runs the leg.
        for leg in plan.legs:
            append_line(runs_jsonl, json.dumps(
                {"run": leg.output_dir, "kind": "fork", "mode": plan.mode,
                 "parent": plan.parent_ckpt, "parent_step": plan.parent_step,
                 "leg": leg.name, "start": leg.start, "seed": int(leg.seed),
                 "created": plan.created},
                sort_keys=True, separators=(",", ":"), default=str))
    return plan


def load_start_weights(trainer, ckpt_dir: str) -> None:
    """Put a checkpoint's *weights* into a trainer, without resuming it.

    ``adapt`` needs the parent's weights and nothing else: a fresh optimizer, a
    fresh schedule and a fresh sampler, because Adam moments and a mixture cursor
    carried over from the trunk would be a confound in a steps-to-target number
    that is supposed to be about the weights.

    The trainer's own loader is used rather than a hand-rolled one because
    ``GraphTrainerV2._load_from_checkpoint`` is what pairs the PEFT adapter with
    ``bias_parameters.pt`` — the 2026-07-17 bug is precisely what happens when
    something else loads only the adapter. ``verify`` afterwards re-checks the
    bias-norm fingerprint, so a mispaired adapter and bias file costs a startup
    rather than a whole adaptation curve.
    """
    trainer._load_from_checkpoint(ckpt_dir)
    ckpt_mod.verify(ckpt_dir, model=trainer.model,
                    active_params=getattr(trainer, "active_params", None))


def _periodic_validation(every: int, request_for: Callable, record: Callable):
    """A ``TrainerCallback`` that evaluates every ``every`` steps (adapt, D6).

    Built inside a function rather than at module level so that ``plan_fork``
    stays importable without transformers — a fork config is checked on a login
    node, where nothing heavier than the standard library is available.
    """
    from transformers import TrainerCallback

    class _PeriodicValidation(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step <= 0 or step % every:
                return
            record(step, request_for("periodic", step))

    return _PeriodicValidation()


def fork(from_ckpt: str, mode: str, config: dict, *, registry: Registry = None,
         parent_mixture=None, run_dir: str = None, results_dir: str = None,
         lineage: Lineage = None, trainer_factory: Callable = None,
         validate: Callable = None, runs_jsonl: str = None,
         copy_checkpoint: bool = True) -> ForkResult:
    """Plan a fork, lay it down on disk, and run its legs.

    With no ``trainer_factory`` the fork stops after :func:`prepare_fork`: the
    directories, the copy, the pin, ``fork.json`` and the lineage entries all
    exist and ``ForkResult.ran`` is False. That is the useful shape for
    `validate` mode and for a fork whose legs are submitted as separate Slurm
    jobs.

    ``validate`` is called at the end of every leg, and additionally every
    ``eval_steps`` for ``adapt``. It receives a :class:`ValidationRequest` and
    returns a flat ``{metric: value}`` dict; the fork never interprets the keys
    beyond the criterion's and the target's.
    """
    plan = plan_fork(from_ckpt, mode, config, registry=registry,
                     parent_mixture=parent_mixture, run_dir=run_dir)
    lineage = lineage or Lineage(results_dir or os.path.dirname(plan.run_dir))
    plan = prepare_fork(plan, lineage=lineage, copy_checkpoint=copy_checkpoint,
                        runs_jsonl=runs_jsonl)

    result = ForkResult(plan=plan)
    if trainer_factory is None:
        _write_result(result)
        return result

    for leg in plan.legs:
        result.legs[leg.name] = _run_leg(plan, leg, trainer_factory, validate,
                                         lineage)
    result.ran = True

    if plan.mode == "admit" and plan.criterion:
        # No baseline metrics are computed here: the parent's numbers come from
        # the parent's own validator run, which the caller holds. With none, the
        # verdict is honestly undecided rather than a pass against zeros.
        baseline = config.get("baseline_metrics") or {}
        result.verdict = check_admission(
            plan.criterion, result.legs["admit"].metrics, baseline)
    _write_result(result)
    return result


def _run_leg(plan: ForkPlan, leg: ForkLeg, trainer_factory: Callable,
             validate: Callable, lineage: Lineage) -> LegResult:
    trainer = trainer_factory(leg, plan)
    if trainer is None:
        raise ForkError(f"trainer_factory returned None for leg {leg.name!r}")
    got = os.path.abspath(getattr(trainer.args, "output_dir", ""))
    if got != os.path.abspath(leg.output_dir):
        raise ForkError(
            f"leg {leg.name!r}: the trainer writes to {got}, but the leg's run "
            f"directory is {leg.output_dir}. A fork's checkpoints and its record "
            "have to land in the directory its lineage entry names.")
    # Resume entries from the child's own chunk boundaries belong in the same
    # file as the fork entry; the factory does not have to know that.
    if getattr(trainer, "lineage_hook", None) is None:
        trainer.lineage_hook = lineage.hook(child=leg.output_dir)

    out = LegResult(leg=leg)
    scratch = os.path.join(leg.output_dir, "eval_scratch")

    def request_for(stage: str, step: int) -> dict:
        if validate is None:
            return {}
        return dict(validate(ValidationRequest(
            stage=stage, mode=plan.mode, leg=leg.name, step=step,
            trainer=trainer, validators=plan.validators, scratch_dir=scratch,
            plan=plan)) or {})

    if leg.eval_steps and validate is not None:
        os.makedirs(scratch, exist_ok=True)
        trainer.add_callback(_periodic_validation(
            int(leg.eval_steps), request_for,
            lambda step, metrics: out.history.append((step, metrics))))

    if leg.resume:
        # anneal / admit continue the parent run: optimizer moments, RNG, sampler
        # cursor and schedule position all come back, and the discontinuity check
        # of D5.4 writes its own lineage entry through the hook installed above.
        trainer.prepare_resume(leg.start_checkpoint)
        out.train_output = trainer.train(resume_from_checkpoint=leg.start_checkpoint)
    else:
        if leg.start_checkpoint is not None:
            load_start_weights(trainer, leg.start_checkpoint)
        out.train_output = trainer.train()

    out.final_step = int(trainer.state.global_step)
    if validate is not None:
        os.makedirs(scratch, exist_ok=True)
        out.metrics = request_for("end", out.final_step)
        if out.metrics:
            out.history.append((out.final_step, out.metrics))
    out.steps_to_target = steps_to_target(out.history, plan.target)
    return out


def _write_result(result: ForkResult) -> None:
    """``result.json``, beside the immutable ``fork.json``.

    Two files rather than one rewritten file, because ``fork.json`` is the
    record that `PLAN.md` §5 requires to have been written *before* the fork ran
    — an admit criterion that could be rewritten afterwards is not a criterion.
    """
    _write_json(os.path.join(result.plan.run_dir, "result.json"),
                result.to_json())
