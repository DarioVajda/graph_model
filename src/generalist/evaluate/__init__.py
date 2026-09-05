"""
D7 — evaluation as plugins: the protocol, the registry, the context, the runner.

Every number this harness reports about a model comes from a *validator*: a named
object with a cadence, a declared set of things it needs from the run, a declared
set of metric keys it produces, and a version. Nothing about a metric — the
generation config, the answer extraction, the metric implementation — lives in
the trainer, so a metric can be added, versioned or removed without touching the
training loop, and a checkpoint's record says which version of which validator
produced each number (D7.2).

Three rules carry the design, and each of them exists because of something that
has already gone wrong in this repo:

* **A validator that raises is logged and skipped, never fatal.** This is the
  `_per_example` contract from `molecules/train.py`: analysis must not lose a
  training run that already cost GPU-hours. It holds even when the validator is
  nonsense. What is *not* silent is the record — every failure lands in the run's
  status list with its exception text, which is what turned that same try/except
  from a hiding place into a readable one after it concealed a Tier-B defect for
  a whole campaign.
* **A validator that returns a key it did not declare is an error.** Metric names
  are how two runs are compared months apart; a name that drifts silently makes
  every table that joins on it wrong. The runner detects it, marks the validator
  failed and drops its metrics, so a training run survives it and the smoke run
  — which runs with ``strict=True`` — does not.
* **Training runs never select** (D7.4). :func:`check_selection` refuses any
  selection at all outside a fork, and refuses a key mentioning ``test``
  anywhere, in a fork or not.

The module is free of torch, numpy and RDKit at import time: `validate` mode
resolves a config, builds the validator list and checks the cadences on the login
node. Everything heavy is imported inside :meth:`Validator.run`.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, fields
from typing import Protocol, runtime_checkable

__all__ = [
    "CADENCES", "EVENTS", "NEED_ALIASES", "CONTEXT_FIELDS",
    "EvalContext", "EvalError", "EvalNeedsError", "EvalRun", "ValidatorStatus",
    "Validator", "BaseValidator",
    "build_validators", "check_needs", "check_selection", "get", "names",
    "parse_cadence", "protocol_versions", "register", "run_validators",
    "should_run",
]


class EvalError(ValueError):
    """A validator, cadence or selection that cannot be used. Names the cause."""


class EvalNeedsError(EvalError):
    """A validator asked for something the context does not carry.

    Raised by :func:`check_needs` *before* the validator runs, so the failure
    reads "held_out needs eval_sets" rather than an ``AttributeError`` thrown
    from three frames inside a metric.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Cadence
# ─────────────────────────────────────────────────────────────────────────────

#: The cadence forms D7.1 allows. ``steps:<n>`` fires every *n* optimizer steps;
#: ``milestone`` at whatever the run calls a milestone (a fork point, the end of
#: a chunk); ``end`` once, when training stops; ``manual`` never on its own —
#: only when someone names the validator, which is what `eval` mode does.
CADENCES = ("steps:<n>", "milestone", "end", "manual")

#: The events a run raises. One event per call into :func:`run_validators`.
EVENTS = ("step", "milestone", "end", "manual")

_STEPS_PREFIX = "steps:"


def parse_cadence(cadence) -> tuple:
    """``"steps:500"`` -> ``("steps", 500)``; ``"end"`` -> ``("end", None)``."""
    if not isinstance(cadence, str) or not cadence:
        raise EvalError(f"cadence: must be one of {CADENCES}, got {cadence!r}")
    if cadence.startswith(_STEPS_PREFIX):
        rest = cadence[len(_STEPS_PREFIX):]
        try:
            n = int(rest)
        except ValueError:
            raise EvalError(
                f"cadence: {cadence!r} — 'steps:' must be followed by an integer "
                f"number of steps, got {rest!r}") from None
        if n < 1:
            raise EvalError(f"cadence: {cadence!r} — the step interval must be >= 1")
        return "steps", n
    if cadence in ("milestone", "end", "manual"):
        return cadence, None
    raise EvalError(f"cadence: {cadence!r} is not one of {CADENCES}")


def should_run(cadence, step: int, event: str = "step") -> bool:
    """Does a validator with this cadence fire on this (step, event)?

    One place, so the trainer, `eval` mode and the fork cannot disagree about
    when a validator runs.

    ``manual`` is deliberately asymmetric: the *event* ``"manual"`` fires every
    validator regardless of its cadence — someone naming a validator on the
    command line has already decided — while the *cadence* ``"manual"`` fires on
    nothing else. A ``steps:<n>`` validator does not fire at step 0 (the model
    has taken no step and the number would be noise in the middle of a curve)
    and does not fire on a milestone or at the end unless its own multiple lands
    there; ask for that with a second entry rather than by overloading one.
    """
    kind, n = parse_cadence(cadence)
    if event not in EVENTS:
        raise EvalError(f"event: {event!r} is not one of {EVENTS}")
    if event == "manual":
        return True
    if kind == "steps":
        return event == "step" and int(step) > 0 and int(step) % n == 0
    if kind == "manual":
        return False
    return kind == event


# ─────────────────────────────────────────────────────────────────────────────
# The context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalContext:
    """Everything a validator may ask for, and nothing it may mutate.

    Frozen because a validator is a *measurement*: one that reassigned
    ``ctx.model`` or swapped an eval set would change the run it is reporting
    on. The mutable objects inside it (the model, the sampler) are of course
    still mutable — that is unavoidable — but the shape is fixed here and the
    trainer is wired to it, so a validator added later cannot quietly require a
    new field.

    ``step`` is the only required field. Everything else is optional and a
    validator declares what it actually needs in :attr:`Validator.needs`; the
    runner then refuses to call it with a field missing rather than letting an
    ``AttributeError`` surface from inside a metric.

    ``eval_sets`` is ``{task_name: {split: TaskSource}}`` — the adapter's own
    :class:`adapters.TaskSource` objects, so a validator gets ``__len__``,
    ``__getitem__`` (an ``Example.to_item()`` dict, sidecar included),
    ``lengths()`` and the identity fields. It is *not* a dict of tensors: a
    validator that wants to generate needs the untokenized text and the schema
    sidecar, and one that wants the per-example report needs the graph columns.

    ``config`` is the run's config as a plain dict. It is the one open field, and
    it is open on purpose: a validator that needs something structural from the
    run (``max_spd`` for the geometry report, ``active_params`` for the bias
    norm, the trainer's per-task loss closure for the gradient share) reads it
    from here under a documented key rather than growing a field that only one
    validator uses. The keys the built-ins read are named as constants in
    `evaluate/builtin.py`.
    """

    #: Optimizer step this evaluation describes.
    step: int
    model: object = None
    tokenizer: object = None
    registry: object = None
    mixture: object = None
    arm: str | None = None
    #: ``(segment index, step within segment)`` from `schedule.Schedule.position`.
    schedule_position: tuple | None = None
    eval_sets: dict | None = None
    train_sampler: object = None
    base_model_name: str | None = None
    collator: object = None
    device: object = None
    scratch_dir: str | None = None
    config: dict = field(default_factory=dict)

    def require(self, need: str):
        """The field ``need`` names, or :class:`EvalNeedsError` naming it."""
        name = NEED_ALIASES.get(need, need)
        if name not in CONTEXT_FIELDS:
            raise EvalError(
                f"needs: {need!r} is not an EvalContext field; have "
                f"{sorted(CONTEXT_FIELDS)}")
        value = getattr(self, name)
        if value is None or (name == "eval_sets" and not value):
            raise EvalNeedsError(
                f"the context has no {name!r}; this evaluation cannot run "
                "without it")
        return value

    def sources(self, task: str) -> dict:
        """``{split: TaskSource}`` for one task, empty when it was not built."""
        return dict((self.eval_sets or {}).get(task, {}) or {})


#: Every field name a validator may put in ``needs``.
CONTEXT_FIELDS = frozenset(f.name for f in fields(EvalContext))

#: D7.1 spells one need ``"base_model"``; the context field that carries it is
#: ``base_model_name`` (the harness never holds a second model resident — a
#: validator that wants the base weights loads them itself, once, inside `run`).
NEED_ALIASES = {"base_model": "base_model_name"}


def check_needs(validator, ctx: EvalContext) -> None:
    """Raise :class:`EvalNeedsError` naming the validator and the first field missing."""
    for need in sorted(getattr(validator, "needs", ()) or ()):
        try:
            ctx.require(need)
        except EvalNeedsError as exc:
            raise EvalNeedsError(
                f"{getattr(validator, 'name', validator)}: needs "
                f"{NEED_ALIASES.get(need, need)!r}, but {exc}") from None


# ─────────────────────────────────────────────────────────────────────────────
# The protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Validator(Protocol):
    """D7.1. What the runner requires of anything it will call.

    ``keys(ctx)`` is the addition to D7.1's four attributes, and it is what makes
    "a validator that returns an undeclared key fails the smoke" checkable rather
    than aspirational. It returns the **leaf** metric names — the last path
    segment of every key ``run`` will return. Leaves rather than whole keys
    because the middle of a key is data (a task name, an endpoint, a split) and
    an eval set with one more endpoint in it is not a protocol change; the leaf
    is the name a table joins on and the thing that must not drift.

    ``run(ctx)`` returns a flat dict. Keys may carry a task and any further
    qualification as a ``/``-joined path (``"mol/tox21/endpoint:NR-AR/roc_auc"``);
    the runner prefixes the validator's own name, giving D7.1's
    ``<validator>/<task>/<metric>``. Values are floats, strings or lists —
    anything that survives a round trip through the run record's JSON.
    """

    name: str
    cadence: str
    needs: frozenset
    protocol_version: str

    def keys(self, ctx: EvalContext | None = None) -> set: ...

    def run(self, ctx: EvalContext) -> dict: ...


class BaseValidator:
    """Boilerplate every built-in shares: cadence override, need checking, repr.

    Validators are *instances*, built once per run from the config's list and
    reused across every firing, so a validator may hold state between calls
    (`throughput` measures wall-clock between two of them and could not
    otherwise). That also means an instance belongs to one run and is never
    shared between two.
    """

    name = "validator"
    cadence = "manual"
    needs = frozenset()
    protocol_version = "1"

    #: ``{option: why}`` for options this validator will not honour. A cost or
    #: cadence knob that is read by some validators and ignored by others is a
    #: config that lies: the smoke set capped ``per_example`` at
    #: ``max_samples: 32`` and got a report over all 1000 rows, and nothing said
    #: so. Refusing at build time — `validate` mode, on the login node — is the
    #: only place the answer is free.
    REJECTED_OPTIONS: dict = {}

    def __init__(self, cadence: str | None = None, **options):
        # Options first: `needs` may be a property that consults them.
        self.options = dict(options)
        for key, why in self.REJECTED_OPTIONS.items():
            if key in self.options:
                raise EvalError(f"{self.name}: {key!r} is not honoured here — {why}")
        if cadence is not None:
            self.cadence = cadence
        parse_cadence(self.cadence)                      # fail at build, not at step 500
        for need in self.needs:
            if NEED_ALIASES.get(need, need) not in CONTEXT_FIELDS:
                raise EvalError(
                    f"{self.name}: needs {need!r}, which is not an EvalContext "
                    f"field; have {sorted(CONTEXT_FIELDS)}")

    def option(self, key, default=None):
        return self.options.get(key, default)

    def keys(self, ctx: EvalContext | None = None) -> set:
        raise NotImplementedError(f"{self.name}: must declare its metric keys")

    def run(self, ctx: EvalContext) -> dict:
        raise NotImplementedError(f"{self.name}: must implement run(ctx)")

    def describe(self) -> dict:
        return {"name": self.name, "cadence": self.cadence,
                "protocol_version": self.protocol_version,
                "needs": sorted(self.needs), "options": dict(self.options)}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} cadence={self.cadence}>"


# ─────────────────────────────────────────────────────────────────────────────
# The registry of validators
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATORS: dict = {}


def register(cls, *, replace: bool = False):
    """Register a validator *class* under its ``name``. Returns it, so it decorates."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise EvalError(f"register: {cls!r} has no name")
    if name in _VALIDATORS and not replace:
        raise EvalError(f"{name}: already registered as {_VALIDATORS[name]!r}")
    _VALIDATORS[name] = cls
    return cls


def get(name: str):
    try:
        return _VALIDATORS[name]
    except KeyError:
        raise EvalError(f"{name}: no such validator (have {names()})") from None


def names() -> list:
    return sorted(_VALIDATORS)


def build_validators(specs) -> tuple:
    """The config's ``"validators": [...]`` list -> instances, in config order.

    Each entry is ``{"name": ..., "cadence": ...}`` plus whatever options that
    validator takes. A bad name or a bad cadence fails here — at `validate`, on
    the login node, before a GPU is allocated — which is the whole reason this
    list is resolved eagerly rather than at the first firing.
    """
    out = []
    seen = set()
    for i, spec in enumerate(specs or ()):
        if isinstance(spec, str):
            spec = {"name": spec}
        if not isinstance(spec, dict):
            raise EvalError(
                f"validators[{i}]: must be a name or a dict with a 'name', got "
                f"{spec!r}")
        spec = dict(spec)
        name = spec.pop("name", None)
        if not name:
            raise EvalError(f"validators[{i}]: no 'name'")
        if name in seen:
            raise EvalError(f"{name}: listed twice in 'validators'")
        seen.add(name)
        out.append(get(name)(**spec))
    return tuple(out)


def protocol_versions(validators) -> list:
    """The ``(validator, version)`` pairs D7.2 puts in ``state.json``.

    Over the *configured* set, not the set that happened to fire: a checkpoint's
    record has to say which protocol the run was evaluating under, and a
    validator whose cadence had not come round yet is still part of it.
    """
    return sorted({(v.name, str(v.protocol_version)) for v in validators})


def _release_eval_memory() -> None:
    """Hand a validator's CUDA blocks back to the allocator.

    Called after every validator, success or failure. It is a no-op without
    torch or without a live CUDA context, and it never raises: this runs on the
    path whose entire purpose is that measurement cannot lose a run.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                               # noqa: BLE001
        pass


# ─────────────────────────────────────────────────────────────────────────────
# The runner
# ─────────────────────────────────────────────────────────────────────────────

#: What a validator's metric values may be. Anything else would not survive the
#: run record's JSON round trip, and a dict would silently flatten into keys
#: nothing declared.
_VALUE_TYPES = (bool, int, float, str, list)


@dataclass(frozen=True)
class ValidatorStatus:
    """What happened to one validator on one firing.

    ``state`` is ``"ran"``, ``"not_due"`` (the cadence did not fire) or
    ``"error"`` (it raised, or it returned something the protocol does not
    allow — either way its metrics were dropped and the run continued).
    """

    name: str
    protocol_version: str
    state: str
    message: str = ""
    duration_s: float = 0.0
    n_metrics: int = 0

    @property
    def ok(self) -> bool:
        return self.state != "error"


@dataclass(frozen=True)
class EvalRun:
    """One firing of a validator set: the metrics, and what happened to each."""

    step: int
    event: str
    metrics: dict
    statuses: tuple
    versions: tuple

    def errors(self) -> tuple:
        return tuple(s for s in self.statuses if s.state == "error")

    def ran(self) -> tuple:
        return tuple(s for s in self.statuses if s.state == "ran")

    def status(self, name: str):
        for s in self.statuses:
            if s.name == name:
                return s
        raise KeyError(name)

    def raise_for_errors(self) -> "EvalRun":
        """Turn skipped validators into a failure. The smoke run (T10) calls this.

        A training run must not; that is the whole point of the skip.
        """
        bad = self.errors()
        if bad:
            raise EvalError(
                "validators failed at step "
                f"{self.step}: " + "; ".join(f"{s.name}: {s.message}" for s in bad))
        return self

    def record(self) -> dict:
        """The run record's evaluation fragment."""
        return {
            "step": self.step,
            "event": self.event,
            "metrics": dict(self.metrics),
            "protocol_versions": [list(v) for v in self.versions],
            "statuses": [
                {"name": s.name, "state": s.state, "message": s.message,
                 "duration_s": round(s.duration_s, 4), "n_metrics": s.n_metrics,
                 "protocol_version": s.protocol_version}
                for s in self.statuses
            ],
        }


def run_validators(ctx: EvalContext, validators, event: str = "step",
                   only=None, strict: bool = False) -> EvalRun:
    """Fire every validator whose cadence is due, and collect what they return.

    ``only`` restricts the set by name (what `eval` mode passes); ``strict``
    re-raises instead of skipping, which is for tests and the plumbing smoke —
    never for a training run.

    The contract, in order, per validator:

    1. cadence (:func:`should_run`) — otherwise ``not_due``;
    2. needs (:func:`check_needs`) — a missing field is an error *about the
       wiring*, reported as such, and the other validators still run;
    3. ``run(ctx)`` — any exception is caught, printed with its traceback and
       recorded;
    4. the returned keys are checked against ``keys(ctx)``; an undeclared leaf,
       a non-dict return or a value of an unsupported type fails the validator
       and drops **all** of its metrics, because a validator that returned a name
       nobody declared cannot be trusted about the ones it did.

    Namespacing is the last step, so a validator never sees its own prefix and
    two validators can return the same inner key without colliding.
    """
    if event not in EVENTS:
        raise EvalError(f"event: {event!r} is not one of {EVENTS}")
    validators = tuple(validators)
    wanted = None if only is None else set(only)

    metrics, statuses = {}, []
    for validator in validators:
        version = str(getattr(validator, "protocol_version", "?"))
        if wanted is not None and validator.name not in wanted:
            statuses.append(ValidatorStatus(validator.name, version, "not_due",
                                            "not in the requested set"))
            continue
        if not should_run(validator.cadence, ctx.step, event):
            statuses.append(ValidatorStatus(
                validator.name, version, "not_due",
                f"cadence {validator.cadence} does not fire on {event} at step {ctx.step}"))
            continue

        started = time.monotonic()
        try:
            check_needs(validator, ctx)
            out = validator.run(ctx)
            namespaced = _check_output(validator, out, ctx)
        except Exception as exc:                                    # noqa: BLE001
            if strict:
                raise
            # The `_per_example` contract: measurement never loses a run that
            # already cost GPU-hours. The traceback goes to the log so the
            # failure is diagnosable; the message goes to the record so it is
            # not merely absent.
            print(f"[eval] {validator.name} failed at step {ctx.step} "
                  f"({type(exc).__name__}: {exc}); the run is unaffected.")
            traceback.print_exc()
            statuses.append(ValidatorStatus(
                validator.name, version, "error", f"{type(exc).__name__}: {exc}",
                time.monotonic() - started))
            continue
        finally:
            # Without this the contract above is false for the one failure that
            # matters most. A scoring validator that dies of CUDA OOM leaves its
            # reserved blocks on the card, and the next training step allocates
            # into what is left — the 2026-09-04 shakedown lost the run four
            # steps after `in_mixture` had been caught and skipped, with 8 GB
            # free on a 178 GB card. Releasing after every validator, not only
            # after a failing one, because a validator that succeeded has still
            # just held the largest transient buffers in the run.
            _release_eval_memory()

        metrics.update(namespaced)
        statuses.append(ValidatorStatus(
            validator.name, version, "ran", "", time.monotonic() - started,
            len(namespaced)))

    return EvalRun(step=int(ctx.step), event=event, metrics=metrics,
                   statuses=tuple(statuses),
                   versions=tuple(tuple(p) for p in protocol_versions(validators)))


def _check_output(validator, out, ctx) -> dict:
    """D7.1's undeclared-key rule, and the namespacing. Raises on a violation."""
    if not isinstance(out, dict):
        raise EvalError(
            f"{validator.name}: run(ctx) returned {type(out).__name__}, not a dict "
            "of metrics")
    declared = set(validator.keys(ctx))
    if not declared:
        raise EvalError(
            f"{validator.name}: declares no metric keys, so nothing it returns "
            "can be checked")

    namespaced = {}
    for key, value in out.items():
        if not isinstance(key, str) or not key:
            raise EvalError(f"{validator.name}: metric key {key!r} is not a string")
        leaf = key.rsplit("/", 1)[-1]
        if leaf not in declared:
            raise EvalError(
                f"{validator.name}: returned {key!r}, whose metric name {leaf!r} is "
                f"not declared in keys() ({sorted(declared)}). Metric names are how "
                "two runs are compared; declare it or do not return it.")
        if isinstance(value, _VALUE_TYPES):
            namespaced[f"{validator.name}/{key}"] = value
        else:
            raise EvalError(
                f"{validator.name}: metric {key!r} is a "
                f"{type(value).__name__}; validators return floats, strings or "
                "lists so the run record round-trips through JSON")
    return namespaced


# ─────────────────────────────────────────────────────────────────────────────
# D7.4 — selection
# ─────────────────────────────────────────────────────────────────────────────

#: Separators a metric key is split on before the ``test`` check. A key is a path
#: of names joined by these, and it is a *name* equal to ``test`` that is refused
#: — not the four letters, which would also refuse ``latest``.
_KEY_SEPARATORS = "/_-.:"


def check_selection(selection, *, mode: str = "train"):
    """Refuse a selection this run is not allowed to make. Returns it otherwise.

    Two refusals, both from D7.4:

    * **Training runs do not select.** Selection is a fork's job (D6): the
      annealed checkpoint is the reportable model, and Tier-B val has been
      measured to *anti-rank* the two arms on BBBP (`molecules/PLAN.md` §8.4), so
      a best-val checkpoint here would be chosen by an instrument shown to be
      near-blind. ``mode`` is the fork mode (``anneal`` / ``admit`` / ``adapt``)
      when a fork is asking.
    * **No selection key may mention ``test``.** A run that picks a checkpoint on
      test has reported the maximum of a noisy quantity as if it were an estimate
      of it. The key is split into names and a name equal to ``test`` is refused
      wherever it sits, so ``eval/mol/bace/test/roc_auc`` and ``test_roc_auc``
      are both caught and ``latest_loss`` is not.
    """
    if selection is None:
        return None
    if not isinstance(selection, dict) or "metric" not in selection:
        raise EvalError(
            f"selection: must be a dict with a 'metric' (and optionally a "
            f"'split'), got {selection!r}")
    if mode == "train":
        raise EvalError(
            "selection: a training run does not select — the annealed fork is "
            "the reportable model (D7.4), and Tier-B val anti-ranks the arms. "
            "Declare the selection on the fork that needs it.")

    for label in ("metric", "split"):
        value = selection.get(label)
        if value is None:
            continue
        if not isinstance(value, str):
            raise EvalError(f"selection: {label!r} must be a string, got {value!r}")
        names_in = [value]
        for sep in _KEY_SEPARATORS:
            names_in = [part for chunk in names_in for part in chunk.split(sep)]
        if any(part.lower() == "test" for part in names_in):
            raise EvalError(
                f"selection: {label} {value!r} names the test split. Selecting on "
                "test reports the maximum of a noisy quantity as an estimate of "
                "it; select on val or do not select.")
    return dict(selection)


# The built-ins register themselves on import. Bottom of the file so the module's
# protocol and registry are fully defined by the time they do.
from . import builtin as _builtin        # noqa: E402,F401  (import for side effect)
