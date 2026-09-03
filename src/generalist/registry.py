"""
D2 — what is in the mixture, at what weight, from which build.

One :class:`TaskSpec` per task, held in a :class:`Registry`, which is the only
place that answers those three questions. The registry's :meth:`Registry.snapshot`
goes into the run record and into every checkpoint's ``state.json``, and its
:meth:`Registry.hash` is what a resume compares to decide whether the mixture
changed (D5.4).

:func:`resolve` turns a config's task list into a :class:`Mixture`: normalised
shares, the budget in examples, and the step count at the configured
tokens-per-step. Two things it refuses, both of which have already cost this repo
a run:

* a **held-out** task in a training mixture. `MOLECULE_GENERALIST.md` §4 holds out
  ``bond_path``, ``longest_chain`` and ClinTox; the molecules package refuses to
  *build* the first and third without ``--held-out-eval``. Enforcement in two
  places is deliberate (D2.1) — a mixture config that names one fails at
  ``validate``, before any data is built.
* a **share that contributes nothing**. A task whose share rounds to fewer than
  one example per 1000 steps is in the config, in the report, and absent from the
  gradient. That is the ``--magnetic-groups`` class of bug (`PLAN.md` §10) and it
  is caught here rather than read off a loss curve weeks later.

No torch: ``validate`` mode resolves a whole config on the login node.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .schema import ANSWER_KINDS, SCHEMA_VERSION

KINDS = ("corpus", "generator")
LOSS_NORMS = ("per_example", "per_token")

#: D2.2. A task must be worth at least one example per this many steps, or the
#: mixture is refused. 1000 steps is the smallest horizon over which "this task
#: is training" is a claim anyone would make about a run.
MIN_EXAMPLES_PER = 1000


class RegistryError(ValueError):
    """A registry or mixture that cannot be resolved. The message names the cause."""


# ─────────────────────────────────────────────────────────────────────────────
# TaskSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    """One task: identity, how it draws, how it is scored, and what built it.

    ``weight``, ``passes`` and ``cap_per_pass`` are *defaults*; a mixture entry
    overrides them, which is what lets one registry serve several configs.

    Three fields are filled in by the adapter at build time rather than written
    in a config, because they are properties of the built data and not knobs:

    ``train_size``  examples in the task's train split (a corpus's pass cap is
                    ``passes × train_size``, which is what bounds the budget).
                    For a held-out task it is the size of its ``held_out`` split:
                    that split is what an ``adapt`` fork trains on, and it is the
                    only run that ever trains one.
    ``mean_tokens`` mean rendered length in tokens, which is what turns
                    ``tokens_per_step`` into examples per step (D4.4). A
                    captioning task and a one-token task cannot share a number
                    here, so it cannot be a constant.
    ``build_version`` hash of the adapter's inputs (D3.2). A spec whose
                    ``build_version`` differs from the one a checkpoint trained
                    on is a mixture change: legal, but it forces a re-warm and a
                    lineage entry.

    ``verify`` is the ``(prediction, example) -> bool`` free-label check of
    `PLAN.md` §3.2. It is stored but never serialised — :meth:`Registry.snapshot`
    records only whether one exists, so the registry hash stays stable across
    two processes that built the same callable.
    """

    name: str
    domain: str
    adapter: str
    kind: str = "corpus"
    answer_kind: str = "token"
    held_out: bool = False
    weight: float = 1.0
    loss_norm: str = "per_example"
    passes: int = 1
    cap_per_pass: int | None = None
    metric: str = "exact_match"
    verify: Callable | None = None
    max_new_tokens: int | None = None
    build_version: str = "unset"
    eval_splits: tuple = ("val", "test")
    #: Mean rendered length in tokens; set by the adapter after build.
    mean_tokens: float | None = None
    #: Examples in the train split; set by the adapter after build.
    train_size: int | None = None
    #: The question text (or its template) this task is routed by. Recorded so a
    #: run record says what the model was actually shown, since the question is
    #: the *only* thing that tells it which task it is on.
    question_template: str | None = None

    def __post_init__(self):
        if not self.name:
            raise RegistryError("name: a task spec needs a name")
        if self.kind not in KINDS:
            raise RegistryError(
                f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.answer_kind not in ANSWER_KINDS:
            raise RegistryError(
                f"{self.name}: answer_kind must be one of {ANSWER_KINDS}, got "
                f"{self.answer_kind!r}")
        if self.loss_norm not in LOSS_NORMS:
            raise RegistryError(
                f"{self.name}: loss_norm must be one of {LOSS_NORMS}, got "
                f"{self.loss_norm!r}")
        if self.weight < 0:
            raise RegistryError(f"{self.name}: weight must be >= 0, got {self.weight}")
        if self.passes < 1:
            raise RegistryError(f"{self.name}: passes must be >= 1, got {self.passes}")
        self.eval_splits = tuple(self.eval_splits)

    def snapshot(self) -> dict:
        """JSON-serialisable, callable-free view of this spec."""
        out = {k: v for k, v in asdict(self).items() if k != "verify"}
        out["eval_splits"] = list(self.eval_splits)
        out["has_verify"] = self.verify is not None
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class Registry:
    """The task table. Insertion order never reaches the snapshot or the hash."""

    def __init__(self, specs: Iterable[TaskSpec] = ()):
        self._specs: dict[str, TaskSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: TaskSpec) -> TaskSpec:
        if not isinstance(spec, TaskSpec):
            raise RegistryError(f"register: expected a TaskSpec, got {spec!r}")
        if spec.name in self._specs:
            raise RegistryError(f"{spec.name}: already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> TaskSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise RegistryError(
                f"{name}: not registered (have {self.names()})") from None

    def names(self) -> list:
        return sorted(self._specs)

    def __contains__(self, name) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return (self._specs[n] for n in self.names())

    def snapshot(self) -> dict:
        """The serialised registry state, in a deterministic key order.

        Callables are excluded (a bound method's identity differs between
        processes and would make the hash useless); ``has_verify`` records that
        one was configured, which is the part a run record needs.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "tasks": {name: self._specs[name].snapshot() for name in self.names()},
        }

    def hash(self) -> str:
        return _hash(self.snapshot())


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Held-out enforcement
# ─────────────────────────────────────────────────────────────────────────────

#: Prefix the molecules adapter registers its tasks under.
MOLECULE_PREFIX = "mol/"


def molecule_held_out_names() -> set:
    """The molecules package's held-out declaration, as registry names.

    The authoritative list lives in `molecules/data.py` (``HELD_OUT_TIER_A_TASKS``,
    ``HELD_OUT_DATASETS``) — this reads it rather than restating it, so the two
    enforcement points of D2.1 can never disagree about ``bond_path`` or ClinTox.

    ``longest_chain`` joined the tuples on 2026-09-02 for `MOLECULE_GENERALIST.md`
    §4, so all three holdouts come through here. A spec may still carry
    ``held_out=True`` on its own — :func:`resolve` refuses a task if *either*
    source says held out, which is what lets a task be held out for one campaign
    without amending the molecules package.

    Imported lazily: `molecules/data.py` pulls RDKit and networkx, and this module
    is imported by everything.
    """
    from ..experiments.molecules.data import HELD_OUT_DATASETS, HELD_OUT_TIER_A_TASKS

    return {f"{MOLECULE_PREFIX}{name}"
            for name in tuple(HELD_OUT_TIER_A_TASKS) + tuple(HELD_OUT_DATASETS)}


def is_held_out(spec: TaskSpec) -> bool:
    """True if either enforcement source says this task never trains."""
    if spec.held_out:
        return True
    return spec.name in molecule_held_out_names()


# ─────────────────────────────────────────────────────────────────────────────
# D2.2 — resolving a mixture
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MixtureEntry:
    """One resolved task line: what it draws and how much of it the run sees."""

    name: str
    kind: str
    weight: float
    share: float
    passes: int
    cap_per_pass: int | None
    mean_tokens: float
    #: Corpus: ``passes × train_size``, the most this task may contribute before
    #: it would repeat past its pass cap. Generators are unbounded (D4.2 draws a
    #: fresh pass), so ``None``.
    available: int | None
    examples: int


@dataclass(frozen=True)
class Mixture:
    """A resolved mixture: shares, budget, steps, and the per-task example counts."""

    entries: tuple
    shares: dict
    budget_examples: int
    examples_per_step: float
    steps: int
    per_task_examples: dict
    tokens_per_step: int
    mean_tokens: float
    binding_task: str
    registry_hash: str

    def hash(self) -> str:
        """The ``mixture hash`` D5.4 compares across a resume."""
        return _hash({
            "registry_hash": self.registry_hash,
            "tokens_per_step": self.tokens_per_step,
            "entries": [
                {"name": e.name, "weight": e.weight, "passes": e.passes,
                 "cap_per_pass": e.cap_per_pass}
                for e in sorted(self.entries, key=lambda e: e.name)
            ],
        })

    def table(self) -> str:
        """The mixture table ``validate`` mode prints."""
        rows = ["  task                       share    examples   passes",
                "  " + "-" * 54]
        for e in sorted(self.entries, key=lambda e: -e.share):
            rows.append(f"  {e.name:<24} {e.share:7.4f} {e.examples:>10d} "
                        f"{e.passes:>8d}")
        rows.append(f"  budget {self.budget_examples} examples over {self.steps} "
                    f"steps at {self.tokens_per_step} tokens/step "
                    f"(bound by {self.binding_task})")
        return "\n".join(rows)


def resolve(registry: Registry, mixture, tokens_per_step: int,
            steps: int | None = None,
            min_examples_per: int = MIN_EXAMPLES_PER,
            allow_held_out=()) -> Mixture:
    """Config task list -> :class:`Mixture`.

    ``mixture`` is a list of ``{"name", "weight", "passes"?, "cap_per_pass"?}``;
    the optional keys override the spec's defaults.

    ``allow_held_out`` names the held-out tasks this particular resolve may
    admit. It exists for exactly one caller: D6's ``adapt`` fork, which trains on
    one held-out task on purpose to measure adaptation efficiency, and which is
    safe because the fork is a leaf — it never merges back and the parent never
    sees the task. The default is empty, so every other path keeps D2.1's
    unconditional refusal. Naming the exception in the API is better than the
    alternative of resolving against an aliased shadow registry at the call site:
    a sanctioned exception that has to be smuggled past a check stops looking
    like a sanctioned exception the second time someone needs one.

    ``steps`` overrides the budget rule: the budget becomes
    ``ceil(steps x examples_per_step)`` and the "no corpus task" refusal is
    skipped, since a step count already bounds the run. This is for a smoke run —
    a fixed 200 steps over three tasks, one of which may be a generator — where
    the finite sources' pass caps are not the thing being tested and a 12000-example
    budget would be the whole point of the run rather than a warm-up. A corpus may
    then be asked for more than ``passes x train_size``; the sampler retires it and
    logs, which is the behaviour a smoke run wants rather than a refusal.
    ``min_examples_per`` lowers (or, at ``0``, disables) the sub-threshold floor for
    the same reason: over 200 steps a task that would train on nothing over 1000 is
    not the bug that check is looking for.

    **The budget is set by the finite sources.** For each corpus task,
    ``passes × train_size`` is the most it may contribute before it starts
    repeating past its pass cap. Since a task's example count is
    ``share × budget``, the largest budget under which *no* corpus exceeds its
    cap is ``min over corpora of available / share`` — the corpus with the
    smallest ratio binds, and every other one lands at or under its cap.
    Generators never bound it: D4.2 draws a fresh pass from the train-role pool
    each time, so they have no cap to exceed. This is `MOLECULE_GENERALIST.md`
    §2's rule — "the total budget is defined by the finite sources" — and it is
    why the step count is computed and recorded rather than configured.

    ``examples_per_step`` follows from D4.4: ``tokens_per_step`` fixes the
    effective batch in tokens, so the number of examples in a step is that
    budget divided by the share-weighted mean example length. A mixture with a
    captioning task in it has fewer examples per step than one without, which is
    the whole reason ``batch_size`` is derived rather than configured.
    """
    entries_in = _normalise_entries(mixture)
    if tokens_per_step is None or tokens_per_step <= 0:
        raise RegistryError(
            f"tokens_per_step: must be a positive int, got {tokens_per_step!r}")

    permitted = set(allow_held_out or ())
    specs = {}
    for name, _entry in entries_in.items():
        spec = registry.get(name)              # RegistryError if unregistered
        if is_held_out(spec) and name not in permitted:
            raise RegistryError(
                f"{name}: held out and must never enter a training mixture "
                f"(spec.held_out={spec.held_out}, molecules declaration="
                f"{name in molecule_held_out_names()}). Evaluate it with the "
                f"'held_out' validator instead.")
        specs[name] = spec

    weights = {}
    for name, entry in entries_in.items():
        weight = float(entry.get("weight", specs[name].weight))
        if not math.isfinite(weight) or weight <= 0:
            raise RegistryError(
                f"{name}: weight must be a positive finite number, got {weight!r}")
        weights[name] = weight
    total_weight = sum(weights.values())
    shares = {name: w / total_weight for name, w in weights.items()}

    mean_tokens = {}
    for name, spec in specs.items():
        if spec.mean_tokens is None or spec.mean_tokens <= 0:
            raise RegistryError(
                f"{name}: mean_tokens is {spec.mean_tokens!r}; the adapter sets it "
                "after build and resolve needs it to turn tokens_per_step into "
                "examples per step")
        mean_tokens[name] = float(spec.mean_tokens)

    passes, caps, available = {}, {}, {}
    for name, spec in specs.items():
        entry = entries_in[name]
        p = int(entry.get("passes", spec.passes))
        if p < 1:
            raise RegistryError(f"{name}: passes must be >= 1, got {p}")
        passes[name] = p
        caps[name] = entry.get("cap_per_pass", spec.cap_per_pass)
        if spec.kind == "corpus":
            if spec.train_size is None or spec.train_size <= 0:
                raise RegistryError(
                    f"{name}: train_size is {spec.train_size!r}; a corpus task's "
                    "pass cap is passes x train_size and the budget is computed "
                    "from it")
            available[name] = p * int(spec.train_size)
        else:
            available[name] = None

    weighted_mean_tokens = sum(shares[n] * mean_tokens[n] for n in shares)
    examples_per_step = tokens_per_step / weighted_mean_tokens

    finite = {n: a for n, a in available.items() if a is not None}
    if steps is not None:
        steps = int(steps)
        if steps < 1:
            raise RegistryError(f"steps: must be a positive int, got {steps!r}")
        budget_examples = int(math.ceil(steps * examples_per_step))
        binding_task = f"steps={steps}"
    else:
        if not finite:
            raise RegistryError(
                "budget: the mixture has no corpus task, so nothing bounds it. "
                "Generators draw a fresh pass every time (D4.2) and never set a "
                "budget; add a corpus task or configure a step count explicitly.")
        binding_task = min(finite, key=lambda n: finite[n] / shares[n])
        budget_examples = int(math.floor(finite[binding_task] / shares[binding_task]))
        if budget_examples < 1:
            raise RegistryError(
                f"budget: resolves to {budget_examples} examples; {binding_task} has "
                f"{finite[binding_task]} available at share {shares[binding_task]:.6f}")
        steps = int(math.floor(budget_examples / examples_per_step))
        if steps < 1:
            raise RegistryError(
                f"budget: {budget_examples} examples at {examples_per_step:.2f} "
                f"examples/step is under one step; lower tokens_per_step "
                f"({tokens_per_step}) or widen the mixture")

    if min_examples_per:
        for name, share in shares.items():
            # D2.2: a task in the config, in the report, and absent from the gradient.
            if share * examples_per_step * min_examples_per < 1.0:
                raise RegistryError(
                    f"{name}: share {share:.3e} is under one example per "
                    f"{min_examples_per} steps at {examples_per_step:.2f} "
                    "examples/step, so the task would train on nothing while still "
                    "appearing in the mixture. Raise its weight or drop it.")

    per_task_examples = {n: int(math.floor(shares[n] * budget_examples))
                         for n in shares}
    entries = tuple(
        MixtureEntry(
            name=n, kind=specs[n].kind, weight=weights[n], share=shares[n],
            passes=passes[n], cap_per_pass=caps[n], mean_tokens=mean_tokens[n],
            available=available[n], examples=per_task_examples[n],
        )
        for n in sorted(shares)
    )
    return Mixture(
        entries=entries, shares=shares, budget_examples=budget_examples,
        examples_per_step=examples_per_step, steps=steps,
        per_task_examples=per_task_examples, tokens_per_step=int(tokens_per_step),
        mean_tokens=weighted_mean_tokens, binding_task=binding_task,
        registry_hash=registry.hash(),
    )


def _normalise_entries(mixture) -> dict:
    """``[{"name": ..., "weight": ...}, ...]`` -> ``{name: entry}``, checked."""
    if not mixture:
        raise RegistryError("mixture: empty; a run needs at least one task")
    out = {}
    for i, entry in enumerate(mixture):
        if not isinstance(entry, dict):
            raise RegistryError(
                f"mixture[{i}]: must be a dict with a 'name', got {entry!r}")
        name = entry.get("name")
        if not name:
            raise RegistryError(f"mixture[{i}]: no 'name'")
        if name in out:
            raise RegistryError(f"{name}: listed twice in the mixture")
        out[name] = entry
    return out
