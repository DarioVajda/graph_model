"""
D4 — weights to a draw plan, a resumable sampler, mixed batches, and the two-level
loss accounting.

Four things live here, and each one exists because of a specific way a mixture run
goes wrong:

* **The draw plan is a pure function of** ``(mixture hash, seed, step)`` (D4.1).
  A run restored at step *k* has to draw at *k + 1* exactly what an uninterrupted
  run would have drawn, or a resume quietly changes the data order and every
  before/after comparison across a chunk boundary is noise. So the sampler holds
  no state that is not in :meth:`MixtureSampler.state_dict` — per-step counts come
  from a step-indexed random stream, and the only carried state is a cursor and a
  pass id per task.
* **Generators refresh per pass** (D4.2). A generator task's source is re-requested
  at every pass boundary through the trainer-supplied ``get_source``; a corpus task
  gets a fresh permutation per pass and stops at ``passes``.
* **Batches are mixed, and keyed by the shape the collator builds** (D4.3,
  `GRAPH_GENERALIST.md` §3). Homogeneous batches would make per-task gradient
  noise a function of task share, which is exactly the quantity the
  mixture-weight readout is trying to measure. Examples are bucketed by their
  *padded* ``(node count, token length)`` — a task-agnostic key, and the one the
  compiled kernels guard on — and dealt round-robin within their bucket, so a
  micro-batch is homogeneous only when its bucket is. Every rank runs the same
  bucket at the same micro-step, and a step comes out as exactly
  ``accumulation_steps`` micro-batches per rank.
* **A run never silently runs out of data.** :meth:`MixtureSampler.check_supply`
  replays the draw plan to the end of the run before the first step and refuses a
  corpus that would retire part-way, or a generator pass that was never built.
* **Two-level normalisation** (D4.3/D7a). Each example's loss is divided by its own
  loss-span length; the batch loss is the mean over the examples of the *optimizer
  step*, not of the micro-batch. Dividing by the micro-batch count instead is the
  standard accumulation footgun: it makes a task's gradient share depend on how the
  step happened to be chopped up. :class:`MixtureLoss` takes the step's example
  count and T3 pins that it is invariant to accumulation and to rank count.

The sampler talks to data only through two protocols, so it is testable without a
tokenizer or a real dataset:

``TaskSource``   ``__len__``, ``__getitem__(i) -> item dict`` (a ``TextGraphDataset``
                 item with the schema sidecar and ``ds_label == task``),
                 ``lengths() -> (num_nodes, num_tokens)``, and the attributes
                 ``task``, ``split``, ``arm``, ``pass_id``.
``get_source``   ``(task, pass_id) -> TaskSource``, supplied by the trainer; it wraps
                 ``adapter.load(task, "train", arm, pass_id)`` and caches on disk so
                 a resume does not regenerate a pass.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from typing import Callable, NamedTuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Keys :class:`MixtureDataset` stows on each item and :func:`wrap_collator` strips
#: before the base collator sees them. ``GraphCollatorV2`` reads named keys and
#: ignores everything else, so leaving them on is harmless — but a collator that is
#: swapped later must not have to know about them, so the side channel is explicit.
SIDE_KEYS = ("task_id", "example_index", "step", "num_tokens", "pad_shape")

#: Coarse, task-agnostic bucket ladders for D4.3, used when the sampler is not told
#: the collator's own (:func:`ladder_shape`). Powers of two from these floors; the
#: point is only that the table is a function of size and not of task, so a
#: micro-batch's padding waste is bounded without the task ever entering the key.
NODE_BUCKET_MIN = 8
TOKEN_BUCKET_MIN = 32

#: Steps :meth:`MixtureSampler.derive_accumulation_steps` replays to find the
#: heaviest step. The per-step composition is a multinomial, so the heaviest of a
#: few hundred is within a few percent of the heaviest of a whole run.
ACCUMULATION_PROBE_STEPS = 500

STATE_VERSION = 1


class MixtureError(ValueError):
    """A sampler that cannot draw. The message names the task or the step."""


# ─────────────────────────────────────────────────────────────────────────────
# Task ids
# ─────────────────────────────────────────────────────────────────────────────

def task_ids_for(mixture) -> dict:
    """``{task name: int}`` in sorted-name order.

    The integer is what travels in ``batch["task_ids"]`` and in the per-task loss
    table, because a name cannot ride in a tensor. Sorted order makes the id a
    function of the mixture's task *set* alone: two processes that resolved the
    same mixture agree, and a config that lists its tasks in a different order
    does not renumber anything. Adding or removing a task renumbers, which is why
    the ids are never written into a checkpoint — the names are (``state.json``'s
    registry snapshot), and the table is rebuilt from them.
    """
    names = _mixture_names(mixture)
    return {name: i for i, name in enumerate(sorted(names))}


def _mixture_names(mixture) -> list:
    entries = getattr(mixture, "entries", None)
    if entries is not None:
        return [e.name for e in entries]
    return list(mixture)


# ─────────────────────────────────────────────────────────────────────────────
# The draw plan
# ─────────────────────────────────────────────────────────────────────────────

class Draw(NamedTuple):
    """One drawn example: which task, which row, and of which pass.

    ``pass_id`` is carried because a generator's source *is* the pass — row 7 of
    pass 3 and row 7 of pass 4 are different molecules (D4.2). Materialising an
    item after the sampler has rolled over would otherwise silently fetch the
    wrong row.
    """

    task: str
    index: int
    pass_id: int


def _seed(*parts) -> int:
    """A 64-bit stream seed from a tuple of identifiers.

    SHA-256 rather than :func:`hash` because Python's string hash is salted per
    process: a resume in a new process must reproduce the same permutation.
    """
    joined = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


class MixtureSampler:
    """The draw plan for a resolved mixture: counts per step, rows per task, batches.

    Args:
        mixture: a :class:`~src.generalist.registry.Mixture`.
        seed: the run seed. ``(mixture.hash(), seed)`` fixes every stream.
        get_source: ``(task, pass_id) -> TaskSource``, the trainer's loader.
        accumulation_steps: micro-batches per optimizer step; with ``world_size``
            it turns ``tokens_per_step`` into a per-micro-batch token budget (D4.4).
        world_size: ranks the step is split across. ``batches_for_step`` returns
            *all* of a step's micro-batches; :class:`MixtureDataset` hands rank *r*
            the slice ``[r::world_size]``, so every rank runs an identical sampler
            and no cross-rank coordination is needed.
        shape_fn: ``(num_nodes, num_tokens) -> (padded nodes, padded tokens)``,
            the shape the collator will build for one row. The bucket key and
            every cost below are in these units, because they are what the GPU
            holds. ``None`` uses :func:`ladder_shape`, the coarse power-of-two
            ladder, for a caller (a test, a non-flex collator) with no ladder of
            its own.
        micro_batch_tokens: the padded-token budget of one micro-batch on one
            rank. ``None`` derives it from ``tokens_per_step``, as before the
            budget was padded: a raw-token number, which is only right when rows
            are long against the collator's length ladder.
        micro_batch_node_pairs: the second budget, ``rows x padded nodes²`` per
            micro-batch per rank — what the dense pair bias holds. 0 means none.

    The sampler is a cursor: :meth:`batches_for_step` may only be called for the
    step it is currently at, and advances it. Restoring a :meth:`state_dict` is the
    only way to go back.
    """

    def __init__(self, mixture, seed: int, get_source: Callable,
                 accumulation_steps: int = 1, world_size: int = 1,
                 shape_fn: Callable | None = None,
                 micro_batch_tokens: float | None = None,
                 micro_batch_node_pairs: float = 0):
        if accumulation_steps < 1:
            raise MixtureError(
                f"accumulation_steps must be >= 1, got {accumulation_steps}")
        if world_size < 1:
            raise MixtureError(f"world_size must be >= 1, got {world_size}")
        if micro_batch_tokens is not None and micro_batch_tokens <= 0:
            raise MixtureError(
                f"micro_batch_tokens must be > 0, got {micro_batch_tokens}")
        if micro_batch_node_pairs < 0:
            raise MixtureError(
                f"micro_batch_node_pairs must be >= 0, got {micro_batch_node_pairs}")

        self.mixture = mixture
        self.seed = int(seed)
        self.get_source = get_source
        self.accumulation_steps = int(accumulation_steps)
        self.world_size = int(world_size)
        self.mixture_hash = mixture.hash()

        self.entries = {e.name: e for e in mixture.entries}
        self.tasks = sorted(self.entries)
        self.task_ids = task_ids_for(mixture)
        self.examples_per_step = float(mixture.examples_per_step)
        self.shape_fn = shape_fn or ladder_shape
        #: Padded tokens one micro-batch may hold on one rank. Given, it is a
        #: memory budget and `derive_accumulation_steps` sizes the accumulation
        #: to it. Otherwise the step's budget is ``tokens_per_step`` split across
        #: the accumulation micro-batches and the ranks, which is what makes
        #: ``batch_size`` derived (D4.4).
        self.padded_budget = micro_batch_tokens is not None
        self.micro_batch_tokens = (
            float(micro_batch_tokens) if micro_batch_tokens is not None
            else float(mixture.tokens_per_step)
            / (self.accumulation_steps * self.world_size))
        self.micro_batch_node_pairs = float(micro_batch_node_pairs)

        # Draw probabilities, in `self.tasks` order, renormalised so numpy's
        # multinomial never trips its sum > 1 check on float error.
        p = np.array([mixture.shares[t] for t in self.tasks], dtype=np.float64)
        self._p = p / p.sum()

        self.step = 0
        self.cursor = {t: 0 for t in self.tasks}
        self.pass_id = {t: 0 for t in self.tasks}
        self.exhausted = set()

        self._sources: dict = {}
        self._perms: dict = {}
        self._warned_over_budget = False

    # ── the pure part ────────────────────────────────────────────────────────

    def examples_in_step(self, k: int) -> int:
        """How many examples step *k* draws, as a pure function of *k*.

        ``examples_per_step`` is a float (D4.4: it is a token budget divided by a
        mean length, not a configured integer). Rounding it every step would drift
        the realised token budget; carrying a fractional accumulator fixes that,
        and writing the accumulator in closed form —
        ``floor((k+1)·e) - floor(k·e)`` — keeps the count a pure function of *k*
        rather than of how many steps have been taken, which is what D4.1 needs
        for a resume.
        """
        e = self.examples_per_step
        return int(math.floor((k + 1) * e)) - int(math.floor(k * e))

    def fraction_at(self, k: int) -> float:
        """The fractional accumulator entering step *k*; recorded in the state."""
        v = k * self.examples_per_step
        return float(v - math.floor(v))

    def counts_for_step(self, k: int) -> dict:
        """``{task: count}`` for step *k*: D4.1's deterministic multinomial.

        The stream is seeded by ``(mixture hash, seed, k)`` alone, so the *plan*
        for a step never depends on what happened before it. A task that has
        exhausted its passes still appears in the plan; :meth:`draw_step` drops
        its slots rather than redistributing them, because redistribution would
        make step *k*'s composition depend on history and there would be no
        resumable draw plan left.
        """
        n = self.examples_in_step(k)
        if n <= 0:
            return {t: 0 for t in self.tasks}
        rng = np.random.default_rng(_seed(self.mixture_hash, self.seed, "step", k))
        counts = rng.multinomial(n, self._p)
        return {t: int(c) for t, c in zip(self.tasks, counts)}

    # ── sources and permutations ─────────────────────────────────────────────

    def source(self, task: str, pass_id: int):
        """The task's source for a pass, through the trainer's ``get_source``.

        Cached per ``(task, pass_id)`` and trimmed to the two most recent passes of
        each task: a step that straddles a pass boundary needs both, and nothing
        needs a third. A generator's ``get_source`` is expensive (it materialises a
        fresh draw), so the cache is what keeps D4.2 from re-running a pass per
        micro-batch.
        """
        key = (task, pass_id)
        src = self._sources.get(key)
        if src is None:
            src = self.get_source(task, pass_id)
            if src is None:
                raise MixtureError(
                    f"{task}: get_source returned None for pass {pass_id}")
            self._sources[key] = src
            for stale in [k for k in self._sources
                          if k[0] == task and k[1] < pass_id - 1]:
                self._sources.pop(stale, None)
                self._perms.pop(stale, None)
        return src

    def _permutation(self, task: str, pass_id: int, n: int) -> np.ndarray:
        key = (task, pass_id)
        perm = self._perms.get(key)
        if perm is None or len(perm) != n:
            rng = np.random.default_rng(
                _seed(self.mixture_hash, self.seed, "perm", task, pass_id))
            perm = rng.permutation(n)
            self._perms[key] = perm
        return perm

    def _max_passes(self, task: str):
        """``passes`` bounds a corpus; a generator is unbounded.

        This mirrors ``registry.resolve``, where a corpus contributes
        ``passes x train_size`` to the budget and a generator contributes ``None``
        because D4.2 draws a fresh pass every time. Capping a generator here would
        contradict the budget the mixture was resolved under.
        """
        entry = self.entries[task]
        return int(entry.passes) if entry.kind == "corpus" else None

    def _retire(self, task: str, reason: str) -> None:
        # A warning, not info: a task leaving the mixture changes the experiment,
        # and `check_supply` has already refused it unless the run allowed it.
        if task not in self.exhausted:
            self.exhausted.add(task)
            logger.warning("mixture: %s is exhausted at step %d (%s); its slots "
                           "are dropped from here on", task, self.step, reason)

    def _take(self, task: str, count: int) -> list:
        """``count`` rows of ``task``, walking the cursor and rolling passes."""
        out = []
        while count > 0 and task not in self.exhausted:
            pass_id = self.pass_id[task]
            src = self.source(task, pass_id)
            n = len(src)
            if n == 0:
                self._retire(task, f"pass {pass_id} is empty")
                break
            perm = self._permutation(task, pass_id, n)
            cursor = self.cursor[task]
            take = min(count, n - cursor)
            out.extend(Draw(task, int(perm[cursor + i]), pass_id)
                       for i in range(take))
            cursor += take
            count -= take
            if cursor >= n:
                max_passes = self._max_passes(task)
                if max_passes is not None and pass_id + 1 >= max_passes:
                    self.cursor[task] = cursor
                    self._retire(task, f"{max_passes} pass(es) consumed")
                    break
                self.pass_id[task] = pass_id + 1
                self.cursor[task] = 0
                # D4.2: the next pass is requested at the boundary, not lazily on
                # the next step, so a generator's fresh draw is already in hand.
                self.source(task, pass_id + 1)
            else:
                self.cursor[task] = cursor
        return out

    def draw_step(self, k: int) -> list:
        """The examples of step *k* as a flat list of :class:`Draw`; advances to *k+1*."""
        if k != self.step:
            raise MixtureError(
                f"step {k} requested but the sampler is at step {self.step}; the "
                "cursors only move forward. Restore a state_dict to go back.")
        counts = self.counts_for_step(k)
        draws = []
        for task in self.tasks:
            c = counts[task]
            if c and task not in self.exhausted:
                draws.extend(self._take(task, c))
        self.step = k + 1
        return draws

    # ── supply: will the run have the data it plans to draw? ─────────────────

    def plan_supply(self, end_step: int) -> dict:
        """Replay the draw plan from the current step to ``end_step``, moving nothing.

        Returns ``{task: {"draws", "last_pass", "retires_at"}}``: how many rows the
        task will hand over, the highest pass it will open, and the step at which
        a corpus will hit its pass cap (``None`` if it does not). The plan is the
        same pure per-step multinomial :meth:`draw_step` uses, and the cursor walk
        is :meth:`_take`'s, so this is what the run *will* do, not an estimate.
        A generator's later passes are taken to be the size of its current one —
        every builder here draws a fixed count per pass.
        """
        cursor = dict(self.cursor)
        pass_id = dict(self.pass_id)
        exhausted = set(self.exhausted)
        size = {t: len(self.source(t, pass_id[t])) for t in self.tasks
                if t not in exhausted}
        out = {t: {"draws": 0, "last_pass": pass_id[t], "retires_at": None}
               for t in self.tasks}
        for k in range(self.step, int(end_step)):
            for task, count in self.counts_for_step(k).items():
                while count > 0 and task not in exhausted:
                    n = size[task]
                    if n == 0:
                        exhausted.add(task)
                        out[task]["retires_at"] = k
                        break
                    take = min(count, n - cursor[task])
                    cursor[task] += take
                    count -= take
                    out[task]["draws"] += take
                    if cursor[task] >= n:
                        cap = self._max_passes(task)
                        if cap is not None and pass_id[task] + 1 >= cap:
                            exhausted.add(task)
                            out[task]["retires_at"] = k
                            break
                        pass_id[task] += 1
                        cursor[task] = 0
                        out[task]["last_pass"] = pass_id[task]
        return out

    def check_supply(self, end_step: int, allow_exhaustion: bool = False) -> dict:
        """Refuse a run that would run out of data before ``end_step``.

        Two ways a run quietly stops being the experiment it was configured as,
        both measured on the molecule forks before this existed:

        * **a corpus past its ``passes`` cap retires** and its share is dropped
          for the rest of the run. A 4,456-step decay over a fork config sized
          for 1,114 retired BACE, BBBP and ChEBI-20 mid-run, and nothing said so.
          Refused unless ``allow_exhaustion`` — a smoke run budgeted by a step
          count wants it — and logged as a warning when allowed.
        * **a generator asks for a pass ``data_prep`` never built**, and the run
          dies with a build error at that pass boundary, hours in. Always refused:
          no setting makes a missing file loadable. Only the last pass the run
          will open is loaded here, through ``get_source`` directly so the
          sampler's two-pass cache is not disturbed.

        Tasks already exhausted when this runs (a resume past a retirement that
        was allowed then) are logged and not refused again. Returns the plan.
        """
        plan = self.plan_supply(end_step)
        problems = []
        if self.exhausted:
            logger.warning("mixture: already exhausted at step %d: %s",
                           self.step, sorted(self.exhausted))
        for task in self.tasks:
            row = plan[task]
            if task in self.exhausted:
                continue
            if row["retires_at"] is not None:
                n = len(self.source(task, self.pass_id[task]))
                message = (
                    f"{task} runs out at step {row['retires_at']} of {end_step}: "
                    f"its cap is {self._max_passes(task)} pass(es) of {n} rows. "
                    f"Raise its passes, shorten the run, or set allow_exhaustion "
                    f"if a task leaving the mixture part-way is what this run is for")
                if allow_exhaustion:
                    logger.warning("mixture: %s", message)
                else:
                    problems.append(message)
            last = row["last_pass"]
            if (self.entries[task].kind == "generator"
                    and last > self.pass_id[task]):
                try:
                    self.get_source(task, last)
                except Exception as exc:                 # noqa: BLE001 - reported
                    problems.append(
                        f"{task} needs pass {last} by step {end_step} and it does "
                        f"not load ({type(exc).__name__}: {exc}). Build the passes "
                        f"this run consumes with data_prep before starting it")
        if problems:
            raise MixtureError(
                f"the mixture cannot supply steps {self.step}..{end_step}:\n  "
                + "\n  ".join(problems))
        return plan

    # ── D4.3/D4.4 batching ───────────────────────────────────────────────────

    def batches_for_step(self, k: int) -> list:
        """Step *k*'s examples as ``accumulation_steps x world_size`` micro-batches;
        advances to *k+1*.

        **Shape-keyed and rank-synchronised** (`GRAPH_GENERALIST.md` §3). Each
        example is keyed by the padded ``(nodes, tokens)`` the collator will build
        for it (``shape_fn``). A step is cut into *groups*; a group is one shape
        and ``world_size`` micro-batches of it, one per rank, and the batches are
        returned group by group, so :class:`MixtureDataset`'s ``[r::world_size]``
        hands every rank the same shape at the same micro-step. Before this, rank
        *r* took every ``world_size``-th batch of a size-sorted list, so the ranks
        ran different buckets side by side, waited on the slowest, and each
        compiled shapes the others did not.

        **Within a group the batches are dealt** round-robin from a task-ordered
        list. Dealing rather than slicing is what keeps them mixed: contiguous
        slices would be task-homogeneous exactly when a task has a distinctive
        size, which is the common case.

        **Every rank gets the same row count**, so a bucket's count is cut to a
        multiple of ``world_size``. The few left over from each bucket are pooled
        and grouped with each other by size, each group padded to its largest
        member — promoted, never dropped or deferred: deferring would
        under-sample whatever is unusually sized. Only the last of those, when the
        step's total is not a multiple of ``world_size``, leaves some ranks one
        row short.

        **Costs are what the collator builds**: ``rows x padded tokens`` against
        ``micro_batch_tokens``, and ``rows x padded nodes²`` against
        ``micro_batch_node_pairs`` when it is set. A bucket is split into as many
        groups as those budgets ask for, and the step is then reshaped to
        exactly ``accumulation_steps`` groups — merging the cheapest *pair* or
        splitting the dearest group — here rather than per rank in the trainer,
        because a reshape each rank did on its own rows could pick different
        merges and break the shape sync it exists to keep. HF needs the fixed
        count; `derive_accumulation_steps` chooses one that keeps the reshaped
        groups inside the budget.

        Batching only regroups: a step's per-task counts are the draw plan's
        whatever the ranks and the budgets, and the loss is normalised over the
        step (D4.3), so none of this moves the gradient.
        """
        draws = self.draw_step(k)
        if not draws:
            return []
        ws = self.world_size
        if len(draws) < ws * self.accumulation_steps:
            raise MixtureError(
                f"step {k} drew {len(draws)} example(s), fewer than "
                f"accumulation_steps x world_size = "
                f"{self.accumulation_steps} x {ws}: some micro-batch would be "
                f"empty. Lower accumulation_steps or raise tokens_per_step.")

        groups = self._reshape(self._natural_groups(draws, self._shapes(draws)))
        groups.sort(key=lambda g: (_shape_order(g.shape), _draw_order(g.items[0])))
        return [MicroBatch(g.items[r::ws], g.shape) for g in groups for r in range(ws)]

    def _natural_groups(self, draws, shapes) -> list:
        """A step's groups as the budgets cut them, before `_reshape`."""
        ws = self.world_size
        buckets = defaultdict(list)
        for d in draws:
            buckets[shapes[d]].append(d)

        groups, pool = [], []
        for shape in sorted(buckets, key=_shape_order):
            items = sorted(buckets[shape], key=_draw_order)
            keep = len(items) - len(items) % ws
            pool.extend(items[keep:])
            if keep:
                groups.extend(self._split_bucket(shape, items[:keep]))
        if pool:
            pool.sort(key=lambda d: (_shape_order(shapes[d]), _draw_order(d)))
            chunks = [pool[i:i + ws] for i in range(0, len(pool), ws)]
            short = chunks.pop() if len(chunks[-1]) < ws else []
            if short and chunks:
                chunks[-1].extend(short)
            elif short:
                # Fewer leftovers than ranks and no chunk to join: they go into
                # the bucket group they cost least in once it is padded to cover
                # them, so a full group is not pushed over the budget.
                def joined(g):
                    return _cover([g.shape] + [shapes[d] for d in short])
                host = min(range(len(groups)), key=lambda n: (
                    self._cost(joined(groups[n]), len(groups[n].items) + len(short)),
                    _draw_order(groups[n].items[0])))
                g = groups[host]
                groups[host] = _Group(joined(g),
                                      sorted(g.items + short, key=_draw_order))
            for chunk in chunks:
                groups.append(_Group(_cover(shapes[d] for d in chunk),
                                     sorted(chunk, key=_draw_order)))
        return groups

    def _shapes(self, draws) -> dict:
        """``{draw: padded (nodes, tokens)}`` through ``shape_fn``."""
        lengths = {}
        for key in {(d.task, d.pass_id) for d in draws}:
            lengths[key] = self.source(*key).lengths()
        out = {}
        for d in draws:
            nodes, tokens = lengths[(d.task, d.pass_id)]
            out[d] = tuple(int(v) for v in
                           self.shape_fn(int(nodes[d.index]), int(tokens[d.index])))
        return out

    def _row_cap(self, shape) -> int:
        """Rows per rank one micro-batch of ``shape`` may hold under the budgets."""
        nodes, tokens = shape
        cap = self.micro_batch_tokens // tokens
        if self.micro_batch_node_pairs:
            cap = min(cap, self.micro_batch_node_pairs // (nodes * nodes))
        return max(1, int(cap))

    def _cost(self, shape, n_items: int) -> float:
        """A group's load on one rank, as a fraction of the tighter budget."""
        nodes, tokens = shape
        rows = math.ceil(n_items / self.world_size)
        cost = rows * tokens / self.micro_batch_tokens
        if self.micro_batch_node_pairs:
            cost = max(cost, rows * nodes * nodes / self.micro_batch_node_pairs)
        return cost

    def _split_bucket(self, shape, items: list) -> list:
        """One bucket's rows (a multiple of ``world_size``) as budget-sized groups."""
        ws = self.world_size
        rows = len(items) // ws
        n_groups = max(1, math.ceil(rows / self._row_cap(shape)))
        sizes = [ws * (rows // n_groups + (g < rows % n_groups))
                 for g in range(n_groups)]
        members = [[] for _ in range(n_groups)]
        g = 0
        for d in items:
            while len(members[g]) >= sizes[g]:
                g = (g + 1) % n_groups
            members[g].append(d)
            g = (g + 1) % n_groups
        return [_Group(shape, m) for m in members]

    def _reshape(self, groups: list) -> list:
        """Exactly ``accumulation_steps`` groups, by padded cost.

        `trainer.align_to_accumulation`'s rule lifted to whole groups: merge the
        pair whose *merged* cost is lowest (a merge pads both to the larger
        shape), split the dearest group that has two rows a rank to split. The
        ties break on each group's first draw, so every rank, and every resume,
        makes the same choice.
        """
        target = self.accumulation_steps
        ws = self.world_size
        while len(groups) > target:
            best = None
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    shape = _cover((groups[i].shape, groups[j].shape))
                    n = len(groups[i].items) + len(groups[j].items)
                    key = (self._cost(shape, n), _draw_order(groups[i].items[0]),
                           _draw_order(groups[j].items[0]))
                    if best is None or key < best[0]:
                        best = (key, i, j, shape)
            key, i, j, shape = best
            if (self.padded_budget and key[0] > 1.0
                    and not self._warned_over_budget):
                # A step past the probe `derive_accumulation_steps` sized the
                # accumulation on cut more groups than it; said once, not per step.
                self._warned_over_budget = True
                logger.warning(
                    "mixture: step %d needs more than %d micro-batches; a merged "
                    "one is %.2fx micro_batch_tokens and may not fit in memory",
                    self.step - 1, target, key[0])
            merged = _Group(shape, sorted(groups[i].items + groups[j].items,
                                          key=_draw_order))
            groups = [g for n, g in enumerate(groups) if n not in (i, j)] + [merged]
        while len(groups) < target:
            splittable = [g for g in groups if len(g.items) >= 2 * ws]
            if not splittable:                           # guarded in the caller
                raise MixtureError("cannot reach accumulation_steps micro-batches")
            dearest = max(splittable, key=lambda g: (
                self._cost(g.shape, len(g.items)), _draw_order(g.items[0])))
            groups.remove(dearest)
            # Half the rows a rank, taken as every second item of the task-ordered
            # list so both halves stay mixed. Both keep the group's shape.
            head = ws * ((len(dearest.items) // ws) // 2)
            picked = set(range(0, len(dearest.items), 2)[:head])
            first = [d for n, d in enumerate(dearest.items) if n in picked]
            rest = [d for n, d in enumerate(dearest.items) if n not in picked]
            groups.extend([_Group(dearest.shape, first),
                           _Group(dearest.shape, rest)])
        return groups

    # ── accumulation sized to a padded budget ────────────────────────────────

    def derive_accumulation_steps(self, probe_steps: int = ACCUMULATION_PROBE_STEPS
                                  ) -> int:
        """The accumulation that keeps the heaviest step's micro-batches in budget.

        ``tokens_per_step`` fixes a step's example count from *raw* lengths
        (D4.4), and HF fixes the number of micro-batches a step is cut into. With
        a fixed accumulation, a mixture whose rows are short against the
        collator's 512-token length ladder then puts many times its raw tokens on
        the card: in the 2026-10-04 graph smoke a step of 33-token GraphQA rows
        came out as micro-batches of 64 rows padded to 512, and the logits alone
        ran an 80 GB card out of memory. So when ``micro_batch_tokens`` is a
        padded budget, the accumulation is derived from it instead of configured:
        the heaviest padded step over the next ``probe_steps``, divided by the
        budget, rounded up. Sets and returns ``accumulation_steps``.

        "Heaviest" is the step's group count as `batches_for_step` cuts it
        before reshaping (`_natural_groups`), not its mean padded volume over
        the budget. The volume is a lower bound only: groups of different
        shapes do not pack, and when the accumulation is below the natural
        count `_reshape` merges groups and pads both to the larger shape. On the
        2026-10-05 2-rank smoke the volume rule gave 2 micro-batches a rank
        where the bucketing made more, and the merged ones ran an 80 GB card out
        of memory. With the accumulation at the natural maximum a step is only
        ever split, never merged, so every micro-batch stays within budget.

        The replay draws nothing: it walks the cursors as `plan_supply` does,
        on each task's current pass, and past that pass's end it wraps round
        the same permutation. That stands in for the next pass's rows; a
        corpus repeats its rows, and a generator's passes are draws from one
        distribution. No pass is loaded.
        """
        if not self.padded_budget:
            raise MixtureError(
                "derive_accumulation_steps needs micro_batch_tokens: without a "
                "padded budget there is nothing to size the accumulation against")
        steps = range(self.step, self.step + int(probe_steps))
        live = [t for t in self.tasks if t not in self.exhausted]
        size = {t: len(self.source(t, self.pass_id[t])) for t in live}
        perm = {t: self._permutation(t, self.pass_id[t], size[t])
                for t in live if size[t]}
        cursor = {t: self.cursor[t] for t in perm}
        lengths = {t: self.source(t, self.pass_id[t]).lengths() for t in perm}
        shape_of = {}

        heaviest = 1
        for k in steps:
            draws, shapes = [], {}
            for task, count in self.counts_for_step(k).items():
                if task not in perm:
                    continue
                n, pass_id = size[task], self.pass_id[task]
                for i in range(count):
                    index = int(perm[task][(cursor[task] + i) % n])
                    d = Draw(task, index, pass_id)
                    if d not in shape_of:
                        nodes, tokens = lengths[task]
                        shape_of[d] = tuple(int(v) for v in self.shape_fn(
                            int(nodes[index]), int(tokens[index])))
                    draws.append(d)
                    shapes[d] = shape_of[d]
                cursor[task] = (cursor[task] + count) % n
            if len(set(draws)) < len(draws):
                # A step wider than a pass repeats a row; the draws must stay
                # distinct to key the buckets, and the count is what matters.
                draws = [Draw(d.task, d.index, -1 - i) for i, d in enumerate(draws)]
                shapes = {d: shape_of[Draw(d.task, d.index, self.pass_id[d.task])]
                          for d in draws}
            if draws:
                heaviest = max(heaviest, len(self._natural_groups(draws, shapes)))
        smallest = min(self.examples_in_step(k) for k in steps)
        if heaviest * self.world_size > smallest:
            raise MixtureError(
                f"the padded budget asks for {heaviest} micro-batches a rank, but a "
                f"step can draw as few as {smallest} examples over "
                f"{self.world_size} rank(s): raise micro_batch_tokens or "
                f"tokens_per_step")
        self.accumulation_steps = int(heaviest)
        return self.accumulation_steps

    # ── D4.1 state ───────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """JSON-serialisable; this is what ``checkpoint.py`` writes as ``sampler.json``."""
        return {
            "version": STATE_VERSION,
            "mixture_hash": self.mixture_hash,
            "seed": self.seed,
            "step": self.step,
            "cursor": dict(self.cursor),
            "pass_id": dict(self.pass_id),
            "exhausted": sorted(self.exhausted),
            "fraction": self.fraction_at(self.step),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a cursor vector. A changed mixture is a warning, not an error.

        D5.4 makes a mixture change legal (it forces a re-warm and a lineage
        entry), so a task that is no longer in the mixture is dropped and a task
        that is new starts at pass 0 — refusing here would make the resume path
        unable to do the one thing the design says it may.
        """
        if not state:
            raise MixtureError("load_state_dict: empty sampler state")
        if state.get("mixture_hash") != self.mixture_hash:
            logger.warning(
                "mixture: resuming a sampler whose state was written under mixture "
                "hash %s but this run resolves to %s; per-task cursors are matched "
                "by name and new tasks start at pass 0",
                state.get("mixture_hash"), self.mixture_hash)
        self.step = int(state["step"])
        self.cursor = {t: int(state.get("cursor", {}).get(t, 0)) for t in self.tasks}
        self.pass_id = {t: int(state.get("pass_id", {}).get(t, 0)) for t in self.tasks}
        self.exhausted = {t for t in state.get("exhausted", ()) if t in self.entries}
        self._sources.clear()
        self._perms.clear()


def _bucket_up(value, minimum: int) -> int:
    """The smallest power-of-two multiple of ``minimum`` that covers ``value``."""
    v = int(minimum)
    while v < int(value):
        v *= 2
    return v


def ladder_shape(nodes: int, tokens: int) -> tuple:
    """The default ``shape_fn``: the coarse power-of-two ladder of D4.3."""
    return _bucket_up(nodes, NODE_BUCKET_MIN), _bucket_up(tokens, TOKEN_BUCKET_MIN)


class MicroBatch(list):
    """A micro-batch's draws, and the padded ``(nodes, tokens)`` of its group.

    The shape has to travel: a merged group holds rows of several sizes, and the
    collator pads a batch to its *own* widest row, so two ranks holding
    different rows of one group would build different shapes. `MixtureDataset`
    stamps it on each item as ``pad_shape`` and `wrap_collator` pads to it.
    """

    def __init__(self, draws=(), shape=None):
        super().__init__(draws)
        self.shape = tuple(shape) if shape is not None else None


class _Group(NamedTuple):
    """One micro-step of a step: a padded shape and its rows across all ranks."""

    shape: tuple
    items: list


def _cover(shapes) -> tuple:
    """The smallest shape every one of ``shapes`` pads into."""
    shapes = list(shapes)
    return max(s[0] for s in shapes), max(s[1] for s in shapes)


def _shape_order(shape) -> tuple:
    # Tokens first: they set the logits and the attention, so a step's buckets
    # run from cheapest to dearest in the dimension that dominates the memory.
    return shape[1], shape[0]


def _draw_order(d: Draw) -> tuple:
    return d.task, d.pass_id, d.index


# ─────────────────────────────────────────────────────────────────────────────
# The dataset and the collator wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MixtureDataset(torch.utils.data.IterableDataset):
    """The sampler's draws as collate-ready micro-batches.

    Yields one *list of items* per micro-batch by default, which is the
    ``DataLoader(dataset, batch_size=None, collate_fn=...)`` idiom: with automatic
    batching off, the collate function is applied to whatever the dataset yields,
    so a variable-size micro-batch (D4.4 derives the size from a token budget, so
    it varies) passes through unchanged. Set ``yield_batches=False`` to get
    individual items instead, for a caller that does its own grouping.

    Each item is a shallow copy of the source's item plus ``task_id``,
    ``example_index`` and ``step`` — :func:`wrap_collator` strips those and
    re-attaches them to the batch dict as tensors, because the trainer's per-task
    accounting needs ``batch["task_ids"]`` and the per-example report needs to know
    which row of which task it is looking at.

    With ``world_size > 1`` rank *r* takes micro-batches ``[r::world_size]`` of
    every step. Every rank runs an identical sampler over identical draws, so the
    split needs no communication and each rank's slice is a mixed sample of the
    step rather than a contiguous block of one bucket.
    """

    def __init__(self, sampler: MixtureSampler, start_step: int = 0,
                 end_step: int | None = None, rank: int = 0, world_size: int = 1,
                 yield_batches: bool = True):
        super().__init__()
        if rank < 0 or rank >= world_size:
            raise MixtureError(f"rank {rank} is not in [0, {world_size})")
        self.sampler = sampler
        self.start_step = int(start_step)
        self.end_step = end_step
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.yield_batches = bool(yield_batches)
        #: ``(task, pass_id) -> num_tokens`` per example, memoised because
        #: ``TaskSource.lengths()`` copies both lists on every call and ``_item``
        #: runs once per example.
        self._token_lengths: dict = {}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is not None and info.num_workers > 1:
            # Every worker would run the same sampler and emit the same steps.
            # Sharding by worker is possible but would interleave the steps out of
            # order, and the sampler is the run's data order — so refuse rather
            # than silently train on duplicates.
            raise MixtureError(
                "MixtureDataset is a single-stream IterableDataset; run the "
                "DataLoader with num_workers <= 1 (the sampler is the data order "
                "and workers would duplicate it).")

        step = self.start_step
        while self.end_step is None or step < self.end_step:
            batches = self.sampler.batches_for_step(step)
            if not batches and self.sampler.exhausted == set(self.sampler.tasks):
                logger.info("mixture: every task is exhausted at step %d; the "
                            "stream ends here", step)
                return
            for batch in batches[self.rank::self.world_size]:
                items = [self._item(d, step) for d in batch]
                shape = getattr(batch, "shape", None)
                if shape is not None:
                    for item in items:
                        item["pad_shape"] = shape
                if self.yield_batches:
                    yield items
                else:
                    yield from items
            step += 1

    def _item(self, draw: Draw, step: int) -> dict:
        source = self.sampler.source(draw.task, draw.pass_id)
        item = dict(source[draw.index])
        item["task_id"] = self.sampler.task_ids[draw.task]
        item["example_index"] = int(draw.index)
        item["step"] = int(step)
        # The same token count `batches_for_step` buckets on, carried forward so
        # that whatever regroups these micro-batches downstream can size them the
        # way the sampler did. `align_to_accumulation` reshapes a step to exactly
        # `accumulation_steps` groups, and it has to merge by padded token cost:
        # keying on example count instead makes the many one-item batches that a
        # long-sequence bucket produces look like the *cheapest* things to merge,
        # and collapsing them together builds a micro-batch many times over
        # budget.
        item["num_tokens"] = int(self._tokens_for(draw.task, draw.pass_id)
                                 [draw.index])
        return item

    def _tokens_for(self, task: str, pass_id: int):
        key = (task, pass_id)
        if key not in self._token_lengths:
            self._token_lengths[key] = self.sampler.source(task,
                                                           pass_id).lengths()[1]
        return self._token_lengths[key]


def wrap_collator(base_collator: Callable, task_ids_key: str = "task_ids") -> Callable:
    """Strip D4's side-channel keys, collate, and re-attach them as tensors.

    ``GraphCollatorV2`` reads named keys and ignores the rest, so this is not
    needed to keep it from raising — it is needed because the batch dict has to
    come back out with ``task_ids`` on it, and because a collator swapped in later
    must not have to know that the mixture stows anything on an item.
    """

    def collate(items):
        stripped = [{k: v for k, v in item.items() if k not in SIDE_KEYS}
                    for item in items]
        batch = _padded_to(base_collator, items)(stripped)
        batch[task_ids_key] = torch.tensor(
            [int(item.get("task_id", -1)) for item in items], dtype=torch.long)
        batch["example_index"] = torch.tensor(
            [int(item.get("example_index", -1)) for item in items], dtype=torch.long)
        batch["step"] = torch.tensor(
            [int(item.get("step", -1)) for item in items], dtype=torch.long)
        return batch

    return collate


def _padded_to(base_collator: Callable, items) -> Callable:
    """``base_collator``, padding at least to the items' ``pad_shape``.

    A collator with ladders (``pad_to_block``) is shallow-copied with each ladder
    floored at the group's shape, so a rank whose rows happen to be short still
    builds the shape every other rank builds at this micro-step. The copy shares
    everything else; the base collator is not touched. Without a stamped shape,
    or a collator that has no ladder to floor, this is the base collator.
    """
    import copy

    from ..models.flex_kernel import bucketize

    shapes = [item.get("pad_shape") for item in items if item.get("pad_shape")]
    if not shapes or not getattr(base_collator, "pad_to_block", False):
        return base_collator
    nodes, tokens = _cover(shapes)
    floored = copy.copy(base_collator)
    len_ladder, node_ladder = base_collator.len_buckets, base_collator.node_buckets
    floored.len_buckets = lambda value: bucketize(max(int(value), tokens), len_ladder)
    floored.node_buckets = lambda value: bucketize(max(int(value), nodes), node_ladder)
    return floored


# ─────────────────────────────────────────────────────────────────────────────
# D4.3 — two-level loss accounting
# ─────────────────────────────────────────────────────────────────────────────

def count_examples_in_step(task_ids, accumulation_steps: int = 1,
                           world_size: int | None = None) -> int:
    """Examples in the whole optimizer step, from one micro-batch's ``task_ids``.

    The exact count is ``len(MixtureSampler.draw_step(k))`` and a trainer that has
    the sampler in hand should pass that. This is the fallback for a trainer that
    does not: it assumes the step's micro-batches are the same size, which is true
    only when the bucket ladder happened to make them so. It is here because the
    alternative fallback — normalising by the micro-batch — is the accumulation
    footgun D4.3 exists to close, and an approximate step count is much closer to
    right than a per-micro-batch one.
    """
    local = int(task_ids.shape[0]) * int(accumulation_steps)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        total = torch.tensor([local], dtype=torch.long)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return int(total.item())
    return local * int(world_size or 1)


class MixtureLoss:
    """Per-example normalisation, then a mean over the *optimizer step* (D4.3).

    Args:
        loss_norm: ``{task: "per_example" | "per_token"}`` — the registry's table.
            Keys may be task names (then ``task_ids`` must be given, so they can
            be translated to the integers that ride in the batch) or task ids.
            Missing tasks default to ``per_example``.
        task_ids: ``{name: id}`` from :func:`task_ids_for`, needed only to
            translate a name-keyed ``loss_norm``.
        ddp_scale: multiply the returned loss by ``world_size`` when
            ``torch.distributed`` is initialised. DDP averages gradients across
            ranks; the loss here is already normalised by the *global* example
            count, so without this the step would be divided by the world size
            twice. Off only for a caller that reduces gradients itself.

    ``per_example`` divides an example's summed loss by its own span length, so
    every example contributes one unit and a task's gradient share equals its
    example share in expectation. ``per_token`` divides by the micro-batch's mean
    span instead, which keeps the *task*-level share matched to the example share
    while leaving long examples inside a task weighted more than short ones.
    """

    def __init__(self, loss_norm: dict | None = None, task_ids: dict | None = None,
                 ddp_scale: bool = True):
        self.ddp_scale = bool(ddp_scale)
        self._by_id: dict = {}
        for key, norm in (loss_norm or {}).items():
            if norm not in ("per_example", "per_token"):
                raise MixtureError(
                    f"{key}: loss_norm must be 'per_example' or 'per_token', got "
                    f"{norm!r}")
            if isinstance(key, str):
                if not task_ids or key not in task_ids:
                    raise MixtureError(
                        f"{key}: loss_norm is keyed by task name but no task_ids "
                        "table was given to translate it; pass task_ids_for(mixture)")
                self._by_id[int(task_ids[key])] = norm
            else:
                self._by_id[int(key)] = norm
        self._has_per_token = any(v == "per_token" for v in self._by_id.values())

    def norm_for(self, task_id) -> str:
        return self._by_id.get(int(task_id), "per_example")

    def per_example_losses(self, token_losses, label_mask, task_ids):
        """The ``(B,)`` vector of per-example losses, still differentiable.

        Split out of :meth:`__call__` because the gradient-share readout (D4.3)
        needs one task's rows of *this* micro-batch as a scalar it can backward
        through, and it must be the same quantity the step actually summed —
        including ``per_token``'s mean span, which is a property of the whole
        micro-batch and would come out different if a caller recomputed it over a
        subset of the rows.
        """
        mask = label_mask.to(token_losses.dtype)
        spans = mask.sum(dim=-1)
        summed = (token_losses * mask).sum(dim=-1)

        # An example with no supervised token contributes nothing, but must not
        # divide by zero on the way there.
        safe_spans = spans.clamp(min=1.0)
        per_example = summed / safe_spans

        if self._has_per_token:
            mean_span = spans.sum() / max(int(spans.shape[0]), 1)
            mean_span = torch.clamp(mean_span, min=1.0)
            per_token = summed / mean_span
            use_per_token = torch.tensor(
                [self.norm_for(t) == "per_token" for t in task_ids.tolist()],
                dtype=torch.bool, device=token_losses.device)
            per_example = torch.where(use_per_token, per_token, per_example)

        return torch.where(spans > 0, per_example, torch.zeros_like(per_example))

    def __call__(self, token_losses, label_mask, task_ids, examples_in_step):
        """``(loss, {task_id: (sum_loss, n_examples)})``.

        Args:
            token_losses: ``(B, T)`` per-token loss, already shifted to align with
                ``labels`` (the caller owns the shift; HF's ``labels`` convention
                and this repo's collator both put the answer span in the prompt
                node's tokens).
            label_mask: ``(B, T)`` truthy on supervised tokens — normally
                ``labels != -100``.
            task_ids: ``(B,)`` long, from ``batch["task_ids"]``.
            examples_in_step: examples in the whole optimizer step, across
                accumulation micro-batches and ranks. The one number that makes
                the accounting invariant to how the step was chopped up.
        """
        if examples_in_step is None or int(examples_in_step) <= 0:
            raise MixtureError(
                f"examples_in_step must be a positive int, got "
                f"{examples_in_step!r}; it is the whole optimizer step's example "
                "count (D4.3), not the micro-batch's")

        per_example = self.per_example_losses(token_losses, label_mask, task_ids)

        loss = per_example.sum() / float(examples_in_step)
        if self.ddp_scale and torch.distributed.is_available() \
                and torch.distributed.is_initialized():
            loss = loss * float(torch.distributed.get_world_size())

        per_task = {}
        detached = per_example.detach()
        for i, tid in enumerate(task_ids.tolist()):
            total, n = per_task.get(int(tid), (0.0, 0))
            per_task[int(tid)] = (total + float(detached[i]), n + 1)
        return loss, per_task


def measure_grad_share(model, loss_fn_per_task: Callable, tasks,
                       params=None) -> dict:
    """Fraction of the summed gradient L2 norm attributable to each task (D4.3).

    ``loss_fn_per_task(task)`` runs the model on that task's rows of the current
    optimizer step and returns their contribution to the loss — exactly the term
    :class:`MixtureLoss` summed for it. The caller owns it because only the
    trainer knows how to feed the model. It may return ``None`` (the step did not
    sample this task, which is skipped rather than scored zero), a single scalar,
    or an **iterable of scalars** — one per micro-batch — whose gradients are
    summed before the norm is taken. The iterable is consumed lazily and a
    generator is the expected shape: each scalar is backwarded before the next is
    asked for, so exactly one graph is alive at a time.

    **Nothing here retains a graph.** That is not a preference: on the flex path
    the backward is compiled, and a compiled backward with donated buffers refuses
    ``retain_graph=True`` outright ("This backward function was compiled with
    non-empty donated buffers…"). Summing a step's micro-batch losses into one
    scalar and backwarding once would need exactly that, and the alternative —
    flipping ``torch._functorch.config.donated_buffer`` off — would change how
    every subsequent kernel is compiled for the sake of a diagnostic. Summing the
    *gradients* instead is the same quantity and needs no retained graph.

    The readout is the crudest thing that answers the question: the L2 norm of
    each task's summed gradient over the trainable parameters, normalised to sum
    to one. It says nothing about interference between tasks — the norms do not
    add up to the norm of the sum — and it is not meant to; the claim being
    checked is "task *t*'s share of the mixture is the share it has in the
    gradient". It costs one forward and one backward per (task, micro-batch) pair
    the task appears in, which is why it runs on a cadence and not every step.
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    params = list(params)
    if not params:
        raise MixtureError("measure_grad_share: the model has no trainable "
                           "parameters to attribute a gradient to")

    norms = {}
    for task in tasks:
        losses = loss_fn_per_task(task)
        if losses is None:
            continue
        if isinstance(losses, torch.Tensor):
            losses = (losses,)
        summed = [None] * len(params)
        measured = 0
        for loss in losses:
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for i, g in enumerate(grads):
                if g is None:
                    continue
                g = g.detach().double()
                summed[i] = g if summed[i] is None else summed[i] + g
            measured += 1
        if not measured:
            continue
        total = torch.zeros((), dtype=torch.float64)
        for g in summed:
            if g is not None:
                total = total + g.pow(2).sum()
        norms[task] = float(total.sqrt())

    denom = sum(norms.values())
    if denom <= 0:
        # Every task produced a zero gradient. Reporting 1/n shares would read as
        # a balanced mixture; report zeros, which reads as what it is.
        return {task: 0.0 for task in norms}
    return {task: norm / denom for task, norm in norms.items()}
