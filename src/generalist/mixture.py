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
* **Batches are mixed** (D4.3). Homogeneous batches would make per-task gradient
  noise a function of task share, which is exactly the quantity the mixture-weight
  readout is trying to measure. Examples are bucketed by ``(node count, token
  length)`` — a task-agnostic key — and dealt round-robin across the micro-batches
  of their bucket, so a micro-batch is homogeneous only when its bucket is.
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
SIDE_KEYS = ("task_id", "example_index", "step")

#: Coarse, task-agnostic bucket ladders for D4.3. Powers of two from these floors;
#: the point is only that the table is a function of size and not of task, so a
#: micro-batch's padding waste is bounded without the task ever entering the key.
NODE_BUCKET_MIN = 8
TOKEN_BUCKET_MIN = 32

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

    The sampler is a cursor: :meth:`batches_for_step` may only be called for the
    step it is currently at, and advances it. Restoring a :meth:`state_dict` is the
    only way to go back.
    """

    def __init__(self, mixture, seed: int, get_source: Callable,
                 accumulation_steps: int = 1, world_size: int = 1):
        if accumulation_steps < 1:
            raise MixtureError(
                f"accumulation_steps must be >= 1, got {accumulation_steps}")
        if world_size < 1:
            raise MixtureError(f"world_size must be >= 1, got {world_size}")

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
        #: Padded tokens one micro-batch may hold. The whole step's budget is
        #: ``tokens_per_step``; it is split across the accumulation micro-batches
        #: and the ranks, which is what makes ``batch_size`` derived (D4.4).
        self.micro_batch_tokens = (float(mixture.tokens_per_step)
                                   / (self.accumulation_steps * self.world_size))

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
        if task not in self.exhausted:
            self.exhausted.add(task)
            logger.info("mixture: %s is exhausted at step %d (%s); its slots are "
                        "dropped from here on", task, self.step, reason)

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

    # ── D4.3/D4.4 batching ───────────────────────────────────────────────────

    def batches_for_step(self, k: int) -> list:
        """Step *k*'s examples grouped into micro-batches; advances to *k+1*.

        Bucketed by ``(node count, token length)`` on the coarse, task-agnostic
        ladder, then *dealt* round-robin across the micro-batches of each bucket
        from a task-ordered list. Dealing rather than slicing is what makes the
        batches mixed: contiguous slices of a length-sorted list would be
        task-homogeneous exactly when a task has a distinctive size, which is the
        common case (a captioning task's items are all long).

        The padded token total of a micro-batch is
        ``len(batch) x max tokens in batch``, and the bucket bounds the second
        factor, so capping the count at ``micro_batch_tokens // token bucket``
        keeps every batch under budget. One example longer than the whole budget
        still gets its own batch — refusing it here would fail a run at step
        *n* over data that was fine to build.
        """
        draws = self.draw_step(k)
        if not draws:
            return []

        lengths = {}
        for task in {d.task for d in draws}:
            for pass_id in {d.pass_id for d in draws if d.task == task}:
                nodes, tokens = self.source(task, pass_id).lengths()
                lengths[(task, pass_id)] = (nodes, tokens)

        buckets = defaultdict(list)
        for d in draws:
            nodes, tokens = lengths[(d.task, d.pass_id)]
            buckets[(_bucket_up(nodes[d.index], NODE_BUCKET_MIN),
                     _bucket_up(tokens[d.index], TOKEN_BUCKET_MIN))].append(d)

        batches = []
        for key in sorted(buckets):
            items = sorted(buckets[key], key=lambda d: (d.task, d.index))
            cap = max(1, int(self.micro_batch_tokens // key[1]))
            n_batches = max(1, math.ceil(len(items) / cap))
            groups = [[] for _ in range(n_batches)]
            for i, d in enumerate(items):
                groups[i % n_batches].append(d)
            batches.extend(groups)
        return batches

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
        return item


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
        batch = base_collator(stripped)
        batch[task_ids_key] = torch.tensor(
            [int(item.get("task_id", -1)) for item in items], dtype=torch.long)
        batch["example_index"] = torch.tensor(
            [int(item.get("example_index", -1)) for item in items], dtype=torch.long)
        batch["step"] = torch.tensor(
            [int(item.get("step", -1)) for item in items], dtype=torch.long)
        return batch

    return collate


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
