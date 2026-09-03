"""
D5.1 — the trainer: a mixture dataloader, per-task loss, a horizon-free schedule,
an extended checkpoint, and a resume that refuses to be wrong quietly.

:class:`GeneralistTrainer` is ``GraphTrainerV2`` plus the four things a mixture
run needs and HF's ``Trainer`` does not provide. Everything ``GraphTrainerV2``
already does is kept untouched — the two optimizer groups (LoRA at ``lr``, the
graph bias at ``bias_lr``, tagged ``is_bias``), the bias-aware ``save_model`` /
``_load_best_model`` / ``_load_from_checkpoint``, and ``remove_unused_columns
=False``. What is added:

1. **The mixture dataloader** (D4). ``MixtureDataset`` is a single-stream
   ``IterableDataset`` that yields a *list of items* per micro-batch, so the
   loader runs with ``batch_size=None`` and the collator is applied to whatever
   the dataset yields. Two HF defaults have to be turned off around it, and both
   are silent when they are wrong: automatic batching (``batch_size=None``) and
   batch skipping on resume (``ignore_data_skip=True`` — the sampler's cursor is
   what restores position, and letting HF skip on top of that would advance the
   stream twice).

2. **Per-task loss accounting** (D4.3). The loss goes through
   :class:`~src.generalist.mixture.MixtureLoss`, which normalises by the *whole
   optimizer step's* example count. The count comes from
   ``sampler.examples_in_step(k)`` — a pure function of the step — never from the
   micro-batch's row count, because normalising per micro-batch makes a task's
   gradient share depend on how the step happened to be chopped up.

3. **Exactly one normalisation.** transformers 4.50.3 divides the loss returned
   by ``compute_loss`` by ``gradient_accumulation_steps``, but only when
   ``not model_accepts_loss_kwargs and compute_loss_func is None`` (see
   ``Trainer.training_step``). ``MixtureLoss`` has already divided by the step's
   example count, so that second division must not stand. Rather than assume
   which branch this model takes — every model in this repo has ``**kwargs`` in
   its forward, so today the branch is *not* taken, and a backbone without it
   would flip that silently — :meth:`GeneralistTrainer._hf_accumulation_scale`
   evaluates HF's own condition and multiplies back exactly when HF will divide.
   ``tests/generalist/test_resume.py`` pins the invariance under *both* branches.

4. **The schedule, the checkpoint and the resume** (D5.2–D5.4). The schedule is
   segment-based and horizon-free, so :meth:`create_scheduler` ignores HF's
   ``num_training_steps`` on purpose; the checkpoint is finished by
   ``checkpoint.finalize`` with ``COMPLETE`` written last; and
   :meth:`GeneralistTrainer.resume` refuses an incomplete directory, an
   architecture-hash or schema-version mismatch, and a bias-norm fingerprint that
   does not match — the 2026-07-17 adapter-only-reload bug is what that last
   check exists for.

``load_best_model_at_end`` stays off (D5.1): a training run does not select,
forks do (D6), and the reload path is the known bias-restore trap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import TrainerCallback

from . import checkpoint as ckpt_mod
from .mixture import MixtureDataset, MixtureLoss, wrap_collator
from .schedule import Schedule, make_lr_scheduler
from .schema import SCHEMA_VERSION
from ..utils.text_graph_trainer_v2 import GraphTrainerV2

logger = logging.getLogger(__name__)

#: Where a re-warm restarts from, as a fraction of the full LR (D5.2). Low enough
#: that stale Adam moments are re-estimated against the new loss surface before
#: they move the weights far, high enough that a chunk boundary does not cost a
#: full warmup's worth of steps.
REWARM_FROM_FACTOR = 0.1


class TrainerError(RuntimeError):
    """The trainer refuses to run. The message names what would have gone wrong."""


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

def architecture_hash(model) -> str:
    """A hash over every parameter's ``(name, shape, dtype)``.

    D5.4 step 1 refuses a resume across a changed architecture, because the load
    would either raise deep inside ``load_state_dict`` or — worse, with
    ``strict=False`` on the bias path — succeed on the subset that still matches
    and train a model that is half restored. Names and shapes are the whole of
    what "would this checkpoint load" depends on; values are not part of it.
    """
    parts = [f"{name}:{tuple(p.shape)}:{p.dtype}"
             for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0])]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def hardware_name() -> str:
    """The hardware class recorded in ``state.json`` (a D5.4 discontinuity key).

    ``torch.cuda.get_device_name`` when there is a device, a stable literal
    otherwise: the string is compared across a resume, so it must not carry a
    host name, a device index or anything else that changes between two jobs on
    equivalent nodes.
    """
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:                                    # pragma: no cover
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Step-aligned micro-batches
# ─────────────────────────────────────────────────────────────────────────────

def align_to_accumulation(batches: list, target: int) -> list:
    """Reshape one step's micro-batches into exactly ``target`` non-empty ones.

    HF's inner loop is rigid about this: it takes ``gradient_accumulation_steps``
    micro-batches from the iterator and calls ``optimizer.step()`` on a counter
    that is ``(micro-batch index + 1) % gradient_accumulation_steps``. The
    sampler, meanwhile, derives its micro-batch count from a *token* budget
    (D4.4) and a bucket ladder, so a step yields however many batches the bucket
    arithmetic produced — usually near ``accumulation_steps × world_size``, but
    not exactly it. If the two ever disagree, HF's optimizer steps drift out of
    phase with the sampler's steps and every per-task count, every LR lookup and
    every resume position is off by a varying amount. So the stream is reshaped
    here, once, where it is visible.

    Reshaping only ever moves examples between micro-batches; it never adds,
    drops or reorders them, so the step's loss and gradient are unchanged (the
    loss is normalised per example over the whole step, and padding is masked).
    What it does change is padding waste, which is why it merges the *smallest*
    batches and splits the *largest*: both keep the batch sizes as even as the
    bucketing left them.
    """
    if target < 1:
        raise TrainerError(f"gradient_accumulation_steps must be >= 1, got {target}")
    # Carry each example's position in the incoming stream alongside it, so the
    # emission order below can be fixed without the helper having to know what an
    # item is.
    groups, position = [], 0
    for batch in batches:
        if not batch:
            continue
        groups.append([(position + i, item) for i, item in enumerate(batch)])
        position += len(batch)
    if not groups:
        return []
    total = sum(len(g) for g in groups)
    if total < target:
        raise TrainerError(
            f"an optimizer step drew {total} example(s) but "
            f"gradient_accumulation_steps is {target}: there is nothing to put in "
            f"the remaining micro-batches. Lower gradient_accumulation_steps or "
            f"raise tokens_per_step so a step holds at least one example per "
            f"micro-batch.")

    while len(groups) > target:
        groups.sort(key=len)
        smallest = groups.pop(0)
        groups[0].extend(smallest)
    while len(groups) < target:
        groups.sort(key=len, reverse=True)
        biggest = groups.pop(0)
        half = len(biggest) // 2
        groups.extend([biggest[:half], biggest[half:]])
    # Deterministic order: the sorts above key on size alone, so two runs could
    # otherwise emit the same groups in a different order and the "identical
    # batch sequence" guarantee of D4.1 would be a coin flip. Ordering by each
    # group's earliest incoming position also keeps an untouched stream in the
    # order the sampler produced it.
    groups.sort(key=lambda g: min(p for p, _ in g))
    return [[item for _p, item in g] for g in groups]


class StepAlignedBatches(IterableDataset):
    """``MixtureDataset``'s micro-batches, regrouped so each step yields exactly
    ``accumulation_steps`` of them (see :func:`align_to_accumulation`).

    Step boundaries are read off the items themselves (``item["step"]``, which
    ``MixtureDataset`` stamps), not off a counter kept here: the sampler is the
    authority on what a step contains, and a second counter would be one more
    thing that can drift.

    **The reader runs ahead of the trainer, and the checkpoint must not.**
    Regrouping needs a whole step in hand, so the boundary is only visible once
    the first micro-batch of the *next* step has arrived — and ``MixtureDataset``
    draws a step in full before yielding any of it, and HF prefetches an update's
    micro-batches before running them. By the time a checkpoint is written the
    live sampler has therefore drawn one or two steps more than the trainer has
    optimized, and checkpointing ``sampler.state_dict()`` as it stands would
    resume the run past examples it never trained on. So a snapshot is taken at
    each step boundary and filed under the step a resume would start at;
    :meth:`state_at` hands the trainer the one that matches its ``global_step``.
    """

    #: How many boundary snapshots to keep. Two would do — the reader is at most
    #: two steps ahead — and the third is slack, since the cost is a dict of ints.
    KEEP_SNAPSHOTS = 4

    def __init__(self, inner: MixtureDataset, accumulation_steps: int):
        super().__init__()
        self.inner = inner
        self.sampler = inner.sampler
        self.accumulation_steps = int(accumulation_steps)
        self.snapshots: dict = {}

    def state_at(self, step: int):
        """The sampler state a resume at ``step`` needs, or ``None``."""
        return self.snapshots.get(int(step))

    def _snapshot(self, resume_step: int) -> None:
        self.snapshots[resume_step] = self.sampler.state_dict()
        for stale in [k for k in self.snapshots
                      if k < resume_step - self.KEEP_SNAPSHOTS]:
            self.snapshots.pop(stale, None)

    def __iter__(self):
        pending, current = [], None
        for batch in self.inner:
            if not batch:
                continue
            step = int(batch[0]["step"])
            if step != current:
                if pending:
                    yield from align_to_accumulation(pending,
                                                     self.accumulation_steps)
                    pending = []
                current = step
                # The sampler has just finished drawing `step`, so its cursors
                # are exactly what a run that has completed `step` — i.e. one
                # about to draw `step + 1` — must restore.
                self._snapshot(step + 1)
            pending.append(batch)
        if pending:
            yield from align_to_accumulation(pending, self.accumulation_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class _StepAccountingCallback(TrainerCallback):
    """Closes the per-task accumulator at every optimizer step.

    A callback rather than a hook inside ``compute_loss`` because only the
    trainer loop knows where a step ends: ``compute_loss`` runs once per
    micro-batch and cannot tell the last one from the others.
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._close_optimizer_step()


# ─────────────────────────────────────────────────────────────────────────────
# The trainer
# ─────────────────────────────────────────────────────────────────────────────

class GeneralistTrainer(GraphTrainerV2):
    """``GraphTrainerV2`` + the mixture, the schedule and the extended checkpoint.

    Args:
        sampler: a :class:`~src.generalist.mixture.MixtureSampler` already built
            over the resolved mixture and the run's ``get_source``. Its
            ``accumulation_steps`` and ``world_size`` must match the
            ``TrainingArguments``, and are checked here rather than left to
            produce a quietly mis-sized token budget.
        schedule: the live :class:`~src.generalist.schedule.Schedule`. The LR
            lambdas close over this object, so appending a segment on resume is
            reflected without rebuilding the scheduler.
        registry: the :class:`~src.generalist.registry.Registry` the mixture was
            resolved from. Optional; it supplies the per-task ``loss_norm`` table
            and the snapshot written into ``state.json``.
        loss_norm: overrides the table derived from ``registry``.
        rewarm_steps: length of the re-warm segment a discontinuity appends on
            resume (D5.4 step 3). ``None`` means "the run's warmup length", which
            is the only defensible default; a schedule with no warmup segment and
            no explicit value makes a discontinuous resume an error rather than a
            guess.
        rewarm_from: where that re-warm starts, as a fraction of full LR.
        config_hash: the run config's hash, recorded in ``state.json``.
        lineage_hook: ``(entry: dict) -> None``, called where D6's
            ``lineage.py`` will write ``results/lineage.json``. Until that unit
            lands the entries are logged and dropped; nothing here depends on
            them being persisted.
        save_total_limit: how many complete checkpoints to keep. Taken over from
            HF because ``checkpoint.rotate`` is pin-aware and HF's rotation is
            not: a checkpoint a fork was taken from must survive, and HF would
            delete it.
    """

    def __init__(self, *args, sampler=None, schedule: Optional[Schedule] = None,
                 registry=None, loss_norm: Optional[dict] = None,
                 rewarm_steps: Optional[int] = None,
                 rewarm_from: float = REWARM_FROM_FACTOR,
                 config_hash: Optional[str] = None,
                 lineage_hook: Optional[Callable] = None,
                 save_total_limit: Optional[int] = None,
                 **kwargs):
        if sampler is None:
            raise TrainerError("sampler: GeneralistTrainer needs a MixtureSampler")
        if schedule is None:
            raise TrainerError("schedule: GeneralistTrainer needs a Schedule")

        training_args = kwargs.get("args") or (args[1] if len(args) > 1 else None)
        if training_args is not None:
            if training_args.load_best_model_at_end:
                # D5.1. Selection is a fork's job, Tier-B val has been measured to
                # anti-rank arms, and the reload path is the bias-restore trap.
                raise TrainerError(
                    "load_best_model_at_end must stay off on a generalist training "
                    "run (DESIGN.md D5.1): a training run has no selection metric, "
                    "and the best-model reload is the 2026-07-17 bias trap.")
            # The sampler's cursor restores the position on resume (D4.1). HF's
            # own skip-ahead would then advance the stream a second time and the
            # resumed run would silently train on different data.
            training_args.ignore_data_skip = True
            if save_total_limit is None:
                save_total_limit = training_args.save_total_limit
            # Rotation moves to checkpoint.rotate: it is by step number, keeps
            # PINNED checkpoints a fork's lineage points at, and never deletes an
            # incomplete directory another job may still be writing.
            training_args.save_total_limit = None

        self.sampler = sampler
        self.mixture = sampler.mixture
        self.schedule = schedule
        self.registry = registry
        self.rewarm_steps = rewarm_steps
        self.rewarm_from = float(rewarm_from)
        self.config_hash = config_hash
        self.lineage_hook = lineage_hook
        self.save_total_limit = save_total_limit

        self.task_names = {i: name for name, i in sampler.task_ids.items()}
        if loss_norm is None and registry is not None:
            loss_norm = {e.name: registry.get(e.name).loss_norm
                         for e in self.mixture.entries}
        self.mixture_loss = MixtureLoss(loss_norm=loss_norm,
                                        task_ids=sampler.task_ids)

        #: ``[(step, {task: mean loss})]`` — one entry per optimizer step. The
        #: log line only carries the logging-window mean (HF logs every
        #: ``logging_steps``), so the per-step series lives here for the
        #: validators and the run record.
        self.task_loss_history: list = []
        self.examples_per_task: dict = {name: 0 for name in sampler.task_ids}
        self.tokens_seen: int = 0
        self._acc_task_loss: dict = {}
        self._window_task_loss: dict = {}
        self._resumed_from: Optional[str] = None
        self._train_stream: Optional[StepAlignedBatches] = None
        #: The micro-batches of the optimizer step in progress, for the D7
        #: gradient-share readout. Inputs only — see `per_task_loss_fn`.
        self._step_batches: list = []
        self._step_batches_step: Optional[int] = None

        super().__init__(*args, **kwargs)

        if self.args.load_best_model_at_end:                 # pragma: no cover
            raise TrainerError("load_best_model_at_end must stay off (D5.1)")
        if sampler.accumulation_steps != self.args.gradient_accumulation_steps:
            raise TrainerError(
                f"the sampler was built with accumulation_steps="
                f"{sampler.accumulation_steps} but the run uses "
                f"gradient_accumulation_steps="
                f"{self.args.gradient_accumulation_steps}. The sampler divides "
                f"tokens_per_step by that number to size a micro-batch (D4.4), so "
                f"a mismatch silently changes the realised batch.")
        if sampler.world_size != max(int(self.args.world_size), 1):
            raise TrainerError(
                f"the sampler was built with world_size={sampler.world_size} but "
                f"the run has world_size={self.args.world_size}; every rank must "
                f"run an identical sampler for the rank slicing of D4.3 to cover "
                f"the step exactly once.")
        self.add_callback(_StepAccountingCallback(self))

    # ── D4: the dataloader ──────────────────────────────────────────────────

    def get_train_dataloader(self) -> DataLoader:
        """The mixture stream, with HF's batching and sharding both out of the way.

        Three deliberate departures from ``Trainer.get_train_dataloader``:

        * ``batch_size=None`` — the dataset already yields a micro-batch (a list
          of items), because D4.4 derives the size from a token budget and it
          varies from step to step. With automatic batching on, the loader would
          wrap each list in another list of one.
        * ``num_workers=0`` — a worker process gets a *copy* of the sampler, so
          the cursor would advance there and ``sampler.state_dict()`` in the
          parent would checkpoint a position the run never had. ``MixtureDataset``
          refuses more than one worker for a different reason (duplicate
          streams); zero is the only number that also keeps the checkpoint
          honest.
        * no ``accelerator.prepare`` — with more than one rank, accelerate would
          shard the stream a second time on top of ``MixtureDataset``'s own
          ``batches[rank::world_size]``. Device placement is not lost by skipping
          it: ``Trainer._prepare_inputs`` moves the batch in ``training_step``.
        """
        if self.train_dataset is not None:
            logger.info("generalist: train_dataset is ignored; the mixture sampler "
                        "is the data order (D4.1)")
        if self.args.dataloader_num_workers > 0:
            logger.warning(
                "generalist: dataloader_num_workers=%d requested; forcing 0 so the "
                "sampler cursor that the checkpoint writes is the one that drew the "
                "batches", self.args.dataloader_num_workers)

        rank = max(int(self.args.process_index), 0)
        world = max(int(self.args.world_size), 1)
        stream = MixtureDataset(self.sampler, start_step=self.sampler.step,
                                rank=rank, world_size=world, yield_batches=True)
        dataset = StepAlignedBatches(stream, self.args.gradient_accumulation_steps)
        self._train_stream = dataset
        return DataLoader(
            dataset,
            batch_size=None,
            collate_fn=wrap_collator(self.data_collator),
            num_workers=0,
            # Pinning only means anything with a CUDA device to copy into, and a
            # collated graph batch carries `None` columns (magnetic_V and friends
            # when the arm does not use them) that the pinner has to walk past.
            pin_memory=bool(self.args.dataloader_pin_memory
                            and torch.cuda.is_available()),
        )

    # ── D5.2: the schedule ──────────────────────────────────────────────────

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Build the segment schedule's ``LambdaLR``; ``num_training_steps`` is unused.

        That argument is HF's horizon, and the trunk has none (D5.2): the run is
        a warmup into an open-ended stable phase, extended by a re-warm on every
        discontinuous resume and closed only by an anneal fork. Taking the
        argument and ignoring it is the honest shape here — HF passes
        ``max_steps``, which is a *chunk* length, and using it would make the LR
        depend on how the run was sliced into Slurm jobs.

        ``self.optimizer`` is ``GraphTrainerV2``'s, so the bias groups already
        carry ``is_bias: True`` and :func:`make_lr_scheduler` gives them the
        ``bias_factor`` lambda while everything else gets ``factor``.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = make_lr_scheduler(
                self.optimizer if optimizer is None else optimizer, self.schedule)
        return self.lr_scheduler

    # ── D4.3: the loss ──────────────────────────────────────────────────────

    def _hf_accumulation_scale(self) -> float:
        """What to multiply the loss by so exactly one normalisation survives.

        ``Trainer.training_step`` (4.50.3, line ~3756) ends with::

            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

        ``MixtureLoss`` has already divided by the *whole step's* example count,
        so a micro-batch's contribution is its share of the step and the
        micro-batches sum to the step's loss. HF's division would shrink that by
        another factor of ``gradient_accumulation_steps`` — the classic
        double-division — so it is multiplied back here, under HF's own
        condition rather than under an assumption about it. Every model in this
        repo has ``**kwargs`` in its forward, which makes
        ``model_accepts_loss_kwargs`` true and the branch dead; a backbone
        without it would flip that with no other symptom than a silently smaller
        effective LR under accumulation.
        """
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            return float(self.args.gradient_accumulation_steps)
        return 1.0

    def _token_losses(self, model, inputs: dict, labels):
        """``(token_losses, mask, outputs)`` for one micro-batch.

        Split out of :meth:`compute_loss` so the gradient-share readout can run
        the same forward and the same shift on a stored micro-batch; a readout
        that shifted differently would be measuring a different loss than the
        one the step took a gradient of.
        """
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Shift once, here. The collator places the answer span inside the prompt
        # node's tokens and HF's convention is that position t predicts t+1, so
        # the mask and the losses have to be built on the same shifted view or the
        # supervised span moves by one token.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)

        flat_labels = shift_labels.reshape(-1)
        flat_mask = mask.reshape(-1)
        if not bool(flat_mask.any()):
            raise TrainerError(
                "a training micro-batch has no supervised token; either the "
                "collator dropped the label span or an adapter emitted an example "
                "with an empty answer")
        # Cross-entropy on the supervised positions only. The answer span is a
        # handful of tokens out of a packed sequence, and the dense form would
        # materialise a (B·L, |V|) float32 tensor for them — 2 GB at 1B scale.
        selected = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1])[flat_mask].float(),
            flat_labels[flat_mask], reduction="none")
        token_losses = torch.zeros(flat_labels.shape, dtype=selected.dtype,
                                   device=selected.device)
        token_losses = token_losses.masked_scatter(flat_mask, selected)
        token_losses = token_losses.view(shift_labels.shape)
        return token_losses, mask, outputs

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        """Per-example loss, normalised over the whole optimizer step (D4.3).

        ``num_items_in_batch`` is HF's *token* count for the update and is
        deliberately unused: D7a normalises per example, and the example count
        comes from ``sampler.examples_in_step(k)`` — a pure function of the step
        index, so it is the same number whether the step was run as one
        micro-batch or four, on one rank or eight.

        A batch without ``task_ids`` is an evaluation batch (the eval loaders do
        not go through the mixture), and falls back to the model's own loss.
        """
        if "task_ids" not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

        task_ids = inputs.pop("task_ids")
        example_index = inputs.pop("example_index", None)
        step_col = inputs.pop("step", None)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise TrainerError("a training batch reached compute_loss without labels")

        step = self._step_of(step_col)
        examples_in_step = self.sampler.examples_in_step(step)

        token_losses, mask, outputs = self._token_losses(model, inputs, labels)
        loss, per_task = self.mixture_loss(token_losses, mask, task_ids,
                                           examples_in_step)

        # The gradient-share readout (D7) needs a task's rows of real micro-
        # batches as something it can backward through, and only the trainer is
        # ever holding them: `MixtureSampler.draw_step` is a cursor, so drawing a
        # batch inside a validator would advance the run being measured.
        #
        # The whole step is kept, not the last micro-batch of it. A share
        # measured on one micro-batch is a share of four examples out of the
        # step's twenty-five: it swings by a factor of five between firings, and
        # a task with 0.3 of the mixture is *absent* from it often enough to
        # report `nan` about a third of the time. Neither of T10's two claims —
        # the share tracks the weight, no task is at zero — survives that. What
        # is kept is the inputs, never the graph: each graph here is freed by the
        # backward that follows it, and `per_task_loss_fn` runs its own forward
        # per micro-batch, one at a time. One step of activation-free inputs is a
        # few hundred kilobytes.
        if step != self._step_batches_step:
            self._step_batches = []
            self._step_batches_step = step
        self._step_batches.append({"inputs": inputs, "labels": labels,
                                   "task_ids": task_ids,
                                   "examples_in_step": examples_in_step})

        self._record_micro_batch(per_task, inputs, step)
        loss = loss * self._hf_accumulation_scale()
        return (loss, outputs) if return_outputs else loss

    def _micro_batch_loss(self, batch: dict, rows):
        """One micro-batch's contribution to ``task``'s term of the step loss."""
        token_losses, mask, _ = self._token_losses(
            self.model, batch["inputs"], batch["labels"])
        per_example = self.mixture_loss.per_example_losses(
            token_losses, mask, batch["task_ids"])
        # The same normalisation the step used: a task's rows divided by the
        # whole step's example count, so summing over the step's micro-batches
        # reproduces exactly the term `MixtureLoss` contributed for that task.
        return per_example[rows].sum() / float(batch["examples_in_step"])

    def per_task_loss_fn(self) -> Callable:
        """The closure ``measure_grad_share`` backwards through, one task at a time.

        The readout wants each task's contribution to the loss of *real* training
        micro-batches, still attached to a graph. Only the trainer can produce
        one: a validator that drew its own batch would move
        ``MixtureSampler.draw_step``, which is a cursor, and desynchronise the run
        it is measuring. So the trainer keeps the step's micro-batch inputs and
        this closure runs a fresh forward over them — the training step's own
        graphs are gone by then, freed by the backwards that followed them.

        **The whole step, one micro-batch at a time.** ``loss_for(task)`` returns
        a *generator* of scalars, one per micro-batch that holds a row of that
        task, and it is lazy on purpose: the caller backwards through each scalar
        before asking for the next, so exactly one graph is alive at a time. That
        is not a refinement — ``measure_grad_share`` cannot retain a graph at all,
        because on the flex path the backward is compiled and a compiled backward
        with donated buffers refuses ``retain_graph=True`` outright (measured, not
        anticipated, at step 50 of the first smoke run). Summing the scalars first
        and backwarding once would hold the whole step's activations and reproduce
        that failure; yielding them does neither.

        A task with no rows anywhere in the step returns ``None`` before paying
        for a forward at all. Skipped, not zero: a zero share would read as "this
        task contributes nothing to the gradient" rather than "the step did not
        sample it".

        The forward goes through ``self.model``, not the DDP wrapper, so the
        readout is per-rank and does not trip an all-reduce that no other rank is
        waiting on. Dropout is left in whatever mode the caller has the model in;
        the validators run this between steps, where the model is still in train
        mode, which is the mode the measured gradient was taken in.
        """
        def loss_for(task: str):
            if not self._step_batches:
                raise TrainerError(
                    "grad_share fired before any training micro-batch had run, so "
                    "there is nothing to attribute a gradient to; move its cadence "
                    "past step 0")
            task_id = self.sampler.task_ids.get(task)
            if task_id is None:
                return None
            present = [(batch, batch["task_ids"] == task_id)
                       for batch in self._step_batches]
            present = [(batch, rows) for batch, rows in present if bool(rows.any())]
            if not present:
                return None
            return (self._micro_batch_loss(batch, rows) for batch, rows in present)

        return loss_for

    def per_task_batch_counts(self) -> dict:
        """``{task: examples}`` over the step :meth:`per_task_loss_fn` measures.

        The gradient-share readout is a statement about a finite sample, and how
        many examples of each task were in it is the difference between a share
        worth reading and noise. Because the sample is now the whole optimizer
        step, these are the step's own per-task example counts, and their shares
        are what the mixture actually drew — which is the number a share of the
        gradient should be compared against, rather than the configured weight it
        is a noisy draw from. Returned as a plain dict so the validator can report
        it without reaching into the trainer.
        """
        counts: dict = {}
        for batch in self._step_batches:
            for tid in batch["task_ids"].tolist():
                name = self.task_names.get(int(tid))
                if name is not None:
                    counts[name] = counts.get(name, 0) + 1
        return counts

    def _step_of(self, step_col) -> int:
        """The optimizer step a micro-batch belongs to, from the batch itself.

        ``MixtureDataset`` stamps every item with the step that drew it, so the
        batch carries the answer and no counter has to be trusted. It is checked
        against HF's ``global_step`` because a disagreement means the two loops
        have fallen out of phase, and every count and LR lookup downstream would
        be attributed to the wrong step.
        """
        if step_col is None:
            raise TrainerError("a mixture batch arrived without its 'step' column")
        step = int(step_col.reshape(-1)[0])
        if step != int(self.state.global_step):
            raise TrainerError(
                f"micro-batch belongs to sampler step {step} but the trainer is on "
                f"optimizer step {self.state.global_step}. The dataloader and the "
                f"accumulation counter have fallen out of phase — every per-task "
                f"count, LR and resume position from here on would be wrong.")
        return step

    def _record_micro_batch(self, per_task: dict, inputs: dict, step: int) -> None:
        for task_id, (total, n) in per_task.items():
            acc_total, acc_n = self._acc_task_loss.get(task_id, (0.0, 0))
            self._acc_task_loss[task_id] = (acc_total + total, acc_n + n)
        mask = inputs.get("attention_mask")
        if mask is not None:
            self.tokens_seen += int(mask.sum())

    def _close_optimizer_step(self) -> None:
        """Fold the step's per-task sums into the history and the log window."""
        step = int(self.state.global_step)
        means = {}
        for task_id, (total, n) in self._acc_task_loss.items():
            name = self.task_names.get(task_id, str(task_id))
            self.examples_per_task[name] = self.examples_per_task.get(name, 0) + n
            if n:
                means[name] = total / n
            w_total, w_n = self._window_task_loss.get(name, (0.0, 0))
            self._window_task_loss[name] = (w_total + total, w_n + n)
        self.task_loss_history.append((step, means))
        self._acc_task_loss = {}

    def log(self, logs: dict, *args, **kwargs) -> None:
        """Add the window's per-task mean loss to the training log line.

        Only to the *training* line (the one carrying ``loss``): an evaluation
        log with training losses folded in would put two different quantities
        under names that look like one family.
        """
        if "loss" in logs and self._window_task_loss:
            for name, (total, n) in sorted(self._window_task_loss.items()):
                if n:
                    logs[f"task_loss/{name}"] = round(total / n, 6)
                logs[f"task_examples/{name}"] = n
            self._window_task_loss = {}
        super().log(logs, *args, **kwargs)

    # ── D5.3: checkpoints ───────────────────────────────────────────────────

    def run_state(self) -> dict:
        """The identity half of ``state.json``: what a resume compares (D5.4).

        Kept apart from :meth:`checkpoint_state` because the discontinuity check
        runs *before* there is a checkpoint to write — it compares this dict
        against the one the parent checkpoint recorded.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "mixture_hash": self.mixture.hash(),
            "tokens_per_step": int(self.mixture.tokens_per_step),
            "lr": float(self.args.learning_rate),
            "bias_lr": float(self.bias_lr if self.bias_lr is not None
                             else self.args.learning_rate),
            "hardware": hardware_name(),
            "architecture_hash": architecture_hash(self.model),
            "config_hash": self.config_hash,
        }

    def sampler_state_for_checkpoint(self) -> dict:
        """The sampler position *the trainer* is at, not the one the reader is at.

        See :class:`StepAlignedBatches`: the dataloader is one or two steps ahead
        of the optimizer by construction, so the live ``state_dict()`` would
        resume the run past examples it never trained on. The stream files a
        snapshot at every step boundary under the step a resume would start from;
        this picks the one matching ``global_step``.
        """
        stream = getattr(self, "_train_stream", None)
        snapshot = stream.state_at(self.state.global_step) if stream else None
        if snapshot is not None:
            return snapshot
        logger.warning(
            "generalist: no sampler snapshot for step %d; falling back to the live "
            "cursor, which may be ahead of the trainer. A resume from this "
            "checkpoint could skip examples.", self.state.global_step)
        return self.sampler.state_dict()

    def checkpoint_state(self) -> dict:
        """Everything ``state.json`` carries; ``finalize`` adds the fingerprint.

        The mixture is written out entry by entry, not just as its hash. A hash
        answers "is this the same mixture" and nothing else, and an ``anneal`` or
        ``admit`` fork (D6) has to *train the parent's mixture* — so without the
        entries a fork has to be told what the parent was doing, by a caller that
        may not know. With them a fork is self-contained from a checkpoint path,
        which is what ``resume --from latest`` already is.
        """
        state = self.run_state()
        state.update({
            "mixture_entries": [
                {"name": e.name, "weight": float(e.weight),
                 "passes": int(e.passes), "cap_per_pass": e.cap_per_pass}
                for e in self.mixture.entries],
            "step": int(self.state.global_step),
            "global_step": int(self.state.global_step),
            "examples_per_task": dict(self.examples_per_task),
            "tokens_seen": int(self.tokens_seen),
            "seed": int(self.sampler.seed),
            "schedule_position": list(self.schedule.position(self.state.global_step)),
            "resumed_from": self._resumed_from,
        })
        if self.registry is not None:
            state["registry_snapshot"] = self.registry.snapshot()
        return state

    def _save_checkpoint(self, model, trial):
        """HF's checkpoint, then ours, then ``COMPLETE``.

        Order matters twice over: ``finalize`` records the directory's file list
        and the bias norm as they stand when it runs, so it must come after HF's
        writes and after ``GraphTrainerV2.save_model``'s ``bias_parameters.pt``
        (which ``super()._save_checkpoint`` triggers); and ``COMPLETE`` is the
        last byte written, which is what makes a chunk killed mid-write
        unresumable rather than subtly wrong.
        """
        super()._save_checkpoint(model, trial)

        run_dir = self._get_output_dir(trial=trial)
        ckpt_dir = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")
        if not self.args.should_save:
            return
        ckpt_mod.finalize(
            ckpt_dir,
            model=self.model,
            active_params=self.active_params,
            schedule=self.schedule,
            sampler_state=self.sampler_state_for_checkpoint(),
            state=self.checkpoint_state(),
        )
        if self.save_total_limit:
            ckpt_mod.rotate(run_dir, keep=int(self.save_total_limit))

    # ── D5.4: resume ────────────────────────────────────────────────────────

    def prepare_resume(self, from_: str = "latest") -> str:
        """D5.4 steps 1–3: refuse, restore, and re-warm if anything moved.

        Returns the resolved checkpoint directory, which the caller hands to
        ``train(resume_from_checkpoint=...)`` — the model, optimizer and RNG come
        back through HF and ``GraphTrainerV2._load_from_checkpoint``, the
        schedule and the sampler come back here.

        The three refusals are all cases where continuing produces *numbers*
        rather than a crash: an incomplete directory is a partial write; a
        different architecture hash loads a subset of the parameters and trains
        the rest from init; a different schema version masks the loss span
        differently, which moves every metric without moving anything visible.
        """
        ckpt_dir = self._resolve_checkpoint(from_)

        # (1) whole, and every file it claims still present. Without `model=` this
        # is the structural check only; the bias fingerprint is checked below,
        # after the hashes, so a mismatched architecture is named as such rather
        # than surfacing as a confusing load error.
        prev = ckpt_mod.verify(ckpt_dir)

        current = self.run_state()
        if prev.get("architecture_hash") not in (None, current["architecture_hash"]):
            raise ckpt_mod.CheckpointError(
                f"{ckpt_dir} was written by a model with architecture hash "
                f"{prev.get('architecture_hash')!r}; this run's is "
                f"{current['architecture_hash']!r}. The parameter shapes differ, so "
                "the restore would load a subset and train the rest from init.")
        if str(prev.get("schema_version")) != str(current["schema_version"]):
            raise ckpt_mod.CheckpointError(
                f"{ckpt_dir} was trained under example schema version "
                f"{prev.get('schema_version')!r}, this run renders version "
                f"{current['schema_version']!r}. A different render masks a "
                "different span, which shifts every metric silently rather than "
                "raising.")

        # (2) the bias tensors load and their norm matches the fingerprint. This
        # is the load, not an extra one — `verify` loads them into the model.
        ckpt_mod.verify(ckpt_dir, model=self.model,
                        active_params=self.active_params)

        schedule, sampler_state, _state = ckpt_mod.restore_extras(ckpt_dir)
        self.schedule = schedule
        if self.lr_scheduler is not None:
            # The lambdas close over the Schedule object; a restore replaces the
            # object, so a scheduler already built would keep serving the old one.
            self.lr_scheduler = None
        if sampler_state:
            self.sampler.load_state_dict(sampler_state)

        self.examples_per_task.update(prev.get("examples_per_task") or {})
        self.tokens_seen = int(prev.get("tokens_seen") or 0)
        self._resumed_from = ckpt_dir

        # (3) discontinuities: legal, recorded, and each one costs a re-warm.
        changed = ckpt_mod.discontinuities(prev, current)
        at = int(prev.get("step") if prev.get("step") is not None
                 else self.sampler.step)
        rewarm_steps = self._maybe_rewarm(at, changed)
        self._note_lineage({
            "event": "resume",
            "parent": ckpt_dir,
            "parent_step": int(prev.get("step") or 0),
            "causes": changed,
            "rewarm_steps": rewarm_steps,
            "schedule": self.schedule.to_json(),
        })
        return ckpt_dir

    def _maybe_rewarm(self, at_step: int, causes) -> int:
        """Append the re-warm a discontinuity calls for; return its length.

        Zero when nothing changed, and zero when the run has not reached its
        open-ended stable phase yet: a schedule can only be extended past its
        last segment, and a run still inside its warmup is *already* climbing
        from a low LR — a second warmup laid on top of the first would be a
        strictly worse version of what is already happening.
        """
        if not causes:
            return 0
        last = self.schedule.segments[-1]
        if not (last.is_open and at_step >= last.start):
            logger.warning(
                "generalist: resuming across %s at step %d, which is still inside "
                "the %s segment; no re-warm appended (the run has not reached full "
                "LR yet)", ", ".join(causes), at_step, last.kind)
            return 0
        steps = self._resolved_rewarm_steps(causes)
        self.schedule.append_rewarm(at_step=at_step, rewarm_steps=steps,
                                    from_factor=self.rewarm_from)
        logger.warning(
            "generalist: resuming across %s; appended a %d-step re-warm from "
            "%.3g x LR at step %d", ", ".join(causes), steps, self.rewarm_from,
            at_step)
        return steps

    def resume(self, from_: str = "latest", **train_kwargs):
        """:meth:`prepare_resume` and then continue training (D5.4 step 4)."""
        ckpt_dir = self.prepare_resume(from_)
        return self.train(resume_from_checkpoint=ckpt_dir, **train_kwargs)

    def _resolve_checkpoint(self, from_: str) -> str:
        if from_ in (None, "latest"):
            found = ckpt_mod.latest(self.args.output_dir)
            if found is None:
                raise ckpt_mod.CheckpointError(
                    f"resume --from latest: no complete checkpoint under "
                    f"{self.args.output_dir}")
            return found
        if not os.path.isdir(from_):
            raise ckpt_mod.CheckpointError(f"{from_} is not a directory")
        return from_

    def _resolved_rewarm_steps(self, causes) -> int:
        if self.rewarm_steps:
            return int(self.rewarm_steps)
        for seg in self.schedule.segments:
            if seg.kind == "warmup" and seg.steps:
                return int(seg.steps)
        raise TrainerError(
            f"resuming across {', '.join(causes)} needs a re-warm (D5.4 step 3) but "
            "the run configures no rewarm_steps and its schedule has no warmup "
            "segment to take a length from. Set rewarm_steps explicitly.")

    def _note_lineage(self, entry: dict) -> None:
        """Where D6's ``lineage.py`` will append to ``results/lineage.json``.

        Kept as a hook so the resume path is complete before that unit lands:
        with no hook configured the entry is logged and dropped, and nothing in
        the resume depends on it having been written.
        """
        if self.lineage_hook is not None:
            self.lineage_hook(entry)
        else:
            logger.info("generalist: lineage entry (no sink configured yet): %s",
                        json.dumps(entry, default=str, sort_keys=True))
