"""Horizon-free WSD schedule as absolute-step segments (DESIGN.md D5.2).

HF's ``warmup_stable_decay`` needs ``warmup + stable + decay == num_training_steps``
and silently runs the remainder at ``min_lr`` when that does not hold. A trunk has
no horizon: it is trained in chunks, resumed, re-warmed when the mixture changes,
and only *forked* runs ever decay. So the schedule here is a list of segments in
**absolute steps**, the last of which may be open-ended, and a resume rebuilds the
lambdas from the serialised segments instead of from a step count that does not
exist.

Two properties are worth stating because everything else follows from them:

* A segment carries a **factor**, not a learning rate. The factor multiplies each
  parameter group's own base LR, so one schedule serves both the LoRA group (at
  ``lr``) and the bias group (at ``bias_lr``) and their ratio is preserved for
  free — no place in the code has to know what ``bias_lr / lr`` is.
* ``ratio_start`` / ``ratio_end`` are a second multiplier applied to the bias
  groups only. They default to 1.0, which is exactly today's behaviour (constant
  ratio), and they are the hook for decaying the ratio toward 1 on the trunk
  (PLAN.md §5) without inventing a mechanism then.

The schedule's position is ``(segment index, step within segment)``: once
segments have been appended, ``global_step`` alone no longer identifies where the
run is, because the same step number means different things before and after a
truncation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional


SCHEDULE_FORMAT_VERSION = 1

KINDS = ("warmup", "stable", "rewarm", "decay")
SHAPES = ("linear", "cosine")


class ScheduleError(ValueError):
    """A schedule was built, mutated or queried outside its contract."""


@dataclass
class Segment:
    """One contiguous stretch of steps with a linear (or cosine) factor ramp.

    ``start`` is absolute. ``steps is None`` marks an open-ended segment, which is
    legal only as the last segment and only for ``kind == "stable"`` — an
    open-ended warmup or decay has no meaning, and an open segment in the middle
    would make every later segment's start undefined.
    """

    kind: str
    start: int
    steps: Optional[int]
    factor_start: float
    factor_end: float
    ratio_start: float = 1.0
    ratio_end: float = 1.0
    shape: str = "linear"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ScheduleError(f"unknown segment kind {self.kind!r}; expected one of {KINDS}")
        if self.shape not in SHAPES:
            raise ScheduleError(f"unknown segment shape {self.shape!r}; expected one of {SHAPES}")
        if self.start < 0:
            raise ScheduleError(f"segment start must be >= 0, got {self.start}")
        if self.steps is None:
            if self.kind != "stable":
                raise ScheduleError(f"only a stable segment may be open-ended, not {self.kind!r}")
        elif self.steps <= 0:
            raise ScheduleError(f"segment steps must be > 0 or None, got {self.steps}")

    @property
    def is_open(self) -> bool:
        return self.steps is None

    @property
    def end(self) -> Optional[int]:
        return None if self.steps is None else self.start + self.steps

    def contains(self, step: int) -> bool:
        if step < self.start:
            return False
        return self.steps is None or step < self.start + self.steps

    def _t(self, step: int) -> float:
        """Progress through the segment in [0, 1]; 0 for an open segment."""
        if self.steps is None or self.steps <= 0:
            return 0.0
        t = (step - self.start) / self.steps
        return min(max(t, 0.0), 1.0)

    def factor(self, step: int) -> float:
        return _interp(self.factor_start, self.factor_end, self._t(step), self.shape)

    def ratio(self, step: int) -> float:
        # The ratio always moves linearly. Its shape is not a knob anyone has
        # asked for, and a cosine ratio under a cosine decay would be two curves
        # to reason about instead of one.
        return _interp(self.ratio_start, self.ratio_end, self._t(step), "linear")


def _interp(a: float, b: float, t: float, shape: str) -> float:
    if shape == "cosine":
        return b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * t))
    return a + (b - a) * t


@dataclass
class Schedule:
    """An ordered, contiguous list of segments starting at step 0."""

    segments: list = field(default_factory=list)

    def __post_init__(self):
        self.segments = [s if isinstance(s, Segment) else Segment(**s) for s in self.segments]
        self._validate()

    # ── invariants ──────────────────────────────────────────────────────────

    def _validate(self):
        if not self.segments:
            raise ScheduleError("a schedule needs at least one segment")
        if self.segments[0].start != 0:
            raise ScheduleError(f"the first segment must start at 0, got {self.segments[0].start}")
        for i, seg in enumerate(self.segments):
            last = i == len(self.segments) - 1
            if seg.is_open and not last:
                raise ScheduleError(f"segment {i} ({seg.kind}) is open-ended but is not the last")
            if not last:
                nxt = self.segments[i + 1]
                if nxt.start != seg.end:
                    raise ScheduleError(
                        f"segments {i} and {i + 1} are not contiguous: "
                        f"{seg.kind} ends at {seg.end}, {nxt.kind} starts at {nxt.start}")

    # ── queries ─────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True while the schedule can still be extended (D5.2: rewarm/decay)."""
        return self.segments[-1].is_open

    def end_step(self) -> Optional[int]:
        """Last step the schedule covers, or None while it is open-ended."""
        return self.segments[-1].end

    def position(self, step: int) -> tuple:
        """``(segment index, step within segment)`` for an absolute step.

        Past the end of a closed schedule the position stays in the last segment
        with an offset beyond its length — a decayed run that is still being
        stepped is at ``lr_min``, not off the schedule.
        """
        step = self._check_step(step)
        for i, seg in enumerate(self.segments):
            if seg.contains(step):
                return i, step - seg.start
        last = len(self.segments) - 1
        return last, step - self.segments[last].start

    def factor(self, step: int) -> float:
        """LR multiplier for the LoRA (and any non-bias) group."""
        step = self._check_step(step)
        for seg in self.segments:
            if seg.contains(step):
                return seg.factor(step)
        return self.segments[-1].factor_end

    def bias_factor(self, step: int) -> float:
        """LR multiplier for the bias groups: ``factor × ratio``."""
        step = self._check_step(step)
        for seg in self.segments:
            if seg.contains(step):
                return seg.factor(step) * seg.ratio(step)
        last = self.segments[-1]
        return last.factor_end * last.ratio_end

    def _check_step(self, step: int) -> int:
        step = int(step)
        if step < 0:
            raise ScheduleError(f"step must be >= 0, got {step}")
        return step

    # ── mutation: the three things a run does to its own schedule ───────────

    @classmethod
    def training(cls, warmup_steps: int, lr_min_factor: float = 0.0) -> "Schedule":
        """A fresh training run: linear warmup into an open-ended stable phase.

        ``lr_min_factor`` is where the warmup starts (0 by default, i.e. LR 0 at
        step 0). ``warmup_steps == 0`` means no warmup segment at all, so step 0
        already runs at the full LR.
        """
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ScheduleError(f"warmup_steps must be >= 0, got {warmup_steps}")
        segments = []
        if warmup_steps > 0:
            segments.append(Segment("warmup", 0, warmup_steps, float(lr_min_factor), 1.0))
        segments.append(Segment("stable", warmup_steps, None, 1.0, 1.0))
        return cls(segments)

    def append_rewarm(self, at_step: int, rewarm_steps: int, from_factor: float,
                      ratio_end: float = 1.0) -> "Schedule":
        """Truncate the open stable at ``at_step`` and re-warm back to full LR.

        This is what a resume does when it finds a discontinuity (D5.4 step 3):
        the mixture, the token budget, the LRs or the hardware changed, so the
        optimizer's moments no longer describe the loss surface it is about to
        see. The run continues in a new open-ended stable phase afterwards.
        """
        rewarm_steps = int(rewarm_steps)
        if rewarm_steps <= 0:
            raise ScheduleError(f"rewarm_steps must be > 0, got {rewarm_steps}")
        at_step, ratio_at = self._truncate_open_stable(at_step)
        self.segments.append(Segment("rewarm", at_step, rewarm_steps,
                                     float(from_factor), 1.0,
                                     ratio_start=ratio_at, ratio_end=float(ratio_end)))
        self.segments.append(Segment("stable", at_step + rewarm_steps, None, 1.0, 1.0,
                                     ratio_start=float(ratio_end), ratio_end=float(ratio_end)))
        self._validate()
        return self

    def append_decay(self, at_step: int, decay_steps: int, min_factor: float,
                     shape: str = "cosine", ratio_end: Optional[float] = None) -> "Schedule":
        """Truncate the open stable at ``at_step`` and decay to ``min_factor``.

        After this the schedule is CLOSED: nothing more can be appended, and the
        factor past ``at_step + decay_steps`` stays at ``min_factor``. This is the
        anneal fork (D6) — the parent keeps its own, still-open schedule, because
        a fork copies the checkpoint rather than continuing it.
        """
        decay_steps = int(decay_steps)
        if decay_steps <= 0:
            raise ScheduleError(f"decay_steps must be > 0, got {decay_steps}")
        if shape not in SHAPES:
            raise ScheduleError(f"unknown decay shape {shape!r}; expected one of {SHAPES}")
        at_step, ratio_at = self._truncate_open_stable(at_step)
        factor_at = 1.0  # the stable phase is at full LR by construction
        self.segments.append(Segment("decay", at_step, decay_steps, factor_at, float(min_factor),
                                     ratio_start=ratio_at,
                                     ratio_end=ratio_at if ratio_end is None else float(ratio_end),
                                     shape=shape))
        self._validate()
        return self

    def _truncate_open_stable(self, at_step: int) -> tuple:
        """Cut the trailing open stable segment at ``at_step``.

        Returns ``(at_step, ratio at that step)``. A zero-length remainder drops
        the stable segment entirely rather than leaving a 0-step segment, which
        would break the "steps > 0" invariant every other query relies on.
        """
        at_step = int(at_step)
        last = self.segments[-1]
        if not last.is_open:
            raise ScheduleError(
                "the schedule is closed (it ends with a "
                f"{last.kind} segment); nothing can be appended to it")
        if at_step < last.start:
            raise ScheduleError(
                f"cannot append at step {at_step}: the open stable segment starts at "
                f"{last.start}, so that step lies in an earlier, already-fixed segment")
        ratio_at = last.ratio(at_step)
        if at_step == last.start:
            self.segments.pop()
            if not self.segments:
                # Dropping the only segment would leave an empty schedule; the
                # appended segment will start at 0, which is still valid.
                pass
        else:
            last.steps = at_step - last.start
        return at_step, ratio_at

    # ── serialisation ───────────────────────────────────────────────────────

    def to_json(self) -> dict:
        """JSON-ready dict; ``checkpoint.finalize`` writes it as schedule.json."""
        return {"version": SCHEDULE_FORMAT_VERSION,
                "segments": [asdict(s) for s in self.segments]}

    @classmethod
    def from_json(cls, data) -> "Schedule":
        """Rebuild from ``to_json``'s dict (or the same thing as a JSON string)."""
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        version = data.get("version", SCHEDULE_FORMAT_VERSION)
        if version != SCHEDULE_FORMAT_VERSION:
            raise ScheduleError(
                f"schedule.json is format version {version}, this code writes "
                f"{SCHEDULE_FORMAT_VERSION}")
        return cls([Segment(**s) for s in data["segments"]])

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{s.kind}@{s.start}+{'inf' if s.steps is None else s.steps}"
            f"[{s.factor_start:g}->{s.factor_end:g}]" for s in self.segments)
        return f"Schedule({parts})"


#: Built on first use by :func:`_scheduler_class`. This module stays importable
#: without torch — `validate` mode resolves a mixture and prints a schedule on a
#: login node — so the subclass cannot be defined at module level.
_SCHEDULER_CLASS = None


def _scheduler_class():
    """``LambdaLR`` that keeps the live :class:`Schedule` out of ``scheduler.pt``.

    The lambdas have to reach the schedule somehow, and hanging it on the
    scheduler as ``gtlm_schedule`` is the readable way to do it — but
    ``LRScheduler.state_dict`` copies the whole of ``__dict__`` except the
    optimizer and the lambdas, so a plain attribute *is* in the saved state. The
    schedule would then be pickled into ``scheduler.pt``, and HF's resume loads
    that file with torch's ``weights_only=True``, which refuses to unpickle a
    class it was never told about. Every resume would fail on a file nothing
    reads: the schedule's own record is ``schedule.json``, and the only thing
    ``scheduler.pt`` has to carry across is ``last_epoch``.

    ``load_state_dict`` drops the key too, so a checkpoint written before this
    still loads.
    """
    global _SCHEDULER_CLASS
    if _SCHEDULER_CLASS is None:
        from torch.optim.lr_scheduler import LambdaLR

        class ScheduleLambdaLR(LambdaLR):
            __doc__ = _scheduler_class.__doc__

            def state_dict(self) -> dict:
                state = super().state_dict()
                state.pop("gtlm_schedule", None)
                return state

            def load_state_dict(self, state_dict: dict) -> None:
                super().load_state_dict({k: v for k, v in state_dict.items()
                                         if k != "gtlm_schedule"})

        _SCHEDULER_CLASS = ScheduleLambdaLR
    return _SCHEDULER_CLASS


def make_lr_scheduler(optimizer, schedule: Schedule):
    """``LambdaLR`` with one lambda per parameter group.

    Groups marked ``is_bias`` (that is how ``GraphTrainerV2.create_optimizer``
    tags the graph-bias groups) follow ``bias_factor``; everything else follows
    ``factor``. Each lambda multiplies its *own* group's base LR, so the LoRA
    group runs at ``lr × factor`` and the bias group at ``bias_lr × bias_factor``.

    This is the object handed to HF as ``optimizers=(opt, sched)``. HF saves the
    returned scheduler's ``state_dict`` into ``scheduler.pt``; the lambdas are
    rebuilt from ``schedule.json`` on resume, so all that state_dict has to carry
    across is ``last_epoch``.
    """
    def _factor(step, _s=schedule):
        return _s.factor(step)

    def _bias_factor(step, _s=schedule):
        return _s.bias_factor(step)

    lambdas = [_bias_factor if g.get("is_bias", False) else _factor
               for g in optimizer.param_groups]
    sched = _scheduler_class()(optimizer, lr_lambda=lambdas)
    sched.gtlm_schedule = schedule
    return sched
