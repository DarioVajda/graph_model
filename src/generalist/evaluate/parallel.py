"""
D7 under DDP: shard a validator's work across the ranks, gather the metrics.

Training is sharded and evaluation was not. Every rank held the same eval sets,
drew the same ``eval_indices`` and scored all of them, so a four-rank run did the
scoring pass four times and reported one copy of it. That is invisible in the
metrics — every rank computes the same number — and it is the whole cost of a
milestone: the first firing of the arm-2 campaign took **133 minutes** against
34 minutes of training for the 2,000 steps it interrupted.

**The unit of sharding is a whole scoring target, never a slice of one.** A
validator hands :func:`run_sharded` the list of independent things it has to
score — a ``(task, split)`` pair, one view of a split, one task's permutation
sweep — and each of those is computed end to end on exactly one rank, from the
same ``eval_indices`` and in the same batches it would have used alone. So a
metric is bit-identical to what a single-rank run produces, and the four-rank
number can be checked against the one-rank number rather than merely believed.
Splitting the *rows* of one target would not have that property: the batch a row
lands in decides its padding and its neighbours in a bf16 reduction, and the
margin moves by around an eighth on this model (`builtin.PermSpread`'s control
measures exactly that). It would also make the derived quantities — an AUROC, a
majority-class floor — a combination of partial statistics rather than the
statistic itself.

**Balance.** Targets are wildly unequal: a teacher-forced yes/no over 500 rows is
one forward pass per batch, while 500 captions at 128 new tokens each is 500
prefills and 64,000 decode steps. Round-robin over a sorted list would put the
two generative tasks on one rank and leave the others idle, so the assignment is
longest-processing-time-first against a declared cost — the classic greedy 4/3
bound, and far closer than that on this distribution. The cost is an estimate and
only ever decides *placement*; nothing it gets wrong can move a number.

**Failure, and why it cannot hang the run.** A collective that one rank reaches
and another does not deadlocks until the watchdog kills the job, which would turn
D7's "a validator that raises is logged and skipped" into "a validator that
raises on one rank loses the run". So the local work is run inside a guard and
the *exception is carried through the gather as data*: every rank calls the
collective exactly once per :func:`run_sharded` call, whatever happened locally,
and every rank then raises or returns identically. The gather runs on its own
gloo group with a long timeout, because the ranks arrive at it at genuinely
different times — that is what balance is for, not something to be perfect at —
and NCCL's watchdog aborts the process rather than raising.
"""

from __future__ import annotations

from . import EvalError

__all__ = ["assign", "gather_metrics", "run_sharded", "world"]

#: How long a rank waits at an evaluation gather before giving up. Generous
#: because the thing being waited for is another rank's scoring pass, not a
#: message: an imbalance of a few minutes is normal and a stall is the only
#: reason to ever hit this. It is not NCCL's watchdog — see :func:`_group`.
GATHER_TIMEOUT_S = 3 * 60 * 60


def world() -> tuple:
    """``(rank, world_size)``, and ``(0, 1)`` whenever there is no process group.

    Deliberately tolerant: `validate` mode, the tests and a single-card run all
    go through the same validator code, and none of them has torch.distributed
    initialised.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), max(int(dist.get_world_size()), 1)
    except Exception:                                               # noqa: BLE001
        pass
    return 0, 1


def assign(units, costs, world_size: int) -> list:
    """Longest-processing-time-first: ``[[unit, ...], ...]``, one list per rank.

    Deterministic given the same inputs, which is what makes it safe to run
    independently on every rank instead of computing it once and broadcasting:
    ties break on the unit's original index, so every rank derives the same
    assignment and no rank has to be told what the others are doing.

    The returned lists are in original order, so a rank's log reads in the order
    the validator declared its targets.
    """
    units, costs = list(units), [float(c) for c in costs]
    if len(units) != len(costs):
        raise EvalError(
            f"assign: {len(units)} units and {len(costs)} costs; every unit needs "
            "exactly one cost estimate")
    buckets = [[] for _ in range(world_size)]
    load = [0.0] * world_size
    for i in sorted(range(len(units)), key=lambda j: (-costs[j], j)):
        r = min(range(world_size), key=lambda r: (load[r], r))
        buckets[r].append(i)
        load[r] += costs[i]
    return [[units[i] for i in sorted(b)] for b in buckets]


_GROUP = None
_GROUP_TRIED = False


def _group():
    """A gloo process group for the metric gathers, built once. ``None`` to fall back.

    Gloo rather than the run's NCCL group for one reason: NCCL's watchdog aborts
    the whole process when a collective outlives its timeout, so an evaluation
    imbalance would kill a training run that is otherwise healthy. Gloo raises
    instead, which the D7 guard can catch and report. Falling back to the default
    group is still better than not gathering at all — it only means a stall is
    fatal rather than reported.
    """
    global _GROUP, _GROUP_TRIED

    if _GROUP_TRIED:
        return _GROUP
    _GROUP_TRIED = True
    try:
        import datetime

        import torch.distributed as dist

        _GROUP = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=GATHER_TIMEOUT_S))
    except Exception as exc:                                        # noqa: BLE001
        print(f"[eval] no gloo group for the metric gather ({type(exc).__name__}: "
              f"{exc}); falling back to the default process group")
        _GROUP = None
    return _GROUP


def gather_metrics(local: dict, error) -> tuple:
    """All-gather ``(metrics, error)`` from every rank. ``(merged, [error, ...])``.

    Every rank ends with the same merged dict and the same error list, so what
    happens next — a return or a raise — happens the same way everywhere. Keys
    cannot collide across ranks: a unit is scored on one rank and the keys it
    produces name it.
    """
    import torch.distributed as dist

    size = max(int(dist.get_world_size()), 1)
    slots = [None] * size
    dist.all_gather_object(slots, (dict(local), error), group=_group())

    merged, errors = {}, []
    for rank, slot in enumerate(slots):
        metrics, failure = slot if slot is not None else ({}, "no payload")
        merged.update(metrics)
        if failure:
            errors.append(f"rank {rank}: {failure}")
    return merged, errors


def run_sharded(units, work, cost=None) -> dict:
    """Run ``work(unit)`` over this rank's share of ``units`` and gather the rest.

    ``work`` returns the metric dict for one unit; the dicts are merged. ``cost``
    is ``unit -> float`` and decides only the placement (see :func:`assign`);
    without it the units are assumed equal, which is right when they are slices
    of the same shape and wrong enough to matter when they are not.

    Single-rank runs take the direct path — no process group, no gather, and an
    exception propagates with its own type, exactly as it did before any of this
    existed. That is the path the tests and every current campaign cell run on.
    """
    units = list(units)
    rank, size = world()
    if size == 1:
        out = {}
        for unit in units:
            out.update(work(unit))
        return out

    costs = [1.0] * len(units) if cost is None else [float(cost(u)) for u in units]
    mine = assign(units, costs, size)[rank]

    out, error = {}, None
    try:
        for unit in mine:
            out.update(work(unit))
    except Exception as exc:                                        # noqa: BLE001
        # Carried to the other ranks as data. Raising here would skip the
        # collective below and leave every other rank waiting on a message that
        # is never sent — the one failure mode that turns a skipped validator
        # into a lost run.
        out, error = {}, f"{type(exc).__name__}: {exc}"

    merged, errors = gather_metrics(out, error)
    if errors:
        raise EvalError("; ".join(errors))
    return merged
