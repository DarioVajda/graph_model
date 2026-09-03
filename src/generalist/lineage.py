"""D6 — ``results/lineage.json``: every branching and resume event, append-only.

`PLAN.md` §3.4 asks for these fields to be fixed *before* the first admission
rather than after the first unexplained regression, and this module is that
fixing. One record answers, for any run directory in ``results/``: what it came
from, at which step, under what mode, and what changed on the way.

**One record type, two producers.** A fork (D6) and a discontinuous resume
(D5.4) are the same kind of event — a run continues from a checkpoint under a
config that is not quite the parent's — and they were about to be written in two
shapes, because the trainer's ``lineage_hook`` already emits
``{event, parent, parent_step, causes, rewarm_steps, schedule}`` while D6's table
names ``{child, parent, parent_step, mode, config_diff, created}``.
:class:`LineageEntry` is the union: the D6 fields are always present, ``mode`` is
the fork mode (``None`` for a resume), and ``causes`` / ``rewarm_steps`` /
``schedule`` carry what only a resume knows. Two formats in one file would mean
every reader downstream has to branch on which one it got, and the first reader
to forget would silently skip half the history.

``config_diff`` and ``causes`` look redundant and are not. A fork knows both the
parent's and the child's values, so its diff is ``{key: {"parent": …, "child":
…}}``. A resume is told by ``checkpoint.discontinuities`` only *which* keys moved
— the trainer holds the new values and the checkpoint holds the old ones, but the
hook is handed neither — so a resume entry names the keys in ``causes`` and
leaves ``config_diff`` empty rather than inventing values for it.

**Append-only across concurrent jobs.** Two Slurm chunks can hold the same run
directory open (a resume submitted while the previous chunk is still finishing
its last save is ordinary), so an append must not be able to lose an entry. A
JSON *array* cannot do that: appending to one is read-modify-write, and two
writers racing on it lose whichever finished first. So the file is **JSON Lines**
— one complete record per line — and an append is:

1. serialise the record to a single compact line (no embedded newlines);
2. ``open(O_WRONLY | O_CREAT | O_APPEND)``, so the kernel resolves the write
   offset to the current end of file under the inode lock, and two appenders can
   never target the same bytes;
3. ``flock(LOCK_EX)`` around the write, so a write that the filesystem splits
   into more than one physical write cannot be interleaved with another
   process's;
4. one ``write()`` of the whole line, then ``fsync``.

(2) is what makes an entry impossible to *overwrite*; (3) is what makes a line
impossible to *split*. Where ``flock`` is unavailable — some network filesystems
refuse it — the append still runs, and a record big enough that (4) might not be
one physical write logs a warning rather than pretending. Reading is
correspondingly defensive: a malformed line is counted and reported, never
raised, because one bad line is not a reason to lose the other three hundred.

The file keeps D6's name, ``lineage.json``, even though its content is JSON Lines
— that name is what `PLAN.md` §3.4 and D6 both refer to, and :func:`read` gives a
clear error rather than a ``JSONDecodeError`` if something ever writes an array
into it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:                                                     # pragma: no cover
    import fcntl
except ImportError:                                      # pragma: no cover
    fcntl = None

#: D6's name for the file. Its content is JSON Lines; see the module docstring.
LINEAGE_FILENAME = "lineage.json"

#: Bumped when the *record* changes shape. Written on every line so a reader can
#: tell a v1 entry from a later one without guessing from which keys are present.
LINEAGE_FORMAT_VERSION = 1

#: What produced the entry. A fork branches the run; a resume continues it.
EVENTS = ("fork", "resume")

#: D6's fork modes. They live here rather than in ``fork.py`` because the record
#: is the lower layer — ``fork.py`` imports this module, not the other way round.
FORK_MODES = ("anneal", "admit", "adapt")

#: Above this, a single ``write()`` is not guaranteed to be one physical write on
#: every filesystem, so an append without ``flock`` could in principle interleave.
#: POSIX guarantees atomicity up to ``PIPE_BUF``; 4096 is its floor.
ATOMIC_APPEND_LIMIT = 4096


class LineageError(RuntimeError):
    """A lineage record is malformed, or the file is not the format we write."""


# ─────────────────────────────────────────────────────────────────────────────
# The record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LineageEntry:
    """One branching or resume event.

    The first six fields are D6's; the rest are what a resume adds and a fork
    leaves empty. ``child`` and ``parent`` are paths — the child's *run
    directory* and the parent's *checkpoint directory* — because those are what
    identifies a run on disk and what a report has to be able to open.

    ``created`` is UTC. Entries from two Slurm jobs on two nodes end up in one
    file and local time would not order them.
    """

    child: str
    parent: str
    parent_step: int
    event: str = "fork"
    mode: Optional[str] = None
    config_diff: dict = field(default_factory=dict)
    #: Resume only: the ``checkpoint.DISCONTINUITY_KEYS`` that moved (D5.4 step 3).
    causes: tuple = ()
    #: Length of the re-warm the event appended, 0 for none, ``None`` if unknown.
    rewarm_steps: Optional[int] = None
    #: ``Schedule.to_json()`` as it stands after the event.
    schedule: Optional[dict] = None
    note: str = ""
    created: str = ""
    version: int = LINEAGE_FORMAT_VERSION

    def __post_init__(self):
        if self.event not in EVENTS:
            raise LineageError(
                f"event must be one of {EVENTS}, got {self.event!r}")
        if self.event == "fork" and self.mode not in FORK_MODES:
            raise LineageError(
                f"a fork entry needs a mode from {FORK_MODES}, got {self.mode!r}")
        if self.event == "resume" and self.mode is not None:
            raise LineageError(
                f"a resume entry has no fork mode; got mode={self.mode!r}")
        if not self.child:
            raise LineageError("child: a lineage entry needs the child run directory")
        if not self.parent:
            raise LineageError("parent: a lineage entry needs the parent checkpoint")
        object.__setattr__(self, "parent_step", int(self.parent_step))
        object.__setattr__(self, "causes", tuple(self.causes or ()))
        object.__setattr__(self, "config_diff", dict(self.config_diff or {}))
        if not self.created:
            object.__setattr__(self, "created", utc_now())

    def to_json(self) -> dict:
        """A JSON-ready dict in a fixed key order (the on-disk record)."""
        return {
            "version": int(self.version),
            "event": self.event,
            "child": self.child,
            "parent": self.parent,
            "parent_step": int(self.parent_step),
            "mode": self.mode,
            "config_diff": dict(self.config_diff),
            "causes": list(self.causes),
            "rewarm_steps": self.rewarm_steps,
            "schedule": self.schedule,
            "note": self.note,
            "created": self.created,
        }

    @classmethod
    def from_json(cls, data) -> "LineageEntry":
        if isinstance(data, (str, bytes)):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise LineageError(f"a lineage record must be an object, got {data!r}")
        version = int(data.get("version", LINEAGE_FORMAT_VERSION))
        if version > LINEAGE_FORMAT_VERSION:
            raise LineageError(
                f"lineage record is format version {version}, this code reads "
                f"{LINEAGE_FORMAT_VERSION}")
        known = {f: data.get(f) for f in
                 ("child", "parent", "parent_step", "event", "mode", "config_diff",
                  "causes", "rewarm_steps", "schedule", "note", "created")}
        known["version"] = version
        known["event"] = known["event"] or "fork"
        return cls(**{k: v for k, v in known.items() if v is not None or k in
                      ("mode", "rewarm_steps", "schedule")})


def utc_now() -> str:
    """ISO 8601 in UTC, to the second. One clock for entries from many nodes."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ─────────────────────────────────────────────────────────────────────────────
# The file
# ─────────────────────────────────────────────────────────────────────────────

def _lock(fd) -> bool:
    """``flock(LOCK_EX)``; False when the filesystem will not take one.

    Not fatal: the ``O_APPEND`` open is what stops one entry overwriting
    another, and the lock only removes the remaining (small) chance that a long
    line is split across two physical writes. Failing the whole append because a
    network filesystem has no ``flock`` would trade a rare interleaving for a
    certain loss.
    """
    if fcntl is None:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return True
    except OSError as exc:                                # pragma: no cover
        logger.debug("lineage: flock unavailable (%s); relying on O_APPEND", exc)
        return False


def _unlock(fd, locked: bool) -> None:
    if locked and fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:                                   # pragma: no cover
            pass


def append_line(path: str, line: str) -> None:
    """Append one line to ``path``, atomically enough for two Slurm jobs.

    See the module docstring for what "atomically enough" means and why the two
    mechanisms (``O_APPEND`` and ``flock``) are both there.
    """
    if "\n" in line.strip("\n"):
        raise LineageError("a lineage record must serialise to a single line")
    data = (line.rstrip("\n") + "\n").encode("utf-8")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        locked = _lock(fd)
        if not locked and len(data) > ATOMIC_APPEND_LIMIT:  # pragma: no cover
            logger.warning(
                "lineage: appending %d bytes to %s without a lock; only the first "
                "%d are guaranteed to land as one write, so a concurrent append "
                "could interleave with this one", len(data), path,
                ATOMIC_APPEND_LIMIT)
        try:
            written = 0
            while written < len(data):
                written += os.write(fd, data[written:])
            os.fsync(fd)
        finally:
            _unlock(fd, locked)
    finally:
        os.close(fd)


class Lineage:
    """``results/lineage.json`` — the append-only record of every branch.

    Construct it with the file path or with the ``results/`` directory holding
    it; both are common at the call sites (a fork knows its results root, the
    trainer is handed a path).
    """

    def __init__(self, path: str):
        path = str(path)
        if os.path.isdir(path) or not path.endswith(".json"):
            path = os.path.join(path, LINEAGE_FILENAME)
        self.path = os.path.abspath(path)

    def __repr__(self) -> str:
        return f"Lineage({self.path!r})"

    # ── writing ─────────────────────────────────────────────────────────────

    def append(self, entry) -> LineageEntry:
        """Append one entry and return it (with ``created`` filled in)."""
        if isinstance(entry, dict):
            entry = LineageEntry.from_json(entry)
        if not isinstance(entry, LineageEntry):
            raise LineageError(f"append: expected a LineageEntry, got {entry!r}")
        append_line(self.path, json.dumps(entry.to_json(), sort_keys=True,
                                          separators=(",", ":"), default=str))
        return entry

    def record_fork(self, *, child: str, parent: str, parent_step: int, mode: str,
                    config_diff: dict = None, schedule: dict = None,
                    rewarm_steps: Optional[int] = None,
                    note: str = "") -> LineageEntry:
        """D6's entry: this run directory branched off that checkpoint."""
        return self.append(LineageEntry(
            child=child, parent=parent, parent_step=parent_step, event="fork",
            mode=mode, config_diff=config_diff or {}, schedule=schedule,
            rewarm_steps=rewarm_steps, note=note))

    def record_trainer_event(self, entry: dict, *, child: str,
                             note: str = "") -> LineageEntry:
        """Normalise ``GeneralistTrainer``'s hook payload into a record.

        The trainer emits ``{event, parent, parent_step, causes, rewarm_steps,
        schedule}`` and does not know its own run directory in those terms, so
        ``child`` is supplied by whoever installed the hook (see :meth:`hook`).
        Anything the trainer did not send keeps the record's default rather than
        being guessed at — a resume genuinely does not know the parent's config
        *values*, only which keys moved.
        """
        if not isinstance(entry, dict):
            raise LineageError(f"lineage hook: expected a dict, got {entry!r}")
        event = entry.get("event", "resume")
        return self.append(LineageEntry(
            child=child,
            parent=entry.get("parent", ""),
            parent_step=entry.get("parent_step", 0),
            event=event,
            mode=entry.get("mode") if event == "fork" else None,
            config_diff=entry.get("config_diff") or {},
            causes=entry.get("causes") or (),
            rewarm_steps=entry.get("rewarm_steps"),
            schedule=entry.get("schedule"),
            note=note or entry.get("note", ""),
        ))

    def hook(self, child: str, *, note: str = "") -> Callable:
        """The callable ``GeneralistTrainer(lineage_hook=…)`` expects.

        ``child`` is the run directory the trainer is writing into; the trainer
        does not carry it in the payload, and a record whose ``child`` is unknown
        cannot be joined to anything.
        """
        if not child:
            raise LineageError(
                "hook: needs the child run directory — an entry that does not say "
                "which run it belongs to cannot be read back into a lineage")

        def lineage_hook(entry: dict) -> LineageEntry:
            return self.record_trainer_event(entry, child=child, note=note)

        return lineage_hook

    # ── reading ─────────────────────────────────────────────────────────────

    def read(self) -> list:
        """Every entry, in file order. Malformed lines are warned about, not raised."""
        entries, malformed = self.read_with_errors()
        if malformed:
            logger.warning(
                "lineage: %d unreadable line(s) in %s (line numbers %s); the file "
                "is append-only, so these are the remains of a write that did not "
                "complete and the entries around them are intact",
                len(malformed), self.path, [n for n, _ in malformed][:5])
        return entries

    def read_with_errors(self) -> tuple:
        """``(entries, [(line number, raw line), …])``.

        The second half is what makes "append-only" checkable: a caller that
        cares (a report, a test) can assert it is empty instead of trusting that
        every append landed whole.
        """
        if not os.path.exists(self.path):
            return [], []
        entries, malformed = [], []
        with open(self.path) as fh:
            for number, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                if number == 1 and line.startswith("["):
                    raise LineageError(
                        f"{self.path} starts with '[' — it holds a JSON array, but "
                        "this file is JSON Lines (one record per line) so that two "
                        "jobs can append to it without losing each other's entries")
                try:
                    entries.append(LineageEntry.from_json(json.loads(line)))
                except (json.JSONDecodeError, LineageError, TypeError):
                    malformed.append((number, raw.rstrip("\n")))
        return entries, malformed

    def entries_for(self, child: str) -> list:
        """Everything recorded about one run directory."""
        target = os.path.abspath(child)
        return [e for e in self.read() if os.path.abspath(e.child) == target]

    def children_of(self, parent: str) -> list:
        """Every fork taken from one checkpoint directory."""
        target = os.path.abspath(parent)
        return [e for e in self.read()
                if e.event == "fork" and os.path.abspath(e.parent) == target]

    def ancestry(self, child: str) -> list:
        """``child`` back to the root, oldest first.

        Each hop is the fork entry whose ``child`` is the current run; the walk
        stops at a run with no fork entry (a run trained from base) and refuses
        to loop, since a hand-edited file could in principle contain a cycle.
        """
        by_child: dict = {}
        for entry in self.read():
            if entry.event == "fork":
                by_child.setdefault(os.path.abspath(entry.child), entry)
        chain, seen, cursor = [], set(), os.path.abspath(child)
        while cursor in by_child and cursor not in seen:
            seen.add(cursor)
            entry = by_child[cursor]
            chain.append(entry)
            cursor = os.path.abspath(os.path.dirname(entry.parent))
        return list(reversed(chain))
