"""What a checkpoint contains, and the guarantee that it is whole (DESIGN.md D5.3).

This module is an *extension* of the HF Trainer checkpoint directory, not a
replacement for it. HF still writes the adapter, ``optimizer.pt``,
``scheduler.pt``, ``trainer_state.json`` and the RNG states;
``GraphTrainerV2.save_model`` still writes ``bias_parameters.pt`` beside them.
What is missing from that directory is everything a *resumable, forkable* run
needs and one guarantee:

* ``schedule.json`` — the segments and where in them the run is (D5.2). A step
  number is not enough once segments have been appended.
* ``sampler.json`` — the mixture cursor vector and pass ids (D4.1), without which
  a resume redraws examples the run has already seen.
* ``state.json`` — step, per-task counts, the hashes a resume and a fork have to
  compare, and the bias-norm fingerprint.
* ``COMPLETE`` — written last, atomically. A chunk killed mid-write leaves a
  directory without it, and nothing ever resumes from such a directory. This is
  what makes the sbatch chain (D8.3) requeue-safe.

The bias-norm fingerprint earns its place from the 2026-07-17 reload bug: an
adapter reloaded without its bias tensors trains and evaluates perfectly happily
at silently wrong numbers. ``verify`` recomputes the norm from
``bias_parameters.pt`` and refuses the checkpoint if it disagrees with
``state.json``, so that failure mode costs a startup, not a campaign.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import Optional

from .schedule import Schedule


#: Bumped when the *layout* of these files changes (not when the D1 example
#: schema changes — that version is the trainer's and travels in `state`).
CHECKPOINT_FORMAT_VERSION = 1

COMPLETE_MARKER = "COMPLETE"
PINNED_MARKER = "PINNED"
STATE_FILE = "state.json"
SCHEDULE_FILE = "schedule.json"
SAMPLER_FILE = "sampler.json"
BIAS_FILE = "bias_parameters.pt"

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

#: The keys whose change forces a re-warm on resume (D5.4 step 3).
DISCONTINUITY_KEYS = ("mixture_hash", "tokens_per_step", "lr", "bias_lr", "hardware")

_MISSING = object()


class CheckpointError(RuntimeError):
    """A checkpoint is incomplete, inconsistent, or does not match the model."""


# ── writing ─────────────────────────────────────────────────────────────────


def bias_norm(model, active_params) -> Optional[float]:
    """L2 norm over every parameter whose name contains one of ``active_params``.

    One number over the whole graph-bias channel: it is both the resume
    fingerprint and the ``bias_norm`` validator's readout (D7.3), and the same
    quantity `feedback-verify-nulls-are-real` asks for before any null is
    believed. ``None`` when there is no bias arm to measure.
    """
    if model is None or not active_params:
        return None
    import torch

    total = torch.zeros((), dtype=torch.float64)
    seen = 0
    for name, param in model.named_parameters():
        if any(act in name for act in active_params):
            total += param.detach().to(torch.float64).pow(2).sum().cpu()
            seen += 1
    if seen == 0:
        return None
    return float(total.sqrt())


def _write_json(path: str, payload) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _relative_files(ckpt_dir: str) -> list:
    """Every file in the directory, relative, markers and temporaries excluded."""
    out = []
    for root, _dirs, files in os.walk(ckpt_dir):
        for name in files:
            if name in (COMPLETE_MARKER, PINNED_MARKER) or name.startswith(".tmp."):
                continue
            out.append(os.path.relpath(os.path.join(root, name), ckpt_dir))
    return sorted(out)


def finalize(ckpt_dir: str, *, model, active_params, schedule: Schedule,
             sampler_state: dict, state: dict) -> None:
    """Add the harness's files to an HF checkpoint dir and mark it complete.

    Call this *after* HF has written its own files and after
    ``GraphTrainerV2.save_model`` has written ``bias_parameters.pt``, because
    ``state.json`` records the file list and the bias norm as they stand at this
    moment; anything written afterwards is invisible to ``verify``.

    ``state`` is the trainer's own dict (step, per-task counts, hashes, config,
    registry snapshot, …) and is written through unchanged apart from the four
    keys added here.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    _write_json(os.path.join(ckpt_dir, SCHEDULE_FILE), schedule.to_json())
    _write_json(os.path.join(ckpt_dir, SAMPLER_FILE), dict(sampler_state or {}))

    files = _relative_files(ckpt_dir)
    if STATE_FILE not in files:
        files = sorted(files + [STATE_FILE])

    payload = dict(state or {})
    payload["bias_norm"] = bias_norm(model, active_params)
    payload.setdefault("active_params", list(active_params) if active_params else [])
    # The trainer owns `schema_version` (the D1 example schema) and must set it;
    # it is deliberately NOT defaulted here, because a resume compares it against
    # the running schema, and a checkpoint-format number standing in for it would
    # pass or fail that comparison for the wrong reason.
    if "schema_version" not in payload:
        raise CheckpointError("state must carry `schema_version` (the D1 example schema "
                              "version); a checkpoint without it cannot be checked on resume")
    payload["ckpt_format_version"] = CHECKPOINT_FORMAT_VERSION
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    payload["files"] = files
    _write_json(os.path.join(ckpt_dir, STATE_FILE), payload)

    # COMPLETE last, and via a rename, so that a process killed at any point
    # leaves either no marker or a whole one — never a truncated one that a
    # resume would believe.
    tmp = os.path.join(ckpt_dir, ".tmp." + COMPLETE_MARKER)
    with open(tmp, "w") as fh:
        fh.write(json.dumps({"step": payload.get("step"),
                             "written_at": payload["written_at"]}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.join(ckpt_dir, COMPLETE_MARKER))


# ── enumeration ─────────────────────────────────────────────────────────────


def is_complete(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, COMPLETE_MARKER))


def checkpoint_step(ckpt_dir: str) -> Optional[int]:
    """The step in a ``checkpoint-<step>`` directory name, or None."""
    m = _CKPT_RE.match(os.path.basename(os.path.normpath(ckpt_dir)))
    return int(m.group(1)) if m else None


def list_checkpoints(run_dir: str) -> list:
    """Every ``checkpoint-<step>`` under ``run_dir``, ordered by step ascending.

    By step number, not mtime: a resumed chunk rewrites nothing, but a copied or
    rsynced run directory has mtimes that say nothing about training order.
    """
    if not os.path.isdir(run_dir):
        return []
    found = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        step = checkpoint_step(path)
        if step is not None and os.path.isdir(path):
            found.append((step, path))
    return [path for _step, path in sorted(found)]


def latest(run_dir: str) -> Optional[str]:
    """The newest *complete* checkpoint under ``run_dir``, or None.

    ``resume --from latest`` resolves through here (D5.4). An incomplete
    directory with a higher step is ignored, not an error: it is what a chunk
    killed mid-write leaves behind.
    """
    complete = [p for p in list_checkpoints(run_dir) if is_complete(p)]
    return complete[-1] if complete else None


# ── reading ─────────────────────────────────────────────────────────────────


def read_state(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, STATE_FILE)
    if not os.path.exists(path):
        raise CheckpointError(f"{path} is missing; this is not a harness checkpoint")
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path} is not valid JSON: {exc}") from exc


def verify(ckpt_dir: str, *, model=None, active_params=None) -> dict:
    """Refuse a checkpoint that cannot be resumed from, and return its state.

    Checks, in the order a resume cares about them (D5.4 steps 1–2):
      1. the ``COMPLETE`` marker exists — otherwise the directory is a partial
         write and everything below it is meaningless;
      2. every file ``state.json`` claims is still there;
      3. ``bias_parameters.pt`` is present whenever the run has a bias arm;
      4. with a ``model``, the bias tensors load and their norm matches the
         fingerprint to 1e-6 relative — the 2026-07-17 pairing bug.

    Note that (4) *loads* the tensors into ``model``: on the resume path that is
    the load, not an extra one.
    """
    if not is_complete(ckpt_dir):
        raise CheckpointError(
            f"{ckpt_dir} has no {COMPLETE_MARKER} marker — it was written by a job that "
            "did not finish, and resuming from it would silently use partial state")

    state = read_state(ckpt_dir)

    missing = [f for f in state.get("files", [])
               if not os.path.exists(os.path.join(ckpt_dir, f))]
    if missing:
        raise CheckpointError(
            f"{ckpt_dir} is missing {len(missing)} file(s) named in {STATE_FILE}: {missing[:5]}")

    if active_params is None:
        active_params = state.get("active_params") or None

    bias_path = os.path.join(ckpt_dir, BIAS_FILE)
    if active_params and not os.path.exists(bias_path):
        raise CheckpointError(
            f"{ckpt_dir} has no {BIAS_FILE} but the run trains {list(active_params)} — "
            "the graph-bias weights were never saved, so the adapter alone is not the model")

    if model is not None and active_params:
        from ..models.io import load_bias_parameters

        load_bias_parameters(model, ckpt_dir)
        recomputed = bias_norm(model, active_params)
        expected = state.get("bias_norm")
        if expected is None and recomputed is None:
            # The flat arm. `active_params` names the bias group whatever the
            # arm is, but on a single-node graph there is no bias module to hold
            # those parameters, so `bias_norm` finds none and the checkpoint
            # recorded none. Both absent is the two sides agreeing that there is
            # nothing to pair — the pairing bug this check exists for needs a
            # bias channel to mis-pair. One absent and the other present is
            # still an error, and that is the case that matters.
            return state
        if expected is None or recomputed is None:
            raise CheckpointError(
                f"{ckpt_dir}: no bias-norm fingerprint to check against "
                f"(state.json bias_norm={expected}, recomputed={recomputed})")
        scale = max(abs(expected), 1e-12)
        if abs(recomputed - expected) / scale > 1e-6:
            raise CheckpointError(
                f"{ckpt_dir}: bias norm {recomputed!r} does not match the fingerprint "
                f"{expected!r} written with the checkpoint. The adapter and "
                f"{BIAS_FILE} do not belong to each other; training on this pairing "
                "would report numbers for a model that was never trained.")

    return state


def restore_extras(ckpt_dir: str) -> tuple:
    """``(Schedule, sampler_state, state)`` — the resume path's half of D5.4.

    The model, optimizer and RNG come back through HF and ``GraphTrainerV2``;
    these three are ours.
    """
    state = read_state(ckpt_dir)
    with open(os.path.join(ckpt_dir, SCHEDULE_FILE)) as fh:
        schedule = Schedule.from_json(json.load(fh))
    sampler_path = os.path.join(ckpt_dir, SAMPLER_FILE)
    sampler_state = {}
    if os.path.exists(sampler_path):
        with open(sampler_path) as fh:
            sampler_state = json.load(fh)
    return schedule, sampler_state, state


# ── pinning and rotation ────────────────────────────────────────────────────


def pin(ckpt_dir: str, reason: str = "") -> None:
    """Exempt a checkpoint from rotation — a fork was taken from it (D5.3)."""
    with open(os.path.join(ckpt_dir, PINNED_MARKER), "w") as fh:
        fh.write(json.dumps({"reason": reason,
                             "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def is_pinned(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, PINNED_MARKER))


def rotate(run_dir: str, keep: int) -> dict:
    """Keep the newest ``keep`` complete checkpoints; delete the rest, except:

    * **pinned** ones, which a fork's lineage points at and which must survive
      for its parent to be reproducible;
    * **incomplete** ones, which are never deleted because a concurrent job may
      be in the middle of writing them. They are returned so the caller can say
      so; a stale one costs disk, a deleted live one costs the run.

    Returns ``{"kept", "deleted", "pinned", "incomplete"}`` of paths.
    """
    all_ckpts = list_checkpoints(run_dir)
    incomplete = [p for p in all_ckpts if not is_complete(p)]
    complete = [p for p in all_ckpts if is_complete(p)]

    keep = max(int(keep), 0)
    cut = max(len(complete) - keep, 0)
    recent, older = complete[cut:], complete[:cut]

    kept, deleted, pinned = list(recent), [], []
    for path in older:
        if is_pinned(path):
            pinned.append(path)
            kept.append(path)
            continue
        shutil.rmtree(path)
        deleted.append(path)

    return {"kept": sorted(kept), "deleted": sorted(deleted),
            "pinned": sorted(pinned), "incomplete": sorted(incomplete)}


# ── resume discontinuities ──────────────────────────────────────────────────


def discontinuities(prev_state: dict, new_state: dict) -> list:
    """Which of the D5.4 keys changed between a checkpoint and the run about to
    continue it.

    Pure: it names the causes, it does not decide anything. None of these is an
    error — the trainer appends a re-warm segment and writes a lineage entry
    naming what it found. A key absent on one side and present on the other
    counts as changed; absent on both does not.
    """
    prev_state = prev_state or {}
    new_state = new_state or {}
    changed = []
    for key in DISCONTINUITY_KEYS:
        before = prev_state.get(key, _MISSING)
        after = new_state.get(key, _MISSING)
        if before is _MISSING and after is _MISSING:
            continue
        if before is _MISSING or after is _MISSING or before != after:
            changed.append(key)
    return changed
