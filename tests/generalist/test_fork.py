"""T7 — branching and lineage (DESIGN.md D6).

Five things have to hold for a fork to be worth anything, and each of them fails
silently rather than loudly if it does not:

1. **An anneal actually reaches ``lr_min``.** The reportable model is the one the
   decay produced, so the last optimizer step has to run at the LR the schedule
   advertises. Read off ``optimizer.param_groups`` during training, never off the
   schedule object — asking the schedule whether it agrees with itself proves
   nothing about what the optimizer applied.
2. **The parent is pinned and untouched.** A fork that edits its parent's
   checkpoint makes the trunk's series mean two different things depending on
   when it was read, and pinning is what keeps the branch point alive through the
   parent's own rotation.
3. **The lineage record is complete and append-only.** Two Slurm jobs share a run
   directory routinely, and an entry lost to a racing append is a hole in the one
   file `PLAN.md` §3.4 says has to be trustworthy before the first admission.
4. **The two ``adapt`` legs differ only in their starting weights.** The whole
   adaptation-efficiency number is the difference between them; any other
   difference makes it unattributable.
5. **An incomplete checkpoint is refused.** It is a partial write, and a fork
   from one reports numbers for a model that was never finished.

Everything runs on the tiny CPU model from ``tiny_run.py``.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil

import pytest
from transformers import TrainerCallback

from src.generalist import checkpoint as ckpt_mod
from src.generalist.fork import (
    ADMISSION_PARTS,
    ALL_VALIDATORS,
    ForkError,
    check_admission,
    check_criterion,
    fork,
    plan_fork,
    steps_to_target,
)
from src.generalist.lineage import Lineage, LineageEntry, LineageError
from src.generalist.registry import TaskSpec
from tests.generalist.tiny_run import (
    ACTIVE_PARAMS,
    MEAN_TOKENS,
    TASKS,
    TRAIN_SIZE,
    LearningRateProbe,
    build_registry,
    build_trainer,
)

PARENT_STEPS = 4
PARENT_WARMUP = 2
LR = 1e-2
BIAS_LR = 5e-2

HELD_OUT_TASK = "t/held"
CANDIDATE_TASK = "t/candidate"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _extra_registry():
    """The tiny registry plus a held-out task and an admission candidate.

    ``tiny_run.build_registry`` has no held-out task (nothing it is used for
    needed one) and this file may not edit it, so the two extra specs are added
    here. ``held_out=True`` is the registry's own enforcement source, so
    ``is_held_out`` answers without reaching into the molecules package.
    """
    registry = build_registry()
    for name, held in ((HELD_OUT_TASK, True), (CANDIDATE_TASK, False)):
        registry.register(TaskSpec(
            name=name, domain="tiny", adapter="tiny", kind="corpus",
            answer_kind="token", held_out=held, weight=1.0, passes=8,
            metric="exact_match", build_version="tiny-1",
            mean_tokens=MEAN_TOKENS, train_size=TRAIN_SIZE))
    return registry


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    """A finished four-step trunk run with one complete checkpoint.

    The warmup is two steps so the parent is *past* it at step 4 — a schedule can
    only be extended past its trailing open segment, so a fork taken mid-warmup
    is a different (and separately tested) situation.
    """
    root = tmp_path_factory.mktemp("fork_parent")
    run_dir = str(root / "trunk")
    registry = _extra_registry()
    trainer, _model, _sampler, _schedule = build_trainer(
        run_dir, max_steps=PARENT_STEPS, warmup_steps=PARENT_WARMUP,
        save_steps=PARENT_STEPS, lr=LR, bias_lr=BIAS_LR, seed=0,
        registry=registry)
    trainer.train()
    ckpt = ckpt_mod.latest(run_dir)
    assert ckpt is not None, "the parent run wrote no complete checkpoint"
    return {
        "root": root,
        "run_dir": run_dir,
        "ckpt": ckpt,
        "state": ckpt_mod.read_state(ckpt),
        "registry": registry,
        "mixture": [{"name": name, "weight": 1.0} for name in TASKS],
    }


def _snapshot(directory: str) -> dict:
    """``{relative path: sha256}`` for every file, PINNED excluded.

    ``PINNED`` is excluded because a fork is *supposed* to write it; everything
    else in a parent checkpoint must come back byte-identical.
    """
    out = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name == ckpt_mod.PINNED_MARKER:
                continue
            path = os.path.join(root, name)
            with open(path, "rb") as fh:
                out[os.path.relpath(path, directory)] = hashlib.sha256(
                    fh.read()).hexdigest()
    return out


class StartWeights(TrainerCallback):
    """Records the graph-bias norm at the moment training starts.

    ``on_train_begin`` fires after ``fork`` has put the starting weights in and
    before the first step, which is the only window in which "what did this leg
    start from" is observable from outside.
    """

    def __init__(self):
        self.bias_norm = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.bias_norm = ckpt_mod.bias_norm(model, ACTIVE_PARAMS)


def make_factory(registry, record: dict):
    """A ``trainer_factory`` over ``tiny_run.build_trainer``.

    Everything the leg fixes — the output directory, the schedule, the mixture,
    the seed and the step budget — is taken from the leg rather than chosen here,
    which is exactly what D8's real factory will do.
    """
    def factory(leg, _plan):
        probe = LearningRateProbe()
        start = StartWeights()
        trainer, model, sampler, schedule = build_trainer(
            leg.output_dir, max_steps=leg.max_steps, lr=LR, bias_lr=BIAS_LR,
            seed=leg.seed, schedule=leg.schedule, mixture=leg.mixture,
            registry=registry, callbacks=[probe, start])
        record[leg.name] = {"trainer": trainer, "model": model, "probe": probe,
                            "start": start, "sampler": sampler,
                            "schedule": schedule}
        return trainer

    return factory


# ─────────────────────────────────────────────────────────────────────────────
# lineage.py — the record
# ─────────────────────────────────────────────────────────────────────────────

class TestLineageRecord:

    def test_every_d6_field_is_present_on_a_fork_entry(self, tmp_path):
        lineage = Lineage(str(tmp_path))
        entry = lineage.record_fork(
            child="/runs/child", parent="/runs/trunk/checkpoint-4", parent_step=4,
            mode="anneal", config_diff={"decay_steps": {"parent": None, "child": 3}},
            schedule={"version": 1, "segments": []}, note="milestone")

        for field in ("child", "parent", "parent_step", "mode", "config_diff",
                      "created"):
            assert field in entry.to_json(), f"D6 field {field} missing"
        assert entry.created.endswith("Z"), "created must be UTC"
        (read_back,) = lineage.read()
        assert read_back.to_json() == entry.to_json()

    def test_a_resume_becomes_the_same_record_type(self, tmp_path):
        """The trainer's hook payload and a fork entry are one shape on disk.

        Two shapes in one append-only file would make every reader branch on
        which one it got, and the first reader to forget silently skips half the
        history.
        """
        lineage = Lineage(str(tmp_path))
        hook = lineage.hook(child="/runs/child")
        hook({"event": "resume", "parent": "/runs/child/checkpoint-8",
              "parent_step": 8, "causes": ["mixture_hash", "hardware"],
              "rewarm_steps": 5, "schedule": {"version": 1, "segments": []}})
        lineage.record_fork(child="/runs/child", parent="/runs/trunk/checkpoint-4",
                            parent_step=4, mode="anneal")

        entries = lineage.read()
        assert [e.event for e in entries] == ["resume", "fork"]
        assert set(entries[0].to_json()) == set(entries[1].to_json())
        resume, forked = entries
        assert resume.mode is None and resume.causes == ("mixture_hash", "hardware")
        assert resume.rewarm_steps == 5
        # A resume is told which keys moved, not what they moved between, so it
        # records the names and does not invent values for config_diff.
        assert resume.config_diff == {}
        assert forked.mode == "anneal"

    def test_a_fork_entry_needs_a_mode_and_a_resume_refuses_one(self):
        with pytest.raises(LineageError):
            LineageEntry(child="c", parent="p", parent_step=1, event="fork")
        with pytest.raises(LineageError):
            LineageEntry(child="c", parent="p", parent_step=1, event="fork",
                         mode="polish")
        with pytest.raises(LineageError):
            LineageEntry(child="c", parent="p", parent_step=1, event="resume",
                         mode="anneal")

    def test_a_json_array_is_refused_rather_than_half_read(self, tmp_path):
        path = tmp_path / "lineage.json"
        path.write_text(json.dumps([{"child": "c", "parent": "p",
                                     "parent_step": 1, "mode": "anneal"}]))
        with pytest.raises(LineageError, match="JSON Lines"):
            Lineage(str(path)).read()

    def test_ancestry_walks_back_to_the_root(self, tmp_path):
        lineage = Lineage(str(tmp_path))
        lineage.record_fork(child="/r/a", parent="/r/trunk/checkpoint-4",
                            parent_step=4, mode="anneal")
        lineage.record_fork(child="/r/b", parent="/r/a/checkpoint-8",
                            parent_step=8, mode="adapt")
        assert [e.child for e in lineage.ancestry("/r/b")] == ["/r/a", "/r/b"]
        assert [e.child for e in lineage.children_of("/r/trunk/checkpoint-4")] \
            == ["/r/a"]


def _append_worker(path: str, tag: str, count: int) -> None:
    """One process's share of the concurrent-append test. Module level so the
    child can reach it after a fork."""
    lineage = Lineage(path)
    for i in range(count):
        lineage.record_fork(child=f"/runs/{tag}-{i:03d}",
                            parent="/runs/trunk/checkpoint-4", parent_step=4,
                            mode="anneal",
                            note="a note long enough that the line is not trivial "
                                 "and a split write would be visible " * 3)


class TestLineageIsAppendOnly:

    def test_two_processes_appending_lose_nothing(self, tmp_path):
        """The reason the file is JSON Lines under ``O_APPEND`` and a lock.

        A JSON array cannot survive this: appending to one is read-modify-write,
        and whichever process finished first would have its entries overwritten.
        """
        path = str(tmp_path / "lineage.json")
        per_process = 25
        ctx = multiprocessing.get_context("fork")
        procs = [ctx.Process(target=_append_worker, args=(path, tag, per_process))
                 for tag in ("alpha", "beta")]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(120)
            assert proc.exitcode == 0, "an appending process died"

        entries, malformed = Lineage(path).read_with_errors()
        assert malformed == [], f"interleaved or truncated lines: {malformed[:3]}"
        assert len(entries) == 2 * per_process
        expected = {f"/runs/{tag}-{i:03d}" for tag in ("alpha", "beta")
                    for i in range(per_process)}
        assert {e.child for e in entries} == expected


# ─────────────────────────────────────────────────────────────────────────────
# Refusals — plan only, no training
# ─────────────────────────────────────────────────────────────────────────────

class TestForkRefusals:

    def test_an_incomplete_checkpoint_is_refused(self, parent, tmp_path):
        broken = str(tmp_path / "checkpoint-4")
        shutil.copytree(parent["ckpt"], broken)
        os.remove(os.path.join(broken, ckpt_mod.COMPLETE_MARKER))
        with pytest.raises(ForkError, match="COMPLETE"):
            plan_fork(broken, "anneal", {"run_dir": str(tmp_path / "child")},
                      registry=parent["registry"],
                      parent_mixture=parent["mixture"])

    def test_an_unknown_mode_is_refused(self, parent, tmp_path):
        with pytest.raises(ForkError, match="mode"):
            plan_fork(parent["ckpt"], "polish", {"run_dir": str(tmp_path / "c")},
                      registry=parent["registry"],
                      parent_mixture=parent["mixture"])

    def test_a_fork_into_the_parents_own_directory_is_refused(self, parent):
        with pytest.raises(ForkError, match="parent"):
            plan_fork(parent["ckpt"], "anneal",
                      {"run_dir": parent["run_dir"]},
                      registry=parent["registry"],
                      parent_mixture=parent["mixture"])

    def test_anneal_reads_the_parents_mixture_off_the_checkpoint(self, parent,
                                                                 tmp_path):
        """`state.json` records the entry list beside the hash, so an anneal is
        self-contained from a checkpoint path — a hash alone would say whether two
        mixtures match and never what either of them was."""
        plan = plan_fork(parent["ckpt"], "anneal", {"run_dir": str(tmp_path / "c")},
                         registry=parent["registry"])
        named = [e["name"] for e in plan.legs[0].mixture_config]
        assert named == [e["name"] for e in parent["mixture"]]

    def test_anneal_without_a_recorded_mixture_is_refused(self, parent, tmp_path):
        """An older checkpoint, or one written by something that did not record
        the entries, is refused rather than annealed on a guessed mixture."""
        stripped = shutil.copytree(parent["ckpt"], str(tmp_path / "bare"))
        state_path = os.path.join(stripped, "state.json")
        with open(state_path) as f:
            state = json.load(f)
        state.pop("mixture_entries", None)
        with open(state_path, "w") as f:
            json.dump(state, f)
        with pytest.raises(ForkError, match="mixture"):
            plan_fork(stripped, "anneal", {"run_dir": str(tmp_path / "c")},
                      registry=parent["registry"])


def _spent(parent, tmp_path, consumed: dict) -> str:
    """A copy of the parent checkpoint that has already drawn ``consumed``."""
    spent = shutil.copytree(parent["ckpt"], str(tmp_path / "spent"))
    state_path = os.path.join(spent, "state.json")
    with open(state_path) as f:
        state = json.load(f)
    state["examples_per_task"] = consumed
    with open(state_path, "w") as f:
        json.dump(state, f)
    return spent


class TestAForkInheritsTheParentsSpentBudget:
    """A corpus's ``passes x train_size`` is spent by the trunk *and* every leg.

    The trunk's budget rule sizes the run at the largest budget no corpus
    overruns, so a trunk that ran to its own step count leaves a leg nothing to
    draw. That has to be a plan-time refusal: at training time it surfaces as a
    short final optimizer step and an accumulation error, which names the batch
    shape and not the cause.
    """

    #: 8 of the 8 passes over 64 rows, less a handful.
    NEARLY_SPENT = {"t/alpha": 510, "t/beta": 510, "t/gamma": 510}

    def test_an_anneal_off_a_spent_trunk_is_refused_by_name(self, parent, tmp_path):
        spent = _spent(parent, tmp_path, self.NEARLY_SPENT)
        with pytest.raises(ForkError, match="passes") as excinfo:
            plan_fork(spent, "anneal", {"run_dir": str(tmp_path / "c")},
                      registry=parent["registry"])
        message = str(excinfo.value)
        assert "t/alpha" in message, "the refusal must name the short task"
        assert "2 example(s) left" in message, "and what is actually left"

    def test_raising_passes_gives_the_anneal_room_to_run(self, parent, tmp_path):
        spent = _spent(parent, tmp_path, self.NEARLY_SPENT)
        plan = plan_fork(spent, "anneal",
                         {"run_dir": str(tmp_path / "c"),
                          "passes": {name: 16 for name in TASKS}},
                         registry=parent["registry"])
        (leg,) = plan.legs
        assert {e["passes"] for e in leg.mixture_config} == {16}
        assert {e.passes for e in leg.mixture.entries} == {16}, \
            "the resolved mixture, not just the config, must carry the override"

    def test_passes_may_only_name_a_task_the_fork_already_trains(self, parent,
                                                                 tmp_path):
        with pytest.raises(ForkError, match="does not train"):
            plan_fork(parent["ckpt"], "anneal",
                      {"run_dir": str(tmp_path / "c"),
                       "passes": {"t/nonesuch": 16}},
                      registry=parent["registry"])

    def test_a_trunk_with_budget_left_needs_no_override(self, parent, tmp_path):
        """The default path stays default — the check is a refusal, not a
        requirement to restate the parent's passes in every fork config."""
        plan = plan_fork(parent["ckpt"], "anneal",
                         {"run_dir": str(tmp_path / "c")},
                         registry=parent["registry"])
        assert plan.legs[0].mixture.entries, "a plan with room should still plan"

    def test_admit_inherits_the_same_refusal(self, parent, tmp_path):
        spent = _spent(parent, tmp_path, self.NEARLY_SPENT)
        with pytest.raises(ForkError, match="passes"):
            plan_fork(spent, "admit",
                      {"run_dir": str(tmp_path / "c"), "budget_steps": 4,
                       "rewarm_steps": 2,
                       "candidate": {"name": CANDIDATE_TASK, "weight": 1.0},
                       "criterion": _criterion()},
                      registry=parent["registry"])

    def test_adapt_on_an_in_mixture_task_is_refused(self, parent, tmp_path):
        config = {"task": TASKS[0], "budget_steps": 4, "eval_steps": 2,
                  "target": {"metric": "held_out/x/exact_match", "value": 0.5},
                  "run_dir": str(tmp_path / "c")}
        with pytest.raises(ForkError, match="not held out"):
            plan_fork(parent["ckpt"], "adapt", config,
                      registry=parent["registry"])
        config["allow_in_mixture_task"] = True
        plan = plan_fork(parent["ckpt"], "adapt", config,
                         registry=parent["registry"])
        assert len(plan.legs) == 2

    def test_a_target_on_the_test_split_is_refused(self, parent, tmp_path):
        with pytest.raises(ForkError, match="test split"):
            plan_fork(parent["ckpt"], "adapt",
                      {"task": HELD_OUT_TASK, "budget_steps": 4, "eval_steps": 2,
                       "target": {"metric": "held_out/t/held/test_roc_auc",
                                  "value": 0.5},
                       "run_dir": str(tmp_path / "c")},
                      registry=parent["registry"])


# ─────────────────────────────────────────────────────────────────────────────
# anneal
# ─────────────────────────────────────────────────────────────────────────────

class TestAnneal:

    @pytest.fixture(scope="class")
    def annealed(self, parent, tmp_path_factory):
        root = tmp_path_factory.mktemp("anneal")
        record: dict = {}
        result = fork(
            parent["ckpt"], "anneal",
            {"run_dir": str(root / "fork"), "decay_steps": 3,
             "min_factor": 0.25, "decay_shape": "linear"},
            registry=parent["registry"], parent_mixture=parent["mixture"],
            results_dir=str(root), trainer_factory=make_factory(
                parent["registry"], record))
        return {"result": result, "record": record, "root": root}

    def test_the_last_step_runs_at_lr_min(self, annealed):
        """Read off the optimizer, not the schedule.

        The decay's endpoint is a step the model must actually take: a segment of
        ``decay_steps`` interpolates over ``[start, start + decay_steps]``, so the
        fork runs ``decay_steps + 1`` steps and the last one is at exactly
        ``min_factor``. Stopping one earlier would make the reportable model the
        one from just before the anneal finished.
        """
        probe = annealed["record"]["anneal"]["probe"]
        last = PARENT_STEPS + 3          # decay spans steps 4..7
        lora_lr, bias_lr = probe.lrs_at(last)
        assert lora_lr == pytest.approx(LR * 0.25, rel=1e-9)
        assert bias_lr == pytest.approx(BIAS_LR * 0.25, rel=1e-9)

    def test_the_lr_falls_monotonically_through_the_decay(self, annealed):
        probe = annealed["record"]["anneal"]["probe"]
        seen = [probe.lrs_at(step)[0]
                for step in range(PARENT_STEPS, PARENT_STEPS + 4)]
        assert seen[0] == pytest.approx(LR, rel=1e-9)
        assert all(a > b for a, b in zip(seen, seen[1:])), seen

    def test_the_bias_ratio_is_held_through_the_decay(self, annealed):
        """One curve serves both groups (D5.2): ``bias_lr / lr`` never moves."""
        probe = annealed["record"]["anneal"]["probe"]
        for step in range(PARENT_STEPS, PARENT_STEPS + 4):
            lora_lr, bias_lr = probe.lrs_at(step)
            assert bias_lr / lora_lr == pytest.approx(BIAS_LR / LR, rel=1e-9)

    def test_the_child_is_a_new_run_directory_with_its_own_record(self, annealed):
        plan = annealed["result"].plan
        assert os.path.isfile(os.path.join(plan.run_dir, "fork.json"))
        assert os.path.isfile(os.path.join(plan.run_dir, "result.json"))
        with open(os.path.join(plan.run_dir, "fork.json")) as fh:
            record = json.load(fh)
        assert record["mode"] == "anneal"
        assert record["parent"] == plan.parent_ckpt
        assert record["validators"] == [ALL_VALIDATORS]
        assert record["config_diff"]["decay_steps"]["child"] == 3
        assert plan.leg("anneal").start_checkpoint.startswith(plan.run_dir)

    def test_the_lineage_has_the_fork_and_the_childs_own_resume(self, annealed):
        entries = Lineage(str(annealed["root"])).read()
        leg_dir = annealed["result"].plan.leg("anneal").output_dir
        mine = [e for e in entries if os.path.abspath(e.child)
                == os.path.abspath(leg_dir)]
        assert [e.event for e in mine] == ["fork", "resume"], \
            "the fork entry, then the child's own restore of the copied checkpoint"
        assert mine[0].mode == "anneal"
        assert mine[0].parent_step == PARENT_STEPS
        assert mine[1].causes == (), "nothing moved, so no discontinuity"

    def test_the_schedule_is_closed_and_the_parents_is_not(self, annealed, parent):
        child = annealed["result"].plan.leg("anneal").schedule
        assert not child.is_open
        assert child.segments[-1].kind == "decay"
        parent_schedule, _s, _st = ckpt_mod.restore_extras(parent["ckpt"])
        assert parent_schedule.is_open, "the parent keeps its open-ended stable"


# ─────────────────────────────────────────────────────────────────────────────
# The parent survives
# ─────────────────────────────────────────────────────────────────────────────

class TestTheParentIsUntouched:

    def test_pinned_complete_and_byte_identical(self, parent, tmp_path):
        before = _snapshot(parent["ckpt"])
        fork(parent["ckpt"], "anneal",
             {"run_dir": str(tmp_path / "child"), "decay_steps": 2},
             registry=parent["registry"], parent_mixture=parent["mixture"],
             results_dir=str(tmp_path))

        assert ckpt_mod.is_pinned(parent["ckpt"]), \
            "the branch point must survive the parent's own rotation"
        assert ckpt_mod.is_complete(parent["ckpt"])
        assert _snapshot(parent["ckpt"]) == before, \
            "a fork wrote into its parent's checkpoint"
        # And it still verifies: `files` intact, bias fingerprint intact.
        ckpt_mod.verify(parent["ckpt"])

    def test_a_pinned_checkpoint_survives_rotation(self, parent, tmp_path):
        run_dir = str(tmp_path / "rotating")
        for step in (2, 4, 6):
            shutil.copytree(parent["ckpt"],
                            os.path.join(run_dir, f"checkpoint-{step}"))
        branch_point = os.path.join(run_dir, "checkpoint-2")
        fork(branch_point, "anneal",
             {"run_dir": str(tmp_path / "child2"), "decay_steps": 2},
             registry=parent["registry"], parent_mixture=parent["mixture"],
             results_dir=str(tmp_path))
        report = ckpt_mod.rotate(run_dir, keep=1)
        assert branch_point in report["pinned"]
        assert os.path.isdir(branch_point)

    def test_the_childs_copy_carries_the_childs_schedule(self, parent, tmp_path):
        """The appended segment has to live in the copy the child resumes.

        ``prepare_resume`` restores the schedule *from the checkpoint it
        resumes*, so a decay written anywhere else is overwritten on the first
        step — and the fork would train at the stable LR while its record said
        it annealed.
        """
        result = fork(parent["ckpt"], "anneal",
                      {"run_dir": str(tmp_path / "child3"), "decay_steps": 2,
                       "min_factor": 0.5},
                      registry=parent["registry"],
                      parent_mixture=parent["mixture"], results_dir=str(tmp_path))
        copy = result.plan.leg("anneal").start_checkpoint
        copied_schedule, _s, _st = ckpt_mod.restore_extras(copy)
        assert copied_schedule.segments[-1].kind == "decay"
        assert copied_schedule.factor(PARENT_STEPS + 2) == pytest.approx(0.5)
        # Rewriting schedule.json must not break the file list state.json carries.
        ckpt_mod.verify(copy)


# ─────────────────────────────────────────────────────────────────────────────
# adapt
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_validate(request):
    """A stand-in for the ``evaluate/`` hook: a metric that rises with the step.

    The parent leg climbs four times faster than base, so it crosses the target
    inside the budget and base does not — which is the shape of the number
    `PLAN.md` §3.3 asks for.
    """
    rate = 0.25 if request.leg == "parent" else 0.05
    return {"held_out/t/held/exact_match": rate * request.step}


class TestAdapt:

    @pytest.fixture(scope="class")
    def adapted(self, parent, tmp_path_factory):
        root = tmp_path_factory.mktemp("adapt")
        record: dict = {}
        result = fork(
            parent["ckpt"], "adapt",
            {"run_dir": str(root / "fork"), "task": HELD_OUT_TASK,
             "budget_steps": 3, "eval_steps": 1, "warmup_steps": 1,
             "target": {"metric": "held_out/t/held/exact_match", "value": 0.5,
                        "direction": "max"}},
            registry=parent["registry"], results_dir=str(root),
            trainer_factory=make_factory(parent["registry"], record),
            validate=_adapt_validate)
        return {"result": result, "record": record, "root": root}

    def test_two_legs_one_from_the_parent_one_from_base(self, adapted):
        legs = {leg.name: leg for leg in adapted["result"].plan.legs}
        assert set(legs) == {"parent", "base"}
        assert legs["parent"].start_checkpoint is not None
        assert legs["base"].start_checkpoint is None

    def test_the_two_legs_have_identical_configs(self, adapted):
        legs = {leg.name: leg for leg in adapted["result"].plan.legs}
        assert legs["parent"].fingerprint() == legs["base"].fingerprint()
        assert legs["parent"].schedule.to_json() == legs["base"].schedule.to_json()
        assert legs["parent"].mixture.hash() == legs["base"].mixture.hash()
        assert legs["parent"].max_steps == legs["base"].max_steps
        assert legs["parent"].seed == legs["base"].seed
        # And nothing else about them is the same:
        assert legs["parent"].output_dir != legs["base"].output_dir

    def test_they_differ_only_in_their_starting_weights(self, adapted, parent):
        """The one difference, made observable.

        The tiny model's SPD bias table is zero-initialised, so a leg that
        started from base has bias norm exactly 0 at ``on_train_begin`` while one
        that started from the parent has the parent's fingerprint.
        """
        from_parent = adapted["record"]["parent"]["start"].bias_norm
        from_base = adapted["record"]["base"]["start"].bias_norm
        assert from_base == pytest.approx(0.0, abs=1e-12)
        assert from_parent == pytest.approx(parent["state"]["bias_norm"], rel=1e-6)
        assert from_parent > 0.0, "the parent's bias channel never left its init"

    def test_a_fresh_schedule_and_optimizer_not_a_resume(self, adapted):
        """``adapt`` takes the weights and nothing else.

        Adam moments and a sampler cursor carried over from the trunk's mixture
        would be a confound in a number that is supposed to be about the weights,
        so both legs start at step 0 under a fresh warmup.
        """
        for name in ("parent", "base"):
            leg = adapted["result"].plan.leg(name)
            assert leg.resume is False
            assert leg.schedule.segments[0].kind == "warmup"
            assert leg.schedule.segments[0].start == 0
            assert adapted["record"][name]["trainer"].state.global_step == 3

    def test_it_trains_the_held_out_task_only(self, adapted):
        for leg in adapted["result"].plan.legs:
            assert [e["name"] for e in leg.mixture_config] == [HELD_OUT_TASK]
            assert [e.name for e in leg.mixture.entries] == [HELD_OUT_TASK]
            # The held-out task is admitted by name through resolve's
            # `allow_held_out`, so the record carries the real name and nothing
            # else — no alias, no renamed spec.
            assert HELD_OUT_TASK in json.dumps(leg.to_json())

    def test_steps_to_target_is_the_first_crossing(self, adapted):
        table = adapted["result"].adaptation_table()
        assert table == {"parent": 2, "base": None}, table
        parent_leg = adapted["result"].legs["parent"]
        assert [step for step, _m in parent_leg.history] == [1, 2, 3, 3]

    def test_a_leg_that_never_arrives_records_none_not_a_gap(self, adapted):
        base = adapted["result"].legs["base"]
        assert base.steps_to_target is None
        assert base.history, "the leg was still evaluated at every eval_steps"


class TestStepsToTarget:

    def test_the_first_crossing_wins_not_the_best_value(self):
        history = [(10, {"m": 0.6}), (20, {"m": 0.4}), (30, {"m": 0.9})]
        assert steps_to_target(history, {"metric": "m", "value": 0.5}) == 10

    def test_a_min_direction_target(self):
        history = [(10, {"m": 2.0}), (20, {"m": 0.4})]
        target = {"metric": "m", "value": 0.5, "direction": "min"}
        assert steps_to_target(history, target) == 20

    def test_a_missing_metric_is_skipped_not_counted(self):
        history = [(10, {}), (20, {"m": 0.9})]
        assert steps_to_target(history, {"metric": "m", "value": 0.5}) == 20


# ─────────────────────────────────────────────────────────────────────────────
# admit
# ─────────────────────────────────────────────────────────────────────────────

def _criterion(**overrides):
    base = {"seeds": 3, "aggregate": "mean"}
    for part in ADMISSION_PARTS:
        rule = "improves" if part == "candidate" else "no_regression"
        base[part] = {"metric": f"{part}/score", "rule": rule, "tolerance": 0.5}
    base.update(overrides)
    return base


class TestAdmit:

    def _plan(self, parent, tmp_path, **config):
        merged = {"run_dir": str(tmp_path / "admit_child"),
                  "candidate": {"name": CANDIDATE_TASK, "weight": 0.5},
                  "budget_steps": 8, "rewarm_steps": 2,
                  "criterion": _criterion()}
        merged.update(config)
        return plan_fork(parent["ckpt"], "admit", merged,
                         registry=parent["registry"],
                         parent_mixture=parent["mixture"])

    def test_the_candidate_joins_the_mixture_at_its_configured_weight(
            self, parent, tmp_path):
        plan = self._plan(parent, tmp_path)
        entries = {e["name"]: e["weight"] for e in plan.leg("admit").mixture_config}
        assert set(entries) == set(TASKS) | {CANDIDATE_TASK}
        assert entries[CANDIDATE_TASK] == 0.5
        assert plan.config_diff["candidate"]["child"] == CANDIDATE_TASK

    def test_a_rewarm_is_appended_and_the_schedule_stays_open(self, parent,
                                                              tmp_path):
        plan = self._plan(parent, tmp_path)
        kinds = [s.kind for s in plan.leg("admit").schedule.segments]
        assert kinds == ["warmup", "stable", "rewarm", "stable"]
        assert plan.leg("admit").schedule.is_open
        assert plan.leg("admit").max_steps == PARENT_STEPS + 8

    def test_a_candidate_already_in_the_mixture_is_refused(self, parent, tmp_path):
        with pytest.raises(ForkError, match="already in the parent"):
            self._plan(parent, tmp_path,
                       candidate={"name": TASKS[0], "weight": 0.5})

    def test_a_candidate_without_a_weight_is_refused(self, parent, tmp_path):
        with pytest.raises(ForkError, match="weight"):
            self._plan(parent, tmp_path, candidate={"name": CANDIDATE_TASK})

    def test_the_criterion_must_be_in_the_config_before_the_fork_runs(
            self, parent, tmp_path):
        with pytest.raises(ForkError, match="criterion"):
            self._plan(parent, tmp_path, criterion=None)
        with pytest.raises(ForkError, match="missing"):
            self._plan(parent, tmp_path,
                       criterion={k: v for k, v in _criterion().items()
                                  if k != "text_only"})
        with pytest.raises(ForkError, match="seeds"):
            self._plan(parent, tmp_path,
                       criterion={k: v for k, v in _criterion().items()
                                  if k != "seeds"})

    def test_a_part_cannot_quietly_change_its_rule(self, parent, tmp_path):
        bad = _criterion()
        bad["held_out"] = dict(bad["held_out"], rule="improves")
        with pytest.raises(ForkError, match="PLAN.md"):
            self._plan(parent, tmp_path, criterion=bad)


class TestAdmissionCriterion:

    def test_all_four_parts_holding_is_a_pass(self):
        criterion = check_criterion(_criterion())
        baseline = {f"{p}/score": 1.0 for p in ADMISSION_PARTS}
        metrics = dict(baseline, **{"candidate/score": 2.0})
        verdict = check_admission(criterion, metrics, baseline)
        assert verdict.decided and verdict.passed

    def test_a_regression_beyond_the_bar_fails(self):
        criterion = check_criterion(_criterion())
        baseline = {f"{p}/score": 1.0 for p in ADMISSION_PARTS}
        metrics = dict(baseline, **{"candidate/score": 2.0, "held_out/score": 0.2})
        verdict = check_admission(criterion, metrics, baseline)
        assert verdict.decided and not verdict.passed
        assert "held_out" in verdict.reason

    def test_a_candidate_that_only_moves_within_the_noise_bar_fails(self):
        criterion = check_criterion(_criterion())
        baseline = {f"{p}/score": 1.0 for p in ADMISSION_PARTS}
        metrics = dict(baseline, **{"candidate/score": 1.1})
        verdict = check_admission(criterion, metrics, baseline)
        assert verdict.decided and not verdict.passed

    def test_a_missing_suite_is_undecided_and_never_a_pass(self):
        """The seam being honest: the regression suites land with the trunk, so
        a fork run before them must not read as a clean gate."""
        criterion = check_criterion(_criterion())
        baseline = {f"{p}/score": 1.0 for p in ADMISSION_PARTS}
        metrics = {k: v for k, v in baseline.items() if "text_only" not in k}
        metrics["candidate/score"] = 2.0
        verdict = check_admission(criterion, metrics, baseline)
        assert verdict.decided is False
        assert verdict.passed is None
        assert verdict.missing == ("text_only",)


# ─────────────────────────────────────────────────────────────────────────────
# Planning without running
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanOnly:

    def test_a_fork_without_a_trainer_factory_lays_everything_down(
            self, parent, tmp_path):
        """The shape `validate` mode and a Slurm-submitted leg both use."""
        result = fork(parent["ckpt"], "anneal",
                      {"run_dir": str(tmp_path / "child"), "decay_steps": 2},
                      registry=parent["registry"],
                      parent_mixture=parent["mixture"], results_dir=str(tmp_path),
                      runs_jsonl=str(tmp_path / "runs.jsonl"))
        assert result.ran is False
        assert result.legs == {}
        leg = result.plan.leg("anneal")
        assert ckpt_mod.is_complete(leg.start_checkpoint)
        assert ckpt_mod.is_pinned(leg.start_checkpoint)
        assert Lineage(str(tmp_path)).entries_for(leg.output_dir)
        with open(str(tmp_path / "runs.jsonl")) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert rows[0]["mode"] == "anneal" and rows[0]["kind"] == "fork"

    def test_the_default_decay_is_a_tenth_of_the_parents_steps(self, parent,
                                                              tmp_path):
        plan = plan_fork(parent["ckpt"], "anneal",
                         {"run_dir": str(tmp_path / "c")},
                         registry=parent["registry"],
                         parent_mixture=parent["mixture"])
        decay = plan.leg("anneal").schedule.segments[-1]
        assert decay.kind == "decay"
        assert decay.steps == max(1, round(0.10 * PARENT_STEPS))
        assert decay.factor_end == pytest.approx(0.1), "MOLECULE_GENERALIST §7: lr/10"

    def test_an_absolute_lr_min_is_converted_against_the_parents_lr(
            self, parent, tmp_path):
        plan = plan_fork(parent["ckpt"], "anneal",
                         {"run_dir": str(tmp_path / "c"), "decay_steps": 2,
                          "lr_min": LR / 4},
                         registry=parent["registry"],
                         parent_mixture=parent["mixture"])
        assert plan.leg("anneal").schedule.segments[-1].factor_end \
            == pytest.approx(0.25)
