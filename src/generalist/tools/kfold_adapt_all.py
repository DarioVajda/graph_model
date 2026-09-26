"""Submit every adaptation leg of `KFOLD_TRANSFER.md`, queued behind its trunk.

The study is twenty-two forks across four folds and sixteen tasks, each needing
a task name, a fold, a trunk checkpoint, and — because `--task` rewrites only
the metric key's task segment — sometimes a metric and a threshold of its own.
Twenty-two hand-typed commands is how the wrong threshold reaches a queue, which
is the one failure §6a exists to prevent. This builds them from the fold table
instead.

    src/generalist/tools/kfold_adapt_all.py --dry-run
    src/generalist/tools/kfold_adapt_all.py --queue-behind-trunk

Like `anneal_all.py` it is **idempotent** and does not wait: a fork that already
has a `result.json` is skipped, a fold whose trunk has not landed is reported
and skipped, and `--queue-behind-trunk` submits against the trunk's Slurm job id
with an `afterok` dependency so the legs start themselves overnight.

**Three things this encodes that a command line would not.**

*Which tasks run.* Sixteen of nineteen. Tox21 and SIDER are gradient tasks and
never anchor numbers (§1), and ClinTox has no train split to fork on (§3), so
none of the three can be a leg.

*Which fork config, metric and threshold each task takes.* Four configs by task
kind; ChEBI-20 needs `--target-metric` because its answer_kind is `text` where
g2s's is `smiles`; BACE, BBBP and HIV are the only tasks whose threshold exists
today and each needs its own.

*How many parents a task has.* One, except `bond_path` and `longest_chain`,
which are held out of every mixture and so have four valid parents. Their extra
three forks run `--starts parent` alone: the base leg does not depend on the
parent, so the fold A fork's base leg is the shared reference and running it
four times would be the same run four times.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
FORKS = "src/generalist/configs/forks/"
PROBES = "src/generalist/configs/probes/"

#: fold -> the probe config holding that fold's trunk cells.
FOLD_CONFIG = {
    "A": PROBES + "010_kfold_fold_a.jsonc",
    "B": PROBES + "011_kfold_fold_b.jsonc",
    "C": PROBES + "012_kfold_fold_c.jsonc",
    "D": PROBES + "013_kfold_fold_d.jsonc",
}

#: The four per-task-kind fork configs of §9.
TIER_A = FORKS + "adapt_kfold_tier_a.jsonc"
HELD_OUT_FAMILY = FORKS + "adapt_kfold_held_out_family.jsonc"
TIER_B = FORKS + "adapt_kfold_tier_b.jsonc"
GENERATION = FORKS + "adapt_kfold_generation.jsonc"

#: Wall clock per fork config. **A fork is two legs in one job**, and
#: `run_cli.sh` defaults to two hours, which covers neither. At the ~2.2 s/it a
#: single-GPU graph-arm leg runs at, a Tier-A fork is 2 x 2,000 steps plus
#: eighty evaluations; a generation fork is 2 x 3,000 plus sixty generative
#: evaluations of up to 500 samples, which is the slow part and not the
#: training. Generous rather than tight: a leg that hits the wall at 90 % has
#: produced nothing, and an over-long limit costs only queue position.
WALL_CLOCK = {
    TIER_A: "10:00:00",
    HELD_OUT_FAMILY: "10:00:00",
    TIER_B: "06:00:00",
    GENERATION: "20:00:00",
}

#: task -> (fork config, metric override or None, threshold or None).
#:
#: A `None` threshold is not an oversight: the config carries an `anchor` saying
#: where the number will come from, the leg records its full curve, and the
#: crossing is read off it once the anchor lands (§6a). Passing a stand-in here
#: would be the failure that convention replaced.
TASKS = {
    # Fold A, Tier-A families that train in every other mixture.
    "ring_membership": (TIER_A, None, None),
    "aromatic_ring": (TIER_A, None, None),
    "ring_size": (TIER_A, None, None),
    "ring_count": (TIER_A, None, None),
    # Fold B, the same kind.
    "fg_presence": (TIER_A, None, None),
    "fg_count": (TIER_A, None, None),
    "fg_atom_membership": (TIER_A, None, None),
    "stereo_potential": (TIER_A, None, None),
    "stereo_assigned": (TIER_A, None, None),
    # Held out of every mixture, so `held_out` claims them and `in_mixture`
    # skips them; the fork config asks `held_out` for the test split by name.
    "bond_path": (HELD_OUT_FAMILY, None, None),
    "longest_chain": (HELD_OUT_FAMILY, None, None),
    # Fold C. The only three thresholds the study owns today: 0.95 x the
    # three-seed specialist test AUROC of `molecules/TODO.md` §6.
    "bace": (TIER_B, None, 0.7792),
    "bbbp": (TIER_B, None, 0.6703),
    "hiv": (TIER_B, None, 0.7306),
    # Fold D. g2s is `smiles` and scores roundtrip_match, which is the fork
    # config's default; ChEBI-20 is `text` and scores bleu2, which is not.
    "g2s": (GENERATION, None, None),
    "chebi20": (GENERATION, "in_mixture/mol/chebi20/test/bleu2", None),
}

#: Held out of every mixture, so every fold is a valid parent (§6a). The fold
#: that nominally owns them in `TRANSFER_FOLDS` runs both legs; the other three
#: run the parent leg alone against that shared base leg.
EVERY_FOLD = ("bond_path", "longest_chain")


def legs():
    """``(fold, task, fork_config, metric, value, starts)`` for all 22 forks."""
    from src.generalist.config import TRANSFER_FOLDS

    owner = {task: fold for fold, tasks in TRANSFER_FOLDS.items()
             for task in tasks}
    out = []
    for fold in sorted(TRANSFER_FOLDS):
        for task in TRANSFER_FOLDS[fold]:
            if task not in TASKS:
                continue          # tox21, sider, clintox — no anchor, no split
            config, metric, value = TASKS[task]
            out.append((fold, task, config, metric, value, None))
    for task in EVERY_FOLD:
        config, metric, value = TASKS[task]
        for fold in sorted(TRANSFER_FOLDS):
            if fold == owner[task]:
                continue          # already emitted above, with both legs
            out.append((fold, task, config, metric, value, "parent"))
    return out


def trunk(config_path, cell_config):
    """``(checkpoint, why)`` — the trunk to fork from, or why there is none."""
    from src.generalist.checkpoint import COMPLETE_MARKER

    step = cell_config.max_steps
    if not step:
        return None, "max_steps is 0; this tool forks a pinned-horizon run"
    path = os.path.join(cell_config.run_dir(), f"checkpoint-{step}")
    if not os.path.isdir(path):
        return None, f"trunk has not reached step {step}"
    if not os.path.exists(os.path.join(path, COMPLETE_MARKER)):
        return None, f"checkpoint-{step} is mid-write (no {COMPLETE_MARKER})"
    return path, ""


def trunk_job(cell_config) -> str:
    """The queued Slurm job id of this cell's trunk, or "".

    `chain.sh` names every chunk ``gen_<run_name>_c<i>``, so the queue is the
    lookup table and no job id has to be written down. The last chunk is the one
    a fork must follow.
    """
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
        capture_output=True, text=True).stdout
    best, best_chunk = "", -1
    prefix = f"gen_{cell_config.run_name}_c"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        job, name = parts
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            chunk = int(name[len(prefix):])
            if chunk > best_chunk:
                best, best_chunk = job, chunk
    return best


def fork_dir(cell_config, task) -> str:
    """Where this leg lands — and it has to be named, not defaulted.

    `fork`'s default child name is ``<parent>-<mode>-<step>``, which carries no
    task. Fold A runs six adapt forks off one trunk, so all six would resolve to
    the same directory and overwrite each other's record. The task goes in the
    name here and travels on `--run-dir`.
    """
    return (f"{cell_config.run_dir()}-adapt-{task.split('/')[-1]}"
            f"-{cell_config.max_steps}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="A,B,C,D",
                        help="restrict to these folds.")
    parser.add_argument("--only", default=None,
                        help="restrict to these task names, comma-separated.")
    parser.add_argument("--seed-cell", default=None,
                        help="fold A has three trunk seeds; fork off this cell "
                             "(default: the config's first, which is seed 0). "
                             "The adaptation number is a ratio within one "
                             "parent, so one seed is the unit here and the "
                             "other two are the trunk-variance measurement.")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--gpu-brands", default="GPU_MEM:80GB|GPU_MEM:180GB|GPU_MEM:288GB",
                        help="§11: a graph-arm job needs more than 40 GB, and "
                             "`GPU_BRD:A100` matches both the 80 GB ana and the "
                             "40 GB axa. Constrain on memory, which is a real "
                             "feature, not on brand.")
    parser.add_argument("--inductor-cache", default=".inductor_cache/generalist",
                        help="share compiled flex kernels with the trunks.")
    parser.add_argument("--queue-behind-trunk", action="store_true",
                        help="submit against a trunk that is still queued, held "
                             "by an afterok dependency on it, instead of "
                             "skipping. This is what lets the whole study go out "
                             "in one evening.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted, submit nothing.")
    args = parser.parse_args(argv)

    from src.generalist.config import RunConfig, config_cells

    folds = {part.strip().upper() for part in args.folds.split(",")
             if part.strip()}
    only = ({part.strip() for part in args.only.split(",") if part.strip()}
            if args.only else None)

    submitted, waiting, done = [], [], []
    for fold, task, fork_config, metric, value, starts in legs():
        if fold not in folds or (only and task not in only):
            continue
        config_path = FOLD_CONFIG[fold]
        cells = config_cells(config_path)
        cell = args.seed_cell if args.seed_cell in cells else sorted(cells)[0]
        cell_config = RunConfig(**cells[cell])
        label = f"fold {fold} / mol/{task}"

        out_dir = fork_dir(cell_config, task)
        if os.path.exists(os.path.join(out_dir, "result.json")):
            done.append(label)
            continue

        checkpoint, why = trunk(config_path, cell_config)
        depend = ""
        if checkpoint is None and args.queue_behind_trunk:
            # The checkpoint does not exist yet and does not have to: a
            # pinned-horizon trunk ends at `max_steps`, so its path is known
            # before it is written and `fork` re-checks it on the way in.
            job = trunk_job(cell_config)
            if job:
                checkpoint = os.path.join(cell_config.run_dir(),
                                          f"checkpoint-{cell_config.max_steps}")
                depend = f"afterok:{job}"
        if checkpoint is None:
            waiting.append((label, why))
            continue

        cmd = [os.path.join(REPO, "src/generalist/tools/run_cli.sh"), "fork",
               "--from", checkpoint, "--mode", "adapt",
               "--fork-config", fork_config,
               "--config", config_path, "--cell", cell,
               "--task", f"mol/{task}", "--run-dir", out_dir,
               "--held-out-by", fold]
        if metric:
            cmd += ["--target-metric", metric]
        if value is not None:
            cmd += ["--target-value", str(value)]
        if starts:
            cmd += ["--starts", starts]

        print(f"[adapt]  {label}\n         from {checkpoint}"
              + f"\n         wall clock {WALL_CLOCK[fork_config]}"
              + (f"\n         held until {depend}" if depend else "")
              + (f"\n         starts {starts}" if starts else "")
              + (f"\n         target {value}" if value is not None
                 else "\n         target deferred to its anchor"))
        if args.dry_run:
            submitted.append(label)
            continue
        # The seed rides in the job name. Fold A's three trunks each get their
        # own legs for the same six tasks, so without it the queue shows three
        # rows called `adapt_A_ring_membership` and there is no way to tell from
        # `squeue` which trunk a job belongs to. The run dirs were always
        # distinct; only the label was ambiguous.
        suffix = cell.rsplit("_", 1)[-1]
        seeded = (f"_{suffix}" if suffix[:1] == "s" and suffix[1:].isdigit()
                  else "")
        env = dict(os.environ, GPU=args.gpus, WAIT="0",
                   INDUCTOR_CACHE=args.inductor_cache,
                   GPUS=args.gpu_brands,
                   TIME=WALL_CLOCK[fork_config],
                   NAME=f"adapt_{fold}_{task}{seeded}")
        if depend:
            env["DEPENDENCY"] = depend
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        if rc == 0:
            submitted.append(label)
        else:
            waiting.append((label, f"submission exited {rc}"))

    for label in done:
        print(f"[skip]   {label}: already has a result.json")
    for label, why in waiting:
        print(f"[wait]   {label}: {why}")
    print(f"\n{len(submitted)} submitted, {len(waiting)} waiting, "
          f"{len(done)} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
