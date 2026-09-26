"""Submit the `anneal` fork of every cell of a config whose trunk has landed.

A generalist cell is **not reportable until its anneal fork has run**: the trunk
stops mid-WSD-stable-phase by construction, so its last checkpoint is not the
model any comparison should use. The fork is deliberately not part of `chain.sh`
— it is a separate submission per cell — which is exactly how a finished campaign
comes to produce no quotable number despite having cost the compute.

This closes that gap without anyone watching a queue. It is **idempotent**: a
cell whose trunk has not finished is skipped and reported, a cell whose fork has
already run is skipped and reported, and only the ready ones are submitted. Run
it as often as you like while the trunks are still going.

    src/generalist/tools/anneal_all.py --config <cfg>
    src/generalist/tools/anneal_all.py --config <cfg> --dry-run

Waiting is deliberately not built in. A watcher process that has to outlive a
14-hour trunk is a thing to babysit; re-running a command that does nothing until
there is something to do is not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
FORK_CONFIG = "src/generalist/configs/forks/anneal_molecule_generalist.jsonc"


def trunk_state(config, cell_config):
    """``(checkpoint, why)`` — the checkpoint to fork from, or why there is none."""
    from src.generalist.checkpoint import COMPLETE_MARKER

    run_dir = cell_config.run_dir()
    step = cell_config.max_steps
    if not step:
        return None, "max_steps is 0; this tool forks a pinned-horizon run"
    ckpt = os.path.join(run_dir, f"checkpoint-{step}")
    if not os.path.isdir(ckpt):
        return None, f"trunk has not reached step {step}"
    if not os.path.exists(os.path.join(ckpt, COMPLETE_MARKER)):
        return None, f"checkpoint-{step} is mid-write (no {COMPLETE_MARKER})"
    return ckpt, ""


def trunk_job(cell_config) -> str:
    """The queued Slurm job id of this cell's trunk, or "" if it is not queued.

    `chain.sh` names every chunk ``gen_<run_name>_c<i>``, which makes the queue
    itself the lookup table — no job id has to be written down or passed along.
    The *last* chunk is the one an anneal must follow, so the highest ``_c``
    suffix wins.
    """
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
        capture_output=True, text=True).stdout
    best, best_chunk = "", -1
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        job, name = parts
        prefix = f"gen_{cell_config.run_name}_c"
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            chunk = int(name[len(prefix):])
            if chunk > best_chunk:
                best, best_chunk = job, chunk
    return best


def fork_done(cell_config) -> bool:
    """Has this cell's anneal already produced a result?

    `result.json` sits one level *above* the `anneal/` directory that holds the
    checkpoints, which is not where anyone looks the first time.
    """
    fork_dir = f"{cell_config.run_dir()}-anneal-{cell_config.max_steps}"
    return os.path.exists(os.path.join(fork_dir, "result.json"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--fork-config", default=FORK_CONFIG)
    parser.add_argument("--gpus", default="1", help="GPUs per fork job.")
    parser.add_argument("--gpu-brands", default="B300|B200",
                        help="`run_cli.sh`'s brand list. **Not optional in "
                             "practice on the graph arm**: the default spans "
                             "every fast brand including 40 GB A100s, and a "
                             "graph-arm fork placed on one dies with a CUDA OOM "
                             "after it has queued, waited and loaded — which is "
                             "how six forks were lost on 2026-09-11. The trunk "
                             "gets its brands from the config; a fork has no "
                             "config to read them from, so it has to be told.")
    parser.add_argument("--time", default="08:00:00",
                        help="wall clock per fork. `run_cli.sh` defaults to two "
                             "hours and an anneal does not fit in it: the decay "
                             "is only a tenth of the trunk, but it ends with the "
                             "full validator suite over every task in the "
                             "mixture, and the generative ones are most of that "
                             "time. A fork that walls at 95 % has produced "
                             "nothing.")
    parser.add_argument("--inductor-cache", default=".inductor_cache/generalist",
                        help="share compiled flex kernels with the trunk's cache. "
                             "Point it at the one the config names, or the fork "
                             "spends its first minutes re-autotuning shapes the "
                             "trunk already compiled.")
    parser.add_argument("--queue-behind-trunk", action="store_true",
                        help="submit a cell whose trunk is still QUEUED, held by "
                             "a Slurm dependency on it, instead of skipping it. "
                             "Turns 'run this again when the trunks land' into "
                             "something that happens without anyone present.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted, submit nothing.")
    args = parser.parse_args(argv)

    from src.generalist.config import RunConfig, config_cells

    submitted, waiting, done = [], [], []
    for cell, values in config_cells(args.config).items():
        config = RunConfig(**values)
        if fork_done(config):
            done.append(cell)
            continue
        ckpt, why = trunk_state(args.config, config)
        depend = ""
        if ckpt is None and args.queue_behind_trunk:
            # The checkpoint does not exist yet *and does not have to*: the fork
            # is queued behind the trunk that will write it, so Slurm starts it
            # only once that job has exited ok. The path is deterministic — a
            # pinned-horizon run ends at `max_steps` — so naming it before it
            # exists is safe, and `fork` re-checks it anyway.
            #
            # This is what makes the rest of a campaign survive the shell that
            # launched it: `--wait` cannot outlive a session, a dependency can.
            job = trunk_job(config)
            if job:
                ckpt = os.path.join(config.run_dir(), f"checkpoint-{config.max_steps}")
                depend = f"afterok:{job}"
        if ckpt is None:
            waiting.append((cell, why))
            continue
        cmd = [os.path.join(REPO, "src/generalist/tools/run_cli.sh"), "fork",
               "--from", ckpt, "--mode", "anneal",
               "--fork-config", args.fork_config,
               "--config", args.config, "--cell", cell]
        print(f"[anneal] {cell}\n         from {ckpt}"
              + (f"\n         held until {depend}" if depend else ""))
        if args.dry_run:
            submitted.append(cell)
            continue
        # WAIT=0: submit and return. Blocking here would serialise six forks
        # that have no reason to wait for each other, and would tie the whole
        # batch to one shell staying alive.
        env = dict(os.environ, GPU=args.gpus, WAIT="0",
                   INDUCTOR_CACHE=args.inductor_cache,
                   GPUS=args.gpu_brands,
                   TIME=args.time,
                   NAME=f"anneal_{cell}")
        if depend:
            env["DEPENDENCY"] = depend
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        (submitted if rc == 0 else waiting).append(
            cell if rc == 0 else (cell, f"submission exited {rc}"))

    for cell in done:
        print(f"[skip]   {cell}: fork already has a result.json")
    for cell, why in waiting:
        print(f"[wait]   {cell}: {why}")
    print(f"\n{len(submitted)} submitted, {len(waiting)} waiting, {len(done)} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
