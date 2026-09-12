"""Score every annealed cell of a config on the clean property instrument.

`tools/notation_probe.py --checkpoint` is the instrument §8 and §9 quote for
property prediction: it drops the union of truncated rows across the arms that
exist, so every arm scores the *identical* molecules. It takes one cell at a
time, and a campaign has six — which is six commands to remember, six chances to
point one at the wrong checkpoint, and no record of which ones are still missing.

This is `tools/anneal_all.py`'s shape for the scoring step, and it is idempotent
for the same reason: a cell whose anneal has not finished is skipped and
reported, a cell already scored is skipped and reported, and only the ready ones
are submitted. Run it as often as you like while the anneals are still going.

    src/generalist/tools/probe_all.py --config <cfg>
    src/generalist/tools/probe_all.py --config <cfg> --dry-run

**The backbone has to be passed through.** `notation_probe` builds its adapter
config from ``--model-name``, and that is what decides which *build* it reads —
base weights and instruct weights resolve to different `build_version`s because
the prompt format differs. Taking it off the cell's own config rather than the
probe's default is what stops an instruct checkpoint being scored against
base-format data, which would not raise: the sources exist, they are simply the
wrong ones.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
PROBE_DIR = "src/generalist/results/notation_probe"


def anneal_checkpoint(config):
    """``(checkpoint, why)`` — the annealed model to score, or why there is none.

    An anneal's reportable model is its LAST step, which is
    ``parent_step + decay_steps + 1`` and not a round number; rather than
    recompute the schedule here, take the highest ``checkpoint-N`` under the
    fork's ``anneal/`` directory. `result.json` beside it is what says the leg
    finished, so it is checked first.
    """
    fork_dir = f"{config.run_dir()}-anneal-{config.max_steps}"
    if not os.path.exists(os.path.join(fork_dir, "result.json")):
        if not os.path.isdir(fork_dir):
            return None, "no anneal fork yet"
        return None, "anneal has not written result.json"
    anneal = os.path.join(fork_dir, "anneal")
    steps = [int(n.split("-")[1]) for n in os.listdir(anneal)
             if n.startswith("checkpoint-") and n.split("-")[1].isdigit()]
    if not steps:
        return None, "anneal finished but left no checkpoint"
    return os.path.join(anneal, f"checkpoint-{max(steps)}"), ""


def already_scored(config) -> bool:
    return os.path.exists(os.path.join(
        REPO, PROBE_DIR, f"trained_{config.run_name}.json"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="1", help="GPUs per scoring job.")
    parser.add_argument("--max-samples", default="500",
                        help="500 is what every campaign number was read at.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from src.generalist.config import RunConfig, config_cells

    submitted, waiting, done = [], [], []
    for cell, values in config_cells(args.config).items():
        config = RunConfig(**values)
        if already_scored(config):
            done.append(cell)
            continue
        ckpt, why = anneal_checkpoint(config)
        if ckpt is None:
            waiting.append((cell, why))
            continue
        cmd = [os.path.join(REPO, "src/generalist/tools/run_cli.sh"),
               "--run-config", args.config, "--cell", cell,
               "--checkpoint", ckpt, "--out", PROBE_DIR,
               "--max-samples", args.max_samples,
               # The cell's own backbone, not the probe's default — see the
               # module docstring for what reading the wrong build looks like.
               "--model-name", config.model_name]
        print(f"[probe]  {cell}\n         from {ckpt}")
        if args.dry_run:
            submitted.append(cell)
            continue
        env = dict(os.environ, GPU=args.gpus, WAIT="0",
                   RUNMOD="src.generalist.tools.notation_probe",
                   GPUS="B300|B200", NAME=f"probe_{cell}")
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        (submitted if rc == 0 else waiting).append(
            cell if rc == 0 else (cell, f"submission exited {rc}"))

    for cell in done:
        print(f"[skip]   {cell}: already scored")
    for cell, why in waiting:
        print(f"[wait]   {cell}: {why}")
    print(f"\n{len(submitted)} submitted, {len(waiting)} waiting, {len(done)} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
