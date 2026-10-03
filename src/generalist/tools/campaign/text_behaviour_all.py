"""Run the `text_behaviour` validator over every annealed cell of a config.

`tools/campaign/probe_all.py`'s shape for the text-behaviour measurement, and
idempotent for the same reason: a cell whose anneal has not finished is skipped
and reported, a cell already measured is skipped and reported, and only the
ready ones are submitted.

    src/generalist/tools/campaign/text_behaviour_all.py --config <cfg>
    src/generalist/tools/campaign/text_behaviour_all.py --config <cfg> --dry-run

**The validator set is overridden, on purpose.** These checkpoints were trained
before `text_behaviour` existed, and adding it to `DEFAULT_VALIDATORS` would
rename every run that used that set — ``validator_specs`` is inside
``config_hash``. So the cells keep the set they trained under and this job asks
for the ``text`` set instead, through the generated ``--validators`` flag. `eval`
mode writes both the config's hash and the checkpoint's into its record, so the
override is visible in the artifact rather than implied by the command line.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
OUT_DIR = "src/generalist/results/text_behaviour"


def anneal_checkpoint(config):
    """``(checkpoint, why)`` — the annealed model to measure, or why there is none.

    `probe_all.py`'s rule, and for the same reason: an anneal's reportable
    model is its last step, which is ``parent_step + decay_steps + 1`` and not a
    round number, so take the highest ``checkpoint-N`` rather than recompute the
    schedule here.
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


def out_dir(cell: str) -> str:
    return os.path.join(REPO, OUT_DIR, cell)


def already_measured(cell: str) -> bool:
    directory = out_dir(cell)
    return os.path.isdir(directory) and any(
        n.startswith("eval_step") and n.endswith(".json")
        for n in os.listdir(directory))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="1", help="GPUs per measurement job.")
    parser.add_argument("--gpu-brands", default="B300|B200",
                        help="`run_cli.sh`'s brand list. The graph arm's forward "
                             "pass is what sized the anneal's memory, and this "
                             "job holds two logit tensors at once; the default "
                             "keeps it off the 40 GB A100s for the same reason "
                             "`anneal_all.py` does.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from src.generalist.config import RunConfig, config_cells

    submitted, waiting, done = [], [], []
    for cell, values in config_cells(args.config).items():
        config = RunConfig(**values)
        if already_measured(cell):
            done.append(cell)
            continue
        ckpt, why = anneal_checkpoint(config)
        if ckpt is None:
            waiting.append((cell, why))
            continue
        cmd = [os.path.join(REPO, "src/generalist/tools/launch/run_cli.sh"), "eval",
               "--config", args.config, "--cell", cell,
               "--checkpoint", ckpt,
               "--validators", "text",
               "--only-validators", "text_behaviour",
               "--out", out_dir(cell)]
        print(f"[text]   {cell}\n         from {ckpt}")
        if args.dry_run:
            submitted.append(cell)
            continue
        env = dict(os.environ, GPU=args.gpus, WAIT="0",
                   GPUS=args.gpu_brands, NAME=f"text_{cell}")
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        (submitted if rc == 0 else waiting).append(
            cell if rc == 0 else (cell, f"submission exited {rc}"))

    for cell in done:
        print(f"[skip]   {cell}: already measured")
    for cell, why in waiting:
        print(f"[wait]   {cell}: {why}")
    print(f"\n{len(submitted)} submitted, {len(waiting)} waiting, {len(done)} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
