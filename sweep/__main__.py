"""
``python3 -m sweep <experiment_module> <config.json>``

Generic sweep runner CLI. The experiment is named explicitly (a dotted module
like ``src.experiments.expressiveness``, or the equivalent ``src/experiments/...``
path); the config is a JSONC sweep file. The runner expands the config and runs
each resolved configuration against the experiment (locally or via sbatch).
"""

import sys

from .execute import run_sweep


USAGE = "usage: python3 -m sweep <experiment_module> <config.json>"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    experiment, config_path = argv
    run_sweep(experiment, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
