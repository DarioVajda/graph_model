"""
``sweep`` — a generic, experiment-agnostic sweep runner.

Turns a single JSONC sweep config into a list of flat, fully-resolved per-run
parameter dicts, then dispatches each as its own subprocess — locally
(sequential) or as Slurm/sbatch jobs. It knows nothing about any particular
experiment: the experiment module is named on the command line, every config key
is rendered to a CLI flag, and the experiment's own argparse program decides what
each flag means (and may reject ones it does not support).

    python3 -m sweep <experiment_module> <config.json>

e.g.  python3 -m sweep src.experiments.expressiveness configs/my_sweep.jsonc

Modules:
    expand   config loading + sweep expansion (the cartesian/bundle semantics)
    execute  local / sbatch dispatch (renders params -> flags, invokes the experiment)
    report   read a sweep's runs.jsonl -> aggregate table
"""
