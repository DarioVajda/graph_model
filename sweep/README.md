# `sweep` — generic sweep runner

Turns one JSONC config into a resolved set of runs and dispatches each as its
own subprocess — locally (sequential) or as Slurm jobs. It is
experiment-agnostic: the experiment module is named on the command line, every
config key is rendered to a CLI flag, and the experiment's own argparse program
decides what each flag means (see
[`src/experiments/template`](../src/experiments/template) for the
experiment-side contract and a copy-me reference layout).

```bash
python3 -m sweep <experiment_module> <config.jsonc>   # run every resolved config
python3 -m sweep.report <sweep_dir>                   # aggregate runs.jsonl → report.md
```

Run from the **repo root** — the config path resolves relative to the current
directory, but experiments generally assume repo-root-relative paths.

## How the config expands

The **shape** of each value decides how it sweeps (no marker keywords):

| Value in the JSON | Meaning |
|---|---|
| scalar (`1e-3`, `"v2-flex"`) | fixed in every run |
| list of scalars (`[0, 1, 2]`) | a sweep **axis** (one run per value) |
| list of lists (`[[640,768],[512]]`) | an axis over a list-valued param |
| list of objects (`[{…}, {…}]`) | a **bundle**: params that vary *together*; each object's keys flatten into the run, and the bundle's label disappears |

The run set is the **cartesian product** of all axes (a bundle is one axis).
Keys use the **singular** form (`impl` / `k_hop` / `seed`) — a list value is
what makes them sweep — and map 1:1 to the experiment's CLI flags (bools as
`--x`/`--no-x`, lists comma-joined). A key may be defined in exactly one place
(top-level *or* one bundle); a collision is an error. The reserved keys
`name`, `results_dir`, and `execution` are consumed by the runner, never swept.

## Outputs

Each run appends one line to `<results_dir>/<sweep_name>/runs.jsonl`. The sweep
dir also holds the source config verbatim (`config.json`/`.jsonc`), per-run
`resolved/*.json`, and `logs/`. After a **local** sweep a `report.md` is
written automatically; aggregate any sweep (e.g. after sbatch jobs finish) with
`python3 -m sweep.report <sweep_dir>`.

## Slurm / sbatch execution

Set `execution.mode: "sbatch"` in the config; `python3 -m sweep …` then builds
and submits the jobs. `granularity: "per_config"` submits one job per run;
`"single"` one job running every config in sequence; `max_concurrent` caps
per-config jobs via a throttled job array (`--array=0-(K-1)%N`). The exact
commands are always written to `<sweep_dir>/sbatch_commands.sh` and the per-job
scripts to `<sweep_dir>/jobs/` (submitted unless `dry_run: true` or `sbatch` is
absent).

```jsonc
"execution": {
  "mode": "sbatch",
  "sbatch": {
    "granularity": "per_config",
    "max_concurrent": 4,          // omit for no cap
    "partition": "frida", "account": "povejmo",
    "gpus": "B200:1", "cpus": 16, "mem": "64G", "time": "24:00:00",
    "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh",
    "dry_run": false              // true: write sbatch_commands.sh + jobs/ without submitting
  }
}
```

Each job runs `sweep/slurm_launch.sh` *inside* the container (an `srun` step
within the sbatch allocation — bare `srun` is blocked on `frida`). The
container matters: the B200 hosts are py3.12 but the project `.venv` is py3.10,
so jobs run in the py3.10 container; the launcher puts `.venv/bin` first on
`PATH` (a bare `python` is the venv interpreter), forwards `HOME` (HF/wandb
creds + model cache), runs `login.sh`, and sets a per-job
`TORCHINDUCTOR_CACHE_DIR`.
