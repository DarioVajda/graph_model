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
`"single"` one job running every config in sequence; `max_concurrent` submits the
per-config jobs as one job array (`--array=0-(K-1)%N`) instead of K separate jobs,
capped at N running at a time — set it to K unless you mean to throttle. The exact
commands are always written to `<sweep_dir>/sbatch_commands.sh` and the per-job
scripts to `<sweep_dir>/jobs/` (submitted unless `dry_run: true` or `sbatch` is
absent).

```jsonc
"execution": {
  "mode": "sbatch",
  "sbatch": {
    "granularity": "per_config",
    "max_concurrent": 4,          // = number of runs; selects the ARRAY path
    "partition": "frida", "account": "povejmo",
    "gpus": "B200", "cpus": 16, "mem": "64G", "time": "24:00:00",
    // or a list to accept any of several types (rendered as --gres gpu:N
    // --constraint GPU_BRD:B200|GPU_BRD:B300): "gpus": ["B200", "B300"],
    "gpus_per_config": 1,         // GPUs per run == DDP ranks; >1 uses torchrun
    "inductor_cache": null,       // path to a compile cache SHARED across the sweep
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
creds + model cache), runs `login.sh`, and sets `TORCHINDUCTOR_CACHE_DIR`.

### GPUs per config, and DDP

### `max_concurrent`: set it to the number of runs, don't omit it

**This field does two things, and only one of them is a throttle.** Setting it at
all is what selects the **job array** path (`--array=0-(K-1)%N`, one job id, one
`array_map.tsv`); omitting it falls through to submitting **K independent sbatch
jobs** (`execute.py:441`). The array is almost always what you want, so the house
default is

```jsonc
"max_concurrent": <total number of runs>   // array, no effective throttle
```

which is `%K` on K tasks — every task eligible at once, with all the array
machinery. Set it *lower* than K only for the one situation the throttle is for:
an **idle** cluster, where a large array would claim every GPU on the partition
just because nothing else wants them. On a busy partition a lower cap does nothing
useful — Slurm's fair-share scheduler is already arbitrating between users, and
capping your own array only makes your sweep finish later without releasing
anything to anyone else.

What the array buys you, concretely: one job id, so a mistake is fixable in place
on every queued task at once, without resubmitting —

```bash
scontrol update jobid=<id> ArrayTaskThrottle=0                  # lift the cap
scontrol update jobid=<id> Features="GPU_BRD:H100|GPU_BRD:A100" # widen hardware
```

Running tasks are unaffected; queued ones become eligible immediately. Slurm
refuses both for tasks that have already started and names them — that refusal is
a report, not a failure. With K separate jobs you would be looping over K ids.

The array also **shuffles** index→config (seeded on the sweep name) so a
demand-sorted config list can't cluster the heavy runs into one wave — which is
why array task *n* is not run *n*; read `array_map.tsv` for the mapping.

### `gpus` names node *features*, not gres names

A list renders to `--constraint GPU_BRD:X|GPU_BRD:Y`, so each entry must be a value
of the node's `GPU_BRD` feature — which is **not** always the gres name:

| node | gres name | `GPU_BRD` | memory |
|---|---|---|---|
| `ixb*` | `gpu:B200:8` / `gpu:B300:8` | `B200` / `B300` | 180 GB |
| `ixh` | `gpu:H100:8` | `H100` | 80 GB |
| `ana` | `gpu:A100_80GB:8` | **`A100`** | 80 GB |
| `axa` | `gpu:A100:8` | `A100` | 40 GB |

Writing `"A100_80GB"` produces `GPU_BRD:A100_80GB`, which matches no node: the job
is accepted and then **pends forever with no error**. Check a candidate with
`scontrol show node <n> | grep ActiveFeatures` before adding it. Note `A100` admits
both 80 GB and 40 GB cards — constrain on memory separately if a run needs 80.

Check what a config will actually request without submitting:

```python
from sweep.expand import load_config
from sweep.execute import _gpu_args, _gpus_per_config
sb = load_config("<config>.jsonc")["execution"]["sbatch"]
print(_gpu_args(sb["gpus"], _gpus_per_config(sb)))
```

`gpus_per_config` (default 1) is the **single source of truth** for how many GPUs
one run gets. It drives both the `--gres` count and the launcher: at 1 the job
script calls `python -m <module>`, above 1 it calls
`torchrun --standalone --nproc_per_node N`, which is what makes an HF `Trainer`
initialise a process group and run DDP.

Because one number drives both, naming a count in `gpus` as well is a config
**error** rather than a silent divergence — a gres of 2 with `--nproc_per_node 1`
wastes a card, and the reverse hangs in NCCL waiting on a rank that has no GPU.
Write the type in `gpus` and the count in `gpus_per_config`:

```jsonc
"gpus": ["H100"],          // types only
"gpus_per_config": 2       // GPUs == DDP ranks
```

`"H100:1"` with the default `gpus_per_config: 1` still passes, so existing configs
are unaffected and render byte-identical sbatch args.

**Two things DDP does not do for you.** The experiment must have a
distribution-aware train sampler: a custom sampler returned unconditionally from
`_get_train_sampler` bypasses HF's `DistributedSampler` wrapping, so every rank
iterates identical indices and computes an identical gradient — no error, just N
GPUs doing one GPU's work. And `accumulation_steps` is **per-device**, so at
`gpus_per_config: 2` the effective batch doubles unless you halve it. Neither is
rewritten for you.

DDP reduces the latency of one run; it does not increase throughput. If the sweep
has independent runs to spend GPUs on, `granularity: per_config` is the better
lever.

### Sharing the inductor compile cache

By default each job gets its own `TORCHINDUCTOR_CACHE_DIR` so parallel jobs cannot
thrash a shared one. When every run of a sweep compiles the *same* shapes, that
default makes each job re-pay identical work: measured on the context sweep, 13
distinct cell shapes cost ~86 min of Triton codegen and PTX compilation before step
1, of which only ~2 min is GPU autotune benchmarking — the rest is CPU-bound and
byte-identical across runs.

`inductor_cache` names a directory to share instead. It is used **only if it
already exists**, which makes sharing opt-in and self-healing: a fresh clone, or a
cache that was never populated, silently falls back to the per-job path and just
compiles. Reproducibility never depends on a cache being present.
