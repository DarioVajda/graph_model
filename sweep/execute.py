"""
Dispatch: expand a sweep config and run each resolved config as its own process.

The runner is experiment-agnostic and the dependency points one way — runner ->
experiment. The experiment module is named on the command line; the runner never
imports it. Each resolved parameter dict is rendered to CLI flags and the
experiment is invoked as ``python -m <module> <flags> --runs-jsonl ... --run-name
... --sweep-id ...``. The experiment is a standalone argparse program that runs
exactly one config and logs its own record to ``--runs-jsonl``.

Flag-rendering conventions (the experiment's argparse must match):
    bool   -> ``--key`` / ``--no-key``   (argparse.BooleanOptionalAction)
    list   -> ``--key v1,v2`` (comma-joined)
    None   -> omitted
    scalar -> ``--key value``

Sweep directory layout (``<results_dir>/<sweep_name>/``)::

    config.json[c]       the original sweep config, verbatim (extension mirrors the source)
    resolved/<name>.json one fully-resolved parameter dict per run (provenance)
    logs/<name>.log      that run's stdout/stderr (local mode)
    jobs/<label>.sh      the command(s) a Slurm job runs (sbatch mode)
    runs.jsonl           one line per finished run (written by the runs themselves)
    report.md            written by report.py (auto after a local sweep)

Local mode runs the resolved configs sequentially in this process. sbatch mode
submits Slurm jobs whose container `srun` step runs ``slurm_launch.sh`` on a
generated job script, either as one sequential job (``granularity: single``) or
one job per config (``granularity: per_config``).

For ``granularity: per_config`` an optional ``execution.sbatch.max_concurrent``
caps how many configs run at once: instead of one independent job per config, the
configs are submitted as a single Slurm job array (``--array=0-(K-1)%N``) so Slurm
keeps at most N tasks running and backfills each freed slot with the next pending
config until all are done. The index->config assignment is shuffled (seeded on the
sweep name) so a demand-sorted config list can't cluster all the heavy runs into
one wave; the mapping is recorded in ``array_map.tsv``. Leave it unset for the
default behavior (one independent job per config, all queued at once).
"""

import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time

from . import expand as expand_mod
from . import report as report_mod

# This file lives at <repo>/sweep/execute.py.
_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_SWEEP_DIR, ".."))
LAUNCHER = os.path.join(_SWEEP_DIR, "slurm_launch.sh")
DEFAULT_RESULTS_ROOT = os.path.join(REPO_ROOT, "sweeps")

# Fallbacks for the sbatch resource block; every one is overridable from config.
_SBATCH_DEFAULTS = {
    "granularity": "per_config",   # "single" | "per_config"
    "cpus": 16,
    "mem": "64G",
    "time": "24:00:00",
    "mounts": "/shared:/shared",
}


# ── module + flag rendering ──────────────────────────────────────────────────
def normalize_module(experiment):
    """Accept either a dotted module (``a.b.c``) or a path (``a/b/c``)."""
    mod = experiment.strip().strip("/")
    if mod.endswith(".py"):
        mod = mod[:-3]
    return mod.replace("/", ".")


def render_flags(params):
    """Render a resolved parameter dict to argparse flags (see module docstring).

    dict values render as ``k=v`` pairs comma-joined (``--max-nodes
    webqsp=512,cwq=1024``) — the per-dataset knob form; the experiment's
    argparse type parses it back into a dict.
    """
    argv = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, dict):
            argv += [flag, ",".join(f"{k}={v}" for k, v in value.items())]
        elif isinstance(value, (list, tuple)):
            argv += [flag, ",".join(str(v) for v in value)]
        elif value is None:
            continue
        else:
            argv += [flag, str(value)]
    return argv


def _experiment_tag(experiment_module):
    """Short experiment name for Slurm job names (``src.experiments.kgqa`` -> ``kgqa``)."""
    parts = [p for p in experiment_module.split(".") if p and p != "__main__"]
    return parts[-1] if parts else experiment_module


def _run_argv(experiment_module, params, runs_jsonl, run_name, sweep_id):
    """Full ``-m <module> <flags> <bookkeeping>`` argv (without the python exe)."""
    return ["-m", experiment_module, *render_flags(params),
            "--runs-jsonl", runs_jsonl, "--run-name", run_name, "--sweep-id", sweep_id]


# ── run names + sweep dir ────────────────────────────────────────────────────
def _short(value):
    if isinstance(value, (list, tuple)):
        value = "-".join(str(v) for v in value)
    return str(value).replace("/", "-").replace(" ", "")[:24]


def _run_names(runs):
    """Readable, unique per-run names from the keys that vary across runs."""
    serial = [{k: json.dumps(v, sort_keys=True) for k, v in r.items()} for r in runs]
    keys = list(runs[0]) if runs else []
    varying = [k for k in keys if len({s.get(k) for s in serial}) > 1]
    width = max(4, len(str(len(runs) - 1)) if runs else 1)
    names = []
    for i, run in enumerate(runs):
        tag = "_".join(f"{k}{_short(run[k])}" for k in varying if k in run)
        names.append(f"{i:0{width}d}" + (f"_{tag}" if tag else ""))
    return names


def _prepare_sweep_dir(config_path, results_root, meta, runs):
    """Create ``<results_root>/<sweep_name>/`` and return a ``paths`` dict."""
    sweep_name = meta.get("name") or time.strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(results_root, sweep_name)
    resolved_dir = os.path.join(sweep_dir, "resolved")
    logs_dir = os.path.join(sweep_dir, "logs")
    os.makedirs(resolved_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Preserve the submitted config verbatim, mirroring its extension so a JSONC
    # source (comments/trailing commas) is not saved under a misleading .json name.
    config_ext = os.path.splitext(config_path)[1] or ".json"
    with open(config_path) as src, open(os.path.join(sweep_dir, f"config{config_ext}"), "w") as dst:
        dst.write(src.read())

    names = _run_names(runs)
    for run, name in zip(runs, names):
        with open(os.path.join(resolved_dir, f"{name}.json"), "w") as f:
            json.dump(run, f, indent=2)   # provenance; the experiment is invoked via flags

    return {"sweep_name": sweep_name, "sweep_dir": sweep_dir, "logs_dir": logs_dir,
            "runs_jsonl": os.path.join(sweep_dir, "runs.jsonl"), "names": names}


# ── entry ────────────────────────────────────────────────────────────────────
def run_sweep(experiment, config_path, default_results_root=None, python_exe=None):
    """Expand ``config_path`` and dispatch every resolved run against ``experiment``."""
    python_exe = python_exe or sys.executable
    experiment_module = normalize_module(experiment)
    meta, runs = expand_mod.load_and_expand(config_path)
    if not runs:
        raise expand_mod.SweepError("Sweep expanded to zero runs.")

    results_root = meta.get("results_dir") or default_results_root or DEFAULT_RESULTS_ROOT
    paths = _prepare_sweep_dir(config_path, results_root, meta, runs)
    execution = meta.get("execution") or {"mode": "local"}
    mode = execution.get("mode", "local")

    print(f"[sweep] '{paths['sweep_name']}' -> {len(runs)} run(s), experiment={experiment_module}, mode={mode}")

    if mode == "local":
        _dispatch_local(experiment_module, runs, paths, python_exe)
    elif mode == "sbatch":
        _dispatch_sbatch(experiment_module, runs, paths, execution.get("sbatch") or {})
    else:
        raise expand_mod.SweepError(f"Unknown execution mode {mode!r} (expected 'local' or 'sbatch').")
    return paths["sweep_dir"]


def _dispatch_local(experiment_module, runs, paths, python_exe):
    failures = 0
    for run, name in zip(runs, paths["names"]):
        argv = [python_exe, *_run_argv(experiment_module, run, paths["runs_jsonl"], name, paths["sweep_name"])]
        log_path = os.path.join(paths["logs_dir"], f"{name}.log")
        print(f"\n$ {' '.join(shlex.quote(a) for a in argv)}")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = proc.wait()
        print(f"[sweep] run {name}: {'ok' if rc == 0 else f'FAILED (rc={rc})'}")
        failures += rc != 0
    report_mod.write_report(paths["sweep_dir"])
    print(f"[sweep] local done: {len(runs) - failures}/{len(runs)} ok -> {paths['sweep_dir']}")


# ── sbatch ───────────────────────────────────────────────────────────────────
def _write_job_script(jobs_dir, label, argv_lines):
    """Write a job script that runs each invocation with the venv `python`.

    The launcher puts the project .venv first on PATH, so a bare `python` here is
    the venv interpreter (avoids the `source activate` + `set -u` pitfall).
    """
    path = os.path.join(jobs_dir, f"{label}.sh")
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -uo pipefail\n")
        for argv in argv_lines:
            f.write("python " + " ".join(shlex.quote(a) for a in argv) + "\n")
    return path


def _launch_env():
    """Submit-time context forwarded to ``slurm_launch.sh``.

    A job must run against the project it was *submitted from* — that project's
    working dir, its venv, and its login script — not against wherever the
    installed ``sweep`` package physically lives (which, once ``sweep`` is
    pip-installed into another repo, is a different tree entirely). We capture
    that context here and forward it through the same ``env`` list that already
    carries ``HOME``; the launcher prefers these and falls back to its own
    location only when they are absent.

    Submitting from this repo's root with its ``.venv`` active — the historical
    invocation — makes these resolve to exactly the values the launcher used to
    compute for itself, so submissions here are byte-for-byte unchanged.
    """
    root = os.getcwd()
    venv_bin = os.path.dirname(os.path.abspath(sys.executable))
    env = [f"SWEEP_PROJECT_ROOT={root}", f"SWEEP_VENV_BIN={venv_bin}"]
    login = os.path.join(root, "login.sh")
    if os.path.isfile(login):
        env.append(f"SWEEP_LOGIN={login}")
    return env


def _srun_wrap(label, job_script, sb):
    """``srun --container-image=... slurm_launch.sh <label> <job_script>`` wrap.

    Bare `srun` is blocked on the cluster, so this is an `srun` step *inside* the
    sbatch allocation; the container gives the py3.10 base the .venv needs. HOME is
    forwarded so HF/wandb creds + the model cache resolve inside it.
    """
    home = os.environ.get("HOME", "")
    inner = ["bash", LAUNCHER, label, job_script]
    srun = ["srun"]
    if sb.get("container"):
        srun += [f"--container-image={sb['container']}",
                 f"--container-mounts={sb.get('mounts', _SBATCH_DEFAULTS['mounts'])}"]
    srun += ["env", f"HOME={home}", "PYTHONUNBUFFERED=1", *_launch_env(), *inner]
    return " ".join(shlex.quote(a) for a in srun)


def _array_wrap(labels, scripts, sb):
    """``--wrap`` body for a job array: each task runs its own job script.

    The task's index into ``labels``/``scripts`` is ``$SLURM_ARRAY_TASK_ID``. The
    two indexed args are emitted raw (not shlex-quoted) so the shell expands them;
    everything else is quoted normally.
    """
    home = os.environ.get("HOME", "")
    srun = ["srun"]
    if sb.get("container"):
        srun += [f"--container-image={sb['container']}",
                 f"--container-mounts={sb.get('mounts', _SBATCH_DEFAULTS['mounts'])}"]
    srun += ["env", f"HOME={home}", "PYTHONUNBUFFERED=1", *_launch_env(), "bash", LAUNCHER]
    srun_str = " ".join(shlex.quote(a) for a in srun)
    labels_arr = " ".join(shlex.quote(x) for x in labels)
    scripts_arr = " ".join(shlex.quote(x) for x in scripts)
    body = (f'LABELS=({labels_arr}); SCRIPTS=({scripts_arr}); i="$SLURM_ARRAY_TASK_ID"; '
            f'{srun_str} "${{LABELS[$i]}}" "${{SCRIPTS[$i]}}"')
    # sbatch --wrap scripts run under /bin/sh, which has no arrays: force bash.
    return "exec bash -c " + shlex.quote(body)


def _gpu_args(gpus):
    """Render ``gpus`` to sbatch args.

    A string (``"B200:1"``) pins the type inside the gres request. A list
    (``["B200:1", "B300:1"]``) means "any of these types": Slurm gres can't
    express an OR, so it becomes a generic count plus a feature constraint —
    ``--gres gpu:1 --constraint GPU_BRD:B200|GPU_BRD:B300`` (each node carries a
    ``GPU_BRD:<type>`` feature). The per-type counts must agree.
    """
    if not isinstance(gpus, (list, tuple)):
        return ["--gres", f"gpu:{gpus}"]
    types, counts = [], set()
    for entry in gpus:
        gpu_type, _, count = str(entry).partition(":")
        types.append(gpu_type)
        counts.add(count or "1")
    if len(counts) != 1:
        raise expand_mod.SweepError(
            f"execution.sbatch.gpus list must use one count across all types, got {gpus!r}.")
    return ["--gres", f"gpu:{counts.pop()}",
            "--constraint", "|".join(f"GPU_BRD:{t}" for t in types)]


def _sbatch_argv(jobname, logpath, wrap, sb, array=None):
    """Assemble the ``sbatch`` argv from the resource block (missing -> defaults)."""
    if not sb.get("partition"):
        raise expand_mod.SweepError("execution.sbatch.partition is required for sbatch mode.")
    if not sb.get("gpus"):
        raise expand_mod.SweepError("execution.sbatch.gpus is required for sbatch mode (e.g. 'B200:1').")
    argv = ["sbatch", "-p", str(sb["partition"]), *_gpu_args(sb["gpus"]),
            "-c", str(sb.get("cpus", _SBATCH_DEFAULTS["cpus"])),
            "--mem", str(sb.get("mem", _SBATCH_DEFAULTS["mem"])),
            "-t", str(sb.get("time", _SBATCH_DEFAULTS["time"])),
            "-J", jobname, "-o", logpath]
    if array:
        argv += ["--array", array]
    if sb.get("account"):
        argv += ["-A", str(sb["account"])]
    if sb.get("nodelist"):
        argv += ["-w", str(sb["nodelist"])]
    argv += ["--wrap", wrap]
    return argv


def _dispatch_sbatch(experiment_module, runs, paths, sb):
    """Submit the sweep as Slurm jobs (one sequential job, or one job per config)."""
    granularity = sb.get("granularity", _SBATCH_DEFAULTS["granularity"])
    sweep_name = paths["sweep_name"]
    # Slurm job names lead with the experiment so squeue groups them by experiment.
    job_prefix = _experiment_tag(experiment_module)
    jobs_dir = os.path.join(paths["sweep_dir"], "jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    def argv_for(run, name):
        return _run_argv(experiment_module, run, paths["runs_jsonl"], name, sweep_name)

    jobs = []   # (jobname, sbatch_argv)
    if granularity == "single":
        lines = [argv_for(run, name) for run, name in zip(runs, paths["names"])]
        job_script = _write_job_script(jobs_dir, sweep_name, lines)
        wrap = _srun_wrap(sweep_name, job_script, sb)
        logpath = os.path.join(paths["logs_dir"], f"{sweep_name}.slurm.out")
        jobs.append((sweep_name, _sbatch_argv(f"{job_prefix}_{sweep_name}", logpath, wrap, sb)))
    elif granularity == "per_config":
        # One job script per config either way; how they're submitted depends on
        # whether a concurrency cap is set.
        entries = []   # (name, label, job_script)
        for run, name in zip(runs, paths["names"]):
            label = f"{sweep_name}_{name}"
            job_script = _write_job_script(jobs_dir, label, [argv_for(run, name)])
            entries.append((name, label, job_script))

        max_concurrent = sb.get("max_concurrent")
        if max_concurrent:
            # Single throttled job array: Slurm keeps <=N tasks running and fills a
            # freed slot with the next pending index until all K are done. Shuffle
            # index->config (seeded on sweep name) so a demand-sorted config list
            # can't cluster the heavy runs into one wave; record the mapping.
            order = list(range(len(entries)))
            random.Random(sweep_name).shuffle(order)
            labels = [entries[j][1] for j in order]
            scripts = [entries[j][2] for j in order]
            with open(os.path.join(paths["sweep_dir"], "array_map.tsv"), "w") as f:
                f.write("task\tname\n")
                for task, j in enumerate(order):
                    f.write(f"{task}\t{entries[j][0]}\n")
            wrap = _array_wrap(labels, scripts, sb)
            array_spec = f"0-{len(entries) - 1}%{int(max_concurrent)}"
            logpath = os.path.join(paths["logs_dir"], f"{sweep_name}_%A_%a.slurm.out")
            jobs.append((sweep_name, _sbatch_argv(f"{job_prefix}_{sweep_name}", logpath, wrap, sb,
                                                  array=array_spec)))
        else:
            for name, label, job_script in entries:
                wrap = _srun_wrap(label, job_script, sb)
                logpath = os.path.join(paths["logs_dir"], f"{name}.slurm.out")
                jobs.append((name, _sbatch_argv(f"{job_prefix}_{label}", logpath, wrap, sb)))
    else:
        raise expand_mod.SweepError(
            f"Unknown sbatch granularity {granularity!r} (expected 'single' or 'per_config').")

    # Always record the exact commands for reproducibility / manual resubmission.
    cmds_path = os.path.join(paths["sweep_dir"], "sbatch_commands.sh")
    with open(cmds_path, "w") as f:
        f.write("#!/usr/bin/env bash\n# sbatch commands for this sweep (auto-generated).\n")
        for _, argv in jobs:
            f.write(" ".join(shlex.quote(a) for a in argv) + "\n")
    print(f"[sweep] wrote {len(jobs)} sbatch command(s) to {cmds_path}")

    if bool(sb.get("dry_run")) or shutil.which("sbatch") is None:
        why = "dry_run set" if sb.get("dry_run") else "sbatch not found on this host"
        print(f"[sweep] NOT submitting ({why}). Submit later with: bash {cmds_path}")
        return

    submitted = []
    for name, argv in jobs:
        out = subprocess.run(argv, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"[sweep] sbatch FAILED for {name}: {out.stderr.strip()}")
            continue
        job_id = out.stdout.strip().split()[-1]   # "Submitted batch job 12345"
        submitted.append((job_id, name))
        print(f"[sweep] submitted {name} -> job {job_id}")
    with open(os.path.join(paths["sweep_dir"], "jobs.txt"), "w") as f:
        for job_id, name in submitted:
            f.write(f"{job_id}\t{name}\n")
    print(f"[sweep] {len(submitted)}/{len(jobs)} job(s) submitted. When they finish, aggregate with:\n"
          f"    python3 -m sweep.report {paths['sweep_dir']}")
