#!/usr/bin/env bash
# =============================================================================
# slurm_launch.sh — thin per-job entry, run INSIDE the pyxis container.
# =============================================================================
# The sweep runner (execute.py) submits jobs whose `srun --container-image=...`
# step invokes this script inside the project container (a py3.10 base, so the
# project .venv resolves — the bare B200 host is py3.12 and would break imports).
# All orchestration (sweep expansion, which configs run where) is done in Python;
# this only sets up the environment and runs a generated job script.
#
#   slurm_launch.sh <label> <job_script.sh>
#     <label>        tag for this job's torch-inductor compile cache
#     <job_script>   a generated script of `python -m <experiment> ...` lines
#
# HOME must be exported by the caller (execute.py forwards the submit-time HOME via
# the srun env) so HF / wandb credentials + the cached model are found.
# =============================================================================
set -uo pipefail

LABEL="$1"; shift
JOB_SCRIPT="$1"; shift

# Project root: prefer the submit-time working dir forwarded by execute.py
# (SWEEP_PROJECT_ROOT) so a job runs against the project it was submitted FROM —
# which, once `sweep` is pip-installed into another repo, is NOT this script's
# tree. Fall back to two levels up (sweep/..) for manual/legacy invocation.
REPO="${SWEEP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

: "${HOME:?HOME must be set (HF/wandb creds + model cache live under it)}"

# Put the venv FIRST on PATH so a bare `python` in the job script is the venv
# interpreter — `source .venv/bin/activate` trips under `set -u`, and if that is
# swallowed the job silently runs on the system python and dies on imports.
# The venv comes from the submitting interpreter (SWEEP_VENV_BIN); fall back to
# the project's own .venv/bin.
VENV_BIN="${SWEEP_VENV_BIN:-$REPO/.venv/bin}"
[ -x "$VENV_BIN/python" ] || { echo "FATAL: venv python not found at $VENV_BIN/python" >&2; exit 1; }
export PATH="$VENV_BIN:$PATH"

# Login script: submit-time SWEEP_LOGIN, else the project's own login.sh. Skip
# quietly if none exists (a consuming project need not have one).
LOGIN="${SWEEP_LOGIN:-$REPO/login.sh}"
if [ -f "$LOGIN" ]; then
    bash "$LOGIN" >/dev/null 2>&1 && echo "[launch] hf+wandb login ok" || echo "[launch] WARN login failed" >&2
else
    echo "[launch] no login script at $LOGIN; skipping" >&2
fi

# Inductor compile cache. Default is per-job so parallel jobs don't thrash a shared
# one. But when every run of a sweep compiles the SAME shapes, that default makes each
# job re-pay a cost that is pure duplication: measured on the context sweep, 13 distinct
# cell shapes take ~86 min of Triton codegen + PTX compilation before step 1, and only
# ~2 min of that is the GPU autotune benchmarking -- the rest is CPU-bound work whose
# result is byte-identical across runs.
#
# So SWEEP_INDUCTOR_CACHE (set from execution.sbatch.inductor_cache) names a cache to
# SHARE. It is used only if the directory already exists, which makes the sharing
# opt-in AND self-healing: a fresh clone, or a config whose cache was never populated,
# silently falls back to the per-job path and just compiles. Reproducibility never
# depends on a cache being present.
if [ -n "${SWEEP_INDUCTOR_CACHE:-}" ] && [ -d "${SWEEP_INDUCTOR_CACHE}" ]; then
    export TORCHINDUCTOR_CACHE_DIR="${SWEEP_INDUCTOR_CACHE}"
    echo "[launch] inductor cache: SHARED $TORCHINDUCTOR_CACHE_DIR"
else
    export TORCHINDUCTOR_CACHE_DIR="$REPO/.inductor_cache/$LABEL"
    if [ -n "${SWEEP_INDUCTOR_CACHE:-}" ]; then
        echo "[launch] inductor cache: ${SWEEP_INDUCTOR_CACHE} absent -> per-job $TORCHINDUCTOR_CACHE_DIR" >&2
    else
        echo "[launch] inductor cache: per-job $TORCHINDUCTOR_CACHE_DIR"
    fi
fi
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo "[launch] $(date '+%F %T') label=$LABEL host=$(hostname) script=$JOB_SCRIPT"
bash "$JOB_SCRIPT"
rc=$?
echo "[launch] $(date '+%F %T') done (rc=$rc)"
exit $rc
