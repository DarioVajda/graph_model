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

# Per-job compile cache so parallel jobs don't thrash a shared inductor cache.
export TORCHINDUCTOR_CACHE_DIR="$REPO/.inductor_cache/$LABEL"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo "[launch] $(date '+%F %T') label=$LABEL host=$(hostname) script=$JOB_SCRIPT"
bash "$JOB_SCRIPT"
rc=$?
echo "[launch] $(date '+%F %T') done (rc=$rc)"
exit $rc
