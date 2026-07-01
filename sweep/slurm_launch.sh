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

# Repo root = two levels up from this script (sweep/..).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${HOME:?HOME must be set (HF/wandb creds + model cache live under it)}"

# Put the venv FIRST on PATH so a bare `python` in the job script is the venv
# interpreter — `source .venv/bin/activate` trips under `set -u`, and if that is
# swallowed the job silently runs on the system python and dies on imports.
[ -x "$REPO/.venv/bin/python" ] || { echo "FATAL: venv python not found at $REPO/.venv/bin/python" >&2; exit 1; }
export PATH="$REPO/.venv/bin:$PATH"

bash login.sh >/dev/null 2>&1 && echo "[launch] hf+wandb login ok" || echo "[launch] WARN login.sh failed" >&2

# Per-job compile cache so parallel jobs don't thrash a shared inductor cache.
export TORCHINDUCTOR_CACHE_DIR="$REPO/.inductor_cache/$LABEL"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

echo "[launch] $(date '+%F %T') label=$LABEL host=$(hostname) script=$JOB_SCRIPT"
bash "$JOB_SCRIPT"
rc=$?
echo "[launch] $(date '+%F %T') done (rc=$rc)"
exit $rc
