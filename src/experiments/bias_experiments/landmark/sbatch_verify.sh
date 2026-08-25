#!/usr/bin/env bash
# Dry-run every landmark sweep run through the real parser + validator.
# Imports torch via the experiment module, so it is a compute-node job.
#   ./src/experiments/bias_experiments/landmark/sbatch_preflight.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"; cd "$REPO"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$REPO/job_logs"
SCRIPT="$REPO/job_logs/landmark_verify_${STAMP}.sh"
LOG="$REPO/job_logs/landmark_verify_${STAMP}_%j.out"
{ echo "#!/usr/bin/env bash"; echo "set -x"
  echo "python -m src.experiments.bias_experiments.landmark.verify_live ${VERIFY_ARGS:-}"
} > "$SCRIPT"; chmod +x "$SCRIPT"
WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh landmark_verify_${STAMP} $SCRIPT"
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 8 --mem 32G -t "00:30:00" \
      -J "landmark_verify" -o "$LOG" --wrap "$WRAP" | grep -oE '^[0-9]+$' | head -1)
[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }
echo "submitted preflight -> job $JOB"; echo "  log: ${LOG/\%j/$JOB}"
