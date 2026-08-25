#!/usr/bin/env bash
# Read 042's per-head gain out of live checkpoints: did the gate open, and how
# far? Imports torch, so it is a compute-node job (no CPU-heavy work, tiny slot).
#   DIAGNORM_ARGS="--prefix 042_webqsp_landmark_norm" ./…/sbatch_diagnorm.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"; cd "$REPO"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$REPO/job_logs"
SCRIPT="$REPO/job_logs/landmark_diagnorm_${STAMP}.sh"
LOG="$REPO/job_logs/landmark_diagnorm_${STAMP}_%j.out"
{ echo "#!/usr/bin/env bash"; echo "set -x"
  echo "python -m src.experiments.bias_experiments.landmark.diagnose_norm ${DIAGNORM_ARGS:-}"
} > "$SCRIPT"; chmod +x "$SCRIPT"
WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh landmark_diagnorm_${STAMP} $SCRIPT"
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 4 --mem 16G -t "00:20:00" \
      -J "landmark_diagnorm" -o "$LOG" --wrap "$WRAP" | grep -oE '^[0-9]+$' | head -1)
[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }
echo "submitted diagnose_norm -> job $JOB"; echo "  log: ${LOG/\%j/$JOB}"
