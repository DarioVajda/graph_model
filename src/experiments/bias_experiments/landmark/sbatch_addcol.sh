#!/usr/bin/env bash
# Add the `landmark` column to existing .gtds caches, on a compute node.
# Unpickles and rewrites multi-GB Arrow tables — never a login-node job.
#   ADDCOL_ARGS="--roots <dir> --only dev" ./…/sbatch_addcol.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"; cd "$REPO"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"; mkdir -p "$REPO/job_logs"
SCRIPT="$REPO/job_logs/landmark_addcol_${STAMP}.sh"
LOG="$REPO/job_logs/landmark_addcol_${STAMP}_%j.out"
{ echo "#!/usr/bin/env bash"; echo "set -x"
  echo "python -m src.experiments.bias_experiments.landmark.add_landmark_column ${ADDCOL_ARGS:-}"
} > "$SCRIPT"; chmod +x "$SCRIPT"
WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh landmark_addcol_${STAMP} $SCRIPT"
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem 192G -t "04:00:00" \
      -J "landmark_addcol" -o "$LOG" --wrap "$WRAP" | grep -oE '^[0-9]+$' | head -1)
[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }
echo "submitted addcol -> job $JOB"; echo "  log: ${LOG/\%j/$JOB}"
