#!/usr/bin/env bash
# =============================================================================
# sbatch_diagnostic.sh — FACTORIZED_RWPE_BIAS.md Phase 0, on ONE CPU node.
# =============================================================================
# Runs `feature_diagnostic.py` over the WebQSP cache `021`/`023` trained on and
# three GraphQA tasks. No GPU: the work is fp64 matrix powers on graphs of ~50-512
# nodes, which is CPU-bound and finishes in minutes, and a CPU-only job does not
# queue behind the multi-day training jobs that own the GPUs.
#
# It reads 111 MB+ pickles and holds every node's feature matrix in RAM, so it is
# NOT a login-node job.
#
#   ./src/experiments/bias_experiments/factorized_rwpe/sbatch_diagnostic.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

SCRIPT="$REPO/job_logs/rwpe_diagnostic_${STAMP}.sh"
LOG="$REPO/job_logs/rwpe_diagnostic_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "python -m src.experiments.bias_experiments.factorized_rwpe.feature_diagnostic ${DIAG_ARGS:-}"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh rwpe_diagnostic_${STAMP} $SCRIPT"

# 32 cores because the fp64 `bmm` chain is the whole runtime and it threads well;
# 96G because the pooled node x 72 feature matrix for WebQSP train sits in RAM
# alongside the unpickled graphs. 2 h is generous for what is expected to be ~10 min.
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 32 --mem 96G -t "02:00:00" \
      -J "rwpe_diagnostic" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

# FRIDA prints backfill chatter on stdout, so `--parsable | strip` is NOT the job
# id and a later --dependency or scancel would silently target nothing.
[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted rwpe diagnostic -> job $JOB"
echo "  log: ${LOG/\%j/$JOB}"
