#!/usr/bin/env bash
# =============================================================================
# sbatch_diagnostic.sh — LANDMARK_BIAS.md Phase 0, on ONE CPU node.
# =============================================================================
# Runs `anchor_diagnostic.py` over two WebQSP caches (base + isolated question
# node) and three GraphQA tasks. No GPU: the work is per-graph APSP, betweenness
# and a min-reduction over anchors, all CPU-bound, and a CPU-only job does not
# queue behind the multi-day training jobs that own the GPUs.
#
# It unpickles 100 MB+ graph files and holds an (N,N) APSP per graph, so it is
# NOT a login-node job.
#
#   ./src/experiments/bias_experiments/landmark/sbatch_diagnostic.sh
#   DIAG_ARGS="--only webqsp --max-graphs 500" ./…/sbatch_diagnostic.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

SCRIPT="$REPO/job_logs/landmark_gold_${STAMP}.sh"
LOG="$REPO/job_logs/landmark_gold_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "python -m src.experiments.bias_experiments.landmark.gold_coverage ${DIAG_ARGS:-}"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh landmark_gold_${STAMP} $SCRIPT"

# 16 cores: the hot loops are scipy APSP and networkx betweenness, both largely
# single-threaded per graph, so cores buy little beyond BLAS on the oracle
# reduction. 64G because only one graph's (N,N) matrices are live at a time —
# the accumulators are histograms, not stored features. 4 h is generous:
# betweenness is O(NM) per graph and 6 rules x 4 k share one APSP.
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem 64G -t "04:00:00" \
      -J "landmark_gold" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

# FRIDA prints backfill chatter on stdout, so `--parsable | strip` is NOT the job
# id and a later --dependency or scancel would silently target nothing.
[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted landmark diagnostic -> job $JOB"
echo "  log: ${LOG/\%j/$JOB}"
