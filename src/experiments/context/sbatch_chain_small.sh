#!/usr/bin/env bash
# =============================================================================
# sbatch_chain_small.sh — the SMALL-CELL LEARNABILITY CONTROL for the k-hop task.
# =============================================================================
# Phase A (008/009/010) asks "at which k does the chain break?" That question is
# only meaningful if the model can learn the chain AT ALL. At epoch 1.0 of the
# Phase A graph arm every k sits at eval_em 0.01-0.05 — which is precisely the
# rate you get by copying a RANDOM content node's code (1/62 = 0.016). Format
# learned, traversal not.
#
# Two things could produce that, and they call for opposite responses:
#   (a) the task is too hard for the BUDGET (2 epochs x 2000 graphs), or
#   (b) the task is too hard for the MODEL/representation, at any budget.
#
# This build isolates (b) by making the task as easy as it can possibly be while
# staying a genuine pointer chase:
#   * N = 16 and 32 (14 / 30 distractors instead of 62),
#   * T = 64  -> L ~ 1.1k / 2.1k instead of 8k, so a step is ~7x cheaper,
#   * n_train 8000 instead of 2000, and the training configs run many epochs.
#
# If k=1 still floors HERE, hop count is not the axis worth sweeping and Phase A
# needs a redesign rather than more k values.
#
# Deliberately NOT a variant of sbatch_chain_data.sh: that script's flags are
# pinned to 008_chain_data.jsonc and a mismatch there would orphan Phase A's
# caches. This is a separate keyspace (`_h{k}` + its own grid/n_train tags).
#
# CPU-only for the same measured reason as sbatch_chain_data.sh: the build is
# HF `datasets.map` plumbing, not the eigendecomposition. Do NOT use the `amd`
# partition (MI210/ROCm cannot start this CUDA container).
#
# Idempotent: a split already on disk is skipped, so re-running resumes.
#
#   ./src/experiments/context/sbatch_chain_small.sh [PARTITION]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${1:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

# Must match 011/012 — these knobs form the cache key.
HOPS=(1 2 3 4)
COMMON="--mode data_prep --node-counts 16,32 --token-counts 64 --max-train-len 16384 \
--n-train 8000 --n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --spd --max-spd 8 --no-rrwp --magnetic \
--magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25"

STAMP="$(date +%Y%m%d_%H%M%S)"
for k in "${HOPS[@]}"; do
  SCRIPT="$REPO/job_logs/ctx_small_h${k}_${STAMP}.sh"
  LOG="$REPO/job_logs/ctx_small_h${k}_${STAMP}_%j.out"
  {
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "python -m src.experiments.context $COMMON --hops $k"
  } > "$SCRIPT"
  chmod +x "$SCRIPT"

  WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_small_h${k}_${STAMP} $SCRIPT"

  echo "[submit] hops=$k -> $LOG"
  sbatch -p "$PARTITION" -A povejmo -c 16 --mem 64G -t 04:00:00 \
         -J "ctx_small_h${k}" -o "$LOG" --wrap "$WRAP"
done
