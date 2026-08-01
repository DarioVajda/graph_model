#!/usr/bin/env bash
# =============================================================================
# sbatch_chain_hard8k.sh — the HARD cell at the data budget that actually worked.
# =============================================================================
# This closes a 2x2 that was otherwise missing a corner. The two axes that have
# been moving together are CELL SIZE and DATA/BUDGET:
#
#                        n_train=2000, 500 steps     n_train=8000, 2000 steps
#   N=16/32, T=64  (easy)          -                  control 011/012  -> graph WINS
#   N=64,    T=128 (hard)   Phase A 009/010           <- THIS BUILD
#                           (both arms floor)
#
# Phase A floored at every k>=2 for BOTH arms, but it only ever ran 500 steps,
# and on the EASY cell the flat arm did not cross EM 0.9 until step ~1500. Phase
# A was under-budget by 3x; it is not evidence about the cell. 013/014 raise the
# hard cell to 1250 steps while holding n_train=2000, which tests budget but
# leaves the model re-visiting only 2000 unique graphs -- at 5 epochs that starts
# to reward memorisation over learning the traversal.
#
# This build instead matches the control's DATA (8000 unique graphs) at the hard
# cell, so the only variable left between it and 011/012 is the cell itself.
# That is the run that answers "does the graph arm's k=4 advantage survive 62
# distractors and an 8k-token context?"
#
# MEMORY: request 384G, not 64/96G. Measured MaxRSS on the SAME cell at n_train=2000
# was 61.7 GB against a 64 G request — i.e. Phase A's builds only just fit. At 8000
# graphs a 96 G request OOM-killed all four hop counts (119549-119552, ~18-27 min in,
# empty output dirs). The N=64 cell is what drives this, not n_train alone: the
# small-cell 8000-graph builds peaked at only ~38 GB. frida nodes carry 515 GB-2 TB,
# so the headroom is free.
#
# Cost: ~1.3 GB and ~40 min of CPU per hop count. GPU cost is the real constraint
# downstream -- at 11.4 s/it the graph arm gets ~1890 steps inside the 6 h wall,
# so plan ~1.5 epochs (1500 steps), which is where the control's graph arm had
# already solved k=2 (step ~500) and k=4 (step ~750).
#
# Separate keyspace from both Phase A (`tr2000`) and the control (`n16-32_t64`),
# so nothing here can orphan an existing cache.
#
# CPU-only: the build is HF `datasets.map` plumbing, not the eigendecomposition.
# Do NOT use the `amd` partition (MI210/ROCm cannot start this CUDA container) --
# sbatch prints a hint suggesting it for CPU-only jobs; ignore it.
#
# Idempotent: a split already on disk is skipped, so re-running resumes.
#
#   ./src/experiments/context/sbatch_chain_hard8k.sh [PARTITION]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${1:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

# Phase A's cell (node/token counts) at the control's n_train. These knobs form
# the cache key and must match 015/016.
HOPS=(1 2 3 4)
COMMON="--mode data_prep --node-counts 64 --token-counts 128 --max-train-len 16384 \
--n-train 8000 --n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --spd --max-spd 8 --no-rrwp --magnetic \
--magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25"

STAMP="$(date +%Y%m%d_%H%M%S)"
for k in "${HOPS[@]}"; do
  SCRIPT="$REPO/job_logs/ctx_hard8k_h${k}_${STAMP}.sh"
  LOG="$REPO/job_logs/ctx_hard8k_h${k}_${STAMP}_%j.out"
  {
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "python -m src.experiments.context $COMMON --hops $k"
  } > "$SCRIPT"
  chmod +x "$SCRIPT"

  WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_hard8k_h${k}_${STAMP} $SCRIPT"

  echo "[submit] hops=$k -> $LOG"
  sbatch -p "$PARTITION" -A povejmo -c 16 --mem 384G -t 06:00:00 \
         -J "ctx_hard8k_h${k}" -o "$LOG" --wrap "$WRAP"
done
