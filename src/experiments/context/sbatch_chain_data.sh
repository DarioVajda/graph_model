#!/usr/bin/env bash
# =============================================================================
# sbatch_chain_data.sh — build the Phase A k-hop datasets, one job per hop count.
# =============================================================================
# Six independent .gtds trees (keyed by `_h{hops}`), built in PARALLEL on compute
# nodes. Sequentially this is ~3 h; in parallel it is one build's wall time.
#
# **No GPU is requested, on purpose** — and unlike the note in
# sbatch_data_prep.sh, the reason is now measured. The build is dominated by HF
# `datasets.map` overhead at batch_size=1 (~2.0 s/graph at N=128), NOT by the
# spectral decomposition: the magnetic eigendecomposition itself is 2.8 ms/graph
# single-threaded at N=128 (17.9 ms with 32 threads — thread contention on a tiny
# matrix). A GPU cannot touch CPU-side plumbing, so attaching one would only add
# queue time.
#
# NB: sbatch prints a hint suggesting the `amd` partition for GPU-less jobs — DO
# NOT follow it. `amd` is the MI210/ROCm node and cannot start this CUDA
# container ("pyxis: container start failed", job 119348, dead in 27 s).
#
# Idempotent: a split already on disk is skipped, so re-running resumes.
#
#   ./src/experiments/context/sbatch_chain_data.sh [PARTITION]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${1:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

# Must match 008_chain_data.jsonc — these are the same knobs, and a mismatch
# builds a tree the training configs will not find.
HOPS=(1 2 3 4 6 8)
COMMON="--mode data_prep --node-counts 64 --token-counts 128 --max-train-len 16384 \
--n-train 2000 --n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --spd --max-spd 8 --no-rrwp --magnetic \
--magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25"

STAMP="$(date +%Y%m%d_%H%M%S)"
for k in "${HOPS[@]}"; do
  SCRIPT="$REPO/job_logs/ctx_chain_h${k}_${STAMP}.sh"
  LOG="$REPO/job_logs/ctx_chain_h${k}_${STAMP}_%j.out"
  {
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "python -m src.experiments.context $COMMON --hops $k"
  } > "$SCRIPT"
  chmod +x "$SCRIPT"

  WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_chain_h${k}_${STAMP} $SCRIPT"

  echo "[submit] hops=$k -> $LOG"
  sbatch -p "$PARTITION" -A povejmo -c 16 --mem 64G -t 04:00:00 \
         -J "ctx_chain_h${k}" -o "$LOG" --wrap "$WRAP"
done
