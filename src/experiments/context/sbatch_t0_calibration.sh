#!/usr/bin/env bash
# =============================================================================
# sbatch_t0_calibration.sh — T0 for the context experiment (README §A.12).
# =============================================================================
# Measures model fwd+bwd latency and peak memory for every (N, T) cell under the
# knobs this experiment trains with (flex, k_hop=0, bias + decoder
# checkpointing), on the GPU the training run will actually request.
#
# This exists because the plan's cost numbers are interpolated from
# results_h100/full_model.md, which predates both checkpointing modes AND was
# measured on different hardware. Nothing downstream (max_train_len, num_epochs,
# the sbatch time/mem in 003) should be trusted until this has run.
#
#   ./src/experiments/context/sbatch_t0_calibration.sh [GPU_TYPE]   # default B200
#
# Output: src/experiments/context/results/calibration/{calibration.jsonl,.md}
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

GPU_TYPE="${1:-B200}"
GPU_TAG="$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="src/experiments/context/results/calibration_${GPU_TAG}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs" "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT="$REPO/job_logs/ctx_t0_${STAMP}.sh"
LOG="$REPO/job_logs/ctx_t0_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  echo "python -m src.experiments.context.calibrate --out-dir $OUT_DIR --cells all --compile-mode default"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_t0_${STAMP} $SCRIPT"

echo "[submit] T0 calibration on $GPU_TYPE -> $LOG"
sbatch -p frida -A povejmo --gres="gpu:${GPU_TYPE}:1" -c 8 --mem=128G -t 08:00:00 \
       -J "ctx_t0_${GPU_TAG}" -o "$LOG" --wrap "$WRAP"
