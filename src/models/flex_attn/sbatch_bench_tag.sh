#!/usr/bin/env bash
# =============================================================================
# sbatch_bench_tag.sh — submit the real-input TAG benchmark to a compute node.
# =============================================================================
# Thin launcher: builds a job script of `python -m src.models.flex_attn.bench_real`
# lines and hands it to the sweep runner's `slurm_launch.sh` inside the project
# container, exactly as a sweep job would be launched.
#
#   ./src/models/flex_attn/sbatch_bench_tag.sh [GPU_TYPE] [DATASET ...]
#
#   GPU_TYPE   default H100 — matches the existing results_h100* scaling figures,
#              which must not be mixed with numbers from another GPU.
#   DATASET    default: cora. Any of cora pubmed ogbn-arxiv reddit.
#
# Results land in src/models/flex_attn/results_<gpu>_tag/{tag.jsonl,tag.md}.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

GPU_TYPE="${1:-H100}"; shift || true
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(cora)

GPU_TAG="$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="src/models/flex_attn/results_${GPU_TAG}_tag"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SCRIPT="$REPO/job_logs/bench_tag_${STAMP}.sh"
LOG="$REPO/job_logs/bench_tag_${STAMP}_%j.out"

mkdir -p "$REPO/job_logs" "$OUT_DIR"

{
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "nvidia-smi"
    for ds in "${DATASETS[@]}"; do
        # The paper's setting: k_hop=0, gradient checkpointing on, B=1.
        echo "python -m src.models.flex_attn.bench_real --experiment tag --arm $ds \\"
        echo "    --methods eager flex sdpa --n-batches 24 --passes 3 \\"
        echo "    --out-dir $OUT_DIR"
        # Same inputs without gradient checkpointing: isolates the attention
        # backend from the recompute the paper's config also pays.
        echo "python -m src.models.flex_attn.bench_real --experiment tag --arm $ds \\"
        echo "    --methods eager flex sdpa --n-batches 24 --passes 3 \\"
        echo "    --no-gradient-checkpointing --out-dir $OUT_DIR"
    done
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh bench_tag_${STAMP} $JOB_SCRIPT"

echo "[submit] gpu=$GPU_TYPE datasets=${DATASETS[*]}"
echo "[submit] job script: $JOB_SCRIPT"
echo "[submit] log:        $LOG"
# 6h: the cold pass compiles/autotunes one flex kernel per distinct (L, N) bucket
# (≤10 for the TAG length distributions) at max-autotune, which dominates setup.
sbatch -p frida -A povejmo --gres="gpu:${GPU_TYPE}:1" -c 8 --mem=64G -t 06:00:00 \
       -J "bench_tag_${GPU_TAG}" -o "$LOG" --wrap "$WRAP"
