#!/usr/bin/env bash
# =============================================================================
# sbatch_bench_graphmask.sh — add the mask-matched plain-LLM reference point.
# =============================================================================
# `sdpa` (plain causal, flash-eligible) stays exactly as it is: it is the
# deliberate theoretical floor — what the model would cost if the graph structure
# were abandoned entirely, block skipping included. This script adds a SECOND
# reference beside it, not a replacement:
#
#   sdpa-graphmask — the same stock LLM, no bias, but handed GTLM's own dense
#   structural mask (causal relaxed to bidirectional between prefix tokens, plus
#   padding and the K-hop gate). Measured on Cora, GTLM's mask admits 0.697 of
#   the L*L matrix against plain causal's 0.500 — so `sdpa` alone understates the
#   attention work GTLM is obliged to do by ~29%. This arm prices the mask shape
#   on its own, with the bias still absent.
#
# `sdpa` is re-run alongside it in the same process so the ratio between them is
# measured under identical conditions rather than across jobs.
#
#   ./src/models/flex_attn/sbatch_bench_graphmask.sh [GPU_TYPE]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

GPU_TYPE="${1:-H100}"
GPU_TAG="$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="src/models/flex_attn/results_${GPU_TAG}_tag"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT="$REPO/job_logs/bench_gmask_${STAMP}.sh"
LOG="$REPO/job_logs/bench_gmask_${STAMP}_%j.out"
mkdir -p "$REPO/job_logs" "$OUT_DIR"

M="sdpa sdpa-graphmask"
{
    echo "#!/usr/bin/env bash"; echo "set -x"; echo "nvidia-smi"
    for ds in cora pubmed ogbn-arxiv reddit; do
        echo "python -m src.models.flex_attn.bench_real --experiment tag --arm $ds \\"
        echo "    --methods $M --n-batches 24 --passes 3 --out-dir $OUT_DIR"
    done
    # Cora without gradient checkpointing too, matching the existing pair.
    echo "python -m src.models.flex_attn.bench_real --experiment tag --arm cora \\"
    echo "    --methods $M --n-batches 24 --passes 3 --no-gradient-checkpointing --out-dir $OUT_DIR"
    # GraphQA at natural L — where its fair comparison lives (no flex here, so
    # bucket padding would only add noise).
    for arm in standard/node_count incidence/node_count; do
        echo "python -m src.models.flex_attn.bench_real --experiment graphqa --arm $arm \\"
        echo "    --methods $M --pad-mode batch --n-batches 24 --passes 3 --out-dir $OUT_DIR"
    done
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh bench_gmask_${STAMP} $SCRIPT"

echo "[submit] log: $LOG"
# No flex arms -> no autotune; these are minutes, not hours.
sbatch -p frida -A povejmo --gres="gpu:${GPU_TYPE}:1" -c 8 --mem=64G -t 02:00:00 \
       -J "bgmask" -o "$LOG" --wrap "$WRAP"
