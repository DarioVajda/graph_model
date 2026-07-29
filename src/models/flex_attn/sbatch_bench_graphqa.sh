#!/usr/bin/env bash
# =============================================================================
# sbatch_bench_graphqa.sh — the real-input benchmark on GraphQA.
# =============================================================================
# GraphQA is the *opposite* regime from the TAG benchmarks: L ~29 tokens
# (standard encoding) to ~150 (incidence), against TAG's ~600-1300. The point of
# running it is to establish where flex stops paying, not to claim a speedup.
#
# Fairness note — why two pad modes per arm. Flex requires block-aligned L and
# the block size is 128, so a 29-token GraphQA sequence is padded ~4x however
# tight the ladder. Charging eager for that same padding (the equal-L setting
# used for TAG, where padding was 5-27%) would make eager look slow for a cost
# it would never actually pay. So each arm is measured twice:
#
#   bucket  — every arm at the same padded L. Apples-to-apples between arms.
#   batch   — dense arms at their natural per-batch L (flex cannot run here).
#
# The production question "which backend should GraphQA use?" is answered by
# comparing flex@bucket against eager@batch, NOT eager@bucket.
#
#   ./src/models/flex_attn/sbatch_bench_graphqa.sh [GPU_TYPE]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

GPU_TYPE="${1:-H100}"
GPU_TAG="$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="src/models/flex_attn/results_${GPU_TAG}_tag"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs" "$OUT_DIR"

submit () {              # submit <label> <body...>
    local label="$1"; shift
    local stamp; stamp="$(date +%Y%m%d_%H%M%S)_$label"
    local script="$REPO/job_logs/bench_gqa_${stamp}.sh"
    local log="$REPO/job_logs/bench_gqa_${stamp}_%j.out"
    { echo "#!/usr/bin/env bash"; echo "set -x"; echo "nvidia-smi"; printf '%s\n' "$@"; } > "$script"
    chmod +x "$script"
    local wrap="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh bench_gqa_${stamp} $script"
    echo "[submit] $label -> $log"
    sbatch -p frida -A povejmo --gres="gpu:${GPU_TYPE}:1" -c 8 --mem=64G -t 06:00:00 \
           -J "bgqa_${label}" -o "$log" --wrap "$wrap"
}

BASE="python -m src.models.flex_attn.bench_real --experiment graphqa --n-batches 24 --passes 3 --out-dir $OUT_DIR"

# Two tasks per encoding — node_count (shortest prompts) and shortest_path
# (longest) — to show the result is a property of the regime, not one task.
for arm in standard/node_count standard/shortest_path \
           incidence/node_count incidence/shortest_path; do
    label="$(echo "$arm" | tr '/' '_')"
    submit "$label" \
        "$BASE --arm $arm --len-bucket-multiple 128 --methods eager flex flex-nobias sdpa" \
        "$BASE --arm $arm --pad-mode batch --methods eager sdpa"
done

# The production 512 ladder on the shortest arm: documents how much of any flex
# deficit is the ladder rather than the kernel (94% padding at L=29).
submit "standard_ladder512" \
    "$BASE --arm standard/node_count --len-bucket-multiple 512 --methods eager flex sdpa"
