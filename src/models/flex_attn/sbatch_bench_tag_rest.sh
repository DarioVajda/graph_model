#!/usr/bin/env bash
# =============================================================================
# sbatch_bench_tag_rest.sh — the remaining TAG datasets + a fresh parity run.
# =============================================================================
# Companion to sbatch_bench_tag.sh: one job per dataset (so a slow dataset does
# not block the others), plus one job that re-runs the flex<->eager parity tests
# on this GPU. The rebuttal claims "no reported number changes"; that claim
# should rest on a test run from this week, not on the test file's existence.
#
#   ./src/models/flex_attn/sbatch_bench_tag_rest.sh [GPU_TYPE]
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
    local script="$REPO/job_logs/bench_tag_${stamp}.sh"
    local log="$REPO/job_logs/bench_tag_${stamp}_%j.out"
    { echo "#!/usr/bin/env bash"; echo "set -x"; echo "nvidia-smi"; printf '%s\n' "$@"; } > "$script"
    chmod +x "$script"
    local wrap="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh bench_tag_${stamp} $script"
    echo "[submit] $label -> $log"
    sbatch -p frida -A povejmo --gres="gpu:${GPU_TYPE}:1" -c 8 --mem=64G -t 06:00:00 \
           -J "btag_${label}" -o "$log" --wrap "$wrap"
}

METHODS="eager flex flex-nobias sdpa"

for ds in pubmed ogbn-arxiv reddit; do
    submit "$ds" \
        "python -m src.models.flex_attn.bench_real --experiment tag --arm $ds --methods $METHODS --n-batches 24 --passes 3 --out-dir $OUT_DIR" \
        "python -m src.models.flex_attn.bench_real --experiment tag --arm $ds --methods $METHODS --n-batches 24 --passes 3 --no-gradient-checkpointing --out-dir $OUT_DIR"
done

# Cora again, now including the flex-nobias decomposition arm (the first Cora job
# predates it). Same out-dir; summarize() folds every record into tag.md.
submit "cora_decomp" \
    "python -m src.models.flex_attn.bench_real --experiment tag --arm cora --methods $METHODS --n-batches 24 --passes 3 --out-dir $OUT_DIR"

# Tier-3 evidence: flex == eager, forward and gradients, every bias type, k∈{0,2}.
submit "parity" \
    "python -m pytest tests/models/test_flex_attention.py -v --tb=short" \
    "python -m pytest tests/models/test_modeling_gtlm_llama_v2.py -v --tb=short"
