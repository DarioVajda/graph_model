#!/usr/bin/env bash
# =============================================================================
# sbatch_bench_rerun.sh — re-measure after the allocator-stall fix.
# =============================================================================
# `run_method` used to call torch.cuda.empty_cache() immediately before the timed
# loop, so the first timed step re-acquired the whole allocator pool from the
# driver. Measured cost: up to 6.6 s against a 103 ms median, which nearly
# doubled the mean of a ~100 ms arm. An untimed warmup step now absorbs it.
#
# Two jobs:
#   gmask  — the sdpa vs sdpa-graphmask reference points (the run that exposed
#            the bug; its numbers are unusable and must be replaced).
#   verify — a full 5-method re-run on the two datasets that anchor the headline
#            table, to confirm the published speedups survive the fix.
#
#   ./src/models/flex_attn/sbatch_bench_rerun.sh [GPU_TYPE]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

GPU_TYPE="${1:-H100}"
GPU_TAG="$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="src/models/flex_attn/results_${GPU_TAG}_tag"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs" "$OUT_DIR"

submit () {
    local label="$1"; shift
    local stamp; stamp="$(date +%Y%m%d_%H%M%S)_$label"
    local script="$REPO/job_logs/bench_rerun_${stamp}.sh"
    local log="$REPO/job_logs/bench_rerun_${stamp}_%j.out"
    { echo "#!/usr/bin/env bash"; echo "set -x"; echo "nvidia-smi"; printf '%s\n' "$@"; } > "$script"
    chmod +x "$script"
    local wrap="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh bench_rerun_${stamp} $script"
    echo "[submit] $label -> $log"
    sbatch -p frida -A povejmo --gres="gpu:${GPU_TYPE}:1" -c 8 --mem=64G -t 03:00:00 \
           -J "brerun_${label}" -o "$log" --wrap "$wrap"
}

B="python -m src.models.flex_attn.bench_real"
GM="sdpa sdpa-graphmask"

# ── the mask-matched reference points, redone ──
{
  cmds=()
  for ds in cora pubmed ogbn-arxiv reddit; do
    cmds+=("$B --experiment tag --arm $ds --methods $GM --n-batches 24 --passes 3 --out-dir $OUT_DIR")
  done
  for arm in standard/node_count incidence/node_count; do
    cmds+=("$B --experiment graphqa --arm $arm --methods $GM --pad-mode batch --n-batches 24 --passes 3 --out-dir $OUT_DIR")
  done
  submit gmask "${cmds[@]}"
}

# ── headline table re-verification: every arm, the two anchor datasets ──
submit verify \
    "$B --experiment tag --arm cora       --methods eager flex flex-nobias sdpa-graphmask sdpa --n-batches 24 --passes 3 --out-dir $OUT_DIR" \
    "$B --experiment tag --arm ogbn-arxiv --methods eager flex flex-nobias sdpa-graphmask sdpa --n-batches 24 --passes 3 --out-dir $OUT_DIR"
