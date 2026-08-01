#!/usr/bin/env bash
# =============================================================================
# sbatch_mainsweep_data.sh — build the MAIN SWEEP dataset (README §A.10).
# =============================================================================
# The k-MIXTURE build. Every previous build pinned one k; this one draws k per
# graph from {1,2,3,4} (`data.sample_hops`, mirroring `sample_cell`) and the
# QUESTION node states the drawn k, so the model must read the task off the input
# instead of assuming a constant. Test splits are built per (N, T, k) — 16 cells x
# 4 k = 64 evaluation conditions.
#
# AXES (see README §A.10 for why each exclusion):
#   N = 16,32,64,128    (8 excluded: a k=4 chain occupies 5 of its 6 content nodes)
#   T = 64,128,256,512  (32 excluded: the fan_out=2 needle is 29 tokens, so a
#                        32-token node leaves 3 tokens of filler and pins every
#                        needle to offset 0 — `validate()` now rejects it)
#   k = 1,2,3,4         (0 excluded: it selects a DIFFERENT builder — a star with
#                        no content-content edges — and is already done as 003)
#   fan_out = 2         (the SPD shortcut is absent; fan_out=1 is the ablation)
#
# 13 of 16 cells fit max_train_len=16384 and form the training distribution; the
# other 3 ((64,512), (128,256), (128,512)) are evaluated as length extrapolation.
#
# WHY SHARDED. The train split is materialised whole in RAM during `datasets.map`
# and the peak scales with graphs x tokens: the N=64 cell hit 61.7 GB at
# n_train=2,000, and a 96 G request OOM-killed the 8k builds (119549-119552).
# Extrapolated, a 16,000-graph build at 8,042 mean tokens does not fit any node
# here. So the blueprint range is split into 16 contiguous shards built as 16
# parallel jobs (~1,000 graphs each, projecting ~31 GB), then concatenated by
# `--mode data_merge` (TextGraphDataset.__add__).
#
# The dev + test-grid job is NOT sharded and does not need to be: `build_split`
# saves each split before starting the next, so its peak is one split (<=200
# graphs), not the 12,800 graphs of the whole grid.
#
# MERGE MEMORY IS THE ONE UNMEASURED PEAK. Sharding bounds the BUILD; the merge
# still materialises every graph in one process. On-disk size is 20.4 KB per 1k
# tokens per graph, so the artifact is only ~2.6 GB and the merge should sit well
# under the 256 G requested — but that is an inference, not a measurement. If it
# OOMs, the fallback is to concatenate lazily at load time instead.
#
# REQUEST A GPU, despite this being mostly `datasets.map` plumbing. The magnetic
# Laplacian eigendecomposition (`compute_magnetic_lap`) defaults to `use_gpu=True`
# and silently falls back to CPU when CUDA is masked -- and the fallback is ~200x
# slower. Measured on this very build: shards that happened to land where a GPU was
# visible finished the decomposition in 13-26 SECONDS for 1000 graphs; shard 12 got
# no GPU allocation and was still going after 41 MINUTES at 4.07 s/graph, having to
# be killed and rerun. One cheap GPU makes it deterministic.
#
# Do NOT use the `amd` partition (MI210/ROCm cannot start this CUDA container) --
# sbatch prints a hint suggesting it for CPU-only jobs; ignore it.
#
# Idempotent: a split already on disk is skipped, so re-running resumes.
#
# ORDER: shards + dev/test can all run at once; the merge MUST wait for every
# shard, so it is submitted with a dependency on them.
#
#   ./src/experiments/context/sbatch_mainsweep_data.sh [PARTITION]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${1:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

SHARDS=16
# Must match 021/022/023 exactly — these knobs form the cache key.
COMMON="--node-counts 16,32,64,128 --token-counts 64,128,256,512 \
--hop-counts 1,2,3,4 --fan-out 2 --max-train-len 16384 \
--n-train 16000 --n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --spd --max-spd 8 --no-rrwp --magnetic \
--magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25 --train-shards $SHARDS"

STAMP="$(date +%Y%m%d_%H%M%S)"

submit() {   # submit <tag> <mem> <time> <extra-flags...>
  local tag="$1" mem="$2" walltime="$3"; shift 3
  local script="$REPO/job_logs/ctx_ms_${tag}_${STAMP}.sh"
  local log="$REPO/job_logs/ctx_ms_${tag}_${STAMP}_%j.out"
  {
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "python -m src.experiments.context $COMMON $*"
  } > "$script"
  chmod +x "$script"

  local wrap="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_ms_${tag}_${STAMP} $script"

  # `sbatch` on this cluster prints an MOTD banner to stdout alongside --parsable's
  # job id, so a bare $(...) capture returns the banner too. That silently produced a
  # malformed --dependency list and the merge job was rejected with "Job dependency
  # problem". Keep only the first all-digits line.
  sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem "$mem" -t "$walltime" \
         --gres gpu:1 -J "ctx_ms_${tag}" -o "$log" "${DEP[@]}" --wrap "$wrap" \
    | grep -oE '^[0-9]+' | head -1
}

DEP=()
echo "[submit] $SHARDS train shards + dev/test grid"
SHARD_IDS=()
for i in $(seq 0 $((SHARDS - 1))); do
  jid=$(submit "shard${i}" 64G 08:00:00 --mode data_prep --train-shard "$i")
  SHARD_IDS+=("$jid")
  echo "   shard $i -> $jid"
done

# --train-shard omitted => dev + the 64-condition test grid, no monolithic train.
# 192G, not the shards' 64G: `build_split` saves each split before starting the next, so
# the peak is ONE split — but the biggest is the (128,512) extrapolation cell at 64,576
# tokens x 200 graphs, which projects to ~50 GB on the 3.86 GB-per-M-token-graphs figure
# the N=64 build measured. 64G would have been a coin flip.
jid=$(submit "devtest" 192G 12:00:00 --mode data_prep)
echo "   dev+test -> $jid"

# The merge needs every shard on disk; let Slurm enforce that rather than a human.
DEP=(--dependency="afterok:$(IFS=:; echo "${SHARD_IDS[*]}")")
jid=$(submit "merge" 256G 04:00:00 --mode data_merge)
echo "   merge    -> $jid  (waits for all $SHARDS shards)"

echo
echo "When everything is COMPLETED, run the pre-registered structural audit BEFORE"
echo "any GPU time — it is the gate the last experiment's mislabeled statistic slipped past:"
echo "  ./.venv/bin/python -m src.experiments.context.analysis.audit_cells"
