#!/usr/bin/env bash
# =============================================================================
# sbatch_context4k_data.sh — build the 4k-capped context mixture for D2.
# =============================================================================
# A direct copy of src/experiments/context/sbatch_mainsweep_data.sh with ONE
# knob changed: --max-train-len 16384 -> 4096. Everything else is byte-identical,
# because every other flag is part of `data_config_key()` and a drift there would
# either orphan this build or silently reuse the 16k one.
#
# What the cap changes. `train_cells()` keeps only cells whose packed length fits
# the cap, so the training distribution shrinks from 13 cells to 6:
#
#   in-cap at 4096:  (16,64) 960   (16,128) 1856  (16,256) 3648
#                    (32,64) 1984  (32,128) 3904  (64,64)  4032
#   the other 10 cells are still built, and scored as length extrapolation.
#
# The cache key carries `cap4096`, so this cannot collide with the existing
# `cap16384` tree — both can live on disk at once.
#
# Sharded exactly like the 16k build: 16 train shards + a dev/test job + a merge
# that waits on the shards. Idempotent — a split already on disk is skipped.
#
# Memory is lower than the 16k build's because the biggest thing this writes is
# unchanged (the dev/test grid still contains the (128,512) extrapolation cell),
# so devtest keeps its 192G while the shards drop to 48G: a train shard now packs
# at most 4,032 tokens per graph instead of 16,192.
#
#   ./src/experiments/bias_sharing/sbatch_context4k_data.sh [PARTITION]
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${1:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

SHARDS=16
# Must match 003_context4k_g_sweep.jsonc exactly — these knobs form the cache key.
COMMON="--node-counts 16,32,64,128 --token-counts 64,128,256,512 \
--hop-counts 1,2,3,4 --fan-out 2 --max-train-len 4096 \
--n-train 16000 --n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --spd --max-spd 8 --no-rrwp --magnetic \
--magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25 --train-shards $SHARDS"

STAMP="$(date +%Y%m%d_%H%M%S)"

submit() {   # submit <tag> <mem> <time> <extra-flags...>
  local tag="$1" mem="$2" walltime="$3"; shift 3
  local script="$REPO/job_logs/ctx4k_${tag}_${STAMP}.sh"
  local log="$REPO/job_logs/ctx4k_${tag}_${STAMP}_%j.out"
  {
    echo "#!/usr/bin/env bash"
    echo "set -x"
    echo "python -m src.experiments.context $COMMON $*"
  } > "$script"
  chmod +x "$script"

  local wrap="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx4k_${tag}_${STAMP} $script"

  # sbatch prints an MOTD banner alongside --parsable's job id on this cluster, so
  # the id has to be picked out of it. ANCHOR BOTH ENDS: the banner's clock line
  # ("18:07:24 up 47 days") starts with digits, so the `^[0-9]+` that
  # sbatch_mainsweep_data.sh uses captures "18" for every job and the merge's
  # --dependency comes out as afterok:18:18:... ("Job dependency problem").
  sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem "$mem" -t "$walltime" \
         --gres gpu:1 -J "ctx4k_${tag}" -o "$log" "${DEP[@]}" --wrap "$wrap" \
    | grep -oE '^[0-9]+$' | head -1
}

DEP=()
echo "[submit] $SHARDS train shards + dev/test grid  (cap 4096)"
SHARD_IDS=()
for i in $(seq 0 $((SHARDS - 1))); do
  jid=$(submit "shard${i}" 48G 06:00:00 --mode data_prep --train-shard "$i")
  SHARD_IDS+=("$jid")
  echo "   shard $i -> $jid"
done

# --train-shard omitted => dev + the 64-condition test grid, no monolithic train.
# Still 192G: the grid contains the (128,512) extrapolation cell regardless of the
# TRAIN cap, and that cell is what sets the peak.
jid=$(submit "devtest" 192G 12:00:00 --mode data_prep)
echo "   dev+test -> $jid"

DEP=(--dependency="afterok:$(IFS=:; echo "${SHARD_IDS[*]}")")
jid=$(submit "merge" 256G 04:00:00 --mode data_merge)
echo "   merge    -> $jid  (waits for all $SHARDS shards)"
MERGE_ID="$jid"

echo
echo "Merge job id: $MERGE_ID"
echo "Training (003_context4k_g_sweep.jsonc) must wait for it:"
echo "  python3 -m sweep src.experiments.context \\"
echo "      src/experiments/bias_sharing/configs/003_context4k_g_sweep.jsonc"
