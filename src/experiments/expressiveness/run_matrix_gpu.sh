#!/usr/bin/env bash
# =============================================================================
# run_matrix_gpu.sh — run one or more expressiveness training runs on ONE GPU.
# =============================================================================
#
# WHAT IT DOES
#   You hand it a list of "specs". Each spec is one unit of work. The script
#   walks the list in order, on a single GPU, doing one run at a time:
#
#       <size>:<seed>:<k>    train ONE config (num_nodes=size, seed, k_hop=k)
#       PREP:<size>          only build that size's datasets, don't train
#
#   Before any run it makes sure the needed dataset exists (building it if not),
#   then launches `python -m src.experiments.expressiveness` with the right
#   buckets / batch size / hyperparameters for that size.
#
# USAGE (run directly, from inside the container)
#   run_matrix_gpu.sh <label> <spec> [<spec> ...]
#     <label>   a tag for this job; names its logs + its compile cache dir
#     <spec>    one of the two forms above
#
#   Examples:
#     run_matrix_gpu.sh gpu0 500:0:2 500:1:2 500:2:2   # 3 seeds, run in sequence
#     run_matrix_gpu.sh gpu0 PREP:1000                 # just build the datasets
#
# SUBMIT AS A SLURM JOB (copy + edit NAME / SPECS)
#   EXP=/shared/workspace/povejmo/graph_model/src/experiments/expressiveness
#   SQSH=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
#   NAME=n200_k1          # job name + log filename prefix
#   SPECS="200:0:1"       # one or more "<size>:<seed>:<k>" (space-separated)
#
#   sbatch -p frida --gres=gpu:B200:1 -c 16 --mem=64G -t 6:00:00 \
#     -J "$NAME" -o "$EXP/job_logs/slurm_${NAME}_%j.out" \
#     --wrap "srun --container-image=$SQSH --container-mounts=/shared:/shared \
#             env PYTHONUNBUFFERED=1 bash $EXP/run_matrix_gpu.sh $NAME $SPECS"
#   SBATCH_EXAMPLE
#
# =============================================================================
set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
# This runs INSIDE the transformers_deepspeed container (py3.10 base), so the
# project's py3.10 .venv resolves correctly. HOME points at the shared home so
# the HF / wandb credentials + model cache are found.
REPO=/shared/workspace/povejmo/graph_model
export HOME=/shared/home/dario.vajda
cd "$REPO"
set +u; source .venv/bin/activate; set -u   # (.venv references unbound vars)
PY=python

# ── Per-job logging + compile cache ──────────────────────────────────────────
LABEL="$1"; shift                            # first arg = the job tag
EXPDIR="$REPO/src/experiments/expressiveness"
LOGDIR="$EXPDIR/job_logs"
DATADIR="$EXPDIR/datasets"
mkdir -p "$LOGDIR"
GPULOG="$LOGDIR/$LABEL.log"                   # this job's orchestration log
export TORCHINDUCTOR_CACHE_DIR="$REPO/.inductor_cache/$LABEL"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$GPULOG"; }

# ── Per-size settings ────────────────────────────────────────────────────────
# For each graph size we fix:
#   LEN_BUCKETS  : flex packed-token-length ladder
#   NODE_BUCKETS : flex node-count ladder (drives the (N,N) bias padding)
#   MIN / MAX    : node-count range graphs are drawn from (0.8x .. 1.2x size)
# All bucket values are multiples of 128 so they're valid for k=0 and k>0 alike.
declare -A LEN_BUCKETS NODE_BUCKETS MIN_NODES MAX_NODES
set_size() {  # set_size <size> <len_buckets> <node_buckets> <min> <max>
  LEN_BUCKETS[$1]=$2; NODE_BUCKETS[$1]=$3; MIN_NODES[$1]=$4; MAX_NODES[$1]=$5
}
#         size  LEN_BUCKETS                NODE_BUCKETS   MIN   MAX
set_size   100  "128,256"                  "128"           80   120
set_size   200  "256,384"                  "256"          160   240
set_size   500  "640,768"                  "512,640"      400   600
set_size  1000  "1408,1664,1792"           "1024,1280"    800  1200
set_size  2000  "3200,3712,4096,4224"      "2048,2432"   1600  2400

# ── Fixed training knobs (shared by every run; edit here to adapt the sweep) ──
TRAIN_SIZE=2000          # number of training graphs   (dataset tag "2k")
EVAL_SIZE=200            # number of eval graphs        (dataset tag "200")
EPOCHS=25
EVAL_STEPS=75
EFFECTIVE_BATCH=32       # batch_size * accumulation_steps, held constant
MAGNETIC_M=128
WANDB_PROJECT=GraphLLM

# ── Build a size's datasets if missing ───────────────────────────────────────
ensure_dataset() {  # ensure_dataset <size>
  local size=$1
  local mn=${MIN_NODES[$size]} mx=${MAX_NODES[$size]}
  # Filenames follow load_or_create_dataset's convention (2000 -> "2k").
  local train="$DATADIR/2k_hard_n${mn}-${mx}_rcm_dataset.gtds"
  local evald="$DATADIR/200_hard_n${mn}-${mx}_rcm_dataset.gtds"

  if [ -d "$train" ] && [ -d "$evald" ]; then
    log "datasets size=$size (n${mn}-${mx}) present — skip gen"; return 0
  fi

  # flock: if several GPUs want this size at once, one builds, the rest wait.
  log "ensuring datasets size=$size (n${mn}-${mx}) under flock ..."
  flock "/tmp/expr_ds_n${mn}-${mx}.lock" "$PY" -c "
from src.experiments.expressiveness.datasets import load_or_create_dataset as L
from src.experiments.expressiveness.config import MODEL_NAME as M
for n in ($TRAIN_SIZE, $EVAL_SIZE):   # train set, then eval set
    L(n, easy=False, model_name=M, min_nodes=$mn, max_nodes=$mx, magnetic_m=$MAGNETIC_M)
" >> "$GPULOG" 2>&1

  local rc=$?
  if [ $rc -ne 0 ] || [ ! -d "$train" ] || [ ! -d "$evald" ]; then
    log "ERROR dataset gen for size=$size failed (rc=$rc, train present=$([ -d "$train" ]&&echo y||echo n) eval present=$([ -d "$evald" ]&&echo y||echo n)) — aborting"
    return 1
  fi
  log "datasets size=$size ready"
}

# ── Train one (size, seed, k) config ─────────────────────────────────────────
run_one() {  # run_one <size> <seed> <k>
  local size=$1 seed=$2 k=$3
  ensure_dataset "$size" || { log "SKIP n=$size s=$seed k=$k (dataset unavailable)"; return 1; }

  # Keep the effective batch (= bs * acc) constant across sizes. At 2000 nodes
  # the (B,H,N,N) bias is huge, so use a smaller micro-batch there.
  local bs=4
  [ "$size" = "2000" ] && bs=2
  local acc=$(( EFFECTIVE_BATCH / bs ))

  local rl="$LOGDIR/run_n${size}_s${seed}_k${k}.log"   # this run's training log
  log "START n=$size seed=$seed k=$k (bs=$bs acc=$acc) -> $rl"
  "$PY" -m src.experiments.expressiveness \
    --impls v2-flex --k-hops "$k" --seeds "$seed" \
    --num-nodes "$size" \
    --len-buckets "${LEN_BUCKETS[$size]}" --node-buckets "${NODE_BUCKETS[$size]}" \
    --magnetic-m "$MAGNETIC_M" --bias-lr 1e-3 --lora --lora-r 8 --lr 1e-4 \
    --batch-size "$bs" --accumulation-steps "$acc" \
    --train-dataset-size "$TRAIN_SIZE" --eval-dataset-size "$EVAL_SIZE" \
    --num-epochs "$EPOCHS" --eval-steps "$EVAL_STEPS" --report-to "$WANDB_PROJECT" \
    > "$rl" 2>&1
  local rc=$?
  log "END   n=$size seed=$seed k=$k rc=$rc"
}

# ── Main: log in, then process each spec in order ────────────────────────────
log "=== job $LABEL start on $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} | specs: $* ==="
bash login.sh >>"$GPULOG" 2>&1 && log "hf+wandb login ok" || log "WARN login.sh rc=$?"

for spec in "$@"; do
  case "$spec" in
    PREP:*)
      ensure_dataset "${spec#PREP:}"          # build datasets only
      ;;
    *)
      IFS=: read -r size seed k <<< "$spec"   # "<size>:<seed>:<k>"
      run_one "$size" "$seed" "$k"
      ;;
  esac
done

log "=== job $LABEL DONE ==="
