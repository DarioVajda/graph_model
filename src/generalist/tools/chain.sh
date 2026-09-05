#!/usr/bin/env bash
# =============================================================================
# chain.sh — submit a generalist run as a chain of dependent sbatch chunks (D8.3)
# =============================================================================
# `frida` caps walltime at 7 days and a trunk run is longer, so a long run is N
# jobs rather than one: chunk 1 is `train`, every chunk after it is
# `resume --from latest`, and each waits on its predecessor with
# `--dependency=afterany`.
#
#   src/generalist/tools/chain.sh src/generalist/configs/runs/001_molecule_generalist_graph_s0.jsonc
#   CHUNKS=8 TIME=12:00:00 src/generalist/tools/chain.sh <config.jsonc>
#   DRY_RUN=1 src/generalist/tools/chain.sh <config.jsonc>     # write, don't submit
#
# Every value comes from the config file (its `execution.sbatch` and `chain`
# blocks, folded onto the RunConfig by `load_config_file`); the environment
# variables below override one at a time.
#
#   CHUNKS TIME PARTITION ACCOUNT GPUS GPUS_PER_CONFIG CPUS MEM
#   INDUCTOR_CACHE DEPENDENCY DRY_RUN
#
# Three properties this script exists to have:
#
# * **`--time` is the window, not the workload** (`feedback-fit-jobs-to-window`).
#   A chunk is however much of the run fits before the wall clock runs out; it is
#   never sized to "how long the training should take", because that number is
#   not known and being wrong about it costs a whole job.
# * **Requeue-safe.** `checkpoint.finalize` writes `COMPLETE` last, and
#   `resume --from latest` only ever resolves to a directory that has one. A
#   chunk killed mid-write therefore leaves a partial checkpoint that the next
#   chunk cannot see, and the run restarts from the previous complete one. At
#   most `save_steps` steps are lost per killed chunk, and never a corrupt state.
# * **`afterany`, not `afterok`.** A chunk killed by the time limit exits
#   non-zero, and that is the *expected* end of a chunk rather than a failure.
#   With `afterok` the chain would stop exactly when it was working as designed.
#
# Job scripts live under `/shared` (`feedback-submit-to-slurm`): the node-local
# scratch a job writes to is gone by the time the next chunk starts, and a chain
# that cannot re-read its own scripts is not a chain. The inductor cache is
# shared across the chunks, because every chunk compiles the same flex kernels
# and that work is CPU-bound and byte-identical between them
# (`project-ddp-flex-bucketing`).
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
  echo "usage: $0 <config.jsonc> [chunks]" >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "FATAL: no config at $CONFIG" >&2
  exit 2
fi

# The submitting interpreter is the project venv, not the login node's python:
# `validate` resolves the whole config, and that reads RDKit and networkx. The
# job scripts themselves say plain `python`, which `slurm_launch.sh` puts the
# venv first on PATH for.
PYTHON="${PYTHON:-$REPO/.venv/bin/python3}"
[ -x "$PYTHON" ] || PYTHON="python3"

# One call, one source of truth. `validate --print-shell` resolves the config
# exactly as a training job will, so a config this script accepts is a config
# that runs — and a typo fails here, before anything is queued.
SETTINGS="$("$PYTHON" -m src.generalist validate --config "$CONFIG" --print-shell)"
rc=$?
if [ $rc -ne 0 ] || [ -z "$SETTINGS" ]; then
  echo "FATAL: the config did not validate; nothing submitted" >&2
  echo "$SETTINGS" >&2
  exit $rc
fi
eval "$SETTINGS"

CHUNKS="${2:-${CHUNKS:-$GEN_CHUNKS}}"
TIME="${TIME:-$GEN_TIME}"
PARTITION="${PARTITION:-$GEN_PARTITION}"
ACCOUNT="${ACCOUNT:-$GEN_ACCOUNT}"
GPUS="${GPUS:-$GEN_GPUS}"
GPUS_PER_CONFIG="${GPUS_PER_CONFIG:-$GEN_GPUS_PER_CONFIG}"
CPUS="${CPUS:-$GEN_CPUS}"
MEM="${MEM:-$GEN_MEM}"
DEPENDENCY="${DEPENDENCY:-$GEN_DEPENDENCY}"
CONTAINER="${CONTAINER:-$GEN_CONTAINER}"
INDUCTOR_CACHE="${INDUCTOR_CACHE:-$GEN_INDUCTOR_CACHE}"
DRY_RUN="${DRY_RUN:-0}"

CHAIN_DIR="$REPO/src/generalist/results/chain/$GEN_RUN_NAME"
LOG_DIR="$CHAIN_DIR/logs"
mkdir -p "$LOG_DIR"

# The shared cache is opt-in AND self-healing: `slurm_launch.sh` uses
# SWEEP_INDUCTOR_CACHE only if the directory exists, so creating it here is what
# turns sharing on, and a run that never populates it just compiles.
if [ -n "$INDUCTOR_CACHE" ]; then
  case "$INDUCTOR_CACHE" in
    /*) ;;
    *) INDUCTOR_CACHE="$REPO/$INDUCTOR_CACHE" ;;
  esac
  mkdir -p "$INDUCTOR_CACHE"
fi

# `gpus_per_config` is the single source of truth for the rank count: at 1 the
# job runs `python -m`, above 1 it runs `torchrun` (sweep/README.md). Naming a
# count in GPUS as well is how a job ends up with two ranks and one card.
GRES="gpu:$GPUS_PER_CONFIG"
CONSTRAINT=""
if [ -n "$GPUS" ]; then
  IFS='|' read -r -a _brands <<< "$GPUS"
  for brand in "${_brands[@]}"; do
    [ -z "$brand" ] && continue
    CONSTRAINT="${CONSTRAINT:+$CONSTRAINT|}GPU_BRD:$brand"
  done
fi

if [ "$GPUS_PER_CONFIG" -gt 1 ]; then
  RUNNER="torchrun --standalone --nproc_per_node $GPUS_PER_CONFIG -m src.generalist"
else
  RUNNER="python -m src.generalist"
fi

echo "[chain] run       $GEN_RUN_NAME"
echo "[chain] output    $GEN_RUN_DIR"
echo "[chain] config    $CONFIG (hash $GEN_CONFIG_HASH)"
echo "[chain] chunks    $CHUNKS x $TIME on $PARTITION ($GRES${CONSTRAINT:+, $CONSTRAINT})"
echo "[chain] scripts   $CHAIN_DIR"

PREV=""
for i in $(seq 1 "$CHUNKS"); do
  SCRIPT="$CHAIN_DIR/chunk_$i.sh"
  if [ "$i" -eq 1 ]; then
    BODY="$RUNNER train --config $CONFIG"
  else
    BODY="$RUNNER resume --from latest --config $CONFIG"
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "# chunk $i/$CHUNKS of $GEN_RUN_NAME — generated by src/generalist/tools/chain.sh"
    echo "set -x"
    echo "cd $REPO"
    # Chunk 1 is the only one that may `train`. If it is requeued after a
    # checkpoint exists, `train` refuses (it would restart the schedule beside a
    # live run), so it falls through to a resume — which is what a requeued first
    # chunk means.
    if [ "$i" -eq 1 ]; then
      echo "$BODY || $RUNNER resume --from latest --config $CONFIG"
    else
      echo "$BODY"
    fi
    echo "rc=\$?; echo CHAIN_CHUNK_${i}_EXIT=\$rc; exit \$rc"
  } > "$SCRIPT.tmp.$$"
  chmod +x "$SCRIPT.tmp.$$"
  # Replace the inode instead of truncating it. A running chunk is a `bash
  # chunk_1.sh` that reads the file by byte offset as it goes, so rewriting the
  # same inode under it makes it resume mid-line in the new text — it executed
  # half of a config path as a command and exited 127, which is how the first
  # smoke run's requeue fallback was lost. `mv` leaves the running shell on the
  # old inode, where it stays consistent to the end.
  mv -f "$SCRIPT.tmp.$$" "$SCRIPT"

  WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
SWEEP_LOGIN=$REPO/login.sh SWEEP_INDUCTOR_CACHE=$INDUCTOR_CACHE \
bash $REPO/sweep/slurm_launch.sh gen_${GEN_RUN_NAME}_c$i $SCRIPT"

  ARGS=(-p "$PARTITION" -A "$ACCOUNT" -c "$CPUS" --mem "$MEM" -t "$TIME"
        --gres "$GRES" -J "gen_${GEN_RUN_NAME}_c$i"
        -o "$LOG_DIR/chunk_$i.out")
  [ -n "$CONSTRAINT" ] && ARGS+=(--constraint "$CONSTRAINT")
  [ -n "$PREV" ] && ARGS+=(--dependency "$DEPENDENCY:$PREV")

  if [ "$DRY_RUN" != "0" ]; then
    echo "[chain] (dry run) sbatch ${ARGS[*]} --wrap '<srun ... $SCRIPT>'"
    PREV="<chunk_$i>"
    continue
  fi

  # `--parsable` promises one bare job id, and on this cluster it does not
  # deliver one: the login banner (uptime, maintenance windows) is printed on
  # stdout ahead of it, so a naive capture hands the next chunk a dependency
  # argument several lines long. Take the last line that is *only* an id.
  # Capturing the id is the step that fails silently: a chain whose second chunk
  # depends on an empty job id is submitted with no dependency at all, and two
  # chunks then train the same run at once.
  JOB="$(sbatch --parsable "${ARGS[@]}" --wrap "$WRAP" \
         | tr -d '\r' | grep -Eo '^[0-9]+(_[0-9]+)?$' | tail -1)"
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$JOB" ]; then
    echo "FATAL: sbatch failed for chunk $i (rc=$rc); the chain stops here" >&2
    exit 1
  fi
  echo "[chain] chunk $i -> job $JOB${PREV:+ (after $PREV)}"
  PREV="$JOB"
done

echo "[chain] logs: $LOG_DIR/chunk_*.out"
