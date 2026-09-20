#!/usr/bin/env bash
# =============================================================================
# run_cli.sh — run one `python -m src.generalist ...` on a COMPUTE node.
# =============================================================================
# The single-job counterpart to chain.sh: submits one sbatch job that runs the
# harness inside the project container and BLOCKS until it finishes, so the exit
# code is the command's. Output lands in
# src/generalist/results/job_logs/<stamp>.out and the tail is echoed here.
#
#   src/generalist/tools/run_cli.sh data_prep --config src/generalist/configs/probes/000_smoke.jsonc
#   GPU=1 src/generalist/tools/run_cli.sh eval --checkpoint <ckpt> --config <cfg>
#   GPU=4 GPUS="B300|B200" src/generalist/tools/run_cli.sh fork --from <ckpt> ...
#
# Env overrides: PARTITION (frida), CPUS (16), MEM (64G), TIME (02:00:00),
# GPU (0 -> CPU-only; 1 -> one GPU; >1 -> that many ranks under torchrun), NAME,
# INDUCTOR_CACHE, GPUS (brand list, default all Blackwell/H100/A100), EXCLUDE
# (node list to keep off), WAIT (1 blocks and echoes the tail, 0 submits and
# prints the job id — which is what several of these at once needs).
#
# GPU is the rank count as well as the card count, which is `chain.sh`'s rule
# (`gpus_per_config` is the single source of truth) for the same reason: naming
# the two separately is how a job ends up with four ranks and one card. Above 1
# the body runs under `torchrun --standalone`, and the harness picks the world
# size up from the environment — the mixture sampler splits each optimizer step
# across the ranks (`micro_batch_tokens` is `tokens_per_step / (accumulation_steps
# x world_size)`), so the step is unchanged and only the wall clock moves.
# `accumulation_steps` has to come down by the same factor to hold the
# micro-batch, and it is a CLI flag: `--accumulation-steps`.
#
# GPUS takes `chain.sh`'s `|`-separated brand list ("B300|B200") and renders it
# as a `GPU_BRD:` constraint. The default spans every fast brand, which is right
# for a scoring pass and wrong for anything whose per-rank peak exceeds 80 GB —
# name the brands when the run has a measured peak.
#
# INDUCTOR_CACHE names a directory to share compiled flex kernels with, the way
# `execution.sbatch.inductor_cache` does for a sweep or a chain. There is no
# config parsing here, so it has to be passed by hand; point it at the run's own
# cache when the job scores that run's checkpoints and the compile is free
# instead of a few minutes of autotuning per shape bucket. Empty (the default)
# means a per-job cache, which is correct for a one-off.
#
# `data_prep`, `eval` and `fork` are the modes that belong here: each is one job
# of a length that is known before it starts — a build, a scoring pass, or an
# anneal, which trains exactly `decay_steps + 1` steps. `train` and `resume` go
# through chain.sh, which owns the chunking and the dependency discipline a run
# of unknown length needs.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

CONTAINER="${CONTAINER:-/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh}"
# The repo venv is pinned at transformers 4.50.3, which the GTLM attention
# internals are written against. A step that shares nothing with training — the
# §9.4 writer, which needs 5.5+ to load a gemma4 checkpoint — points this at its
# own venv instead of dragging the trunk's environment forward.
VENV_BIN="${VENV_BIN:-$REPO/.venv/bin}"
PARTITION="${PARTITION:-frida}"
CPUS="${CPUS:-16}"
MEM="${MEM:-64G}"
TIME="${TIME:-02:00:00}"
GPU="${GPU:-0}"
GPUS="${GPUS:-}"
EXCLUDE="${EXCLUDE:-}"
WAIT="${WAIT:-1}"
INDUCTOR_CACHE="${INDUCTOR_CACHE:-}"
if [ -n "$INDUCTOR_CACHE" ]; then
  case "$INDUCTOR_CACHE" in /*) ;; *) INDUCTOR_CACHE="$REPO/$INDUCTOR_CACHE" ;; esac
fi

LOG_DIR="$REPO/src/generalist/results/job_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
NAME="${NAME:-gen_${1:-cli}}"
SCRIPT="$LOG_DIR/$STAMP.sh"
LOG="$LOG_DIR/$STAMP.out"

# RUNMOD names the module to run, so a one-off tool under `tools/` gets the
# container, the constraint list and the log discipline the harness modes get
# rather than a second launcher that drifts from this one.
#   RUNMOD=src.generalist.tools.notation_probe GPU=1 src/generalist/tools/run_cli.sh --out ...
RUNMOD="${RUNMOD:-src.generalist}"

if [ "$GPU" -gt 1 ] 2>/dev/null; then
  RUNNER="torchrun --standalone --nproc_per_node $GPU -m $RUNMOD"
else
  RUNNER="python -m $RUNMOD"
fi

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "cd $REPO"
  echo "$RUNNER $* ; rc=\$?"
  echo "echo CLI_EXIT=\$rc; exit \$rc"
} > "$SCRIPT"
chmod +x "$SCRIPT"

# MELLANOX_VISIBLE_DEVICES=none skips the enroot mellanox hook, which fails on
# nodes without rdma_cm (ixb7); single-node jobs need no InfiniBand (CLAUDE.md).
WRAP="srun --export=ALL,MELLANOX_VISIBLE_DEVICES=none \
--container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$VENV_BIN \
SWEEP_INDUCTOR_CACHE=$INDUCTOR_CACHE \
SWEEP_LOGIN=$REPO/login.sh bash $REPO/sweep/slurm_launch.sh ${NAME}_$STAMP $SCRIPT"

GPU_ARGS=()
if [ "$GPU" != "0" ]; then
  CONSTRAINT='GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100|GPU_BRD:A100'
  if [ -n "$GPUS" ]; then
    CONSTRAINT=""
    IFS='|' read -r -a _brands <<< "$GPUS"
    for brand in "${_brands[@]}"; do
      [ -z "$brand" ] && continue
      CONSTRAINT="${CONSTRAINT:+$CONSTRAINT|}GPU_BRD:$brand"
    done
  fi
  GPU_ARGS=(--gres "gpu:$GPU" --constraint "$CONSTRAINT")
fi
[ -n "$EXCLUDE" ] && GPU_ARGS+=(--exclude "$EXCLUDE")

echo "[cli] submitting: $RUNNER $*"
echo "[cli] log: $LOG"

SB=(-p "$PARTITION" -A povejmo -c "$CPUS" --mem "$MEM" -t "$TIME"
    "${GPU_ARGS[@]}" -J "$NAME" -o "$LOG" --wrap "$WRAP")

# DEPENDENCY queues this behind another job ("afterok:12345"), which is what a
# step that has to follow a run it cannot wait for needs — an anneal fork owed by
# a trunk that is still going. Slurm then holds it whether or not the shell that
# submitted it is still alive, which a `WAIT=1` block does not.
[ -n "${DEPENDENCY:-}" ] && SB+=(--dependency "$DEPENDENCY")

if [ "$WAIT" = "0" ]; then
  # Submit and return. `--parsable` promises one bare job id and does not
  # deliver one here — the login banner is printed on stdout ahead of it — so
  # take the last line that is only digits, the way `chain.sh` does.
  JOB="$(sbatch --parsable "${SB[@]}" 2>&1 | grep -E '^[0-9]+$' | tail -n 1)"
  if [ -z "$JOB" ]; then
    echo "[cli] submission failed; no job id"
    exit 1
  fi
  echo "[cli] job $JOB (not waiting)"
  exit 0
fi

sbatch --wait "${SB[@]}" >/dev/null
rc=$?
echo "[cli] ---- tail of $LOG ----"
tail -n 60 "$LOG" 2>/dev/null
echo "[cli] exit=$rc"
exit $rc
