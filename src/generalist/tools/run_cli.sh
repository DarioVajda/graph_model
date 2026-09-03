#!/usr/bin/env bash
# =============================================================================
# run_cli.sh — run one `python -m src.generalist ...` on a COMPUTE node.
# =============================================================================
# The single-job counterpart to chain.sh: submits one sbatch job that runs the
# harness inside the project container and BLOCKS until it finishes, so the exit
# code is the command's. Output lands in
# src/generalist/results/job_logs/<stamp>.out and the tail is echoed here.
#
#   src/generalist/tools/run_cli.sh data_prep --config src/generalist/configs/000_smoke.jsonc
#   GPU=1 src/generalist/tools/run_cli.sh eval --checkpoint <ckpt> --config <cfg>
#
# Env overrides: PARTITION (frida), CPUS (16), MEM (64G), TIME (02:00:00),
# GPU (0 -> CPU-only; 1 -> one GPU of any Blackwell/H100/A100 class), NAME,
# INDUCTOR_CACHE.
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
PARTITION="${PARTITION:-frida}"
CPUS="${CPUS:-16}"
MEM="${MEM:-64G}"
TIME="${TIME:-02:00:00}"
GPU="${GPU:-0}"
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

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "cd $REPO"
  echo "python -m src.generalist $* ; rc=\$?"
  echo "echo CLI_EXIT=\$rc; exit \$rc"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
SWEEP_INDUCTOR_CACHE=$INDUCTOR_CACHE \
SWEEP_LOGIN=$REPO/login.sh bash $REPO/sweep/slurm_launch.sh ${NAME}_$STAMP $SCRIPT"

GPU_ARGS=()
if [ "$GPU" != "0" ]; then
  GPU_ARGS=(--gres "gpu:$GPU" --constraint 'GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100|GPU_BRD:A100')
fi

echo "[cli] submitting: python -m src.generalist $*"
echo "[cli] log: $LOG"
sbatch --wait -p "$PARTITION" -A povejmo -c "$CPUS" --mem "$MEM" -t "$TIME" \
       "${GPU_ARGS[@]}" -J "$NAME" -o "$LOG" --wrap "$WRAP" >/dev/null
rc=$?
echo "[cli] ---- tail of $LOG ----"
tail -n 60 "$LOG" 2>/dev/null
echo "[cli] exit=$rc"
exit $rc
