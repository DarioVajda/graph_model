#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — run pytest on a COMPUTE node, never on the login node.
# =============================================================================
# Submits one CPU-only sbatch job that runs `python -m pytest <args>` inside the
# project container (the same launcher the sweep runner uses, so the venv, HOME
# and login script are set up identically), and BLOCKS until it finishes so the
# exit code is the test result. The full pytest output lands in
# src/generalist/results/test_logs/<stamp>.out; the tail is echoed here.
#
#   src/generalist/tools/run_tests.sh tests/generalist/test_schema.py -q
#   src/generalist/tools/run_tests.sh tests/generalist -x -q
#   GPU=1 src/generalist/tools/run_tests.sh tests/generalist/test_smoke_gpu.py -q -s
#
# Env overrides: PARTITION (frida), CPUS (8), MEM (32G), TIME (01:00:00),
# GPU (0 -> CPU-only; 1 -> one GPU of any Blackwell/H100/A100 class).
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
PARTITION="${PARTITION:-frida}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-01:00:00}"
GPU="${GPU:-0}"

LOG_DIR="$REPO/src/generalist/results/test_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
SCRIPT="$LOG_DIR/$STAMP.sh"
LOG="$LOG_DIR/$STAMP.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "cd $REPO"
  echo "python -m pytest $* ; rc=\$?"
  echo "echo GATE_EXIT=\$rc; exit \$rc"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
SWEEP_LOGIN=$REPO/login.sh bash $REPO/sweep/slurm_launch.sh gen_tests_$STAMP $SCRIPT"

GPU_ARGS=()
if [ "$GPU" != "0" ]; then
  GPU_ARGS=(--gres "gpu:$GPU" --constraint 'GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100|GPU_BRD:A100')
fi

echo "[tests] submitting: pytest $*"
echo "[tests] log: $LOG"
sbatch --wait -p "$PARTITION" -A povejmo -c "$CPUS" --mem "$MEM" -t "$TIME" \
       "${GPU_ARGS[@]}" -J "gen_tests" -o "$LOG" --wrap "$WRAP" >/dev/null
rc=$?
echo "[tests] ---- tail of $LOG ----"
tail -n 60 "$LOG" 2>/dev/null
echo "[tests] exit=$rc"
exit $rc
