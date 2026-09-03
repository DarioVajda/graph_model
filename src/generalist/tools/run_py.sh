#!/usr/bin/env bash
# =============================================================================
# run_py.sh — run one project script on a COMPUTE node, in the project container.
# =============================================================================
# The sibling of run_cli.sh for things that are not a `python -m src.generalist`
# mode: a one-off analysis, a comparison against another campaign's artifacts, a
# measurement that wants RDKit or torch. Same container, same blocking sbatch,
# same log directory.
#
#   src/generalist/tools/run_py.sh src/generalist/tools/compare_bace_split.py
#   GPU=1 src/generalist/tools/run_py.sh path/to/script.py --flag value
#
# Env overrides: PARTITION (frida), CPUS (16), MEM (64G), TIME (02:00:00),
# GPU (0 -> CPU-only), NAME, INDUCTOR_CACHE — all as in run_cli.sh.
#
# Nothing here belongs in a campaign's results: a script run this way is a
# measurement, and where its answer matters it goes into the write-up, not into
# a log nobody reads again.
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
NAME="${NAME:-gen_py}"
SCRIPT="$LOG_DIR/$STAMP.sh"
LOG="$LOG_DIR/$STAMP.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "cd $REPO"
  # `python -m src.generalist` puts the repo root on sys.path for free;
  # `python path/to/script.py` puts the *script's* directory there instead, so
  # `import src.…` fails from a tools/ script. PYTHONPATH is what makes the two
  # entry points see the same tree.
  echo "PYTHONPATH=$REPO python $* ; rc=\$?"
  echo "echo PY_EXIT=\$rc; exit \$rc"
} > "$SCRIPT.tmp.$$"
chmod +x "$SCRIPT.tmp.$$"
mv -f "$SCRIPT.tmp.$$" "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
SWEEP_INDUCTOR_CACHE=$INDUCTOR_CACHE \
SWEEP_LOGIN=$REPO/login.sh bash $REPO/sweep/slurm_launch.sh ${NAME}_$STAMP $SCRIPT"

GPU_ARGS=()
if [ "$GPU" != "0" ]; then
  GPU_ARGS=(--gres "gpu:$GPU" --constraint 'GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100|GPU_BRD:A100')
fi

echo "[py] submitting: python $*"
echo "[py] log: $LOG"
sbatch --wait -p "$PARTITION" -A povejmo -c "$CPUS" --mem "$MEM" -t "$TIME" \
       "${GPU_ARGS[@]}" -J "$NAME" -o "$LOG" --wrap "$WRAP" >/dev/null
rc=$?
echo "[py] ---- tail of $LOG ----"
tail -n 80 "$LOG" 2>/dev/null
echo "[py] exit=$rc"
exit $rc
