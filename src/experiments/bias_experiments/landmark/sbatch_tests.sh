#!/usr/bin/env bash
# =============================================================================
# sbatch_tests.sh — the landmark Phase 1 correctness gate, on ONE GPU.
# =============================================================================
# No training job is submitted until this is green. The properties it pins are
# the ones whose failure is INVISIBLE in a training curve — an absent,
# mis-indexed, or non-equivariant bias trains fine and reads as "landmark did not
# help", which is the conclusion the sweep exists to draw.
#
# Runs the landmark suite plus the two suites its plumbing could break: the
# collator contract and the flex/eager parity path.
#
#   ./src/experiments/bias_experiments/landmark/sbatch_tests.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

GPU="${GPU:-A100}"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

SCRIPT="$REPO/job_logs/landmark_tests_${STAMP}.sh"
LOG="$REPO/job_logs/landmark_tests_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  echo "rc=0"
  echo "python -m pytest tests/models/test_landmark_bias.py -v || rc=1"
  # Regression guard: the landmark column touches __getitem__ and _collate_features,
  # so anything that reads a batch is in the blast radius.
  echo "python -m pytest tests/utils/ -v -x -q || rc=1"
  echo "python -m pytest tests/models/test_linear_magnetic_bias.py -q || rc=1"
  echo "python -m pytest tests/models/test_flex_cpu.py -q || rc=1"
  echo "python -m pytest tests/experiments/test_graphqa_flags.py -q || rc=1"
  echo 'echo "GATE_EXIT=$rc"; exit $rc'
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh landmark_tests_${STAMP} $SCRIPT"

JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo --gres "gpu:${GPU}:1" -c 8 --mem 64G \
      -t "01:00:00" -J "landmark_tests" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }
echo "submitted landmark gate -> job $JOB"
echo "  log: ${LOG/\%j/$JOB}"
