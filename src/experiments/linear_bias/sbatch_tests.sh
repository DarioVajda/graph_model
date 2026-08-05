#!/usr/bin/env bash
# =============================================================================
# sbatch_tests.sh — the Phase 1 correctness gate, on ONE GPU.
# =============================================================================
# LINEAR_BIAS.md §5.2: no Phase 2 training job is submitted until this is green.
# The four gate properties (permutation invariance, no-silent-no-op, zero-init
# inertness, backward compatibility) are the ones whose failure is INVISIBLE in a
# training curve — a run with a mis-indexed or entirely absent bias trains fine
# and reads as "linearization hurt", which is the conclusion the sweep exists to
# draw. Everything else here guards the same blast radius more cheaply.
#
# A GPU is required for the flex suite (flex-vs-eager parity and bias-checkpoint
# recompute-safety have no CPU path); the rest is CPU-light but runs here anyway
# so the gate is one artifact with one exit code.
#
#   ./src/experiments/linear_bias/sbatch_tests.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

GPU="${GPU:-A100}"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

SCRIPT="$REPO/job_logs/linear_bias_tests_${STAMP}.sh"
LOG="$REPO/job_logs/linear_bias_tests_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  echo "rc=0"
  # New suite first: it is the one that can fail for reasons the others cannot see.
  echo "python -m pytest tests/models/test_linear_magnetic_bias.py -v || rc=1"
  # Priority order from LINEAR_BIAS.md §5.3.
  echo "python -m pytest tests/models/test_flex_cpu.py -v || rc=1"
  echo "python -m pytest tests/models/test_v2_ragged_magnetic_padding.py -v || rc=1"
  echo "python -m pytest tests/models/test_modeling_gtlm_llama_v2.py -q || rc=1"
  echo "python -m pytest tests/models/test_flex_attention.py -q || rc=1"
  echo "python -m pytest tests/models/test_graph_bias.py -q || rc=1"
  echo "python -m pytest tests/utils/test_collator_bucketing.py -q || rc=1"
  echo "python -m pytest tests/models/test_bias_regularization.py -q || rc=1"
  echo "python -m pytest tests/models/test_bias_sharing.py -q || rc=1"
  echo "python -m pytest tests/experiments/test_magnetic_groups_cli.py -q || rc=1"
  # Full sweep last: catches collateral damage anywhere else in the repo, which
  # matters because this change touched shared code (_folded_spectral's split
  # point) and not only the new class.
  echo "python -m pytest tests -q -x --ignore=tests/models/test_linear_magnetic_bias.py || rc=1"
  echo 'echo "GATE_EXIT=$rc"'
  echo 'exit $rc'
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh linear_bias_tests_${STAMP} $SCRIPT"

JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem 128G -t "02:00:00" \
      --gres "gpu:${GPU}:1" -J "linear_bias_tests" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted gate -> job $JOB on $GPU"
echo "  log: ${LOG/\%j/$JOB}"
