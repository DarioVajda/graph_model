#!/usr/bin/env bash
# =============================================================================
# sbatch_tests.sh — the MIXED_BIAS Phase 1 correctness gate, on ONE GPU.
# =============================================================================
# MIXED_BIAS.md §4.2: no Phase 2 training job is submitted until this is green.
# The gate properties (permutation invariance under a DEGENERATE spectrum, no
# silent no-op, zero-init inertness without a dead saddle, backward
# compatibility) are the ones whose failure is INVISIBLE in a training curve — a
# run with an absent or mis-grouped bias trains fine and reads as "the magnitude
# channel didn't help", which is exactly the conclusion the sweep exists to draw.
#
# A GPU is required for the flex suite (flex-vs-eager parity and bias-checkpoint
# recompute-safety have no CPU path); the rest is CPU-light but runs here anyway
# so the gate is one artifact with one exit code.
#
#   ./src/experiments/mixed_bias/sbatch_tests.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

# ANY NVIDIA datacenter GPU on `frida`. The gate does not care which — it needs a
# CUDA device for the flex suite and nothing else — and the cluster is routinely
# saturated with multi-day jobs, so pinning one brand can leave a 45-minute gate
# queued for hours behind a pool it has no reason to prefer. MI210 is excluded
# (AMD; the flex kernel path is CUDA-only) and GH200 lives in a different
# partition.
GPU="${GPU:-A100|H100|B200|B300|L4}"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

SCRIPT="$REPO/job_logs/mixed_bias_tests_${STAMP}.sh"
LOG="$REPO/job_logs/mixed_bias_tests_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  echo "rc=0"
  # New suite first: it is the one that can fail for reasons the others cannot see.
  echo "python -m pytest tests/models/test_mixed_magnetic_bias.py -v || rc=1"
  # The predecessor gate: the hybrid INHERITS its phase channel from
  # LinearMagneticBias, and _phase_bias / _phase_factors were split out of it, so
  # a regression there is a regression in arm 4.
  echo "python -m pytest tests/models/test_linear_magnetic_bias.py -v || rc=1"
  # Priority order from MIXED_BIAS.md §4.2.
  echo "python -m pytest tests/models/test_flex_cpu.py -v || rc=1"
  echo "python -m pytest tests/models/test_v2_ragged_magnetic_padding.py -v || rc=1"
  echo "python -m pytest tests/models/test_modeling_gtlm_llama_v2.py -q || rc=1"
  echo "python -m pytest tests/models/test_flex_attention.py -q || rc=1"
  echo "python -m pytest tests/models/test_graph_bias.py -q || rc=1"
  echo "python -m pytest tests/models/test_bias_regularization.py -q || rc=1"
  echo "python -m pytest tests/experiments/test_graphqa_flags.py -q || rc=1"
  # Full sweep last: catches collateral damage anywhere else in the repo, which
  # matters because this change touched shared code (GraphConfigMixin's
  # exclusivity rule, LinearMagneticBias's forward, three experiment configs) and
  # not only the two new classes.
  echo "python -m pytest tests -q --ignore=tests/models/test_mixed_magnetic_bias.py || rc=1"
  echo 'echo "GATE_EXIT=$rc"'
  echo 'exit $rc'
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh mixed_bias_tests_${STAMP} $SCRIPT"

# A generic gres plus a GPU_BRD feature constraint, NOT `--gres gpu:A100:1`.
# The two are not equivalent: the gres form pins the gres TYPE, and the cluster's
# 80GB A100 node (ana) registers its gres as `A100_80GB`, so the type form
# silently excludes it and the job waits behind the two 40GB nodes. The feature
# `GPU_BRD:A100` is carried by all three. This is the same rendering the sweep
# configs get from `"gpus": ["A100"]` (sweep/execute.py:_gpu_args).
#
# Small and short on purpose: the whole suite is ~1 min of CPU plus a few minutes
# of flex compilation, so 45 min / 8 CPUs / 64G is generous — and a small, short
# job is the one Slurm's backfill can slot into a gap without delaying the
# multi-day jobs ahead of it.
CONSTRAINT="$(echo "$GPU" | sed 's/[^|]*/GPU_BRD:&/g')"
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 8 --mem 64G -t "00:45:00" \
      --gres "gpu:1" --constraint "$CONSTRAINT" \
      -J "mixed_bias_tests" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted gate -> job $JOB on $GPU"
echo "  log: ${LOG/\%j/$JOB}"
