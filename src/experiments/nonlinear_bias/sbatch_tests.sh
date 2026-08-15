#!/usr/bin/env bash
# =============================================================================
# sbatch_tests.sh — the NON_LINEAR_BIAS §7 correctness gate, on ONE GPU.
# =============================================================================
# No training job is submitted until this is green. The gate properties
# (permutation invariance under a DEGENERATE spectrum, the pooling AXIS, padded
# rows staying finite, zero-init inertness without a dead saddle, and the trunk
# receiving gradient across the checkpoint boundary) are the ones whose failure is
# INVISIBLE in a training curve: each produces a run that trains cleanly and reads
# as "the non-linear pooled head didn't help", which is exactly the conclusion
# the sweep exists to draw.
#
# A GPU is required for the flex suite — including this arm's own flex-vs-eager
# parity test, since the WebQSP sweep runs graph_attn_impl=flex — and for the
# bias-checkpoint recompute tests. The rest is CPU-light but runs here anyway so
# the gate is one artifact with one exit code.
#
#   ./src/experiments/nonlinear_bias/sbatch_tests.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

# ANY NVIDIA datacenter GPU on `frida`. The gate does not care which — it needs a
# CUDA device for the flex suite and nothing else — and pinning one brand can
# leave a 45-minute gate queued for hours behind a pool it has no reason to
# prefer. MI210 is excluded (AMD; the flex path is CUDA-only) and GH200 lives in
# a different partition.
GPU="${GPU:-A100|H100|B200|B300|L4}"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

SCRIPT="$REPO/job_logs/nonlinear_bias_tests_${STAMP}.sh"
LOG="$REPO/job_logs/nonlinear_bias_tests_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  echo "rc=0"
  # New suite first: it is the one that can fail for reasons the others cannot see.
  echo "python -m pytest tests/models/test_magnetic_nonlinear_bias.py -v || rc=1"
  # The suites this change could plausibly break. `_finalize` was refactored to a
  # module-level `finalize_node_bias` (shared with the new head), GraphAttentionBias
  # gained a kwarg, GraphContext gained a field, and GraphCausalLMMixin gained a
  # module — all of which every existing arm runs through.
  echo "python -m pytest tests/models/test_mixed_magnetic_bias.py -v || rc=1"
  echo "python -m pytest tests/models/test_linear_magnetic_bias.py -v || rc=1"
  echo "python -m pytest tests/models/test_gated_linear_magnetic_bias.py -v || rc=1"
  echo "python -m pytest tests/models/test_flex_cpu.py -v || rc=1"
  echo "python -m pytest tests/models/test_v2_ragged_magnetic_padding.py -v || rc=1"
  echo "python -m pytest tests/models/test_modeling_gtlm_llama_v2.py -q || rc=1"
  echo "python -m pytest tests/models/test_modeling_gtlm_bloom.py -q || rc=1"
  echo "python -m pytest tests/models/test_flex_attention.py -q || rc=1"
  echo "python -m pytest tests/models/test_bias_sharing.py -q || rc=1"
  echo "python -m pytest tests/models/test_graph_bias.py -q || rc=1"
  echo "python -m pytest tests/models/test_bias_regularization.py -q || rc=1"
  # Full sweep last: catches collateral damage anywhere else in the repo, which
  # matters because this change touched shared code (bias.py's finalize, three
  # model modules, GraphConfigMixin's exclusivity rule, two experiment configs)
  # and not only the two new classes.
  echo "python -m pytest tests -q --ignore=tests/models/test_magnetic_nonlinear_bias.py || rc=1"
  # The sweep configs, through the real parser and validator.
  echo "python -m src.experiments.nonlinear_bias.preflight || rc=1"
  echo 'echo "GATE_EXIT=$rc"'
  echo 'exit $rc'
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh nonlinear_bias_tests_${STAMP} $SCRIPT"

# A generic gres plus a GPU_BRD feature constraint, NOT `--gres gpu:A100:1`. The
# two are not equivalent: the gres form pins the gres TYPE, and the cluster's 80GB
# A100 node (ana) registers its gres as `A100_80GB`, so the type form silently
# excludes it and the job waits behind the two 40GB nodes. The feature
# `GPU_BRD:A100` is carried by all three.
CONSTRAINT="$(echo "$GPU" | sed 's/[^|]*/GPU_BRD:&/g')"
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 8 --mem 64G -t "00:45:00" \
      --gres "gpu:1" --constraint "$CONSTRAINT" \
      -J "nonlinear_bias_tests" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted gate -> job $JOB on $GPU"
echo "  log: ${LOG/\%j/$JOB}"
