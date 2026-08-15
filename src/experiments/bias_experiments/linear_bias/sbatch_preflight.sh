#!/usr/bin/env bash
# =============================================================================
# sbatch_preflight.sh — run preflight.py inside the container, on a compute node.
# =============================================================================
# preflight.py imports the real experiment entrypoints, and those pull in the
# data stack (networkx, torch_geometric, ...) which is not installed on the login
# node — so "just run it locally" fails at import, not at the check. It needs no
# GPU, so this asks for CPUs only and finishes in well under a minute.
#
#   ./src/experiments/bias_experiments/linear_bias/sbatch_preflight.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPO/job_logs"

# Must live under /shared: the session scratchpad in /tmp is node-local and the
# compute node would see an empty path.
SCRIPT="$REPO/job_logs/linear_bias_preflight_${STAMP}.sh"
LOG="$REPO/job_logs/linear_bias_preflight_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "python -m src.experiments.bias_experiments.linear_bias.preflight"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh linear_bias_preflight_${STAMP} $SCRIPT"

JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 8 --mem 32G -t "00:20:00" \
      -J "linear_bias_preflight" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted preflight -> job $JOB"
echo "  log: ${LOG/\%j/$JOB}"
