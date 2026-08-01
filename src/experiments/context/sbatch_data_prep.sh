#!/usr/bin/env bash
# =============================================================================
# sbatch_data_prep.sh — build the context experiment's .gtds tree on a compute node.
# =============================================================================
# The build is CPU-bound and takes ~25 min for the full grid, which does not
# belong on a login node.
#
# **No GPU is requested, on purpose.** The only GPU-eligible work is
# compute_shortest_path_distances + compute_magnetic_lap, which already default
# to use_gpu=True and would light up automatically — but at N <= 128 they run at
# 830 and 519 graphs/s even on CPU, i.e. ~13 s of a ~25 min build. The bottleneck
# is single-threaded tokenizer work in the text synthesis loop, which has no GPU
# path and does not benefit from extra cores either. Attaching a GPU would only
# add queue time. (Falls back to CPU automatically when no CUDA device is
# visible, so nothing breaks if a GPU is attached anyway.)
#
# Idempotent: a split already on disk is skipped, so re-running after an
# interruption resumes rather than restarting.
#
#   ./src/experiments/context/sbatch_data_prep.sh [CONFIG] [PARTITION]
#     CONFIG    defaults to configs/001_data_prep.jsonc
#     PARTITION defaults to frida. NB: sbatch prints a hint suggesting the `amd`
#               partition for GPU-less jobs — DO NOT follow it for this job. `amd`
#               is the MI210/ROCm node and cannot start this CUDA container:
#               "pyxis: container start failed", job dead in 27 s (tried
#               2026-07-29, job 119348). A CPU-only job still has to run
#               somewhere the container works, i.e. frida.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

CONFIG="${1:-src/experiments/context/configs/001_data_prep.jsonc}"
PARTITION="${2:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
mkdir -p "$REPO/job_logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT="$REPO/job_logs/ctx_dataprep_${STAMP}.sh"
LOG="$REPO/job_logs/ctx_dataprep_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "python -m sweep src.experiments.context $CONFIG"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh ctx_dataprep_${STAMP} $SCRIPT"

echo "[submit] data_prep ($CONFIG) -> $LOG"
sbatch -p "$PARTITION" -A povejmo -c 16 --mem 64G -t 04:00:00 \
       -J "ctx_dataprep" -o "$LOG" --wrap "$WRAP"
