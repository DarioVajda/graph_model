#!/usr/bin/env bash
# =============================================================================
# sbatch_phase0.sh — Phase 0 (P0a fit + P0b rank spectrum) on ONE GPU.
# =============================================================================
# Phase 0 is offline: it never trains, it re-derives each trained magnetic bias
# from `bias_parameters.pt` and asks how much of it a linear head reproduces, over
# the M-grid (LINEAR_BIAS.md §2.6, §4).
#
# GPU rather than CPU because the per-pair spectral features are a dense
# (P x M) @ (M x d_mag) contraction per graph per layer per M — small in memory,
# but 6 seeds x 16 layers x 5 M values of it is minutes on a card and the better
# part of an hour on the login node.
#
# A100 by default: the fit accumulates in float64 (R^2 near 1 is a small
# difference of large moments, so fp32 accumulation would eat the answer), and
# A100/H100 have real fp64 throughput where B200/B300 do not.
#
#   ./src/experiments/linear_bias/sbatch_phase0.sh            # both sweeps, 3 seeds
#   ./src/experiments/linear_bias/sbatch_phase0.sh smoke      # one seed, few graphs
#   GPU=H100 ./src/experiments/linear_bias/sbatch_phase0.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

MODE="${1:-full}"
GPU="${GPU:-A100}"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
OUT_DIR="$REPO/src/experiments/linear_bias/results"
mkdir -p "$REPO/job_logs" "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

# Which datasets to measure. Both by default; the single-dataset modes exist so a
# fixed failure can be re-run without redoing the half that already succeeded.
DO_WEBQSP=1; DO_CONTEXT=1
case "$MODE" in
  smoke)   SEEDS="0"; BATCHES=4;  BSZ=4; PAIRS=4000; SPECG=2; WALL="00:30:00" ;;
  full)    SEEDS="0 1 2"; BATCHES=16; BSZ=4; PAIRS=8000; SPECG=4; WALL="03:00:00" ;;
  webqsp)  SEEDS="0 1 2"; BATCHES=16; BSZ=4; PAIRS=8000; SPECG=4; WALL="03:00:00"
           DO_CONTEXT=0 ;;
  context) SEEDS="0 1 2"; BATCHES=16; BSZ=4; PAIRS=8000; SPECG=4; WALL="03:00:00"
           DO_WEBQSP=0 ;;
  *) echo "usage: $0 [smoke|full|webqsp|context]" >&2; exit 2 ;;
esac

SCRIPT="$REPO/job_logs/linear_bias_p0_${MODE}_${STAMP}.sh"
LOG="$REPO/job_logs/linear_bias_p0_${MODE}_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  [ "$DO_WEBQSP" = 1 ] && for s in $SEEDS; do
    # The recipe is replayed from the sweep's OWN job script, never retyped, so a
    # drifted recipe fails at parse time instead of silently measuring a
    # different model than the checkpoint was trained as.
    echo "python -m src.experiments.linear_bias.phase0 \\"
    echo "  --job \$(ls src/experiments/bias_sharing/results/002_webqsp_g_sweep/jobs/*_seed${s}_magnetic_groups0.sh) \\"
    echo "  --run-dir \$(ls -d checkpoints/kgqa/002_webqsp_g_sweep_*_seed${s}_magnetic_groups0) \\"
    echo "  --m-grid 8,16,32,64,128 --batches $BATCHES --batch-size $BSZ --pairs $PAIRS \\"
    echo "  --spectrum-graphs $SPECG --seed ${s} --out $OUT_DIR/p0_webqsp_seed${s}.json"
  done
  [ "$DO_CONTEXT" = 1 ] && for s in $SEEDS; do
    echo "python -m src.experiments.linear_bias.phase0 \\"
    echo "  --job \$(ls src/experiments/bias_sharing/results/003_context4k_g_sweep/jobs/*_seed${s}_magnetic_groups0.sh) \\"
    echo "  --run-dir \$(ls -d checkpoints/context/003_context4k_g_sweep_*_seed${s}_magnetic_groups0) \\"
    echo "  --m-grid 8,16,32,64,128 --batches $BATCHES --batch-size $BSZ --pairs $PAIRS \\"
    echo "  --spectrum-graphs $SPECG --seed ${s} --out $OUT_DIR/p0_context_seed${s}.json"
  done
  echo "python -m src.experiments.linear_bias.analyse --results $OUT_DIR"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh linear_bias_p0_${MODE}_${STAMP} $SCRIPT"

# Anchor both ends of the job-id match: the cluster MOTD's clock line
# ("15:15:06 up 48 days") also starts with digits, so an unanchored grep picks
# up the wrong number and the submission is then untracked.
JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem 128G -t "$WALL" \
      --gres "gpu:${GPU}:1" -J "linear_bias_p0_${MODE}" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

if [ -z "${JOB:-}" ]; then
  echo "SUBMISSION FAILED — no job id captured" >&2; exit 1
fi

echo "submitted phase0 $MODE -> job $JOB on $GPU"
echo "  log: ${LOG/\%j/$JOB}"
echo "  out: $OUT_DIR"
