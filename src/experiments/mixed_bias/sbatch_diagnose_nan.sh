#!/usr/bin/env bash
# =============================================================================
# sbatch_diagnose_nan.sh — reproduce the 020 arm-4 divergence with per-group
# gradient logging, on ONE GPU.
# =============================================================================
# The flags below are copied VERBATIM from the job script the diverged run
# actually executed:
#   results/020_context4k_mixed/jobs/020_context4k_mixed_0017_seed1_..._bias_lr0.02.sh
# Only three things differ, none of which touch the training math:
#   * the entrypoint is diagnose_nan (which wraps context/__main__ and observes);
#   * --runs-jsonl points at a diagnostic file, NOT the sweep's runs.jsonl;
#   * --run-name / --sweep-id are relabelled so this cannot be mistaken for a
#     sweep run when the results are aggregated.
#
# B200/B300 is requested deliberately: the divergence was observed on ixb2, and
# bf16 + flex numerics are not identical across GPU generations, so a different
# brand risks simply not reproducing it.
#
#   ./src/experiments/mixed_bias/sbatch_diagnose_nan.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
STAMP="$(date +%Y%m%d_%H%M%S)"
DIAGDIR="$REPO/src/experiments/mixed_bias/results/diagnostic"
mkdir -p "$REPO/job_logs" "$DIAGDIR"

SCRIPT="$REPO/job_logs/mixed_bias_diag_${STAMP}.sh"
LOG="$REPO/job_logs/mixed_bias_diag_${STAMP}_%j.out"

{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi --query-gpu=name --format=csv,noheader"
  echo "export DIAG_OUT=$DIAGDIR/grad_norms_${STAMP}.jsonl"
  echo "export DIAG_MAX_STEPS=${DIAG_MAX_STEPS:-700}"
  echo "python -m src.experiments.mixed_bias.diagnose_nan \
--magnetic-magnitude-dim 64 --magnetic-magnitude-repr-dim 256 \
--hop-counts 1,2,3,4 --fan-out 2 --node-counts 16,32,64,128 \
--token-counts 64,128,256,512 --max-train-len 4096 --n-train 16000 \
--n-dev 200 --n-test 200 --code-len 3 --id-pool 4096 --data-seed 42 \
--model-name meta-llama/Llama-3.2-1B --lora-r 64 --lora-dropout 0.15 --dtype bf16 \
--max-spd 8 --no-rrwp --magnetic-dim 128 --magnetic-m 128 --magnetic-q 0.25 \
--k-hop 0 --graph-attn-impl flex --compile-mode default --num-epochs 6 \
--batch-size 1 --accumulation-steps 8 --lr 0.0001 --gradient-checkpointing \
--num-workers 4 --eval-steps 500 --wandb-project GraphLLM --seed 1 \
--no-spd --no-magnetic --no-magnetic-linear --no-magnetic-magnitude \
--magnetic-hybrid --magnetic-m-collate 64 --bias-self-node --bias-lr 0.02 \
--runs-jsonl $DIAGDIR/diagnostic_runs.jsonl \
--run-name DIAGNOSTIC_arm4_hybrid_lr0.02_seed1 --sweep-id mixed_bias_diagnostic"
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh mixed_bias_diag_${STAMP} $SCRIPT"

JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem 128G -t "01:30:00" \
      --gres "gpu:1" --constraint "GPU_BRD:B200|GPU_BRD:B300" \
      -J "mixed_bias_diag" -o "$LOG" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

[ -z "${JOB:-}" ] && { echo "SUBMISSION FAILED — no job id captured" >&2; exit 1; }

echo "submitted diagnostic -> job $JOB"
echo "  slurm log : ${LOG/\%j/$JOB}"
echo "  grad norms: $DIAGDIR/grad_norms_${STAMP}.jsonl"
