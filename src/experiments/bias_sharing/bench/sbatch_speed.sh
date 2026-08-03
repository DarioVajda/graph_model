#!/usr/bin/env bash
# =============================================================================
# sbatch_speed.sh — run bench/speed.py on ONE GPU.
# =============================================================================
# Everything the benchmark compares must sit on one card: step time is only
# meaningful within a fixed device, and the whole point is arm-vs-arm deltas.
# B200 by default because that is what `002_webqsp_g_sweep` ran on, so the
# `--source webqsp` cell can be cross-checked against README §4's own s/it
# (§4 measures a Trainer "it" = 4 accumulated micro-steps + an optimizer step;
# this measures one micro-step).
#
#   ./src/experiments/bias_sharing/bench/sbatch_speed.sh                 # everything
#   ./src/experiments/bias_sharing/bench/sbatch_speed.sh smoke           # 2 min sanity
#   GPU=B300 ./src/experiments/bias_sharing/bench/sbatch_speed.sh
#
# Memory: the magnetic bias materializes a (B, N, N, magnetic_dim) intermediate,
# which at N=4096 / dim=128 is 4.3 GB in bf16 before the SiLU copy. 512 GB of host
# RAM and a whole 180 GB card are requested so an N=4096 arm fails on physics, not
# on a stingy allocation.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

MODE="${1:-full}"
GPU="${GPU:-B200}"
PARTITION="${PARTITION:-frida}"
CONTAINER="/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
OUT_DIR="$REPO/src/experiments/bias_sharing/results/bench"
mkdir -p "$REPO/job_logs" "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

case "$MODE" in
  smoke)
    # One small synthetic point, both endpoints plus the floor. Proves the whole
    # chain (config -> batch -> model -> timed step) before committing hours.
    ARGS="--source synth --nodes 512 --arms g0 g16 llm --n-batches 1 \
--warmup-passes 1 --passes 2 --out $OUT_DIR/smoke_${GPU}.jsonl"
    WALL="00:40:00"
    ;;
  synth)
    # --passes 8: token counts are sampled i.i.d. per seed, so batches at one N
    # can land in different length buckets (N=512 spans L 1536/2048, N=1024
    # 3072/4096, N=4096 12288/16384; only N=2048 is single-shape). Timings are
    # therefore read PER SHAPE, and 8 passes give each shape enough samples for
    # its own median instead of one pooled median that lands wherever the middle
    # step happens to fall.
    ARGS="--source synth --passes 8 --arms g0 g1 g2 g4 g8 g16 nobias llm llm_causal \
--out $OUT_DIR/speed.jsonl"
    WALL="08:00:00"
    ;;
  nocompile)
    # Same synthetic grid with plain torch.compile instead of the autotuning
    # default, to price the autotune wall time (over an hour at L=15360) against
    # what it buys per step. Appends to the SAME file: report.py keys rows by
    # compile mode and prints the two side by side. Run it AFTER the autotuned
    # sweep, on the same GPU type, or the comparison is confounded.
    ARGS="--source synth --flex-compile-mode default --passes 8 \
--arms g0 g1 g2 g4 g8 g16 llm --out $OUT_DIR/speed.jsonl"
    WALL="06:00:00"
    ;;
  nobias)
    # Just the isolation arm across the synthetic grid: GTLM with the graph mask
    # and flex kernel but no bias modules. Splits llm -> GTLM into "mask+kernel"
    # and "bias", which no other pair of arms separates. Same compile mode as the
    # autotuned grid it is compared against.
    ARGS="--source synth --arms nobias --out $OUT_DIR/speed.jsonl"
    WALL="03:00:00"
    ;;
  biasmodes)
    # Split the bias machinery into forward gather and backward atomic scatter,
    # which `nobias` cannot do: make_score_mod returns None when there is no bias,
    # so that arm removes the score_mod entirely rather than making it cheap.
    # Runs bench_isolation's --bias-mode {none,frozen,full} across our node counts
    # at int64 node_ids (production; the int32 win was never ported to
    # src/models/flex_kernel.py).
    ARGS="__BIASMODES__"
    WALL="03:00:00"
    ;;
  causal)
    # The `llm` floor is fed the same padded tensors GTLM gets, and an
    # attention_mask containing zeros makes transformers build an explicit 4D mask
    # — which disables sdpa's is_causal fast path, so the floor computes the full
    # square that flex's BlockMask skips. WebQSP is 61% padding, so this is not a
    # rounding error. `llm_causal` drops the mask to recover the fast path and
    # measure the best case a plain LLM actually reaches at these lengths.
    ARGS="--source synth webqsp graphqa context --arms llm_causal \
--out $OUT_DIR/speed.jsonl"
    WALL="02:00:00"
    ;;
  recheck512)
    # The N=512 default-mode row had two cells (g2 1.38x, g16 1.30x) far off the
    # ~1.02x every other cell shows, and both had first/median < 0.78 — the first
    # timed step was FASTER than the median, i.e. the run slowed down partway
    # through. That is contention, not a compile mode. Re-measure the whole row
    # (not just the two cells) so the replacement is internally consistent, with
    # more passes so a single perturbed step cannot move the median.
    ARGS="--source synth --nodes 512 --flex-compile-mode default --passes 24 \
--out $OUT_DIR/speed.jsonl"
    WALL="01:00:00"
    ;;
  real)
    ARGS="--source webqsp graphqa context --out $OUT_DIR/speed.jsonl"
    WALL="04:00:00"
    ;;
  full)
    ARGS="--source synth webqsp graphqa context --out $OUT_DIR/speed.jsonl"
    WALL="08:00:00"
    ;;
  *)
    echo "usage: $0 [smoke|synth|nocompile|nobias|biasmodes|causal|recheck512|real|full]" >&2; exit 2 ;;
esac

SCRIPT="$REPO/job_logs/bias_bench_${MODE}_${STAMP}.sh"
LOG="$REPO/job_logs/bias_bench_${MODE}_${STAMP}_%j.out"
{
  echo "#!/usr/bin/env bash"
  echo "set -x"
  echo "nvidia-smi"
  if [ "$MODE" = "biasmodes" ]; then
    echo "python -m src.experiments.bias_sharing.bench.bias_modes"
  else
    echo "python -m src.experiments.bias_sharing.bench.speed $ARGS"
  fi
} > "$SCRIPT"
chmod +x "$SCRIPT"

WRAP="srun --container-image=$CONTAINER --container-mounts=/shared:/shared \
env HOME=$HOME PYTHONUNBUFFERED=1 SWEEP_PROJECT_ROOT=$REPO SWEEP_VENV_BIN=$REPO/.venv/bin \
bash $REPO/sweep/slurm_launch.sh bias_bench_${MODE}_${STAMP} $SCRIPT"

# Anchor both ends of the job-id match: the cluster MOTD's clock line
# ("13:56:24 up 47 days") also starts with digits (see sbatch_context4k_data.sh).
# AFTER=<jobid> chains this run behind another, so two timing runs never share a
# node. Co-tenancy is exactly the confound these measurements cannot absorb.
DEP=()
[ -n "${AFTER:-}" ] && DEP=(--dependency="afterany:${AFTER}")

JOB=$(sbatch --parsable -p "$PARTITION" -A povejmo -c 16 --mem 256G -t "$WALL" \
      --gres "gpu:${GPU}:1" -J "bias_bench_${MODE}" -o "$LOG" "${DEP[@]}" --wrap "$WRAP" \
      | grep -oE '^[0-9]+$' | head -1)

echo "submitted $MODE -> job $JOB on $GPU"
echo "  log: ${LOG/\%j/$JOB}"
echo "  out: $OUT_DIR"
