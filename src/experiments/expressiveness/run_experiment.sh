#!/usr/bin/env bash
# Run the full expressiveness experiment matrix for a SINGLE seed:
#   impl=v2-flex, k_hop in {0,1}, sizes in {500, 1000, 2000}.
#
# Each run logs to the wandb project "GraphLLM" and appends its hyperparameters +
# results (accuracy, ms/step, peak GB, token/block sparsity) to
# src/experiments/expressiveness/results/train_runs.jsonl.
#
# Usage (from the repo root):
#   bash src/experiments/expressiveness/run_experiment.sh <seed>
#   e.g.  bash src/experiments/expressiveness/run_experiment.sh 0
#
# Run seeds one at a time (0, then 1, then 2). The FIRST seed generates the
# datasets and compiles the flex kernels; later seeds reuse both from disk, so
# seed 0 is the slow one and seeds 1/2 are much faster.
set -euo pipefail

SEED="${1:?usage: run_experiment.sh <seed>}"

# ── knobs shared across all sizes (edit here) ────────────────────────────────
MAGNETIC_M=128       # eigenvectors kept; truncation is required at scale
TRAIN_SIZE=2000      # train graphs per size
EVAL_SIZE=500        # eval graphs per size
BIAS_LR=1e-3         # graph-bias LR. 1e-2 was tuned for 25-node graphs and is too
                     # hot at scale (the bias overshoots, then collapses to the
                     # 50/50 marginal). Lowered for the larger graphs.

# LoRA: adapt the (otherwise frozen) backbone so it can better *read* the bias
# signal. The token stream has no edge info and the model is prefix-permutation-
# invariant, so the bias remains the only source of graph signal. Set USE_LORA=0
# for a pure-frozen run (e.g. to isolate the LR/component/max_spd fixes first).
USE_LORA=1
LORA_R=8
LORA_LR=1e-5         # LoRA adapter LR (maps to --lr; the only "base" trainable params)

COMMON=(
  --impls v2-flex
  --k-hops 0,1
  --seeds "$SEED"
  --magnetic-m "$MAGNETIC_M"
  --bias-lr "$BIAS_LR"
  --train-dataset-size "$TRAIN_SIZE"
  --eval-dataset-size "$EVAL_SIZE"
  --num-epochs 12
  --report-to GraphLLM
)
if [ "$USE_LORA" = "1" ]; then
  COMMON+=(--lora --lora-r "$LORA_R" --lr "$LORA_LR")
fi

run() {  # run <num_nodes> <len_buckets> <node_buckets>
  echo -e "\n######## seed=$SEED  N=$1  (lora=$USE_LORA) ########"
  python3 -m src.experiments.expressiveness "${COMMON[@]}" \
    --num-nodes "$1" --len-buckets "$2" --node-buckets "$3"
}

run 500  640,768             512,640
run 1000 1408,1664,1792      1024,1280
run 2000 3200,3712,4096,4224 2048,2432

echo -e "\n######## seed=$SEED complete — results appended to src/experiments/expressiveness/results/train_runs.jsonl ########"
