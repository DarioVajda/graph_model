#!/bin/bash
#SBATCH --job-name=kgqa-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=src/experiments/kgqa/logs/train_%j.out
#SBATCH --error=src/experiments/kgqa/logs/train_%j.err

# Real KGQA (SR-WebQSP) training run on the new GTLM stack.
# Defaults: 5 epochs, batch_size=2, accum=4, lora_r=16, k_hop=2, flex/bf16,
# full-dev generative eval; best checkpoint selected on eval_f1.
#
# Submit from the repo root:   sbatch src/experiments/kgqa/train.sbatch
# HF + wandb creds are already cached (~/.cache/huggingface/token, ~/.netrc);
# torch 2.6.0+cu124 bundles its CUDA runtime, so no `module load` is required.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p src/experiments/kgqa/logs

nvidia-smi -L || { echo "No GPU visible on this node" >&2; exit 1; }

exec .venv/bin/python -m src.experiments.kgqa --graph_attn_impl flex
