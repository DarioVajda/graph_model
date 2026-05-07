#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn[01-06,08-10]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=3b_cora
#SBATCH --output=job_logs/benchmark_%j.out

cd ~/graph_model
pwd

# USE WITH A100/H100 GPUS:
# Activate the virtual environment
source .venv/bin/activate

# Execute the login script
source ./login.sh

# Run the python benchmark training script three times
for ((i=0; i<3; i++)); do
  echo "Starting training run $i..."

  # python3 -m src.experiments.benchmarks \
  #   --model_name=meta-llama/Llama-3.2-3B \
  #   --dataset_name=cora_hops2_neighbors60_target_abstract \
  #   --num_epochs=15 \
  #   --lora_r=64 \
  #   --learning_rate=6e-5 \
  #   --bias_learning_rate=2e-2
  python3 -m src.experiments.benchmarks \
    --model_name=meta-llama/Llama-3.2-3B \
    --dataset_name=pubmed_hops2_neighbors30 \
    --num_epochs=4 \
    --lora_r=64 \
    --learning_rate=6e-5 \
    --bias_learning_rate=4e-2 \
    --eval_every=80
  # python3 -m src.experiments.benchmarks \
  #   --model_name=meta-llama/Llama-3.2-1B \
  #   --dataset_name=ogbn-arxiv_hops2_neighbors30 \
  #   --num_epochs=1 \
  #   --lora_r=64 \
  #   --learning_rate=2e-4 \
  #   --bias_learning_rate=4e-2
  # python3 -m src.experiments.benchmarks \
  #   --model_name=meta-llama/Llama-3.2-1B \
  #   --dataset_name=reddit_hops2_neighbors30_truncated_text_128 \
  #   --num_epochs=10 \
  #   --lora_r=16 \
  #   --learning_rate=3e-5 \
  #   --bias_learning_rate=5e-3
done