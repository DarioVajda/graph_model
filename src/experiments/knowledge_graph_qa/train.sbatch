#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn[01-06,08-10]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --job-name=train_graph_model
#SBATCH --output=job_logs/benchmark_%j.out

cd /d/hpc/home/dv70648/graph_model
pwd

# USE WITH A100/H100 GPUS:
# Activate the virtual environment
source .venv/bin/activate

# Execute the login script
source ./login.sh

# Run the python custom task training script
python3 -m src.experiments.knowledge_graph_qa \
  --dataset_name=kg_qa \
  --model_name=meta-llama/Llama-3.2-1B \
  --lora_r=32 \
  --batch_size=4 \
  --accumulation_steps=4 \
  --learning_rate=5e-4 \
  --bias_learning_rate=3e-2 \
  --num_epochs=10
# python3 -m src.experiments.knowledge_graph_qa.train_llm_baseline \
#   --dataset_name=kg_qa \
#   --model_name=meta-llama/Llama-3.2-1B \
#   --lora_r=32 \
#   --batch_size=2 \
#   --accumulation_steps=8 \
#   --learning_rate=5e-4 \
#   --num_epochs=10
  