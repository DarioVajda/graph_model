#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn[01-06,08-10]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=graphqa
#SBATCH --output=job_logs/graph_qa_%j.out

cd ~/graph_model
pwd

# USE WITH A100/H100 GPUS:
# Activate the virtual environment
source .venv/bin/activate

# Execute the login script
source ./login.sh

# Run the python benchmark script
python3 -m src.experiments.graphqa.full_experiment