#!/bin/bash
#SBATCH --partition=frida
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:2
#SBATCH --time=12:00:00
#SBATCH --job-name=graph_benchmark
#SBATCH --output=job_logs/benchmark_%j.out

# --- The script below uses srun to launch the container on the allocated compute node ---

srun \
  --container-image=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh \
  --container-mounts=/shared/workspace/povejmo:/shared/workspace/povejmo \
  --output=/shared/workspace/povejmo/graph_model/src/experiments/benchmarks/job_logs/benchmark_%j.out \
  bash -c "
    cd /shared/workspace/povejmo/graph_model || exit 1
    
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Execute the login script
    ./login.sh
    
    # Run the python benchmark script
    python3 -m src.experiments.benchmarks
  "