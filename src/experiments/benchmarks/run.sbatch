#!/bin/bash
#SBATCH --partition=frida
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=train_graph_model
#SBATCH --output=job_logs/benchmark_%j.out

container_image=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

srun \
  --container-image=${container_image} \
  --container-mounts=/shared/workspace/povejmo:/shared/workspace/povejmo \
  --output=job_logs/benchmark_%j.out \
  bash -c "
    cd /shared/workspace/povejmo/graph_model || exit 1
    pwd

    # USE WITH A100/H100 GPUS:
    # Activate the virtual environment
    source .venv/bin/activate

    # Execute the login script
    ./login.sh
    
    # Run the python benchmark script
    python3 -m src.experiments.benchmarks --dataset_name=pubmed_hops2_neighbors30 --num_epochs=4 --lora_r=32
  "