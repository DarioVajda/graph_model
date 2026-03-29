#!/bin/bash
#SBATCH --partition=frida
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:B200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=test_graph_model
#SBATCH --output=job_logs/test_benchmark_%j.out

container_image=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

srun \
  --container-image=${container_image} \
  --container-mounts=/shared/workspace/povejmo:/shared/workspace/povejmo \
  --output=job_logs/test_benchmark_%j.out \
  bash -c "
    cd /shared/workspace/povejmo/graph_model || exit 1
    pwd

    # USE WITH A100/H100/B200 GPUS:
    # Activate the virtual environment
    source .venv/bin/activate

    # Execute the login script
    ./login.sh
    
    # Run the python benchmark script
    python3 -m src.experiments.benchmarks.test \
    --checkpoint_path=./checkpoints/pubmed_hops2_neighbors30_lora_v5/checkpoint-960 \
    --dataset_name=pubmed_hops2_neighbors30
  "