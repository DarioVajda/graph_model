#!/bin/bash
#SBATCH --partition=frida
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:B200:1
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
    
    # Run the python benchmark training script
    # python3 -m src.experiments.benchmarks \
    #   --dataset_name=cora_hops2_neighbors20_random_abstracts_p0.2 \
    #   --num_epochs=20 \
    #   --lora_r=32 \
    #   --learning_rate=3e-4 \
    #   --bias_learning_rate=5e-2
    python3 -m src.experiments.benchmarks \
      --dataset_name=reddit_hops2_neighbors60_more_target_text \
      --num_epochs=10 \
      --lora_r=32 \
      --learning_rate=1e-4 \
      --bias_learning_rate=2e-2
    # python3 -m src.experiments.benchmarks \
    #   --dataset_name=ogbn-arxiv_hops2_neighbors20_random_abstracts_p0.2 \
    #   --num_epochs=3 \
    #   --lora_r=32 \
    #   --learning_rate=2e-4 \
    #   --bias_learning_rate=4e-2
  "