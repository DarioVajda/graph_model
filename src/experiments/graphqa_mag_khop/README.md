# graphqa_mag_khop

## Purpose

This experiment evaluates how capping two structural hyperparameters affects model performance on GraphQA tasks:

- **M** (`--magnetic_m`): the number of magnetic Laplacian eigenvectors used per graph. Lower M means a coarser spectral representation of graph structure.
- **K** (`--k_hop`): the maximum hop distance for the attention mask. Nodes beyond K hops from each other are blocked from attending to one another.

The core question is whether restricting either quantity — forcing the model to work with less spectral information or a more local attention field — helps, hurts, or has no significant effect on task accuracy.

## Setup

The model uses **only the magnetic Laplacian bias** (SPD, RRWP, and other biases are disabled). This isolates the contribution of the magnetic bias and makes the M/K sweep interpretable.

Datasets are loaded from the pre-processed GraphQA datasets (`src/experiments/graphqa/processed_datasets`). Eigenvector truncation to M is applied at collation time — no re-processing required.

## Usage

```bash
python -m src.experiments.graphqa_mag_khop \
    --task shortest_path \
    --k_hop 2 \
    --magnetic_m 8
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--task` | `shortest_path` | GraphQA task to train on |
| `--graph_type` | `standard` | `standard` or `incidence` |
| `--k_hop` | `0` | K-hop attention gate (0 = disabled) |
| `--magnetic_m` | `0` | Max eigenvectors to keep (0 = keep all) |
| `--lora_r` | `16` | LoRA rank (0 = no LoRA) |
| `--wandb_project` | `GraphLLM` | WandB project (`none` to disable) |

WandB runs are named `GraphQA_{task}_K={K}_M={M}` (with `_incidence` suffix when using the incidence graph type).
