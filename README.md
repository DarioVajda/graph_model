# Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning

### Dario Vajda

**Preprint:** [arXiv:2605.10247](https://arxiv.org/abs/2605.10247)

## Abstract

Current state-of-the-art approaches for graph processing with LLMs typically rely on multi-step pipelines with Graph Neural Network (GNN) encoders that compress rich textual attributes into solitary tokens. This work introduces **GTLM** (Graph Transformer Language Model), enabling pretrained LLMs to process graph topologies while fully preserving semantic information. By injecting graph-aware attention biases directly into LLM attention modules, GTLM adds only 0.015% additional parameters while providing theoretical guarantees of node permutation equivariance and exact backward compatibility with the pretrained base model. A 1B-parameter GTLM matches or exceeds the performance of 7B-parameter state-of-the-art models on Text-Attributed Graph benchmarks and significantly surpasses baselines on GraphQA tasks. Analysis of the learned attention heads shows that GTLM implicitly learns to simulate message passing, enabling true algorithmic reasoning within LLMs.

## This Codebase

This repository contains all of the code used for this research project — it includes low-level architecture implementations, data processing, experiments, and evaluation scripts. 

File structure:
```
graph_model/
  └── plots/                            # Visualizations and plots generated during data analysis.
  └── src/                              # The main source code directory.
      ├── experiments/                  # specific experiment setups and configurations.
      │   ├── backward_compatibility/   # Code for the backward compatibility test.
      │   ├── benchmarks/               # Scripts used for TAG benchmark training and evauluation.
      │   ├── expressiveness/           # Code for expressiveness analysis (also used for the message passing visualisation)
      │   ├── graphqa/                  # Code for GraphQA benchmark experiments.
      │   ├── graphqa_mag_khop/         # Ablation: effect of capping magnetic eigenvectors (M) and attention hop distance (K).
      │   ├── our_tests/                # Family Tree and Knowledge Graph QA experiments
      │   ├── permutation_equivariance/ # Code for permutation the equivariance test.
      │   └── template/                 # Minimal experiment template — copy this to start a new experiment.
      ├── models/                       # Implementation of the custom architecture.
      └── utils/                        # general-purpose utility functions and helper scripts.
  ├── .gitignore                        # specifies intentionally untracked files to ignore.
  ├── hf_login.sh                       # Helper script for logging into the Hugging Face Hub. (private)
  └── wandb_login.sh                    # Helper script for logging into Weights & Biases (WandB). (private)
  ├── login.sh                          # A script running hf_login and wandb_login.
  ├── README.md                         # Project documentation (this file).
  ├── requirements.in                   # High-level specification of project dependencies.
  ├── requirements.txt                  # A generated and locked file of exact project dependencies.
```

## Working with GTLM

The dependencies are managed with `pip-tools`, so that the process of tracking the required libraries is easier (which are saved in `requirements.in`).

### Environment setup:
```
python -m venv .venv        # Create a new python environment
source .venv/bin/activate   # Activate the environment
pip install pip-tools       # Download pip-tools
pip-compile                 # Compile the full requirements.txt file
pip-sync                    # Install all requirements
```

From now on, we will assume that each program is ran inside of the `.venv` environment.

The instructions for each individual experiment can be found in the appropriate directories inside `src/experiments/`, while the core architectural logic is implemented in `src/models/`.

## Starting a New Experiment

Copy `src/experiments/template/` to a new directory and edit three files:

**`load_data.py`** — build your dataset. Each graph is a NetworkX `DiGraph` with:
- A `text` attribute on every node (the node's natural-language description)
- A `prompt_node` key in `graph.graph` (the index of the node whose tokens the model generates)
- The prompt node's text must contain both the question and the expected answer

Pass a list of such graphs to `TextGraphDataset`, call the feature-computation methods you need (`compute_shortest_path_distances`, `compute_rrwp`, `compute_magnetic_lap`, …), tokenize, then call `compute_labels` to mask the question tokens to `-100` so the model is supervised on the answer only. Return `(train_dataset, eval_dataset, test_dataset)`.

**`__main__.py`** — configure `BIAS_PARAMS` to choose which graph attention biases to use (SPD, RRWP, magnetic Laplacian, …), set sensible defaults for the CLI arguments, and update `EXPERIMENT_NAME`. Everything else can stay as-is.

**`test.py`** — update the `load_data` call to match your new `load_data.py` signature.

### Training

```bash
python -m src.experiments.<your_experiment> \
    --num_epochs 20 \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --lora_r 16 \
    --k_hop 0 \
    --wandb_project GraphLLM
```

Checkpoints are saved under `checkpoints/<experiment_name>/<run_name>/`.

### Evaluation

```bash
python -m src.experiments.<your_experiment>.test \
    --checkpoint_path checkpoints/<experiment_name>/<run_name>/checkpoint-<step>
```

Results are appended to `src/experiments/<your_experiment>/results.json`.

## Citation

This repository contains the code for the following preprint. Ongoing work extends GTLM with K-hop attention masking, a Flex Attention implementation, and thorough evaluation on Knowledge Graph question answering and relational deep learning.

```bibtex
@misc{vajda2026teachingllmsgraphsunifying,
      title={Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning}, 
      author={Dario Vajda},
      year={2026},
      eprint={2605.10247},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.10247}, 
}
```
