# Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning

### Anonymous Author, University Name


## This Codebase

This is a repository with all of the code used for this research project — it includes low-level architecture implementations, data processing, experiments, and evaluation scripts. 

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
      │   ├── our_tests/                # Family Tree and Knowledge Graph QA experiments
      │   └── permutation_equivariance/ # Code for permutation the equivariance test.
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

The dependencies are managed with `pip-tools`, so that the process of tracking the required libraries is easier.

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