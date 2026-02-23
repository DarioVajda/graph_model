from ...utils import set_wandb_project, GraphTrainer, TextGraphDataset, GraphCollator
from ...models.llama_attn_bias import GraphLlamaForCausalLM

from .data_gen import create_and_save_dataset, dataset_path_and_size

import torch
import os
import random
from transformers import TrainingArguments

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(model_name, device, bias_type):
    model = GraphLlamaForCausalLM.from_pretrained(model_name, bias_type=bias_type)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

def select_active_params(model, active_params=None):
    """
    Sets requires_grad=True for parameters whose names contain any of the substrings in active_params.
    If active_params is None, all parameters are frozen (requires_grad=False).
    If active_params is "all", all parameters are set to requires_grad=True.
    """
    if active_params is None:
        active_params = [] # freeze all parameters
    elif active_params == "all":
        active_params = [""] # activate all parameters by matching any substring (empty string matches all names)

    for name, param in model.named_parameters():
        if any(active_param in name for active_param in active_params):
            param.requires_grad = True
        else:
            param.requires_grad = False

def training_run(model, train_datasetm, eval_dataset, collator, run_name, num_epochs=3, batch_size=8, learning_rate=5e-5):
    training_args = TrainingArguments(
        # Basic training arguments:
        num_train_epochs=num_epochs,                            # Total number of training epochs to perform
        output_dir=f"./checkpoints/{run_name}",                 # Directory to save checkpoints and logs
        logging_steps=4,                                        # Log training metrics every 5 steps
        per_device_train_batch_size=8,                          # Batch size per device during training
        gradient_accumulation_steps=4,                          # Number of steps to accumulate gradients before performing an optimizer step
        gradient_checkpointing=False,                           # Gradient checkpointing to save memory

        # Evaluation arguments:
        evaluation_strategy="steps",                            # Evaluate every eval_steps during training
        eval_steps=50,                                          # Number of steps between evaluations
        load_best_model_at_end=True,                            # Load the best model at the end of training
        metric_for_best_model="eval_loss",                      # Metric to use for selecting the best model
        greater_is_better=False,                                # Lower eval_loss is better
        save_total_limit=5,                                     # Maximum number of checkpoints to store

        # WandB logging:
        report_to="wandb",                                      # Report training metrics to Weights & Biases
        run_name=run_name,                                      # Name of the WandB run for better organization
        
        # Learning rate scheduler:
        learning_rate=learning_rate,                            # The initial learning rate for Adam
        lr_scheduler_type="cosine_with_min_lr",                 # Type of learning rate scheduler to use
        lr_scheduler_kwargs={"min_lr": learning_rate/10},       # Additional arguments for the learning rate scheduler
        warmup_steps=STEPS_PER_EPOCH,                           # Number of steps for the warmup phase (when the learning rate is increasing linearly)
        weight_decay=0.1,                                       # Weight decay to apply (if not zero)
    )

    # initialize using the custom class
    trainer = GraphTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    BIAS_TYPE = "combined" # options: "none", "spd", "spectral", "combined"
    TRAIN_DATASET_SIZE = 1_000
    EVAL_DATASET_SIZE = 200
    MODEL_NAME = "meta-llama/Llama-3.2-1B"

    set_wandb_project("GraphLLM")
    device = get_device()

    model = init_model(model_name=MODEL_NAME, device=device, bias_type=BIAS_TYPE)

    # --------------------------------------------------------------------------
    #region ----------------------- LOAD DATASET -------------------------------
    # --------------------------------------------------------------------------
    train_dataset_path, _ = dataset_path_and_size(TRAIN_DATASET_SIZE)
    if not os.path.exists(train_dataset_path):
        print(f"Dataset not found at {train_dataset_path}. Creating new dataset...")
        create_and_save_dataset(dataset_size=TRAIN_DATASET_SIZE, min_nodes=10, max_nodes=20, spectral_dims=16, model_name=MODEL_NAME)

    train_dataset = TextGraphDataset.load(train_dataset_path)

    eval_dataset_path, _ = dataset_path_and_size(EVAL_DATASET_SIZE)
    if not os.path.exists(eval_dataset_path):
        print(f"Dataset not found at {eval_dataset_path}. Creating new dataset...")
        create_and_save_dataset(dataset_size=EVAL_DATASET_SIZE, min_nodes=10, max_nodes=20, spectral_dims=16, model_name=MODEL_NAME)

    eval_dataset = TextGraphDataset.load(eval_dataset_path)
    collator = GraphCollator()
    #endregion
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #region -------- FINE TUNE ONLY THE CUSTOM BIAS-RELATED PARAMETERS ---------
    # --------------------------------------------------------------------------
    print("Fine-tuning only the custom bias-related parameters...")
    select_active_params(model, active_params=["spd_weights", "spectral_weights"])



    #endregion
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #region ------------------- FINE TUNE ALL PARAMETERS -----------------------
    # --------------------------------------------------------------------------
    print("Fine-tuning all parameters...")
    select_active_params(model, active_params="all")

    

    #endregion
    # --------------------------------------------------------------------------