from ...utils import set_wandb_project, GraphTrainer, TextGraphDataset, GraphCollator
from ...models.llama_attn_bias import GraphLlamaForCausalLM, GraphLlamaConfig

from .load_data import load_dataset

import torch
import os, json
import datetime
import random
from transformers import TrainingArguments, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#region -------- Code for model intialization and parameter selection --------
def init_model(model_name, device, bias_params):
    config = GraphLlamaConfig.from_pretrained(model_name, **bias_params)
    model = GraphLlamaForCausalLM.from_pretrained(
        model_name, 
        config=config, 
        attn_implementation="sdpa",
        allowed_seq_lens=[1024, 2048, 4096],
        allowed_node_counts=[32, 64, 128],
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def select_active_params(model, active_params=None, lora=None):
    """
    Applies LoRA if a configuration dictionary is provided.
    Sets requires_grad=True for parameters whose names contain any of the substrings in active_params.
    If active_params is None, no additional parameters are unfrozen.
    If active_params is "all", all parameters are set to requires_grad=True.
    """
    # apply LoRA if configuration is provided
    if lora is not None:
        print("Applying LoRA with config:", lora)
        lora_config = LoraConfig(
            r=lora.get("r", 8),
            lora_alpha=lora.get("lora_alpha", 16),
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=lora.get("lora_dropout", 0.05),
            bias=lora.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        # wrap the model with Low-Rank Adapters
        model = get_peft_model(model, lora_config)

    # handle custom active paramters (usually those used for computing the graph-based attention biases)
    if active_params == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif active_params is not None:
        for name, param in model.named_parameters():
            if any(active_param in name for active_param in active_params):
                param.requires_grad = True

    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model, 
    differentiating between LoRA adapters and custom active parameters.
    """
    trainable_lora_params = 0
    trainable_custom_params = 0
    all_param = 0
    
    print("List of custom active parameters (requires_grad=True):")
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        
        if param.requires_grad:
            # PEFT always includes 'lora' in the adapter weight names
            if "lora" in name.lower():
                trainable_lora_params += num_params
                print(f" - {name} ({num_params:,} params) [LoRA Adapter]")
            else:
                trainable_custom_params += num_params
                print(f" - {name} ({num_params:,} params)")
                
    total_trainable = trainable_lora_params + trainable_custom_params
    
    print("\n" + "="*50)
    print("TRAINABLE PARAMETER SUMMARY")
    print("="*50)
    print(f"LoRA Adapters:       {trainable_lora_params:>15,}")
    print(f"Custom Graph Biases: {trainable_custom_params:>15,}")
    print("-" * 50)
    print(f"Total Trainable:     {total_trainable:>15,}")
    print(f"Total Model Params:  {all_param:>15,}")
    print(f"Trainable %:         {100 * total_trainable / all_param:>14.4f}%")
    print("="*50 + "\n")
#endregion

#region -------- Code for custom evaluation --------
class PreprocessLogitsEM:
    def __call__(self, logits, labels):
        # 1. Unpack logits if the model returns a tuple (common in HF models)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        # 2. Get the predicted token IDs via argmax
        preds = torch.argmax(logits, dim=-1)
        
        # 3. Shift predictions to align with labels
        # The logit at sequence index 't' predicts the label at 't+1'
        shifted_preds = torch.full_like(preds, fill_value=-100)
        shifted_preds[:, 1:] = preds[:, :-1]
        
        return shifted_preds

def compute_exact_match(eval_preds):
    preds, labels = eval_preds
    
    exact_matches = 0
    total = len(labels)
    
    for i in range(total):
        # Find the valid label tokens (ignoring -100 padding)
        valid_indices = labels[i] != -100
        
        if not np.any(valid_indices):
            continue
            
        example_preds = preds[i][valid_indices]
        example_labels = labels[i][valid_indices]
        
        # Exact Match: ALL predicted tokens must match the ground truth
        if np.array_equal(example_preds, example_labels):
            exact_matches += 1
            
    return {
        "em_accuracy": float(exact_matches) / total if total > 0 else 0.0,
    }
#endregion

def training_run(
    model, 
    train_dataset, 
    eval_dataset, 
    collator, 
    run_name, 
    num_epochs=3, 
    batch_size=8, 
    learning_rate=5e-5, 
    bias_learning_rate=1e-3,
    accumulation_steps=4, 
    pad_token_id=None,
    active_params=None,
):
    # if label_options is None or pad_token_id is None:
    #     raise ValueError("Label options and pad token ID must be provided for the training run.")

    STEPS_PER_EPOCH = len(train_dataset) // batch_size // accumulation_steps
    EVAL_EVERY = 20

    training_args = TrainingArguments(
        # Basic training arguments:
        num_train_epochs=num_epochs,                            # Total number of training epochs to perform
        output_dir=f"./checkpoints/{run_name}",                 # Directory to save checkpoints and logs
        logging_steps=1,                                        # Log training metrics every 5 steps
        per_device_train_batch_size=batch_size,                 # Batch size per device during training
        gradient_accumulation_steps=accumulation_steps,         # Number of steps to accumulate gradients before performing an optimizer step
        # torch_compile=True,                                     # Use PyTorch 2.0's torch.compile for potential speedup (requires PyTorch 2.0+)
        gradient_checkpointing=True,                            # Enable gradient checkpointing to save memory (trades compute for memory)
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Use non-reentrant checkpointing to be compatible with PyTorch 2.0's torch.compile and avoid issues with certain operations (like in our custom attention)

        # Evaluation arguments:
        eval_strategy="steps",                                  # Evaluate every eval_steps during training
        eval_steps=EVAL_EVERY,                                  # Number of steps between evaluations
        save_strategy="steps",                                  # Save a checkpoint based on save_steps
        save_steps=EVAL_EVERY,                                  # Number of steps between saving checkpoints
        metric_for_best_model="eval_em_accuracy",               # Metric to use for determining the best model
        greater_is_better=True,                                 # Higher classification_accuracy is better
        save_total_limit=3,                                     # Maximum number of checkpoints to store
        load_best_model_at_end=True,                            # Load the best model at the end of training

        # WandB logging:
        report_to="wandb",                                      # Report training metrics to Weights & Biases
        run_name=run_name,                                      # Name of the WandB run for better organization
        
        # Learning rate scheduler:
        learning_rate=learning_rate,                            # The initial learning rate for Adam
        lr_scheduler_type="cosine_with_min_lr",                 # Type of learning rate scheduler to use
        lr_scheduler_kwargs={"min_lr": learning_rate/10},       # Additional arguments for the learning rate scheduler
        warmup_steps=STEPS_PER_EPOCH*2,                           # Number of steps for the warmup phase (when the learning rate is increasing linearly)
        weight_decay=0.1,                                       # Weight decay to apply (if not zero)
    )

    preprocess_logits = PreprocessLogitsEM()

    # initialize using the custom class
    trainer = GraphTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,

        # Evaluation parameters
        compute_metrics=compute_exact_match,
        # preprocess_logits_for_metrics=preprocess_logits,

        # set the active parameters for bias saving in the trainer callback
        active_params=active_params,

        # set the custom bias learning rate to create two distinct parameter groups
        bias_lr=bias_learning_rate,
    )

    trainer.train()

def save_run_metadata(run_name, bias_params, dataset_name, base_model, active_params, lr, bias_lr, lora_config, num_epochs):
    """
    Save the metadata of the training run to the run_metadata.json file with the run_name being the key (if there are multiple runs with the same base name, append "_v2", "_v3", etc. to the run name).

    Returns:
    run_name --> The final run name used for this training run (which may have a version suffix if there were duplicate names).
    """
    metadata_path = "./src/experiments/benchmarks/run_metadata.json"
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            json.dump({}, f)
    with open(metadata_path, "r") as f:
        run_metadata = json.load(f)
    
    if run_name in run_metadata:
        version = 2
        new_run_name = f"{run_name}_v{version}"
        while new_run_name in run_metadata:
            version += 1
            new_run_name = f"{run_name}_v{version}"
        run_name = new_run_name

    # determine what is the exact time of the run
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Create a dictionary specifically for the new entry
    new_entry = {
        run_name: {
            "date_time": date_time,
            "bias_params": bias_params,
            "dataset_name": dataset_name,
            "base_model": base_model,
            "active_params": active_params,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "bias_learning_rate": bias_lr,
            "lora_config": lora_config,
        }
    }

    # 2. Prepend the new entry to the existing metadata by unpacking both
    # The new_entry comes first, so it stays at the top!
    run_metadata = {**new_entry, **run_metadata}

    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=4)
    
    return run_name

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    #region ----------------------- CONFIGURATION ------------------------------
    # --------------------------------------------------------------------------
    dataset_dir = "./src/experiments/benchmarks/processed_data/cora_hops2_neighbors30"
    dataset_name = dataset_dir.split("/")[-1]
    BIAS_PARAMS = { 
        "spd": True, 
        "max_spd": 8, 
        "laplacian": False, 
        "rwse": False, 
        "rrwp": True, 
        "max_rw_steps": 16,
        "magnetic": True,
        "magnetic_dim": 32,
        "magnetic_q": 0.25
    }
    # MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    ACTIVE_PARAMS = ["spd_weights", "laplacian_weights", "rwse_weights", "rrwp_proj", "magnetic_"] # options: list of parameter name substrings to activate, or "all" to activate all parameters, or None to freeze all parameters
    LR = 3e-5
    BIAS_LR=5e-3
    NUM_EPOCHS = 20

    LORA_R = 8
    LORA_CONFIG = {
        "r": LORA_R,
        "lora_alpha": LORA_R*2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.00,
        "bias": "none",
    }
    # LORA_CONFIG = None

    # Create a unique run name and save the run metadata
    RUN_NAME = f"{dataset_name}_{'_lora' if LORA_CONFIG else ''}"
    RUN_NAME = save_run_metadata(
        run_name=RUN_NAME,
        bias_params=BIAS_PARAMS,
        dataset_name=dataset_name,
        base_model=MODEL_NAME,
        active_params=ACTIVE_PARAMS,
        lr=LR,
        bias_lr=BIAS_LR,
        num_epochs=NUM_EPOCHS,
        lora_config=LORA_CONFIG,
    )
    #endregion
    # --------------------------------------------------------------------------

    set_wandb_project("GraphLLM")
    device = get_device()

    model, tokenizer = init_model(model_name=MODEL_NAME, device=device, bias_params=BIAS_PARAMS)

    # --------------------------------------------------------------------------
    #region ----------------------- LOAD DATASETS ------------------------------
    # --------------------------------------------------------------------------
    datasets = load_dataset(dataset_dir, train=True, val=True, test=False)
    train_dataset = datasets["train"]
    eval_dataset = datasets["val"]

    collator = GraphCollator()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    #endregion
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #region ---------------- FINE TUNE SELECTED PARAMETERS ---------------------
    # --------------------------------------------------------------------------
    print("Fine-tuning these parameters: ", ACTIVE_PARAMS)
    model = select_active_params(model, active_params=ACTIVE_PARAMS, lora=LORA_CONFIG)
    print_trainable_parameters(model)

    training_run(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        run_name=RUN_NAME,
        num_epochs=NUM_EPOCHS,
        batch_size=4,
        learning_rate=LR,
        bias_learning_rate=BIAS_LR,
        accumulation_steps=8,
        active_params=ACTIVE_PARAMS,
    )

    #endregion
    # --------------------------------------------------------------------------