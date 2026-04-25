from ...utils import set_wandb_project, GraphTrainer, TextGraphDataset, GraphCollator
from ...models.llama_attn_bias import GraphLlamaForCausalLM, GraphLlamaConfig

from .load_data import load_dataset
from .train_utils import get_device, compute_exact_match

import torch
import os, json
import datetime
import random
from transformers import TrainingArguments, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
import numpy as np
import gc

#region -------- Code for model intialization and parameter selection --------
def init_model(model_name, device, bias_params):
    config = GraphLlamaConfig.from_pretrained(model_name, **bias_params)
    model = GraphLlamaForCausalLM.from_pretrained(
        model_name, 
        config=config, 
        attn_implementation="sdpa",
        # allowed_seq_lens=[1024, 2048, 4096, 8192],
        # allowed_node_counts=[64, 128],
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

def training_run(
    model, 
    train_dataset, 
    eval_dataset, 
    test_dataset,
    collator, 
    run_name, 
    num_epochs=3, 
    batch_size=8, 
    learning_rate=5e-5, 
    bias_learning_rate=1e-3,
    accumulation_steps=4, 
    pad_token_id=None,
    active_params=None,
    eval_every=40,
    gradient_checkpointing=True,
):
    if gradient_checkpointing:
        print("Gradient checkpointing is ENABLED. This will save memory but may increase training time.")
    else:
        print("Gradient checkpointing is DISABLED. This may lead to out-of-memory errors if the model or batch size is too large.")

    STEPS_PER_EPOCH = len(train_dataset) // batch_size // accumulation_steps
    TOTAL_STEPS = STEPS_PER_EPOCH * num_epochs
    EVAL_EVERY = eval_every

    training_args = TrainingArguments(
        # Basic training arguments:
        num_train_epochs=num_epochs,                            # Total number of training epochs to perform
        output_dir=f"./checkpoints/{run_name}",                 # Directory to save checkpoints and logs
        logging_steps=1,                                        # Log training metrics every 5 steps
        per_device_train_batch_size=batch_size,                 # Batch size per device during training
        gradient_accumulation_steps=accumulation_steps,         # Number of steps to accumulate gradients before performing an optimizer step
        # torch_compile=True,                                     # Use PyTorch 2.0's torch.compile for potential speedup (requires PyTorch 2.0+)
        gradient_checkpointing=gradient_checkpointing,          # Enable gradient checkpointing to save memory (trades compute for memory)
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,

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
        warmup_steps=TOTAL_STEPS // 10,                         # Number of steps for the warmup phase
        weight_decay=0.1,                                       # Weight decay to apply (if not zero)
    )

    # initialize using the custom class
    trainer = GraphTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,

        # Evaluation parameters
        compute_metrics=compute_exact_match,

        # set the active parameters for bias saving in the trainer callback
        active_params=active_params,

        # set the custom bias learning rate to create two distinct parameter groups
        bias_lr=bias_learning_rate,
    )

    # 1. Execute the main training loop
    trainer.train()

    # =========================================================================
    # 2. TEST DATASET EVALUATION LOOP ON TOP 3 SAVED CHECKPOINTS
    # =========================================================================
    print("\n" + "="*70)
    print("Training Complete. Locating Top 3 Saved Checkpoints for Evaluation...")
    print("="*70)

    output_dir = training_args.output_dir
    # Get all checkpoint directories left by the Trainer (since save_total_limit=3)
    checkpoint_dirs = [
        os.path.join(output_dir, d) 
        for d in os.listdir(output_dir) 
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    eval_results = []
    
    # Iterate and test each saved checkpoint
    for ckpt_dir in checkpoint_dirs:
        print(f"\nEvaluating checkpoint: {ckpt_dir}")
        
        # Load the checkpoint model using your custom loader logic
        ckpt_model = GraphLlamaForCausalLM.from_pretrained(ckpt_dir)
        ckpt_model.to(get_device())
        ckpt_model.eval()

        # Dummy arguments strictly for evaluation to disable wandb logging
        eval_args = TrainingArguments(
            output_dir="./eval_temp",
            per_device_eval_batch_size=batch_size,
            report_to="none", 
            do_train=False,
            do_eval=True,
        )

        eval_trainer = GraphTrainer(
            model=ckpt_model,
            args=eval_args,
            eval_dataset=test_dataset,
            data_collator=collator,
            compute_metrics=compute_exact_match,
        )

        # Run evaluation and prepend 'test_' to metric keys
        metrics = eval_trainer.evaluate(metric_key_prefix="test")
        eval_results.append((ckpt_dir, metrics))

        # Cleanup memory to prevent OOM before loading the next checkpoint
        del ckpt_model
        del eval_trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Sort results descending based on exact match accuracy on the test set
    eval_results.sort(key=lambda x: x[1].get("test_em_accuracy", 0.0), reverse=True)

    # Prepare to append to the JSON file
    output_file = "./src/experiments/benchmarks/test_results.json"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    suffixes = ["_first", "_second", "_third"]
    
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION RESULTS")
    print("="*70)
    
    for i, (ckpt_dir, metrics) in enumerate(eval_results):
        suffix = suffixes[i] if i < len(suffixes) else f"_{i+1}th"
        
        # Create a clean key for the JSON file 
        checkpoint_name = os.path.basename(os.path.normpath(ckpt_dir))
        run_key = f"{run_name}_{checkpoint_name}{suffix}"
        
        # Avoid overriding previous testing iterations manually triggered
        if run_key in all_results:
            version = 2
            while f"{run_key}_v{version}" in all_results:
                version += 1
            run_key = f"{run_key}_v{version}"

        # Place this new evaluation at the top of the dictionary
        all_results = {run_key: metrics, **all_results}

        # Print cleanly to the console
        print(f"\n--- Checkpoint Rank {i+1} {suffix.strip('_').upper()} ---")
        print(f"Path: {ckpt_dir}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    # Write everything back to disk
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nAll checkpoint metrics successfully appended to {output_file}.")


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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune GraphLLaMA on a specified dataset with configurable parameters.")

    # general parameters
    parser.add_argument("--dataset_name", type=str, default="cora_hops2_neighbors60_target_abstract", help="Directory containing the processed dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B", help="Pre-trained model name or path.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Number of steps to accumulate gradients before performing an optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--bias_learning_rate", type=float, default=5e-2, help="Learning rate for the bias parameters.")
    parser.add_argument("--eval_every", type=int, default=40, help="Number of steps between evaluations.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing (useful for debugging or if memory is not a concern).")

    # which parameters to activate and train
    parser.add_argument("--active_params", nargs="+", default=["spd_weights", "laplacian_weights", "rwse_weights", "rrwp_proj", "magnetic_"], help="List of parameter name substrings to activate for training. Use 'all' to activate all parameters.")
    parser.add_argument("--lora_r", type=int, default=32, help="Rank for LoRA adapters. If not using LoRA, set to 0.")

    # bias related parameters
    parser.add_argument("--no_spd", action="store_true", help="Whether to use the shortest path distance bias in the model.")
    parser.add_argument("--no_laplacian", action="store_true", help="Whether to use the Laplacian eigenvector bias in the model.")
    parser.add_argument("--no_rwse", action="store_true", help="Whether to use the random walk structural encoding bias in the model.")
    parser.add_argument("--no_rrwp", action="store_true", help="Whether to use the random walk with restart probability bias in the model.")
    parser.add_argument("--no_magnetic", action="store_true", help="Whether to use the magnetic bias in the model.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # parse command line arguments
    args = parse_args()

    # --------------------------------------------------------------------------
    #region ----------------------- CONFIGURATION ------------------------------
    # --------------------------------------------------------------------------
    dataset_dir = f"./src/experiments/benchmarks/processed_data/{args.dataset_name}"
    dataset_name = dataset_dir.split("/")[-1]
    BIAS_PARAMS = { 
        "spd": not args.no_spd, 
        "max_spd": 8, 
        "laplacian": not args.no_laplacian, 
        "rwse": not args.no_rwse, 
        "rrwp": not args.no_rrwp, 
        "max_rw_steps": 16,
        "magnetic": not args.no_magnetic,
        "magnetic_dim": 32,
        "magnetic_q": 0.25
    }
    MODEL_NAME = args.model_name
    ACTIVE_PARAMS = args.active_params
    LR = args.learning_rate
    BIAS_LR=args.bias_learning_rate
    NUM_EPOCHS = args.num_epochs

    LORA_R = args.lora_r
    LORA_CONFIG = {
        "r": LORA_R,
        "lora_alpha": LORA_R*2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    }
    if LORA_R == 0: # if rank is set to 0, don't use LoRA at all
        LORA_CONFIG = None

    # Create a unique run name and save the run metadata
    RUN_NAME = f"{dataset_name}{'_lora' if LORA_CONFIG else ''}"
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
    EVAL_EVERY = args.eval_every
    #endregion
    # --------------------------------------------------------------------------

    set_wandb_project("GraphLLM")
    device = get_device()

    model, tokenizer = init_model(model_name=MODEL_NAME, device=device, bias_params=BIAS_PARAMS)

    # --------------------------------------------------------------------------
    #region ----------------------- LOAD DATASETS ------------------------------
    # --------------------------------------------------------------------------
    
    datasets = load_dataset(dataset_dir, train=True, val=True, test=True)
    train_dataset = datasets["train"]
    eval_dataset = datasets["val"]
    test_dataset = datasets["test"]

    collator = GraphCollator()

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

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
        test_dataset=test_dataset,      # <--- MODIFICATION: Pass test dataset
        collator=collator,
        run_name=RUN_NAME,
        num_epochs=NUM_EPOCHS,
        batch_size=args.batch_size,
        learning_rate=LR,
        bias_learning_rate=BIAS_LR,
        accumulation_steps=args.accumulation_steps,
        active_params=ACTIVE_PARAMS,
        eval_every=EVAL_EVERY,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    #endregion
    # --------------------------------------------------------------------------