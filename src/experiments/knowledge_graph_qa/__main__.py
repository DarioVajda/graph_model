from ...utils import set_wandb_project, GraphTrainer, TextGraphDataset, GraphCollator
from ...models.llama_attn_bias import GraphLlamaForCausalLM, GraphLlamaConfig

from .data_load import load_dataset
from .train_utils import get_device, compute_exact_match

import torch
import os, json
import datetime
import random
from transformers import TrainingArguments, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model
import numpy as np

#region -------- Code for model intialization and parameter selection --------
def init_model(model_name, device, bias_params):
    config = GraphLlamaConfig.from_pretrained(model_name, **bias_params)
    model = GraphLlamaForCausalLM.from_pretrained(
        model_name, 
        config=config, 
        attn_implementation="sdpa",
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
        warmup_steps=TOTAL_STEPS // 10,                         # Number of steps for the warmup phase (when the learning rate is increasing linearly)
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

    trainer.train()

    if test_dataset is None:
        print("No test dataset provided. Skipping final evaluation.")
        return

    # =====================================================================
    # Evaluate the best model on the test dataset
    # =====================================================================    
    print("\n" + "="*50)
    print("Training Complete. Evaluating Best Model on Test Dataset...")
    print("="*50)
    
    # By passing metric_key_prefix="test", the metrics will show up in wandb 
    # as test_em_accuracy and test_em_f1, keeping them distinct from eval metrics.
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    
    print("\nFinal Test Set Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print("="*50 + "\n")


def save_run_metadata(run_name, bias_params, dataset_name, base_model, active_params, lr, bias_lr, lora_config, num_epochs):
    """
    Save the metadata of the training run to the run_metadata_graph.json file with the run_name being the key (if there are multiple runs with the same base name, append "_v2", "_v3", etc. to the run name).

    Returns:
    run_name --> The final run name used for this training run (which may have a version suffix if there were duplicate names).
    """
    metadata_path = "./src/experiments/knowledge_graph_qa/run_metadata_graph.json"
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
    parser.add_argument("--dataset_name", type=str, default="kg_qa", help="Directory containing the processed dataset. Should be 'kg_qa' or 'family'.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Pre-trained model name or path.")
    parser.add_argument("--num_epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients before performing an optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the LoRA parameters.")
    parser.add_argument("--bias_learning_rate", type=float, default=5e-2, help="Learning rate for the bias parameters.")
    parser.add_argument("--eval_every", type=int, default=40, help="Number of steps between evaluations.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing (useful for debugging or if memory is not a concern).")

    # which parameters to activate and train
    parser.add_argument("--active_params", nargs="+", default=["spd_weights", "laplacian_weights", "rwse_weights", "rrwp_proj", "magnetic_"], help="List of parameter name substrings to activate for training. Use 'all' to activate all parameters.")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank for LoRA adapters. If not using LoRA, set to 0.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # parse command line arguments
    args = parse_args()
    # python3 -m src.experiments.knowledge_graph_qa --dataset_name=family --model_name=meta-llama/Llama-3.2-1B --lora_r=32 --batch_size=4 --accumulation_steps=4 --learning_rate=1e-4 --bias_learning_rate=1e-2 --num_epochs=8
    # python3 -m src.experiments.knowledge_graph_qa --dataset_name=family --model_name=meta-llama/Llama-3.2-3B --lora_r=64 --batch_size=2 --accumulation_steps=8 --learning_rate=1e-4 --bias_learning_rate=1e-2 --num_epochs=8
    # python3 -m src.experiments.knowledge_graph_qa --dataset_name=family --model_name=meta-llama/Llama-3.1-8B --lora_r=64 --batch_size=1 --accumulation_steps=16 --learning_rate=1e-5 --bias_learning_rate=1e-2 --num_epochs=8

    # --------------------------------------------------------------------------
    #region ----------------------- CONFIGURATION ------------------------------
    # --------------------------------------------------------------------------
    if args.dataset_name not in ["kg_qa", "family"]:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}. Must be 'kg_qa' or 'family'.")
    
    if args.dataset_name == "kg_qa":
        dataset_dir = "./src/experiments/knowledge_graph_qa/graph_datasets/dataset_40-60"
    else:
        dataset_dir = "./src/experiments/knowledge_graph_qa/family_tree_graph_dataset"
    dataset_name = f"graph_{args.dataset_name}"
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

    model_size = "1B" if "1b" in MODEL_NAME.lower() else ("3B" if "3b" in MODEL_NAME.lower() else ("8B" if "8b" in MODEL_NAME.lower() else "unknown_size"))

    # Create a unique run name and save the run metadata
    RUN_NAME = f"{model_size}_graph_{args.dataset_name}{'_lora' if LORA_CONFIG else ''}"
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
    train_dataset, eval_dataset, test_dataset = load_dataset(dataset_dir, type='graph')

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
        test_dataset=test_dataset,
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