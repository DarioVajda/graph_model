import torch
import os, json
import datetime
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer
)
from peft import LoraConfig, get_peft_model

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#region -------- Code for model initialization and parameter selection --------
def init_model(model_name, device):
    # Using standard AutoModelForCausalLM with Flash Attention 2
    # Note: Flash Attention requires bf16 or fp16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 
    )
    model.to(device)
    
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token exists for batched training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def select_active_params(model, lora=None):
    """
    Applies LoRA if a configuration dictionary is provided.
    """
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
        model = get_peft_model(model, lora_config)

    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model (LoRA).
    """
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f" - {name} ({num_params:,} params) [LoRA Adapter]")
                
    print("\n" + "="*50)
    print("TRAINABLE PARAMETER SUMMARY")
    print("="*50)
    print(f"Total Trainable:     {trainable_params:>15,}")
    print(f"Total Model Params:  {all_param:>15,}")
    print(f"Trainable %:         {100 * trainable_params / all_param:>14.4f}%")
    print("="*50 + "\n")
#endregion

#region -------- Code for Dummy Dataset --------
class DummyLengthDataset(Dataset):
    """
    Generates dummy data where ~80% of samples are 1024 tokens 
    and ~20% of samples are 2048 tokens long.
    """
    def __init__(self, num_samples, vocab_size):
        self.data = []
        for _ in range(num_samples):
            # 80% chance for 1024, 20% chance for 2048
            seq_len = 1024 if random.random() < 0.8 else 2048
            
            # Generate random token IDs
            input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
            
            self.data.append({
                "input_ids": input_ids,
                "labels": input_ids.clone() # Causal LM standard
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PaddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features):
        # Find the max length in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        batch_input_ids = []
        batch_labels = []
        
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            
            # Pad input_ids with pad_token_id
            padded_inputs = torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            # Pad labels with -100 (standard ignore index for PyTorch CrossEntropyLoss)
            padded_labels = torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            
            batch_input_ids.append(padded_inputs)
            batch_labels.append(padded_labels)
            
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels)
        }
#endregion

def training_run(
    model, 
    train_dataset, 
    eval_dataset, 
    run_name, 
    num_epochs=3, 
    batch_size=8, 
    learning_rate=5e-5, 
    accumulation_steps=4,
    pad_token_id=None,
):

    STEPS_PER_EPOCH = max(1, len(train_dataset) // batch_size // accumulation_steps)
    EVAL_EVERY = 20

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,                            
        output_dir=f"./checkpoints/{run_name}",                 
        logging_steps=1,                                        
        per_device_train_batch_size=batch_size,                 
        gradient_accumulation_steps=accumulation_steps,         
        gradient_checkpointing=True,                            
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        eval_strategy="steps",                                  
        eval_steps=EVAL_EVERY,                                  
        save_strategy="steps",                                  
        save_steps=EVAL_EVERY,                                  
        save_total_limit=3,                                     
        load_best_model_at_end=False,                           
        # report_to="wandb",                                      
        run_name=run_name,                                      
        learning_rate=learning_rate,                            
        lr_scheduler_type="cosine_with_min_lr",                 
        lr_scheduler_kwargs={"min_lr": learning_rate/10},       
        warmup_steps=STEPS_PER_EPOCH*2,                           
        weight_decay=0.1,
        # Ensure fp16/bf16 is passed for Flash Attention compatibility
        bf16=True, 
    )

    # Use standard HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PaddingCollator(pad_token_id),
    )

    trainer.train()

def save_run_metadata(run_name, dataset_name, base_model, lr, lora_config, num_epochs):
    """Simplified metadata tracking for the baseline run."""
    metadata_path = "run_metadata.json"
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

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_entry = {
        run_name: {
            "date_time": date_time,
            "type": "Vanilla Baseline Speed Test",
            "dataset_name": dataset_name,
            "base_model": base_model,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "lora_config": lora_config,
        }
    }

    run_metadata = {**new_entry, **run_metadata}

    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=4)
    
    return run_name

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    #region ----------------------- CONFIGURATION ------------------------------
    # --------------------------------------------------------------------------
    dataset_name = "dummy_baseline_data"
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    LR = 3e-5
    NUM_EPOCHS = 20

    LORA_R = 8
    LORA_CONFIG = {
        "r": LORA_R,
        "lora_alpha": LORA_R*2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.00,
        "bias": "none",
    }

    RUN_NAME = f"baseline_speed_test_{'_lora' if LORA_CONFIG else ''}"
    # RUN_NAME = save_run_metadata(
    #     run_name=RUN_NAME,
    #     dataset_name=dataset_name,
    #     base_model=MODEL_NAME,
    #     lr=LR,
    #     num_epochs=NUM_EPOCHS,
    #     lora_config=LORA_CONFIG,
    # )
    #endregion
    # --------------------------------------------------------------------------

    # Commenting out wandb init so you don't clutter your graph project if you don't want to
    # import wandb
    # wandb.init(project="GraphLLM_Baseline") 
    
    device = get_device()
    model, tokenizer = init_model(model_name=MODEL_NAME, device=device)

    # --------------------------------------------------------------------------
    #region ----------------------- LOAD DATASETS ------------------------------
    # --------------------------------------------------------------------------
    # Create dummy datasets. Using 1000 train samples and 100 eval samples as placeholders
    print("Generating synthetic speed-test dataset...")
    vocab_size = model.config.vocab_size
    train_dataset = DummyLengthDataset(num_samples=1000, vocab_size=vocab_size)
    eval_dataset = DummyLengthDataset(num_samples=100, vocab_size=vocab_size)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    #endregion
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #region ---------------- FINE TUNE SELECTED PARAMETERS ---------------------
    # --------------------------------------------------------------------------
    model = select_active_params(model, lora=LORA_CONFIG)
    print_trainable_parameters(model)

    training_run(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        run_name=RUN_NAME,
        num_epochs=NUM_EPOCHS,
        batch_size=4,
        learning_rate=LR,
        accumulation_steps=8,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    #endregion