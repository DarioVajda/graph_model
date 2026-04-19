import torch
import os
import json
import datetime
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

from .data_load import load_dataset
from .train_utils import get_device

# ------------------------------------------------------------------------------
# Dataset Wrapper for Text Baseline
# ------------------------------------------------------------------------------
class TextQADataset(Dataset):
    """Wraps the list of dictionaries from JSONL into a PyTorch Dataset."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
            # 'text' is dropped as the model only expects tensors
        }

# ------------------------------------------------------------------------------
# Model Initialization & LoRA Application
# ------------------------------------------------------------------------------
def init_model(model_name, device):
    print(f"Initializing standard causal LM: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def select_active_params(model, lora=None):
    """Applies LoRA configuration to the base model."""
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
    trainable_params = 0
    all_param = 0
    
    print("List of active parameters (requires_grad=True):")
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

# ------------------------------------------------------------------------------
# Metrics Evaluation Adapters
# ------------------------------------------------------------------------------
def preprocess_logits_for_metrics(logits, labels):
    """
    Computes argmax over vocabulary to prevent OOM errors during eval.
    Returns: Tensor of shape (batch_size, sequence_length)
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_exact_match(eval_preds):
    preds, labels = eval_preds
    
    # Causal LM Token Shift: 
    # The predicted token at index `i` corresponds to the target label at index `i + 1`.
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    
    y_true, y_pred = [], []
    
    for i in range(len(labels)):
        valid = labels[i] != -100
        if not np.any(valid): 
            continue
        
        y_true.append("-".join(map(str, labels[i][valid].tolist())))
        y_pred.append("-".join(map(str, preds[i][valid].tolist())))

    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])

    unique_classes = list(set(y_true))
    majority_class = max(set(y_true), key=y_true.count) if y_true else ""
    tp = { cl: 0 for cl in unique_classes }
    fp = { cl: 0 for cl in unique_classes }
    fn = { cl: 0 for cl in unique_classes }
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            if pred in unique_classes:
                fp[pred] += 1
            else:
                if majority_class:
                    fp[majority_class] += 1
            fn[true] += 1
            
    f1_scores = {
        cl: (2 * tp[cl]) / (2 * tp[cl] + fp[cl] + fn[cl]) if (2 * tp[cl] + fp[cl] + fn[cl]) > 0 else 0.0
        for cl in unique_classes
    }
    macro_f1 = np.mean(list(f1_scores.values())) if f1_scores else 0.0

    return {
        "em_accuracy": float(accuracy),
        "em_f1": float(macro_f1),
    }

# ------------------------------------------------------------------------------
# Training Run & Setup
# ------------------------------------------------------------------------------
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
    accumulation_steps=4, 
    eval_every=40,
    gradient_checkpointing=True,
):
    if gradient_checkpointing:
        print("Gradient checkpointing is ENABLED.")
    else:
        print("Gradient checkpointing is DISABLED.")

    STEPS_PER_EPOCH = len(train_dataset) // batch_size // accumulation_steps
    TOTAL_STEPS = max(1, STEPS_PER_EPOCH * num_epochs)

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=f"./checkpoints/{run_name}",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        metric_for_best_model="eval_em_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": learning_rate/10},
        warmup_steps=TOTAL_STEPS // 10,
        weight_decay=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_exact_match,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    if test_dataset is None:
        print("No test dataset provided. Skipping final evaluation.")
        return
  
    print("\n" + "="*50)
    print("Training Complete. Evaluating Best Model on Test Dataset...")
    print("="*50)
    
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    
    print("\nFinal Test Set Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print("="*50 + "\n")


def save_run_metadata(run_name, dataset_name, base_model, lr, lora_config, num_epochs):
    metadata_path = "./src/experiments/knowledge_graph_qa/run_metadata_text.json"
    if not os.path.exists(os.path.dirname(metadata_path)):
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a baseline text LLM on a specified dataset.")

    parser.add_argument("--dataset_name", type=str, default="family", help="Directory containing the processed dataset. Should be 'kg_qa' or 'family'.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Pre-trained model name or path.")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for LoRA parameters.")
    parser.add_argument("--eval_every", type=int, default=40, help="Number of steps between evaluations.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing.")
    parser.add_argument("--lora_r", type=int, default=32, help="Rank for LoRA adapters. If 0, LoRA is disabled.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name not in ["kg_qa", "family"]:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}. Must be 'kg_qa' or 'family'.")
    
    if args.dataset_name == "kg_qa":
        dataset_dir = "./src/experiments/knowledge_graph_qa/text_datasets/dataset_40-60"
    else:
        dataset_dir = "./src/experiments/knowledge_graph_qa/family_tree_text_dataset"
        
    dataset_name = f"text_{args.dataset_name}"
    MODEL_NAME = args.model_name
    LR = args.learning_rate
    NUM_EPOCHS = args.num_epochs

    LORA_R = args.lora_r
    LORA_CONFIG = {
        "r": LORA_R,
        "lora_alpha": LORA_R*2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
    } if LORA_R > 0 else None

    model_size = "1B" if "1b" in MODEL_NAME.lower() else ("3B" if "3b" in MODEL_NAME.lower() else ("8B" if "8b" in MODEL_NAME.lower() else "unknown_size"))

    RUN_NAME = f"{model_size}_text_{args.dataset_name}{'_lora' if LORA_CONFIG else ''}"
    RUN_NAME = save_run_metadata(
        run_name=RUN_NAME,
        dataset_name=dataset_name,
        base_model=MODEL_NAME,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        lora_config=LORA_CONFIG,
    )
    EVAL_EVERY = args.eval_every

    import wandb
    wandb.init(project="GraphLLM", name=RUN_NAME)

    device = get_device()
    model, tokenizer = init_model(model_name=MODEL_NAME, device=device)

    # --------------------------------------------------------------------------
    # Load Datasets (Using text type)
    # --------------------------------------------------------------------------
    raw_train, raw_val, raw_test = load_dataset(dataset_dir, type='text')

    train_dataset = TextQADataset(raw_train)
    eval_dataset = TextQADataset(raw_val)
    test_dataset = TextQADataset(raw_test)

    # DataCollatorForSeq2Seq is excellent here as it seamlessly pads input_ids 
    # with `tokenizer.pad_token_id` and labels with `-100`.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # --------------------------------------------------------------------------
    # Fine Tune Parameters
    # --------------------------------------------------------------------------
    model = select_active_params(model, lora=LORA_CONFIG)
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
        accumulation_steps=args.accumulation_steps,
        eval_every=EVAL_EVERY,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )