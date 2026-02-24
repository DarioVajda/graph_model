from ...utils import set_wandb_project, GraphTrainer, TextGraphDataset, GraphCollator
from ...models.llama_attn_bias import GraphLlamaForCausalLM

from .data_gen import create_and_save_dataset, dataset_path_and_size

import torch
import os
import random
from transformers import TrainingArguments, AutoTokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(model_name, device, bias_type, max_spd=32):
    model = GraphLlamaForCausalLM.from_pretrained(model_name, bias_type=bias_type, max_spd=max_spd, attn_implementation="eager")
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

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

def smuggle_prediction_step(super_prediction_step, model, inputs, prediction_loss_only, ignore_keys=None):
    # Run the standard evaluation step
    loss, logits, labels = super_prediction_step(
        model, inputs, prediction_loss_only, ignore_keys
    )
    
    # SMUGGLE THE DATA (to the preprocess_logits_for_metrics function)
    batch_size = len(inputs["input_ids"])
    prediction_token_indices = torch.full((batch_size,), -1, device=logits.device)
    if labels is not None and not prediction_loss_only:

        for i in range(batch_size):
            example_input_len = sum([ input_ids.shape[0] for input_ids in inputs["input_ids"][i] ])
            prediction_token_indices[i] = example_input_len - 2 # the second-to-last token is the one where the model should predict the labels
        
        # Turn labels into a tuple: (actual_labels, prediction_token_indices)
        labels = (labels, prediction_token_indices)
        
    return loss, logits, labels

class PreprocessLogits:
    def __init__(self, label_options, pad_token_id):
        self.label_options = label_options
        self.pad_token_id = pad_token_id

    def __call__(self, logits, labels):
        """
        Takes the raw logits from the model's forward pass and the true labels
        Returns the probabilities of the label options
        """
        labels, prediction_token_indices = labels # unpack the tuple we created in the smuggled prediction step

        batch_size = logits.shape[0]
        probs = torch.zeros((batch_size, 3), device=logits.device) # tensor[label_option_1_prob, label_option_2_prob, label_id (0/1)]

        probability_distributions = torch.softmax(logits, dim=-1) # get the probability distribution for the second-to-last token (the last token is just the end-of-sequence token)
        for i in range(batch_size):
            prediction_token_index = prediction_token_indices[i]
            if prediction_token_index == -1:
                raise ValueError("Prediction token index not found in labels. Make sure the smuggle_prediction_step is correctly adding the prediction token indices to the labels.")
            for j, label_option in enumerate(self.label_options):
                probs[i, j] = probability_distributions[i, prediction_token_index, label_option[0]] # get the probability of the correct label option at the prediction token index
            
            probs[i, 2] = 0 if labels[i][-1].item() == self.label_options[0][0] else (1 if labels[i][-1].item() == self.label_options[1][0] else -1) # get the true label (0 or 1, or -1 if not found)

        # Get the token ID with the highest probability
        return probs

class ComputeMetrics:
    def __init__(self, label_options):
        self.label_options = label_options
    
    def __call__(self, eval_preds):
        probs, (labels, prediction_token_indices) = eval_preds

        total = 0.0
        positive_prediction_count = 0.0
        correct = 0.0
        for i in range(probs.shape[0]):
            if probs[i, 2] == -1:
                raise ValueError("True label not found in labels. Make sure the smuggle_prediction_step is correctly adding the prediction token indices to the labels.")
            if probs[i, 2] == 0 and probs[i, 0] > probs[i, 1]: # true label is "Yes" and model predicts "Yes"
                correct += 1
            elif probs[i, 2] == 1 and probs[i, 1] > probs[i, 0]: # true label is "No" and model predicts "No"
                correct += 1

            if probs[i, 0] > probs[i, 1]: # model predicts "Yes"
                positive_prediction_count += 1

            total += 1
        
        # The Trainer will automatically log these keys to W&B!
        return {
            "classification_accuracy": float(correct) / float(total) if total > 0 else 0.0,
            "positive_prediction_ratio": float(positive_prediction_count) / float(total) if total > 0 else 0.0,
        }

def training_run(
    model, 
    train_dataset, 
    eval_dataset, 
    collator, 
    run_name, 
    num_epochs=3, 
    batch_size=8, 
    learning_rate=5e-5, 
    accumulation_steps=4, 
    label_options=None, 
    pad_token_id=None
):
    if label_options is None or pad_token_id is None:
        raise ValueError("Label options and pad token ID must be provided for the training run.")

    STEPS_PER_EPOCH = len(train_dataset) // batch_size // accumulation_steps

    training_args = TrainingArguments(
        # Basic training arguments:
        num_train_epochs=num_epochs,                            # Total number of training epochs to perform
        output_dir=f"./checkpoints/{run_name}",                 # Directory to save checkpoints and logs
        logging_steps=1,                                        # Log training metrics every 5 steps
        per_device_train_batch_size=batch_size,                 # Batch size per device during training
        gradient_accumulation_steps=accumulation_steps,         # Number of steps to accumulate gradients before performing an optimizer step
        gradient_checkpointing=False,                           # Gradient checkpointing to save memory

        # Evaluation arguments:
        eval_strategy="steps",                                  # Evaluate every eval_steps during training
        eval_steps=25,                                          # Number of steps between evaluations
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

    preprocess_logits_for_metrics = PreprocessLogits(label_options=label_options, pad_token_id=pad_token_id)
    compute_metrics = ComputeMetrics(label_options=label_options)

    # initialize using the custom class
    trainer = GraphTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,

        # custom evaluation function and logits preprocessor
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        custom_prediction_step=smuggle_prediction_step,
    )

    trainer.train()


if __name__ == "__main__":
    BIAS_TYPE = "combined" # options: "none", "spd", "laplacian", "combined"
    TRAIN_DATASET_SIZE = 10_000
    EVAL_DATASET_SIZE = 500
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    ACTIVE_PARAMS = ["spd_weights", "spectral_weights"]
    RUN_NAME = f"new_params_only_{BIAS_TYPE}"
    # ACTIVE_PARAMS = "all"
    # RUN_NAME = f"all_params_{BIAS_TYPE}"

    set_wandb_project("GraphLLM")
    device = get_device()

    model, tokenizer = init_model(model_name=MODEL_NAME, device=device, bias_type=BIAS_TYPE, max_spd=4)

    # --------------------------------------------------------------------------
    #region ----------------------- LOAD DATASETS ------------------------------
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

    possible_labels = [" Yes", " No" ]
    tokenized_possible_labels = [tokenizer(label, add_special_tokens=False).input_ids for label in possible_labels]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    #endregion
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #region ---------------- FINE TUNE SELECTED PARAMETERS ---------------------
    # --------------------------------------------------------------------------
    print("Fine-tuning these parameters: ", ACTIVE_PARAMS)
    select_active_params(model, active_params=ACTIVE_PARAMS)

    print("List of active parameters (requires_grad=True):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")
    print('-'*50)

    print("Example bias parameters before fine-tuning:")
    for name, param in model.named_parameters():
        if "1.self_attn.spd_weights" in name or "1.self_attn.spectral_weights" in name:
            print(f" - {name}: shape {param.shape}\n")
            print(param.data)
            print('-'*50)

    training_run(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        run_name=RUN_NAME,
        num_epochs=10,
        batch_size=4,
        learning_rate=1e-3,
        accumulation_steps=8,
        label_options=tokenized_possible_labels,
        pad_token_id=pad_token_id,
    )

    # inspect some of the bias parameters after fine-tuning (model.layers.1.self_attn.spd_weights, model.layers.1.self_attn.spectral_weights)
    print("Bias parameters after fine-tuning:")
    for name, param in model.named_parameters():
        if "1.self_attn.spd_weights" in name or "1.self_attn.spectral_weights" in name:
            print(f" - {name}: shape {param.shape}\n")
            print(param.data)
            print('-'*50)

    #endregion
    # --------------------------------------------------------------------------