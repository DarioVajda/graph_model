from .data_gen import prepare_dataset, generate_graph_dataset
from ...models.llama import GraphLlamaForCausalLM
from transformers import LlamaForCausalLM, LlamaConfig

import os
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import random

# read the HF_TOKEN from the ~/hf_token.txt file
with open(os.path.expanduser("./hf_token.txt"), "r") as f:
    HF_TOKEN = f.read().strip()
os.environ["HF_TOKEN"] = HF_TOKEN

def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print("Processed dataset not found. Generating new dataset...")
        MIN_NODES = 10
        MAX_NODES = 20
        SPECTRAL_DIMS = 8
        DATASET_SIZE = 2500

        dataset = generate_graph_dataset(DATASET_SIZE, min_size=MIN_NODES, max_size=MAX_NODES)
        print(f"Generated {len(dataset)} examples.")

        model_name = "meta-llama/Llama-3.2-1B"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        prepare_dataset(dataset, tokenizer, graph_model=True, max_nodes=MAX_NODES, spectral_dims=SPECTRAL_DIMS, save_path=dataset_path)
        print(f"Processed dataset with {DATASET_SIZE} examples.")

    print(f"Loading processed dataset from {dataset_path}...")
    data = torch.load(dataset_path, weights_only=False)
    print(f"Loaded dataset with {len(data['dataset'])} examples.")
    return data['hyperparameters'], data['dataset']

def split_dataset(dataset, val_ratio):
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    
    train_set = dataset[:total_size - val_size]
    val_set = dataset[total_size - val_size:]

    return train_set, val_set

def create_batches(dataset, batch_size, pad_token_id):
    """
    Create batches from the dataset.

    Args:
        dataset (list): List of dataset examples, each one is a dict with "graph", "x", "y", "label", "input_ids, "labels", "node_ids", and "node_spectral_features".
        batch_size (int): Size of each batch.
        pad_token_id (int): Token ID used for padding sequences.
    """
    batches_lists = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batches_lists.append(batch)
    
    # convert lists of dicts to dict of padded tensors
    batches = []
    for batch in batches_lists:
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        num_nodes = batch[0]["node_spectral_features"].size(0)
        node_ids = pad_sequence([item["node_ids"] for item in batch], batch_first=True, padding_value=num_nodes-1)
        node_spectral_features = torch.stack([item["node_spectral_features"] for item in batch], dim=0) # these should already be the same size [ max_num_nodes, spectral_dims ]

        attention_mask = (input_ids != pad_token_id).long()

        batches.append({
            "input_ids": input_ids,
            "labels": labels,
            "node_ids": node_ids,
            "node_spectral_features": node_spectral_features,
            "attention_mask": attention_mask,
        })
        
    return batches

def model_inputs(model_type, batch):
    assert model_type in ["llama", "graph_llama"], "model_type must be either 'llama' or 'graph_llama'"

    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"], # <--- Pass it here
        "labels": batch["labels"]
    }
    
    if model_type == "graph_llama":
        inputs.update({
            "node_ids": batch["node_ids"],
            "node_spectral_features": batch["node_spectral_features"]
        })
    
    return inputs

def train_model(
    model_type, 
    model_name, 
    train_set, 
    val_set, 
    hyperparameters, 
    batch_size=8, 
    epochs=3, 
    eval_every=100, 
    optimizer_params={"learning_rate": 1e-3, "min_lr_factor": 0.1, "warmup_steps": 20}
):
    assert model_type in ["llama", "graph_llama"], "model_type must be either 'llama' or 'graph_llama'"

    print("\n" + "="*50)
    print("Starting training for model type:", model_type)
    print("="*50)

    # load the config for the model
    config = LlamaConfig.from_pretrained(model_name)
    print("Model config loaded.")

    # load the model
    if model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            model_name, 
            config=config, 
        )
    else:  # graph_llama
        model = GraphLlamaForCausalLM.from_pretrained(
            model_name, 
            config=config, 
            spectral_dims=hyperparameters['spectral_dims'], 
            strict=False, 
        )
    print("Model loaded. Class: ", model.__class__.__name__)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    print(f"Pad token exists: {pad_token_id}" if tokenizer.pad_token_id is not None else f"Pad token does not exist, using eos token as pad token: {pad_token_id}")

    # create batches for the training and validation sets
    train_batches = create_batches(train_set, batch_size, pad_token_id)
    NUM_TRAIN_EXAMPLES = len(train_set)
    val_batches = create_batches(val_set, batch_size, pad_token_id)
    NUM_VAL_EXAMPLES = len(val_set)
    print(f"Created {len(train_batches)} training batches and {len(val_batches)} validation batches.")

    # calculate total training steps
    total_steps = len(train_batches) * epochs
    print(f"Total training steps: {total_steps}")

    # read optimizer parameters
    lr = optimizer_params.get('learning_rate', 5e-5)
    min_lr_factor = optimizer_params.get('min_lr_factor', 0.1)
    # warmup_steps = optimizer_params.get('warmup_steps', total_steps//5)
    warmup_steps = total_steps//5

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # create the optimizer learning rate scheduler
    lin_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cos_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=lr * min_lr_factor
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[lin_scheduler, cos_scheduler],
        milestones=[warmup_steps]
    )
    print("Optimizer and scheduler set up.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # move everything to the device
    model.to(device)
    for key in train_batches[0].keys():
        train_batches = [{k: v.to(device) for k, v in batch.items()} for batch in train_batches]
        val_batches = [{k: v.to(device) for k, v in batch.items()} for batch in val_batches]
    print("Moved model and data to device.")

    # training loop
    steps_performed = 0
    total_train_loss = 0.0
    train_losses = []
    val_losses = []
    
    # copy the model to initialise the "best_model"
    if model_type == "llama":
        best_model = LlamaForCausalLM.from_pretrained(model_name, config=config)
    else:  # graph_llama
        best_model = GraphLlamaForCausalLM.from_pretrained(model_name, config=config, spectral_dims=hyperparameters['spectral_dims'], strict=False)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        # shuffle the training batches
        random.shuffle(train_batches)

        for batch in tqdm(train_batches, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(**model_inputs(model_type, batch))
            loss = outputs.loss
            loss.backward()

            if False:
                #region --- DIAGNOSTIC PROBE START ---
                # Find the spectral_freqs parameter
                freq_param = None
                for name, param in model.named_parameters():
                    if "spectral_freqs" in name:
                        freq_param = param
                        break

                if freq_param is not None:
                    # 1. Check if Gradient exists at all
                    if freq_param.grad is None:
                        print(f"⚠️ FATAL: {name} .grad is None! The graph is broken.")
                    else:
                        # 2. Check the magnitude of the gradient
                        grad_mean = freq_param.grad.abs().mean().item()
                        grad_max = freq_param.grad.abs().max().item()
                        
                        # 3. Check the magnitude of the weights themselves
                        weight_mean = freq_param.data.abs().mean().item()
                        
                        print(f"DEBUG [{steps_performed}]: Freq Grads -> Mean: {grad_mean:.2e}, Max: {grad_max:.2e} | Weight Mean: {weight_mean:.2e}")
                        
                        # 4. CRITICAL: Check if gradients are effectively zero
                        if grad_max < 1e-8:
                            print("⚠️ WARNING: Gradients are effectively zero (Vanishing Gradient).")
                # --- DIAGNOSTIC PROBE END ---
                #endregion
                # exit()

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            steps_performed += 1

            # Evaluate every eval_every steps
            if steps_performed % eval_every == 0:
                avg_train_loss = total_train_loss / eval_every
                train_losses.append(avg_train_loss)
                total_train_loss = 0.0

                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for val_batch in tqdm(val_batches, desc=f"Evaluating at Step {steps_performed}"):
                        val_outputs = model(**model_inputs(model_type, val_batch))
                        val_loss = val_outputs.loss
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_batches)
                val_losses.append(avg_val_loss)

                # Save the best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model.load_state_dict(model.state_dict())
                    print(f"New best model found at step {steps_performed} with val loss {best_val_loss:.4f}")

                print(f"Epoch [{epoch+1}/{epochs}], Step [{steps_performed}/{total_steps}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                model.train()

    print("Training complete.")
    print("Best Validation Loss:", best_val_loss)
    torch.save(best_model.state_dict(), f"easy_{model_type}_model.pth")

def evaluate_model(model_name, model_type, model_path, test_set, batch_size=8, spectral_dims=None, report_file=None):
    """
    Evaluate the model on the test set by calculating the average loss and the accuracy of predicting the correct next token.
    """
    config = LlamaConfig.from_pretrained(model_name)
    print("Model config loaded.")

    if model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(model_name, config=config)
    else:  # graph_llama
        model = GraphLlamaForCausalLM.from_pretrained(model_name, config=config, spectral_dims=spectral_dims, strict=False)

    model.load_state_dict(torch.load(model_path))

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    print(f"Pad token exists: {pad_token_id}" if tokenizer.pad_token_id is not None else f"Pad token does not exist, using eos token as pad token: {pad_token_id}")

    test_batches = create_batches(test_set, batch_size, pad_token_id)
    
    model.eval()
    total_test_loss = 0.0
    total_correct_predictions = 0

    # move everything to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    for key in test_batches[0].keys():
        test_batches = [{k: v.to(device) for k, v in batch.items()} for batch in test_batches]
    print("Moved model and data to device.")

    report = []

    with torch.no_grad():
        for test_batch in tqdm(test_batches, desc="Evaluating"):
            test_outputs = model(**model_inputs(model_type, test_batch))
            test_loss = test_outputs.loss
            total_test_loss += test_loss.item()

            # --- 1. SHIFTING (The Fix) ---
            # remove the last logit
            shift_logits = test_outputs.logits[..., :-1, :].contiguous()
            # remove the first label
            shift_labels = test_batch["labels"][..., 1:].contiguous()

            # --- 2. CALCULATE ACCURACY ---
            # predictions are now aligned
            predictions = torch.argmax(shift_logits, dim=-1)
            
            # mask out padding (-100) so we only count the "Yes/No" token
            mask = shift_labels != -100
            correct = (predictions == shift_labels) & mask
            total_correct_predictions += correct.sum().item()

            # --- 3. GENERATE REPORT (Updated to use shifted versions) ---
            
            # create a display version of predictions where -100 is replaced by pad token
            pred_ids_for_display = predictions.clone()
            # mask based on the *shifted* labels now
            pred_ids_for_display[~mask] = pad_token_id 

            for i in range(len(test_batch["input_ids"])):
                if len(report) >= 20:
                    break
                
                # want the probability of the token that isn't -100
                valid_indices = (shift_labels[i] != -100).nonzero(as_tuple=True)[0]
                
                if len(valid_indices) > 0:
                    # get the index of the "Yes/No" token in the sequence
                    target_idx = valid_indices[0].item()
                    
                    # get the raw probability for that token from the SHIFTED logits
                    probs = torch.softmax(shift_logits[i, target_idx], dim=-1)
                    correct_label_id = shift_labels[i, target_idx].item()
                    prob_value = probs[correct_label_id].item()
                else:
                    prob_value = 0.0

                # decode for readability
                # note: input_ids are still the original full context
                untokenized_inputs = tokenizer.decode(test_batch["input_ids"][i], skip_special_tokens=True)
                
                # outputs and predictions should use the shifted versions to align with what we just tested
                untokenized_outputs = tokenizer.decode(shift_labels[i][mask[i]], skip_special_tokens=True)
                untokenized_predictions = tokenizer.decode(predictions[i][mask[i]], skip_special_tokens=True)

                report.append({
                    "input_ids": test_batch["input_ids"][i].tolist(),
                    "labels": shift_labels[i].tolist(),         # Use shifted labels
                    "predictions": pred_ids_for_display[i].tolist(), # Use aligned predictions
                    "probability": prob_value,
                    "untokenized_inputs": untokenized_inputs,
                    "untokenized_outputs": untokenized_outputs,
                    "untokenized_predictions": untokenized_predictions,
                })

    # calculate average test loss and accuracy
    avg_test_loss = total_test_loss / len(test_set)
    accuracy = total_correct_predictions / len(test_set)
    print("Test Loss:", avg_test_loss)
    print("Accuracy:", accuracy)

    if report_file is not None:
        with open(report_file, "w") as f:
            for example in report:
                f.write(json.dumps(example) + "\n")


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EPOCHS = 10
# DATASET_NAME = "processed_dataset.pt"
DATASET_NAME = "processed_easy_dataset.pt"

if __name__ == "__main__":
    # check if there is a processed dataset
    dataset_path = f"src/experiments/expressiveness/{DATASET_NAME}"
    parameters, dataset = load_dataset(dataset_path)

    # split into train and val
    train_set, val_set = split_dataset(dataset, val_ratio=0.2)
    train_set = train_set[:800]
    val_set = val_set[:200]
    print(f"Dataset split into {len(train_set)} training and {len(val_set)} validation examples.")

    # Train the GraphLlama model on the training set and evaluate on the validation set
    train_model("graph_llama", MODEL_NAME, train_set, val_set, parameters, epochs=EPOCHS, batch_size=16, eval_every=10)

    # Train the simple Llama model on the training set and evaluate on the validation set
    train_model("llama", MODEL_NAME, train_set, val_set, parameters, epochs=EPOCHS, batch_size=16, eval_every=10)

    # Evaluate the model on the test set
    evaluate_model(MODEL_NAME, "graph_llama", "easy_graph_llama_model.pth", val_set, spectral_dims=parameters['spectral_dims'], report_file="easy_graph_llama_report.jsonl")
    evaluate_model(MODEL_NAME, "llama", "easy_llama_model.pth", val_set, report_file="easy_llama_report.jsonl")
