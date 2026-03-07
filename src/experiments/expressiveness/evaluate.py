from ...utils import TextGraphDataset, GraphCollator
from ...models.llama_attn_bias import GraphLlamaForCausalLM

from .data_gen import create_and_save_dataset, dataset_path_and_size

from transformers import AutoTokenizer
import torch
import os
import random
from tqdm import tqdm

from accelerate.utils import send_to_device

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()


def evaluate_model(model, test_dataset, label_options):
    model.eval()
    model.to(DEVICE)
    collator = GraphCollator()

    correct_predictions = 0
    total_predictions = 0

    for i, example in enumerate(tqdm(test_dataset)):
    # for i, example in enumerate(test_dataset):
        batched_inputs = collator([example])
        batched_inputs = send_to_device(batched_inputs, DEVICE)

        labels = batched_inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Labels must be provided in the inputs for loss computation.")
        
        # FORWARD PASS
        outputs = model(
            input_ids=None, 
            input_graph_batch=batched_inputs, 
            labels=labels, 
            output_logits=True,
        )

        # compare the predicted logits on the second-to-last token for the two label options (e.g. " Yes" and " No")
        logits = outputs.logits
        prediction_logits = logits[0, -2, :]
        prediction_distribution = torch.softmax(prediction_logits, dim=-1)
        # print("P(Yes):", prediction_distribution[label_options[0]].item())
        # print("P(No):", prediction_distribution[label_options[1]].item())

        model_prediction = "Yes" if prediction_distribution[label_options[0]].item() > prediction_distribution[label_options[1]].item() else "No"
        # print("Model Prediction:", model_prediction)

        true_label = "Yes" if labels[0][-1].item() == label_options[0][0] else ("No" if labels[0][-1].item() == label_options[1][0] else "Unknown")
        # print("True Label:", true_label)
        # print("-" * 50)
        # print(f"{i+1} - True: {true_label:3}, Predicted: {model_prediction:3}")

        total_predictions += 1
        if true_label != "Unknown" and model_prediction != "Unknown" and model_prediction == true_label:
            # print("Correct prediction!")
            correct_predictions += 1

    print(f"Accuracy: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions:.4f}")



if __name__ == "__main__":
    # Load the test dataset
    TEST_DATASET_SIZE = 1_000
    EASY = False
    test_dataset_path, _ = dataset_path_and_size(TEST_DATASET_SIZE, easy=EASY)
    if not os.path.exists(test_dataset_path):
        print(f"Test dataset not found at {test_dataset_path}. Creating new dataset...")
        create_and_save_dataset(dataset_size=TEST_DATASET_SIZE, min_nodes=10, max_nodes=20, spectral_dims=16, model_name="meta-llama/Llama-3.2-1B", easy=EASY)
    test_dataset = TextGraphDataset.load(test_dataset_path)

    # Load the model
    # trained_model_path = "./checkpoints/HARD_magnetic(dim=32,q=0.25)/checkpoint-3120"
    trained_model_path = "./checkpoints/HARD_lora_rrwp(16)+magnetic(dim=32,q=0.25)/checkpoint-3120"
    model = GraphLlamaForCausalLM.from_pretrained(trained_model_path, attn_implementation="eager")
    print(f"Loaded model from {trained_model_path}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    possible_labels = [" Yes", " No" ]
    tokenized_possible_labels = [tokenizer(label, add_special_tokens=False).input_ids for label in possible_labels]

    # Evaluate the model on the test dataset
    evaluate_model(model, test_dataset, label_options=tokenized_possible_labels)