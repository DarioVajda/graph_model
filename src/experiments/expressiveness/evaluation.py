"""
Custom v0 evaluation (v2 uses ``GraphTrainerV2``'s own prompt-span metrics).

v0 scores the Yes/No answer token directly: the prediction step smuggles the
per-example answer-token index to ``PreprocessLogits``, which extracts the two
label probabilities, and ``ComputeMetrics`` turns those into accuracy.
"""

import torch


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
            example_input_len = sum([input_ids.shape[0] for input_ids in inputs["input_ids"][i]])
            prediction_token_indices[i] = example_input_len - 2  # the second-to-last token is the prediction site

        # Turn labels into a tuple: (actual_labels, prediction_token_indices)
        labels = (labels, prediction_token_indices)

    return loss, logits, labels


class PreprocessLogits:
    def __init__(self, label_options, pad_token_id):
        self.label_options = label_options
        self.pad_token_id = pad_token_id

    def __call__(self, logits, labels):
        labels, prediction_token_indices = labels  # unpack the tuple created in the smuggled prediction step

        batch_size = logits.shape[0]
        probs = torch.zeros((batch_size, 3), device=logits.device)  # [label_1_prob, label_2_prob, label_id]

        probability_distributions = torch.softmax(logits, dim=-1)
        for i in range(batch_size):
            prediction_token_index = prediction_token_indices[i]
            if prediction_token_index == -1:
                raise ValueError("Prediction token index not found in labels.")
            for j, label_option in enumerate(self.label_options):
                probs[i, j] = probability_distributions[i, prediction_token_index, label_option[0]]
            probs[i, 2] = 0 if labels[i][-1].item() == self.label_options[0][0] else (1 if labels[i][-1].item() == self.label_options[1][0] else -1)

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
                raise ValueError("True label not found in labels.")
            if probs[i, 2] == 0 and probs[i, 0] > probs[i, 1]:
                correct += 1
            elif probs[i, 2] == 1 and probs[i, 1] > probs[i, 0]:
                correct += 1
            if probs[i, 0] > probs[i, 1]:
                positive_prediction_count += 1
            total += 1

        return {
            "classification_accuracy": float(correct) / float(total) if total > 0 else 0.0,
            "positive_prediction_ratio": float(positive_prediction_count) / float(total) if total > 0 else 0.0,
        }
