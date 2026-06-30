"""
Custom v0 evaluation (v2 uses ``GraphTrainerV2``'s own prompt-span metrics).

v0 reports **exact-match accuracy** (``em_accuracy``) — the same metric v2 uses —
so the two implementations are directly comparable. The prediction step smuggles
the per-example answer-token index to ``PreprocessLogits``, which takes the
``argmax`` over the full vocabulary at that site (NOT just the two label options)
and pairs it with the true answer token; ``ComputeMetrics`` then reports the
fraction of exact matches. For the single-token `` Yes``/`` No`` answer this is
identical to v2's prompt-span exact match.
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
    """Reduce (B, L, V) logits to (B, 2) = [predicted_token, true_token] at the
    answer site, so the eval loop accumulates a tiny tensor instead of full
    logits. The prediction is the ``argmax`` over the *whole* vocabulary (exact
    match), not a restricted Yes/No comparison."""

    def __init__(self, label_options, pad_token_id):
        self.label_options = label_options
        self.pad_token_id = pad_token_id

    def __call__(self, logits, labels):
        labels, prediction_token_indices = labels  # unpacked from the smuggled prediction step

        batch_size = logits.shape[0]
        out = torch.zeros((batch_size, 2), dtype=torch.long, device=logits.device)  # [pred_token, true_token]
        for i in range(batch_size):
            prediction_token_index = prediction_token_indices[i]
            if prediction_token_index == -1:
                raise ValueError("Prediction token index not found in labels.")
            out[i, 0] = torch.argmax(logits[i, prediction_token_index])  # predicted token over full vocab
            out[i, 1] = labels[i][-1]                                    # true answer token
        return out


class ComputeMetrics:
    """Exact-match accuracy from (B, 2) = [pred_token, true_token]; also reports
    how often the model emitted the first label option (`` Yes``) so we can spot a
    collapsed/degenerate predictor."""

    def __init__(self, label_options):
        self.label_options = label_options

    def __call__(self, eval_preds):
        preds, _ = eval_preds                     # preds: (N, 2) = [pred_token, true_token]
        pred_tok, true_tok = preds[:, 0], preds[:, 1]
        total = len(pred_tok)
        if total == 0:
            return {"em_accuracy": 0.0, "positive_prediction_ratio": 0.0}
        correct = float((pred_tok == true_tok).sum())
        positive = float((pred_tok == self.label_options[0][0]).sum())  # predicted " Yes"
        return {
            "em_accuracy": correct / total,
            "positive_prediction_ratio": positive / total,
        }
