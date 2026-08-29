"""
Tier-B scoring: ROC-AUC from a generative model, via the logit margin.

Ported from `src/experiments/relbench/evaluate.py`, which already solved this
problem for this repo, minus its relbench dependencies (metrics come from sklearn
here, and the task object is a plain spec). Keeping the readout *identical* is the
point: it is the same quantity, computed the same way, so a molecule AUROC and a
relbench AUROC mean the same thing.

Score = ``logit(" Yes") - logit(" No")`` at the answer position, in fp32. Not the
softmax probability: the margin is strictly monotone in it, avoids bf16 saturation
ties, and AUROC/AP care only about ranking.

**Two traps carried over from that experiment, both silent if missed.**

1. `sigmoid` before the threshold metrics. `roc_auc` and `average_precision` are
   rank-based and unmoved by it, but `accuracy` and `f1` threshold at 0.5 and are
   meaningless on an unbounded margin.
2. Tie collapse. bf16 quantises the margin to ~1/8 (`project-gtlm-margin-quantization`),
   and a saturated model can produce a handful of distinct scores across a whole
   split. `n_distinct` and `tied_pair_fraction` are in every record from the first
   run, not retrofitted after a suspicious number. HIV (~3.5% positives) is where
   this will bite hardest.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

YES_WORD, NO_WORD = " Yes", " No"


def answer_token_ids(tokenizer, words=(YES_WORD, NO_WORD)):
    """Token ids for the two label words, asserting each is a single token.

    A multi-token label word would make the margin a comparison of *first* tokens,
    which may not distinguish the classes at all. Checked rather than assumed.
    """
    ids = []
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"label word {word!r} is {len(encoded)} tokens ({encoded}) under "
                f"{tokenizer.name_or_path}; the logit-margin readout needs one.")
        ids.append(encoded[0])
    if ids[0] == ids[1]:
        raise ValueError(f"label words {words} share token id {ids[0]}.")
    return tuple(ids)


def make_margin_preprocessor(yes_id, no_id):
    """``preprocess_logits_for_metrics``: (B, L, V) -> (B, 3), inside the eval step.

    Columns are ``(logit_yes, logit_no, true_token_id)``. The true id travels with
    the score because reading the target from the same gather removes any chance of
    the two drifting apart. Never materialises ``(B, L, V)`` across the eval loop.
    """
    def preprocess(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        # logits[:, t] predicts token t+1, so the answer token at position t is
        # scored by the logits at t-1.
        scoring = logits[:, :-1].float()
        answers = labels[:, 1:]

        supervised = answers != -100
        if not bool(supervised.any()):
            raise ValueError("an eval batch has no supervised token; label masking is broken.")
        first = torch.argmax(supervised.int(), dim=1)

        rows = torch.arange(scoring.shape[0], device=scoring.device)
        picked = scoring[rows, first]                    # (B, V)
        true = answers[rows, first].to(picked.dtype)     # (B,)
        return torch.stack([picked[:, yes_id], picked[:, no_id], true], dim=1)

    return preprocess


def tied_pair_fraction(margin, y_true):
    """Share of (positive, negative) pairs sharing a score.

    Each tied pair contributes 0.5 to AUROC instead of 0 or 1, so this is exactly
    how much of the metric is decided by a coin flip rather than by the model.
    """
    margin, y_true = np.asarray(margin), np.asarray(y_true)
    pos, neg = margin[y_true == 1], margin[y_true == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    values, counts_pos = np.unique(pos, return_counts=True)
    tied = sum(c * int((neg == v).sum()) for v, c in zip(values, counts_pos))
    return float(tied) / (len(pos) * len(neg))


def make_margin_metrics(yes_id):
    """``compute_metrics`` for a binary Tier-B task, on ``sigmoid(margin)``."""
    def compute_metrics(eval_preds):
        preds = (eval_preds.predictions if hasattr(eval_preds, "predictions")
                 else eval_preds[0])
        preds = np.asarray(preds, dtype=np.float64)
        margin = preds[:, 0] - preds[:, 1]
        y_true = (preds[:, 2] == yes_id).astype(np.float64)
        y_score = 1.0 / (1.0 + np.exp(-margin))          # trap 1: before thresholding

        out = {}
        if len(np.unique(y_true)) < 2:
            # A single-class split is legitimate on a smoke run; AUROC is undefined.
            out.update(roc_auc=float("nan"), average_precision=float("nan"))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["average_precision"] = float(average_precision_score(y_true, y_score))
        y_pred = (y_score >= 0.5).astype(np.float64)
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        # trap 2: is the metric being decided by the model or by bf16 rounding?
        out["n_distinct"] = float(len(np.unique(np.round(margin, 6))))
        out["tied_pair_fraction"] = tied_pair_fraction(margin, y_true)
        out["margin_mean"] = float(margin.mean())
        out["pos_rate"] = float(y_true.mean())
        return out

    return compute_metrics
