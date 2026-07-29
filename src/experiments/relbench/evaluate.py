"""Score RelBench predictions off the LM head, with relbench's own metric implementations.

There is no head and no probe here. A binary task is read as
``logit(" yes") - logit(" no")`` at the position that predicts the answer token, in fp32 --
the point of the experiment is to measure GTLM, not an MLP trained on top of it (PLAN.md
7.1). The margin is strictly monotone in the two-way renormalized ``P(yes)``, which is what
AUROC needs, and it avoids the bf16 saturation ties a raw softmax probability would produce.

Three things here are load-bearing and each fails silently if wrong.

**Never materialize ``(B, L, V)``.** HF accumulates whatever ``preprocess_logits_for_metrics``
returns across every eval batch. Returning logits would be ~2 GB per 1k examples at Llama's
128k vocab. :func:`make_margin_preprocessor` reduces to two floats per example inside the
step, so accumulation is ``(N, 2)``.

**Sigmoid before ``task.evaluate``.** relbench's ``f1`` and ``accuracy`` threshold at
``pred >= 0.5``; on a raw unbounded margin they are meaningless (a model predicting "no"
everywhere at margin -3 still scores f1 as though it predicted all-positive). The baselines
apply it too -- ``examples/gnn_entity.py`` does ``pred = torch.sigmoid(pred)`` for binary
tasks. ``roc_auc`` and ``average_precision`` are rank-based and unmoved by it, so the headline
number is the same either way; the other three would be garbage.

**Ties collapse AUROC.** Logged explicitly as ``n_distinct``: if the margin has few distinct
values the ranking is mostly arbitrary and AUROC drifts toward 0.5 for reasons that have
nothing to do with the model (PLAN.md 11).
"""

import os

import numpy as np
import torch

from relbench.base import TaskType


# -- label tokens -------------------------------------------------------------

def answer_token_ids(tokenizer, words=(" yes", " no")):
    """Token ids for the two label words, asserting each is a single token.

    A multi-token label word would make the margin a comparison of *first* tokens, which may
    not distinguish the classes at all. Checked here rather than assumed: the fallback if it
    ever fires is a different pair of words, not a different readout.
    """
    ids = []
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"label word {word!r} is {len(encoded)} tokens ({encoded}) under "
                f"{tokenizer.name_or_path}; the logit-margin readout needs one. Pick a "
                f"single-token pair (PLAN.md 7.1 suggests ' A'/' B').")
        ids.append(encoded[0])
    if ids[0] == ids[1]:
        raise ValueError(f"label words {words} share token id {ids[0]}.")
    return tuple(ids)


# -- the readout --------------------------------------------------------------

def make_margin_preprocessor(yes_id, no_id):
    """``preprocess_logits_for_metrics``: (B, L, V) -> (B, 3), inside the eval step.

    Columns are ``(logit_yes, logit_no, true_token_id)``. The true id travels alongside
    because ``compute_metrics`` receives labels that HF may have padded differently, and
    reading the target from the same gather that produced the score removes any chance of
    the two drifting apart.
    """
    def preprocess(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        # logits[:, t] predicts token t+1, so the answer token at position t is scored by
        # the logits at t-1.
        scoring = logits[:, :-1].float()
        answers = labels[:, 1:]

        supervised = answers != -100
        if not bool(supervised.any()):
            raise ValueError("an eval batch has no supervised token; label masking is broken.")
        # First supervised position per example: the label token itself. Later positions are
        # EOS and carry no class information.
        first = torch.argmax(supervised.int(), dim=1)

        rows = torch.arange(scoring.shape[0], device=scoring.device)
        picked = scoring[rows, first]                       # (B, V)
        true = answers[rows, first].to(picked.dtype)        # (B,)
        return torch.stack([picked[:, yes_id], picked[:, no_id], true], dim=1)

    return preprocess


def make_compute_metrics(task, yes_id):
    """relbench's own metric functions, on ``sigmoid(logit_yes - logit_no)``.

    Returned keys are the metric functions' ``__name__``s, so ``metric_for_best_model``
    reads ``eval_roc_auc`` -- the same quantity, under the same name, that the baselines
    tune and report.
    """
    if task.task_type != TaskType.BINARY_CLASSIFICATION:
        raise NotImplementedError(
            f"{task.task_type} needs the numeric_text readout (PLAN.md 7.2), not implemented.")

    def compute_metrics(eval_preds):
        preds = eval_preds.predictions if hasattr(eval_preds, "predictions") else eval_preds[0]
        preds = np.asarray(preds, dtype=np.float64)
        margin = preds[:, 0] - preds[:, 1]
        y_true = (preds[:, 2] == yes_id).astype(np.float64)
        y_score = 1.0 / (1.0 + np.exp(-margin))

        out = {}
        if len(np.unique(y_true)) < 2:
            # roc_auc raises on a single-class split; a smoke run hits this legitimately.
            out["roc_auc"] = float("nan")
        else:
            for fn in task.metrics:
                out[fn.__name__] = float(fn(y_true, y_score))

        out["n_distinct"] = float(len(np.unique(np.round(margin, 6))))
        out["margin_mean"] = float(margin.mean())
        out["pos_rate"] = float(y_true.mean())
        return out

    return compute_metrics


# -- final scoring, through relbench itself -----------------------------------

def save_predictions(path, row_ids, margin, y_true):
    """Persist the raw scores. Without them, a tie-collapse question cannot be answered
    after the fact -- `n_distinct` says ties exist but not how much AUROC they cost, and the
    run would have to be repeated just to look."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, row_id=np.asarray(row_ids), margin=np.asarray(margin),
                        y_true=np.asarray(y_true))


def tied_pair_fraction(margin, y_true):
    """Share of (positive, negative) pairs sharing a score -- each contributes 0.5 to AUROC
    instead of 0 or 1, so this is exactly how much of the metric is being decided by a coin
    flip rather than by the model."""
    margin, y_true = np.asarray(margin), np.asarray(y_true)
    pos, neg = margin[y_true == 1], margin[y_true == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    values, counts_pos = np.unique(pos, return_counts=True)
    tied = sum(c * int((neg == v).sum()) for v, c in zip(values, counts_pos))
    return float(tied) / (len(pos) * len(neg))


def evaluate_split(trainer, dataset, task, split, yes_id, row_ids=None, save_to=None):
    """Score one split and cross-check against ``task.evaluate``.

    ``compute_metrics`` derives the target from the gathered answer token, which is robust
    but is *our* bookkeeping. This additionally runs relbench's own comparison against the
    task table -- positional, exactly as the baselines do -- and requires the two to agree.
    A mismatch means the built cache and the task table have drifted out of alignment, which
    is the one failure mode that produces a plausible number from wrong data.
    """
    metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=f"final_{split}")
    preds = trainer.predict(dataset, metric_key_prefix=f"pred_{split}").predictions
    preds = np.asarray(preds, dtype=np.float64)
    y_score = 1.0 / (1.0 + np.exp(-(preds[:, 0] - preds[:, 1])))

    table = task.get_table(split, mask_input_cols=False)
    if row_ids is not None and len(row_ids) != len(table.df):
        # A capped val build. Index the target rather than assuming identity -- relbench
        # only checks lengths, so an unindexed comparison here would score against the
        # wrong rows for every example past the first gap.
        import copy
        table = copy.copy(table)
        table.df = table.df.iloc[list(row_ids)].reset_index(drop=True)

    margin = preds[:, 0] - preds[:, 1]
    y_true = (preds[:, 2] == yes_id).astype(np.float64)
    if save_to:
        save_predictions(save_to, row_ids if row_ids is not None else range(len(margin)),
                         margin, y_true)

    official = task.evaluate(y_score, table)
    official["tied_pair_fraction"] = tied_pair_fraction(margin, y_true)
    for name, value in official.items():
        ours = metrics.get(f"final_{split}_{name}")
        if ours is not None and not np.isnan(ours) and abs(ours - value) > 1e-6:
            raise ValueError(
                f"{split}: our {name}={ours:.6f} disagrees with task.evaluate={value:.6f}. "
                f"The cached graphs and the task table are misaligned; predictions are being "
                f"scored against the wrong rows.")
    return {f"{split}_{k}": v for k, v in official.items()}, metrics
