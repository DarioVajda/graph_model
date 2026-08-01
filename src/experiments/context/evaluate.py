"""
Loss, metrics and grid scoring for the context experiment.

Everything here exists because of one fact: the supervised span is ~4 tokens at
the end of a sequence that is up to 65k tokens long. Running the model the
ordinary way computes an ``(B, L, V)`` logit tensor — 4.2 GB of bf16 at L=16k,
16.5 GB at L=64k — of which all but the last handful of rows multiply into
``-100`` labels.

``GTLMLlamaForCausalLM.forward`` already accepts ``logits_to_keep`` (an int or an
index tensor) and applies ``lm_head`` only to that slice, but it then hands the
FULL labels to HuggingFace's loss function, which mis-shapes. So the slice cannot
be passed through the stock Trainer, and the two places that would materialize
the full tensor — training loss and evaluation — are overridden here instead:

  * :meth:`ContextGraphTrainer.compute_loss` — sliced forward + its own shifted CE.
  * :meth:`ContextGraphTrainer.prediction_step` — sliced forward, returning the
    logits **and labels of the same window** so that the shared
    ``shift_logits_for_metrics`` / ``make_compute_metrics`` helpers keep working
    unchanged and best-checkpoint selection goes through the normal HF path.

Nothing in ``src/models/`` changes.

Metric (README §A.9): **teacher-forced greedy exact match** — the
answer tokens are present in the input and the model must place the argmax on
every one of them (plus EOS). It upper-bounds free-running greedy EM because
there is no error propagation across the answer; name it "teacher-forced EM" in
any figure caption.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ...utils import GraphTrainerV2


# ── The window ─────────────────────────────────────────────────────────────────

def window_start(labels):
    """First index of the logits window: one before the earliest supervised label.

    The logit at position ``t`` predicts the token at ``t+1``, so predicting the
    first supervised label needs the logit one position earlier. Taking the
    minimum over the batch makes one window serve every row (they are equal under
    cell-homogeneous batching; padding only ever adds a few positions).
    """
    supervised = labels != -100
    if not bool(supervised.any()):
        return max(0, labels.shape[1] - 1)
    first = torch.argmax(supervised.int(), dim=1).min().item()
    return max(0, int(first) - 1)


def windowed_forward(model, inputs, labels, **kwargs):
    """Forward pass that only computes logits over the supervised window.

    Returns ``(logits, window_labels)`` — both covering positions
    ``[start, L)``, i.e. aligned exactly as the full-sequence tensors would be.
    """
    start = window_start(labels)
    keep = labels.shape[1] - start
    outputs = model(**inputs, logits_to_keep=keep, **kwargs)
    return outputs.logits, labels[:, start:]


def windowed_loss(logits, window_labels, num_items_in_batch=None):
    """Shifted cross-entropy over the window (the stock causal-LM loss, sliced)."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = window_labels[:, 1:].to(shift_logits.device)
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    if num_items_in_batch is not None:
        # Match HF's ForCausalLMLoss: sum / the batch's token count, so gradient
        # accumulation normalizes over tokens rather than over micro-batches.
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100,
                               reduction="sum") / num_items_in_batch
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)


# ── Trainer ────────────────────────────────────────────────────────────────────

class ContextGraphTrainer(GraphTrainerV2):
    """``GraphTrainerV2`` with a windowed loss/eval and cell-homogeneous batching.

    ``train_sampler`` is injected rather than built here: the sampler needs the
    per-graph cell, which the dataset knows and this class does not.
    """

    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def _get_train_sampler(self, *args, **kwargs):
        if self._train_sampler is not None:
            return self._train_sampler
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        logits, window_labels = windowed_forward(model, inputs, labels)
        loss = windowed_loss(logits, window_labels, num_items_in_batch)
        if return_outputs:
            return loss, {"logits": logits}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluate on the same window, returning aligned logits + labels.

        HF's own loop would call the model with ``labels`` and no
        ``logits_to_keep``; ``preprocess_logits_for_metrics`` cannot save us
        because it runs only *after* the full logits exist.
        """
        inputs = self._prepare_inputs(inputs)
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        with torch.no_grad():
            logits, window_labels = windowed_forward(model, inputs, labels)
            loss = windowed_loss(logits, window_labels).detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits.detach(), window_labels)


# ── Grid scoring ───────────────────────────────────────────────────────────────

def _classify(pred_ids, gold_ids, pred_text, gold_code, distractor_codes):
    """Exact match, and — when it fails — *why* (README §A.9).

    A failure that names another node's code is a **selection** failure (the model
    retrieved, from the wrong node); one that names no code at all is a
    **representation** failure. The ratio between them across the grid is the
    mechanistic story behind wherever the accuracy contour falls.
    """
    em = pred_ids == gold_ids
    if em:
        return "em"
    if pred_text in distractor_codes:
        return "distractor"
    if pred_text == gold_code:
        # Right code, wrong tokenization/EOS — count separately from a clean hit.
        return "code_no_eos"
    return "malformed"


@torch.no_grad()
def grid_eval(model, dataset, collator, tokenizer, device=None, batch_size=1,
              max_samples=None, verbose=False):
    """Teacher-forced greedy EM over one grid cell (or any built split).

    Returns a metrics dict: ``em`` plus the failure decomposition, the count, and
    the packed length actually fed to the model (what the heatmap annotates).
    """
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device

    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    counts = {"em": 0, "distractor": 0, "code_no_eos": 0, "malformed": 0}
    code_correct = 0
    packed_len = None

    for lo in range(0, n, batch_size):
        items = [dataset[i] for i in range(lo, min(lo + batch_size, n))]
        batch = collator(items)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        labels = batch.pop("labels")
        packed_len = int(batch["input_ids"].shape[1])

        logits, window_labels = windowed_forward(model, batch, labels)
        preds = logits[:, :-1, :].argmax(dim=-1)
        gold = window_labels[:, 1:]

        for row in range(gold.shape[0]):
            g = dataset.graphs[lo + row].graph
            mask = gold[row] != -100
            pred_ids = preds[row][mask].tolist()
            gold_ids = gold[row][mask].tolist()
            # Drop the trailing EOS before decoding, so the text is just the code.
            text_ids = [i for i in pred_ids if i != tokenizer.eos_token_id]
            pred_text = tokenizer.decode(text_ids).strip()
            distractors = {c for c in g.get("codes", []) if c != g["gold_code"]}
            counts[_classify(pred_ids, gold_ids, pred_text, g["gold_code"], distractors)] += 1
            # Retrieval, independent of the EOS convention: the code tokens alone.
            # The supervised span is [code tokens..., EOS], so dropping the last
            # position compares exactly the code. An UNTRAINED arm scores em=0
            # everywhere because it continues the text instead of emitting EOS,
            # which says nothing about whether it found the needle — this is the
            # metric the arms can actually be compared on.
            if pred_ids[:-1] == gold_ids[:-1]:
                code_correct += 1

        if verbose and (lo // max(1, batch_size)) % 25 == 0:
            print(f"[grid_eval] {lo + len(items)}/{n}")

    if was_training:
        model.train()

    total = max(1, sum(counts.values()))
    return {
        "em": counts["em"] / total,
        "code_acc": code_correct / total,
        "distractor_rate": counts["distractor"] / total,
        "code_no_eos_rate": counts["code_no_eos"] / total,
        "malformed_rate": counts["malformed"] / total,
        "n": sum(counts.values()),
        "packed_len": packed_len,
    }


def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for ``k`` successes in ``n`` trials.

    Reported per cell because at n=200 the half-width is ~6.9 pp at p=0.5: the grid
    resolves a transition contour, not 5 pp cell-to-cell differences (README §A.9).
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
