"""
Generation-based, set-level evaluator for KGQA.

The teacher-forced token-EM in ``GraphTrainerV2.make_compute_metrics`` is
inadequate here: KGQA needs the model to *generate* an answer set and be scored
as a set. This module:

  * truncates each eval example's prompt node at the "Answer:" delimiter,
  * runs ``model.generate`` (greedy) from that prefix with the graph bias,
  * parses the continuation into an answer set, normalizes it, and
  * scores macro **Hits@1 / F1 / Hit\*** against the FULL gold set
    (``graph['gold_answers']``, stored by process_dataset).

``KGQAGraphTrainer`` wires this into the standard eval schedule by overriding
``evaluate`` so the generative metrics feed ``metric_for_best_model``.
"""

import re

import numpy as np
import torch

from ...utils import GraphCollatorV2, GraphTrainerV2

_ARTICLES = {"a", "an", "the"}


# --------------------------------------------------------------------------- #
# Text normalization + parsing (RoG/GNN-RAG style)
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)                      # strip punctuation
    toks = [t for t in s.split() if t not in _ARTICLES]  # drop articles
    return " ".join(toks)


def parse_answer_set(text: str):
    """Split a generated 'a, b, c' continuation into a normalized, de-duped list."""
    out, seen = [], set()
    for part in text.split(","):
        n = normalize(part)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _find_prefix_len(ids, question_end):
    """Index just past the 'Answer:' delimiter token-subsequence (start of answers)."""
    qe = list(question_end)
    for i in range(len(ids) - len(qe) + 1):
        if list(ids[i : i + len(qe)]) == qe:
            return i + len(qe)
    return None


def _set_f1(pred, gold):
    predset, goldset = set(pred), set(gold)
    if not predset and not goldset:
        return 1.0
    if not predset or not goldset:
        return 0.0
    tp = len(predset & goldset)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(predset), tp / len(goldset)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generative_eval(model, dataset, tokenizer, collator, question_end,
                    max_new_tokens=128, device=None, max_samples=None, prefix="eval"):
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    hits1, f1s, hitstar = [], [], []
    for i in range(n):
        item = dataset[i]
        pn = int(item["prompt_node"])
        ids = list(item["input_ids"][pn])
        cut = _find_prefix_len(ids, question_end)
        if cut is None:
            continue

        # Truncate the prompt node to "{question}\nAnswer:" and drop labels.
        gen_item = dict(item)
        gen_item["input_ids"] = [list(x) for x in item["input_ids"]]
        gen_item["input_ids"][pn] = ids[:cut]
        gen_item.pop("labels", None)

        batch = collator([gen_item])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0][batch["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred = parse_answer_set(text)
        gold = [normalize(a) for a in dataset.graphs[i].graph.get("gold_answers", [])]
        goldset = set(gold)
        hits1.append(1.0 if pred and pred[0] in goldset else 0.0)
        hitstar.append(1.0 if any(p in goldset for p in pred) else 0.0)
        f1s.append(_set_f1(pred, goldset))

    if was_training:
        model.train()

    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {f"{prefix}_hits1": m(hits1), f"{prefix}_f1": m(f1s), f"{prefix}_hit_star": m(hitstar)}


# --------------------------------------------------------------------------- #
# Trainer that appends generative metrics to the eval schedule
# --------------------------------------------------------------------------- #
class KGQAGraphTrainer(GraphTrainerV2):
    """GraphTrainerV2 whose ``evaluate`` also runs generative set-level scoring."""

    def __init__(self, *args, gen_tokenizer=None, gen_collator=None, question_end=None,
                 gen_max_new_tokens=128, gen_max_samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_tokenizer = gen_tokenizer
        self._gen_collator = gen_collator or GraphCollatorV2(tokenizer=gen_tokenizer)
        self._question_end = question_end
        self._gen_max_new_tokens = gen_max_new_tokens
        self._gen_max_samples = gen_max_samples

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        gen = generative_eval(
            self.model, ds, self._gen_tokenizer, self._gen_collator, self._question_end,
            max_new_tokens=self._gen_max_new_tokens, max_samples=self._gen_max_samples,
            device=self.args.device, prefix=metric_key_prefix,
        )
        metrics.update(gen)
        self.log(gen)
        return metrics
