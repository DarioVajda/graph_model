"""
Generation-based, set-level evaluator for KGQA.

The teacher-forced token-EM in ``GraphTrainerV2.make_compute_metrics`` is
inadequate here: KGQA needs the model to *generate* an answer set and be scored
as a set. This module:

  * truncates each eval example's prompt node at the "Answer:" delimiter,
  * runs ``model.generate`` (greedy) from that prefix with the graph bias,
  * parses the comma-separated continuation into an answer list, and
  * scores macro **Hits@1 / F1 / Hit\*** against the FULL gold set
    (``graph['gold_answers']``, stored by process_dataset).

The primary metrics replicate **GNN-RAG's** ``evaluate_results.py`` verbatim
(normalized-substring ``match``; F1 whose precision denominator is the parsed
prediction count while matching runs over the joined prediction string;
Hits@1 = first parsed answer, Hit = any) so our numbers are directly comparable
to their reported ones. Our stricter exact-set-equality variants are logged too,
suffixed ``_strict``.

``KGQAGraphTrainer`` wires this into the standard eval schedule by overriding
``evaluate`` so the generative metrics feed ``metric_for_best_model``.
"""

import re
import string

import numpy as np
import torch

from ...utils import GraphCollatorV2, GraphTrainerV2


# --------------------------------------------------------------------------- #
# GNN-RAG scoring, ported verbatim from llm/src/qa_prediction/evaluate_results.py
# (github.com/cmavro/GNN-RAG) — keep in lockstep with theirs for fair comparison.
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    """GNN-RAG answer match: normalized gold (s2) is a substring of prediction (s1)."""
    return normalize(s2) in normalize(s1)


def eval_f1(prediction, answer):
    """GNN-RAG F1: golds matched against the JOINED prediction string; precision
    denominator = number of parsed prediction items."""
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def eval_hit1(prediction, answer):
    """1 iff the FIRST parsed prediction matches any gold."""
    for a in answer:
        if match(prediction[0], a):
            return 1
    return 0


def eval_hit(prediction, answer):
    """1 iff any gold appears in the joined prediction string (our Hit*)."""
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            return 1
    return 0


# --------------------------------------------------------------------------- #
# Parsing + strict secondary metrics
# --------------------------------------------------------------------------- #
def parse_answer_list(text: str):
    """Split a generated 'a1, a2, ...' continuation into raw (un-normalized) parts.

    GNN-RAG splits its generations on newlines; ours are trained to be
    comma-separated, so the comma is our delimiter. Matching normalizes later.
    """
    return [p.strip() for p in text.split(",") if p.strip()]


def _find_prefix_len(ids, question_end):
    """Index just past the 'Answer:' delimiter token-subsequence (start of answers)."""
    qe = list(question_end)
    for i in range(len(ids) - len(qe) + 1):
        if list(ids[i : i + len(qe)]) == qe:
            return i + len(qe)
    return None


def _strict_set_f1(pred, gold):
    """Exact-equality set-F1 over normalized strings (our stricter secondary metric)."""
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

    # Flex attention needs block-aligned lengths, but generation batches are
    # unbucketed (the prompt node must stay last, so we can't pad past it).
    # Run the whole loop on the dense eager path: decode steps (q_len == 1) use
    # it regardless, this just extends that to the prefill forward.
    impl = getattr(model.config, "graph_attn_impl", None)
    if impl == "flex":
        model.config.graph_attn_impl = "eager"

    hits1, f1s, hitstar = [], [], []
    s_hits1, s_f1s, s_hitstar = [], [], []
    for i in range(n):
        item = dataset[i]
        pn = int(item["prompt_node"])
        ids = list(item["input_ids"][pn])
        cut = _find_prefix_len(ids, question_end)
        gold = [a for a in dataset.graphs[i].graph.get("gold_answers", []) if a]
        if cut is None or not gold:
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

        pred = parse_answer_list(text)

        # ── primary: GNN-RAG metrics (benchmark-comparable) ──
        if pred:
            hits1.append(float(eval_hit1(pred, gold)))
            hitstar.append(float(eval_hit(pred, gold)))
            f1s.append(float(eval_f1(pred, gold)[0]))
        else:
            hits1.append(0.0)
            hitstar.append(0.0)
            f1s.append(0.0)

        # ── secondary: strict exact-set-equality on normalized strings ──
        pred_n, goldset = [], set(normalize(a) for a in gold)
        for p in pred:
            pn_ = normalize(p)
            if pn_ and pn_ not in pred_n:
                pred_n.append(pn_)
        s_hits1.append(1.0 if pred_n and pred_n[0] in goldset else 0.0)
        s_hitstar.append(1.0 if any(p in goldset for p in pred_n) else 0.0)
        s_f1s.append(_strict_set_f1(pred_n, goldset))

    if impl == "flex":
        model.config.graph_attn_impl = "flex"
    if was_training:
        model.train()

    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        f"{prefix}_hits1": m(hits1), f"{prefix}_f1": m(f1s), f"{prefix}_hit_star": m(hitstar),
        f"{prefix}_hits1_strict": m(s_hits1), f"{prefix}_f1_strict": m(s_f1s),
        f"{prefix}_hit_star_strict": m(s_hitstar),
    }


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

    def set_gen_max_samples(self, n):
        """Switch the generative-eval cap (cheap in-training cap -> full final scoring)."""
        self._gen_max_samples = n

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
