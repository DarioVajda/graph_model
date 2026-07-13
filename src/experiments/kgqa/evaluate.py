"""
Generation-based, set-level evaluator for KGQA.

The teacher-forced token-EM in ``GraphTrainerV2.make_compute_metrics`` is
inadequate here: KGQA needs the model to *generate* an answer set and be scored
as a set. This module:

  * truncates each eval example's prompt node at the "Answer:" delimiter,
  * runs ``model.generate`` (greedy) from that prefix with the graph bias,
  * parses the continuation into an answer list (separator per data-format
    version: "," for dfv2, "\n" for dfv3), and
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

import random
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
def parse_answer_list(text: str, sep: str = ","):
    """Split a generated answer-list continuation into raw (un-normalized) parts.

    ``sep`` must mirror the separator the data was BUILT with
    (``RunConfig.answer_parse_sep``): "," for dfv2 targets, "\n" for dfv3 —
    where it matches GNN-RAG's own newline-split generations exactly.
    Matching normalizes later; empty segments are dropped, so "\n\n" drift in
    generations is harmless.
    """
    return [p.strip() for p in text.split(sep) if p.strip()]


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
def eval_indices(n_total, max_samples):
    """The example indices one generative eval scores.

    When ``max_samples`` caps the split (the in-training ``gen_eval_samples``
    path — E3.3: CWQ dev is 14× WebQSP's, full-dev per checkpoint is too slow),
    the subsample is a FIXED seeded draw, not first-n: stable across checkpoints
    (comparable selection curve within a run) and across runs (same constant
    seed), and unbiased w.r.t. dataset order.
    """
    if max_samples is None or max_samples >= n_total:
        return range(n_total)
    return sorted(random.Random(f"gen-eval:{n_total}").sample(range(n_total), max_samples))


@torch.no_grad()
def generative_eval(model, dataset, tokenizer, collator, question_end,
                    max_new_tokens=128, device=None, max_samples=None, prefix="eval",
                    answer_sep=",", include_strict=False):
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device

    # Flex attention needs block-aligned lengths, but generation batches are
    # unbucketed (the prompt node must stay last, so we can't pad past it).
    # Run the whole loop on the dense eager path: decode steps (q_len == 1) use
    # it regardless, this just extends that to the prefill forward.
    impl = getattr(model.config, "graph_attn_impl", None)
    if impl == "flex":
        model.config.graph_attn_impl = "eager"

    hits1, f1s, hitstar = [], [], []
    s_hits1, s_f1s, s_hitstar = [], [], []
    for i in eval_indices(len(dataset), max_samples):
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

        pred = parse_answer_list(text, sep=answer_sep)

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
        # (opt-in via log_strict_metrics — E3.2; code path kept)
        if include_strict:
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
    out = {f"{prefix}_hits1": m(hits1), f"{prefix}_f1": m(f1s),
           f"{prefix}_hit_star": m(hitstar)}
    if include_strict:
        out.update({f"{prefix}_hits1_strict": m(s_hits1), f"{prefix}_f1_strict": m(s_f1s),
                    f"{prefix}_hit_star_strict": m(s_hitstar)})
    return out


# --------------------------------------------------------------------------- #
# Trainer that appends generative metrics to the eval schedule
# --------------------------------------------------------------------------- #
class PerDatasetEvalMixin:
    """``evaluate`` fan-out over a dict of eval datasets (E3.1, TODO_cwq).

    Eval datasets are ALWAYS scored separately, each under its own namespace:
    a ``{name: dataset}`` dict (the constructor ``eval_dataset`` or an explicit
    ``evaluate(eval_dataset=...)`` argument) yields ``{prefix}_{name}_{metric}``
    for every metric, plus the checkpoint-selection scalar
    ``{prefix}_sel_f1`` = mean of the per-dataset F1s, or the single
    ``selection_dataset`` one when set (``metric_for_best_model`` points at it).

    The in-training path recurses with the dataset's *name* (a str), which HF's
    ``get_eval_dataloader`` resolves through its keyed cache — persistent
    dataloader workers stay alive per dataset. Subclasses implement
    ``_generative_metrics(handle, prefix)`` for their arm's generation loop.
    """

    _selection_dataset = None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        target = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(target, dict):
            metrics = {}
            for name in target:
                # None-arg (in-training) recursion goes by NAME -> HF's keyed
                # dataloader cache; explicit dicts (final scoring) pass objects.
                handle = name if eval_dataset is None else target[name]
                metrics.update(self.evaluate(handle, ignore_keys,
                                             f"{metric_key_prefix}_{name}"))
            key = f"{metric_key_prefix}_sel_f1"
            sel = (metrics[f"{metric_key_prefix}_{self._selection_dataset}_f1"]
                   if self._selection_dataset else
                   float(np.mean([metrics[f"{metric_key_prefix}_{n}_f1"] for n in target])))
            metrics[key] = sel
            self.log({key: sel})
            return metrics

        metrics = super().evaluate(target, ignore_keys, metric_key_prefix)
        gen = self._generative_metrics(target, metric_key_prefix)
        metrics.update(gen)
        self.log(gen)
        return metrics


class KGQAGraphTrainer(PerDatasetEvalMixin, GraphTrainerV2):
    """GraphTrainerV2 whose ``evaluate`` also runs generative set-level scoring.

    D4 (2026-07-08): optional boundary-token loss re-weighting — a strictly
    training-side lever against recall-side under-generation (D0.1 verdict:
    under-generating stops have small logP(EOS)-logP("\\n") margins, i.e. a
    soft stopping decision). Supervised tokens equal to ``boundary_token_id``
    (the dfv3 "\\n" answer separator) get weight ``boundary_loss_weight``;
    weights are renormalized so the MEAN weight over supervised tokens is 1 —
    total loss scale (and thus the effective lr) matches the unweighted path,
    only the emphasis shifts toward the continue-vs-stop decision. Generation
    logic and target format are untouched.
    """

    def __init__(self, *args, gen_tokenizer=None, gen_collator=None, question_end=None,
                 gen_max_new_tokens=128, gen_max_samples=None, gen_answer_sep=",",
                 boundary_loss_weight=1.0, boundary_token_id=None,
                 selection_dataset=None, log_strict_metrics=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_tokenizer = gen_tokenizer
        self._gen_collator = gen_collator or GraphCollatorV2(tokenizer=gen_tokenizer)
        self._question_end = question_end
        self._gen_max_new_tokens = gen_max_new_tokens
        self._gen_max_samples = gen_max_samples
        self._gen_answer_sep = gen_answer_sep
        self._selection_dataset = selection_dataset
        self._log_strict_metrics = log_strict_metrics
        self._boundary_loss_weight = boundary_loss_weight
        self._boundary_token_id = boundary_token_id
        if boundary_loss_weight != 1.0 and boundary_token_id is None:
            raise ValueError("boundary_loss_weight != 1 requires boundary_token_id.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self._boundary_loss_weight == 1.0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = boundary_weighted_loss(
            outputs.logits, labels, self._boundary_loss_weight,
            self._boundary_token_id, num_items_in_batch=num_items_in_batch)
        return (loss, outputs) if return_outputs else loss

    def set_gen_max_samples(self, n):
        """Switch the generative-eval cap (cheap in-training cap -> full final scoring)."""
        self._gen_max_samples = n

    def _generative_metrics(self, handle, prefix):
        ds = self.eval_dataset[handle] if isinstance(handle, str) else handle
        return generative_eval(
            self.model, ds, self._gen_tokenizer, self._gen_collator, self._question_end,
            max_new_tokens=self._gen_max_new_tokens, max_samples=self._gen_max_samples,
            device=self.args.device, prefix=prefix,
            answer_sep=self._gen_answer_sep, include_strict=self._log_strict_metrics,
        )


def boundary_weighted_loss(logits, labels, boundary_weight, boundary_token_id,
                           num_items_in_batch=None):
    """Causal-LM CE with the boundary token's positions re-weighted (D4).

    Weights are renormalized so the MEAN weight over supervised tokens is 1
    (total loss scale — and thus effective lr — matches the unweighted path).
    Accounting mirrors the HF default: sum/num_items during training
    (grad-accum-correct), per-token mean when num_items is unknown (eval).
    """
    shift_logits = logits.float()[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    flat = shift_labels.view(-1)
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), flat,
        ignore_index=-100, reduction="none")
    mask = flat != -100
    weights = torch.ones_like(losses)
    weights[flat == boundary_token_id] = boundary_weight
    weighted = (losses * weights)[mask].sum() * (mask.sum() / weights[mask].sum())
    denom = num_items_in_batch if num_items_in_batch is not None else mask.sum()
    return weighted / denom
