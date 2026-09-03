"""
The scorers `in_mixture` and `held_out` share (D7.3).

One function, :func:`score_source`, takes a built ``TaskSource`` and its
``TaskSpec`` and returns the metrics for that task's answer kind (D1.1):

===========  ==========================  ===========================================
kind         how                         metrics
===========  ==========================  ===========================================
``token``    teacher-forced              ``em_accuracy``
``yesno``    teacher-forced logit margin ``roc_auc`` … + the tie diagnostics
``text``     greedy generation           ``bleu2`` ``bleu4`` ``rouge_l`` ``meteor``
``smiles``   greedy generation           ``validity`` ``roundtrip_match`` ``exact_match``
===========  ==========================  ===========================================

**Nothing here is a second implementation of a metric that already exists.** The
exact-match path is `src/utils`'s ``make_compute_metrics`` and
``shift_logits_for_metrics``; the margin path is `molecules/evaluate.py`'s
``make_margin_preprocessor`` / ``make_margin_metrics``, which carries the
sigmoid-before-thresholding trap and the tie diagnostics with it; the SMILES path
is `adapters/molecules.py`'s ``smiles_scores``; the index subsample is kgqa's
``eval_indices``. Only the caption metrics are written out, and only because
nothing in the image implements them (see `captions.py`).

**Why both ``n_distinct`` and ``tied_pair_fraction`` are always reported on a
``yesno`` task.** The score is a logit margin read in fp32 off a bf16 forward,
and bf16 quantises it to about 1/8 (`project-gtlm-margin-quantization`). A low
``n_distinct`` is therefore a *precision artifact* and not by itself evidence
that the model has collapsed — the quantisation puts a floor under how many
distinct values a split can show whatever the model does. ``tied_pair_fraction``
is the quantity that actually bounds the claim: every tied (positive, negative)
pair contributes 0.5 to AUROC instead of 0 or 1, so it says exactly how much of
the reported number was decided by a coin flip. Reporting one without the other
invites the wrong reading in both directions, so the pair travels together and
`in_mixture` declares both.

Generation is greedy (``do_sample=False``, ``num_beams=1``) with the task's
``max_new_tokens`` from the registry, from a prompt truncated at the supervised
span — the answer boundary the schema already fixed, read back off the item's
``labels`` column, so the evaluation prompt is byte-identical to the training
prompt up to the answer.
"""

from __future__ import annotations

__all__ = [
    "answer_start", "eval_indices", "endpoints_of", "generate_predictions",
    "margin_array", "score_source", "teacher_forced",
]

#: How many examples a validator scores per task by default. Generation is the
#: cost here — a caption at 128 new tokens is two orders of magnitude dearer than
#: a teacher-forced yes/no — and a cadence of every 500 steps means this runs
#: dozens of times in a run. The cap is a fixed seeded draw, not the first *n*,
#: so the curve within a run and the comparison across runs are both stable
#: (kgqa's ``eval_indices``). ``None`` scores the whole split, which is what the
#: end-of-run and milestone firings use.
DEFAULT_MAX_SAMPLES = None

#: Micro-batch for the teacher-forced paths. Small because eval batches are not
#: bucketed and a mixed-length batch pads to its longest member.
DEFAULT_BATCH_SIZE = 8


def eval_indices(n_total: int, max_samples):
    """kgqa's fixed seeded subsample, reused rather than re-derived."""
    from ...experiments.kgqa.evaluate import eval_indices as _eval_indices

    return list(_eval_indices(n_total, max_samples))


def answer_start(item: dict) -> int:
    """First supervised position in the prompt node's tokens.

    ``labels`` is aligned to the *prompt node's* token list (D1.2 and the
    collator's contract), so this is the index the answer begins at within that
    node — which is exactly where a generation prompt has to be cut.
    """
    labels = item.get("labels")
    if labels is None:
        raise ValueError(
            "the item carries no 'labels' column, so the answer boundary cannot "
            "be located; generation would score the model against a prompt it "
            "was never given")
    for i, value in enumerate(labels):
        if int(value) != -100:
            return i
    raise ValueError("the item's labels are entirely -100: no supervised span")


def _sidecar(item: dict) -> dict:
    from ..schema import SIDECAR_KEY

    return item.get(SIDECAR_KEY) or {}


def endpoints_of(source, indices) -> list:
    """``meta["endpoint"]`` per selected example, ``None`` where there is none.

    Tox21 and SIDER are one task each with ~12 and ~27 endpoints
    (`MOLECULE_GENERALIST.md` §1); a single AUROC over their union mixes
    populations with wildly different base rates and is not a number anyone can
    act on, so the endpoint travels in ``meta`` and the breakdown is reported
    beside the pooled figure.
    """
    return [(_sidecar(source[i]).get("meta") or {}).get("endpoint") for i in indices]


def _answers(source, indices) -> list:
    return [_sidecar(source[i]).get("answer", "") for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Teacher-forced
# ─────────────────────────────────────────────────────────────────────────────

def _batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def _to_device(batch, device):
    import torch

    if device is None:
        return batch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def teacher_forced(model, collator, source, indices, device=None,
                   batch_size: int = DEFAULT_BATCH_SIZE, preprocess=None):
    """Run the forward pass over ``indices`` and return ``(predictions, labels)``.

    ``preprocess(logits, labels)`` is HF's ``preprocess_logits_for_metrics`` — the
    same callables the molecules trainer passes, so the reduction from
    ``(B, L, V)`` to something storable happens inside the loop and the full
    logits are never held across it.

    Rows from different batches are padded to a common width before stacking:
    eval batches here are unbucketed, so two batches genuinely differ in ``L``,
    and the exact-match scorer reads each row under its own label mask, for which
    the pad value ``-100`` is inert.
    """
    import numpy as np
    import torch

    from ..schema import SIDECAR_KEY

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for chunk in _batches(list(indices), batch_size):
            items = [{k: v for k, v in source[i].items() if k != SIDECAR_KEY}
                     for i in chunk]
            batch = _to_device(collator(items), device)
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = out.logits if hasattr(out, "logits") else out[0]
            label = batch["labels"]
            reduced = preprocess(logits, label) if preprocess is not None else logits
            preds.append(reduced.detach().float().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    if was_training and hasattr(model, "train"):
        model.train()
    if not preds:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=np.int64)
    return _pad_stack(preds, 0.0), _pad_stack(labels, -100)


def _pad_stack(arrays, pad):
    """Stack ``(B_i, W_i)`` blocks into one ``(sum B_i, max W_i)`` array."""
    import numpy as np

    width = max(a.shape[1] for a in arrays)
    if all(a.shape[1] == width for a in arrays):
        return np.concatenate(arrays, axis=0)
    out = []
    for a in arrays:
        if a.shape[1] < width:
            a = np.pad(a, ((0, 0), (0, width - a.shape[1])), constant_values=pad)
        out.append(a)
    return np.concatenate(out, axis=0)


def margin_array(model, tokenizer, collator, source, indices, device=None,
                 batch_size: int = DEFAULT_BATCH_SIZE):
    """The ``(N, 3)`` ``(logit_yes, logit_no, true_token_id)`` readout, in order.

    Exactly what `molecules/evaluate.py`'s preprocessor produces inside the
    trainer's eval loop, so an AUROC computed here and one computed there are the
    same quantity.
    """
    from ...experiments.molecules.evaluate import (
        answer_token_ids, make_margin_preprocessor,
    )

    yes_id, no_id = answer_token_ids(tokenizer)
    preds, _labels = teacher_forced(
        model, collator, source, indices, device=device, batch_size=batch_size,
        preprocess=make_margin_preprocessor(yes_id, no_id))
    return preds, yes_id


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_predictions(model, tokenizer, collator, source, indices,
                         max_new_tokens: int = 64, device=None) -> tuple:
    """Greedy continuations from the answer boundary. ``(predictions, targets)``.

    One example per call: generation batches cannot be bucketed (the prompt node
    must stay last in the packed sequence, so nothing may be padded past it),
    which is the same constraint kgqa's generative eval works under and the
    reason it also runs one at a time.

    Flex attention needs block-aligned lengths and these batches are not aligned,
    so the whole loop runs on the dense eager path — decode steps use it anyway,
    this only extends that to the prefill.
    """
    import torch

    from ..schema import SIDECAR_KEY

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    config = getattr(model, "config", None)
    impl = getattr(config, "graph_attn_impl", None)
    if impl == "flex":
        config.graph_attn_impl = "eager"

    predictions, targets = [], []
    with torch.no_grad():
        for i in indices:
            item = source[i]
            side = _sidecar(item)
            start = answer_start(item)
            prompt_node = int(item["prompt_node"])

            gen = {k: v for k, v in item.items() if k not in (SIDECAR_KEY, "labels")}
            gen["input_ids"] = [list(x) for x in item["input_ids"]]
            gen["input_ids"][prompt_node] = gen["input_ids"][prompt_node][:start]

            batch = _to_device(collator([gen]), device)
            out = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=getattr(tokenizer, "eos_token_id", None))
            new_tokens = out[0][batch["input_ids"].shape[1]:]
            predictions.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
            targets.append(side.get("answer", ""))

    if impl == "flex":
        config.graph_attn_impl = "flex"
    if was_training and hasattr(model, "train"):
        model.train()
    return predictions, targets


# ─────────────────────────────────────────────────────────────────────────────
# The metric leaves, per answer kind
# ─────────────────────────────────────────────────────────────────────────────

#: What :func:`score_source` returns for each answer kind. Declared as data
#: because `in_mixture` and `held_out` have to state their keys *before* they run
#: (D7.1) and the only thing that decides them is the kind.
METRIC_KEYS = {
    "token": ("em_accuracy", "n"),
    "yesno": ("roc_auc", "average_precision", "accuracy", "f1", "n_distinct",
              "tied_pair_fraction", "margin_mean", "pos_rate", "n"),
    "text": ("bleu2", "bleu4", "rouge_l", "meteor", "n"),
    "smiles": ("validity", "roundtrip_match", "exact_match",
               "stereo_marks_emitted", "n"),
}

#: Sub-key an endpoint breakdown lands under: ``endpoint:NR-AR/roc_auc``.
ENDPOINT_PREFIX = "endpoint:"


def score_source(model, tokenizer, collator, source, spec, device=None,
                 max_samples=DEFAULT_MAX_SAMPLES,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 per_endpoint: bool = True) -> dict:
    """Score one built ``(task, split, arm)`` and return its metrics.

    Keys are metric leaves, except the per-endpoint breakdown which is
    ``endpoint:<name>/<leaf>``. The caller prefixes the task and split.
    """
    kind = spec.answer_kind
    if kind not in METRIC_KEYS:
        raise ValueError(f"{spec.name}: answer_kind {kind!r} has no scorer")

    n_total = len(source)
    indices = eval_indices(n_total, max_samples) if n_total else []

    if kind == "token":
        return _score_token(model, collator, source, indices, device, batch_size)
    if kind == "yesno":
        return _score_yesno(model, tokenizer, collator, source, indices, device,
                            batch_size, per_endpoint)
    predictions, targets = generate_predictions(
        model, tokenizer, collator, source, indices,
        max_new_tokens=spec.max_new_tokens or 64, device=device)
    if kind == "smiles":
        from ..adapters.molecules import smiles_scores

        return dict(smiles_scores(predictions, targets))
    from .captions import caption_metrics

    return dict(caption_metrics(predictions, targets))


def _score_token(model, collator, source, indices, device, batch_size) -> dict:
    from ...utils import make_compute_metrics, shift_logits_for_metrics

    if not len(indices):
        return {"em_accuracy": 0.0, "n": 0}
    preds, labels = teacher_forced(
        model, collator, source, indices, device=device, batch_size=batch_size,
        preprocess=shift_logits_for_metrics)
    out = make_compute_metrics()((preds.astype("int64"), labels.astype("int64")))
    return {"em_accuracy": float(out["em_accuracy"]), "n": len(indices)}


def _score_yesno(model, tokenizer, collator, source, indices, device, batch_size,
                 per_endpoint) -> dict:
    from ...experiments.molecules.evaluate import make_margin_metrics

    if not len(indices):
        return {k: (0 if k == "n" else float("nan")) for k in METRIC_KEYS["yesno"]}

    preds, yes_id = margin_array(model, tokenizer, collator, source, indices,
                                 device=device, batch_size=batch_size)
    compute = make_margin_metrics(yes_id)
    out = {k: float(v) for k, v in compute((preds,)).items()}
    out["n"] = len(indices)

    if per_endpoint:
        endpoints = endpoints_of(source, indices)
        groups: dict = {}
        for row, endpoint in enumerate(endpoints):
            if endpoint is not None:
                groups.setdefault(str(endpoint), []).append(row)
        # A single-endpoint corpus (BACE, BBBP, HIV) would only restate the
        # pooled number under a second name, and a duplicated number reads as an
        # independent measurement in any table that averages these.
        if len(groups) > 1:
            for endpoint, rows in sorted(groups.items()):
                scored = compute((preds[rows],))
                for key, value in scored.items():
                    out[f"{ENDPOINT_PREFIX}{endpoint}/{key}"] = float(value)
                out[f"{ENDPOINT_PREFIX}{endpoint}/n"] = len(rows)
    return out
