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
    "left_padding", "margin_array", "prompt_item", "score_source",
    "scoring_cost", "teacher_forced",
]

#: How many examples a validator scores per task by default. Generation is the
#: cost here — a caption at 128 new tokens is two orders of magnitude dearer than
#: a teacher-forced yes/no — and a cadence of every 500 steps means this runs
#: dozens of times in a run. The cap is a fixed seeded draw, not the first *n*,
#: so the curve within a run and the comparison across runs are both stable
#: (kgqa's ``eval_indices``). ``None`` scores the whole split, which is what the
#: end-of-run and milestone firings use.
DEFAULT_MAX_SAMPLES = None

#: Row cap for the teacher-forced paths. It is an upper bound, not the batch
#: size: `DEFAULT_BATCH_TOKENS` is what actually closes a batch.
DEFAULT_BATCH_SIZE = 8

#: Padded-token budget for one teacher-forced micro-batch, which is the quantity
#: that has to be bounded rather than the row count. A batch costs
#: ``rows x longest_row`` positions and the logits tensor is that times the
#: vocabulary, so a fixed row count prices a batch by its *mean* length while the
#: allocator is charged its *maximum*. The 2026-09-04 shakedown lost `in_mixture`
#: to exactly that: one long molecule pushed the block-aligned pad to 8192, and
#: `(8, 8192, 128256)` in fp32 is 32 GiB — the allocation in the OOM, to the byte.
#: 8192 positions is ~4 GB of fp32 logits, so a single row that long is still
#: affordable and eight ordinary ones still batch together. D4.4 makes this same
#: argument for the training batch; the eval path had not inherited it.
#:
#: The generation path groups against the same budget, where it is conservative
#: rather than binding: a prefill projects one position through the vocabulary
#: (`causal_lm.prepare_inputs_for_generation` asks for ``logits_to_keep=1``) and
#: the KV cache for eight rows of 8192 positions is a few hundred MB. Sharing the
#: budget is not about the memory there — it is what makes a generation batch
#: land on ``(B, L, N)`` shapes the scoring batches have already compiled.
DEFAULT_BATCH_TOKENS = 8192


def scoring_cost(spec, n: int) -> float:
    """Roughly what scoring ``n`` examples of ``spec`` costs, in forward passes.

    Used only to balance the ranks (`parallel.assign`), so it has one job: put
    the two orders of magnitude between a teacher-forced yes/no and a caption on
    the right side of each other. A teacher-forced row is one forward over its
    batch; a generated row is a prefill plus ``max_new_tokens`` sequential decode
    steps, and the decode steps are what a milestone spends its time on. Nothing
    this gets wrong can move a metric — a mis-costed target is scored on the same
    rows in the same batches, just possibly on a busier rank.
    """
    if getattr(spec, "answer_kind", None) in ("text", "smiles"):
        return float(n) * float(getattr(spec, "max_new_tokens", None) or 64)
    return float(n)


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


def row_length(item) -> int:
    """Padded positions one example occupies, before block alignment.

    ``input_ids`` is a list of per-node token lists on the graph arm and a single
    list on the flat arm; both are the same question — how many positions does
    this row put in the packed sequence.
    """
    ids = item.get("input_ids") if hasattr(item, "get") else None
    if not ids:
        return 0
    first = ids[0]
    if isinstance(first, (list, tuple)):
        return sum(len(node) for node in ids)
    return len(ids)


def token_batches(source, indices, max_rows: int = DEFAULT_BATCH_SIZE,
                  budget: int = DEFAULT_BATCH_TOKENS):
    """Group ``indices`` so that ``rows x longest_row`` stays under ``budget``.

    ``max_rows`` remains an upper bound, so a split of uniformly short rows
    batches exactly as it did before. A single row over budget is yielded alone
    rather than dropped — refusing to score the largest molecules would be a
    silent change to what the metric covers, which is worse than one expensive
    batch.

    Order is preserved. The scorers read each row under its own label mask and
    `_pad_stack` already handles blocks of different widths, so how rows are
    grouped cannot move a number.

    **Row counts come out on a power-of-two ladder**, so a greedy group of seven
    is emitted as ``4 + 2 + 1``. The batch dimension is a compile guard exactly
    like ``L`` and ``N``: `GraphCollatorV2` buckets those two precisely so the
    flex kernel sees few distinct shapes, and a budget that closes a batch
    wherever the tokens happen to run out would hand back the variety that
    bucketing was there to remove. Splitting *down* rather than padding *up* is
    what keeps the budget a budget — padding five rows to eight would put
    ``rows x longest`` back over the ceiling this function exists to hold.
    """
    def emit(rows):
        # Largest power of two first, so the common full batch stays one launch.
        start = 0
        while start < len(rows):
            take = 1 << (len(rows) - start).bit_length() - 1
            yield rows[start:start + take]
            start += take

    batch, longest = [], 0
    for index in indices:
        length = max(1, row_length(source[index]))
        candidate = max(longest, length)
        if batch and (len(batch) + 1 > max_rows
                      or (len(batch) + 1) * candidate > budget):
            yield from emit(batch)
            batch, longest = [index], length
            continue
        batch.append(index)
        longest = candidate
    if batch:
        yield from emit(batch)


def _to_device(batch, device):
    import torch

    if device is None:
        return batch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def teacher_forced(model, collator, source, indices, device=None,
                   batch_size: int = DEFAULT_BATCH_SIZE, preprocess=None,
                   batch_tokens: int = DEFAULT_BATCH_TOKENS):
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
        for chunk in token_batches(source, list(indices), batch_size, batch_tokens):
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
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 batch_tokens: int = DEFAULT_BATCH_TOKENS):
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
        preprocess=make_margin_preprocessor(yes_id, no_id),
        batch_tokens=batch_tokens)
    return preds, yes_id


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def prompt_item(item) -> dict:
    """One item with its prompt node truncated at the answer boundary.

    The generation input: everything the training item carries except the answer
    tokens and the columns that only supervision uses. Split out of the loop
    because the batching below has to measure these before it can group them —
    the row length that decides a batch is the *truncated* one.
    """
    from ..schema import SIDECAR_KEY

    start = answer_start(item)
    prompt_node = int(item["prompt_node"])
    gen = {k: v for k, v in item.items() if k not in (SIDECAR_KEY, "labels")}
    gen["input_ids"] = [list(x) for x in item["input_ids"]]
    gen["input_ids"][prompt_node] = gen["input_ids"][prompt_node][:start]
    return gen


class _Prepared:
    """A list of built items, addressed the way :func:`token_batches` addresses a source."""

    def __init__(self, items):
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def left_padding(collator):
    """``collator`` with the padding moved to the front, or ``None`` if it cannot be.

    A shallow copy rather than a second construction: the collator carries the
    tokenizer, the bucket ladders and the block size, and a generation batch that
    bucketed differently from a training batch would compile its own flex kernels
    for no reason. ``None`` for a collator with no such knob (the v0 ragged
    batcher), which is the signal to fall back to one row at a time — the layout
    is the correctness condition for batching, not a preference.
    """
    import copy

    if not hasattr(collator, "padding_side"):
        return None
    if getattr(collator, "padding_side") == "left":
        return collator
    out = copy.copy(collator)
    out.padding_side = "left"
    return out


def generate_predictions(model, tokenizer, collator, source, indices,
                         max_new_tokens: int = 64, device=None,
                         batch_size: int = DEFAULT_BATCH_SIZE,
                         batch_tokens: int = DEFAULT_BATCH_TOKENS) -> tuple:
    """Greedy continuations from the answer boundary. ``(predictions, targets)``.

    **Batched, left-padded, and on the same fast path as the rest of the run.**
    All three used to be otherwise, and the three are one change: generation ran
    a row at a time, on the dense eager kernel, from a right-padded batch. The
    docstring this replaces gave a reason for the first two that does not hold
    here — that a generation batch "cannot be bucketed (the prompt node must stay
    last in the packed sequence, so nothing may be padded past it)". The prompt
    node does have to stay last, and `GraphCollatorV2` under ``pad_to_block``
    pads past it *at any batch size, including one*: the packed length is rounded
    up to a 512 bucket, so a 200-token prompt is followed by 312 pads whether it
    is alone in the batch or not. Batch size was never what put padding after the
    prompt node; the padding side was. Hence `left_padding`, after which the last
    real token is at ``L-1`` for every row and a batch of eight is exactly eight
    copies of the case that already worked.

    Two things the same change fixes:

    * **RoPE on the generated tokens.** ``prepare_inputs_for_generation`` numbers
      a new token ``position_ids[:, -1] + 1``, continuing the prompt node's local
      counter. Under right padding ``position_ids[:, -1]`` is a pad, which is 0,
      so every continuation was numbered 1, 2, 3 … and RoPE placed it *before*
      the prompt it was answering. Left padding makes that read the prompt node's
      last position, which is what the expression was written for.
    * **The prefill kernel.** ``use_flex`` needs ``q_len == kv_len`` and a
      block-aligned length, and a bucketed prompt batch is both — so forcing
      ``graph_attn_impl`` to eager was giving up the fused kernel on the one pass
      in the evaluation that is quadratic in the sequence length. Decode steps
      still fall back to eager inside the model, where ``q_len == 1`` and flex
      buys nothing; that fallback is automatic and needs nothing here.

      **What flex buys on this path is the peak, not the clock.** Batched-eager
      and batched-flex come out level on time — generation is decode-dominated
      and decode is eager either way — but the eval's peak allocation is 23.8 GB
      against 10.5 GB, with the one-row-at-a-time path it replaces at 13.1 GB. So
      batching on the eager prefill would have pushed the evaluation *above* the
      path it replaces, beside a training step that already peaks at 100.6 GB on
      a 178 GB card. That is the shape of the 2026-09-04 OOM, and flex is what
      keeps this batching safe to fire mid-training.

    Row counts come off `token_batches`' power-of-two ladder against the same
    token budget the teacher-forced paths use, so the ``(B, L, N)`` shapes a
    generation batch compiles are shapes the scoring batches have already
    compiled. That is what keeps batched generation from filling dynamo's cache
    with a second family of kernels — measured, the whole evaluation touches 25
    distinct triples on the graph arm and 1 on the flat arm against a cap of 128,
    which is *fewer* than the path this replaces (27 and 2), because a batch of
    one is no longer a shape of its own.
    """
    import torch

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    prompts = _Prepared([prompt_item(source[i]) for i in indices])
    targets = _answers(source, indices)
    gen_collator = left_padding(collator)
    if gen_collator is None:
        gen_collator, batch_size = collator, 1

    predictions = [""] * len(prompts)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        for chunk in token_batches(prompts, list(range(len(prompts))), batch_size,
                                   batch_tokens):
            batch = _to_device(gen_collator([prompts[j] for j in chunk]), device)
            out = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=pad_id)
            new_tokens = out[:, batch["input_ids"].shape[1]:]
            for row, j in enumerate(chunk):
                predictions[j] = tokenizer.decode(
                    new_tokens[row], skip_special_tokens=True).strip()

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
                 per_endpoint: bool = True,
                 batch_tokens: int = DEFAULT_BATCH_TOKENS) -> dict:
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
        return _score_token(model, collator, source, indices, device, batch_size,
                            batch_tokens)
    if kind == "yesno":
        return _score_yesno(model, tokenizer, collator, source, indices, device,
                            batch_size, per_endpoint, batch_tokens)
    predictions, targets = generate_predictions(
        model, tokenizer, collator, source, indices,
        max_new_tokens=spec.max_new_tokens or 64, device=device,
        batch_size=batch_size, batch_tokens=batch_tokens)
    if kind == "smiles":
        from ..adapters.molecules import smiles_scores

        return dict(smiles_scores(predictions, targets))
    from .captions import caption_metrics

    return dict(caption_metrics(predictions, targets))


def _score_token(model, collator, source, indices, device, batch_size,
                 batch_tokens=DEFAULT_BATCH_TOKENS) -> dict:
    from ...utils import make_compute_metrics, shift_logits_for_metrics

    if not len(indices):
        return {"em_accuracy": 0.0, "n": 0}
    preds, labels = teacher_forced(
        model, collator, source, indices, device=device, batch_size=batch_size,
        preprocess=shift_logits_for_metrics, batch_tokens=batch_tokens)
    out = make_compute_metrics()((preds.astype("int64"), labels.astype("int64")))
    return {"em_accuracy": float(out["em_accuracy"]), "n": len(indices)}


def _score_yesno(model, tokenizer, collator, source, indices, device, batch_size,
                 per_endpoint, batch_tokens=DEFAULT_BATCH_TOKENS) -> dict:
    from ...experiments.molecules.evaluate import make_margin_metrics

    if not len(indices):
        return {k: (0 if k == "n" else float("nan")) for k in METRIC_KEYS["yesno"]}

    preds, yes_id = margin_array(model, tokenizer, collator, source, indices,
                                 device=device, batch_size=batch_size,
                                 batch_tokens=batch_tokens)
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
