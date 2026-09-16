"""Has the adapter cost the backbone its assistant behaviour?

"A general-text held-out loss, adapter-on against adapter-off" was owed from arm 2
until 2026-09-12, and nothing measured it. This is that, plus the three things a
loss would not catch; the read-out is `MOLECULE_GENERALIST.md` §7.6.

**Why the control is exact here, and free.** Every campaign so far is D4 arm A:
the backbone is frozen and `base_exact` reports ``max_abs_diff`` of exactly 0.0,
so everything the project adds lives in the LoRA adapter and the bias channel.
On a **single-node** graph every structural bias is identically zero — `SPDBias`
multiplies by ``(spd > 0)`` and a lone node's only distance is 0; `MagneticBias`
finalizes through ``finalize_node_bias(..., bias_self_node=False)``, which masks
``i == j`` — so a plain text prompt, which is exactly `dataset.build_flat_example`'s
shape, reaches the model as base weights plus the adapter and nothing else.
Switching the adapter off therefore *is* the base model, with no second
checkpoint resident and no corpus to choose. That is the whole reason this
validator can be paired rather than absolute.

It also means this is the logit comparison `base_exact` says it cannot make
("on this architecture there is no such batch"). There is one: a single-node
graph. `base_exact` remains the right check for what it measures — the weights,
exactly, rather than within a bf16 tolerance — and this one measures what the
adapter *does*, which is the half nothing covered.

**Four things, and only the first is a loss.**

* ``base_continuation_nll`` — the adapter-off greedy continuation, scored under
  the adapter. "How surprised is the trained model by what the backbone would
  have said." No reference corpus is needed, which is the point: any text set we
  picked would be a guess, and the backbone's own output is the distribution we
  are trying not to move.
* ``kl_mean`` — mean per-token ``KL(off || on)`` over those same positions. This
  is precisely the quantity `PLAN.md` §6's KL-to-base self-distillation would
  minimise, so it says how much work that mechanism would have to do before we
  commit to it.
* **Register**: ``new_tokens_mean``, ``chars_mean``, ``single_token_rate``,
  ``empty_rate``. Two thirds of this mixture is single-token answers, and a model
  that has learned to reply ` Yes` to everything has a perfectly ordinary
  perplexity. Paired against adapter-off on the identical prompts.
* ``caption_rate`` — answers that are a **ChEBI-20 molecule caption** instead of
  an answer to the question. This is the one that found something, and the only
  one that could have: asked what the greenhouse effect is, a graph cell replies
  "The molecule that is the simplest member of the class of benzenes … It has a
  role as a non-polar solvent", which is fluent, correctly terminated, of
  ordinary length, and has nothing to do with the question. Every other number
  here scores it as healthy.
* ``stop_rate`` — did it terminate inside the budget, which is
  :data:`MAX_NEW_TOKENS` and is sized so that the *backbone* mostly clears it. We
  deliberately taught a stop token on ~45-character answers (§8); the failure
  mode this watches for is that lesson generalising to prose.

**Two conditions, because one of them is a departure we chose.** `plain` is the
prompt as the run's own format writes it. `system` prepends a system turn — which
the chat format supports and **no training example has ever contained**, since
the stock template injects today's date and would make a build's bytes depend on
the day it ran (`MOLECULE_GENERALIST.md` §6). The trunk will be served with system prompts on
every request, so the cost of that choice is worth a number rather than an
argument. On a non-chat format there is no system turn to write and the condition
is skipped.

The prompt set is `text_probes.json`, fixed and versioned beside this file: the
value of these numbers is entirely in one checkpoint's being comparable to
another's months later, so the set is never regenerated in place.
"""

from __future__ import annotations

import json
import os

from . import BaseValidator, register

#: Beside this module, so the set travels with the code that reads it.
PROBES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "text_probes.json")

#: Conditions, in report order. ``system`` is chat-only — see the module docstring.
CONDITIONS = ("plain", "system")

#: The adapter states compared. ``off`` is the base model exactly (module docstring).
ADAPTER_STATES = ("on", "off")

#: Rows per forward pass on the divergence pass. Deliberately small and not taken
#: from `scorers.DEFAULT_BATCH_SIZE`: that pass holds **two** full ``(B, L, V)``
#: logit tensors at once — the adapter's and the backbone's — to take a KL
#: between them, and V is 128k here. The generation pass uses the normal ladder.
DIVERGENCE_BATCH_SIZE = 4

#: The generation cap, and it has to clear a *natural* answer or ``stop_rate``
#: stops being about stopping.
#:
#: Measured at 128 on the first firing: the backbone itself terminated on only
#: 14.6 % of these prompts, so 85 % of the control column was the cap rather than
#: the model, and "stopped" was reading as a synonym for "wrote a short answer" —
#: which ``chars_mean`` already reports and reports better. 256 puts the cap
#: clear of an unhurried assistant answer, which is what makes a low
#: ``stop_rate`` attributable to the model. It doubles the generation half of the
#: firing; the firing is minutes.
MAX_NEW_TOKENS = 256


def load_probes(path: str = PROBES_PATH) -> dict:
    """The probe set as a dict, with the prompts checked for the fields read here."""
    with open(path) as fh:
        probes = json.load(fh)
    prompts = probes.get("prompts") or []
    if not prompts:
        raise ValueError(f"{path}: carries no prompts")
    for i, prompt in enumerate(prompts):
        for field in ("id", "category", "text"):
            if not prompt.get(field):
                raise ValueError(f"{path}: prompt {i} has no {field!r}")
    return probes


def peft_model(model):
    """The object whose ``disable_adapter()`` turns the LoRA off, or ``None``.

    The model reaches a validator through the trainer and may be wrapped — DDP
    puts it behind ``.module``, and `select_active_params` returns the PEFT model
    itself. Walking rather than asserting a path keeps this working when the
    wrapping changes, and ``None`` is a reportable answer (``applicable`` 0)
    rather than a crash: a run without LoRA has no adapter to switch off and this
    whole comparison is undefined there, not broken.
    """
    seen = 0
    while model is not None and seen < 4:
        if hasattr(model, "disable_adapter"):
            return model
        model = getattr(model, "module", None)
        seen += 1
    return None


def probe_texts(probes: dict, fmt, condition: str) -> list:
    """``[(id, category, node_text), ...]`` — one single-node graph's text each.

    Mirrors `dataset.build_flat_example`: one node carrying the question turn and
    the open assistant turn, which is the same shape the flat arm trains on minus
    a molecule. The answer is empty because generation starts exactly here.
    """
    out = []
    system = (probes.get("system") or "").strip()
    for prompt in probes["prompts"]:
        body = prompt["text"]
        text = f"{fmt.question(body)}{fmt.answer_prefix}"
        if condition == "system":
            if not fmt.question_suffix or not system:
                continue
            # The system turn as the stock template writes it, minus the
            # "Cutting Knowledge Date" block — the same omission the build makes,
            # for the same reason (`MOLECULE_GENERALIST.md` §6).
            head = ("<|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system}{fmt.question_suffix}")
            text = f"{head}{text}"
        out.append((prompt["id"], prompt["category"], text))
    return out


def build_items(texts, tokenizer, config, answers=None):
    """Single-node graph items, featurized exactly as `_materialise` does.

    Going through `TextGraphDataset` rather than hand-writing the columns is the
    point: the SPD and magnetic columns a graph-arm collator reads have shapes
    and names this module should not be a second opinion about. On one node they
    are 1x1 and cost nothing.

    ``answers`` (optional) appends a continuation to each node and supervises it,
    which is what the divergence pass scores; without it every label is -100 and
    the item is a generation prompt.
    """
    import networkx as nx

    from ...utils import TextGraphDataset

    graphs = []
    for i, (_id, _category, text) in enumerate(texts):
        graph = nx.DiGraph()
        body = text if answers is None else f"{text}{answers[i]}"
        graph.add_node(0, text=body, kind="prompt")
        graph.graph["prompt_node"] = 0
        graphs.append(graph)

    # Keyed by the node's full text rather than by position: `compute_labels`
    # hands the callable one example at a time and no index, so anything that
    # depended on call order would be relying on `datasets.map` not batching.
    prefix_lengths = None
    if answers is not None:
        prefix_lengths = {
            f"{text}{answers[i]}":
                len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for i, (_id, _category, text) in enumerate(texts)}

    def labels_for(example):
        node = example["prompt_node"]
        ids = example["input_ids"][node]
        labels = [-100] * len(ids)
        if prefix_lengths is not None:
            start = min(prefix_lengths[example["text"][node]], len(ids) - 1)
            labels[start:] = ids[start:]
        return labels

    ds = TextGraphDataset(graphs)
    ds.tokenize(tokenizer, max_length=int(config.get("max_length", 512)))
    ds.compute_labels(labels_for, num_proc=1)
    ds.compute_shortest_path_distances()
    ds.compute_magnetic_lap(q=float(config.get("magnetic_q", 0.25)),
                            m=int(config.get("magnetic_m", 0)))
    ds.cast_float_features_to_fp32()
    return [ds[i] for i in range(len(ds))]


class _Rows:
    """A list of items addressed the way `scorers.token_batches` addresses a source."""

    def __init__(self, items):
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def generate(model, tokenizer, collator, items, device, max_new_tokens: int,
             batch_tokens: int) -> list:
    """``[(token ids, stopped), ...]`` — greedy continuations, stop token kept.

    `scorers.generate_predictions` decodes with ``skip_special_tokens=True``,
    which is right for scoring an answer and wrong here: whether the model
    emitted the stop token at all is one of the four things being measured, and
    that flag is exactly what stripping removes.
    """
    import torch

    from .scorers import DEFAULT_BATCH_SIZE, _to_device, left_padding, token_batches

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    rows = _Rows(items)
    gen_collator = left_padding(collator)
    batch_size = DEFAULT_BATCH_SIZE
    if gen_collator is None:
        gen_collator, batch_size = collator, 1

    out = [None] * len(items)
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for chunk in token_batches(rows, list(range(len(items))), batch_size,
                                   batch_tokens):
            batch = _to_device(gen_collator([items[j] for j in chunk]), device)
            batch.pop("labels", None)
            generated = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=pad_id)
            fresh = generated[:, batch["input_ids"].shape[1]:]
            for row, j in enumerate(chunk):
                ids = [int(t) for t in fresh[row].tolist()]
                stopped = eos_id in ids
                if stopped:
                    ids = ids[:ids.index(eos_id)]
                out[j] = (ids, stopped)
    if was_training and hasattr(model, "train"):
        model.train()
    return out


def write_predictions(path, texts, rows, tokenizer) -> None:
    """One JSON line per prompt: what the model actually wrote.

    Every other generative measurement in this repo keeps its predictions —
    `tools/g2s_report.py` is the reason the stop-token defect was diagnosable at
    all rather than merely visible. The same applies here and more sharply,
    because these metrics are *means over 48 prompts*: without the rows there is
    no way to tell a small gap that is every prompt shifting from one that is a
    single prompt diverging, and those two call for opposite responses.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for (probe_id, category, prompt), (ids, stopped) in zip(texts, rows):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            fh.write(json.dumps({
                "id": probe_id, "category": category, "prompt": prompt,
                "text": text, "new_tokens": len(ids), "stopped": bool(stopped),
            }, sort_keys=True) + "\n")


#: Phrasings that belong to a ChEBI-20 caption and to nothing else in English.
#: ChEBI descriptions are written to a house style — "The molecule is a …", "It
#: has a role as …", "is functionally related to …" — and ChEBI-20 is 19.3 % of
#: the mixture, so this is the training text most likely to be reached for when a
#: prompt is out of distribution.
CAPTION_MARKERS = (
    r"^\s*The molecule\b",
    r"It has a role as",
    r"is functionally related to",
    r"It is a conjugate (acid|base) of",
    r"member of the class of",
)


def caption_shaped(text: str) -> bool:
    """Is this answer a molecule caption rather than an answer to the question?

    **A lower bound, deliberately.** It fires on a caption's house phrasing, so it
    counts the unambiguous cases and misses an answer that drifts into chemistry
    without reaching for the template. A rate from this is "at least this often",
    which is the useful direction: the number exists to show that something is
    wrong, and understating it cannot manufacture the finding.

    It is checked against adapter-off on the same prompts, where it reads 0 on
    every cell — the backbone never writes like this — so a nonzero rate is the
    adapter's doing and not a quirk of the probe set.
    """
    import re

    return any(re.search(marker, text, re.I) for marker in CAPTION_MARKERS)


def register_metrics(rows, tokenizer) -> dict:
    """The register numbers over ``[(ids, stopped), ...]``.

    ``caption_rate`` is the one that matters and the one nothing else here would
    have caught: a caption-shaped answer to "What is the greenhouse effect?" is
    fluent, correctly terminated, of ordinary length and completely wrong, so it
    is invisible to every other number in this dict.
    """
    if not rows:
        return {}
    texts = [tokenizer.decode(ids, skip_special_tokens=True).strip()
             for ids, _stopped in rows]
    n = len(rows)
    return {
        "n": float(n),
        "new_tokens_mean": sum(len(ids) for ids, _s in rows) / n,
        "chars_mean": sum(len(t) for t in texts) / n,
        "stop_rate": sum(1.0 for _ids, stopped in rows if stopped) / n,
        "single_token_rate": sum(1.0 for ids, _s in rows if len(ids) <= 1) / n,
        "empty_rate": sum(1.0 for t in texts if not t) / n,
        "caption_rate": sum(1.0 for t in texts if caption_shaped(t)) / n,
    }


def divergence(model, adapter, collator, items, device, batch_tokens) -> dict:
    """``base_continuation_nll`` and ``kl_mean`` over the supervised span.

    One pass with the adapter live and one with it disabled, on the same batch,
    so the two logit tensors are comparable position by position. Labels arrive
    from the collator already expanded to the packed sequence, in HF's
    convention — ``labels[:, t]`` is the token *at* position ``t`` — so the
    prediction of it is ``logits[:, t - 1]``, which is the shift below.
    """
    import torch
    import torch.nn.functional as F

    from .scorers import _to_device, token_batches

    rows = _Rows(items)
    nll_sum, kl_sum, count = 0.0, 0.0, 0
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for chunk in token_batches(rows, list(range(len(items))),
                                   DIVERGENCE_BATCH_SIZE, batch_tokens):
            batch = _to_device(collator([items[j] for j in chunk]), device)
            labels = batch.pop("labels")
            on = model(**batch)
            on = (on.logits if hasattr(on, "logits") else on[0]).float()
            with adapter.disable_adapter():
                off = model(**batch)
                off = (off.logits if hasattr(off, "logits") else off[0]).float()

            target = labels[:, 1:]
            mask = target != -100
            if not bool(mask.any()):
                continue
            logp_on = F.log_softmax(on[:, :-1], dim=-1)
            logp_off = F.log_softmax(off[:, :-1], dim=-1)

            picked = logp_on.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            nll_sum += float((-picked * mask).sum())

            kl = (logp_off.exp() * (logp_off - logp_on)).sum(-1)
            kl_sum += float((kl * mask).sum())
            count += int(mask.sum())
    if was_training and hasattr(model, "train"):
        model.train()
    # Always all three keys, even with nothing to score: the runner drops a
    # validator whose returned leaves do not match what it declared, so an empty
    # dict here would throw away the register metrics beside it. ``n`` is the
    # honesty channel — at 0 the two means are 0.0 because nothing was measured,
    # not because the model matched the backbone.
    if not count:
        return {"n": 0.0, "base_continuation_nll": 0.0, "kl_mean": 0.0}
    return {"n": float(count),
            "base_continuation_nll": nll_sum / count,
            "kl_mean": kl_sum / count}


@register
class TextBehaviour(BaseValidator):
    """Assistant behaviour on text-only prompts, adapter-on against adapter-off.

    See the module docstring for what each number is and why the adapter-off leg
    is an exact control rather than an approximation. ``applicable`` is 0 with a
    ``reason`` when there is no adapter to switch off, which is a run this
    comparison is undefined for rather than one it failed on.

    Cadence is ``milestone``: it generates, so it belongs with the expensive half
    of the suite (`config.DEFAULT_VALIDATORS` puts `in_mixture`, `held_out`,
    `base_exact` and `leakage` on the same two firings) rather than on a step
    multiple of its own.
    """

    name = "text_behaviour"
    cadence = "milestone"
    needs = frozenset({"model", "tokenizer", "collator", "scratch_dir"})
    protocol_version = "1"

    #: Every leaf a firing with an adapter produces. `run` emits all of them for
    #: every condition it reaches, including the divergence pass when there was
    #: nothing to score, because the runner drops a validator whose leaves do not
    #: match its declaration.
    MEASURED_KEYS = frozenset({
        "n", "new_tokens_mean", "chars_mean", "stop_rate", "single_token_rate",
        "empty_rate", "caption_rate", "base_continuation_nll", "kl_mean",
        "predictions_path"})

    #: What a firing reports when there is nothing to measure against.
    STATUS_KEYS = frozenset({"applicable", "reason"})

    def keys(self, ctx=None) -> set:
        """Context-aware, because a run without LoRA produces only the status.

        Declaring the measured keys there would promise numbers this validator
        will not return, and the runner checks that promise exactly.
        """
        if ctx is not None and peft_model(ctx.model) is None:
            return set(self.STATUS_KEYS)
        return set(self.MEASURED_KEYS | self.STATUS_KEYS)

    def run(self, ctx) -> dict:
        from ...experiments.molecules.data import prompt_format
        from .scorers import DEFAULT_BATCH_TOKENS

        adapter = peft_model(ctx.model)
        if adapter is None:
            return {"applicable": 0.0,
                    "reason": "no LoRA adapter on this model, so there is no "
                              "base model to compare against; the whole "
                              "comparison is undefined rather than failed"}

        run_config = dict((ctx.config or {}).get("run_config") or {})
        max_new_tokens = int(self.option("max_new_tokens", MAX_NEW_TOKENS))
        batch_tokens = int(self.option("batch_tokens", DEFAULT_BATCH_TOKENS))
        probes = load_probes(self.option("prompts", PROBES_PATH))
        fmt = prompt_format(run_config.get("prompt_style"),
                            run_config.get("model_name") or ctx.base_model_name)

        out = {"applicable": 1.0, "reason": ""}
        for condition in CONDITIONS:
            texts = probe_texts(probes, fmt, condition)
            if not texts:
                continue
            prompts = build_items(texts, ctx.tokenizer, run_config)

            rows = {}
            for state in ADAPTER_STATES:
                if state == "off":
                    with adapter.disable_adapter():
                        rows[state] = generate(
                            ctx.model, ctx.tokenizer, ctx.collator, prompts,
                            ctx.device, max_new_tokens, batch_tokens)
                else:
                    rows[state] = generate(
                        ctx.model, ctx.tokenizer, ctx.collator, prompts,
                        ctx.device, max_new_tokens, batch_tokens)
                for key, value in register_metrics(rows[state], ctx.tokenizer).items():
                    out[f"{condition}/{state}/{key}"] = value
                path = os.path.join(ctx.scratch_dir, self.name,
                                    f"{condition}-{state}.jsonl")
                write_predictions(path, texts, rows[state], ctx.tokenizer)
                out[f"{condition}/{state}/predictions_path"] = path

            # The backbone's own continuation is the reference the adapter is
            # scored against — see the module docstring for why no corpus is
            # picked here.
            answers = [ctx.tokenizer.decode(ids, skip_special_tokens=True)
                       for ids, _stopped in rows["off"]]
            scored = [i for i, answer in enumerate(answers) if answer.strip()]
            if not scored:
                continue
            supervised = build_items(
                [texts[i] for i in scored], ctx.tokenizer, run_config,
                answers=[answers[i] for i in scored])
            for key, value in divergence(ctx.model, adapter, ctx.collator,
                                         supervised, ctx.device,
                                         batch_tokens).items():
                out[f"{condition}/divergence/{key}"] = value
        return out
