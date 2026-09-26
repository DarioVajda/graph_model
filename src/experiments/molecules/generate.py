"""
Greedy caption generation for Tier C, from a trained checkpoint.

Tiers A and B are read teacher-forced — one token's logit margin, or one token's
exact match — so this package never needed to generate. A caption does: BLEU is
computed against what the model actually writes, not against what it would score
for the reference.

**Left padding, and it is the whole reason this is not three lines.** The prompt
node must be the last real token of the packed sequence, because that is the
position generation continues from. Right padding puts pads after it, which makes
``position_ids[:, -1]`` a pad — so every continuation is numbered 1, 2, 3 … and
RoPE places the generated tokens *before* the prompt they answer. Left padding
puts the last real token at ``L-1`` for every row, which is the case that already
worked at batch size one. `src/generalist/evaluate/scorers.py` records the same
finding; this is the molecules-package form of it.
"""

from __future__ import annotations

import torch

from .chebi import CHEBI_QUESTION  # noqa: F401  (imported for callers' convenience)

#: Generation cap. ChEBI captions run to 86 words on the cap-128 build, and a
#: word is more than one token, so 256 leaves room without inviting a runaway.
DEFAULT_MAX_NEW_TOKENS = 256

#: Steps during which the stop token is forbidden.
#:
#: **An empty caption is never a valid answer here**, and without this the model
#: is allowed to give one: an EOS chosen at the first generated position decodes,
#: under ``skip_special_tokens=True``, to the empty string. It is scored as a
#: total miss and it is nearly invisible upstream — one position out of ~50, so
#: even a 24 % probability there moves ``eval_loss`` by about 0.005.
#:
#: Measured on `043` (instruct, WSD), 2026-09-18: 189 / 375 / 788 empty captions
#: of 3,261 across three seeds whose eval_loss curves agreed to three decimals
#: (0.5962 / 0.5938 / 0.5896), against **zero** on every base-backbone cell. On a
#: 200-row sample of the worst seed, lifting the constraint's absence took empties
#: from 46 to 0 and BLEU-2 from 0.3003 to 0.3914, and the recovered rows read as
#: ordinary captions. The pathology is specific to the chat path, where
#: ``<|start_header_id|>assistant<|end_header_id|>\n\n`` followed straight by
#: ``<|eot_id|>`` is the instruct-tuned spelling of an empty turn — a prior the
#: base backbone simply does not carry.
#:
#: It is applied to EVERY arm, not to the ones that need it. A constraint switched
#: on for the arm it rescues is not a measurement.
DEFAULT_MIN_NEW_TOKENS = 5

#: Token budget per batch. Rows per batch fall out of it, so a split of long
#: captions makes smaller batches rather than a larger peak.
#:
#: **2048, not 8192, and the reason is the cap-128 build.** The structural bias is
#: N-by-N in *nodes*, and a 128-heavy-atom molecule is ~270 Levi nodes against the
#: ~130 a cap-64 build tops out at — four times the bias tensor per graph. At 8192
#: this OOMs on a 40 GB card trying to allocate 12.4 GiB in one index op. The
#: token budget bounds sequence length, which is not the quantity that grew.
DEFAULT_BATCH_TOKENS = 2048


def build_model_for_eval(cfg, checkpoint):
    """``(model, tokenizer, collator, device)`` for a trained Tier-C checkpoint.

    The same construction `train.py` uses, so the model being scored is the model
    that was trained — adapters loaded onto the same backbone with the same bias
    config, rather than a re-derived approximation of it.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    from ..expressiveness.training.dispatch import build_collator, build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    # Generation continues from the last real token, so the batch pads on the
    # LEFT. See the module docstring for what right padding does to RoPE here.
    tokenizer.padding_side = "left"

    model, _ = build_model(cfg.impl, cfg.model_name, cfg.model_bias_config(),
                           cfg.k_hop, cfg.k_hop_directed, device,
                           cfg.flex_compile_mode)
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    _load_bias_weights(model, checkpoint,
                       required=(cfg.bias or "none").strip() != "none")
    model.eval()

    collator = build_collator(
        cfg.impl, tokenizer, pad_token_id, cfg.k_hop, cfg.k_hop_directed,
        magnetic_m=cfg.magnetic_m,
        len_buckets=cfg.len_buckets, node_buckets=cfg.node_buckets)
    # The left padding has to be set HERE, on the collator. `GraphCollatorV2`
    # packs the batch itself and reads only its own ``padding_side``; the
    # tokenizer's, set above, never reaches it. Until 2026-09-24 it defaulted to
    # right, so every caption — a batch of one included, since `pad_to_block`
    # rounds L up to a bucket — was continued from a pad at position 0. Found
    # on the 8B cells (055), where ~99 % of captions opened on a stray token
    # ("definite:", "Question 2") while the teacher-forced argmax at the last
    # real prompt token was " The" on every row checked; the 1B tolerated the
    # same misplacement on ~87 % of rows, so every ChEBI score before that date
    # carries it.
    if not hasattr(collator, "padding_side"):
        raise SystemExit(f"{type(collator).__name__} has no padding_side; "
                         "generation needs left padding.")
    collator.padding_side = "left"
    return model, tokenizer, collator, device


def _load_bias_weights(model, checkpoint, required):
    """Restore the structural-bias tensors, which are not LoRA and not in the adapter.

    `ACTIVE_PARAMS` trains ``graph_bias`` alongside the LoRA weights and
    `GraphTrainerV2` saves it beside the adapter as ``bias_parameters.pt``; PEFT's
    own checkpoint carries only the LoRA weights. Loading the adapter and stopping
    there would score a model whose bias tables are back at initialisation — a
    silent ablation of the exact channel the experiment is about, which would show
    up as a worse number and not as an error.

    ``required`` is true on a bias-carrying arm, and then an absent file is fatal
    rather than a warning: `models.io.load_bias_parameters` returning ``None`` is
    indistinguishable from a successful no-op unless someone checks.
    """
    from ...models.io import load_bias_parameters

    result = load_bias_parameters(model, checkpoint)
    if result is None:
        if required:
            raise SystemExit(
                f"{checkpoint} has no bias_parameters.pt, but this run's arm "
                "carries a structural bias. Scoring it would report a number for "
                "a model whose bias tables sit at initialisation.")
        print(f"[generate] no bias parameters in {checkpoint} (correct for "
              "bias 'none')")
        return None
    print(f"[generate] restored bias parameters from {checkpoint}")
    return result


def _answer_text(dataset, index, tokenizer):
    """The reference caption for one row, read off its supervised label span."""
    row = dataset[index]
    labels = row["labels"]
    ids = [t for t in labels if t != -100]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _prompt_only(row, tokenizer):
    """The row with its answer tokens removed, so generation has something to do."""
    import copy

    out = copy.deepcopy(row)
    node = out["prompt_node"]
    labels = out["labels"]
    # The masked prefix IS the prompt: `make_caption_labels` masks exactly the
    # answer-prefix tokens and supervises everything after them, so the count of
    # -100s is where the caption starts.
    keep = sum(1 for t in labels if t == -100)
    out["input_ids"][node] = out["input_ids"][node][:keep]
    # `labels` has to be truncated with it. The collator asserts the two agree on
    # the prompt node's length, and that assert is the only reason this was a
    # clear error rather than a batch whose generated tokens were silently
    # misaligned with their positions.
    out["labels"] = labels[:keep]
    if "attention_mask" in out and isinstance(out["attention_mask"], list):
        try:
            out["attention_mask"][node] = out["attention_mask"][node][:keep]
        except (TypeError, IndexError):
            pass
    return out


def generate_captions(model, tokenizer, collator, dataset, device=None,
                      max_samples=None, batch_tokens=DEFAULT_BATCH_TOKENS,
                      max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                      min_new_tokens=DEFAULT_MIN_NEW_TOKENS):
    """``(predictions, targets, meta)`` over ``dataset``.

    ``max_samples`` takes a deterministic prefix rather than a random sample: the
    splits here are the benchmark's own and are not ordered by anything the model
    sees, and a fixed prefix makes two checkpoints comparable without carrying a
    seed around. ``None`` scores the whole split.
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = list(range(n))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    predictions, targets, meta = [], [], []
    batch, batch_tokens_used = [], 0
    for i in indices:
        row = dataset[i]
        length = sum(len(x) for x in row["input_ids"])
        if batch and batch_tokens_used + length > batch_tokens:
            _run_batch(model, tokenizer, collator, batch, device, pad_id,
                       max_new_tokens, predictions, min_new_tokens)
            batch, batch_tokens_used = [], 0
        batch.append(_prompt_only(row, tokenizer))
        batch_tokens_used += length
        targets.append(_answer_text(dataset, i, tokenizer))
        meta.append({"index": i})
    if batch:
        _run_batch(model, tokenizer, collator, batch, device, pad_id,
                   max_new_tokens, predictions, min_new_tokens)
    return predictions, targets, meta


def _run_batch(model, tokenizer, collator, rows, device, pad_id, max_new_tokens,
               out, min_new_tokens=DEFAULT_MIN_NEW_TOKENS):
    packed = collator(rows)
    packed = {k: (v.to(device) if hasattr(v, "to") else v)
              for k, v in packed.items()}
    packed.pop("labels", None)
    prompt_len = packed["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(**packed, max_new_tokens=max_new_tokens,
                                   min_new_tokens=min_new_tokens,
                                   do_sample=False, num_beams=1,
                                   pad_token_id=pad_id)
    for row in range(generated.shape[0]):
        text = tokenizer.decode(generated[row, prompt_len:],
                                skip_special_tokens=True)
        out.append(text.strip())
