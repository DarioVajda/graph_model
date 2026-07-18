"""
D2.2 — flat (text-serialization) training path: plain HF causal LM + LoRA.

Same base LLM, same capped SR subgraph — but flattened to triple text in the
prompt: no graph collator, no attention biases, no k-hop gate. This is the
control that isolates whether the graph attention biases do anything.

Everything that can be shared with the graph arm IS shared:
  * scoring — ``evaluate``'s GNN-RAG-verbatim functions on the same gold lists;
  * label masking — ``AnswerLabelMasker`` on the same "Answer:" anchor;
  * LoRA — ``select_active_params`` with the same ``cfg.lora_config()``;
  * schedule — cosine_with_min_lr, 10% warmup, wd 0.1, same batch/accum/epochs,
    full-dev generative checkpoint selection on eval_f1 at the same cadence;
  * decoding — greedy, natural EOS stop, same ``gen_max_new_tokens``.

Differences (inherent to the arm): sdpa attention, seq-len ``cfg.seq_len``
(4096 covers 99.6%+ of questions; the ~10 outliers drop trailing triple lines),
dynamic right-padding.

Routed from __main__ via ``--mode flat_train``.
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from ...train import select_active_params, print_trainable_parameters, get_device
from ...utils import set_wandb_project
from ._io import append_jsonl
from .evaluate import (
    PerDatasetEvalMixin, _find_prefix_len, eval_f1, eval_hit, eval_hit1,
    eval_indices, normalize, parse_answer_list, _strict_set_f1)
from .flat_data import flat_data_config_key
from .process_dataset import OUTPUT_ROOT
from .train import EXPERIMENT_NAME, _DEFAULT_RUNS_JSONL, _PRINT_KEYS, result_block


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def compose_text(row):
    """`{question}\\n{h | r | t per line}\\nAnswer: {answers}` (dfv3 newline
    targets; empty target = prompt ends exactly at 'Answer:', labels EOS only)."""
    body = f"{row['question']}\n" + "\n".join(row["lines"]) + "\nAnswer:"
    return body + (f" {row['target']}" if row["target"] else "")


def tokenize_row(row, tokenizer, question_end, seq_len):
    """Tokenize one row; outliers drop trailing triple lines until they fit."""
    lines = row["lines"]
    while True:
        ids = tokenizer(compose_text({**row, "lines": lines}),
                        add_special_tokens=False)["input_ids"]
        ids = ids + [tokenizer.eos_token_id]
        if len(ids) <= seq_len or not lines:
            break
        lines = lines[:-8] if len(lines) > 8 else []
    cut = _find_prefix_len(ids, question_end)
    if cut is None:
        raise ValueError(f"'Answer:' anchor lost while truncating: {row['question']!r}")
    labels = [-100] * cut + ids[cut:]
    return {"input_ids": ids, "labels": labels, "prefix_len": cut,
            "gold_answers": row["gold_answers"]}


class FlatVersionedDataset(Dataset):
    """Mirrors TextGraphDataset's augmentation contract: rows are stored
    [q1_v1..q1_vk, q2_v1..]; len = #questions; __getitem__ samples a version."""

    def __init__(self, rows, versions=1):
        if len(rows) % versions:
            raise ValueError(f"{len(rows)} rows not divisible by versions={versions}")
        self.rows, self.versions = rows, versions

    def __len__(self):
        return len(self.rows) // self.versions

    def __getitem__(self, idx):
        version = random.randint(0, self.versions - 1) if self.versions > 1 else 0
        r = self.rows[idx * self.versions + version]
        return {"input_ids": r["input_ids"], "labels": r["labels"]}


class PadCollator:
    """Right-pad input_ids (with EOS) + labels (with -100), build attention_mask."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        n = max(len(it["input_ids"]) for it in items)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for it in items:
            ids, labels = it["input_ids"], it["labels"]
            pad = n - len(ids)
            batch["input_ids"].append(ids + [self.pad_id] * pad)
            batch["labels"].append(labels + [-100] * pad)
            batch["attention_mask"].append([1] * len(ids) + [0] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# Generative eval (same protocol + scoring as evaluate.generative_eval)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def flat_generative_eval(model, rows, tokenizer, max_new_tokens=128, device=None,
                         max_samples=None, prefix="eval", answer_sep="\n",
                         include_strict=False):
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device

    hits1, f1s, hitstar = [], [], []
    s_hits1, s_f1s, s_hitstar = [], [], []
    for i in eval_indices(len(rows), max_samples):
        row = rows[i]
        gold = [a for a in row["gold_answers"] if a]
        if not gold:
            continue
        ids = torch.tensor([row["input_ids"][: row["prefix_len"]]], device=device)
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = parse_answer_list(text, sep=answer_sep)

        if pred:
            hits1.append(float(eval_hit1(pred, gold)))
            hitstar.append(float(eval_hit(pred, gold)))
            f1s.append(float(eval_f1(pred, gold)[0]))
        else:
            hits1.append(0.0); hitstar.append(0.0); f1s.append(0.0)

        if include_strict:
            pred_n, goldset = [], set(normalize(a) for a in gold)
            for p in pred:
                pn_ = normalize(p)
                if pn_ and pn_ not in pred_n:
                    pred_n.append(pn_)
            s_hits1.append(1.0 if pred_n and pred_n[0] in goldset else 0.0)
            s_hitstar.append(1.0 if any(p in goldset for p in pred_n) else 0.0)
            s_f1s.append(_strict_set_f1(pred_n, goldset))

    if was_training:
        model.train()
    m = lambda xs: float(np.mean(xs)) if xs else 0.0
    out = {f"{prefix}_hits1": m(hits1), f"{prefix}_f1": m(f1s),
           f"{prefix}_hit_star": m(hitstar)}
    if include_strict:
        out.update({f"{prefix}_hits1_strict": m(s_hits1), f"{prefix}_f1_strict": m(s_f1s),
                    f"{prefix}_hit_star_strict": m(s_hitstar)})
    return out


class FlatKGQATrainer(PerDatasetEvalMixin, Trainer):
    """Plain HF Trainer with the per-dataset eval fan-out + generative metrics."""

    def __init__(self, *args, gen_tokenizer=None, gen_rows=None, gen_max_new_tokens=128,
                 gen_max_samples=None, gen_answer_sep="\n",
                 selection_dataset=None, log_strict_metrics=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_tokenizer = gen_tokenizer
        self._gen_rows = gen_rows          # {"eval_<ds>": rows, "test_<ds>": rows}
        self._gen_max_new_tokens = gen_max_new_tokens
        self._gen_max_samples = gen_max_samples
        self._gen_answer_sep = gen_answer_sep
        self._selection_dataset = selection_dataset
        self._log_strict_metrics = log_strict_metrics

    def set_gen_max_samples(self, n):
        self._gen_max_samples = n

    def _generative_metrics(self, handle, prefix):
        return flat_generative_eval(
            self.model, self._gen_rows[prefix], self._gen_tokenizer,
            max_new_tokens=self._gen_max_new_tokens, max_samples=self._gen_max_samples,
            device=self.args.device, prefix=prefix,
            answer_sep=self._gen_answer_sep, include_strict=self._log_strict_metrics)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_flat_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run = (f"{sweep_id}_{run_name}" if sweep_id and run_name
                    else f"kgqa_flat_lora{cfg.lora_r}_s{cfg.seed}")
    output_dir = f"./checkpoints/{EXPERIMENT_NAME}/{internal_run}"
    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    question_end = tokenizer(cfg.question_end_str, add_special_tokens=False)["input_ids"]

    print("Loading + tokenizing flat data...")

    def load_rows(dataset, split):
        view = cfg.for_dataset(dataset)
        path = os.path.join(OUTPUT_ROOT, flat_data_config_key(view, dataset),
                            f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No flat dataset at {path} — build it first with --mode flat_data_prep.")
        rows = [tokenize_row(json.loads(l), tokenizer, question_end, cfg.seq_len)
                for l in open(path)]
        print(f"  {dataset}/{split}: {len(rows)} rows")
        return rows

    trains = [FlatVersionedDataset(load_rows(ds, "train"),
                                   versions=cfg.resolved_versions(ds))
              for ds in cfg.train_datasets]
    train_dataset = trains[0] if len(trains) == 1 else ConcatDataset(trains)
    dev_rows = {ds: load_rows(ds, "dev") for ds in cfg.eval_datasets}
    test_rows = {ds: load_rows(ds, "test") for ds in cfg.eval_datasets}
    eval_sets = {ds: FlatVersionedDataset(rows, versions=1)
                 for ds, rows in dev_rows.items()}
    test_sets = {ds: FlatVersionedDataset(rows, versions=1)
                 for ds, rows in test_rows.items()}

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.torch_dtype, attn_implementation="sdpa")
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model = select_active_params(model, active_params=[], lora=cfg.lora_config())
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
    print_trainable_parameters(model)

    steps_per_epoch = max(1, len(train_dataset) // (cfg.batch_size * cfg.accumulation_steps))
    total_steps = steps_per_epoch * cfg.num_epochs
    gc = cfg.gradient_checkpointing

    training_args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        output_dir=output_dir,
        logging_steps=1,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation_steps,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gc else None,
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=(cfg.num_workers > 0),
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        metric_for_best_model="eval_sel_f1",   # see PerDatasetEvalMixin
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        run_name=internal_run,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg.lr / 10},
        warmup_steps=total_steps // 10,
        weight_decay=0.1,
        seed=cfg.seed,
        data_seed=cfg.seed,
    )

    gen_rows = {}
    for ds in cfg.eval_datasets:
        gen_rows[f"eval_{ds}"] = dev_rows[ds]
        gen_rows[f"test_{ds}"] = test_rows[ds]

    trainer = FlatKGQATrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_sets,   # {name: dataset}
        data_collator=PadCollator(tokenizer.eos_token_id),
        gen_tokenizer=tokenizer,
        gen_rows=gen_rows,
        gen_max_new_tokens=cfg.gen_max_new_tokens,
        gen_max_samples=(cfg.gen_eval_samples if cfg.gen_eval_samples is not None
                         else cfg.gen_max_samples),
        gen_answer_sep=cfg.answer_parse_sep,
        selection_dataset=cfg.selection_dataset,
        log_strict_metrics=cfg.log_strict_metrics,
    )

    trainer.train()

    trainer.set_gen_max_samples(cfg.gen_max_samples)
    print("\n" + "=" * 50 + "\nBest model — dev set(s) (generative):\n" + "=" * 50)
    dev_metrics = trainer.evaluate(eval_dataset=eval_sets, metric_key_prefix="eval")
    for k, v in dev_metrics.items():
        if any(t in k for t in _PRINT_KEYS):
            print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 50 + "\nBest model — test set(s) (generative):\n" + "=" * 50)
    test_metrics = trainer.evaluate(eval_dataset=test_sets, metric_key_prefix="test")
    for k, v in test_metrics.items():
        if any(t in k for t in _PRINT_KEYS):
            print(f"  {k}: {v:.4f}")

    record = {
        "mode": "flat_train", "arm": "flat_text",
        **sweep_meta, "run_name": internal_run,
        "train_datasets": list(cfg.train_datasets),
        "eval_datasets": list(cfg.eval_datasets),
        "selection_dataset": cfg.selection_dataset,
        "model_name": cfg.model_name, "prompt_style": cfg.resolved_prompt_style,
        "lora_r": cfg.lora_r, "lora_dropout": cfg.lora_dropout, "seq_len": cfg.seq_len,
        "lr": cfg.lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "max_steps": cfg.max_steps, "seed": cfg.seed,
        "rel_mode": cfg.rel_mode, "max_nodes": cfg.max_nodes, "n_max": cfg.n_max,
        "versions": cfg.versions, "data_seed": cfg.data_seed,
        "cvt_collapse": cfg.resolved_cvt_collapse("flat"),
        "flat_shuffle_lines": cfg.flat_shuffle_lines,
        **result_block(cfg, dev_metrics, test_metrics),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended flat run to {runs_jsonl}")

    if report_to == "wandb":
        import wandb
        if wandb.run is not None:
            wandb.finish()
