"""Train ONE RelBench configuration and log ONE record.

Protocol, matched deliberately to the baselines' own runner
(``snap-stanford/relbench:examples/gnn_entity.py``): train on the train split, evaluate and
checkpoint every ``eval_steps``, select on **validation** ``roc_auc`` (higher is better),
reload that checkpoint, and score **test** exactly once. The baselines do the same
``model.load_state_dict(state_dict)`` before their final eval, so anything else would not be
comparable.

``GraphTrainerV2`` is not interchangeable with a stock ``Trainer`` here: it saves a checkpoint
in two parts (the LoRA adapter and the graph-bias tensors) and overrides ``_load_best_model``
to restore both. HF's own implementation knew only about the adapter, which silently paired a
best-step adapter with end-of-run bias weights -- a combination that never existed during
training, and understated every reported metric.

There is no prediction head anywhere. The score is read off the LM head
(:mod:`evaluate`), so what gets measured is GTLM rather than a probe fitted on top of it.
"""

import os

import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

from ...models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from ...utils import GraphCollatorV2, GraphTrainerV2, set_wandb_project
from ...train import select_active_params, print_trainable_parameters, get_device
from .config import EXPERIMENT_NAME, EXPERIMENT_DIR
from .data import load_data, make_builders
from .evaluate import (
    answer_token_ids, evaluate_split, make_compute_metrics, make_margin_preprocessor,
)
from ._io import append_jsonl

_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")

ACTIVE_PARAMS = ["graph_bias"]
# The metric the baselines tune on, under relbench's own name for it.
METRIC = "eval_roc_auc"


def _row_ids(dataset):
    return [dataset.graphs[i].graph["row_id"] for i in range(len(dataset))]


def _save_train_record(cfg, run_name, sizes, results, runs_jsonl, sweep_meta=None):
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # -- the experiment's axes --
        "dataset": cfg.dataset, "task": cfg.task,
        "arm": cfg.arm(), "spd": cfg.spd, "rrwp": cfg.rrwp, "magnetic": cfg.magnetic,
        "max_nodes": cfg.max_nodes, "neighbor_sampling": cfg.neighbor_sampling,
        "collapse_links": cfg.collapse_links, "text_mode": cfg.text_mode,
        "max_value_chars": cfg.max_value_chars, "max_node_chars": cfg.max_node_chars,
        "question_node": cfg.question_node, "prompt_node": cfg.prompt_node,
        "seed": cfg.seed,
        # -- hyperparameters --
        "model_name": cfg.model_name, "impl": cfg.impl, "dtype": cfg.dtype,
        "k_hop": cfg.k_hop, "max_spd": cfg.max_spd, "magnetic_dim": cfg.magnetic_dim,
        "magnetic_q": cfg.magnetic_q, "lora": cfg.lora, "lora_r": cfg.lora_r,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "max_steps": cfg.max_steps,
        # -- provenance --
        "data_config_key": cfg.data_config_key(), "dataset_dir": cfg.dataset_dir(),
        "n_train": sizes[0], "n_val": sizes[1], "n_test": sizes[2],
        **results,
    }
    append_jsonl(runs_jsonl or _DEFAULT_RUNS_JSONL, record)


def run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    sweep_meta = {"sweep_id": sweep_id, "sweep_run_name": run_name} if sweep_id else None
    internal_run = run_name or cfg.run_name()

    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    device = get_device()
    train_dataset, val_dataset, test_dataset = load_data(cfg)
    sizes = (len(train_dataset), len(val_dataset), len(test_dataset))

    # The task is needed for its metric list and for the `task.evaluate` cross-check. Built
    # from the same `make_builders` the cache was built with, so there is one code path
    # deciding what "this task" means.
    task, _, _, _ = make_builders(cfg)

    set_seed(cfg.seed)

    config = GTLMLlamaConfig.from_pretrained(
        cfg.model_name, **cfg.bias_params(),
        k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        graph_attn_impl=cfg.backend(),
        **({"flex_compile_mode": cfg.flex_compile_mode} if cfg.backend() == "flex" else {}),
    )
    model = GTLMLlamaForCausalLM.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
        torch_dtype=cfg.torch_dtype())
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    yes_id, no_id = answer_token_ids(tokenizer)
    print(f"[readout] logit(' yes')={yes_id} - logit(' no')={no_id}, fp32 at the answer "
          f"position")

    model = select_active_params(model, active_params=ACTIVE_PARAMS, lora=cfg.lora_config())
    print_trainable_parameters(model)

    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex"), max_spd=cfg.max_spd)

    steps_per_epoch = max(1, len(train_dataset) // cfg.batch_size // cfg.accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    gc = cfg.gradient_checkpointing
    training_args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        output_dir=f"./checkpoints/{EXPERIMENT_NAME}/{internal_run}",
        logging_steps=1,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation_steps,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gc else None,
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=(cfg.num_workers > 0),

        report_to=report_to,
        run_name=internal_run,

        learning_rate=cfg.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg.lr / 10},
        warmup_steps=max(1, total_steps // 10),
        weight_decay=0.1,

        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.eval_steps,
        metric_for_best_model=METRIC, greater_is_better=True,
        save_total_limit=1, load_best_model_at_end=True,

        seed=cfg.seed,
        data_seed=cfg.seed,
    )

    trainer = GraphTrainerV2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(task, yes_id),
        preprocess_logits_for_metrics=make_margin_preprocessor(yes_id, no_id),
        active_params=ACTIVE_PARAMS,
        bias_lr=cfg.bias_lr,
    )

    train_output = trainer.train()

    # Best-val checkpoint is loaded now. Score val (sanity) then test (the reported number),
    # each cross-checked against `task.evaluate` on the task table itself.
    pred_dir = os.path.join(EXPERIMENT_DIR, "results", "predictions")
    val_official, val_metrics = evaluate_split(
        trainer, val_dataset, task, "val", yes_id, row_ids=_row_ids(val_dataset),
        save_to=os.path.join(pred_dir, f"{internal_run}_val.npz"))
    test_official, test_metrics = evaluate_split(
        trainer, test_dataset, task, "test", yes_id, row_ids=_row_ids(test_dataset),
        save_to=os.path.join(pred_dir, f"{internal_run}_test.npz"))

    results = {
        **{f"official_{k}": v for k, v in {**val_official, **test_official}.items()},
        "test_roc_auc": test_official.get("test_roc_auc"),
        "val_roc_auc": val_official.get("val_roc_auc"),
        "test_average_precision": test_official.get("test_average_precision"),
        # Ties collapse AUROC for reasons unrelated to the model; a low count here means the
        # number below is not measuring what it looks like it is measuring.
        "test_n_distinct": test_metrics.get("final_test_n_distinct"),
        # How much of the AUROC is decided by a coin flip rather than by the model.
        "test_tied_pair_fraction": test_official.get("test_tied_pair_fraction"),
        "test_pos_rate": test_metrics.get("final_test_pos_rate"),
        "train_runtime_s": train_output.metrics.get("train_runtime"),
        "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
    }
    print(f"[results] {cfg.dataset}/{cfg.task} [{cfg.arm()}] "
          f"test_roc_auc={results['test_roc_auc']} (val={results['val_roc_auc']}) "
          f"distinct_scores={results['test_n_distinct']}/{sizes[2]} "
          f"runtime={results['train_runtime_s']}s")

    _save_train_record(cfg, internal_run, sizes, results,
                       runs_jsonl=runs_jsonl, sweep_meta=sweep_meta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
