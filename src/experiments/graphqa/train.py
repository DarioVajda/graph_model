"""
Train ONE GraphQA configuration and log ONE record.

Protocol: train on the train split, evaluate + checkpoint every ``eval_steps``,
reload the best-validation checkpoint at the end (HF ``load_best_model_at_end``),
and report that checkpoint's **test** exact match. Selection is on validation
accuracy — the metric the experiment actually reports — rather than validation
loss, so the chosen checkpoint is the most accurate one rather than the most
confident one.

The model is the v2 GTLM stack, built directly as in the kgqa experiment; the
backbone adapter follows from ``model_name`` (``config.BACKBONES``). See ``config.py``
for why the legacy v0 path was dropped.
"""

import os

import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

from ...utils import GraphCollatorV2, GraphTrainerV2, set_wandb_project
from ...utils.text_graph_trainer_v2 import make_compute_metrics, shift_logits_for_metrics
from ...train import select_active_params, print_trainable_parameters, get_device
from .config import EXPERIMENT_NAME, EXPERIMENT_DIR
from .data import load_data
from ._io import append_jsonl

# Fallback results path for standalone runs (no --runs-jsonl from the sweep runner).
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")

# v2 names every graph-bias module under one umbrella prefix.
ACTIVE_PARAMS = ["graph_bias"]
METRIC = "eval_em_accuracy"     # exact match over the answer span


def _save_train_record(cfg, run_name, sizes, results, runs_jsonl, sweep_meta=None):
    """Append this run's axes, hyperparameters and metrics as one JSONL line."""
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # ── the experiment's axes (analysis/prep_table.py groups on these) ──
        "task": cfg.task, "graph_type": cfg.graph_type, "arm": cfg.arm(),
        "question_node": cfg.question_node,
        "spd": cfg.spd, "rrwp": cfg.rrwp, "magnetic": cfg.magnetic,
        # Bias sharing granularity (0 = legacy per-layer). A first-class column:
        # a G sweep is otherwise only distinguishable by parsing the run name.
        "magnetic_groups": cfg.magnetic_groups,
        # First-class columns for the same reason magnetic_groups is one: without
        # them the only way to tell a linear arm from a magnetic one in runs.jsonl
        # is to parse the run name, which is fragile and silently wrong.
        "magnetic_linear": cfg.magnetic_linear,
        "magnetic_magnitude": cfg.magnetic_magnitude,
        "magnetic_hybrid": cfg.magnetic_hybrid,
        "magnetic_linear_v2": cfg.magnetic_linear_v2,
        "magnetic_m_collate": cfg.collate_magnetic_m,
        "bias_self_node": cfg.bias_self_node,
        "seed": cfg.seed,
        # ── hyperparameters ──
        "model_name": cfg.model_name, "impl": cfg.impl, "dtype": cfg.dtype,
        "k_hop": cfg.k_hop, "k_hop_directed": cfg.k_hop_directed,
        "max_spd": cfg.max_spd, "max_rw_steps": cfg.max_rw_steps,
        "magnetic_dim": cfg.magnetic_dim, "magnetic_q": cfg.magnetic_q,
        "magnetic_m": cfg.magnetic_m,
        "lora": cfg.lora, "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_r * 2 if cfg.lora else None,
        "lora_dropout": cfg.lora_dropout,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "eval_steps": cfg.eval_steps, "max_steps": cfg.max_steps,
        "max_length": cfg.max_length,
        # ── data provenance ──
        "val_source": "official" if cfg.has_official_val() else "carved_from_train",
        "train_size": sizes[0], "val_size": sizes[1], "test_size": sizes[2],
        # ── results ──
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    """Train this config; log one record (the best-val checkpoint's test accuracy)."""
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run = f"{sweep_id}_{run_name}" if (sweep_id and run_name) else cfg.run_name()

    report_to = "wandb" if cfg.wandb_project else "none"
    # Set the project BEFORE the trainer is built: its WandbCallback initializes
    # lazily, so setting it afterwards only works by accident of that lazy init.
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    device = get_device()
    train_dataset, val_dataset, test_dataset = load_data(cfg)
    sizes = (len(train_dataset), len(val_dataset), len(test_dataset))

    # Seed before the build so the graph-bias parameter init is determined by
    # `seed` too, not just the data order — this pairs arms within a seed. (HF's
    # Trainer seeds itself, but only once the model already exists.)
    set_seed(cfg.seed)

    # Llama by default; BLOOM (ALiBi) or Gemma-3 (layer-heterogeneous RoPE) when
    # model_name names one — the positional-encoding probes. All three are the same
    # GTLM stack; only the thin backbone adapter differs.
    gtlm_config_cls, gtlm_model_cls = cfg.gtlm_classes()
    config = gtlm_config_cls.from_pretrained(
        cfg.model_name, **cfg.bias_params(),
        k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        graph_attn_impl=cfg.backend(),
        **({"flex_compile_mode": cfg.flex_compile_mode} if cfg.backend() == "flex" else {}),
    )
    model = gtlm_model_cls.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
        torch_dtype=cfg.torch_dtype())
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = select_active_params(model, active_params=ACTIVE_PARAMS, lora=cfg.lora_config())
    print_trainable_parameters(model)

    # The collator's eigenvector cap (0 -> keep all, which is what the cached
    # datasets store); it is NOT the model's magnetic_dim. Gated on `uses_magnetic`,
    # not `magnetic`, so the linear arm is fed the same eigenvectors.
    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=cfg.collate_magnetic_m,
        pad_to_block=(cfg.backend() == "flex"), max_spd=cfg.max_spd)

    steps_per_epoch = max(1, len(train_dataset) // cfg.batch_size // cfg.accumulation_steps)
    gc = cfg.gradient_checkpointing
    training_args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        output_dir=f"./checkpoints/{EXPERIMENT_NAME}/{internal_run}",
        logging_steps=1,
        per_device_train_batch_size=cfg.batch_size,
        # Match eval batch to train batch: flex compiles per (B, L, N) shape family,
        # so HF's default eval batch of 8 would mint a second kernel set.
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
        warmup_steps=steps_per_epoch * 2,
        weight_decay=0.1,

        # Best-val-checkpoint protocol: eval + save every eval_steps, keep the best,
        # reload it at the end for the single test pass.
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
        compute_metrics=make_compute_metrics(include_f1=cfg.include_f1),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        active_params=ACTIVE_PARAMS,
        bias_lr=cfg.bias_lr,
    )

    train_output = trainer.train()
    # The best checkpoint is loaded now: score it on val (sanity) and on test (the
    # reported metric). GraphTrainerV2.get_eval_dataloader invalidates HF's
    # constant-"eval"-key dataloader cache when a different split is passed, so
    # these two calls really do evaluate val and then test.
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    results = {
        "test_accuracy": test_metrics.get("test_em_accuracy"),
        "best_val_accuracy": val_metrics.get(METRIC),
        "test_loss": test_metrics.get("test_loss"),
        "train_runtime_s": train_output.metrics.get("train_runtime"),
        "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
    }
    if cfg.include_f1:
        results["test_f1"] = test_metrics.get("test_em_f1")
    print(f"[results] {cfg.task}/{cfg.graph_type} [{cfg.arm()}] "
          f"test_accuracy={results['test_accuracy']} "
          f"(best-val={results['best_val_accuracy']}) "
          f"runtime={results['train_runtime_s']}s")

    _save_train_record(cfg, internal_run, sizes, results, runs_jsonl=runs_jsonl,
                       sweep_meta=sweep_meta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
