"""
Train ONE probe configuration (task, bias arm, k_hop, seed) and log ONE record.

Protocol (README, suite-wide conventions): train with val evaluation +
checkpointing every ``eval_steps`` steps, reload the best-val checkpoint at the
end (HF ``load_best_model_at_end``), and report that checkpoint's **test**
accuracy. Speed is a first-class result: wall-clock step time, peak memory and
token/block sparsity are recorded alongside accuracy in ``runs.jsonl``.

The v2 model/collator construction and the instrumentation are reused from the
expressiveness experiment (``training/dispatch.py`` / ``instrumentation.py``) —
same machinery, same metric definitions, so numbers are comparable across the
two experiments.
"""

import os

import torch
from transformers import TrainingArguments, set_seed

from ...utils import GraphTrainerV2, make_compute_metrics, shift_logits_for_metrics
from ..expressiveness.training.dispatch import (
    build_model, build_collator, select_active_params, print_trainable_parameters,
)
from ..expressiveness.training.instrumentation import StepMemCallback, measure_density
from ..expressiveness.data.datasets import calculate_label_distribution
from ..expressiveness.config import tokenized_label_options
from .config import EXPERIMENT_NAME, POSSIBLE_LABELS
from .data import load_data
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback results path for standalone runs (no --runs-jsonl from the sweep runner).
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")

ACTIVE_PARAMS = ["graph_bias"]      # v2 bias modules (incl. shared_graph_bias)
METRIC = "eval_em_accuracy"         # prompt-span exact match (the Yes/No token)


def _save_train_record(cfg, run_name, results, runs_jsonl, sweep_meta=None):
    """Append this run's hyperparameters + metrics to its JSONL."""
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # ── the suite's axes ──
        "task": cfg.task, "bias": cfg.bias, "k_hop": cfg.k_hop, "seed": cfg.seed,
        # ── hyperparameters ──
        "model_name": cfg.model_name, "impl": cfg.impl,
        "k_hop_directed": cfg.k_hop_directed,
        "min_nodes": cfg.node_range()[0], "max_nodes": cfg.node_range()[1],
        "train_size": cfg.train_size, "val_size": cfg.val_size, "test_size": cfg.test_size,
        "data_seed": cfg.data_seed, "ordering": cfg.ordering,
        "p_bidirectional": cfg.p_bidirectional,
        "magnetic_dim": cfg.magnetic_dim, "magnetic_q": cfg.magnetic_q,
        "magnetic_m": cfg.magnetic_m, "max_spd": cfg.max_spd,
        "lora": cfg.lora, "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_r * 2,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "eval_steps": cfg.eval_steps, "max_steps": cfg.max_steps,
        "flex_compile_mode": cfg.flex_compile_mode,
        "len_buckets": list(cfg.len_buckets) if cfg.len_buckets else None,
        "node_buckets": list(cfg.node_buckets) if cfg.node_buckets else None,
        # ── results ──
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def run_train_mode(cfg, tokenizer, pad_token_id, runs_jsonl=None, run_name=None, sweep_id=None):
    """Train this config; log one record (best-val checkpoint's test accuracy)."""
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run_name = f"{sweep_id}_{run_name}" if (sweep_id and run_name) else cfg.run_name()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_labels = tokenized_label_options(tokenizer, labels=POSSIBLE_LABELS)
    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    train_dataset, val_dataset, test_dataset = load_data(cfg)
    for name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        yes, no = calculate_label_distribution(ds, tokenized_labels)
        print(f"[data] {name}: {len(ds)} examples — {yes:.1f}% Yes / {no:.1f}% No")

    # Token/block sparsity is a property of the data + k_hop (not of training);
    # measured once on a val subset so the run record carries it.
    density = None
    if cfg.measure_density:
        print(f"[density] measuring token/block sparsity at k_hop={cfg.k_hop} ...")
        density = measure_density(
            val_dataset, tokenizer, pad_token_id, cfg.magnetic_m, cfg.k_hop,
            cfg.k_hop_directed, batch_size=cfg.batch_size,
            num_sample_graphs=cfg.density_sample_graphs,
            num_sample_batches=cfg.density_sample_batches, device=device)

    # Reset RNG before the build so bias-param init + data order are fully
    # determined by `seed` (init-pairs arms within a seed).
    set_seed(cfg.seed)
    model, _ = build_model(cfg.impl, cfg.model_name, cfg.model_bias_config(),
                           cfg.k_hop, cfg.k_hop_directed, device, cfg.flex_compile_mode)
    model = select_active_params(model, active_params=ACTIVE_PARAMS, lora=cfg.lora_config())
    print_trainable_parameters(model)

    collator = build_collator(cfg.impl, tokenizer, pad_token_id, cfg.k_hop,
                              cfg.k_hop_directed, magnetic_m=cfg.magnetic_m,
                              len_buckets=cfg.len_buckets, node_buckets=cfg.node_buckets)

    steps_per_epoch = max(1, len(train_dataset) // cfg.batch_size // cfg.accumulation_steps)
    training_args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        output_dir=f"./checkpoints/{internal_run_name}",
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_steps=1,
        per_device_train_batch_size=cfg.batch_size,
        # Match eval batch to train batch: flex compiles per (B, L, N) shape
        # family; HF's default eval batch of 8 would spawn a second kernel set.
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=(cfg.num_workers > 0),
        report_to=report_to,
        run_name=internal_run_name,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg.lr / 10},
        warmup_steps=steps_per_epoch,
        weight_decay=0.1,
        # Best-val-checkpoint protocol (README): eval + save every eval_steps,
        # keep the best, reload it at the end for the test pass.
        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.eval_steps,
        metric_for_best_model=METRIC, greater_is_better=True,
        save_total_limit=2, load_best_model_at_end=True,
    )

    step_mem = StepMemCallback()
    trainer = GraphTrainerV2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        active_params=ACTIVE_PARAMS,
        bias_lr=cfg.bias_lr,
        callbacks=[step_mem],
    )

    train_output = trainer.train()
    # Best checkpoint is loaded now: score it on val (sanity) and on TEST (the
    # suite's reported metric). GraphTrainerV2.get_eval_dataloader invalidates
    # HF's constant-"eval"-key dataloader cache when a different split object is
    # passed, so these two calls evaluate val and test respectively (not val
    # twice — see the trainer for the HF caching footgun this guards against).
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    perf = step_mem.summary()

    results = {
        "test_accuracy": test_metrics.get("test_em_accuracy"),
        "best_val_accuracy": val_metrics.get(METRIC),
        # Wall-clock speed (the decision rule's cost axis) + memory.
        "train_runtime_s": train_output.metrics.get("train_runtime"),
        "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
        "step_ms_mean": perf.get("step_ms_mean"),
        "step_ms_median": perf.get("step_ms_median"),
        "peak_gb": perf.get("peak_gb"),
        "n_steps": perf.get("n_steps"),
        "token_sparsity": density.get("token_sparsity_mean") if density else None,
        "block_sparsity": density.get("block_sparsity_mean") if density else None,
    }
    print(f"[results] test_accuracy={results['test_accuracy']} "
          f"(best-val={results['best_val_accuracy']}) "
          f"runtime={results['train_runtime_s']}s peak={results['peak_gb']}GB")

    _save_train_record(cfg, internal_run_name, results, runs_jsonl=runs_jsonl, sweep_meta=sweep_meta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
