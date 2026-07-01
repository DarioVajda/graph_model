"""
Train ONE template configuration and log ONE record.

``run_train_mode`` builds the model + dataset for a single ``RunConfig``, runs the
shared :func:`training_run` loop, and appends one JSON line (hyperparameters +
test metrics) to the sweep's ``runs.jsonl`` — the record ``sweep.report`` reads.
When run standalone (no ``--runs-jsonl``) the record falls back to the package's
``results/train_runs.jsonl``.
"""

import os

from ...utils import GraphCollator
from ...train import (
    init_model, select_active_params, print_trainable_parameters,
    training_run, get_device,
)
from .config import EXPERIMENT_NAME
from .data import load_data
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback results path for standalone runs (no --runs-jsonl from the sweep runner).
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")


def _save_train_record(cfg, run_name, results, runs_jsonl, sweep_meta=None):
    """Append this run's hyperparameters + test metrics to its JSONL."""
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # ── hyperparameters ──
        "model_name": cfg.model_name, "k_hop": cfg.k_hop,
        "lora": cfg.lora, "lora_r": cfg.lora_r,
        "spd": cfg.spd, "max_spd": cfg.max_spd,
        "rrwp": cfg.rrwp, "max_rw_steps": cfg.max_rw_steps,
        "magnetic": cfg.magnetic, "magnetic_dim": cfg.magnetic_dim,
        "magnetic_q": cfg.magnetic_q, "magnetic_m": cfg.magnetic_m,
        "num_samples": cfg.num_samples, "max_length": cfg.max_length,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "max_steps": cfg.max_steps, "seed": cfg.seed, "data_seed": cfg.data_seed,
        # ── results (test-set metrics from the best checkpoint) ──
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    """Train this config; log one record.

    ``runs_jsonl`` / ``run_name`` / ``sweep_id`` are the sweep runner's bookkeeping
    (where to log + which sweep/run this is); when absent (a standalone run) the
    record falls back to the package's ``results/train_runs.jsonl`` and the run
    name is derived from the config.
    """
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    # The sweep's run identity names the checkpoint dir (unique per run, so parallel
    # per_config jobs never collide); standalone runs derive a name from the config.
    internal_run_name = f"{sweep_id}_{run_name}" if (sweep_id and run_name) else cfg.run_name()

    device = get_device()
    model, tokenizer = init_model(
        model_name=cfg.model_name,
        device=device,
        bias_params=cfg.bias_params(),
        k_hop=cfg.k_hop,
    )

    print("Loading data...")
    train_dataset, eval_dataset, test_dataset = load_data(cfg, tokenizer)
    collator = GraphCollator(k_hop=cfg.k_hop, magnetic_m=cfg.magnetic_m)
    print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}  Test: {len(test_dataset)}")

    model = select_active_params(model, active_params=list(cfg.active_params), lora=cfg.lora_config())
    print_trainable_parameters(model)

    results = training_run(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        collator=collator,
        run_name=internal_run_name,
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=EXPERIMENT_DIR,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        bias_learning_rate=cfg.bias_lr,
        accumulation_steps=cfg.accumulation_steps,
        active_params=list(cfg.active_params),
        eval_every=cfg.eval_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        include_f1=cfg.include_f1,
        wandb_project=cfg.wandb_project,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        num_workers=cfg.num_workers,
    )

    _save_train_record(cfg, internal_run_name, results, runs_jsonl=runs_jsonl, sweep_meta=sweep_meta)
