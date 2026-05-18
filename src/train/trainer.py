import os
import json
import datetime

from transformers import TrainingArguments

from ..utils import GraphTrainer, set_wandb_project
from .eval import make_compute_metrics, _filter_results


def save_run_metadata(run_name, experiment_dir, base_model, active_params, lr, bias_lr, lora_config, num_epochs, **extra_metadata):
    """
    Persist run metadata to experiment_dir/run_metadata.json.
    Appends a version suffix (_v2, _v3, …) if run_name already exists.
    Returns the final run_name used.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    metadata_path = os.path.join(experiment_dir, "run_metadata.json")

    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            json.dump({}, f)
    with open(metadata_path) as f:
        run_metadata = json.load(f)

    if run_name in run_metadata:
        version = 2
        candidate = f"{run_name}_v{version}"
        while candidate in run_metadata:
            version += 1
            candidate = f"{run_name}_v{version}"
        run_name = candidate

    new_entry = {
        run_name: {
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": base_model,
            "active_params": active_params,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "bias_learning_rate": bias_lr,
            "lora_config": lora_config,
            **extra_metadata,
        }
    }
    run_metadata = {**new_entry, **run_metadata}

    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=4)

    return run_name


def training_run(
    model,
    train_dataset,
    eval_dataset,
    test_dataset,
    collator,
    run_name,
    experiment_name,
    experiment_dir,
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    bias_learning_rate=1e-3,
    accumulation_steps=4,
    active_params=None,
    eval_every=40,
    gradient_checkpointing=True,
    include_f1=False,
    wandb_project=None,
    seed=42,
):
    """
    Run a full training loop with evaluation.

    Args:
        experiment_name: Used to construct checkpoint path: ./checkpoints/{experiment_name}/{run_name}
        experiment_dir:  Directory for run_metadata.json and results.json (the experiment's source folder).
        wandb_project:   WandB project name. Pass None to disable WandB logging.
    """
    print(f"Gradient checkpointing {'ENABLED' if gradient_checkpointing else 'DISABLED'}.")

    if wandb_project is not None:
        set_wandb_project(wandb_project)

    steps_per_epoch = max(1, len(train_dataset) // batch_size // accumulation_steps)
    total_steps = steps_per_epoch * num_epochs

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=f"./checkpoints/{experiment_name}/{run_name}",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,

        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        metric_for_best_model="eval_em_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True,

        report_to="wandb" if wandb_project is not None else "none",
        run_name=run_name,

        learning_rate=learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": learning_rate / 10},
        warmup_steps=total_steps // 10,
        weight_decay=0.1,

        seed=seed,
        data_seed=seed,
    )

    trainer = GraphTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(include_f1=include_f1),
        active_params=active_params,
        bias_lr=bias_learning_rate,
    )

    trainer.train()

    if test_dataset is None:
        print("No test dataset provided. Skipping final evaluation.")
        if wandb_project is not None:
            import wandb
            wandb.finish()
        return None

    print("\n" + "=" * 50)
    print("Training complete. Evaluating best model on test set...")
    print("=" * 50)

    raw_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    results = _filter_results(raw_results)

    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50 + "\n")

    os.makedirs(experiment_dir, exist_ok=True)
    results_path = os.path.join(experiment_dir, "results.json")
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            json.dump({}, f)
    with open(results_path) as f:
        all_results = json.load(f)
    all_results = {run_name: results, **all_results}
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    if wandb_project is not None:
        import wandb
        wandb.finish()

    return results
