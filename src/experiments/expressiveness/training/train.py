"""
Training (small graphs → accuracy / parity) for the expressiveness experiment.

``training_run`` trains one ``(impl, k_hop, seed)`` config and returns both its
eval accuracy and the live speed/memory summary (:class:`StepMemCallback`).
``run_train_mode`` drives the multi-seed × k_hop × impl sweep and emits a
combined table: accuracy + ms/step + peak GB + token/block sparsity, the last
measured standalone via :func:`measure_density`.
"""

import os
import statistics

import torch
from transformers import TrainingArguments, set_seed

from ....utils import GraphTrainer, GraphTrainerV2, make_compute_metrics, shift_logits_for_metrics
from ..config import ACCURACY_METRIC, run_suffix, tokenized_label_options
from .dispatch import (
    parse_impl, build_model, build_collator, active_params_for,
    select_active_params, print_trainable_parameters,
)
from .evaluation import smuggle_prediction_step, PreprocessLogits, ComputeMetrics
from ..data.datasets import calculate_label_distribution, load_or_create_dataset
from .instrumentation import StepMemCallback, measure_density, format_results_table
from .._io import append_jsonl


def make_trainer(impl, model, training_args, train_dataset, eval_dataset, collator,
                 active_params, bias_lr, label_options, pad_token_id, callbacks=None):
    """Build the right Trainer + metrics for ``impl``.

    v0 needs the prediction-step surgery (`smuggle_prediction_step`) to score the
    Yes/No token; v2 emits standard (B, L) labels so the trainer's own
    `shift_logits_for_metrics` + `make_compute_metrics` (prompt-only exact match)
    suffice.
    """
    version, _ = parse_impl(impl)
    if version == "v0":
        if label_options is None or pad_token_id is None:
            raise ValueError("v0 evaluation needs label_options and pad_token_id.")
        return GraphTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=ComputeMetrics(label_options=label_options),
            preprocess_logits_for_metrics=PreprocessLogits(label_options=label_options, pad_token_id=pad_token_id),
            custom_prediction_step=smuggle_prediction_step,
            active_params=active_params,
            bias_lr=bias_lr,
            callbacks=callbacks,
        )
    return GraphTrainerV2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        active_params=active_params,
        bias_lr=bias_lr,
        callbacks=callbacks,
    )


def training_run(
    impl,
    model,
    train_dataset,
    eval_dataset,
    collator,
    run_name,
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    bias_learning_rate=1e-3,
    accumulation_steps=4,
    label_options=None,
    pad_token_id=None,
    active_params=None,
    report_to="none",
    eval_steps=25,
    seed=42,
    max_steps=-1,
    num_workers=0,
):
    """Train one config; return ``(accuracy, step_mem_summary)``.

    ``eval_dataset=None`` runs a no-eval training pass (used by the quick
    large-graph smoke test); ``max_steps>0`` caps the number of optimizer steps.
    """
    version, _ = parse_impl(impl)
    metric = ACCURACY_METRIC[version]
    has_eval = eval_dataset is not None
    steps_per_epoch = max(1, len(train_dataset) // batch_size // accumulation_steps)

    eval_save_kwargs = dict(
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps,
        metric_for_best_model=metric, greater_is_better=True,
        save_total_limit=2, load_best_model_at_end=True,
    ) if has_eval else dict(eval_strategy="no", save_strategy="no", load_best_model_at_end=False)

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        output_dir=f"./checkpoints/{run_name}",
        seed=seed,
        data_seed=seed,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        # Match eval batch to train batch so flex compiles a single batch-size
        # family. HF defaults per_device_eval_batch_size to 8; with static-shape
        # max-autotune that 8 != train's 4 spawns a whole separate set of
        # (L x N x fwd) eval kernels (and post-eval recompile spikes).
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=False,

        # Overlap the per-batch O(N^2) feature construction with GPU compute.
        # persistent_workers avoids re-forking the pool every epoch / eval loop.
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=(num_workers > 0),

        report_to=report_to,
        run_name=run_name,

        learning_rate=learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": learning_rate / 10},
        warmup_steps=steps_per_epoch,
        weight_decay=0.1,
        **eval_save_kwargs,
    )

    step_mem = StepMemCallback()
    trainer = make_trainer(
        impl, model, training_args, train_dataset, eval_dataset, collator,
        active_params=active_params, bias_lr=bias_learning_rate,
        label_options=label_options, pad_token_id=pad_token_id,
        callbacks=[step_mem],
    )

    trainer.train()
    accuracy = trainer.evaluate().get(metric) if has_eval else None
    return accuracy, step_mem.summary()


def _finish_wandb():
    """End the active wandb run so the next (impl, k_hop, seed) starts a fresh one.

    Multiple configs train in one process (the k_hop / impl loops). HF's
    WandbCallback only ``wandb.init()``s when ``wandb.run is None``, so without an
    explicit finish between runs every config after the first silently logs into
    the first run. Calling ``wandb.finish()`` resets that.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def _save_train_record(cfg, impl, k_hop, seed, run_name, accuracy, step_mem, dens,
                       runs_jsonl, sweep_meta=None):
    """Append one training run's hyperparameters + results to its sweep's JSONL."""
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # ── hyperparameters ──
        "impl": impl, "k_hop": k_hop, "seed": seed,
        "model_name": cfg.model_name, "difficulty": cfg.difficulty,
        "num_nodes": cfg.num_nodes, "min_nodes": cfg.min_nodes, "max_nodes": cfg.max_nodes,
        "train_dataset_size": cfg.train_dataset_size, "eval_dataset_size": cfg.eval_dataset_size,
        "ordering": cfg.ordering, "magnetic_m": cfg.magnetic_m,
        "k_hop_directed": cfg.k_hop_directed, "flex_compile_mode": cfg.flex_compile_mode,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "max_steps": cfg.max_steps,
        "lora": cfg.lora, "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "len_buckets": list(cfg.len_buckets) if cfg.len_buckets else None,
        "node_buckets": list(cfg.node_buckets) if cfg.node_buckets else None,
        # ── results ──
        "eval_accuracy": accuracy,
        "step_ms_mean": step_mem.get("step_ms_mean"),
        "step_ms_median": step_mem.get("step_ms_median"),
        "peak_gb": step_mem.get("peak_gb"),
        "n_steps": step_mem.get("n_steps"),
        "token_sparsity": dens.get("token_sparsity_mean") if dens else None,
        "block_sparsity": dens.get("block_sparsity_mean") if dens else None,
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


# Fallback results path when run standalone (no --runs-jsonl passed by the sweep
# runner). Anchored to the experiment package's results/ dir.
_DEFAULT_RUNS_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "train_runs.jsonl")


def run_train_mode(cfg, tokenizer, pad_token_id, runs_jsonl=None, run_name=None, sweep_id=None):
    """Train this config (one (impl, k_hop, seed) — the train loop runs once).

    ``runs_jsonl`` / ``run_name`` / ``sweep_id`` are the sweep runner's bookkeeping
    (where to log + which sweep/run this is); when absent (a standalone run),
    records fall back to the package's ``results/train_runs.jsonl``.
    """
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_labels = tokenized_label_options(tokenizer)
    suffix = run_suffix(cfg.magnetic_m)
    report_to = "wandb" if cfg.wandb_project else "none"

    train_dataset = load_or_create_dataset(
        cfg.train_dataset_size, cfg.is_easy, cfg.model_name,
        cfg.min_nodes, cfg.max_nodes, ordering=cfg.ordering, magnetic_m=cfg.magnetic_m)
    eval_dataset = load_or_create_dataset(
        cfg.eval_dataset_size, cfg.is_easy, cfg.model_name,
        cfg.min_nodes, cfg.max_nodes, ordering=cfg.ordering, magnetic_m=cfg.magnetic_m)

    train_yes, train_no = calculate_label_distribution(train_dataset, tokenized_labels)
    eval_yes, eval_no = calculate_label_distribution(eval_dataset, tokenized_labels)
    print("!" * 100)
    print(f"Training dataset label distribution: {train_yes:.2f}% Yes, {train_no:.2f}% No")
    print(f"Evaluation dataset label distribution: {eval_yes:.2f}% Yes, {eval_no:.2f}% No")
    print("!" * 100)

    # Token/block sparsity is a property of k_hop + the data, so measure it once
    # per k_hop on a random subset of the eval set (backend-independent). Measured
    # up front so each saved run record can carry its sparsity.
    density = {}
    if cfg.measure_density:
        for k_hop in cfg.k_hops:
            print(f"\n[density] measuring token/block sparsity at k_hop={k_hop} ...")
            density[k_hop] = measure_density(
                eval_dataset, tokenizer, pad_token_id, cfg.magnetic_m, k_hop,
                cfg.k_hop_directed, batch_size=cfg.batch_size,
                num_sample_graphs=cfg.density_sample_graphs,
                num_sample_batches=cfg.density_sample_batches, device=device)

    acc = {}      # (impl, k_hop, seed) -> accuracy
    perf = {}     # (impl, k_hop, seed) -> step_mem summary
    for seed in cfg.seeds:
        for k_hop in cfg.k_hops:
            for impl in cfg.impls:
                version, _ = parse_impl(impl)
                # v0 is dense (k_hop-agnostic); run it once at k_hop = k_hops[0].
                if version == "v0" and k_hop != cfg.k_hops[0]:
                    continue

                # Reset RNG before each build so bias-param init + data order are
                # fully determined by `seed`; this also init-pairs k0/k1 within a
                # seed, isolating the k_hop effect from initialization noise.
                set_seed(seed)

                lora_tag = f"_lora{cfg.lora_r}" if cfg.lora else ""
                # Lead with the graph size and no backend/difficulty prefix, so this
                # batch is easy to tell apart from earlier runs on wandb.
                run_name = f"n{cfg.num_nodes}_k{k_hop}_s{seed}{lora_tag}_{suffix}"
                print("\n" + "#" * 100)
                print(f"# Training {impl} (k_hop={k_hop}, seed={seed}) — run '{run_name}'")
                print("#" * 100)

                model, _ = build_model(impl, cfg.model_name, cfg.model_bias_config(), k_hop,
                                       cfg.k_hop_directed, device, cfg.flex_compile_mode)
                active_params = active_params_for(impl)
                model = select_active_params(model, active_params=active_params, lora=cfg.lora_config())
                print_trainable_parameters(model)

                accuracy, step_mem = training_run(
                    impl=impl,
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    collator=build_collator(impl, tokenizer, pad_token_id,
                                            k_hop, cfg.k_hop_directed, magnetic_m=cfg.magnetic_m,
                                            len_buckets=cfg.len_buckets, node_buckets=cfg.node_buckets),
                    run_name=run_name,
                    num_epochs=cfg.num_epochs,
                    batch_size=cfg.batch_size,
                    learning_rate=cfg.lr,
                    bias_learning_rate=cfg.bias_lr,
                    accumulation_steps=cfg.accumulation_steps,
                    label_options=tokenized_labels,
                    pad_token_id=pad_token_id,
                    active_params=active_params,
                    report_to=report_to,
                    eval_steps=cfg.eval_steps,
                    seed=seed,
                    max_steps=cfg.max_steps,
                    num_workers=cfg.num_workers,
                )
                acc[(impl, k_hop, seed)] = accuracy
                perf[(impl, k_hop, seed)] = step_mem
                _save_train_record(cfg, impl, k_hop, seed, run_name, accuracy,
                                   step_mem, density.get(k_hop),
                                   runs_jsonl=runs_jsonl, sweep_meta=sweep_meta)
                if report_to == "wandb":
                    _finish_wandb()
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    _report(cfg, acc, perf, density)


def _report(cfg, acc, perf, density):
    """Print the accuracy mean±std block and the combined metrics table."""
    configs = []
    for k_hop in cfg.k_hops:
        for impl in cfg.impls:
            version, _ = parse_impl(impl)
            if version == "v0" and k_hop != cfg.k_hops[0]:
                continue
            configs.append((impl, k_hop))

    print("\n" + "=" * 72)
    print(f"FINAL EVAL ACCURACY — mean ± std over seeds {tuple(cfg.seeds)}")
    print("=" * 72)
    for (impl, k_hop) in configs:
        accs = [acc[(impl, k_hop, s)] for s in cfg.seeds if acc.get((impl, k_hop, s)) is not None]
        if not accs:
            continue
        mean = sum(accs) / len(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {impl:<10} k_hop={k_hop}: {mean:.4f} ± {std:.4f}   (seeds: {', '.join(f'{a:.4f}' for a in accs)})")
    print("=" * 72)

    rows = []
    for (impl, k_hop) in configs:
        accs = [acc[(impl, k_hop, s)] for s in cfg.seeds if acc.get((impl, k_hop, s)) is not None]
        ms = [perf[(impl, k_hop, s)]["step_ms_mean"] for s in cfg.seeds if (impl, k_hop, s) in perf]
        gb = [perf[(impl, k_hop, s)]["peak_gb"] for s in cfg.seeds if (impl, k_hop, s) in perf]
        d = density.get(k_hop)
        rows.append({
            "config": f"{impl} k={k_hop}",
            "accuracy": (f"{sum(accs)/len(accs):.4f}" if accs else "—"),
            "ms_step": (f"{statistics.mean(ms):.1f}" if ms else "—"),
            "peak_gb": (f"{max(gb):.2f}" if gb else "—"),
            "token_sp": (f"{d['token_sparsity_mean']:.3f}" if d else "—"),
            "block_sp": (f"{d['block_sparsity_mean']:.3f}" if d else "—"),
        })
    print("\n" + format_results_table(rows))
    if density:
        print("token sp = element sparsity on the unpadded graph (data property); "
              "block sp = flex-packed block sparsity (realized skip rate).")
