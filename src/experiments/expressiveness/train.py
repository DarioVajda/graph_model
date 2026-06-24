"""
Training (small graphs → accuracy / parity) for the expressiveness experiment.

``training_run`` trains one ``(impl, k_hop, seed)`` config and returns both its
eval accuracy and the live speed/memory summary (:class:`StepMemCallback`).
``run_train_mode`` drives the multi-seed × k_hop × impl sweep and emits a
combined table: accuracy + ms/step + peak GB + token/block sparsity, the last
measured standalone via :func:`measure_density`.
"""

import statistics

import torch
from transformers import TrainingArguments, set_seed

from ...utils import GraphTrainer, GraphTrainerV2, make_compute_metrics, shift_logits_for_metrics
from .config import ACCURACY_METRIC, run_suffix, tokenized_label_options
from .dispatch import (
    parse_impl, build_model, build_collator, active_params_for,
    select_active_params, print_trainable_parameters,
)
from .evaluation import smuggle_prediction_step, PreprocessLogits, ComputeMetrics
from .datasets import calculate_label_distribution, load_or_create_dataset
from .instrumentation import StepMemCallback, measure_density, format_results_table


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
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=False,

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


def run_train_mode(cfg, tokenizer, pad_token_id):
    """Small-graph sweep: train every (impl, k_hop, seed) and report the combined table."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_labels = tokenized_label_options(tokenizer)
    suffix = run_suffix(cfg.bias_params)

    train_dataset = load_or_create_dataset(
        cfg.train_dataset_size, cfg.is_easy, cfg.bias_params, cfg.model_name,
        cfg.min_nodes, cfg.max_nodes, cfg.spectral_dims, ordering=cfg.ordering)
    eval_dataset = load_or_create_dataset(
        cfg.eval_dataset_size, cfg.is_easy, cfg.bias_params, cfg.model_name,
        cfg.min_nodes, cfg.max_nodes, cfg.spectral_dims, ordering=cfg.ordering)

    train_yes, train_no = calculate_label_distribution(train_dataset, tokenized_labels)
    eval_yes, eval_no = calculate_label_distribution(eval_dataset, tokenized_labels)
    print("!" * 100)
    print(f"Training dataset label distribution: {train_yes:.2f}% Yes, {train_no:.2f}% No")
    print(f"Evaluation dataset label distribution: {eval_yes:.2f}% Yes, {eval_no:.2f}% No")
    print("!" * 100)

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

                run_name = f"{cfg.difficulty}_{impl}_k{k_hop}_s{seed}_{suffix}"
                print("\n" + "#" * 100)
                print(f"# Training {impl} (k_hop={k_hop}, seed={seed}) — run '{run_name}'")
                print("#" * 100)

                model, _ = build_model(impl, cfg.model_name, cfg.bias_params, k_hop,
                                       cfg.k_hop_directed, device, cfg.flex_compile_mode)
                active_params = active_params_for(impl)
                model = select_active_params(model, active_params=active_params, lora=None)
                print_trainable_parameters(model)

                accuracy, step_mem = training_run(
                    impl=impl,
                    model=model,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    collator=build_collator(impl, tokenizer, pad_token_id, cfg.bias_params,
                                            k_hop, cfg.k_hop_directed),
                    run_name=run_name,
                    num_epochs=cfg.num_epochs,
                    batch_size=cfg.batch_size,
                    learning_rate=cfg.lr,
                    bias_learning_rate=cfg.bias_lr,
                    accumulation_steps=cfg.accumulation_steps,
                    label_options=tokenized_labels,
                    pad_token_id=pad_token_id,
                    active_params=active_params,
                    report_to=cfg.report_to,
                    eval_steps=cfg.eval_steps,
                    seed=seed,
                    max_steps=cfg.max_steps,
                )
                acc[(impl, k_hop, seed)] = accuracy
                perf[(impl, k_hop, seed)] = step_mem
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Token/block sparsity is a property of k_hop + the data, so measure it once
    # per k_hop on a random subset of the eval set (backend-independent).
    density = {}
    if cfg.measure_density:
        for k_hop in cfg.k_hops:
            print(f"\n[density] measuring token/block sparsity at k_hop={k_hop} ...")
            density[k_hop] = measure_density(
                eval_dataset, tokenizer, pad_token_id, cfg.bias_params, k_hop,
                cfg.k_hop_directed, batch_size=cfg.batch_size,
                num_sample_graphs=cfg.density_sample_graphs,
                num_sample_batches=cfg.density_sample_batches, device=device)

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
