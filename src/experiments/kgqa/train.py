"""
KGQA (SR-WebQSP) training — run ONE config, log ONE record.

``run_train_mode`` holds what used to be ``__main__.main()``: build the frozen
GTLM backbone + graph bias (optionally LoRA-adapted), train it to *generate* the
answer set for a question given its retrieved KG subgraph, select the best
checkpoint on generative macro-F1 (see ``evaluate.KGQAGraphTrainer``), then score
the best model on dev + test and append one JSONL record for ``sweep.report``.

The sweep runner invokes this via ``__main__`` (``--mode train``, the default) once
per resolved config; ``runs_jsonl`` / ``run_name`` / ``sweep_id`` are its
bookkeeping. Standalone runs fall back to ``results/train_runs.jsonl``.
"""

import os

import torch
from transformers import AutoTokenizer, TrainingArguments

from ...models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from ...utils import GraphCollatorV2, set_wandb_project
from ...utils.text_graph_trainer_v2 import make_compute_metrics, shift_logits_for_metrics
from ...train import select_active_params, print_trainable_parameters, get_device
from .load_data import load_data
from .evaluate import KGQAGraphTrainer
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "kgqa"

# Fallback results path for standalone runs (no --runs-jsonl from the sweep runner).
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")

# Metric substrings worth printing from the final evaluate() dicts.
_PRINT_KEYS = ("f1", "hits1", "hit_star", "loss", "accuracy")


def _run_identity(cfg, run_name, sweep_id):
    """Internal (checkpoint + wandb) run name — unique per run.

    Under a sweep, derive it from the runner's bookkeeping (``sweep_id`` +
    ``run_name``). Standalone, fall back to the old descriptive pattern PLUS the
    seed, so two runs differing only in seed can't collide on the checkpoint dir
    or the wandb run name.
    """
    if sweep_id and run_name:
        return f"{sweep_id}_{run_name}"
    return f"kgqa_lora{cfg.lora_r}_khop{cfg.k_hop}_{cfg.graph_attn_impl}_s{cfg.seed}"


def result_block(cfg, dev, test):
    """The per-dataset result keys of a run record (E3.4, TODO_cwq).

    Every metric is dataset-prefixed (``eval_webqsp_f1``, ``test_cwq_hits1``,
    …) — also for single-dataset runs. Records written before 2026-07-12 carry
    the legacy unprefixed names, which read as ``X == webqsp_X`` (all history
    to that date is WebQSP-only). ``eval_sel_f1`` is the checkpoint-selection
    scalar (mean of per-dataset dev F1s, or the pinned ``selection_dataset``).
    """
    metrics = ("f1", "hits1", "hit_star", "em_accuracy")
    if cfg.log_strict_metrics:
        metrics += ("f1_strict",)
    block = {"eval_sel_f1": dev.get("eval_sel_f1")}
    for ds in cfg.eval_datasets:
        for m in metrics:
            block[f"eval_{ds}_{m}"] = dev.get(f"eval_{ds}_{m}")
            block[f"test_{ds}_{m}"] = test.get(f"test_{ds}_{m}")
    return block


def _save_train_record(cfg, run_name, dev_metrics, test_metrics,
                       runs_jsonl, sweep_meta=None):
    """Append this run's hyperparameters + dev/test results to its sweep's JSONL."""
    dev = dev_metrics or {}
    test = test_metrics or {}
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # ── dataset roles ──
        "train_datasets": list(cfg.train_datasets),
        "eval_datasets": list(cfg.eval_datasets),
        "selection_dataset": cfg.selection_dataset,
        # ── hyperparameters ──
        "model_name": cfg.model_name, "prompt_style": cfg.resolved_prompt_style,
        "lora_r": cfg.lora_r, "k_hop": cfg.k_hop, "k_hop_directed": cfg.k_hop_directed,
        "graph_attn_impl": cfg.graph_attn_impl, "dtype": cfg.dtype,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "max_steps": cfg.max_steps, "seed": cfg.seed,
        "boundary_loss_weight": cfg.boundary_loss_weight,
        "bias_weight_decay": cfg.bias_weight_decay, "lora_dropout": cfg.lora_dropout,
        # data keys (dataset-resolvable knobs may be per-dataset mappings)
        "rel_mode": cfg.rel_mode, "max_nodes": cfg.max_nodes, "n_max": cfg.n_max,
        "versions": cfg.versions, "magnetic_m": cfg.magnetic_m, "data_seed": cfg.data_seed,
        "data_format_version": cfg.data_format_version,
        "cvt_collapse": cfg.resolved_cvt_collapse("graph"),
        # ── results: best-checkpoint dev + test metrics, dataset-prefixed ──
        **result_block(cfg, dev, test),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    """Train this ONE config, then score + log the best checkpoint on dev + test."""
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name

    internal_run = _run_identity(cfg, run_name, sweep_id)
    output_dir = f"./checkpoints/{EXPERIMENT_NAME}/{internal_run}"
    report_to = "wandb" if cfg.wandb_project else "none"
    # Set the wandb project BEFORE building the trainer (its WandbCallback inits
    # lazily; setting it after works only by accident of that lazy init).
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    device = get_device()
    dtype = cfg.torch_dtype

    # Pin the intra-op thread count BEFORE anything compiles: the DataLoader's
    # pin-memory thread calls torch.set_num_threads(1) (process-global) when the
    # first loader starts, and dynamo guards compiled flex kernels on that global
    # state — the flip silently doubles the recompile count for every shape.
    torch.set_num_threads(1)

    # ── Model (frozen backbone + graph bias), new GTLM stack ──────────────────
    config = GTLMLlamaConfig.from_pretrained(
        cfg.model_name, **cfg.bias_params(),
        k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed, graph_attn_impl=cfg.graph_attn_impl,
        # ~14 (L, N) buckets/batch-size family; leave generous headroom so a long
        # run can never exhaust the dynamo cache mid-training (uncompiled flex has
        # no working backward, so exhaustion is fatal, not slow).
        flex_cache_size_limit=128,
    )
    model = GTLMLlamaForCausalLM.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=cfg.graph_attn_impl, torch_dtype=dtype)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Style-dependent anchor: "Answer:" (plain) | assistant header (chat, D3).
    question_end = tokenizer(cfg.question_end_str, add_special_tokens=False)["input_ids"]

    print("Loading data...")
    train_dataset, eval_sets, test_sets = load_data(cfg)
    print(f"Train: {len(train_dataset)} ({','.join(cfg.train_datasets)})  "
          f"Eval: { {ds: len(d) for ds, d in eval_sets.items()} }  "
          f"Test: { {ds: len(d) for ds, d in test_sets.items()} }")

    # magnetic_m is the eigenvector count for the collator — a data/collator knob,
    # NOT the model's magnetic_dim (the old code aliased the two; they only agreed
    # by coincidence).
    magnetic_m = cfg.magnetic_m if (cfg.magnetic or cfg.magnetic_shared) else 0
    train_collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=magnetic_m, pad_to_block=(cfg.graph_attn_impl == "flex"))
    # Generation runs the dense decode path; keep its collator unbucketed.
    gen_collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=magnetic_m, pad_to_block=False)

    active_params = list(cfg.active_params)
    model = select_active_params(model, active_params=active_params, lora=cfg.lora_config())
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
        # Same eval batch size, so the teacher-forced eval pass reuses the train
        # forwards' compiled flex kernels instead of minting a whole second
        # (B=8, L, N) shape family.
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
        # Generative selection metric (see PerDatasetEvalMixin): the mean of the
        # per-dataset dev F1s, or the selection_dataset one — eval_{ds}_f1 are
        # logged alongside either way.
        metric_for_best_model="eval_sel_f1",
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

    trainer = KGQAGraphTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_sets,           # {name: dataset}
        data_collator=train_collator,
        compute_metrics=make_compute_metrics(include_f1=False),       # cheap teacher-forced EM (secondary)
        preprocess_logits_for_metrics=shift_logits_for_metrics,       # bound eval memory
        active_params=active_params, bias_lr=cfg.bias_lr,
        bias_weight_decay=cfg.bias_weight_decay,
        selection_dataset=cfg.selection_dataset,
        log_strict_metrics=cfg.log_strict_metrics,
        # generative (primary) eval — capped during training (checkpoint selection),
        # switched to gen_max_samples (usually the full split) for final scoring below:
        gen_tokenizer=tokenizer, gen_collator=gen_collator, question_end=question_end,
        gen_max_new_tokens=cfg.gen_max_new_tokens,
        gen_max_samples=(cfg.gen_eval_samples if cfg.gen_eval_samples is not None
                         else cfg.gen_max_samples),
        gen_answer_sep=cfg.answer_parse_sep,
        # D4 lever (1.0 = off): re-weight the "\n" answer-boundary token's loss.
        boundary_loss_weight=cfg.boundary_loss_weight,
        boundary_token_id=(tokenizer("\n", add_special_tokens=False)["input_ids"][0]
                           if cfg.boundary_loss_weight != 1.0 else None),
    )

    trainer.train()

    # Best model is loaded (load_best_model_at_end); score it on dev + test so the
    # run record carries both — per dataset, full splits. Both evaluate() calls
    # run the generative set eval.
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

    _save_train_record(cfg, internal_run, dev_metrics, test_metrics,
                       runs_jsonl=runs_jsonl, sweep_meta=sweep_meta)

    if report_to == "wandb":
        import wandb
        if wandb.run is not None:
            wandb.finish()
