"""
Train ONE molecules configuration and log ONE record.

Structurally identical to `probes/train.py` — same model/collator construction,
same instrumentation, same best-val-checkpoint protocol — so numbers are directly
comparable across the two experiments. The only molecule-specific parts are the
dataset and the extra record fields (`arm`, `encoding`, `stereo_tags`).

Speed and memory are first-class results here, not diagnostics: M2's stated job
(PLAN.md §8) is to settle flex-vs-eager at N ~ 52 and to replace the estimated
s/it with a measured one.
"""

import glob
import os

import torch
from transformers import TrainingArguments, set_seed

from ...utils import GraphTrainerV2, make_compute_metrics, shift_logits_for_metrics
from ..expressiveness.training.dispatch import (
    build_collator, build_model, select_active_params, print_trainable_parameters,
)
from ..expressiveness.training.instrumentation import StepMemCallback, measure_density
from .config import EXPERIMENT_NAME
from .analysis import write_per_example_report
from .dataset import load_data, load_dataset_stats
from .evaluate import answer_token_ids, make_margin_metrics, make_margin_preprocessor
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")

ACTIVE_PARAMS = ["graph_bias"]

#: Tier A is scored by exact match on the answer token (probes' metric). Tier B is
#: scored by ROC-AUC from the logit margin, which is the quantity the published
#: baselines report — selecting on exact match there would optimise a different
#: thing than the number we quote (one of the three protocol defects already
#: recorded in this project: `CLAUDE_CONTEXT.md` §4.1).
TIER_METRIC = {"A": "eval_em_accuracy", "B": "eval_roc_auc"}


def _scoring(cfg, tokenizer):
    """``(metric_name, compute_metrics, preprocess_logits)`` for this run's tier."""
    if cfg.tier() == "A":
        return TIER_METRIC["A"], make_compute_metrics(), shift_logits_for_metrics
    yes_id, no_id = answer_token_ids(tokenizer)
    return (TIER_METRIC["B"], make_margin_metrics(yes_id),
            make_margin_preprocessor(yes_id, no_id))


def _save_train_record(cfg, run_name, results, runs_jsonl, sweep_meta=None):
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        # ── the experiment's axes ──
        "task": cfg.task, "arm": cfg.arm, "encoding": cfg.encoding,
        "stereo_tags": cfg.stereo_tags, "bias": cfg.bias, "seed": cfg.seed,
        "question_node": cfg.question_node,
        # ── hyperparameters ──
        "model_name": cfg.model_name, "impl": cfg.impl,
        "k_hop": cfg.k_hop, "k_hop_directed": cfg.k_hop_directed,
        "pool": list(cfg.pool),
        "train_size": cfg.train_size, "val_size": cfg.val_size,
        "test_size": cfg.test_size, "data_seed": cfg.data_seed,
        "ordering": cfg.ordering,
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


def _eval_curve(trainer, metric):
    """The full eval trajectory, so convergence is a recorded fact not a log scrape.

    Without this, "did the arm converge?" can only be answered by grepping slurm
    logs — which is how it was answered the first time, and the logs interleave
    with tqdm's carriage returns badly enough that `grep` silently returns nothing
    unless you remember `-a`. The trajectory is ~12 floats; store it.
    """
    curve = []
    for entry in trainer.state.log_history:
        if metric in entry:
            curve.append({"step": entry.get("step"), "epoch": entry.get("epoch"),
                          metric: entry[metric], "eval_loss": entry.get("eval_loss")})
    return curve


def _convergence(curve, metric):
    """Did the metric stop improving, or did the LR schedule simply run out?

    `feedback-dont-call-floors-early` applies to ceilings too: an arm still
    climbing at the final eval has not been measured, it has been interrupted, and
    its score is a lower bound rather than its ceiling. `tail_gain` is the
    improvement over the last three evals; a run whose curve is still rising there
    ended because cosine annealed the LR to min, not because it converged.

    **`curve` MUST be the training trajectory only.** Read it straight off
    `trainer.state.log_history` at the end of a run and it is not: `trainer.train()`
    restores the best checkpoint (`load_best_model_at_end`), and the `trainer.evaluate`
    call that follows re-scores THAT model on val under the same `eval_` prefix, so
    the curve gains a final point equal by construction to its own maximum. Then
    `tail_gain = max - values[-4] >= 0` always, and the flag fires exactly when a run
    has fallen furthest from its peak — i.e. it reports **overfitting as
    under-training**. That is not hypothetical: it is what M3a's 16 records did (14
    flagged `still_improving`, 2 actually were), and it is why `train_and_eval`
    snapshots the curve before evaluating. `test_the_flag_inverts_on_a_contaminated_curve`
    pins the failure so it cannot come back silently.

    `peak_fraction` is the more direct reading and needs no threshold: a run whose
    best eval sits at 0.2 of the way through has budget to spare, one that peaks at
    1.0 was interrupted.
    """
    if len(curve) < 4:
        return {"tail_gain": None, "still_improving": None,
                "best_eval_index": None, "peak_fraction": None}
    values = [c[metric] for c in curve]
    tail_gain = values[-1] - values[-4]
    best = max(range(len(values)), key=values.__getitem__)
    return {
        "tail_gain": tail_gain,
        # 1pp over three evals is inside eval noise at these split sizes; above it,
        # the run wants more budget before its number means anything.
        "still_improving": bool(tail_gain > 0.01),
        "best_eval_index": best,
        "peak_fraction": best / (len(values) - 1),
    }


def _answer_stats(stats):
    """Majority-class rate of the answers — the floor any score beats.

    Admission criterion 3 in PLAN.md §3.2.4 ("well above the majority-class rate")
    was unverifiable from the run record until this was added: `fg_count` has a
    0.760 base rate, so an arm reporting 0.74 has learned *nothing* and looks like
    a respectable score next to a task whose base rate is 0.285.

    **Which split's floor.** The headline is a TEST number, so the floor it has to
    beat is the TEST split's majority rate, not the corpus-wide one. On Tier B the
    two differ sharply — the scaffold split moves BBBP from 0.822 positive in train
    to 0.524 in test — and quoting 0.765 against a test accuracy compares a score to
    a floor from a different distribution. Tier A now splits by molecule and scaffold
    too, so the same applies there.

    `answers_by_split["test"]` therefore wins whenever it is present, and
    `base_rate_source` records which was used rather than leaving a reader to guess.
    The corpus-wide fallback covers artifacts built before either split existed.

    **`degenerate_test_split`** is the case where the floor is the ceiling: every test
    example carries the same answer, so a constant predictor scores 1.000 and the two
    arms cannot be compared at all. `n_classes` has always exposed this, but only to
    someone who thought to look — and a family in that state reports a *perfect*
    accuracy, which is the last number anyone audits. Measured on `014`:
    `stereo_assigned` lands 1000/1000 test examples on one answer once the pool
    includes HIV, because scaffold-splitting sends the rarest scaffolds to test and
    assigned stereocentres are not among them. Recorded as a boolean so a sweep report
    can void the family instead of printing a win.
    """
    stats = stats or {}
    by_split = stats.get("answers_by_split") or {}
    answers = by_split.get("test") or stats.get("answers") or {}
    source = "test_split" if by_split.get("test") else "all_examples"
    total = sum(answers.values())
    if not total:
        return {"base_rate": None, "answer_distribution": None, "n_classes": 0,
                "base_rate_source": None, "degenerate_test_split": None}
    return {"base_rate": max(answers.values()) / total,
            "answer_distribution": dict(sorted(answers.items(), key=lambda kv: -kv[1])),
            "n_classes": len(answers),
            "base_rate_source": source,
            "degenerate_test_split": len(answers) < 2}


def _per_example(trainer, test_dataset, cfg, run_name, runs_jsonl, yes_id=None):
    """Write the per-example error/geometry report next to the sweep's records.

    Wrapped in a try/except on purpose: this is *analysis*, and a failure here
    must not lose a training run that already cost GPU-hours. A missing report is
    visible in the record as a null; a crashed job at the last line is not.

    That safety net also hid a real defect for the whole campaign: the report had
    never once succeeded on Tier B (0/18 on `001`, 0/3 on `019`, 0/8 on `021`,
    against 20/20 on Tier A's `014`), because it read the margin readout with the
    exact-match unpacking and raised `IndexError` every time. The null looked like
    an absent nicety rather than an untested instrument. `yes_id` is what Tier B
    needs to unpack its own predictions; passing it is what fixed that.
    """
    try:
        out_dir = os.path.join(os.path.dirname(runs_jsonl) or ".", "per_example")
        os.makedirs(out_dir, exist_ok=True)
        return write_per_example_report(
            trainer, test_dataset, cfg, os.path.join(out_dir, f"{run_name}.jsonl"),
            yes_id=yes_id)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[analysis] per-example report failed ({type(exc).__name__}: {exc}); "
              "the training result above is unaffected.")
        return {"per_example_path": None}


def _score_last_checkpoint(trainer, test_dataset, best_step):
    """Test score of the FINAL checkpoint, recorded beside the val-selected one.

    `load_best_model_at_end` reports whichever checkpoint scored best on *val*,
    and PLAN.md §8.3 measured that val is a different **population** from test on
    every Tier-B set — different chemical series, drawn from a disjoint slice of
    the source CSV, and on BBBP saturating by epoch 2. The headline therefore
    rests on a selection signal shown to be unreliable, and nothing in the record
    says what that selection is worth. This turns it into a number: one extra
    evaluation pass, no extra training, no protocol change to what is reported.

    `save_total_limit=2` with `load_best_model_at_end=True` keeps the best
    checkpoint *and* the most recent one, so both are on disk. When the best IS
    the most recent there is only one, `last_is_best` says so, and the last score
    is left null rather than restated — a duplicated number would otherwise look
    like an independent measurement in any table that averages these.

    MUST be called after every field derived from the best model has been
    captured: it swaps the weights in place and does not put them back.
    """
    out = {"test_roc_auc_last": None, "test_accuracy_last": None,
           "last_checkpoint_step": None, "best_checkpoint_step": best_step,
           "last_is_best": None}
    try:
        found = glob.glob(os.path.join(trainer.args.output_dir, "checkpoint-*"))
        steps = sorted(int(os.path.basename(p).rsplit("-", 1)[1]) for p in found)
        if not steps:
            print("[last-ckpt] no checkpoint directories on disk; skipping")
            return out
        last = steps[-1]
        out["last_checkpoint_step"] = last
        out["last_is_best"] = (last == best_step)
        if last == best_step:
            print(f"[last-ckpt] the best checkpoint IS the last ({last}); "
                  "scores coincide, leaving test_roc_auc_last null")
            return out
        print(f"[last-ckpt] scoring checkpoint-{last} (best was {best_step}) ...")
        trainer._load_from_checkpoint(
            os.path.join(trainer.args.output_dir, f"checkpoint-{last}"))
        m = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        out["test_roc_auc_last"] = m.get("test_roc_auc")
        out["test_accuracy_last"] = m.get("test_em_accuracy")
    except Exception as exc:                                  # noqa: BLE001
        # Same contract as `_per_example`: this is measurement *about* the run,
        # and it must never lose a run that already cost GPU-hours.
        print(f"[last-ckpt] failed ({type(exc).__name__}: {exc}); "
              "the training result above is unaffected.")
    return out


def _bias_init_fingerprint(model):
    """L2 norm of the graph-bias parameters, recorded before and after training.

    `feedback-verify-nulls-are-real`: a null result is only reportable once the
    bias modules are shown to have left their initialisation. Recording both ends
    makes that checkable from `runs.jsonl` alone, with no rerun.
    """
    total = 0.0
    for name, param in model.named_parameters():
        if "graph_bias" in name:
            total += float(param.detach().float().pow(2).sum())
    return total ** 0.5


def run_train_mode(cfg, tokenizer, pad_token_id, runs_jsonl=None, run_name=None, sweep_id=None):
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run_name = f"{sweep_id}_{run_name}" if (sweep_id and run_name) else cfg.run_name()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    metric, compute_metrics, preprocess_logits = _scoring(cfg, tokenizer)
    # The per-example report unpacks Tier B's margin readout itself, so it needs the
    # same `yes` id the metric uses. Resolved once, here, rather than re-derived in
    # the analysis path where it could drift from what the scorer actually used.
    yes_id = answer_token_ids(tokenizer)[0] if cfg.tier() == "B" else None

    train_dataset, val_dataset, test_dataset = load_data(cfg)
    for name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        print(f"[data] {name}: {len(ds)} examples")
    print(f"[scoring] tier {cfg.tier()} -> selecting on {metric}")

    density = None
    if cfg.measure_density and cfg.arm == "graph":
        print(f"[density] measuring token/block sparsity at k_hop={cfg.k_hop} ...")
        density = measure_density(
            val_dataset, tokenizer, pad_token_id, cfg.magnetic_m, cfg.k_hop,
            cfg.k_hop_directed, batch_size=cfg.batch_size,
            num_sample_graphs=cfg.density_sample_graphs,
            num_sample_batches=cfg.density_sample_batches, device=device)

    set_seed(cfg.seed)
    model, _ = build_model(cfg.impl, cfg.model_name, cfg.model_bias_config(),
                           cfg.k_hop, cfg.k_hop_directed, device, cfg.flex_compile_mode)
    model = select_active_params(model, active_params=ACTIVE_PARAMS, lora=cfg.lora_config())
    print_trainable_parameters(model)
    bias_norm_init = _bias_init_fingerprint(model)

    collator = build_collator(
        cfg.impl, tokenizer, pad_token_id, cfg.k_hop, cfg.k_hop_directed,
        magnetic_m=cfg.magnetic_m,
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
        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.eval_steps,
        metric_for_best_model=metric, greater_is_better=True,
        save_total_limit=2, load_best_model_at_end=True,
    )

    step_mem = StepMemCallback()
    trainer = GraphTrainerV2(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        active_params=ACTIVE_PARAMS,
        bias_lr=cfg.bias_lr,
        callbacks=[step_mem],
    )

    train_output = trainer.train()
    # SNAPSHOT THE TRAJECTORY HERE, before the two `evaluate` calls below append to
    # `log_history`. `load_best_model_at_end` has already restored the best
    # checkpoint by this point, so the next line re-scores the BEST model on val and
    # logs it under the same `eval_` prefix. Reading the curve afterwards therefore
    # appended a final point equal, by construction, to the maximum — which inverted
    # `still_improving` (see `_convergence`).
    training_curve = _eval_curve(trainer, metric)
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    perf = step_mem.summary()

    # ORDER MATTERS BELOW. Everything that describes the BEST model is captured
    # first, because `_score_last_checkpoint` loads the final checkpoint into the
    # live model in place and does not restore it. Inlining any of these into the
    # results literal underneath would silently re-attribute them to the last
    # checkpoint instead — `bias_norm_final` and the per-example report both read
    # the model, not the metrics.
    bias_norm_final = _bias_init_fingerprint(model)
    per_example = _per_example(trainer, test_dataset, cfg, internal_run_name, runs_jsonl,
                               yes_id=yes_id)
    best_ckpt = trainer.state.best_model_checkpoint
    last_checkpoint = _score_last_checkpoint(
        trainer, test_dataset,
        int(best_ckpt.rsplit("-", 1)[1]) if best_ckpt else None)

    results = {
        # Tier A's headline is exact match; Tier B's is ROC-AUC. Both are written
        # so `sweep.report` can aggregate either without a per-tier special case;
        # the one that does not apply is simply absent.
        "test_accuracy": test_metrics.get("test_em_accuracy"),
        "test_roc_auc": test_metrics.get("test_roc_auc"),
        "test_average_precision": test_metrics.get("test_average_precision"),
        "test_f1": test_metrics.get("test_f1"),
        # Tie-collapse watch (see evaluate.py trap 2) — recorded from the first
        # run rather than retrofitted after a suspicious AUROC.
        "test_n_distinct": test_metrics.get("test_n_distinct"),
        "test_tied_pair_fraction": test_metrics.get("test_tied_pair_fraction"),
        "test_pos_rate": test_metrics.get("test_pos_rate"),
        "selection_metric": metric,
        "best_val_score": val_metrics.get(metric),
        "train_runtime_s": train_output.metrics.get("train_runtime"),
        "train_steps_per_second": train_output.metrics.get("train_steps_per_second"),
        "step_ms_mean": perf.get("step_ms_mean"),
        "step_ms_median": perf.get("step_ms_median"),
        "peak_gb": perf.get("peak_gb"),
        "n_steps": perf.get("n_steps"),
        "token_sparsity": density.get("token_sparsity_mean") if density else None,
        "block_sparsity": density.get("block_sparsity_mean") if density else None,
        # Null-gate evidence (see `_bias_init_fingerprint`).
        "bias_norm_init": bias_norm_init,
        "bias_norm_final": bias_norm_final,
        # Is the headline a ceiling or an interruption? See `_convergence`.
        "eval_curve": training_curve,
        **_convergence(training_curve, metric),
        # What any score has to beat before it means anything. See `_answer_stats`.
        **_answer_stats(load_dataset_stats(cfg)),
        # Is a mistake explained by the molecule's width? See `analysis.py`.
        **per_example,
        # What is checkpoint selection worth, given §8.3? See `_score_last_checkpoint`.
        **last_checkpoint,
    }
    headline = (results["test_accuracy"] if cfg.tier() == "A" else results["test_roc_auc"])
    last = results["test_accuracy_last"] if cfg.tier() == "A" else results["test_roc_auc_last"]
    print(f"[results] tier {cfg.tier()} headline={headline} "
          f"last-ckpt={last if last is not None else 'same as best'} "
          f"(best-val {metric}={results['best_val_score']}) "
          f"runtime={results['train_runtime_s']}s peak={results['peak_gb']}GB "
          f"bias_norm {results['bias_norm_init']:.4g} -> {results['bias_norm_final']:.4g}")
    if results.get("degenerate_test_split"):
        print(f"[results] *** DEGENERATE TEST SPLIT: all {sum(results['answer_distribution'].values())} "
              f"test examples answer {list(results['answer_distribution'])[0]!r}. A constant "
              f"predictor scores 1.000 here, so this family CANNOT compare two arms. "
              f"Void it in the report rather than quoting the accuracy. ***")
    base = results.get("base_rate")
    print(f"[results] base_rate={'?' if base is None else format(base, '.3f')} "
          f"tail_gain={results.get('tail_gain')} "
          f"still_improving={results.get('still_improving')}"
          + ("  <-- BUDGET-LIMITED: this score is a lower bound, not a ceiling"
             if results.get("still_improving") else ""))

    _save_train_record(cfg, internal_run_name, results, runs_jsonl=runs_jsonl,
                       sweep_meta=sweep_meta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
