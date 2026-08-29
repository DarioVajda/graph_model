"""``--mode eval``: re-score a trained checkpoint and write its per-example report.

Why this exists rather than re-training with the analysis wired in: the 004 sweep
already produced 18 trained checkpoints, and the question "does the graph arm fail
on wide molecules?" is a property of those exact models. Re-training to ask it
would cost ~20 GPU-hours and would answer the question about *different* models.
This reads the checkpoints we have, in minutes.

The load path is `GTLMLlamaForCausalLM.from_pretrained`, which restores the LoRA
adapters **and** `bias_parameters.pt` (`src/models/causal_lm.py`). That second half
is the one that was silently missing once before — `project-load-best-model-bias-bug`,
where an adapter-only reload understated every GTLM final metric. So this module
does not trust the reload: it recomputes test accuracy and compares it against the
value the training run recorded, and says loudly when they disagree. A geometry
analysis built on a half-loaded model would look completely plausible and be
entirely wrong.
"""

import json
import os

import torch
from transformers import AutoTokenizer, TrainingArguments

from ...models import GTLMLlamaForCausalLM
from ...utils import GraphTrainerV2, make_compute_metrics, shift_logits_for_metrics
from ..expressiveness.training.dispatch import build_collator
from .analysis import write_per_example_report
from .dataset import load_data
from ._io import append_jsonl


def _recorded_accuracy(runs_jsonl, run_name):
    """The `test_accuracy` the training run wrote, or None if it cannot be found."""
    if not runs_jsonl or not os.path.exists(runs_jsonl):
        return None
    hit = None
    for line in open(runs_jsonl):
        record = json.loads(line)
        if record.get("run_name") == run_name:
            hit = record.get("test_accuracy")
    return hit


def run_eval_mode(cfg, run_name=None, runs_jsonl=None, sweep_meta=None,
                  expect_accuracy=None):
    if not cfg.checkpoint:
        raise ValueError("--mode eval requires --checkpoint <path to a checkpoint dir>")
    if not os.path.isdir(cfg.checkpoint):
        raise ValueError(f"checkpoint {cfg.checkpoint!r} is not a directory")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = run_name or os.path.basename(cfg.checkpoint.rstrip("/"))

    _, _, test_dataset = load_data(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"[eval] loading {cfg.checkpoint}")
    model = GTLMLlamaForCausalLM.from_pretrained(
        cfg.checkpoint, graph_attn_impl=cfg.impl.split("-", 1)[1],
        torch_dtype=torch.bfloat16)
    model.to(device).eval()

    collator = build_collator(
        cfg.impl, tokenizer, pad_token_id, cfg.k_hop, cfg.k_hop_directed,
        magnetic_m=cfg.magnetic_m,
        len_buckets=cfg.len_buckets, node_buckets=cfg.node_buckets)

    trainer = GraphTrainerV2(
        model=model,
        args=TrainingArguments(
            output_dir=f"./checkpoints/_eval_{run_name}",
            per_device_eval_batch_size=cfg.batch_size,
            dataloader_num_workers=cfg.num_workers,
            report_to=[], bf16=True, seed=cfg.seed),
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
    )

    out_dir = os.path.join(os.path.dirname(runs_jsonl) or ".", "per_example") \
        if runs_jsonl else "per_example"
    os.makedirs(out_dir, exist_ok=True)
    summary = write_per_example_report(
        trainer, test_dataset, cfg, os.path.join(out_dir, f"{run_name}.jsonl"))

    # The reload check. `expect_accuracy` comes from the training record; a mismatch
    # means the checkpoint did not come back the way it went in, and every geometry
    # number above it is describing a model that never existed.
    expected = expect_accuracy if expect_accuracy is not None else \
        _recorded_accuracy(runs_jsonl, run_name)
    got = summary["per_example_accuracy"]
    if expected is None:
        print(f"[eval] WARNING: no recorded accuracy to check the reload against "
              f"(got {got:.4f}). Treat the geometry analysis as unverified.")
        summary["reload_verified"] = None
    else:
        drift = abs(got - expected)
        summary["reload_verified"] = bool(drift <= 0.005)
        summary["reload_expected_accuracy"] = expected
        summary["reload_drift"] = drift
        flag = "OK" if drift <= 0.005 else "MISMATCH"
        print(f"[eval] reload check {flag}: recorded={expected:.4f} "
              f"reloaded={got:.4f} drift={drift:.4f}")
        if drift > 0.005:
            print("[eval] The checkpoint did not reload to the model that produced "
                  "the recorded score (cf. project-load-best-model-bias-bug). Do "
                  "NOT read the geometry analysis from this run.")

    if runs_jsonl:
        append_jsonl(runs_jsonl, {
            "mode": "eval", **(sweep_meta or {}),
            "run_name": run_name, "task": cfg.task, "arm": cfg.arm,
            "bias": cfg.bias, "encoding": cfg.encoding, "seed": cfg.seed,
            "checkpoint": cfg.checkpoint, "max_spd": cfg.max_spd,
            **summary,
        })
    return summary
