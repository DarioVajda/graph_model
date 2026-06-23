"""
Slim Trainer for the v2 GTLM-Llama model.

Because :class:`GraphCollatorV2` emits a standard packed batch with ``(B, L)``
``labels`` (``-100`` outside the prompt span), the model's ``forward`` returns a
loss via HuggingFace's own ``ForCausalLMLoss`` — so the heavy customisation of
the v0/v1 ``GraphTrainer`` (``compute_loss``, ``floating_point_ops``, the
prediction-step surgery) is no longer needed.

What remains genuinely custom and opt-in:
  * ``remove_unused_columns=False`` — the collator consumes raw dataset fields
    (``edges``, ``input_ids`` lists, …) that are not model-forward arguments, so
    HF must not strip them before collation.
  * a separate learning rate for the graph-bias parameters (``bias_lr``);
  * a lightweight checkpoint that stores only the base-model pointer + the
    trainable bias parameters (mirrors v1's format).

For evaluation, :func:`shift_logits_for_metrics` (as ``preprocess_logits_for_metrics``)
plus :func:`make_compute_metrics` reproduce the prompt-only exact-match scoring
without overriding ``prediction_step``.
"""

import os
import json

import numpy as np
import torch
from transformers import Trainer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

from ..models.io import save_bias_parameters


class GraphTrainerV2(Trainer):
    """HF Trainer with opt-in bias-LR groups and a lightweight bias checkpoint."""

    def __init__(self, *args, active_params=None, bias_lr=None, **kwargs):
        if kwargs.get("args") is not None:
            # The collator needs raw graph columns that aren't forward arguments.
            kwargs["args"].remove_unused_columns = False
        self.active_params = active_params
        self.bias_lr = bias_lr
        super().__init__(*args, **kwargs)

    # ── Optional: a distinct learning rate for the bias parameters ──────────────

    def create_optimizer(self):
        if self.bias_lr is None and not self.active_params:
            return super().create_optimizer()

        from transformers.utils import is_sagemaker_mp_enabled
        is_smp = is_sagemaker_mp_enabled()
        opt_model = self.model_wrapped if is_smp else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            if getattr(self, "optimizer_cls_and_kwargs", None) is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            groups = {"base_decay": [], "base_no_decay": [], "bias_decay": [], "bias_no_decay": []}
            active_p = self.active_params or []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                is_active = any(act in n for act in active_p)
                has_decay = n in decay_parameters
                key = ("bias" if is_active else "base") + ("_decay" if has_decay else "_no_decay")
                groups[key].append(p)

            base_lr = self.args.learning_rate
            bias_lr = self.bias_lr if self.bias_lr is not None else base_lr
            grouped = [
                {"params": groups["base_decay"], "weight_decay": self.args.weight_decay, "lr": base_lr, "is_bias": False},
                {"params": groups["base_no_decay"], "weight_decay": 0.0, "lr": base_lr, "is_bias": False},
                {"params": groups["bias_decay"], "weight_decay": self.args.weight_decay, "lr": bias_lr, "is_bias": True},
                {"params": groups["bias_no_decay"], "weight_decay": 0.0, "lr": bias_lr, "is_bias": True},
            ]
            grouped = [g for g in grouped if g["params"]]

            if "params" in optimizer_kwargs:
                grouped = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                grouped = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                grouped = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for module in opt_model.modules():
                    if isinstance(module, torch.nn.Embedding):
                        manager.register_module_override(module, "weight", {"optim_bits": 32})

        if is_smp:
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer

    def log(self, logs: dict, *args, **kwargs) -> None:
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get("is_bias", False):
                    logs["bias_learning_rate"] = group["lr"]
                    break
        super().log(logs, *args, **kwargs)

    # ── Optional: lightweight bias-only checkpoint ──────────────────────────────

    def save_model(self, output_dir=None, _internal_call=False):
        if self.active_params is None:
            return super().save_model(output_dir, _internal_call=_internal_call)

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if PeftModel is not None and isinstance(self.model, PeftModel):
            super().save_model(output_dir)

        base_model_name = self.model.config._name_or_path
        with open(os.path.join(output_dir, "graph_bias_config.json"), "w") as f:
            json.dump({"base_model_name_or_path": base_model_name}, f, indent=4)
        self.model.config.save_pretrained(output_dir)
        save_bias_parameters(self.model, output_dir, params=self.active_params)


# ── Evaluation helpers (prompt-only exact match, no prediction_step override) ───

def shift_logits_for_metrics(logits, labels):
    """``preprocess_logits_for_metrics``: turn logits into next-token predictions
    aligned to the label positions (logit at t predicts token t+1)."""
    preds = torch.argmax(logits, dim=-1)
    shifted = torch.full_like(preds, fill_value=-100)
    shifted[:, 1:] = preds[:, :-1]
    return shifted


def make_compute_metrics(include_f1: bool = False):
    """Exact-match (and optional macro-F1) over the prompt span (label != -100)."""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        y_true, y_pred = [], []
        for i in range(len(labels)):
            valid = labels[i] != -100
            if not np.any(valid):
                continue
            y_true.append("-".join(map(str, labels[i][valid].tolist())))
            y_pred.append("-".join(map(str, preds[i][valid].tolist())))

        accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
        result = {"em_accuracy": accuracy}

        if include_f1 and y_true:
            classes = list(set(y_true))
            majority = max(set(y_true), key=y_true.count)
            tp = {c: 0 for c in classes}; fp = {c: 0 for c in classes}; fn = {c: 0 for c in classes}
            for t, p in zip(y_true, y_pred):
                if t == p:
                    tp[t] += 1
                else:
                    fp[p if p in classes else majority] += 1
                    fn[t] += 1
            f1 = {c: (2 * tp[c]) / (2 * tp[c] + fp[c] + fn[c]) if (2 * tp[c] + fp[c] + fn[c]) > 0 else 0.0
                  for c in classes}
            result["em_f1"] = float(np.mean(list(f1.values())))
        return result
    return compute_metrics


def set_wandb_project(project_name: str = "GraphLLM"):
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_WATCH"] = "false"
