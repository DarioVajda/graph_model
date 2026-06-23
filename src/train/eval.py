import os
import json

import numpy as np
import torch
from transformers import TrainingArguments

from ..utils import GraphTrainer, GraphCollator
from ..models.legacy.modeling_gtlm_llama_v1 import KHopLlamaConfig, KHopGraphLlamaForCausalLM
from ..models.io import load_bias_parameters


def make_compute_metrics(include_f1=False):
    """Return a compute_metrics function for use with GraphTrainer."""
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        y_true, y_pred = [], []

        for i in range(len(labels)):
            valid = labels[i] != -100
            if not np.any(valid):
                continue
            y_true.append("-".join(map(str, labels[i][valid].tolist())))
            y_pred.append("-".join(map(str, preds[i][valid].tolist())))

        accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        result = {"em_accuracy": accuracy}

        if include_f1 and y_true:
            unique_classes = list(set(y_true))
            majority_class = max(set(y_true), key=y_true.count)
            tp = {cl: 0 for cl in unique_classes}
            fp = {cl: 0 for cl in unique_classes}
            fn = {cl: 0 for cl in unique_classes}
            for true, pred in zip(y_true, y_pred):
                if true == pred:
                    tp[true] += 1
                else:
                    fp[pred if pred in unique_classes else majority_class] += 1
                    fn[true] += 1
            f1_scores = {
                cl: (2 * tp[cl]) / (2 * tp[cl] + fp[cl] + fn[cl])
                if (2 * tp[cl] + fp[cl] + fn[cl]) > 0 else 0.0
                for cl in unique_classes
            }
            result["em_f1"] = float(np.mean(list(f1_scores.values())))

        return result

    return compute_metrics


def _filter_results(raw_results):
    """Keep only EM accuracy and F1 keys, dropping HF runtime noise."""
    return {k: v for k, v in raw_results.items() if "em_accuracy" in k or "em_f1" in k}


def _load_model_from_checkpoint(checkpoint_path, device):
    """
    Reconstruct model + config from a checkpoint saved by GraphTrainer.
    Handles both plain fine-tuned and LoRA checkpoints automatically.
    """
    from peft import PeftModel

    graph_bias_config_path = os.path.join(checkpoint_path, "graph_bias_config.json")
    with open(graph_bias_config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    # config.json in the checkpoint contains bias_params + k_hop
    config = KHopLlamaConfig.from_pretrained(checkpoint_path)

    model = KHopGraphLlamaForCausalLM.from_pretrained(
        base_model_name, config=config, attn_implementation="eager"
    )

    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, checkpoint_path)

    # Load bias parameters after any PeftModel wrapping so that parameter
    # names match: PeftModel prefixes them with 'base_model.model.' and the
    # saved state dict was recorded from the wrapped model.
    load_bias_parameters(model, checkpoint_path)

    model.to(device)
    return model, config


def evaluate_checkpoint(checkpoint_path, test_dataset, experiment_dir, include_f1=False):
    """
    Load a checkpoint, evaluate on test_dataset, and append filtered results
    to experiment_dir/results.json.
    Returns the results dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_model_from_checkpoint(checkpoint_path, device)

    collator = GraphCollator(k_hop=config.k_hop)

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_eval_batch_size=4,
        report_to="none",
    )

    trainer = GraphTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        compute_metrics=make_compute_metrics(include_f1=include_f1),
        active_params=None,
    )

    raw_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    results = _filter_results(raw_results)

    os.makedirs(experiment_dir, exist_ok=True)
    results_path = os.path.join(experiment_dir, "results.json")
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            json.dump({}, f)
    with open(results_path) as f:
        all_results = json.load(f)

    run_key = f"{os.path.basename(checkpoint_path)}_test"
    all_results = {run_key: results, **all_results}
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50 + "\n")

    return results
