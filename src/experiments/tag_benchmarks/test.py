"""
Re-evaluate a saved checkpoint on a TAG-benchmark test split.

Usage:
    python3 -m src.experiments.tag_benchmarks.test \\
        --checkpoint-path ./checkpoints/tag_benchmarks/<run_name> \\
        --dataset cora --max-neighbors 60 --text-mapping target_abstract

Training already scores the best-val checkpoint on test and records it in the run's
JSONL line, so this is for re-scoring an existing checkpoint (a different test subsample,
a spot check) rather than part of the normal protocol.

Pass the same dataset and bias flags you trained with: the dataset flags select which
cache to load, and the bias flags must match the checkpoint's saved bias modules.

Built on the v2 stack directly. ``src/train/eval.py``'s ``evaluate_checkpoint`` is
v0/v1-only (it constructs ``KHopGraphLlamaForCausalLM`` and the ragged v0 collator), so
it cannot load a v2 checkpoint.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, TrainingArguments

from ...models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from ...models.io import load_bias_parameters
from ...utils import GraphCollatorV2, GraphTrainerV2
from ...utils.text_graph_trainer_v2 import make_compute_metrics, shift_logits_for_metrics
from ...train import get_device
from .config import RunConfig, DATASETS, TEXT_MAPPINGS, IMPLS, DTYPES
from .data import load_data


def parse_args():
    d = RunConfig()
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a TAG-benchmark test split.")
    p.add_argument("--checkpoint-path", required=True,
                   help="Checkpoint directory, e.g. ./checkpoints/tag_benchmarks/<run_name>")

    # ── which cache to load (must match what the checkpoint was trained on) ───
    p.add_argument("--dataset", choices=DATASETS, default=d.dataset)
    p.add_argument("--hops", type=int, default=d.hops)
    p.add_argument("--max-neighbors", type=int, default=d.max_neighbors)
    p.add_argument("--text-mapping", choices=TEXT_MAPPINGS, default=d.text_mapping)
    p.add_argument("--text-mapping-param", type=float, default=d.text_mapping_param)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--test-subsample", type=int, default=d.test_subsample)

    # ── bias flags (must match the checkpoint's saved modules) ────────────────
    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--max-rw-steps", type=int, default=d.max_rw_steps)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)
    p.add_argument("--k-hop", type=int, default=d.k_hop)
    p.add_argument("--k-hop-directed", action=B, default=d.k_hop_directed)

    p.add_argument("--impl", choices=IMPLS, default=d.impl)
    p.add_argument("--dtype", choices=DTYPES, default=d.dtype)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--include-f1", action=B, default=d.include_f1)
    return p.parse_args()


def main():
    args = parse_args()

    # The checkpoint records the backbone it was adapted from.
    config_path = os.path.join(args.checkpoint_path, "graph_bias_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"{config_path} not found — is {args.checkpoint_path} a checkpoint saved by "
            f"GraphTrainerV2?")
    with open(config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    cfg = RunConfig(
        model_name=base_model_name,
        dataset=args.dataset, hops=args.hops, max_neighbors=args.max_neighbors,
        text_mapping=args.text_mapping, text_mapping_param=args.text_mapping_param,
        max_length=args.max_length, test_subsample=args.test_subsample,
        spd=args.spd, max_spd=args.max_spd, rrwp=args.rrwp, max_rw_steps=args.max_rw_steps,
        magnetic=args.magnetic, magnetic_dim=args.magnetic_dim,
        magnetic_q=args.magnetic_q, magnetic_m=args.magnetic_m,
        k_hop=args.k_hop, k_hop_directed=args.k_hop_directed,
        impl=args.impl, dtype=args.dtype, batch_size=args.batch_size,
        include_f1=args.include_f1,
    ).validate()

    device = get_device()
    _, _, test_dataset = load_data(cfg)

    config = GTLMLlamaConfig.from_pretrained(
        cfg.model_name, **cfg.bias_params(),
        k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        graph_attn_impl=cfg.backend(),
    )
    model = GTLMLlamaForCausalLM.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
        torch_dtype=cfg.torch_dtype())

    # LoRA adapters (if any) then the graph-bias tensors, mirroring how the trainer
    # saved them: PEFT adapter files + a separate bias_parameters.pt.
    adapter = os.path.join(args.checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path)
        model = model.merge_and_unload()
    load_bias_parameters(model, args.checkpoint_path)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex"), max_spd=cfg.max_spd)

    trainer = GraphTrainerV2(
        model=model,
        args=TrainingArguments(
            output_dir="./eval_temp",
            per_device_eval_batch_size=cfg.batch_size,
            report_to="none", do_train=False, do_eval=True),
        eval_dataset=test_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(include_f1=cfg.include_f1),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
    )

    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(f"\n--- {cfg.dataset}/{cfg.mapping_name()} [{cfg.arm()}] "
          f"({len(test_dataset)} test graphs) ---")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
