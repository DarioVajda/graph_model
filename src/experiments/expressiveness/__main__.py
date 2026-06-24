"""
Expressiveness experiment — refactor-validation harness (entry point).

Runs the HARD connectivity task across the v0 and v2 GTLM implementations on a
single shared `.gtds` dataset (the `TextGraphDataset` is implementation-agnostic;
only the collator / forward contract / trainer differ), to confirm:

  (a) the refactored v2 model still learns the task — accuracy parity with v0;
  (b) it is faster / leaner at scale — throughput + peak memory + sparsity.

Selectable implementations (``IMPL``):
    v0-eager   GraphLlamaForCausalLM (legacy)  + GraphCollator    + GraphTrainer
    v2-eager   GTLMLlamaForCausalLM            + GraphCollatorV2   + GraphTrainerV2
    v2-flex        "             (pad_to_block) +     "            +     "

Modes (``MODE``):
    train  small graphs, train to convergence  -> accuracy / parity
           (each run also records live ms/step + peak CUDA memory, and token /
            block sparsity is measured standalone on a sampled subset)
    bench  large graphs, few-step throughput + peak CUDA memory + sparsity

This file is a thin dispatcher; the implementation lives in the sibling modules
(config / dispatch / evaluation / datasets / instrumentation / train / bench).
"""

import os

from transformers import AutoTokenizer

from ...utils import set_wandb_project
from .config import RunConfig
from .train import run_train_mode
from .bench import run_bench_mode


def _env_seeds(default):
    raw = os.environ.get("SEEDS")
    if not raw:
        return tuple(default)
    return tuple(int(s) for s in raw.split(",") if s.strip())


if __name__ == "__main__":
    cfg = RunConfig(
        mode=os.environ.get("MODE", "train"),                 # "train" | "bench"
        impls=("v0-eager", "v2-eager"),                       # eager-only: v0↔v2 equivalence + k-hop effect
        k_hops=(0, 1),
        report_to=os.environ.get("REPORT_TO", "none"),        # "none" | "wandb"
        seeds=_env_seeds((0, 1, 2)),
    )

    if cfg.report_to == "wandb":
        set_wandb_project("GraphLLM")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if cfg.mode == "train":
        run_train_mode(cfg, tokenizer, pad_token_id)
    elif cfg.mode == "bench":
        run_bench_mode(cfg)
    else:
        raise ValueError(f"Unknown MODE='{cfg.mode}' (expected 'train' or 'bench').")
