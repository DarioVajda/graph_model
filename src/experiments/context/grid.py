"""
Phase 3 — score a trained checkpoint over the whole (N, T) grid.

One forward per graph per cell, teacher-forced (see ``evaluate.grid_eval``); one
JSON line per cell into ``grid.jsonl``, which ``analysis/plot.py`` turns into the
heatmap and the markdown table.

Cells are scored largest-first. That is deliberate: if the top-right cell is at
ceiling the experiment has no contour to find and the grid should be extended
rather than completed (README §A.12), and this way you learn that in the first few
minutes instead of the last.
"""

import os

from ...train import get_device

from .evaluate import grid_eval, wilson_interval
from .model import build_collator, load_checkpoint_model
from .process_dataset import cell_split_name, load_split
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_GRID_JSONL = os.path.join(EXPERIMENT_DIR, "results", "grid.jsonl")


def run_grid_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    """Evaluate ``cfg.checkpoint_path`` on every (N, T, k) condition; one record each.

    Under a k mixture the grid is a 3-axis product, so a cell contributes one record
    per k rather than one in total. ``hops`` is always written to the record, so a
    single-k grid and a mixture grid are readable by the same analysis code.
    """
    grid_jsonl = runs_jsonl or _DEFAULT_GRID_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name

    import torch
    torch.set_num_threads(1)

    device = get_device()
    model, tokenizer, _ = load_checkpoint_model(cfg.checkpoint_path, cfg, device)
    collator = build_collator(cfg, tokenizer, for_grid=True)

    train_cells = set(cfg.train_cells())
    # Largest first (see module docstring).
    cells = sorted(cfg.selected_cells(), key=lambda c: cfg.cell_length(*c), reverse=True)

    mixed = bool(cfg.hop_counts)
    records = []
    for (n, t) in cells:
      for k in cfg.selected_hops():
        dataset = load_split(cfg, cell_split_name(n, t, k if mixed else None))
        metrics = grid_eval(model, dataset, collator, tokenizer, device=device,
                            batch_size=1, verbose=True)
        lo, hi = wilson_interval(round(metrics["em"] * metrics["n"]), metrics["n"])
        record = {
            "mode": "grid",
            **sweep_meta,
            "checkpoint_path": cfg.checkpoint_path,
            "n_nodes": n, "tokens_per_node": t, "hops": k,
            "expected_len": cfg.cell_length(n, t),
            "in_train_distribution": (n, t) in train_cells,
            "max_train_len": cfg.max_train_len,
            "em_ci_low": lo, "em_ci_high": hi,
            "seed": cfg.seed, "data_seed": cfg.data_seed,
            **metrics,
        }
        append_jsonl(grid_jsonl, record)
        records.append(record)
        print(f"[grid] N={n:4d} T={t:4d} k={k}  L={metrics['packed_len']:6d}  "
              f"EM={metrics['em']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"distractor={metrics['distractor_rate']:.3f}  "
              f"malformed={metrics['malformed_rate']:.3f}"
              f"{'' if (n, t) in train_cells else '   (extrapolation)'}")

    print(f"\n[grid] wrote {len(records)} cell records to {grid_jsonl}")
    return records
