"""
T0 — cost calibration for the context grid (README §A.12).

Every compute and memory number in the plan is interpolated from
``src/models/flex_attn/results_h100/full_model.md``, which (a) predates
bias-checkpointing and decoder gradient checkpointing and (b) was measured on an
H100 while these runs request B200/B300 — and the flex README is explicit that
results directories are per-GPU and cross-directory ratios are meaningless. So
the schedule is not planned off that table; it is measured here first.

    python3 -m src.experiments.context.calibrate --out-dir <dir> [--cells train]

**This measures the real training step, not a synthetic proxy.** The same model,
LoRA config, collator, bucket ladder, windowed loss and optimizer the training
run uses, on graphs built by this experiment's own builder — timed end to end,
including the backward and the optimizer step. What comes out is **wall-clock
s/it at batch 1**, which is the number that actually sizes a job; a
forward+backward microbenchmark is not, because it cannot see the optimizer or
the accumulation.

(``src/models/flex_attn/bench_full_model.py`` was the obvious tool for this and
does not currently run: its monkeypatched attention forward still indexes the
per-batch context as a dict — ``ctx["features"]`` — but the model migrated to the
typed ``GraphContext`` dataclass, so every spec fails with ``TypeError:
'GraphContext' object is not subscriptable``. That is a pre-existing bug in the
benchmark harness, unrelated to this experiment, and it is left untouched here.)

The dataloader is deliberately NOT in the loop: batches are pre-built and reused
across iterations, so these numbers are a *floor* on s/it. Real runs add data
loading on top, which `--num-workers 4` is meant to hide.
"""

import argparse
import json
import os
import time

import torch

from ...train import get_device, print_trainable_parameters, select_active_params
from ...utils import TextGraphDataset
from .config import RunConfig
from .data import build_code_pool, build_id_pool, build_split_graphs, load_corpus
from .evaluate import windowed_forward, windowed_loss
from .model import build_collator, build_model
from .process_dataset import RAW_DATA_DIR, _finalize

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(EXPERIMENT_DIR, "results", "calibration")


def _cell_batch(cfg, tokenizer, corpus, code_pool, id_pool, n, t, collator, device):
    """Build one real batch for cell ``(n, t)`` and move it to the GPU."""
    graphs = build_split_graphs(
        cfg, tokenizer, corpus, code_pool, id_pool, split="calib",
        n_graphs=cfg.batch_size, cell=(n, t), verbose=False)
    ds = TextGraphDataset(graphs, dataset_label=f"calib_{n}x{t}")
    _finalize(ds, cfg, tokenizer)
    batch = collator([ds[i] for i in range(len(ds))])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _time_cell(model, optimizer, batch, n_warmup, n_iter):
    """Warm up, then time ``n_iter`` full training steps. Returns (s/it, peak GB)."""
    labels = batch.pop("labels")

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits, window_labels = windowed_forward(model, batch, labels)
        loss = windowed_loss(logits, window_labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(n_warmup):          # includes the one-time flex compile
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    batch["labels"] = labels
    return elapsed / n_iter, peak_gb


def run_calibration(cfg, out_dir=None, which="all", n_warmup=2, n_iter=5):
    """Time one training step per cell; write ``calibration.jsonl`` + ``.md``."""
    out_dir = out_dir or _DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "calibration.jsonl")

    torch.set_num_threads(1)
    device = get_device()
    model, tokenizer = build_model(cfg, device)
    model = select_active_params(model, active_params=list(cfg.active_params),
                                 lora=cfg.lora_config())
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    print_trainable_parameters(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    corpus = load_corpus(tokenizer, RAW_DATA_DIR, cfg.corpus_tokens, verbose=False)
    code_pool = build_code_pool(tokenizer, cfg.code_len, cfg.id_pool, seed=cfg.data_seed)
    id_pool = build_id_pool(cfg.id_pool)
    collator = build_collator(cfg, tokenizer, for_grid=True)

    train_cells = set(cfg.train_cells())
    cells = cfg.train_cells() if which == "train" else cfg.cells()
    # Smallest first: the cheap cells confirm the harness works before the
    # expensive ones spend an hour discovering it does not.
    cells = sorted(cells, key=lambda c: cfg.cell_length(*c))

    rows = []
    for (n, t) in cells:
        print(f"[calibrate] N={n} T={t} (expected L={cfg.cell_length(n, t)})", flush=True)
        row = {
            "n_nodes": n, "tokens_per_node": t,
            "expected_len": cfg.cell_length(n, t),
            "in_train_distribution": (n, t) in train_cells,
            "ok": False, "error": None, "error_detail": None,
            "s_per_it": None, "peak_gb": None, "packed_len": None,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "k_hop": cfg.k_hop, "magnetic_m": cfg.magnetic_m,
            "batch_size": cfg.batch_size, "lora_r": cfg.lora_r,
            "graph_attn_impl": cfg.graph_attn_impl, "dtype": cfg.dtype,
        }
        try:
            batch = _cell_batch(cfg, tokenizer, corpus, code_pool, id_pool, n, t,
                                collator, device)
            row["packed_len"] = int(batch["input_ids"].shape[1])
            s_it, peak = _time_cell(model, optimizer, batch, n_warmup, n_iter)
            row.update(ok=True, s_per_it=s_it, peak_gb=peak)
            print(f"[calibrate]   -> {s_it:.3f} s/it   {peak:.1f} GB peak   "
                  f"L={row['packed_len']}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            row.update(error="OOM", error_detail=str(e)[:300])
            print("[calibrate]   -> OOM", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            # Keep the DETAIL: an error name alone is unactionable and a failed
            # calibration is expensive to repeat.
            row.update(error=type(e).__name__, error_detail=traceback.format_exc()[-600:])
            print(f"[calibrate]   -> {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            batch = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        rows.append(row)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    md_path = os.path.join(out_dir, "calibration.md")
    with open(md_path, "w") as f:
        f.write("# T0 — context grid cost calibration\n\n")
        f.write(f"Real training steps: {cfg.graph_attn_impl}, k_hop={cfg.k_hop}, "
                f"lora_r={cfg.lora_r}, {cfg.dtype}, batch={cfg.batch_size}, "
                f"gradient_checkpointing={cfg.gradient_checkpointing}, "
                f"magnetic_m={cfg.magnetic_m}. Forward + backward + optimizer step, "
                f"pre-built batches (no dataloader) — a FLOOR on wall-clock s/it.\n\n")
        f.write("| N | T | packed L | in-train | s/it | peak GB |\n")
        f.write("| --: | --: | --: | :-- | --: | --: |\n")
        for r in rows:
            if r["ok"]:
                f.write(f"| {r['n_nodes']} | {r['tokens_per_node']} | {r['packed_len']:,} | "
                        f"{'yes' if r['in_train_distribution'] else 'no'} | "
                        f"{r['s_per_it']:.3f} | {r['peak_gb']:.1f} |\n")
            else:
                f.write(f"| {r['n_nodes']} | {r['tokens_per_node']} | "
                        f"{r['expected_len']:,} (est) | "
                        f"{'yes' if r['in_train_distribution'] else 'no'} | "
                        f"**{r['error']}** | — |\n")
    print(f"[calibrate] wrote {jsonl_path} and {md_path}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.context.calibrate",
        description="T0 cost calibration over the context grid (real training steps).")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--cells", choices=("all", "train"), default="all")
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--max-train-len", type=int, default=RunConfig().max_train_len)
    p.add_argument("--batch-size", type=int, default=RunConfig().batch_size)
    p.add_argument("--compile-mode", default="default",
                   help="'default' (~16 s/shape) is enough for calibration; the real "
                        "run uses max-autotune-no-cudagraphs, which is FASTER per step.")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false", default=True)
    args = p.parse_args(argv)

    cfg = RunConfig(max_train_len=args.max_train_len, batch_size=args.batch_size,
                    compile_mode=args.compile_mode,
                    gradient_checkpointing=args.gradient_checkpointing).validate()
    run_calibration(cfg, out_dir=args.out_dir, which=args.cells,
                    n_warmup=args.n_warmup, n_iter=args.n_iter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
