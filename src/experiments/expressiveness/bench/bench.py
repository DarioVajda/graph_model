"""
Benchmark mode (synthetic large graphs): forward+backward throughput + peak
CUDA memory per implementation, plus token/block sparsity.

This is the *isolated* throughput probe (a hand-rolled few-step loop). The real
training runs measure their own speed/memory live via ``StepMemCallback``; this
mode exists to probe a large size quickly without full training. The graph size
is a single fixed value (``cfg.num_nodes``); sweep multiple sizes by re-invoking.
"""

import time

import torch
from transformers import AutoTokenizer

from .config import MAGNETIC_Q, model_bias_config
from .data_gen import prepare_dataset
from .dispatch import (
    build_model, build_collator, forward_loss, active_params_for, select_active_params,
)
from .instrumentation import measure_density
from .results import append_jsonl, BENCH_RESULTS_PATH
from ...models.flex_kernel import align_len


def _tight_buckets(examples, block_size=128):
    """Single tight (len, node) bucket covering this fixed-size pool: the max
    packed L aligned up to a block, and the max node count. One bucket => one
    compiled shape and ~no padding waste beyond block alignment."""
    max_L = max(sum(len(t) for t in ex["input_ids"]) for ex in examples)
    max_N = max(len(ex["input_ids"]) for ex in examples)
    return [align_len(max_L, block_size)], [max_N]


def bench_impl(impl, examples, model_name, k_hop, k_hop_directed,
               batch_size, num_warmup, num_iters, device, flex_compile_mode,
               magnetic_m=0, len_buckets=None, node_buckets=None):
    """Time a few train steps for one impl on a fixed pool of large graphs.

    Returns (ms_per_step, peak_mem_gb) or (None, None) on CUDA OOM.
    """
    model, tokenizer = build_model(
        impl, model_name, model_bias_config(), k_hop, k_hop_directed, device, flex_compile_mode
    )
    model = select_active_params(model, active_params=active_params_for(impl))
    model.train()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = build_collator(impl, tokenizer, pad_token_id, k_hop, k_hop_directed,
                              magnetic_m=magnetic_m, len_buckets=len_buckets, node_buckets=node_buckets)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def make_batch(step):
        lo = (step * batch_size) % max(1, len(examples))
        chunk = examples[lo:lo + batch_size]
        if len(chunk) < batch_size:
            chunk = examples[:batch_size]
        return collator(chunk)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        # Warmup (absorbs flex's one-time torch.compile on the first step).
        for s in range(num_warmup):
            loss = forward_loss(impl, model, make_batch(s), device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for s in range(num_iters):
            loss = forward_loss(impl, model, make_batch(num_warmup + s), device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms_per_step = (time.perf_counter() - t0) / num_iters * 1000.0
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
        return ms_per_step, peak_gb
    except torch.cuda.OutOfMemoryError:
        return None, None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None, None
        raise
    finally:
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_bench(impls, size, model_name, k_hop, k_hop_directed,
              batch_size, num_warmup, num_iters, num_examples,
              device, flex_compile_mode, magnetic_m=0,
              density_sample_graphs=16, density_sample_batches=8,
              ordering="rcm", len_buckets=None, node_buckets=None):
    """Generate a small in-memory HARD dataset at ``size`` and benchmark every impl."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    print(f"\n[bench] generating {num_examples} HARD graphs of size {size}...")
    ds = prepare_dataset(
        num_examples,
        min_size=size, max_size=size,
        tokenizer_name=model_name,
        easy=False,
        magnetic_q=MAGNETIC_Q,
        magnetic_m=magnetic_m,
        ordering=ordering,
    )
    examples = [ds[i] for i in range(len(ds))]

    # Tight flex buckets from the actual packed lengths (unless the caller pinned
    # them), so the size compiles one shape with minimal padding.
    if len_buckets is None and node_buckets is None:
        len_buckets, node_buckets = _tight_buckets(examples)
        print(f"[bench] size={size} tight buckets: L={len_buckets} N={node_buckets}")

    # Sparsity is a data property — measure once per (size, k_hop).
    dens = measure_density(
        ds, tokenizer, pad_token_id, magnetic_m, k_hop, k_hop_directed,
        batch_size=batch_size,
        num_sample_graphs=min(density_sample_graphs, len(ds)),
        num_sample_batches=density_sample_batches, device=device)

    rows = []
    for impl in impls:
        print(f"[bench] size={size} impl={impl} k_hop={k_hop} ...")
        ms, peak = bench_impl(
            impl, examples, model_name, k_hop, k_hop_directed,
            batch_size, num_warmup, num_iters, device, flex_compile_mode,
            magnetic_m=magnetic_m, len_buckets=len_buckets, node_buckets=node_buckets,
        )
        status = "OK" if ms is not None else "OOM"
        rows.append((size, impl, k_hop, ms, peak, status,
                     dens["token_sparsity_mean"], dens["block_sparsity_mean"]))
        append_jsonl(BENCH_RESULTS_PATH, {
            "mode": "bench",
            # ── hyperparameters ──
            "impl": impl, "k_hop": k_hop, "size": size,
            "model_name": model_name, "magnetic_m": magnetic_m, "ordering": ordering,
            "k_hop_directed": k_hop_directed, "flex_compile_mode": flex_compile_mode,
            "batch_size": batch_size, "num_warmup": num_warmup, "num_iters": num_iters,
            "num_examples": num_examples,
            "len_buckets": list(len_buckets) if len_buckets else None,
            "node_buckets": list(node_buckets) if node_buckets else None,
            # ── results ──
            "status": status, "ms_per_step": ms, "peak_gb": peak,
            "token_sparsity": dens["token_sparsity_mean"],
            "block_sparsity": dens["block_sparsity_mean"],
        })
    print(f"[results] appended {len(impls)} benchmark run(s) to {BENCH_RESULTS_PATH}")

    print("\n" + "=" * 96)
    print(f"{'size':>6} | {'impl':<9} | {'k_hop':>5} | {'ms/step':>9} | {'peak GB':>8} | "
          f"{'token sp':>8} | {'block sp':>8} | status")
    print("-" * 96)
    for size, impl, kh, ms, peak, status, tok, blk in rows:
        ms_str = f"{ms:9.1f}" if ms is not None else f"{'—':>9}"
        peak_str = f"{peak:8.2f}" if peak is not None else f"{'—':>8}"
        print(f"{size:>6} | {impl:<9} | {kh:>5} | {ms_str} | {peak_str} | "
              f"{tok:8.3f} | {blk:8.3f} | {status}")
    print("=" * 96)
    return rows


def run_bench_mode(cfg):
    """Large-graph throughput + memory + sparsity at the fixed size ``cfg.num_nodes``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k_hop in cfg.k_hops:
        print("\n" + "#" * 72)
        print(f"# BENCH — size={cfg.num_nodes}, k_hop={k_hop}")
        print("#" * 72)
        run_bench(
            impls=cfg.impls,
            size=cfg.num_nodes,
            model_name=cfg.model_name,
            k_hop=k_hop,
            k_hop_directed=cfg.k_hop_directed,
            batch_size=cfg.bench_batch_size,
            num_warmup=cfg.bench_num_warmup,
            num_iters=cfg.bench_num_iters,
            num_examples=cfg.bench_num_examples,
            device=device,
            flex_compile_mode=cfg.flex_compile_mode,
            magnetic_m=cfg.magnetic_m,
            density_sample_graphs=cfg.density_sample_graphs,
            density_sample_batches=cfg.density_sample_batches,
            ordering=cfg.ordering,
            len_buckets=cfg.len_buckets, node_buckets=cfg.node_buckets,
        )
