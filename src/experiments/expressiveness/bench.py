"""
Benchmark mode (synthetic large graphs): forward+backward throughput + peak
CUDA memory per implementation, plus token/block sparsity per size.

This is the *isolated* throughput probe (a hand-rolled few-step loop). The real
training runs measure their own speed/memory live via ``StepMemCallback``; this
mode exists to sweep large sizes quickly without full training.
"""

import time

import torch
from transformers import AutoTokenizer

from .data_gen import prepare_dataset
from .dispatch import (
    build_model, build_collator, forward_loss, active_params_for, select_active_params,
)
from .instrumentation import measure_density


def bench_impl(impl, examples, bias_params, model_name, k_hop, k_hop_directed,
               batch_size, num_warmup, num_iters, device, flex_compile_mode):
    """Time a few train steps for one impl on a fixed pool of large graphs.

    Returns (ms_per_step, peak_mem_gb) or (None, None) on CUDA OOM.
    """
    model, tokenizer = build_model(
        impl, model_name, bias_params, k_hop, k_hop_directed, device, flex_compile_mode
    )
    model = select_active_params(model, active_params=active_params_for(impl))
    model.train()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = build_collator(impl, tokenizer, pad_token_id, bias_params, k_hop, k_hop_directed)

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


def run_bench(impls, sizes, bias_params, model_name, k_hop, k_hop_directed,
              batch_size, num_warmup, num_iters, num_examples, spectral_dims,
              device, flex_compile_mode, density_sample_graphs=16, density_sample_batches=8,
              ordering="rcm"):
    """Generate small in-memory HARD datasets at each size and benchmark every impl."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows = []
    for size in sizes:
        print(f"\n[bench] generating {num_examples} HARD graphs of size {size}...")
        ds = prepare_dataset(
            num_examples,
            min_size=size, max_size=size,
            spectral_dims=spectral_dims,
            tokenizer_name=model_name,
            max_rwse_steps=bias_params["max_rw_steps"],
            max_rrwp_steps=bias_params["max_rw_steps"],
            easy=False,
            magnetic_q=bias_params["magnetic_q"],
            ordering=ordering,
        )
        examples = [ds[i] for i in range(len(ds))]

        # Sparsity is a data property — measure once per (size, k_hop).
        dens = measure_density(
            ds, tokenizer, pad_token_id, bias_params, k_hop, k_hop_directed,
            batch_size=batch_size,
            num_sample_graphs=min(density_sample_graphs, len(ds)),
            num_sample_batches=density_sample_batches, device=device)

        for impl in impls:
            print(f"[bench] size={size} impl={impl} k_hop={k_hop} ...")
            ms, peak = bench_impl(
                impl, examples, bias_params, model_name, k_hop, k_hop_directed,
                batch_size, num_warmup, num_iters, device, flex_compile_mode,
            )
            status = "OK" if ms is not None else "OOM"
            rows.append((size, impl, k_hop, ms, peak, status,
                         dens["token_sparsity_mean"], dens["block_sparsity_mean"]))

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
    """Large-graph throughput + memory + sparsity sweep across k_hops."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k_hop in cfg.k_hops:
        print("\n" + "#" * 72)
        print(f"# BENCH — k_hop={k_hop}")
        print("#" * 72)
        run_bench(
            impls=cfg.impls,
            sizes=cfg.bench_sizes,
            bias_params=cfg.bias_params,
            model_name=cfg.model_name,
            k_hop=k_hop,
            k_hop_directed=cfg.k_hop_directed,
            batch_size=cfg.bench_batch_size,
            num_warmup=cfg.bench_num_warmup,
            num_iters=cfg.bench_num_iters,
            num_examples=cfg.bench_num_examples,
            spectral_dims=cfg.spectral_dims,
            device=device,
            flex_compile_mode=cfg.flex_compile_mode,
            density_sample_graphs=cfg.density_sample_graphs,
            density_sample_batches=cfg.density_sample_batches,
            ordering=cfg.ordering,
        )
