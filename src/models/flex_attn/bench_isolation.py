"""
Attention-only (isolation) benchmark: one attention layer, three backends.

Methods
-------
  * ``eager`` — the model's dense path: a ``(B,1,L,L)`` additive structural
    mask + a ``(B,H,L,L)`` token-expanded soft bias, fed to SDPA. (The float
    mask disables the flash kernel, so this is the realistic eager cost.)
  * ``flex``    — ``flex_core``: prebuilt sparse ``BlockMask`` + on-the-fly
    ``node_bias`` gather via ``score_mod``, compiled ``flex_attention``.
  * ``flash``   — plain causal SDPA flash kernel, no bias / no graph structure.
    A *floor* reference, NOT functionally equivalent (it can't express the bias,
    bidirectional prefix, or K-hop) — it answers "what does ideal dense
    attention cost at this shape".

We time **forward and backward separately** (CUDA events) and record **peak
memory** for each pass. For ``flex`` we additionally report the one-time
``compile_ms`` and the per-batch ``blockmask_build_ms`` (amortized across all
layers in the real model, so reported separately rather than folded in).

The node bias is detached to a leaf before timing, so the backward measures the
attention + bias-gather (incl. the scatter-add into ``node_bias`` we flagged as
the real backward cost) but NOT the bias module's internal backprop.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.llama.modeling_llama import repeat_kv

from src.models.flex_attn.inputs import GraphSpec, make_attention_inputs
from src.models.flex_attn.density import compute_density
from src.models.flex_attn import flex_core


# ── timing helpers ────────────────────────────────────────────────────────────

def _cuda_time(fn, n_warmup: int, n_iter: int) -> tuple[float, float]:
    """Mean/std ms over ``n_iter`` calls of ``fn`` (after ``n_warmup`` warmups)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return float(t.mean()), float(t.std())


def _mb(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


# Result is printed as pretty (indented) JSON between these sentinels, so it is
# both human-readable on the terminal and machine-parseable by run_sweep.py.
RESULT_BEGIN = "===RESULT_JSON_BEGIN==="
RESULT_END = "===RESULT_JSON_END==="


def emit_result(res: dict) -> None:
    print(RESULT_BEGIN)
    print(json.dumps(res, indent=2))
    print(RESULT_END)


# ── the three forward/backward closures ───────────────────────────────────────

def _build_eager(ai, scaling):
    """Return (attn_fn(q,k,v) -> out, captured_grad_tensors).

    The bias is a captured leaf (``nb``); q/k/v are supplied by the timing
    harness so it controls their grad lifecycle (reusing requires-grad inputs
    across compiled-flex calls inflates timings ~10×; fresh leaves per step is
    the correct methodology).
    """
    from src.models.graph_attention_v2 import (
        build_dense_structural_mask, expand_node_to_token_bias,
    )
    L = ai["q_len"]
    nb = None if ai["node_bias"] is None else ai["node_bias"].detach().requires_grad_(True)
    Hkv = ai["key"].shape[1]

    def attn_fn(q, k, v):
        n_rep = q.shape[1] // Hkv
        mask = build_dense_structural_mask(
            node_ids=ai["node_ids"], prompt_node=ai["prompt_node"], pad_mask=ai["pad_mask"],
            k_hop_mask=ai["k_hop_mask"], k_hop=ai["k_hop"], q_len=L, kv_len=L,
            dtype=q.dtype, device=q.device,
        )
        attn_mask = mask if nb is None else mask + expand_node_to_token_bias(nb, ai["node_ids"], L, L)
        return F.scaled_dot_product_attention(
            q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), attn_mask=attn_mask, scale=scaling,
        )
    return attn_fn, ([nb] if nb is not None else []), (ai["query"], ai["key"], ai["value"])


def _build_flex(ai, scaling, block_size, dynamic):
    """Flex path. Pads the sequence to a multiple of ``block_size`` (REQUIRED:
    non-aligned L hits a ~14× Triton kernel cliff) and slices the output back."""
    nb = None if ai["node_bias"] is None else ai["node_bias"].detach().requires_grad_(True)
    qp, kp, vp, nidp, pmp, L, Lb = flex_core.pad_to_block(
        ai["query"], ai["key"], ai["value"], ai["node_ids"], ai["pad_mask"], block_size,
    )

    t0 = time.perf_counter()
    bm = flex_core.build_block_mask(
        nidp, ai["prompt_node"], pmp, ai["k_hop_mask"], ai["k_hop"],
        Lb, Lb, block_size=block_size, device=ai["query"].device,
    )
    torch.cuda.synchronize()
    blockmask_build_ms = (time.perf_counter() - t0) * 1e3
    smod = flex_core.make_score_mod(nb, nidp)

    def attn_fn(q, k, v):
        out = flex_core.flex_attention_forward(
            q, k, v, block_mask=bm, score_mod=smod, scaling=scaling,
            enable_gqa=True, dynamic=dynamic,
        )
        return out[:, :, :L]                                    # drop padded query rows
    return attn_fn, ([nb] if nb is not None else []), (qp, kp, vp), blockmask_build_ms


def _build_flash(ai, scaling):
    Hkv = ai["key"].shape[1]

    def attn_fn(q, k, v):
        n_rep = q.shape[1] // Hkv
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(
                q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), is_causal=True, scale=scaling,
            )
    return attn_fn, [], (ai["query"], ai["key"], ai["value"])


# ── one (method, config) run ──────────────────────────────────────────────────

def run_isolation(
    spec: GraphSpec,
    method: str,
    batch_size: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    block_size: int = 128,
    dynamic: bool = False,
    n_warmup: int = 5,
    n_iter: int = 20,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    dev = torch.device("cuda")
    result = {
        "kind": "isolation", "method": method, "ok": False, "error": None,
        "spec": dataclasses.asdict(spec),
        "run": {
            "batch_size": batch_size, "num_heads": num_heads, "num_kv_heads": num_kv_heads,
            "head_dim": head_dim, "block_size": block_size, "dynamic": dynamic,
        },
        "shape": {}, "density": {}, "timing_ms": {}, "memory_mb": {},
    }
    try:
        ai, meta = make_attention_inputs(
            spec, batch_size, num_heads, num_kv_heads, head_dim, dev, dtype=dtype,
        )
        ai["k_hop"] = spec.k_hop
        result["shape"] = {k: meta[k] for k in ("seq_len", "max_num_nodes", "total_tokens")}
        scaling = head_dim ** -0.5

        # Mask density — a property of the graph/mask, reported for EVERY method
        # (including flash) so token- and block-level sparsity are comparable.
        d = compute_density(ai["node_ids"], ai["prompt_node"], ai["pad_mask"],
                            ai["k_hop_mask"], spec.k_hop, block_size=block_size)
        result["density"] = d.as_dict()

        compile_ms = None
        blockmask_build_ms = None
        if method == "eager":
            attn_fn, captured, qkv_src = _build_eager(ai, scaling)
        elif method == "flash":
            attn_fn, captured, qkv_src = _build_flash(ai, scaling)
        elif method == "flex":
            attn_fn, captured, qkv_src, blockmask_build_ms = _build_flex(ai, scaling, block_size, dynamic)
        else:
            raise ValueError(f"unknown method {method!r}")

        q0, k0, v0 = qkv_src
        qd, kd, vd = q0.detach(), k0.detach(), v0.detach()      # pure-forward inputs

        def _fwd_only():
            with torch.no_grad():
                return attn_fn(qd, kd, vd)

        def _train_step():
            for c in captured:
                c.grad = None
            qg = q0.detach().requires_grad_(True)
            kg = k0.detach().requires_grad_(True)
            vg = v0.detach().requires_grad_(True)
            out = attn_fn(qg, kg, vg)
            out.sum().backward()

        # First calls trigger compile (flex); time it as wall-clock, exclude from latency.
        t0 = time.perf_counter()
        _fwd_only(); torch.cuda.synchronize()
        _train_step(); torch.cuda.synchronize()
        if method == "flex":
            compile_ms = (time.perf_counter() - t0) * 1e3

        # ── forward-only timing + peak memory ──
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        fwd_ms, fwd_std = _cuda_time(_fwd_only, n_warmup, n_iter)
        torch.cuda.synchronize()
        peak_fwd_mb = _mb(torch.cuda.max_memory_allocated())

        # ── forward+backward timing + peak memory ──
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        bwd_ms_total, bwd_std = _cuda_time(_train_step, n_warmup, n_iter)
        torch.cuda.synchronize()
        peak_bwd_mb = _mb(torch.cuda.max_memory_allocated())

        result["ok"] = True
        result["timing_ms"] = {
            "fwd": fwd_ms, "fwd_std": fwd_std,
            "fwd_bwd": bwd_ms_total, "fwd_bwd_std": bwd_std,
            "bwd_est": bwd_ms_total - fwd_ms,       # backward ≈ (fwd+bwd) − fwd
            "compile": compile_ms,                  # flex one-time
            "blockmask_build": blockmask_build_ms,  # flex per-batch (amortized over layers)
            "reorder_mean": meta["reorder_time_mean_s"] * 1e3,
            "reorder_max": meta["reorder_time_max_s"] * 1e3,
        }
        result["memory_mb"] = {
            "peak_fwd": peak_fwd_mb,
            "peak_fwd_bwd": peak_bwd_mb,
        }
    except torch.cuda.OutOfMemoryError as e:
        result["error"] = "OOM"
        result["error_detail"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001 — report, don't crash the sweep
        result["error"] = type(e).__name__
        result["error_detail"] = str(e)[:300]
    finally:
        gc.collect(); torch.cuda.empty_cache()
    return result


# ── CLI (one run → one JSON line on stdout) ───────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["eager", "flex", "flash"])
    p.add_argument("--n-nodes", type=int, required=True)
    p.add_argument("--tokens-per-node", type=int, required=True)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--k-hop", type=int, default=0)
    p.add_argument("--k-hop-directed", action="store_true")
    p.add_argument("--ordering", default="rcm", choices=["rcm", "random", "identity"])
    p.add_argument("--directed", action="store_true")
    p.add_argument("--magnetic-m", type=int, default=32)
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=20)
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    spec = GraphSpec(
        n_nodes=a.n_nodes, tokens_per_node=a.tokens_per_node, prompt_tokens=a.prompt_tokens,
        k_hop=a.k_hop, k_hop_directed=a.k_hop_directed, ordering=a.ordering, directed=a.directed,
        magnetic_m=a.magnetic_m, jitter=a.jitter, seed=a.seed,
    )
    res = run_isolation(
        spec, a.method, batch_size=a.batch_size, num_heads=a.num_heads,
        num_kv_heads=a.num_kv_heads, head_dim=a.head_dim, block_size=a.block_size,
        dynamic=a.dynamic, n_warmup=a.n_warmup, n_iter=a.n_iter,
    )
    emit_result(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
