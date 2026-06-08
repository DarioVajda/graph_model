"""
Nsight Compute (ncu) roofline: memory-bound vs compute-bound, fwd vs bwd.

Memory and compute *overlap* on-GPU, so there is no clean "X ms memory + Y ms
compute" split. What ncu *can* give — and what this tool collects — is, per
kernel, the achieved **DRAM throughput** and **compute (SM) throughput** as a
percentage of peak (the SpeedOfLight section). A kernel with mem% ≫ compute% is
memory-bound; the reverse is compute-bound. We attribute kernels to fwd vs bwd
by running two phases (``fwd`` and ``train``) and by kernel name.

Two modes:
  * ``--target`` (inner): run a few iterations of one phase for one config, so
    ncu has something to profile. Not meant to be called directly.
  * driver (default): wrap the target in ``ncu`` for a small representative set
    of configs, parse the CSV, and summarize. ``ncu`` is slow and serializing,
    so keep the set to ~6-10 configs (this is the ``--profile-ncu`` path, off by
    default in the main sweep).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys

import torch

from src.models.flex_attn.inputs import GraphSpec, make_attention_inputs
from src.models.flex_attn import flex_core

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
NCU = "/usr/local/cuda/bin/ncu"

# A small, representative set spanning the regimes that matter:
# small-node (poor block sparsity) ↔ large-node (good), low/high K.
REPRESENTATIVE = [
    dict(n_nodes=512, tokens_per_node=16, k_hop=2),   # good block sparsity
    dict(n_nodes=2048, tokens_per_node=8, k_hop=2),   # large N, small tok/node
    dict(n_nodes=128, tokens_per_node=128, k_hop=2),  # large tok/node
    dict(n_nodes=512, tokens_per_node=16, k_hop=0),   # no K-hop (dense prefix)
    dict(n_nodes=512, tokens_per_node=16, k_hop=8),   # high K
    dict(n_nodes=64, tokens_per_node=8, k_hop=2),     # tiny
]


# ── inner target ──────────────────────────────────────────────────────────────

def _target(args):
    dev = torch.device("cuda")
    spec = GraphSpec(n_nodes=args.n_nodes, tokens_per_node=args.tokens_per_node,
                     k_hop=args.k_hop, ordering=args.ordering, magnetic_m=args.magnetic_m,
                     jitter=0.0, seed=args.seed)
    ai, _ = make_attention_inputs(spec, 1, args.num_heads, args.num_kv_heads,
                                  args.head_dim, dev)
    L = ai["q_len"]; sc = args.head_dim ** -0.5
    bm = flex_core.build_block_mask(ai["node_ids"], ai["prompt_node"], ai["pad_mask"],
                                    ai["k_hop_mask"], spec.k_hop, L, L, device=dev)
    nb = ai["node_bias"].detach().requires_grad_(True)
    smod = flex_core.make_score_mod(nb, ai["node_ids"])
    fa = flex_core.get_flex_attention(dynamic=False)

    def fwd(q, k, v):
        return fa(q, k, v, block_mask=bm, score_mod=smod, scale=sc, enable_gqa=True)

    q0, k0, v0 = ai["query"], ai["key"], ai["value"]
    # Warm up / compile OUTSIDE the profiled region.
    for _ in range(3):
        qg = q0.detach().requires_grad_(True); kg = k0.detach().requires_grad_(True); vg = v0.detach().requires_grad_(True)
        out = fwd(qg, kg, vg)
        if args.phase == "train":
            out.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        qg = q0.detach().requires_grad_(True); kg = k0.detach().requires_grad_(True); vg = v0.detach().requires_grad_(True)
        out = fwd(qg, kg, vg)
        if args.phase == "train":
            out.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return 0


# ── driver ────────────────────────────────────────────────────────────────────

_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
]


def _run_ncu(cfg: dict, phase: str) -> list[dict]:
    cmd = [
        NCU, "--csv", "--target-processes", "all",
        "--profile-from-start", "off",
        "--metrics", ",".join(_METRICS),
        sys.executable, "-m", "src.models.flex_attn.profile_ncu", "--target",
        "--phase", phase,
        "--n-nodes", str(cfg["n_nodes"]), "--tokens-per-node", str(cfg["tokens_per_node"]),
        "--k-hop", str(cfg["k_hop"]), "--iters", "3",
    ]
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    # ncu prints a CSV table to stdout once profiling succeeds.
    out = proc.stdout
    start = out.find('"ID"')
    if start < 0:
        tail = (proc.stderr or "")[-300:]
        hint = ("ncu produced no CSV. ")
        if "driver resource was unavailable" in proc.stderr or "DCGM" in proc.stderr:
            hint += ("DCGM (nv-hostengine) is holding the profiling counters — "
                     "pause it (`dcgmi profile --pause`) or stop nv-hostengine, "
                     "then retry. Use `--analytic` for an ncu-free roofline estimate. ")
        else:
            hint += ("Likely missing GPU performance-counter permission "
                     "(CAP_SYS_ADMIN / admin-enabled counters). ")
        return [{"error": hint + "stderr tail: " + tail}]
    rows = list(csv.DictReader(io.StringIO(out[start:])))
    parsed = []
    for r in rows:
        name = r.get("Kernel Name", "")[:48]
        try:
            sm = float(r.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "nan").replace(",", ""))
            mem = float(r.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "nan").replace(",", ""))
        except ValueError:
            continue
        parsed.append({"kernel": name, "compute_pct": sm, "memory_pct": mem,
                       "bound": "memory" if mem > sm else "compute"})
    return parsed


def _driver(args):
    cfgs = REPRESENTATIVE[:args.max_configs]
    results = []
    for cfg in cfgs:
        for phase in ("fwd", "train"):
            kernels = _run_ncu(cfg, phase)
            results.append({"config": cfg, "phase": phase, "kernels": kernels})
            print(f"\n=== {cfg} | phase={phase} ===")
            for kr in kernels:
                if "error" in kr:
                    print("  " + kr["error"]); continue
                print(f"  {kr['kernel']:<50} compute={kr['compute_pct']:5.1f}%  "
                      f"mem={kr['memory_pct']:5.1f}%  -> {kr['bound']}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    out = os.path.join(_HERE, "results", "ncu_roofline.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out}")
    return 0


# ── analytic roofline (ncu-free fallback) ─────────────────────────────────────

# A100 bf16: ~312 TFLOP/s tensor-core, ~2.0 TB/s HBM → ridge ≈ 156 FLOP/byte.
A100_BF16_TFLOPS = 312.0
A100_HBM_TBPS = 2.0
A100_RIDGE = (A100_BF16_TFLOPS * 1e12) / (A100_HBM_TBPS * 1e12)


def _analytic(args):
    """Estimate memory- vs compute-bound for the flex forward without ncu.

    Memory and compute overlap, so this is the roofline *classification* — it
    compares the kernel's arithmetic intensity (FLOP/byte) to the A100 bf16 ridge
    point. flex is flash-style: Q/K/V/O move once (O(L)); the score_mod adds one
    cached node_bias read per computed pair. FLOPs scale with the *computed*
    (non-skipped) pairs ≈ block_density · L². So AI — and the bound — shifts from
    memory-bound (small L) to compute-bound (large L · density).
    """
    from src.models.flex_attn.density import compute_density
    B, H, d = 1, 32, 64
    bytes_el = 2  # bf16
    print(f"A100 bf16 ridge ≈ {A100_RIDGE:.0f} FLOP/byte\n")
    print(f"{'config':>26} | {'L':>6} | {'blkdens':>7} | {'GFLOP':>8} | {'MB':>7} | "
          f"{'AI':>7} | bound")
    results = []
    for cfg in REPRESENTATIVE[:args.max_configs]:
        spec = GraphSpec(n_nodes=cfg["n_nodes"], tokens_per_node=cfg["tokens_per_node"],
                         k_hop=cfg["k_hop"], ordering="rcm", magnetic_m=32, jitter=0.0)
        ai, meta = make_attention_inputs(spec, B, H, 8, d, torch.device("cuda"))
        L = ai["q_len"]
        dens = compute_density(ai["node_ids"], ai["prompt_node"], ai["pad_mask"],
                               ai["k_hop_mask"], cfg["k_hop"], block_size=128)
        pairs = dens.block_density * L * L * B * H        # computed (q,k) pairs
        flops = 4.0 * pairs * d                            # QK^T (2d) + PV (2d)
        # bytes: Q,K,V,O moved once + one cached node_bias read per computed pair
        bytes_moved = 4 * B * H * L * d * bytes_el + pairs * bytes_el
        ai_intensity = flops / bytes_moved
        bound = "compute" if ai_intensity > A100_RIDGE else "memory"
        row = {"config": cfg, "L": L, "block_density": dens.block_density,
               "gflop": flops / 1e9, "mb": bytes_moved / 1e6,
               "arith_intensity": ai_intensity, "bound": bound}
        results.append(row)
        print(f"{str(cfg):>26} | {L:>6} | {dens.block_density:7.3f} | {flops/1e9:8.2f} | "
              f"{bytes_moved/1e6:7.1f} | {ai_intensity:7.1f} | {bound}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    with open(os.path.join(_HERE, "results", "analytic_roofline.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", action="store_true", help="inner profiled process")
    p.add_argument("--analytic", action="store_true",
                   help="ncu-free roofline estimate (use when DCGM blocks ncu)")
    p.add_argument("--phase", default="fwd", choices=["fwd", "train"])
    p.add_argument("--n-nodes", type=int, default=512)
    p.add_argument("--tokens-per-node", type=int, default=16)
    p.add_argument("--k-hop", type=int, default=2)
    p.add_argument("--ordering", default="rcm")
    p.add_argument("--magnetic-m", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--max-configs", type=int, default=6)
    args = p.parse_args(argv)
    if args.target:
        return _target(args)
    if args.analytic:
        return _analytic(args)
    return _driver(args)


if __name__ == "__main__":
    sys.exit(main())
