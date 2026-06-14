"""
Experiment #10: does narrowing the captured ``node_ids`` (int64 → int32 → int16)
speed up the flex path?

The score_mod's core op gathers ``node_bias[b, h, node_ids[q], node_ids[k]]``
per (q, kv) element, and the structural BlockMask builder indexes the same
``node_ids``. Both bake the tensor's dtype into the compiled Triton: a 64-bit id
means an 8-byte index load per element and 64-bit address arithmetic in the
inner loop. Node counts here are ≤ a few thousand (int16 max 32767), so the cast
is lossless — the only question is whether it buys measurable time, and how that
varies with the regime where the gather dominates (k=0, finding #3) or goes
DRAM-bound (large N, the bias table spilling L2, finding #5).

Two parts:

  * **parity** — flex output must be *bitwise identical* across the three id
    dtypes (only the index width changes, never a value) and match the dense
    reference. Run first, cheap, on the fast-compiling default mode.
  * **timing** — ``run_isolation`` (the real harness: fresh grad leaves, direct
    backward, peak memory) over a few representative configs × {int64, int32,
    int16}, under the production autotune mode + the #6 block-size gate
    (64 when k>0, 128 when k=0). Writes a JSONL + a pivot table with each
    dtype's fwd / bwd / fwd+bwd and the %Δ vs the int64 baseline.

Run (from repo root):

    python -m src.models.flex_attn.exp_node_id_dtype \
        --out-dir src/models/flex_attn/results_h100_nodeid

Add ``--quick`` for a fast smoke (default compile mode, 128-blocks, tiny iters).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from typing import Optional

import torch

from src.models.flex_attn.inputs import GraphSpec, make_attention_inputs
from src.models.flex_attn import flex_core

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Must match bench_isolation.RESULT_BEGIN/END.
RESULT_BEGIN = "===RESULT_JSON_BEGIN==="
RESULT_END = "===RESULT_JSON_END==="

# int16 is NOT a candidate: ``node_ids[q]`` is *used as an index* into
# ``node_bias`` inside the score_mod, and torch requires index tensors to be
# long / int / byte / bool ("tensors used as indices must be long, int, byte or
# bool"). int16 is rejected at trace time; uint8 (byte) would pass but only
# covers ≤255 nodes, so int32 is the narrowest generally-usable width.
DTYPES = ("int64", "int32")

# (label, n_nodes, tokens_per_node, k_hop, block_size). Chosen to span the
# regimes where the gather's share is largest: k=0 is gather-dominated (the
# score_mod's per-token redundancy with nothing to skip, finding #3); large N
# pushes the (B,H,N,N) bias table past L2 so the gather goes DRAM-bound
# (finding #5) — exactly where a narrower index load should help most.
CONFIGS = [
    # label                 N     tpn  k  block
    ("k0_512x32_dense",     512,   32, 0, 128),  # gather-dominated worst case
    ("k2_512x32_op",        512,   32, 2,  64),  # operating point, moderate N
    ("k2_2048x8_largeN",   2048,    8, 2,  64),  # operating point, bias table ≫ L2
]


def block_for(k_hop: int) -> int:
    """The #6 production block-size gate: 64 when k>0 (skips scattered K-hop
    neighbourhoods more tightly), 128 when k=0 (nothing to skip; bigger tiles)."""
    return 64 if k_hop > 0 else 128


def build_grid(nodes, tpn_list, k_hops, max_tokens: int):
    """Cartesian product of (nodes × tpn × k_hop) → config tuples, with the #6
    block gate applied per cell and cells whose est. token count (≈ N·tpn)
    exceeds ``max_tokens`` skipped. Lets one command sweep a superset."""
    configs = []
    for k in k_hops:
        for N in nodes:
            for tpn in tpn_list:
                if N * tpn > max_tokens:
                    continue
                configs.append((f"k{k}_{N}x{tpn}", N, tpn, k, block_for(k)))
    return configs


# ── parity: identical output across id dtypes, and vs the dense reference ──────

@torch.no_grad()
def check_parity(device: torch.device) -> None:
    """flex(int16/int32) must equal flex(int64) bitwise and match dense_reference."""
    dt = torch.bfloat16
    H, Hkv, d = 8, 2, 64
    scaling = d ** -0.5
    print("── parity (flex across id dtypes vs dense reference) ──")
    for k_hop in (0, 2):
        spec = GraphSpec(n_nodes=48, tokens_per_node=8, prompt_tokens=48,
                         k_hop=k_hop, ordering="rcm", magnetic_m=16)
        ai, meta = make_attention_inputs(spec, 2, H, Hkv, d, device, dtype=dt)
        q, key, val = ai["query"].detach(), ai["key"].detach(), ai["value"].detach()

        ref = flex_core.dense_reference(
            q, key, val,
            node_ids=ai["node_ids"], prompt_node=ai["prompt_node"],
            pad_mask=ai["pad_mask"], k_hop_mask=ai["k_hop_mask"], k_hop=k_hop,
            node_bias=ai["node_bias"], scaling=scaling,
        )

        outs = {}
        for name in DTYPES:
            nid = ai["node_ids"] if name == "int64" else ai["node_ids"].to(getattr(torch, name))
            qp, kp, vp, nidp, pmp, L, Lb = flex_core.pad_to_block(
                q, key, val, nid, ai["pad_mask"], 128,
            )
            bm = flex_core.build_block_mask(
                nidp, ai["prompt_node"], pmp, ai["k_hop_mask"], k_hop,
                Lb, Lb, block_size=128, device=device,
            )
            smod = flex_core.make_score_mod(ai["node_bias"], nidp)
            out = flex_core.flex_attention_forward(
                qp, kp, vp, block_mask=bm, score_mod=smod, scaling=scaling,
                compile_mode=None,  # fast default mode — parity is dtype-independent
            )[:, :, :L]
            outs[name] = out.float()

        ref_diff = (outs["int64"] - ref.float()).abs().max().item()
        narrow = {n: (outs[n] - outs["int64"]).abs().max().item()
                  for n in DTYPES if n != "int64"}
        ok = (all(v == 0.0 for v in narrow.values()) and ref_diff < 5e-2)
        narrow_str = "  ".join(f"{n}-vs-int64={v:.0e}" for n, v in narrow.items())
        print(f"  k={k_hop} L={meta['seq_len']:>4}  "
              f"int64-vs-dense max|Δ|={ref_diff:.2e}  {narrow_str}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("parity FAILED — narrowing node_ids changed the result")
        del outs, ai
        gc.collect(); torch.cuda.empty_cache()
    print()


# ── timing sweep ──────────────────────────────────────────────────────────────

def _run_subprocess(N, tpn, k_hop, block, name, compile_mode, n_warmup, n_iter,
                    timeout, seed=0, cache_dir=None) -> dict:
    """One ``bench_isolation`` flex run in a fresh subprocess (OOM/compile-cache
    isolation — running many compiled-flex shapes in one process accumulates
    autotune workspace and OOMs, exactly as ``run_sweep`` found).

    ``cache_dir`` (set per repeat under ``--fresh-autotune``) points
    ``TORCHINDUCTOR_CACHE_DIR`` at a unique dir so each run *re-autotunes from
    scratch* — otherwise repeats hit the on-disk cache and reproduce the same
    kernel pick bit-for-bit, hiding the autotuner's config-selection variance
    (the suspected source of the k=2 small-N backward swing)."""
    cmd = [
        sys.executable, "-m", "src.models.flex_attn.bench_isolation",
        "--method", "flex", "--n-nodes", str(N), "--tokens-per-node", str(tpn),
        "--prompt-tokens", "128", "--k-hop", str(k_hop), "--ordering", "rcm",
        "--magnetic-m", "32", "--block-size", str(block), "--seed", str(seed),
        "--compile-mode", compile_mode or "default",
        "--node-id-dtype", name, "--n-warmup", str(n_warmup), "--n-iter", str(n_iter),
    ]
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if cache_dir is not None:
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TIMEOUT", "timing_ms": {}, "shape": {}, "density": {},
                "run": {"node_id_dtype": name}}
    out = proc.stdout
    b, e = out.find(RESULT_BEGIN), out.find(RESULT_END)
    if b >= 0 and e > b:
        return json.loads(out[b + len(RESULT_BEGIN):e])
    detail = (proc.stderr or proc.stdout)[-300:]
    err = "OOM" if "out of memory" in detail.lower() else "CRASH"
    return {"ok": False, "error": err, "error_detail": detail, "timing_ms": {},
            "shape": {}, "density": {}, "run": {"node_id_dtype": name}}


def run_timing(out_dir: str, compile_mode: Optional[str], n_warmup: int, n_iter: int,
               configs, timeout: int, repeats: int = 1, fresh_autotune: bool = False,
               vary_seed: bool = False) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "node_id_dtype.jsonl")
    cache_root = os.path.join(out_dir, "_inductor_cache") if fresh_autotune else None
    records = []
    with open(jsonl, "w") as fh:
        for (label, N, tpn, k_hop, block) in configs:
            for name in DTYPES:
                for rep in range(repeats):
                    # Fresh inductor cache per repeat → independent autotune pick;
                    # optional fresh input seed → independent graph draw.
                    cache_dir = (os.path.join(cache_root, f"{label}_{name}_r{rep}")
                                 if fresh_autotune else None)
                    seed = rep if vary_seed else 0
                    t0 = time.perf_counter()
                    res = _run_subprocess(N, tpn, k_hop, block, name, compile_mode,
                                          n_warmup, n_iter, timeout, seed=seed,
                                          cache_dir=cache_dir)
                    res["config_label"] = label
                    res["repeat"] = rep
                    res["seed_used"] = seed
                    res["wall_s"] = time.perf_counter() - t0
                    fh.write(json.dumps(res) + "\n"); fh.flush()
                    records.append(res)
                    tg = res.get("timing_ms", {})
                    status = res.get("error") or (
                        f"fwd {tg.get('fwd', float('nan')):.2f}  "
                        f"bwd {tg.get('bwd', float('nan')):.2f}  "
                        f"fwd+bwd {tg.get('fwd_bwd', float('nan')):.2f} ms")
                    rep_tag = f" r{rep}" if repeats > 1 else ""
                    print(f"  [{label:>14}]{rep_tag} {name:<5} "
                          f"L={res.get('shape',{}).get('seq_len','?'):>5} "
                          f"blkSp={res.get('density',{}).get('block_sparsity', float('nan')):.2f}  "
                          f"{status}  ({res['wall_s']:.0f}s)", flush=True)
    print(f"\nwrote {jsonl}")
    return records


# ── pivot table (fwd / bwd / fwd+bwd per dtype, %Δ vs int64) ───────────────────

def _stats(vals):
    """(mean, std, min, n) over the non-None timing samples for one cell."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    std = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5 if n > 1 else 0.0
    return mean, std, min(xs), n


def summarize(records: list[dict]) -> str:
    # group: label -> dtype -> list[record] (one per repeat)
    by_cfg: dict = {}
    for r in records:
        name = r.get("run", {}).get("node_id_dtype", "?")
        by_cfg.setdefault(r["config_label"], {}).setdefault(name, []).append(r)

    n_rep = max((len(v) for byname in by_cfg.values() for v in byname.values()), default=1)
    lines = []
    if n_rep > 1:
        lines.append(f"\n_repeats={n_rep}; cells are mean ± std (min) over independent "
                     f"runs; %Δ is on the means. A real effect should clear the std band._\n")
    for metric in ("fwd", "bwd", "fwd_bwd"):
        lines.append(f"\n### {metric} (ms, and %Δ vs int64)\n")
        lines.append("| config | L | blkSp | " + " | ".join(DTYPES) + " |")
        lines.append("|---|---|---|" + "---|" * len(DTYPES))
        for label, byname in by_cfg.items():
            recs64 = byname.get("int64", [])
            base = _stats([x.get("timing_ms", {}).get(metric) for x in recs64])
            ref = recs64[0] if recs64 else {}
            L = ref.get("shape", {}).get("seq_len", "?")
            blk = ref.get("density", {}).get("block_sparsity", float("nan"))
            cells = []
            for name in DTYPES:
                st = _stats([x.get("timing_ms", {}).get(metric) for x in byname.get(name, [])])
                if st is None:
                    cells.append("—"); continue
                mean, std, mn, n = st
                body = f"{mean:.2f}" + (f" ± {std:.2f} ({mn:.2f})" if n > 1 else "")
                if name != "int64" and base:
                    body += f" ({(mean - base[0]) / base[0] * 100:+.1f}%)"
                cells.append(body)
            lines.append(f"| {label} | {L} | {blk:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="src/models/flex_attn/results_h100_nodeid")
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs",
                   help="'default' for inductor's fast heuristics (forces 128-blocks)")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke: default compile mode, 128-blocks, small iters")
    p.add_argument("--no-parity", action="store_true")
    # ── config grid (default: the original 3 hand-picked configs) ──
    p.add_argument("--nodes", type=int, nargs="+",
                   help="N axis; with --tpn/--k-hops forms a cartesian-product grid "
                        "(superset of the default 3 configs). Block size auto-gated by #6.")
    p.add_argument("--tpn", type=int, nargs="+", help="tokens-per-node axis")
    p.add_argument("--k-hops", type=int, nargs="+", help="K-hop axis (0 = dense gather)")
    p.add_argument("--max-tokens", type=int, default=100000,
                   help="skip grid cells whose est. token count (≈ N·tpn) exceeds this")
    # ── robustness knobs (attack the autotune-variance doubt) ──
    p.add_argument("--repeats", type=int, default=1,
                   help="runs per (config, dtype); >1 reports mean ± std (min)")
    p.add_argument("--fresh-autotune", action="store_true",
                   help="unique TORCHINDUCTOR_CACHE_DIR per repeat so each run "
                        "re-autotunes independently — samples the autotuner's "
                        "config-selection variance (else repeats hit cache, identical)")
    p.add_argument("--vary-seed", action="store_true",
                   help="use a different input seed per repeat (independent graph draw)")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-subprocess timeout (sized for the ~320 s autotune compile)")
    p.add_argument("--summarize", help="path to a saved node_id_dtype.jsonl; print table and exit")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    if a.summarize:
        recs = [json.loads(l) for l in open(a.summarize) if l.strip()]
        print(summarize(recs))
        return 0

    assert torch.cuda.is_available(), "flex_attention needs CUDA"
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    if not a.no_parity:
        check_parity(dev)

    compile_mode = None if (a.quick or a.compile_mode == "default") else a.compile_mode
    # Grid from --nodes/--tpn/--k-hops if any given, else the default 3 configs.
    if a.nodes or a.tpn or a.k_hops:
        nodes = a.nodes or [512, 2048]
        tpn = a.tpn or [8, 32]
        k_hops = a.k_hops or [0, 2]
        configs = build_grid(nodes, tpn, k_hops, a.max_tokens)
    else:
        configs = CONFIGS
    if a.quick:
        configs = [(lbl, N, tpn, k, 128) for (lbl, N, tpn, k, _b) in configs]
    n_warmup = 2 if a.quick else a.n_warmup
    n_iter = 5 if a.quick else a.n_iter

    print(f"── timing (compile_mode={compile_mode or 'default'}, warmup={n_warmup}, "
          f"iter={n_iter}, repeats={a.repeats}, fresh_autotune={a.fresh_autotune}) ──\n"
          f"   {len(configs)} configs × {len(DTYPES)} dtypes × {a.repeats} repeats "
          f"= {len(configs) * len(DTYPES) * a.repeats} runs")
    records = run_timing(a.out_dir, compile_mode, n_warmup, n_iter, configs, a.timeout,
                         repeats=a.repeats, fresh_autotune=a.fresh_autotune,
                         vary_seed=a.vary_seed)

    table = summarize(records)
    print(table)
    md = os.path.join(a.out_dir, "node_id_dtype.md")
    with open(md, "w") as fh:
        fh.write(f"# Experiment #10 — node_id dtype ({' / '.join(DTYPES)})\n\n"
                 f"int16 excluded: torch requires index tensors to be "
                 f"long/int/byte/bool, and node_ids is used as an index.\n\n"
                 f"device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
                 f"compile_mode={compile_mode or 'default'}\n")
        fh.write(table)
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
