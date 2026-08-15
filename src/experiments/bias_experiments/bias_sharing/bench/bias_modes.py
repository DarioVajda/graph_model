"""
Split the flex bias machinery into *gather* and *atomic scatter*, at our shapes.

§6.4 showed that the cost `G` cannot touch scales as O(L²) at fixed `N` — it is
per attention **score**, not per bias **compute**. That rules out SPD compute, but
it does not say which per-score op is expensive, because `bias_sharing`'s `nobias`
arm removes the `score_mod` entirely (`make_score_mod` returns None when there is
no bias, `src/models/flex_kernel.py:298`) and so bundles three things: the forward
gather, the backward atomic scatter-add into `node_bias`, and the mere existence of
a non-trivial `score_mod`.

`src/models/flex_attn/bench_isolation.py` already separates them via
`--bias-mode`:

    none    score_mod=None            the bare masked kernel
    frozen  bias requires_grad=False  + forward gather, NO scatter
    full    bias requires_grad=True   + backward atomic scatter-add

so ``frozen − none`` is the gather and ``full − frozen`` is the scatter. That
decomposition was published at **N=512 on a single attention layer**; this module
re-runs it across `bias_sharing`'s node counts to check the split still holds at
N=4096, where the `(B,H,N,N)` bias table is 1.07 GB and spills L2.

Each cell runs in a **fresh subprocess**, which is what `flex_attn/run_sweep.py`
does and why: "one configuration's OOM cannot poison the others (the only robust
way to guarantee a clean VRAM slate)" (`run_sweep.py:5`). Driving the grid
in-process instead — the obvious thing, and what this module did first — produced
`flex[none]` forward times jumping 0.71 ms at N=1024 to 90.6 ms at N=2048 for 4×
the work, and OOMs at N≥2048 where `speed.py` runs a full 16-layer model on the
same card. Fragmentation accumulated across cells. Do not "optimize" this back
into a single process to share the flex compile cache: each cell autotunes its own
shape anyway, and warm-up is excluded from the timed region.

    python3 -m src.experiments.bias_experiments.bias_sharing.bench.bias_modes --out <path>.jsonl

Why the numbers here are NOT directly comparable to §6.4's: this is one attention
layer with the bias detached to a leaf, so it excludes the bias modules' own
backprop, the other 15 layers, and gradient checkpointing. It measures the *ratio*
between gather and scatter, which is what §6.4 needs; absolute ms belong to
`speed.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# WebQSP's measured profile (bench/webqsp_token_stats.json): 2.99 tokens/node.
# Contention on the scatter scales as tokens-per-node², so this is the one input
# that must match the real data rather than being a round number.
TOKENS_PER_NODE = 3
MAGNETIC_M = 128          # 002_webqsp_g_sweep's setting
K_HOP = 0                 # all three sweeps
NODES = (512, 1024, 2048, 4096)
MODES = ("none", "frozen", "full")

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "results", "bench",
                           "bias_modes.jsonl")


RESULT_BEGIN = "===RESULT_JSON_BEGIN==="      # must match bench_isolation's sentinels
RESULT_END = "===RESULT_JSON_END==="
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))


def _one_cell(n, method, mode, compile_mode, n_warmup, n_iter, node_id_dtype,
              timeout=1800):
    """Run a single bench_isolation config in its own process; return its JSON."""
    cmd = [
        sys.executable, "-m", "src.models.flex_attn.bench_isolation",
        "--method", method,
        "--n-nodes", str(n),
        "--tokens-per-node", str(TOKENS_PER_NODE),
        "--prompt-tokens", "128",
        "--k-hop", str(K_HOP),
        "--ordering", "rcm",
        "--magnetic-m", str(MAGNETIC_M),
        "--batch-size", "1",
        "--num-heads", "32", "--num-kv-heads", "8", "--head-dim", "64",
        "--block-size", "128",
        "--bias-mode", mode or "full",
        "--compile-mode", compile_mode,
        # int64 matches PRODUCTION, not the flex_attn package's tuned default: the
        # collator emits long node_ids (src/utils/text_graph_collator_v2.py:11) and
        # src/models/flex_kernel.py has no cast, so finding #10's free int32 win was
        # never ported to the model path. int32 here would understate what training
        # actually pays.
        "--node-id-dtype", node_id_dtype,
        "--n-warmup", str(n_warmup), "--n-iter", str(n_iter),
        "--seed", "0",
    ]
    env = dict(os.environ,
               PYTHONPATH=_REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TIMEOUT", "method": method, "timing_ms": {}}

    out = proc.stdout
    b, e = out.find(RESULT_BEGIN), out.find(RESULT_END)
    if b >= 0 and e > b:
        return json.loads(out[b + len(RESULT_BEGIN):e])
    detail = (proc.stderr or out)[-300:]
    return {"ok": False, "method": method, "timing_ms": {},
            "error": "OOM" if "out of memory" in detail.lower() else "CRASH",
            "error_detail": detail}


def run_grid(nodes, modes, compile_mode, n_warmup, n_iter, out_path,
             node_id_dtype="int64"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = []
    with open(out_path, "a") as fh:
        for n in nodes:
            for method, mode in [("flex", m) for m in modes] + [("flash_nc", None)]:
                t0 = time.perf_counter()
                res = _one_cell(n, method, mode, compile_mode, n_warmup, n_iter,
                                node_id_dtype)
                res["n_nodes_target"] = n
                res["bias_mode"] = mode
                res["node_id_dtype"] = node_id_dtype
                res["wall_s"] = time.perf_counter() - t0
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                written.append(res)
                t = res.get("timing_ms", {})
                tag = f"{method}[{mode}]" if mode else method
                print(f"  N={n:5d} {tag:16s} fwd {t.get('fwd', float('nan')):8.1f} "
                      f"bwd {t.get('bwd', float('nan')):8.1f} ms"
                      + ("" if res.get("ok") else f"  FAILED {res.get('error')}"),
                      flush=True)
    return written


def report(path: str) -> None:
    """gather = frozen − none; scatter = full − frozen, on the directly-timed bwd."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {}
    for r in rows:
        if r.get("ok"):
            by[(r["n_nodes_target"], r.get("bias_mode") or r["method"])] = r

    def t(n, key, field):
        r = by.get((n, key))
        return r["timing_ms"][field] if r else None

    nodes = sorted({n for n, _ in by})
    print("\n## flex bias machinery: gather vs atomic scatter (one attention layer)\n")
    print(f'{"N":>6} {"L":>7} {"kernel":>9} {"gather":>9} {"scatter":>9} '
          f'{"bias tot":>9} {"bias %":>7} {"scatter % of bias":>18}')
    for n in nodes:
        vals = {m: (t(n, m, "fwd"), t(n, m, "bwd")) for m in MODES}
        if any(v[0] is None for v in vals.values()):
            continue
        tot = {m: vals[m][0] + vals[m][1] for m in MODES}
        kernel, gather, scatter = tot["none"], tot["frozen"] - tot["none"], \
            tot["full"] - tot["frozen"]
        bias = gather + scatter
        L = by[(n, "full")]["shape"]["seq_len"]
        print(f'{n:>6} {L:>7} {kernel:>9.1f} {gather:>9.1f} {scatter:>9.1f} '
              f'{bias:>9.1f} {100 * bias / tot["full"]:>6.0f}% '
              f'{100 * scatter / bias:>17.0f}%')
    print("\n  fwd+bwd ms. gather = frozen − none; scatter = full − frozen.")
    print("  Contention on the scatter scales as tokens-per-node² — every token pair")
    print("  of a node pair adds into ONE address in the (B,H,N,N) bias grad.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, nargs="+", default=list(NODES))
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--node-id-dtype", default="int64",
                   choices=["int64", "int32"],
                   help="int64 matches production; int32 prices finding #10")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)

    if not a.report_only:
        run_grid(a.nodes, a.modes, a.compile_mode, a.n_warmup, a.n_iter, a.out,
                 node_id_dtype=a.node_id_dtype)
    report(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
