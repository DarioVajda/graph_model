"""
Sweep orchestrator.

Runs the 2D (n_nodes × tokens_per_node) sweep, crossed with node-ordering,
K-hop, and method, driving each ``(method, config)`` in a **fresh subprocess** so
that one configuration's OOM cannot poison the others (the only robust way to
guarantee a clean VRAM slate). Configurations whose token budget exceeds
``--max-tokens`` (~100k) are pre-skipped. K-independent methods
(flash/flash_nc/eager — see ``_K_INDEPENDENT``) run once per
(nodes, tpn, ordering) rather than once per K, so extra ``--k-hops`` values
only add flex runs; the markdown table folds K into ``flex-{K}`` columns.

Results are appended as JSON lines to ``--out``; a compact table is printed at
the end. OOM and crashes are recorded as rows with ``ok=False`` rather than
aborting the sweep.

Use ``--kind both`` to run the isolated-attention and full-model sweeps back to
back (writes ``isolation.{jsonl,md}`` and ``full_model.{jsonl,md}``).

Extra mode:
  * ``--recompile-probe`` — a single-process experiment that runs flex across a
    sequence of shapes (varying L, N) with bucketing on/off and reports the
    torch._dynamo recompilation count. This is what actually answers "does
    global L/N bucketing keep torch.compile stable across heterogeneous graphs".
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

# Must match bench_isolation.RESULT_BEGIN/END (kept local so the orchestrator
# doesn't import torch just to read two sentinel strings).
RESULT_BEGIN = "===RESULT_JSON_BEGIN==="
RESULT_END = "===RESULT_JSON_END==="

# Sweep axes (targets; sizes are jittered around these in inputs.py).
GRID_NODES = [8, 32, 128, 512, 2048]
GRID_TPN = [2, 8, 32, 128, 512]


@dataclass
class Cell:
    n_nodes: int
    tokens_per_node: int
    k_hop: int
    ordering: str

    def est_tokens(self, batch_size: int, prompt_tokens: int, jitter: float) -> int:
        per = self.n_nodes * self.tokens_per_node + prompt_tokens
        return int(round(per * batch_size * (1.0 + jitter)))


def build_cells(nodes, tpn, k_hops, orderings, batch_size, prompt_tokens,
                jitter, max_tokens) -> list[Cell]:
    cells = []
    for n, t, k, o in itertools.product(nodes, tpn, k_hops, orderings):
        c = Cell(n, t, k, o)
        if c.est_tokens(batch_size, prompt_tokens, jitter) > max_tokens:
            continue
        cells.append(c)
    return cells


# ── subprocess driver ─────────────────────────────────────────────────────────

def run_subprocess(kind: str, method: str, cell: Cell, *, batch_size: int,
                   prompt_tokens: int, magnetic_m: int, block_size: int,
                   seed: int, timeout: int, bias_mode: str = "full",
                   compile_mode: str = "default") -> dict:
    module = "src.models.flex_attn.bench_full_model" if kind == "full_model" \
        else "src.models.flex_attn.bench_isolation"
    cmd = [
        sys.executable, "-m", module,
        "--method", method,
        "--n-nodes", str(cell.n_nodes),
        "--tokens-per-node", str(cell.tokens_per_node),
        "--prompt-tokens", str(prompt_tokens),
        "--k-hop", str(cell.k_hop),
        "--ordering", cell.ordering,
        "--magnetic-m", str(magnetic_m),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]
    if kind != "full_model":
        bs = block_size if isinstance(block_size, (tuple, list)) else (block_size,)
        cmd += ["--block-size", *[str(x) for x in bs],
                "--bias-mode", bias_mode, "--compile-mode", compile_mode]
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _stub(kind, method, cell, batch_size, "TIMEOUT", bias_mode, block_size)

    # The bench prints pretty JSON between BEGIN/END sentinels.
    out = proc.stdout
    b, e = out.find(RESULT_BEGIN), out.find(RESULT_END)
    if b >= 0 and e > b:
        return json.loads(out[b + len(RESULT_BEGIN):e])
    # No JSON → hard crash (often a non-recoverable CUDA OOM that killed the proc).
    detail = (proc.stderr or proc.stdout)[-300:]
    err = "OOM" if "out of memory" in detail.lower() else "CRASH"
    r = _stub(kind, method, cell, batch_size, err, bias_mode, block_size)
    r["error_detail"] = detail
    return r


def _stub(kind, method, cell, batch_size, err, bias_mode="full",
          block_size=128) -> dict:
    return {
        "kind": kind, "method": method, "ok": False, "error": err,
        "spec": {"n_nodes": cell.n_nodes, "tokens_per_node": cell.tokens_per_node,
                 "k_hop": cell.k_hop, "ordering": cell.ordering},
        "run": {"batch_size": batch_size, "bias_mode": bias_mode,
                "block_size": list(block_size) if isinstance(block_size, tuple) else block_size},
        "shape": {}, "density": {}, "timing_ms": {}, "memory_mb": {},
    }


# ── recompilation probe (single process) ─────────────────────────────────────

def recompile_probe(bucketed: bool, block_size: int = 128) -> dict:
    """Run flex across a sequence of shapes; report torch._dynamo recompiles.

    With ``bucketed=True`` we round L and N up to a coarse ladder before building
    the inputs, so distinct compiled shapes are few; with ``bucketed=False`` we
    feed the raw jittered shapes. The recompile count is the headline number.
    """
    import torch
    import torch.nn.functional as F
    import torch._dynamo as dynamo
    from src.models.flex_attn.inputs import GraphSpec, make_attention_inputs
    from src.models.flex_attn import flex_core

    dev = torch.device("cuda")
    dynamo.reset()
    dynamo.utils.counters.clear()          # the 'unique_graphs' counter is cumulative
    dynamo.config.cache_size_limit = 256

    # A spread of shapes a real run would hit (varying nodes/tokens → varying L,N).
    raw = [(32, 8), (40, 8), (33, 9), (128, 16), (140, 15), (130, 17),
           (512, 8), (520, 9), (500, 8)]
    fa = flex_core.get_flex_attention(dynamic=False)
    for i, (n, t) in enumerate(raw):
        spec = GraphSpec(n_nodes=n, tokens_per_node=t, k_hop=2, ordering="rcm",
                         jitter=0.15, seed=i, magnetic_m=16)
        ai, _ = make_attention_inputs(spec, 1, 32, 8, 64, dev)
        q, k, v = ai["query"].detach(), ai["key"].detach(), ai["value"].detach()
        node_ids, pm, nb = ai["node_ids"], ai["pad_mask"], ai["node_bias"]
        N = nb.shape[-1]

        if bucketed:
            # Bucket BOTH L (to the coarse pow2-with-midpoints ladder) and N
            # (same ladder at step 128): the flex kernel guards on q/k/v length
            # AND on the captured node_bias (B,H,N,N) / node_ids shapes, so
            # both must be stabilized. This is the ladder training should use.
            Lb = flex_core.bucket_len(q.shape[2], block_size)
            Nb = flex_core.bucket_len(N, 128)
            q, k, v, node_ids, pm, _, _ = flex_core.pad_to_block(q, k, v, node_ids, pm, block_size)
            if Lb != q.shape[2]:
                q, k, v, node_ids, pm, _, _ = flex_core.pad_to_block(q, k, v, node_ids, pm, Lb)
            if Nb != N:
                nb = F.pad(nb, (0, Nb - N, 0, Nb - N))     # pad node_bias to bucketed N
            khm = ai["k_hop_mask"]
            if khm is not None and Nb != N:
                khm = F.pad(khm, (0, Nb - N, 0, Nb - N))
        else:
            khm = ai["k_hop_mask"]

        Lq = q.shape[2]
        bm = flex_core.build_block_mask(node_ids, ai["prompt_node"], pm, khm,
                                        spec.k_hop, Lq, Lq, block_size=block_size, device=dev)
        smod = flex_core.make_score_mod(nb, node_ids)
        with torch.no_grad():
            fa(q, k, v, block_mask=bm, score_mod=smod, scale=64 ** -0.5, enable_gqa=True)
        torch.cuda.synchronize()

    n_compiles = dynamo.utils.counters["stats"].get("unique_graphs", -1)
    return {"bucketed": bucketed, "block_size": block_size, "n_shapes": len(raw),
            "unique_graph_compiles": n_compiles}


# ── summary table ─────────────────────────────────────────────────────────────

_METHOD_ORDER = ["flash", "flash_nc", "eager", "flex"]

# Methods only the isolation bench implements; silently inflating the
# full-model grid with them would just produce CRASH rows.
_ISOLATION_ONLY = {"flash_nc"}

# Methods whose cost cannot depend on the K-hop value: flash/flash_nc never see
# the graph at all, and eager's dense SDPA does identical work regardless of
# mask content (verified on the A100 sweep: ≤0.5% timing delta and identical
# peak memory between k=0 and k=2). The sweep runs these once per
# (nodes, tpn, ordering) — at the first swept K — and the markdown table gives
# them a single column, while K-dependent methods (flex) get one column group
# per K, labelled ``<method>-<K>``.
_K_INDEPENDENT = {"flash", "flash_nc", "eager"}

# Methods that carry the node bias, i.e. the only ones the ``--bias-modes``
# backward-decomposition axis (isolation kind only) applies to. flash/flash_nc
# have no bias and run only at the first mode. Non-"full" modes are labelled
# ``<method>[<mode>]`` in the table.
_BIAS_SENSITIVE = {"flex", "eager"}

# Methods the ``--block-sizes`` axis (#6, isolation kind only) applies to —
# only flex consumes the BlockMask. Non-default block sizes are labelled
# ``<method>@<size>`` in the table (e.g. ``flex@64``, ``flex@128x64``).
_BLOCK_SENSITIVE = {"flex"}


def _parse_block_size(s: str):
    """'128' -> 128; '128x64' (or '128,64') -> (128, 64) = (Q_BLOCK, KV_BLOCK)."""
    parts = [int(p) for p in s.replace(",", "x").split("x")]
    return parts[0] if len(parts) == 1 else tuple(parts)


def _bs_label(bs) -> str:
    return "x".join(str(x) for x in bs) if isinstance(bs, (tuple, list)) else str(bs)


def _first(rows, *path, default=None):
    """First non-None nested value across a list of result dicts."""
    for r in rows:
        cur = r
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        if cur not in (None, {}):
            return cur
    return default


def render_markdown(rows: list[dict], kind: str) -> str:
    """Pivot the flat JSONL records into two per-config × method markdown
    tables: **latency** (fwd+bwd ms) and **peak memory** (GB), over identical
    rows.

    The JSONL file stays the source of truth (full detail, one record/line);
    this is the human-readable cross-tab. One row per ``(nodes, tpn,
    ordering)``; K is folded into the columns:

      * K-independent methods (``flash``/``flash_nc``/``eager``) get a single
        column — their cost doesn't depend on K (legacy per-K duplicates
        collapse to the first row seen).
      * K-dependent methods (``flex``) get one column per swept K, labelled
        ``<method>-<K>``, preceded in the latency table by that K's mask
        sparsity (``tokSp-<K>`` / ``blkSp-<K>``).

    ``OOM`` / error tags appear in the latency table; the memory table shows
    ``—`` for failed runs.
    """
    from collections import OrderedDict
    import datetime as _dt

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    densities: dict = {}                  # (rowkey, k) -> density dict
    dep_ks, all_ks = set(), set()
    for r in rows:
        sp = r.get("spec", {})
        key = (sp.get("n_nodes"), sp.get("tokens_per_node"), sp.get("ordering"))
        k = sp.get("k_hop")
        all_ks.add(k)
        m = r["method"]
        run = r.get("run") or {}
        mode = run.get("bias_mode")
        if mode and mode != "full":
            m = f"{m}[{mode}]"            # bias-decomposition variants (#3)
        bs = run.get("block_size", 128)
        if bs not in (None, 128) and r["method"] in _BLOCK_SENSITIVE:
            m = f"{m}@{_bs_label(bs)}"    # block-size variants (#6)
        if m in _K_INDEPENDENT:
            col = m
        else:
            col = (m, k)
            dep_ks.add(k)
        groups.setdefault(key, {}).setdefault(col, r)   # keep-first on duplicates
        if r.get("density") and (key, k) not in densities:
            densities[(key, k)] = r["density"]

    # Per-K column groups come from the K-dependent methods; if none were swept,
    # fall back to all seen K values so the sparsity columns still render.
    ks = sorted(dep_ks or all_ks, key=lambda v: -1 if v is None else v)

    indep = [m for m in _METHOD_ORDER
             if m in _K_INDEPENDENT and any(m in g for g in groups.values())]
    dep_present = {c[0] for g in groups.values() for c in g if isinstance(c, tuple)}
    dep = [m for m in _METHOD_ORDER if m in dep_present]
    dep += sorted(m for m in dep_present if m not in dep)

    def ms_cell(r: dict | None) -> str:
        if r is None:
            return "—"
        if not r.get("ok"):
            return f"**{r.get('error') or 'fail'}**"
        fb = r.get("timing_ms", {}).get("fwd_bwd")
        return f"{fb:.1f}" if fb is not None else "?"

    def gb_cell(r: dict | None) -> str:
        if r is None or not r.get("ok"):
            return "—"
        gb = r.get("memory_mb", {}).get("peak_fwd_bwd")
        return f"{gb/1024:.2f}" if gb is not None else "?"

    methods_str = ", ".join(indep + [f"{m}-{{{','.join(str(k) for k in ks)}}}" for m in dep])
    # Header + legend
    out = [
        f"# FlexAttention sweep — `{kind}`",
        "",
        f"_Generated {_dt.datetime.now():%Y-%m-%d %H:%M}. {len(groups)} configs, "
        f"methods: {methods_str}._",
        "",
        "**Legend.** Two tables over the same configs: **latency** (median "
        "forward+backward, milliseconds; `OOM` and error tags show up here, in "
        "bold) and **peak memory** (during forward+backward, GB; `—` for failed "
        "runs). `flash`/`flash_nc`/`eager` are K-independent (one column; run "
        "once per config); `flex-{K}` = flex with the K-hop-{K} mask. "
        "`tokSp-{K}` / `blkSp-{K}` = token-level / block-level sparsity of that "
        "mask (fraction of attention masked out; block-level is what flex "
        "actually skips — latency table only). `L` = packed sequence length.",
    ]

    def _sp(v):
        return f"{v:.2f}" if v is not None else "—"

    row_keys = sorted(groups, key=lambda t: (t[0] if t[0] is not None else -1,
                                             t[1] if t[1] is not None else -1,
                                             str(t[2])))

    def _table(title: str, cell_fn, with_sparsity: bool) -> list[str]:
        cols = ["nodes", "tpn", "order", "L"] + list(indep)
        for k in ks:
            if with_sparsity:
                cols += [f"tokSp-{k}", f"blkSp-{k}"]
            cols += [f"{m}-{k}" for m in dep]
        align = ["--:", "--:", ":--", "--:"] + ["--:"] * (len(cols) - 4)
        lines = ["", f"## {title}", "",
                 "| " + " | ".join(cols) + " |",
                 "| " + " | ".join(align) + " |"]
        for key in row_keys:
            n, t, order = key
            g = groups[key]
            L = _first(list(g.values()), "shape", "seq_len", default="—")
            cells = [str(n), str(t), str(order), str(L)]
            cells += [cell_fn(g.get(m)) for m in indep]
            for k in ks:
                if with_sparsity:
                    den = densities.get((key, k), {})
                    cells += [_sp(den.get("element_sparsity")),
                              _sp(den.get("block_sparsity"))]
                cells += [cell_fn(g.get((m, k))) for m in dep]
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    out += _table("Latency — forward+backward (ms)", ms_cell, with_sparsity=True)
    out += _table("Peak memory — forward+backward (GB)", gb_cell, with_sparsity=False)
    out.append("")
    return "\n".join(out)


def write_markdown(rows: list[dict], path: str, kind: str) -> None:
    with open(path, "w") as fh:
        fh.write(render_markdown(rows, kind))


def load_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── main ──────────────────────────────────────────────────────────────────────

def _run_kind(kind: str, cells: list[Cell], args) -> None:
    """Run the full method × config grid for one kind, writing {kind}.jsonl/.md."""
    methods = list(args.methods)
    if kind == "full_model":
        skipped = [m for m in methods if m in _ISOLATION_ONLY]
        if skipped:
            print(f"(full_model: skipping isolation-only methods: {', '.join(skipped)})")
            methods = [m for m in methods if m not in _ISOLATION_ONLY]
    # K-independent methods run once per (nodes, tpn, ordering) — at the first
    # swept K — instead of once per K value; re-running them with a new K
    # repeats the identical computation. The --bias-modes and --block-sizes
    # axes (isolation only) apply only to the methods they can affect.
    bias_modes = args.bias_modes if kind == "isolation" else ["full"]
    block_sizes = ([_parse_block_size(s) for s in args.block_sizes]
                   if kind == "isolation" else [128])
    runs, seen_base, n_dedup = [], set(), 0
    for cell in cells:
        base = (cell.n_nodes, cell.tokens_per_node, cell.ordering)
        first_k = base not in seen_base
        seen_base.add(base)
        for method in methods:
            modes = bias_modes if method in _BIAS_SENSITIVE else ["full"]
            sizes = block_sizes if method in _BLOCK_SENSITIVE else block_sizes[:1]
            for mode in modes:
                for bs in sizes:
                    if method in _K_INDEPENDENT and not first_k:
                        n_dedup += 1
                        continue
                    runs.append((cell, method, mode, bs))

    jsonl_path = os.path.join(args.out_dir, f"{kind}.jsonl")
    md_path = os.path.join(args.out_dir, f"{kind}.md")
    total = len(runs)
    print(f"\n=== Sweep ({kind}): {len(cells)} cells × {len(methods)} methods "
          f"(bias modes: {', '.join(bias_modes)}) − {n_dedup} K-duplicate runs "
          f"= {total} runs → {jsonl_path} + {md_path} ===")

    rows, done, t_start = [], 0, time.time()
    with open(jsonl_path, "w") as fh:
        for cell, method, mode, bs in runs:
            done += 1
            r = run_subprocess(kind, method, cell, batch_size=args.batch_size,
                               prompt_tokens=args.prompt_tokens, magnetic_m=args.magnetic_m,
                               block_size=bs, seed=args.seed,
                               timeout=args.timeout, bias_mode=mode,
                               compile_mode=args.compile_mode)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            rows.append(r)
            tag = "ok" if r["ok"] else (r.get("error") or "fail")
            shp, tim, mem, den = (r.get("shape", {}), r.get("timing_ms", {}),
                                  r.get("memory_mb", {}), r.get("density", {}))
            fwd = tim.get("fwd")
            blk_sp = den.get("block_sparsity")
            mlabel = method if mode == "full" else f"{method}[{mode}]"
            if bs != 128:
                mlabel += f"@{_bs_label(bs)}"
            print(f"[{done:>3}/{total}] {kind:<10} n={cell.n_nodes:<4} t={cell.tokens_per_node:<3} "
                  f"k={cell.k_hop} {cell.ordering:<6} {mlabel:<12} -> {tag:<6} "
                  f"L={shp.get('seq_len','?')} "
                  f"blkSparse={f'{blk_sp:.2f}' if blk_sp is not None else '-':<5} "
                  f"fwd={f'{fwd:.2f}ms' if fwd is not None else '-':<8} "
                  f"peakFB={mem.get('peak_fwd_bwd','-')}")
            # Refresh the markdown table as we go, so a partial/interrupted sweep
            # still leaves a readable table.
            write_markdown(rows, md_path, kind)
    print(f"Done ({kind}) in {time.time()-t_start:.0f}s. {sum(r['ok'] for r in rows)}/{total} ok. "
          f"→ {jsonl_path} + {md_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(_HERE, "results_h100"),
                   help="directory to write {kind}.jsonl (detail) and {kind}.md (table); "
                        "results dirs are per-GPU (results_a100 holds the archived A100 sweep)")
    p.add_argument("--kind", default="isolation",
                   choices=["isolation", "full_model", "both"],
                   help="'both' runs the isolated-attention and full-model sweeps back to back")
    p.add_argument("--methods", nargs="+", default=["flash", "eager", "flex"])
    p.add_argument("--bias-modes", nargs="+", default=["full"],
                   choices=["full", "frozen", "none"],
                   help="bias decomposition axis (#3, isolation kind only; applies "
                        "to flex/eager): full = gather + scatter-add grad, frozen = "
                        "gather only, none = bare masked kernel")
    p.add_argument("--compile-mode", default="max-autotune-no-cudagraphs",
                   help="torch.compile mode for the flex kernels (#4, isolation "
                        "kind only — the full-model bench uses flex_core's "
                        "default). The autotuned mode is the decided final "
                        "config; pass 'default' to iterate quickly")
    p.add_argument("--k-hops", nargs="+", type=int, default=[0, 2, 4])
    p.add_argument("--orderings", nargs="+", default=["rcm", "random"])
    p.add_argument("--nodes", nargs="+", type=int, default=GRID_NODES)
    p.add_argument("--tpn", nargs="+", type=int, default=GRID_TPN)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--magnetic-m", type=int, default=32)
    p.add_argument("--block-sizes", nargs="+", default=["128"],
                   help="BlockMask block-size axis (#6, isolation kind, flex "
                        "only): square ('64') or rectangular Q×KV ('128x64')")
    p.add_argument("--max-tokens", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-run subprocess timeout; sized for the ~320 s "
                        "max-autotune compile on top of the bench itself")
    p.add_argument("--recompile-probe", action="store_true")
    p.add_argument("--summarize", metavar="JSONL", default=None,
                   help="regenerate the markdown table (next to the file) from an existing JSONL")
    args = p.parse_args(argv)

    if args.summarize:
        rows = load_rows(args.summarize)
        kind = rows[0].get("kind", "sweep") if rows else "sweep"
        md_path = os.path.splitext(args.summarize)[0] + ".md"
        write_markdown(rows, md_path, kind)
        print(f"Wrote {md_path}")
        return 0

    if args.recompile_probe:
        print("== recompilation probe (single process) ==")
        for bucketed in (False, True):
            r = recompile_probe(bucketed, args.block_size)
            print(f"  bucketed={bucketed!s:>5}: unique_graph_compiles="
                  f"{r['unique_graph_compiles']} over {r['n_shapes']} shapes")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    cells = build_cells(args.nodes, args.tpn, args.k_hops, args.orderings,
                        args.batch_size, args.prompt_tokens, 0.2, args.max_tokens)
    kinds = ["isolation", "full_model"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        _run_kind(kind, cells, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
