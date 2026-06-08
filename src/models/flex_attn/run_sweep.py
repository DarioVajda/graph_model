"""
Sweep orchestrator.

Runs the 2D (n_nodes × tokens_per_node) sweep, crossed with node-ordering,
K-hop, and method, driving each ``(method, config)`` in a **fresh subprocess** so
that one configuration's OOM cannot poison the others (the only robust way to
guarantee a clean VRAM slate). Configurations whose token budget exceeds
``--max-tokens`` (~100k) are pre-skipped.

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
                   seed: int, timeout: int) -> dict:
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
        cmd += ["--block-size", str(block_size)]
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _stub(kind, method, cell, batch_size, "TIMEOUT")

    # The bench prints pretty JSON between BEGIN/END sentinels.
    out = proc.stdout
    b, e = out.find(RESULT_BEGIN), out.find(RESULT_END)
    if b >= 0 and e > b:
        return json.loads(out[b + len(RESULT_BEGIN):e])
    # No JSON → hard crash (often a non-recoverable CUDA OOM that killed the proc).
    detail = (proc.stderr or proc.stdout)[-300:]
    err = "OOM" if "out of memory" in detail.lower() else "CRASH"
    r = _stub(kind, method, cell, batch_size, err)
    r["error_detail"] = detail
    return r


def _stub(kind, method, cell, batch_size, err) -> dict:
    return {
        "kind": kind, "method": method, "ok": False, "error": err,
        "spec": {"n_nodes": cell.n_nodes, "tokens_per_node": cell.tokens_per_node,
                 "k_hop": cell.k_hop, "ordering": cell.ordering},
        "run": {"batch_size": batch_size},
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

    def bucket(x, step):
        return ((x + step - 1) // step) * step

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
            # Bucket BOTH L (to block_size) and N (to a coarse node grid): the
            # flex kernel guards on q/k/v length AND on the captured node_bias
            # (B,H,N,N) / node_ids shapes, so both must be stabilized.
            Lb = bucket(q.shape[2], block_size)
            Nb = bucket(N, 128)
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

_METHOD_ORDER = ["flash", "current", "flex"]


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
    """Pivot the flat JSONL records into a per-config × method markdown table.

    The JSONL file stays the source of truth (full detail, one record/line); this
    is the human-readable cross-tab. Each method contributes two explicit columns:
      * ``<method> ms`` — forward+backward latency (median); or ``OOM`` / an error tag.
      * ``<method> GB`` — peak GPU memory during forward+backward.
    Plus per-config ``tokSp`` (token-level) and ``blkSp`` (block-level) mask sparsity.
    """
    from collections import OrderedDict
    import datetime as _dt

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for r in rows:
        sp = r.get("spec", {})
        key = (sp.get("n_nodes"), sp.get("tokens_per_node"), sp.get("k_hop"), sp.get("ordering"))
        groups.setdefault(key, {})[r["method"]] = r

    present = [m for m in _METHOD_ORDER if any(m in g for g in groups.values())]
    present += [m for g in groups.values() for m in g if m not in present and m not in _METHOD_ORDER]

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

    # Header + legend
    out = [
        f"# FlexAttention sweep — `{kind}`",
        "",
        f"_Generated {_dt.datetime.now():%Y-%m-%d %H:%M}. {len(groups)} configs, "
        f"methods: {', '.join(present)}._",
        "",
        "**Legend.** `… ms` = forward+backward latency (median, milliseconds). "
        "`… GB` = peak GPU memory during forward+backward. "
        "`OOM` = ran out of memory (bold); other bold tags are errors. "
        "`tokSp` / `blkSp` = token-level / block-level mask sparsity "
        "(fraction of attention masked out; block-level is what flex actually skips). "
        "`L` = packed sequence length.",
        "",
    ]

    # Column spec
    cols = ["nodes", "tpn", "k", "order", "L", "tokSp", "blkSp"]
    for m in present:
        cols += [f"{m} ms", f"{m} GB"]
    align = ["--:", "--:", "--:", ":--", "--:", "--:", "--:"] + ["--:"] * (2 * len(present))
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(align) + " |")

    def _sp(v):
        return f"{v:.2f}" if v is not None else "—"

    for key in sorted(groups, key=lambda t: tuple((x if x is not None else -1) for x in t)):
        n, t, k, order = key
        g = groups[key]
        allr = list(g.values())
        L = _first(allr, "shape", "seq_len", default="—")
        cells = [str(n), str(t), str(k), str(order), str(L),
                 _sp(_first(allr, "density", "element_sparsity")),
                 _sp(_first(allr, "density", "block_sparsity"))]
        for m in present:
            cells += [ms_cell(g.get(m)), gb_cell(g.get(m))]
        out.append("| " + " | ".join(cells) + " |")

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
    jsonl_path = os.path.join(args.out_dir, f"{kind}.jsonl")
    md_path = os.path.join(args.out_dir, f"{kind}.md")
    total = len(cells) * len(args.methods)
    print(f"\n=== Sweep ({kind}): {len(cells)} cells × {len(args.methods)} methods "
          f"= {total} runs → {jsonl_path} + {md_path} ===")

    rows, done, t_start = [], 0, time.time()
    with open(jsonl_path, "w") as fh:
        for cell in cells:
            for method in args.methods:
                done += 1
                r = run_subprocess(kind, method, cell, batch_size=args.batch_size,
                                   prompt_tokens=args.prompt_tokens, magnetic_m=args.magnetic_m,
                                   block_size=args.block_size, seed=args.seed, timeout=args.timeout)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                rows.append(r)
                tag = "ok" if r["ok"] else (r.get("error") or "fail")
                shp, tim, mem, den = (r.get("shape", {}), r.get("timing_ms", {}),
                                      r.get("memory_mb", {}), r.get("density", {}))
                fwd = tim.get("fwd")
                blk_sp = den.get("block_sparsity")
                print(f"[{done:>3}/{total}] {kind:<10} n={cell.n_nodes:<4} t={cell.tokens_per_node:<3} "
                      f"k={cell.k_hop} {cell.ordering:<6} {method:<7} -> {tag:<6} "
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
    p.add_argument("--out-dir", default=os.path.join(_HERE, "results"),
                   help="directory to write {kind}.jsonl (detail) and {kind}.md (table)")
    p.add_argument("--kind", default="isolation",
                   choices=["isolation", "full_model", "both"],
                   help="'both' runs the isolated-attention and full-model sweeps back to back")
    p.add_argument("--methods", nargs="+", default=["flash", "current", "flex"])
    p.add_argument("--k-hops", nargs="+", type=int, default=[0, 2, 4])
    p.add_argument("--orderings", nargs="+", default=["rcm", "random"])
    p.add_argument("--nodes", nargs="+", type=int, default=GRID_NODES)
    p.add_argument("--tpn", nargs="+", type=int, default=GRID_TPN)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--magnetic-m", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=900)
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
