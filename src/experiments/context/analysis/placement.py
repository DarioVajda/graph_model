"""
README §3.4 — the placement probe: figures, tables and paired contrasts.

    python3 -m src.experiments.context.analysis.placement \\
        src/experiments/context/results/placement_flat \\
        src/experiments/context/results/placement_graph \\
        [--out-dir src/experiments/context/results/placement_analysis]

Reads the per-item rows (``<sweep dir>/items/*.jsonl``) that ``--mode placement``
writes, and nothing else. Every contrast is paired within a graph: the conditions
differ only in where the chain passages sit, so each graph is its own control.

Design decisions worth not re-litigating:

  * **Uncertainty is a cluster bootstrap over test graphs**, resampling a graph with
    all its seeds and conditions together. The three flat seeds score the same 200
    graphs, so treating (seed, graph) pairs as independent would overstate n by 3x.
    The exact McNemar p-values pool (seed, graph) pairs and are reported only as a
    secondary check; the bootstrap interval is the one the text quotes.
  * **Two metrics, one axis each.** ``code_acc`` is the §3.3 metric; the gold-code
    log-probability is its continuous shadow, which still moves at k=4 where the
    accuracy has hit the floor. They get separate figures, never a dual axis.
  * **The `random` condition is a reproduction check, not a data point.** Its
    accuracy must equal the §3.3 record for the same checkpoint and cell to the
    item; the script says so, or says loudly that it does not.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from ..placement import PLACEMENTS

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
# Categorical slots in fixed order: identity is the arm. Direction is the line style
# and the marker, so no series is told apart by color alone.
ARM_COLOR = {"flat": "#2a78d6", "graph": "#eb6834"}
ARM_LABEL = {"flat": "Flat LLM (LoRA)", "graph": "GTLM"}
DIR_STYLE = {"fwd": ("-", "o"), "rev": ("--", "s")}
DIR_LABEL = {"fwd": "chain in reading order", "rev": "chain reversed"}

N_BOOT = 2000


def _seed_label(checkpoint_path):
    run = checkpoint_path.rstrip("/").split("/")[-2]
    if "s2" in run:
        return "seed2"
    return "seed" + run.rsplit("seed", 1)[-1] if "seed" in run else run


def load_items(sweep_dirs):
    rows = []
    for d in sweep_dirs:
        for path in sorted(glob.glob(os.path.join(d, "items", "*.jsonl"))):
            with open(path) as fh:
                for line in fh:
                    r = json.loads(line)
                    r["seed"] = _seed_label(r["checkpoint_path"])
                    rows.append(r)
    return rows


def index(rows):
    """{(arm, cell, k): {seed: {condition: {item: row}}}}."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        key = (r["arm"], (r["n_nodes"], r["tokens_per_node"]), r["hops"])
        out[key][r["seed"]][r["condition"]][r["item"]] = r
    return out


def _matrix(by_seed, cond, field):
    """(seeds, items) array of ``field`` under ``cond``, items aligned across seeds."""
    seeds = sorted(by_seed)
    items = sorted(set.intersection(*(set(by_seed[s][cond]) for s in seeds)))
    return np.array([[by_seed[s][cond][i][field] for i in items] for s in seeds], float), items


def _boot(values, rng):
    """Mean and 95% cluster-bootstrap CI of a (seeds, items) array, clusters = items."""
    n = values.shape[1]
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = values[:, idx].mean(axis=(0, 2))
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _mcnemar(a, b):
    """Exact McNemar p for paired binary arrays (any shape)."""
    a, b = a.ravel().astype(bool), b.ravel().astype(bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return binomtest(n10, n01 + n10, 0.5).pvalue, n01, n10


def _cond(direction, p):
    return f"{direction}@{p:.2f}"


def _spread(by_seed, conds, field):
    """Best minus worst condition, the pair picked on the full-sample means.

    Picking the pair on the same data makes this an upper-biased descriptive number,
    so it is quoted as "placement moves accuracy by up to X", never tested.
    """
    means = {c: _matrix(by_seed, c, field)[0].mean() for c in conds}
    hi, lo = max(means, key=means.get), min(means, key=means.get)
    return _matrix(by_seed, hi, field)[0] - _matrix(by_seed, lo, field)[0]


def summarize(idx, rng):
    """Per-condition means with CIs, and the paired contrasts, per (arm, cell, k)."""
    summary, contrasts = [], []
    for (arm, cell, k), by_seed in sorted(idx.items()):
        conds = set.intersection(*(set(v) for v in by_seed.values()))
        for cond in sorted(conds):
            acc, _ = _matrix(by_seed, cond, "code_correct")
            lp, items = _matrix(by_seed, cond, "code_logprob")
            summary.append(dict(
                arm=arm, cell=cell, k=k, condition=cond, n_seeds=acc.shape[0],
                n_items=len(items),
                acc=_boot(acc, rng), logp=_boot(lp, rng),
                acc_by_seed=acc.mean(axis=1).round(4).tolist(),
            ))
        if not all(_cond(d, p) in conds for d in ("fwd", "rev") for p in PLACEMENTS):
            continue

        def m(cond, field):
            return _matrix(by_seed, cond, field)[0]

        defs = {
            # Lost in the middle: the edges beat the centre.
            "edges - middle (fwd)": lambda f: (m("fwd@0.00", f) + m("fwd@1.00", f)) / 2 - m("fwd@0.50", f),
            # Recency: the chain next to the question beats the chain at the top.
            "end - start (fwd)": lambda f: m("fwd@1.00", f) - m("fwd@0.00", f),
            # Reading order: the chain in hop order beats it reversed, averaged over p.
            "fwd - rev (mean over p)": lambda f: np.mean(
                [m(_cond("fwd", p), f) - m(_cond("rev", p), f) for p in PLACEMENTS], axis=0),
            # The two ends of the forward sweep against the §3.3 order.
            "fwd@0.00 - random": lambda f: m("fwd@0.00", f) - m("random", f),
            "fwd@1.00 - random": lambda f: m("fwd@1.00", f) - m("random", f),
            # How much placement alone moves the model: best minus worst condition.
            "max - min over conditions": lambda f: _spread(by_seed, conds, f),
        }
        for name, fn in defs.items():
            row = dict(arm=arm, cell=cell, k=k, contrast=name,
                       acc=_boot(fn("code_correct"), rng), logp=_boot(fn("code_logprob"), rng))
            pair = {"edges - middle (fwd)": ("fwd@1.00", "fwd@0.50"),
                    "end - start (fwd)": ("fwd@1.00", "fwd@0.00"),
                    "fwd@0.00 - random": ("fwd@0.00", "random"),
                    "fwd@1.00 - random": ("fwd@1.00", "random")}.get(name)
            if pair:
                p, n01, n10 = _mcnemar(m(pair[0], "code_correct"), m(pair[1], "code_correct"))
                row["mcnemar"] = dict(a=pair[0], b=pair[1], p=p, a_only=n10, b_only=n01)
            contrasts.append(row)
    return summary, contrasts


def answer_position_curve(idx, bins=5):
    """Under `random`, accuracy by where the answer passage happened to land.

    The §3.3 order already scatters the chain, so this reads the position effect off
    the unmodified evaluation — a check that the probe's blocks are not the only
    place it shows up.
    """
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for (arm, cell, k), by_seed in sorted(idx.items()):
        pos, ok = [], []
        for s in by_seed:
            for r in by_seed[s].get("random", {}).values():
                pos.append(r["answer_pos"])
                ok.append(r["code_correct"])
        if not pos:
            continue
        pos, ok = np.array(pos), np.array(ok)
        b = np.clip(np.digitize(pos, edges[1:-1]), 0, bins - 1)
        out.append(dict(arm=arm, cell=cell, k=k, bins=[
            dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=int((b == i).sum()),
                 acc=float(ok[b == i].mean()) if (b == i).any() else None)
            for i in range(bins)]))
    return out


def _resolved_cfg(sweep_dir):
    """The RunConfig a sweep's runs used, rebuilt from its first resolved config."""
    from ..__main__ import build_parser, config_from_args
    path = sorted(glob.glob(os.path.join(sweep_dir, "resolved", "*.json")))[0]
    argv = []
    for key, val in json.load(open(path)).items():
        if key.startswith("only_"):     # one run's slice; the analysis needs every split
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            argv.append(flag if val else "--no-" + key.replace("_", "-"))
        else:
            argv += [flag, str(val)]
    return config_from_args(build_parser().parse_args(argv))


def chain_order_curve(idx, sweep_dir):
    """Under `random`, accuracy by how many of the k chain links read forward.

    A link i -> i+1 reads forward when passage i precedes passage i+1 in the §3.3
    order. This separates reading order from answer position using the unmodified
    evaluation order, with no intervention at all. Needs the test graphs, so it loads
    the pickled graph lists (CPU only).
    """
    import pickle
    from ..flat import content_order
    from ..placement import chain_nodes
    from ..process_dataset import cell_split_name, split_paths, OUTPUT_ROOT
    from ....utils import TextGraphDataset

    cfg = _resolved_cfg(sweep_dir)
    paths = split_paths(cfg, root=cfg.resolved_data_root(OUTPUT_ROOT))
    out = []
    for (arm, cell, k), by_seed in sorted(idx.items()):
        if arm != "flat":
            continue
        base = TextGraphDataset.gtds_path(paths[cell_split_name(cell[0], cell[1], k)])
        with open(os.path.join(base, "graphs.pkl"), "rb") as fh:
            graphs = pickle.load(fh)
        links = {}
        for i, g in enumerate(graphs):
            order = content_order(g, cfg.data_seed + i)
            pos = {node: j for j, node in enumerate(order)}
            chain = chain_nodes(g)
            links[i] = sum(pos[a] < pos[b] for a, b in zip(chain, chain[1:]))
        acc = defaultdict(list)
        for s in by_seed:
            for i, r in by_seed[s].get("random", {}).items():
                acc[links[i]].append(r["code_correct"])
        out.append(dict(arm=arm, cell=cell, k=k, by_links={
            j: dict(n_graphs=sum(1 for v in links.values() if v == j),
                    acc=float(np.mean(acc[j])))
            for j in sorted(acc)}))
    return out


def reproduction_check(summary, reference_paths):
    """`random` against the §3.3 records for the same checkpoint and cell."""
    ref = {}
    for path in reference_paths:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("hops") is None or not r.get("checkpoint_path"):
                    continue
                arm = "graph" if r.get("arm", "").startswith("g") or "graph" in r["checkpoint_path"] else "flat"
                ref[(arm, _seed_label(r["checkpoint_path"]), (r["n_nodes"], r["tokens_per_node"]),
                     r["hops"])] = r.get("code_acc", r.get("em"))
    lines = []
    for s in summary:
        if s["condition"] != "random" or s["n_items"] != 200:
            continue
        for seed, acc in zip(sorted(_seeds_of(s)), s["acc_by_seed"]):
            want = ref.get((s["arm"], seed, s["cell"], s["k"]))
            if want is None:
                continue
            # §3.3 scored the in-memory model at the end of training; the probe reloads
            # the saved adapter onto a bf16 backbone, possibly on another GPU type. That
            # moves a borderline item or two, in either direction, and nothing else —
            # the inputs are byte-identical (tests/experiments/context/test_placement.py).
            # Every probe condition shares one loaded model, so no paired contrast sees it.
            diff = round(abs(acc - want) * 200)
            flag = "ok" if diff == 0 else ("numerics" if diff <= 2 else "MISMATCH")
            lines.append(f"{flag:8s} {s['arm']:5s} {seed} {s['cell'][0]}x{s['cell'][1]} k={s['k']}"
                         f"  probe={acc:.3f}  §3.3={want:.3f}")
    return lines


_SEEDS = {}


def _seeds_of(s):
    return _SEEDS[(s["arm"], s["cell"], s["k"])]


# ── figures ────────────────────────────────────────────────────────────────────

def plot_curves(summary, metric, out_path, cells, ks):
    ylabel = {"acc": "Answer accuracy (code_acc)", "logp": "Gold-code log-prob (nats)"}[metric]
    fig, axes = plt.subplots(len(ks), len(cells), figsize=(3.6 * len(cells), 2.9 * len(ks)),
                             sharex=True, sharey=(metric == "acc"), squeeze=False)
    by = {(s["arm"], s["cell"], s["k"], s["condition"]): s[metric] for s in summary}
    arms = sorted({s["arm"] for s in summary}, key=list(ARM_COLOR).index)
    for i, k in enumerate(ks):
        for j, cell in enumerate(cells):
            ax = axes[i][j]
            for arm in arms:
                c = ARM_COLOR[arm]
                if (arm, cell, k, "random") in by:
                    ax.axhline(by[(arm, cell, k, "random")][0], color=c, lw=1, ls=":", alpha=0.8)
                for d in ("fwd", "rev"):
                    pts = [(p, by.get((arm, cell, k, _cond(d, p)))) for p in PLACEMENTS]
                    pts = [(p, v) for p, v in pts if v is not None]
                    if not pts:
                        continue
                    xs = [p for p, _ in pts]
                    ys = np.array([v for _, v in pts])
                    ls, mk = DIR_STYLE[d]
                    ax.fill_between(xs, ys[:, 1], ys[:, 2], color=c, alpha=0.12, lw=0)
                    ax.plot(xs, ys[:, 0], ls=ls, marker=mk, ms=5, lw=2, color=c,
                            markeredgecolor="white", markeredgewidth=1,
                            label=f"{ARM_LABEL[arm]}, {DIR_LABEL[d]}")
            n, t = cell
            span = "~16k tok" if n * t <= 16384 else "~65k tok, beyond training length"
            ax.set_title(f"k={k}  ·  {n} nodes × {t} tok  ({span})", fontsize=8.5, color=INK)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xticks(PLACEMENTS)
            ax.set_xticklabels(["start", ".25", "middle", ".75", "end\n(next to Q)"])
            if metric == "acc":
                ax.set_ylim(-0.02, 1.02)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=INK)
            if i == len(ks) - 1:
                ax.set_xlabel("Where the chain block sits", fontsize=8.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1))
    labels.append("§3.3 order (random), same model")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ── tables ─────────────────────────────────────────────────────────────────────

def _ci(t, pct=False, fmt="{:.3f}"):
    m, lo, hi = t
    if pct:
        return f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def write_markdown(summary, contrasts, curve, repro, out_path, cells, ks):
    L = ["# Placement probe (README §3.4)", ""]
    L += ["## Reproduction check: `random` vs the §3.3 records", "", "```"] + (repro or ["(no 200-item random rows)"]) + ["```", ""]
    by = {(s["arm"], s["cell"], s["k"], s["condition"]): s for s in summary}
    arms = sorted({s["arm"] for s in summary}, key=list(ARM_COLOR).index)
    for arm in arms:
        L += [f"## {ARM_LABEL[arm]}: code_acc by placement (mean over seeds, 95% cluster-bootstrap CI)", ""]
        head = "| k | cell | random | " + " | ".join(f"fwd@{p:.2f}" for p in PLACEMENTS) + " | " + \
               " | ".join(f"rev@{p:.2f}" for p in PLACEMENTS) + " |"
        L += [head, "|" + "---|" * (2 + 1 + 2 * len(PLACEMENTS))]
        for k in ks:
            for cell in cells:
                if (arm, cell, k, "random") not in by:
                    continue
                vals = [by.get((arm, cell, k, c)) for c in
                        ["random"] + [_cond("fwd", p) for p in PLACEMENTS] + [_cond("rev", p) for p in PLACEMENTS]]
                L.append(f"| {k} | {cell[0]}x{cell[1]} | " +
                         " | ".join("—" if v is None else f"{v['acc'][0]:.3f}" for v in vals) + " |")
        L.append("")
        L += [f"### {ARM_LABEL[arm]}: gold-code log-prob by placement", ""]
        L += [head, "|" + "---|" * (2 + 1 + 2 * len(PLACEMENTS))]
        for k in ks:
            for cell in cells:
                if (arm, cell, k, "random") not in by:
                    continue
                vals = [by.get((arm, cell, k, c)) for c in
                        ["random"] + [_cond("fwd", p) for p in PLACEMENTS] + [_cond("rev", p) for p in PLACEMENTS]]
                L.append(f"| {k} | {cell[0]}x{cell[1]} | " +
                         " | ".join("—" if v is None else f"{v['logp'][0]:.2f}" for v in vals) + " |")
        L.append("")
    L += ["## Paired contrasts (percentage points of code_acc; nats of log-prob)", "",
          "| arm | k | cell | contrast | Δ acc (pp) [95% CI] | Δ log-prob [95% CI] | McNemar (a-only / b-only, p) |",
          "|---|---|---|---|---|---|---|"]
    for c in contrasts:
        mc = c.get("mcnemar")
        mc_s = f"{mc['a_only']} / {mc['b_only']}, p={mc['p']:.2g}" if mc else "—"
        L.append(f"| {c['arm']} | {c['k']} | {c['cell'][0]}x{c['cell'][1]} | {c['contrast']} | "
                 f"{_ci(c['acc'], pct=True)} | {_ci(c['logp'], fmt='{:+.2f}')} | {mc_s} |")
    L += ["", "## Under `random`: code_acc by where the answer passage landed", "",
          "| arm | k | cell | " + " | ".join(f"{b['lo']:.1f}–{b['hi']:.1f}" for b in curve[0]["bins"]) + " |" if curve else "",
          "|---|---|---|" + "---|" * (len(curve[0]["bins"]) if curve else 0)]
    for c in curve:
        L.append(f"| {c['arm']} | {c['k']} | {c['cell'][0]}x{c['cell'][1]} | " +
                 " | ".join("—" if b["acc"] is None else f"{b['acc']:.2f} (n={b['n']})" for b in c["bins"]) + " |")
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dirs", nargs="+")
    ap.add_argument("--out-dir", default="src/experiments/context/results/placement_analysis")
    ap.add_argument("--reference", nargs="*", default=[
        "src/experiments/context/results/flat_trained_grid.jsonl",
        "src/experiments/context/results/mainsweep_grid/grid.jsonl"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chain-order", action="store_true",
                    help="also tabulate `random` accuracy by forward-reading chain links; "
                         "loads the test graphs, so run it on a compute node")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_items(args.sweep_dirs)
    idx = index(rows)
    for key, by_seed in idx.items():
        _SEEDS[key] = sorted(by_seed)
    rng = np.random.default_rng(args.seed)
    summary, contrasts = summarize(idx, rng)
    curve = answer_position_curve(idx)
    links = chain_order_curve(idx, args.sweep_dirs[0]) if args.chain_order else []
    repro = reproduction_check(summary, args.reference)

    cells = sorted({s["cell"] for s in summary}, key=lambda c: (c[0] * c[1], c[0]))
    ks = sorted({s["k"] for s in summary})
    plot_curves(summary, "acc", os.path.join(args.out_dir, "placement_acc.png"), cells, ks)
    plot_curves(summary, "logp", os.path.join(args.out_dir, "placement_logp.png"), cells, ks)
    write_markdown(summary, contrasts, curve, repro, os.path.join(args.out_dir, "placement.md"), cells, ks)
    if links:
        with open(os.path.join(args.out_dir, "placement.md"), "a") as fh:
            fh.write("\n## Under `random`: code_acc by chain links read forward (of k)\n\n"
                     "| k | cell | links forward: acc (graphs) |\n|---|---|---|\n")
            for c in links:
                fh.write(f"| {c['k']} | {c['cell'][0]}x{c['cell'][1]} | " + " · ".join(
                    f"{j}: {v['acc']:.2f} ({v['n_graphs']})" for j, v in c["by_links"].items()) + " |\n")
    with open(os.path.join(args.out_dir, "placement_summary.json"), "w") as fh:
        json.dump(dict(summary=summary, contrasts=contrasts, answer_position=curve,
                       chain_order=links, reproduction=repro),
                  fh, indent=1, default=str)
    print("\n".join(repro))
    print(f"wrote {args.out_dir}/placement_{{acc,logp}}.{{png,pdf}}, placement.md, placement_summary.json")


if __name__ == "__main__":
    main()
