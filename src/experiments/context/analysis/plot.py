"""
Phase 4 — turn ``grid.jsonl`` into the paper figure and its markdown table.

    python3 -m src.experiments.context.analysis.plot <grid.jsonl> [--out-dir DIR]

Design decisions worth not re-litigating:

  * **Sequential, single hue, fixed 0–1 scale.** Accuracy is a magnitude, so the
    ramp is one hue light→dark with a scale legend — never a rainbow, and never
    auto-scaled, because the 16k figure and the 32k diagnostic figure have to be
    comparable by eye.
  * **Every cell is annotated** with its accuracy and its packed length, so
    identity never rests on color alone, and the markdown table is the table view.
  * **The training-cap boundary is drawn on the figure.** Cells above it are
    length extrapolation, which is the first thing a reader should be able to
    check (README §A.6).
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..evaluate import wilson_interval

# One hue, light -> dark. Fixed across every figure this script writes.
CMAP = "Blues"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
BOUNDARY = "#c2410c"        # the max_train_len iso-line


def pick_metric(rows, metric="auto"):
    """Which accuracy column to plot.

    ``code_acc`` (the answer tokens, EOS ignored) is the arm-independent measure and
    the one the results report, so it wins whenever every record carries it. ``em``
    additionally demands the EOS convention, which an untrained arm has never been
    taught — see README §A.9. Older records predate ``code_acc`` and fall back to
    ``em``, which is identical for a trained arm at ceiling.
    """
    if metric != "auto":
        return metric
    return "code_acc" if all(r.get("code_acc") is not None for r in rows) else "em"


def load_grid(path, metric="auto"):
    """Aggregate ``grid.jsonl`` into ``{k: {(n, t): {...}}}``, pooling over seeds/runs.

    Records are keyed by (N, T, **k**), never by (N, T) alone. Pooling the k values
    of one cell would average a condition the graph arm solves perfectly (k=1) with
    one it does not (k=4) and then label the mean as a single accuracy — and the
    surface this experiment measures lives mostly on the k axis, not on T
    (README §3.3). A build that pins one k, or predates the k mixture entirely,
    yields a single group and behaves exactly as before.
    """
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return {}
    column = pick_metric(records, metric)

    rows = defaultdict(list)
    for r in records:
        rows[(int(r.get("hops", 0)), r["n_nodes"], r["tokens_per_node"])].append(r)

    by_k = defaultdict(dict)
    for (k, n, t), group in rows.items():
        n_total = sum(r["n"] for r in group)
        hits = sum(round(r[column] * r["n"]) for r in group)
        lo, hi = wilson_interval(hits, n_total)
        by_k[k][(n, t)] = {
            "em": hits / max(1, n_total),
            "metric": column,
            "ci_low": lo, "ci_high": hi,
            "n": n_total,
            "seeds": len(group),
            "packed_len": max(r["packed_len"] for r in group),
            "distractor_rate": float(np.mean([r["distractor_rate"] for r in group])),
            "code_no_eos_rate": float(np.mean([r.get("code_no_eos_rate", 0.0) for r in group])),
            "malformed_rate": float(np.mean([r["malformed_rate"] for r in group])),
            "in_train": all(r.get("in_train_distribution", True) for r in group),
            "max_train_len": max(r.get("max_train_len", 0) for r in group),
        }
    return {k: by_k[k] for k in sorted(by_k)}


def _axes(cells):
    node_counts = sorted({n for (n, _) in cells})
    token_counts = sorted({t for (_, t) in cells})
    return node_counts, token_counts


def plot_heatmap(cells, out_path, title="GTLM context exhaustion — needle in a graph"):
    """Teacher-forced EM over the (N, T) grid."""
    node_counts, token_counts = _axes(cells)
    grid = np.full((len(node_counts), len(token_counts)), np.nan)
    for i, n in enumerate(node_counts):
        for j, t in enumerate(token_counts):
            if (n, t) in cells:
                grid[i, j] = cells[(n, t)]["em"]

    fig, ax = plt.subplots(figsize=(1.35 * len(token_counts) + 2.2,
                                    1.0 * len(node_counts) + 1.8))
    im = ax.imshow(grid, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")

    for i, n in enumerate(node_counts):
        for j, t in enumerate(token_counts):
            if (n, t) not in cells:
                continue
            c = cells[(n, t)]
            # Ink on light cells, surface on dark ones — the annotation is text, so
            # it wears text colors, not the ramp's.
            colour = "white" if c["em"] > 0.55 else INK
            ax.text(j, i - 0.13, f"{100 * c['em']:.0f}", ha="center", va="center",
                    color=colour, fontsize=12, fontweight="semibold")
            ax.text(j, i + 0.22, f"{c['packed_len'] / 1000:.1f}k", ha="center", va="center",
                    color=colour, fontsize=8.5, alpha=0.75)

    # The training-length boundary: a cell is extrapolation iff it was excluded
    # from the training mixture. The region is monotone in (N, T), so its border
    # draws as a staircase between neighbouring cells.
    for i, n in enumerate(node_counts):
        for j, t in enumerate(token_counts):
            c = cells.get((n, t))
            if c is None or c["in_train"]:
                continue
            left = cells.get((n, token_counts[j - 1])) if j > 0 else None
            below = cells.get((node_counts[i - 1], t)) if i > 0 else None
            if left is not None and left["in_train"]:
                ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color=BOUNDARY, lw=2.5)
            if below is not None and below["in_train"]:
                ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color=BOUNDARY, lw=2.5)

    cap = max((c["max_train_len"] for c in cells.values()), default=0)
    ax.set_xticks(range(len(token_counts)), [str(t) for t in token_counts])
    ax.set_yticks(range(len(node_counts)), [str(n) for n in node_counts])
    ax.set_xlabel("Tokens per node (T)", color=INK)
    ax.set_ylabel("Nodes (N)", color=INK)
    ax.set_title(title, color=INK, pad=12)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(token_counts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(node_counts), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    metric = next(iter(cells.values())).get("metric", "em")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label({"code_acc": "Retrieval accuracy (code_acc)"}.get(
        metric, "Teacher-forced exact match"), color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, length=0)

    if any(not c["in_train"] for c in cells.values()):
        ax.plot([], [], color=BOUNDARY, lw=2.5,
                label=f"trained up to {cap:,} tokens (beyond = extrapolation)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
                  labelcolor=MUTED, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path


def write_table(by_k, out_path):
    """The table view: point estimate, Wilson CI, length, failure decomposition.

    One row per (k, N, T) evaluation condition — the same keying as ``load_grid``,
    so the table never silently merges two k values into one row.
    """
    metric = next((c["metric"] for cells in by_k.values() for c in cells.values()), "em")
    header = "code_acc" if metric == "code_acc" else "EM"
    lines = [
        f"| k | N | T | packed L | in-train | {header} | 95% CI | distractor | code-no-EOS | malformed | n |",
        "| --: | --: | --: | --: | :-- | --: | :-- | --: | --: | --: | --: |",
    ]
    for k, cells in by_k.items():
        node_counts, token_counts = _axes(cells)
        for n in node_counts:
            for t in token_counts:
                c = cells.get((n, t))
                if c is None:
                    continue
                lines.append(
                    f"| {k} | {n} | {t} | {c['packed_len']:,} | {'yes' if c['in_train'] else '**no**'} | "
                    f"{100 * c['em']:.1f} | [{100 * c['ci_low']:.1f}, {100 * c['ci_high']:.1f}] | "
                    f"{100 * c['distractor_rate']:.1f} | {100 * c['code_no_eos_rate']:.1f} | "
                    f"{100 * c['malformed_rate']:.1f} | {c['n']} |")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[plot] wrote {out_path}")
    return text


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m src.experiments.context.analysis.plot",
        description="Heatmap + markdown table from a context grid.jsonl.")
    p.add_argument("grid_jsonl")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the figure/table (default: alongside grid.jsonl).")
    p.add_argument("--title", default="GTLM context exhaustion — needle in a graph")
    p.add_argument("--metric", choices=("auto", "code_acc", "em"), default="auto",
                   help="Accuracy column to plot (default: code_acc when present).")
    args = p.parse_args(argv)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.grid_jsonl))
    os.makedirs(out_dir, exist_ok=True)

    by_k = load_grid(args.grid_jsonl, metric=args.metric)
    if not by_k:
        raise SystemExit(f"{args.grid_jsonl} holds no cell records.")

    # One figure per hop count. A heatmap is a surface over (N, T), so k has to be
    # held fixed within it; the file name carries k so the set is self-describing.
    # A single-k grid keeps the historical unsuffixed name.
    for k, cells in by_k.items():
        suffix = "" if len(by_k) == 1 else f"_k{k}"
        title = args.title if len(by_k) == 1 else f"{args.title} — k={k}"
        plot_heatmap(cells, os.path.join(out_dir, f"fig_context_grid{suffix}.png"), title=title)
    print(write_table(by_k, os.path.join(out_dir, "grid_table.md")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
