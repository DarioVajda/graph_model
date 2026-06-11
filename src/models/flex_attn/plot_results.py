"""
Publication-quality figures for the FlexAttention benchmark results.

Produces four PNG figures from isolation.jsonl and full_model.jsonl:

  fig_isolation_all_methods  — eager / flex (no mask) / flex (k=2) / flash
  fig_isolation_flex_k0      — eager / flex / flash  (k=0 only)
  fig_full_model_all_methods — same 4-method comparison, full model
  fig_full_model_flex_k0     — same 3-method comparison, full model

Each figure: two panels (latency left, peak memory right).
Both axes log-scale.  X-axis ticks at powers of 2; y-axis ticks at powers of 10.
Each plotted x-position is the nearest power of 2 to the true sequence length;
data points that map to the same power of 2 are averaged.  OOM / failed rows
are excluded from averages; lines simply end where data runs out.

Usage:
    python -m src.models.flex_attn.plot_results \
        --results-dir src/models/flex_attn/results \
        --out-dir     src/models/flex_attn/results
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── visual encoding (Paul Tol "bright", colour-blind safe) ────────────────────

_STYLE: dict[str, dict] = {
    "eager":          {"color": "#CC3311", "ls": "-", "marker": "o"},
    "flex (no mask)": {"color": "#0077BB", "ls": "-", "marker": "o"},
    "flex (k=2)":     {"color": "#009988", "ls": "-", "marker": "o"},
    "flex":           {"color": "#0077BB", "ls": "-", "marker": "o"},
    "flash":          {"color": "#EE7733", "ls": "-", "marker": "o"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]



def _series(
    rows: list[dict],
    method: str,
    k_hop: int,
    ordering: str = "rcm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, latency_ms, memory_gb) averaged per (n_nodes, tpn) bucket.

    X is n_nodes * tokens_per_node, which is always an exact power of 2
    (both axes of the sweep are powers of 2).  Rows sharing the same
    (n_nodes, tpn) target are averaged; OOM/failed rows are excluded.
    The returned arrays are sorted by x.
    """
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for r in rows:
        sp = r.get("spec", {})
        if r.get("method") != method:
            continue
        if ordering and sp.get("ordering") != ordering:
            continue
        if sp.get("k_hop") != k_hop:
            continue
        if not r.get("ok"):
            continue
        n   = sp.get("n_nodes")
        t   = sp.get("tokens_per_node")
        ms  = r.get("timing_ms", {}).get("fwd_bwd")
        mb  = r.get("memory_mb", {}).get("peak_fwd_bwd")
        if None in (n, t, ms, mb):
            continue
        buckets[n * t].append((ms, mb / 1024))

    if not buckets:
        return np.array([]), np.array([]), np.array([])

    xs, mss, gbs = [], [], []
    for p2, vals in sorted(buckets.items()):
        ms_mean = np.mean([v[0] for v in vals])
        gb_mean = np.mean([v[1] for v in vals])
        xs.append(p2)
        mss.append(ms_mean)
        gbs.append(gb_mean)

    return np.array(xs, dtype=float), np.array(mss), np.array(gbs)


# ── axis helpers ──────────────────────────────────────────────────────────────

def _setup_x(ax_t, ax_m, all_x: np.ndarray) -> None:
    """Log scale + power-of-2 ticks on x for both panels."""
    pow2s = sorted({int(x) for x in all_x})

    def _xlbl(v: int) -> str:
        return f"{v // 1024}K" if v >= 1024 else str(v)

    for ax in (ax_t, ax_m):
        ax.set_xscale("log")
        ax.set_xticks(pow2s)
        ax.set_xticklabels([_xlbl(v) for v in pow2s])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(which="major", direction="in", length=3)


def _fmt_time(v: float) -> str:
    """Format a millisecond value as e.g. '10ms' or '1s'."""
    if v >= 1000:
        s = v / 1000
        return f"{s:g}s"
    return f"{v:g}ms"


def _fmt_mem(v: float) -> str:
    """Format a GB value as e.g. '100MB' or '10GB'.
    Uses ×1000 for MB conversion so tick labels stay round (100MB not 102.4MB)."""
    if v < 1:
        return f"{v * 1000:g}MB"
    return f"{v:g}GB"


def _smart_yticks(ax, fmt=None) -> None:
    """Adaptive log y-ticks with optional unit formatter.

    Uses pure powers of 10 when the visible range spans ≥ 1.5 decades.
    Falls back to ×1 / ×2 / ×5 multiples when the range is narrower, so
    there are always at least two labelled ticks.
    """
    lo, hi = ax.get_ylim()
    if lo <= 0 or hi <= lo:
        return
    exp_lo = int(np.floor(np.log10(lo)))
    exp_hi = int(np.ceil(np.log10(hi)))

    pow10 = [10**e for e in range(exp_lo, exp_hi + 1) if lo <= 10**e <= hi]

    if len(pow10) >= 2:
        ticks = pow10
    else:
        ticks = sorted({
            sub * 10**e
            for e in range(exp_lo - 1, exp_hi + 2)
            for sub in (1, 2, 5)
            if lo <= sub * 10**e <= hi
        })

    labels = [fmt(v) if fmt else f"{v:g}" for v in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(mticker.NullLocator())


# ── figure builder ────────────────────────────────────────────────────────────

def _annotate_ratios(ax_t, ax_m, drawn: list) -> None:
    """Annotate each non-flash point with its ratio vs flash.

    Annotations are placed to the right of the dot (same y as the data point),
    which naturally separates eager and flex since they have different y-values.
    The last data point of each series is annotated to the left to avoid
    running off the right edge.  A faint white background keeps text readable
    when it crosses a line.
    """
    flash_ms = flash_gb = None
    for label, x, ms, gb in drawn:
        if label == "flash":
            flash_ms = dict(zip(x.tolist(), ms.tolist()))
            flash_gb = dict(zip(x.tolist(), gb.tolist()))
            break
    if flash_ms is None:
        return

    _bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)

    global_last_x = max(xi for _, x, _, _ in drawn for xi in x.tolist())

    for label, x, ms, gb in drawn:
        if label == "flash":
            continue
        color = _STYLE[label]["color"]
        x_list, ms_list, gb_list = x.tolist(), ms.tolist(), gb.tolist()
        series_last_xi = x_list[-1]

        for xi, msi, gbi in zip(x_list, ms_list, gb_list):
            # Flip to left only when the series reaches the rightmost x of the plot
            flip = (xi == series_last_xi) and (xi == global_last_x)
            x_off = -5 if flip else 5
            ha    = "right" if flip else "left"

            if xi in flash_ms and flash_ms[xi] > 0:
                ax_t.annotate(
                    f"{msi / flash_ms[xi]:.1f}×",
                    xy=(xi, msi), xytext=(x_off, 0), textcoords="offset points",
                    fontsize=6.5, color=color, ha=ha, va="center",
                    bbox=_bbox, zorder=5,
                )
            if xi in flash_gb and flash_gb[xi] > 0:
                ax_m.annotate(
                    f"{gbi / flash_gb[xi]:.1f}×",
                    xy=(xi, gbi), xytext=(x_off, 0), textcoords="offset points",
                    fontsize=6.5, color=color, ha=ha, va="center",
                    bbox=_bbox, zorder=5,
                )


def _make_figure(
    rows: list[dict],
    series_spec: list[tuple[str, str, int]],  # (display_label, method, k_hop)
    out_path: str,
    annotate_vs_flash: bool = False,
) -> None:
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    all_x: list[float] = []
    drawn: list[tuple] = []   # (label, x, ms, gb) for axis setup after all series

    for label, method, k_hop in series_spec:
        x, ms, gb = _series(rows, method, k_hop)
        if len(x) == 0:
            continue
        all_x.extend(x.tolist())
        drawn.append((label, x, ms, gb))

    if not all_x:
        plt.close(fig)
        return

    all_x_arr = np.array(sorted(set(all_x)))
    _setup_x(ax_t, ax_m, all_x_arr)
    for ax in (ax_t, ax_m):
        ax.set_yscale("log")

    for label, x, ms, gb in drawn:
        st = _STYLE[label]
        kw = dict(color=st["color"], linestyle=st["ls"], marker=st["marker"])
        ax_t.plot(x, ms, label=label, **kw)
        ax_m.plot(x, gb, **kw)

    if annotate_vs_flash:
        _annotate_ratios(ax_t, ax_m, drawn)

    ax_t.set_xlabel("Prefix sequence length (tokens)")
    ax_t.set_ylabel("Fwd+bwd latency")
    ax_m.set_xlabel("Prefix sequence length (tokens)")
    ax_m.set_ylabel("Peak memory")

    _smart_yticks(ax_t, fmt=_fmt_time)
    _smart_yticks(ax_m, fmt=_fmt_mem)

    ax_t.legend(loc="upper left", frameon=False)

    fig.tight_layout(pad=0.4, w_pad=1.4)
    fig.savefig(out_path + ".png")
    plt.close(fig)
    print(f"Saved {out_path}.png")


# ── main ──────────────────────────────────────────────────────────────────────

_FOUR = [
    ("eager",          "eager", 0),
    ("flex (no mask)", "flex",  0),
    ("flex (k=2)",     "flex",  2),
    ("flash",          "flash", 0),
]
_THREE = [
    ("eager", "eager", 0),
    ("flex",  "flex",  0),
    ("flash", "flash", 0),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="src/models/flex_attn/results")
    p.add_argument("--out-dir",     default="src/models/flex_attn/results")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    for kind in ("isolation", "full_model"):
        jsonl = os.path.join(args.results_dir, f"{kind}.jsonl")
        if not os.path.exists(jsonl):
            print(f"Skipping {jsonl} (not found)")
            continue
        rows = _load(jsonl)
        _make_figure(rows, _FOUR,
                     os.path.join(args.out_dir, f"fig_{kind}_all_methods"))
        _make_figure(rows, _THREE,
                     os.path.join(args.out_dir, f"fig_{kind}_flex_k0"),
                     annotate_vs_flash=True)


if __name__ == "__main__":
    main()
