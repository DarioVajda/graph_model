"""The padded (N, L) of every built graph source, and a ladder fitted to them.

`GRAPH_GENERALIST.md` §3 replaces the collator's fixed ladders (multiples of 512
tokens with 1.5x midpoints, powers of two of nodes floored at 32) by boundaries
fitted to the measured distribution, and chooses the D5 caps against the tails.
Both need the same thing: each source's raw node count and token count per row,
which every build already records in its split sidecar (``num_nodes``,
``num_tokens`` — the lengths the sampler buckets on). So this reads sidecars
only; no dataset is loaded and no GPU is needed.

    src/generalist/tools/launch/run_py.sh src/generalist/tools/reports/shapes.py \\
        --config src/generalist/configs/probes/015_graph_full_build.jsonc

Per source it prints the quantiles of N and L and how much the current ladders
pad them. Then it fits each ladder separately, since the collator pads the two
dimensions independently: for every bucket count K, the K boundaries that
minimise the expected padded cost over the pooled rows. The cost of a token
bucket is its length (the linear layers and the logits scale with L); the cost
of a node bucket is its square (the dense ``(B, H, N, N)`` pair bias). Padding
is per row, because shape-keyed batching puts a row in its own bucket — the
pad-to-the-batch-max of a merged group is not modelled.

Rows are pooled with a weight per source, so a ladder is fitted to what a run
will draw rather than to whichever corpus happens to be largest: every domain
counts the same and splits its share evenly over its tasks (``--weights
domain``, the default while the trunk mixture is open), every task counts the
same (``uniform``, which lets GraphQA's nine near-identical tasks outvote the
rest), or the config's mixture weights (``mixture``).

``--max-tokens`` / ``--max-nodes`` are candidate D5 caps: rows above them are
counted per source and left out of the fit, which is what a cap does to them.
The top boundary is otherwise the single longest row, and the CWQ tail puts that
at more than twice its 99th percentile.

Last, the joint shape set: how many ``(N, L)`` buckets the fitted ladders
actually populate, against the compile cache the shapes share with evaluation.
"""

import argparse
import json
import os
import sys

import numpy as np


def _sources(config, splits):
    """``[(task, split, num_nodes, num_tokens)]`` over every built graph split."""
    from src.generalist.adapters import GRAPH_DOMAINS, get_adapter
    from src.generalist.adapters._graph import _sidecar_path

    out, missing = [], []
    for domain_name in GRAPH_DOMAINS:
        domain = get_adapter(domain_name).DOMAIN_SPEC
        dconf = config.domain_adapter_config(domain_name)
        for info in domain.tasks:
            name = domain.full_name(info.name)
            for split in info.built_splits():
                if split not in splits and not (info.held_out and "held_out" in splits):
                    continue
                side = _sidecar_path(domain.source_path(dconf, name, split, "graph", 0))
                if not os.path.exists(side):
                    missing.append(f"{name}/{split}")
                    continue
                with open(side) as fh:
                    sidecar = json.load(fh)
                out.append((name, split, np.asarray(sidecar["num_nodes"], dtype=np.int64),
                            np.asarray(sidecar["num_tokens"], dtype=np.int64)))
    return out, missing


def _current_ladders():
    from src.models.flex_kernel import default_len_buckets, default_node_buckets

    return default_node_buckets, default_len_buckets


def _apply(ladder, values: np.ndarray) -> np.ndarray:
    """``values`` padded up to ``ladder``: a callable, or sorted boundaries."""
    if callable(ladder):
        return np.array([ladder(int(v)) for v in values], dtype=np.int64)
    bounds = np.asarray(ladder, dtype=np.int64)
    return bounds[np.searchsorted(bounds, values, side="left")]


def fit_ladder(values: np.ndarray, weights: np.ndarray, k: int, grid: int,
               floor: int, cost) -> list:
    """The ``k`` boundaries minimising ``sum w * cost(bucket(v))``.

    Boundaries sit on multiples of ``grid`` no lower than ``floor`` (the
    collator needs block-aligned lengths), and the last is the largest value
    rounded up, so nothing falls off the top. Exact: a 1-D weighted optimal
    quantisation by dynamic programming over the distinct grid points.
    """
    snapped = np.maximum(floor, grid * np.ceil(values / grid).astype(np.int64))
    points, inverse = np.unique(snapped, return_inverse=True)
    mass = np.bincount(inverse, weights=weights, minlength=len(points))
    m = len(points)
    k = min(k, m)
    prefix = np.concatenate([[0.0], np.cumsum(mass)])
    c = np.array([cost(int(p)) for p in points], dtype=np.float64)
    inf = float("inf")
    best = np.full((k + 1, m), inf)
    back = np.zeros((k + 1, m), dtype=np.int64)
    for j in range(m):                                  # one bucket: everything to j
        best[1][j] = c[j] * prefix[j + 1]
    for b in range(2, k + 1):
        for j in range(b - 1, m):
            # last bucket covers points i+1..j and pads them to points[j]
            i = np.arange(b - 2, j)
            cand = best[b - 1][i] + c[j] * (prefix[j + 1] - prefix[i + 1])
            n = int(np.argmin(cand))
            best[b][j] = cand[n]
            back[b][j] = i[n]
    bounds, j = [], m - 1
    for b in range(k, 0, -1):
        bounds.append(int(points[j]))
        j = back[b][j]
    return sorted(bounds)


def _q(values, qs=(0.5, 0.9, 0.99, 1.0)) -> list:
    return [int(np.quantile(values, q)) for q in qs]


def main() -> int:
    from src.generalist.config import RunConfig, load_config_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell", default=None)
    parser.add_argument("--splits", default="train",
                        help="comma list; 'held_out' adds held-out tasks' sets")
    parser.add_argument("--weights", choices=("domain", "uniform", "mixture"),
                        default="domain")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="candidate D5 cap; longer rows leave the fit (0: none)")
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="candidate D5 cap; larger rows leave the fit (0: none)")
    parser.add_argument("--max-buckets", type=int, default=16)
    parser.add_argument("--pick", default=None, metavar="KL,KN",
                        help="bucket counts whose joint shape set to report "
                             "(default: where each ladder flattens)")
    parser.add_argument("--token-grid", type=int, default=128,
                        help="the collator's block_size: L buckets are multiples of it")
    parser.add_argument("--node-grid", type=int, default=16)
    parser.add_argument("--node-floor", type=int, default=32)
    parser.add_argument("--json", default=None, help="write the fitted ladders here")
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config, args.cell)).validate()
    splits = set(args.splits.split(","))
    sources, missing = _sources(config, splits)
    if not sources:
        print("no built graph sources for", sorted(splits))
        return 1

    tasks = sorted({t for t, *_ in sources})
    if args.weights == "mixture":
        shares = {e["name"]: float(e.get("weight", 0.0)) for e in config.mixture_entries()}
        total = sum(shares.values()) or 1.0
        share_of = {t: shares.get(t, 0.0) / total for t in tasks}
    elif args.weights == "domain":
        domains = {t.split("/")[0] for t in tasks}
        share_of = {t: 1.0 / len(domains)
                    / sum(1 for u in tasks if u.split("/")[0] == t.split("/")[0])
                    for t in tasks}
    else:
        share_of = {t: 1.0 / len(tasks) for t in tasks}

    node_now, len_now = _current_ladders()
    caps = []
    if args.max_tokens:
        caps.append(f"max_tokens {args.max_tokens}")
    if args.max_nodes:
        caps.append(f"max_nodes {args.max_nodes}")
    print(f"config {args.config}; splits {sorted(splits)}; weights {args.weights}"
          + (f"; caps {', '.join(caps)}" if caps else ""))
    if missing:
        print(f"not built (skipped): {', '.join(missing)}")
    print(f"\n{'source':<34}{'share':>6}{'rows':>8}  {'N p50/p90/p99/max':<22}"
          f"{'L p50/p90/p99/max':<26}{'L pad':>7}{'N² pad':>8}{'capped':>8}")
    pooled_n, pooled_l, pooled_w = [], [], []
    for task, split, nodes, tokens in sources:
        n_split = sum(1 for t, *_ in sources if t == task)
        l_pad = _apply(len_now, tokens)
        n_pad = _apply(node_now, nodes)
        l_over = l_pad.sum() / max(1, tokens.sum())
        n_over = (n_pad.astype(float) ** 2).sum() / max(1.0, (nodes.astype(float) ** 2).sum())
        keep = np.ones(len(nodes), dtype=bool)
        if args.max_tokens:
            keep &= tokens <= args.max_tokens
        if args.max_nodes:
            keep &= nodes <= args.max_nodes
        capped = 1.0 - keep.mean()
        print(f"{task + '/' + split:<34}{share_of[task] / n_split:>6.3f}{len(nodes):>8}  "
              f"{'/'.join(map(str, _q(nodes))):<22}{'/'.join(map(str, _q(tokens))):<26}"
              f"{l_over:>6.2f}x{n_over:>7.2f}x{100 * capped:>7.2f}%")
        if keep.any():
            pooled_n.append(nodes[keep])
            pooled_l.append(tokens[keep])
            pooled_w.append(np.full(int(keep.sum()), share_of[task] / n_split / keep.sum()))
    nodes = np.concatenate(pooled_n)
    tokens = np.concatenate(pooled_l)
    weights = np.concatenate(pooled_w)
    weights = weights / weights.sum()

    def overhead(pad, raw, power):
        return float((weights * pad.astype(float) ** power).sum()
                     / (weights * raw.astype(float) ** power).sum())

    l_cur = overhead(_apply(len_now, tokens), tokens, 1)
    n_cur = overhead(_apply(node_now, nodes), nodes, 2)
    l_cur_k = len(np.unique(_apply(len_now, tokens)))
    n_cur_k = len(np.unique(_apply(node_now, nodes)))
    print(f"\npooled: {len(nodes)} rows; current ladders pad L {l_cur:.3f}x over "
          f"{l_cur_k} buckets, N² {n_cur:.3f}x over {n_cur_k} buckets")

    fitted = {"tokens": {}, "nodes": {}}
    print(f"\n{'K':>3}  {'L pad':>7}  {'N² pad':>7}")
    for k in range(1, args.max_buckets + 1):
        lb = fit_ladder(tokens, weights, k, args.token_grid, args.token_grid,
                        cost=lambda v: v)
        nb = fit_ladder(nodes, weights, k, args.node_grid, args.node_floor,
                        cost=lambda v: v * v)
        fitted["tokens"][k] = {"bounds": lb, "overhead": overhead(_apply(lb, tokens), tokens, 1)}
        fitted["nodes"][k] = {"bounds": nb, "overhead": overhead(_apply(nb, nodes), nodes, 2)}
        print(f"{k:>3}  {fitted['tokens'][k]['overhead']:>6.3f}x  "
              f"{fitted['nodes'][k]['overhead']:>6.3f}x")

    # The smallest K that matches the current ladder's padding, and the K past
    # which another bucket buys under 1 % — the two ends of the choice.
    def knee(dim, current):
        rows = fitted[dim]
        match = next((k for k in sorted(rows) if rows[k]["overhead"] <= current), None)
        flat = next((k for k in sorted(rows)[1:]
                     if rows[k - 1]["overhead"] - rows[k]["overhead"] < 0.01), None)
        return match, flat

    picked = {}
    for dim, current, cur_k in (("tokens", l_cur, l_cur_k), ("nodes", n_cur, n_cur_k)):
        match, flat = knee(dim, current)
        picked[dim] = flat or args.max_buckets
        print(f"\n{dim}: current ladder uses {cur_k} buckets at {current:.3f}x; "
              f"fitted matches it at K={match}, flattens (<1% per bucket) at K={flat}")
        for k in sorted({x for x in (match, flat) if x}):
            print(f"  K={k}: {fitted[dim][k]['bounds']}")
    if args.pick:
        picked["tokens"], picked["nodes"] = (int(v) for v in args.pick.split(","))

    # The joint shape set: the collator pads N and L independently, so the
    # kernels see every populated (N, L) pair, and each again per row count.
    def shape_set(node_ladder, len_ladder):
        pairs = np.stack([_apply(node_ladder, nodes), _apply(len_ladder, tokens)], axis=1)
        keys, inverse = np.unique(pairs, axis=0, return_inverse=True)
        mass = np.bincount(inverse.ravel(), weights=weights, minlength=len(keys))
        return len(keys), int((mass >= 0.001).sum())

    kl, kn = picked["tokens"], picked["nodes"]
    now_all, now_main = shape_set(node_now, len_now)
    fit_all, fit_main = shape_set(fitted["nodes"][kn]["bounds"],
                                  fitted["tokens"][kl]["bounds"])
    print(f"\njoint (N, L) shapes populated (of them, holding >= 0.1% of the draw):")
    print(f"  current ladders:            {now_all:>4} ({now_main})")
    print(f"  fitted K_L={kl:<2} K_N={kn:<2}:       {fit_all:>4} ({fit_main}); "
          f"L {fitted['tokens'][kl]['overhead']:.3f}x, N² {fitted['nodes'][kn]['overhead']:.3f}x")
    print(f"  tokens {fitted['tokens'][kl]['bounds']}")
    print(f"  nodes  {fitted['nodes'][kn]['bounds']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"config": args.config, "splits": sorted(splits),
                       "weights": args.weights, "missing": missing,
                       "caps": {"max_tokens": args.max_tokens, "max_nodes": args.max_nodes},
                       "current": {"tokens": l_cur, "nodes": n_cur, "shapes": now_all},
                       "picked": {"tokens": kl, "nodes": kn, "shapes": fit_all},
                       "fitted": fitted}, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
