"""Does directed reachability concentrate on the pairs the TASK cares about?

    python -m src.experiments.bias_experiments.landmark.gold_coverage

`anchor_diagnostic.py` §6 measured pair visibility uniformly over all ordered
pairs and found the directed channels non-zero (at init) on only 5.2% of them,
against 100% for the undirected channel. That 19x gap is the entire case for
channel 3 — and it is only a real case if the invisible 95% contains pairs the
model needs.

It might not. SR retrieval is near-shortest-path from the topic entity to the
answer, so the answer node is, by construction, downstream of a topic entity
along a DIRECTED path. If visibility concentrates on gold-answer pairs, the
directed channels already cover what matters and the coverage argument for
channel 3 collapses to "covers many pairs, few useful ones".

Measured, per graph, over ordered pairs (u,v):

  all      — the §6 baseline
  ->gold   — v is a gold-answer node (any u). The pairs that must inform the
             representation of the answer entity.
  gold->   — u is a gold-answer node.
  ->gold_u — v is gold, restricted to u that are NOT gold (drops the trivially
             visible gold->gold block, which is small but self-serving).

Reported for the directed oracle and, as the control, the undirected one. If
`->gold` is far above `all` for the directed oracle, keep the head width and drop
channel 3; if it tracks `all`, channel 3 is buying real coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .anchor_diagnostic import (RESULTS, WEBQSP_BASE, WEBQSP_QNISO, _struct_keys,
                                _top, load_graphs, select_anchors)

import networkx as nx


def gold_mask(G) -> np.ndarray | None:
    """1 for nodes whose text is a gold answer. `process_dataset.py:370` stashes
    `gold_answers` (already restricted to answers grounded in G), and every node
    carries `text`, so this is the same match the evaluator scores on."""
    gold = G.graph.get("gold_answers")
    if not gold:
        return None
    gold = {str(g).strip().lower() for g in gold}
    lab = np.zeros(G.number_of_nodes(), dtype=bool)
    for i, (_, data) in enumerate(G.nodes(data=True)):
        if str(data.get("text", "")).strip().lower() in gold:
            lab[i] = True
    return lab if lab.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=2000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="gold_coverage.json")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    res = []
    for name, path in (("webqsp", os.path.join(WEBQSP_BASE, "train.gtds")),
                       ("webqsp_test", os.path.join(WEBQSP_BASE, "test.gtds")),
                       ("webqsp_qniso", os.path.join(WEBQSP_QNISO, "train.gtds"))):
        if not os.path.exists(path):
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        graphs = load_graphs(path)[:args.max_graphs or None]
        acc = {f"{o}_{s}": [0, 0] for o in ("dir", "und")
               for s in ("all", "to_gold", "from_gold", "to_gold_u")}
        n_graphs = n_gold = 0

        for G in graphs:
            n = G.number_of_nodes()
            if n < 2:
                continue
            gold = gold_mask(G)
            if gold is None:
                continue
            nodes = list(G.nodes())
            pos = {u: i for i, u in enumerate(nodes)}
            e = [(pos[u], pos[v]) for u, v in G.edges()]
            rows, cols = zip(*e) if e else ((), ())
            A = csr_matrix((np.ones(len(rows)), (list(rows), list(cols))), shape=(n, n))
            spd = shortest_path(A, method="D", unweighted=True, directed=True)
            spd_u = shortest_path(A, method="D", unweighted=True, directed=False)
            ncomp, comps = connected_components(A, directed=True, connection="weak")
            sizes = np.bincount(comps, minlength=ncomp)
            deg = np.asarray(A.sum(0)).ravel() + np.asarray(A.sum(1)).ravel()
            sig = {c: tuple(sorted(deg[comps == c].tolist())) for c in range(ncomp)}
            order = sorted(range(ncomp), key=lambda c: (-sizes[c], sig[c]))
            Gi = nx.DiGraph(); Gi.add_nodes_from(range(n)); Gi.add_edges_from(e)
            keys = _struct_keys(Gi, range(n))
            anchors = select_anchors("degree", args.k, comps, sizes, order, spd_u,
                                     {"degree": deg}, keys, np.random.default_rng(0))
            if len(anchors) == 0:
                continue

            for tag, sp in (("dir", spd), ("und", spd_u)):
                src, tgt = sp[:, anchors], sp[anchors, :].T
                o = (src[:, None, :] + tgt[None, :, :]).min(-1)
                vis = np.isfinite(o)
                gcol = np.broadcast_to(gold[None, :], vis.shape)
                grow = np.broadcast_to(gold[:, None], vis.shape)
                for s, m in (("all", np.ones_like(vis)),
                             ("to_gold", gcol), ("from_gold", grow),
                             ("to_gold_u", gcol & ~grow)):
                    acc[f"{tag}_{s}"][0] += int((vis & m).sum())
                    acc[f"{tag}_{s}"][1] += int(m.sum())
            n_graphs += 1
            n_gold += int(gold.sum())

        row = {"dataset": name, "k": args.k, "graphs": n_graphs,
               "gold_nodes_mean": n_gold / max(n_graphs, 1)}
        for key, (num, den) in acc.items():
            row[key] = num / max(den, 1)
        res.append(row)
        print(f"[{name}] {n_graphs} graphs, {time.time()-t0:.0f}s  " + json.dumps(
            {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}), flush=True)

    out = os.path.join(RESULTS, args.out)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")
    print(f"\n{'dataset':14s} {'dir:all':>8s} {'dir:->gold':>11s} {'dir:->gold_u':>13s} "
          f"{'und:all':>8s} {'und:->gold':>11s}")
    for r in res:
        print(f"{r['dataset']:14s} {r['dir_all']:8.4f} {r['dir_to_gold']:11.4f} "
              f"{r['dir_to_gold_u']:13.4f} {r['und_all']:8.4f} {r['und_to_gold']:11.4f}")


if __name__ == "__main__":
    main()
