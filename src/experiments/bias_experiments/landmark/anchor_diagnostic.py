"""Phase 0 for LANDMARK_BIAS.md — measure the FEATURE and the anchor rule, before
building the head.

    python -m src.experiments.bias_experiments.landmark.anchor_diagnostic

`LANDMARK_BIAS.md` proposes a per-node coordinate pair

    D_out[i,j] = SPD(i, a_j),   D_in[i,j] = SPD(a_j, i),    a_j in A, |A| = k

fed to a bilinear head that is initialised as a soft-min over the landmark
distance oracle o(u,v) = min_j [ D_out[u,j] + D_in[v,j] ]. Everything downstream
of D is a design question. D itself is fixed by the data and by the anchor rule,
so it is measured first.

Which Phase 0 this is, and which it is NOT. `linear_bias` fit its trained bias
offline, and its own Conclusion 6 is that offline imitation R^2 did not predict
trained quality — it measured how well a head could FIT a target. This measures
(a) whether the INPUT separates nodes at all and (b) how much of the true graph
metric the coordinates carry. (a) is a kill criterion in the sense
`factorized_rwpe` established: a node whose whole row is UNREACH gets the same
vector as every other such node under any head, at any width, at any LR. (b) is a
screen, not a ranking — the soft-min is only the INITIALISATION and training is
free to move F and G elsewhere, so high fidelity is evidence the metric is
present, while low fidelity with low degeneracy is ambiguous.

The quantities, and what each would settle:

  A. DEGENERACY — can the coordinates tell nodes apart?
     1. unreach_frac    — fraction of coordinate entries that are UNREACH. A rule
                          whose coordinates are mostly "no path" carries no metric.
     2. dead_node_frac  — nodes whose entire 2k row is UNREACH. The direct analogue
                          of `factorized_rwpe`'s 94.5% zero rows, and decisive the
                          same way.
     3. collision_frac  — nodes sharing an identical (D_out, D_in) row with another
                          node of the SAME graph. Those nodes are indistinguishable
                          to the bias by construction.
     4. alphabet        — histogram of the finite distances actually observed. This
                          is what sets d_max: LANDMARK_BIAS.md fixes 16, and SR
                          subgraphs are suspected to have diameter well under that,
                          which would leave most of the table dead.

  B. ORACLE FIDELITY — do the coordinates carry the metric?
     o(u,v) >= d(u,v) always (triangle inequality), so the error is one-sided and
     "gap" is the honest word. Reported STRATIFIED BY d(u,v), which is the reading
     that matters: the hypothesis is fine-grained LOCAL structure, so the gap at
     d in {1,2,3} decides it. An oracle exact at d=6 and useless at d=1 is worthless
     here, and an aggregate number would hide exactly that.
     5. exact_frac      — P(o == d). o = d iff some anchor lies on a shortest u->v
                          path, so this IS the anchor-on-shortest-path hit rate:
                          the mechanism itself, not a proxy. It is also why
                          betweenness is a candidate rule at all.
     6. gap_mean/p90    — how far off the oracle is where it is not exact.
     7. oracle_inf_frac — P(o = inf | d < inf). Coverage failure: no anchor lies on
                          any path at all, so the pair is invisible to the channel.

  C. COMPONENTS — the structure the allocation has to survive.
     8. weak component counts and sizes; how often a graph is disconnected at all.
     9. comps_without_anchor — components that get no anchor because #comps > k.
        Every node in one has an all-UNREACH row by construction.
    10. dir_unreach_within_comp — pairs in the SAME weak component with d = inf.
        This is the residual the component-stratified allocation CANNOT fix, and
        it is the reason the allocation uses weak and not strong components.

ANCHOR RULES. All are component-stratified (LANDMARK_BIAS.md §2): anchors are
apportioned across weakly connected components by size, then the rule runs within
each component. Without that, FPS is undefined on a disconnected graph (every
eccentricity is inf, so the tie-break decides everything) and the centrality rules
pool in the largest component.

  * `betweenness` — the rule the mechanism implies (see 5).
  * `pagerank`    — directed PageRank. Hubs: high coverage, low variance expected.
  * `fps`         — farthest-point / greedy k-center on the undirected skeleton.
                    Selection need not be directed; the COORDINATES are.
  * `degree`      — in+out degree, the cheap centrality proxy.
  * `mixed`       — half pagerank, half fps per component. The hedge.
  * `random`      — uniform. NOT a candidate: a function of the labelling, not of
                    the graph, so it breaks Property 1. It is here as the null that
                    says whether the rule matters at all — if it ties the best
                    structural rule, the whole selection apparatus can be deleted.

Output: one JSON row per (dataset, split, rule, k) under results/, plus printed
summaries. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESULTS = os.path.join(os.path.dirname(__file__), "results")

# The WebQSP caches the KGQA arms actually train on. Both are measured: `base` is
# the recipe `factorized_rwpe`'s Phase 0 used (so the two diagnostics are
# comparable), and `qniso` is the isolated-question-node recipe that produced the
# best graph-native run to date — the one a landmark arm would actually run in,
# and the one whose disconnected QUESTION node makes §C non-hypothetical.
WEBQSP_BASE = os.path.join(
    REPO, "src/experiments/kgqa/processed_datasets",
    "sr-webqsp_meta-llama-Llama-3.2-1B_vlast_1_cap512_nmax50_ver8"
    "_spd64_magq0.25m128_len1024_rcm1_seed42_dfv3",
)
WEBQSP_QNISO = WEBQSP_BASE + "_qnisolated"
GRAPHQA = os.path.join(REPO, "src/experiments/graphqa/processed_datasets/standard")
# The relational tasks — the ones a metric channel should move. `edge_count` is
# the global-aggregation control that a distance feature should NOT help.
GRAPHQA_TASKS = ("shortest_path", "connected_nodes", "edge_count")

RULES = ("betweenness", "pagerank", "fps", "degree", "mixed", "random")
K_VALUES = (8, 16, 32, 64)

D_CAP = 12          # true-distance strata; >= D_CAP folded into the last bucket
GAP_CAP = 12        # oracle-gap histogram cap
ALPHA_CAP = 24      # distance-alphabet histogram cap (must exceed any plausible d_max)
INF = np.inf


# ── anchor selection ──────────────────────────────────────────────────────────

def apportion(k: int, sizes: np.ndarray) -> np.ndarray:
    """Split k anchors across components by size — largest remainder, >=1 each while
    k allows, and never more than a component has nodes.

    Deterministic given (k, sizes, order). Callers pass components in a structural
    order so the result is a function of the graph, not of the labelling.
    """
    n = len(sizes)
    if n == 0:
        return np.zeros(0, dtype=int)
    alloc = np.zeros(n, dtype=int)
    if k <= n:
        # Not enough anchors for one each: the k largest components get one. Every
        # node in the rest is left with an all-UNREACH row, which §C counts.
        alloc[np.argsort(-sizes, kind="stable")[:k]] = 1
        return alloc

    alloc[:] = 1
    remaining = k - n
    # Largest-remainder on the surplus, then clamp-and-redistribute until either
    # the surplus is placed or every component is saturated at its own size.
    while remaining > 0:
        room = sizes - alloc
        live = room > 0
        if not live.any():
            break
        w = sizes * live
        share = w / w.sum() * remaining
        add = np.floor(share).astype(int)
        left = remaining - add.sum()
        if left > 0:
            rem = share - add
            rem[~live] = -1.0
            add[np.argsort(-rem, kind="stable")[:left]] += 1
        add = np.minimum(add, room)
        if add.sum() == 0:                       # nothing placeable this round
            break
        alloc += add
        remaining = k - alloc.sum()
    return alloc


def _struct_keys(G, nodes_all) -> dict:
    """A permutation-invariant tie-break key per node: (in-deg, out-deg, sorted
    neighbour degree multiset). Ties that survive this are between nodes that are
    plausibly isomorphic, where the choice does not affect any measured quantity.
    The final fallback is the index, and it is recorded here rather than hidden:
    the real implementation must carry the same policy and its equivariance is a
    correctness-gate test, not something this script can establish.
    """
    keys = {}
    for u in nodes_all:
        nbr = sorted(G.degree(v) for v in set(G.successors(u)) | set(G.predecessors(u)))
        keys[u] = (G.in_degree(u), G.out_degree(u), tuple(nbr))
    return keys


def _top(scores: np.ndarray, idx: np.ndarray, keys, count: int) -> list:
    """The `count` highest-scoring members of `idx`, ties broken structurally."""
    order = sorted(idx, key=lambda i: (-scores[i], keys[i]))
    return order[:count]


def _fps(sub: np.ndarray, idx: np.ndarray, count: int, keys) -> list:
    """Greedy k-center on the undirected in-component distances `sub` (len(idx)^2).

    Seeded at the max-eccentricity node so the start is structural, not arbitrary —
    on a disconnected graph that quantity is inf for every node, which is exactly
    why the caller runs this per component.
    """
    m = len(idx)
    if count >= m:
        return list(idx)
    ecc = sub.max(1)
    start = min(range(m), key=lambda p: (-ecc[p], -sub[p].sum(), keys[idx[p]]))
    sel = [start]
    dmin = sub[start].copy()
    while len(sel) < count:
        nxt = min((p for p in range(m) if p not in sel),
                  key=lambda p: (-dmin[p], keys[idx[p]]))
        sel.append(nxt)
        dmin = np.minimum(dmin, sub[nxt])
    return [idx[p] for p in sel]


def select_anchors(rule, k, comps, sizes, order, spd_u, scores, keys, rng) -> np.ndarray:
    """Component-stratified anchor selection. Returns node indices, length <= k."""
    alloc = apportion(k, sizes[order])
    picked = []
    for slot, c in enumerate(order):
        count = int(alloc[slot])
        if count == 0:
            continue
        idx = np.flatnonzero(comps == c)
        if count >= len(idx):
            picked.extend(idx.tolist())
            continue
        if rule == "random":
            picked.extend(rng.choice(idx, size=count, replace=False).tolist())
        elif rule == "fps":
            picked.extend(_fps(spd_u[np.ix_(idx, idx)], idx, count, keys))
        elif rule == "mixed":
            half = count // 2
            a = _top(scores["pagerank"], idx, keys, half)
            rest = np.array([i for i in idx if i not in set(a)])
            b = _fps(spd_u[np.ix_(rest, rest)], rest, count - half, keys)
            picked.extend(a + b)
        else:
            picked.extend(_top(scores[rule], idx, keys, count))
    return np.array(sorted(picked), dtype=int)


# ── per-graph measurement ─────────────────────────────────────────────────────

class Acc:
    """Streaming accumulator for one (dataset, split, rule, k) cell."""

    def __init__(self):
        self.coord_entries = 0
        self.coord_unreach = 0
        self.nodes = 0
        self.dead_nodes = 0
        self.colliding = 0
        self.graphs = 0
        self.comps_total = 0
        self.comps_no_anchor = 0
        self.anchors_total = 0
        # gap[d, g]: pairs at true distance d with oracle gap g; last gap column is inf
        self.gap = np.zeros((D_CAP + 2, GAP_CAP + 2), dtype=np.int64)
        self.alphabet = np.zeros(ALPHA_CAP + 2, dtype=np.int64)
        # ── channel 3: the undirected block, measured against the UNDIRECTED metric
        self.gap_und = np.zeros((D_CAP + 2, GAP_CAP + 2), dtype=np.int64)
        self.dead_3k = 0
        self.colliding_3k = 0
        # Pair VISIBILITY: does the channel say anything at all about this pair?
        # This is the quantity the third channel is bought for — it is not a
        # fidelity measure, it is a coverage measure, and on a KG the two diverge
        # by more than an order of magnitude.
        self.pairs_all = 0
        self.visible_dir = 0
        self.visible_und = 0

    def to_dict(self, dcap=D_CAP):
        e = max(self.coord_entries, 1)
        n = max(self.nodes, 1)
        tot = self.gap.sum()
        finite = self.gap[:dcap + 1].sum()
        out = {
            "graphs": self.graphs,
            "nodes": self.nodes,
            "anchors_mean": self.anchors_total / max(self.graphs, 1),
            "unreach_frac": self.coord_unreach / e,
            "dead_node_frac": self.dead_nodes / n,
            "collision_frac": self.colliding / n,
            "comps_no_anchor_frac": self.comps_no_anchor / max(self.comps_total, 1),
        }
        if finite:
            g = self.gap[:dcap + 1]
            out["exact_frac"] = float(g[:, 0].sum() / finite)
            out["oracle_inf_frac"] = float(g[:, -1].sum() / finite)
            gv = np.arange(GAP_CAP + 2, dtype=float)
            gv[-1] = np.nan                       # inf gaps excluded from the mean
            fin = g[:, :-1]
            out["gap_mean"] = float((fin * gv[:-1]).sum() / max(fin.sum(), 1))
        # Stratified by true distance — the reading that decides the hypothesis.
        strata = {}
        for d in range(1, min(7, dcap + 1)):
            row = self.gap[d]
            if row.sum() == 0:
                continue
            fin = row[:-1]
            strata[str(d)] = {
                "pairs": int(row.sum()),
                "exact_frac": float(row[0] / row.sum()),
                "inf_frac": float(row[-1] / row.sum()),
                "gap_mean": float((fin * np.arange(GAP_CAP + 1)).sum() / max(fin.sum(), 1)),
            }
        out["by_distance"] = strata
        out["alphabet"] = (self.alphabet / max(self.alphabet.sum(), 1)).round(5).tolist()
        out["pairs_finite"] = int(finite)
        out["pairs_total"] = int(tot)

        # ── channel 3 ────────────────────────────────────────────────────────
        gu = self.gap_und[:dcap + 1]
        fu = gu.sum()
        if fu:
            out["und_exact_frac"] = float(gu[:, 0].sum() / fu)
            out["und_inf_frac"] = float(gu[:, -1].sum() / fu)
            fin = gu[:, :-1]
            out["und_gap_mean"] = float(
                (fin * np.arange(GAP_CAP + 1)).sum() / max(fin.sum(), 1))
        us = {}
        for d in range(1, min(7, dcap + 1)):
            row = self.gap_und[d]
            if row.sum() == 0:
                continue
            us[str(d)] = {"pairs": int(row.sum()),
                          "exact_frac": float(row[0] / row.sum())}
        out["und_by_distance"] = us
        out["dead_node_frac_3k"] = self.dead_3k / n
        out["collision_frac_3k"] = self.colliding_3k / n
        p = max(self.pairs_all, 1)
        out["pair_visible_dir"] = self.visible_dir / p
        out["pair_visible_und"] = self.visible_und / p
        return out


def measure_graph(acc: Acc, spd, spd_u, anchors, n):
    """Fold one graph's coordinates and oracle into the accumulator."""
    if len(anchors) == 0:
        acc.dead_nodes += n
        acc.dead_3k += n
        acc.nodes += n
        return
    d_out = spd[:, anchors]                       # (n, k) i -> a_j
    d_in = spd[anchors, :].T                      # (n, k) a_j -> i
    d_und = spd_u[:, anchors]                     # (n, k) symmetric skeleton

    live = np.isfinite(d_out)
    acc.coord_entries += d_out.size + d_in.size
    acc.coord_unreach += int((~live).sum() + (~np.isfinite(d_in)).sum())
    acc.anchors_total += len(anchors)
    acc.nodes += n
    acc.dead_nodes += int((~(live | np.isfinite(d_in))).all(1).sum())

    fin = d_out[np.isfinite(d_out)]
    if fin.size:
        acc.alphabet += np.bincount(np.minimum(fin, ALPHA_CAP + 1).astype(int),
                                    minlength=ALPHA_CAP + 2)

    # Identical (D_out, D_in) rows are indistinguishable to the bias. Measured for
    # the directed 2k form and again for the 3k form, so the third channel's
    # contribution to node discriminability is priced, not assumed.
    def _collide(mat):
        r = np.where(np.isfinite(mat), mat, -1.0)
        _, inv, cnt = np.unique(r, axis=0, return_inverse=True, return_counts=True)
        return int((cnt[inv] > 1).sum())

    acc.colliding += _collide(np.concatenate([d_out, d_in], 1))
    acc.colliding_3k += _collide(np.concatenate([d_out, d_in, d_und], 1))
    acc.dead_3k += int((~(np.isfinite(d_out) | np.isfinite(d_in)
                          | np.isfinite(d_und))).all(1).sum())

    # o(u,v) = min_j [ D_out[u,j] + D_in[v,j] ], chunked over u to bound memory.
    # Channel 3 is the same reduction on the symmetric skeleton, scored against the
    # UNDIRECTED truth — it approximates a different metric, so mixing the two into
    # one gap statistic would be meaningless.
    step = max(1, int(4e6 // max(n * len(anchors), 1)))
    for s in range(0, n, step):
        for src, tgt, truth, hist in (
            (d_out[s:s + step], d_in, spd[s:s + step], acc.gap),
            (d_und[s:s + step], d_und, spd_u[s:s + step], acc.gap_und),
        ):
            o = (src[:, None, :] + tgt[None, :, :]).min(-1)            # (chunk, n)
            db = np.where(np.isfinite(truth), np.minimum(truth, D_CAP),
                          D_CAP + 1).astype(int)
            # Only subtract where truth is finite: inf - inf is nan, and those pairs
            # land in the d = inf stratum every finite statistic already excludes.
            gap = np.full_like(o, INF)
            np.subtract(o, truth, out=gap, where=np.isfinite(truth))
            gb = np.where(np.isfinite(gap), np.minimum(gap, GAP_CAP),
                          GAP_CAP + 1).astype(int)
            np.add.at(hist, (db.ravel(), gb.ravel()), 1)
            if hist is acc.gap:
                acc.pairs_all += o.size
                acc.visible_dir += int(np.isfinite(o).sum())
            else:
                acc.visible_und += int(np.isfinite(o).sum())


# ── the run ───────────────────────────────────────────────────────────────────

def load_graphs(gtds_dir: str):
    """Graphs only — `graphs.pkl` is the nx topology; the Arrow table carries the
    N^2 SPD column and is not needed (APSP is recomputed here so the diagnostic
    does not inherit the stored matrix's own clipping)."""
    with open(os.path.join(gtds_dir, "graphs.pkl"), "rb") as f:
        return pickle.load(f)


def run_split(graphs, rules, ks, seed, comp_stats):
    accs = {(r, k): Acc() for r in rules for k in ks}
    rng = np.random.default_rng(seed)

    for gi, G in enumerate(graphs):
        n = G.number_of_nodes()
        if n < 2:
            continue
        nodes = list(G.nodes())
        pos = {u: i for i, u in enumerate(nodes)}
        rows, cols = zip(*(((pos[u], pos[v]) for u, v in G.edges()))) if G.number_of_edges() \
            else ((), ())
        A = csr_matrix((np.ones(len(rows)), (list(rows), list(cols))), shape=(n, n))

        spd = shortest_path(A, method="D", unweighted=True, directed=True)
        spd_u = shortest_path(A, method="D", unweighted=True, directed=False)
        ncomp, comps = connected_components(A, directed=True, connection="weak")
        sizes = np.bincount(comps, minlength=ncomp)

        # Components in a structural order (size desc, then their sorted degree
        # multiset) so the apportionment is a function of the graph.
        deg = np.asarray(A.sum(0)).ravel() + np.asarray(A.sum(1)).ravel()
        sig = {c: tuple(sorted(deg[comps == c].tolist())) for c in range(ncomp)}
        order = sorted(range(ncomp), key=lambda c: (-sizes[c], sig[c]))

        comp_stats["graphs"] += 1
        comp_stats["multi_comp"] += int(ncomp > 1)
        comp_stats["comps"] += ncomp
        comp_stats["largest_frac"] += float(sizes.max() / n)
        same = comps[:, None] == comps[None, :]
        comp_stats["pairs_same_comp"] += int(same.sum())
        comp_stats["pairs_same_comp_dir_inf"] += int((same & ~np.isfinite(spd)).sum())

        Gi = nx.DiGraph()
        Gi.add_nodes_from(range(n))
        Gi.add_edges_from(zip(rows, cols))
        keys = _struct_keys(Gi, range(n))
        scores = {
            "degree": deg,
            "pagerank": np.array([v for _, v in sorted(
                nx.pagerank(Gi, alpha=0.85).items())]),
            "betweenness": np.array([v for _, v in sorted(
                nx.betweenness_centrality(Gi, normalized=True).items())]),
        }

        for rule in rules:
            for k in ks:
                anchors = select_anchors(rule, k, comps, sizes, order, spd_u,
                                         scores, keys, np.random.default_rng(seed + gi))
                acc = accs[(rule, k)]
                acc.graphs += 1
                acc.comps_total += ncomp
                acc.comps_no_anchor += int(ncomp - len(set(comps[anchors].tolist())))
                measure_graph(acc, spd, spd_u, anchors, n)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=2000,
                    help="graphs per split (0 = all); APSP + betweenness are per-graph")
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--k", nargs="+", type=int, default=list(K_VALUES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None, help="dataset names to run")
    ap.add_argument("--out", default="anchor_diagnostic.json")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    targets = [
        ("webqsp", "train", os.path.join(WEBQSP_BASE, "train.gtds")),
        ("webqsp", "test", os.path.join(WEBQSP_BASE, "test.gtds")),
        ("webqsp_qniso", "train", os.path.join(WEBQSP_QNISO, "train.gtds")),
    ]
    for task in GRAPHQA_TASKS:
        targets.append((f"graphqa_{task}", "train", os.path.join(GRAPHQA, task, "train.gtds")))
    if args.only:
        targets = [t for t in targets if t[0] in set(args.only)]

    all_res = []
    for name, split, path in targets:
        if not os.path.exists(path):
            print(f"[skip] {name}/{split}: {path} not found", flush=True)
            continue
        t0 = time.time()
        graphs = load_graphs(path)
        if args.max_graphs:
            graphs = graphs[:args.max_graphs]
        print(f"[{name}/{split}] {len(graphs)} graphs in {time.time()-t0:.1f}s", flush=True)

        comp_stats = defaultdict(float)
        t1 = time.time()
        accs = run_split(graphs, args.rules, args.k, args.seed, comp_stats)
        print(f"[{name}/{split}] measured in {time.time()-t1:.1f}s", flush=True)
        del graphs

        g = max(comp_stats["graphs"], 1)
        topo = {
            "comp_graphs": int(comp_stats["graphs"]),
            "multi_component_frac": comp_stats["multi_comp"] / g,
            "components_mean": comp_stats["comps"] / g,
            "largest_component_frac": comp_stats["largest_frac"] / g,
            # The residual the allocation cannot fix: same weak component, no
            # directed path. Weak connectivity is not directed reachability.
            "dir_unreach_within_comp": (comp_stats["pairs_same_comp_dir_inf"]
                                        / max(comp_stats["pairs_same_comp"], 1)),
        }
        print(f"[{name}/{split}] topology: {json.dumps(topo)}", flush=True)

        for (rule, k), acc in accs.items():
            r = {"dataset": name, "split": split, "rule": rule, "k": k,
                 **acc.to_dict(), **topo}
            all_res.append(r)

    out = os.path.join(RESULTS, args.out)
    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {out}")

    f3 = lambda v: ("%.3f" % v) if isinstance(v, float) else "-"
    hdr = (f"{'dataset/split':22s} {'rule':12s} {'k':>3s} {'unrch':>6s} {'dead':>6s} "
           f"{'coll':>6s} {'exact':>6s} {'gap':>6s} {'oInf':>6s} "
           f"{'ex@1':>6s} {'ex@2':>6s} {'ex@3':>6s}")
    print("\n── directed channels (1-2) ──\n" + hdr)
    for r in all_res:
        bd = r.get("by_distance", {})
        ex = lambda d: f3((bd.get(d) or {}).get("exact_frac"))
        print(f"{r['dataset']+'/'+r['split']:22s} {r['rule']:12s} {r['k']:3d} "
              f"{f3(r['unreach_frac']):>6s} {f3(r['dead_node_frac']):>6s} "
              f"{f3(r['collision_frac']):>6s} {f3(r.get('exact_frac')):>6s} "
              f"{f3(r.get('gap_mean')):>6s} {f3(r.get('oracle_inf_frac')):>6s} "
              f"{ex('1'):>6s} {ex('2'):>6s} {ex('3'):>6s}")

    # What the third channel buys, next to what it costs (2k -> 3k head dims).
    hdr2 = (f"{'dataset/split':22s} {'rule':12s} {'k':>3s} {'uExact':>7s} {'uGap':>6s} "
            f"{'uEx@1':>6s} {'coll2k':>7s} {'coll3k':>7s} {'dead3k':>7s} "
            f"{'visDir':>7s} {'visUnd':>7s}")
    print("\n── channel 3 (undirected) and what it adds ──\n" + hdr2)
    for r in all_res:
        ub = r.get("und_by_distance", {})
        print(f"{r['dataset']+'/'+r['split']:22s} {r['rule']:12s} {r['k']:3d} "
              f"{f3(r.get('und_exact_frac')):>7s} {f3(r.get('und_gap_mean')):>6s} "
              f"{f3((ub.get('1') or {}).get('exact_frac')):>6s} "
              f"{f3(r['collision_frac']):>7s} {f3(r['collision_frac_3k']):>7s} "
              f"{f3(r['dead_node_frac_3k']):>7s} "
              f"{f3(r['pair_visible_dir']):>7s} {f3(r['pair_visible_und']):>7s}")


if __name__ == "__main__":
    main()
