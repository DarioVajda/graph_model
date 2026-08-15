"""Phase 0 for FACTORIZED_RWPE_BIAS.md — measure the FEATURE, before building the head.

    python -m src.experiments.bias_experiments.factorized_rwpe.feature_diagnostic

`FACTORIZED_RWPE_BIAS.md` §3 proposes a per-node encoding built from the diagonals
of directed random-walk transition matrices,

    p_i = [ (M_fwd^t)_ii  for t=1..k ]  ||  [ (M_bwd^t)_ii  for t=1..k ],
    M_fwd = D_out^-1 A,   M_bwd = D_in^-1 A^T,   k = 24,

which is then MLP'd to z_i and factorized into Q/K. Everything downstream of p is
a design question. p itself is a property of the DATA, and it is measurable in
minutes, so it is measured first.

Why this Phase 0 and not the one `linear_bias` ran. That one fit the trained bias
offline and its own Conclusion 6 is that offline imitation R^2 did not predict
trained quality — it measured how well a head could FIT a target, which is a
statement about the head. This measures whether the INPUT distinguishes nodes at
all. A feature that is identically zero on a graph cannot be rescued by any head,
any width, or any learning rate, so a null here is decisive in a way a poor fit
never was.

The quantities, and what each would settle:

  1. zero rows        — a node with p_i = 0 gets z_i = MLP(0), the same vector as
                        every other zero node. The encoding cannot separate them.
                        (M^t)_ii is the return probability of a length-t walk, so
                        p_i = 0 for every node that sits on no directed cycle.
  2. all-zero graphs  — every node zero => the structural bias is one constant per
                        head for that whole example, i.e. no bias at all.
  3. fwd == bwd       — the two halves of the directed block differ only in whether
                        closed walks are weighted by 1/out-degree or 1/in-degree,
                        over the SAME cycle set. Where in- and out-degrees agree the
                        second 24 dims are a literal copy of the first. Counted over
                        non-zero rows only, so it is not just re-reporting (1).
  4. live t           — which of the k walk lengths are ever non-zero. Two distinct
                        deaths show up here: directed cycles are long and rare, and
                        a BIPARTITE graph (which the entity->relation->entity Levi
                        graph is, unless CVT collapse or the PROMPT node breaks it)
                        has no odd cycles at all, killing every odd t.
  5. effective rank   — PCA on the pooled node x d matrix. How many dimensions of
                        the encoding carry variance.
  6. degree R^2       — between-group variance of p when nodes are grouped by their
                        exact (in-deg, out-deg) pair, i.e. the fraction of the
                        encoding that is a function of degree alone and nothing
                        more. `mixed_bias` measured a per-node channel to recover
                        ~100% of GraphQA `node_degree` and 1.8% of WebQSP headroom;
                        a high number here says this feature is the same object.
                        Note diag(M_sym^2)_ii = 1/deg(i) exactly, so the undirected
                        block is expected to score high BY CONSTRUCTION — the
                        question is how much higher than the directed one.
  7. answer AUC       — WebQSP only, and the one that speaks to the task. Can a
                        classifier on p separate gold-answer nodes from the rest,
                        and does it beat a degree-only classifier? WebQSP's metric
                        is whether the model names the answer entity, so a feature
                        that does not separate answer nodes from their neighbours
                        cannot drive that metric however it is projected.

FEATURE SETS. The spec's directed block is measured next to two additions, because
the choice is open and it changes the answer:

  * `dir`        — the doc as written. D_out^-1 is undefined at a sink; those rows
                   are left at zero (the only reading that does not invent edges).
  * `dir_sink`   — `utils/rrwp.py:38`'s convention: a self-loop on every sink. This
                   is what a reimplementation reusing that helper gets by default,
                   and a self-loop makes (M^t)_ii = 1 for all t — a maximal
                   "structural role" manufactured out of a node having no out-edges.
  * `undir`      — the graph symmetrized first, i.e. standard undirected RWSE. On an
                   undirected graph every edge is a 2-cycle, so nothing is
                   structurally zero. This is BOTH the control that separates
                   "directed return probabilities are degenerate here" from
                   "diagonal-only features are degenerate here", AND a proposed
                   third input block in its own right.
  * `dir+undir`  — the concatenation, 3k dims. Free in attention width (d_pos is
                   unchanged; only MLP_PE's input grows), so the question is not
                   cost but whether the union separates anything the parts do not.

Output: one JSON row per (dataset, split, feature set) under results/, plus a
printed summary. CPU only, a few minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RESULTS = os.path.join(os.path.dirname(__file__), "results")

# The WebQSP cache `021`/`023` trained on: dataset webqsp, cap 512, n_max 50,
# rel_mode last_1, cvt collapsed, seed 42, dfv3 (see the config's data block and
# kgqa/config.py:520's key builder). Reading the arm's OWN graphs is the point —
# a diagnostic on a differently-built cache would not be about those results.
WEBQSP = os.path.join(
    REPO, "src/experiments/kgqa/processed_datasets",
    "sr-webqsp_meta-llama-Llama-3.2-1B_vlast_1_cap512_nmax50_ver8"
    "_spd64_magq0.25m128_len1024_rcm1_seed42_dfv3",
)
# GraphQA is the contrast that validates the metric rather than the proposal:
# `mixed_bias` measured the per-node channel at ~100% of headroom on node_degree
# and 39% on shortest_path. A diagnostic that cannot see that split is not
# measuring anything useful about WebQSP either.
GRAPHQA = os.path.join(REPO, "src/experiments/graphqa/processed_datasets/standard")

K_STEPS = 24          # FACTORIZED_RWPE_BIAS.md §3 default
EPS = 1e-12

FEATURE_SETS = {
    "dir":       ("fwd", "bwd"),
    "dir_sink":  ("fwd_sink", "bwd_sink"),
    "undir":     ("undir",),
    "dir+undir": ("fwd", "bwd", "undir"),
}


# ── feature computation ───────────────────────────────────────────────────────

def _return_probs(M: torch.Tensor, k: int) -> torch.Tensor:
    """[ (M^t)_ii for t=1..k ] -> (B, N, k). fp64, diagonal only."""
    B, N, _ = M.shape
    P = torch.eye(N, dtype=M.dtype, device=M.device).expand(B, N, N).clone()
    cols = []
    for _ in range(k):
        P = torch.bmm(P, M)
        cols.append(torch.diagonal(P, dim1=1, dim2=2))
    return torch.stack(cols, dim=2)


def _transition(M: torch.Tensor) -> torch.Tensor:
    deg = M.sum(2, keepdim=True)
    return M / deg.clamp(min=EPS)          # deg == 0 -> an all-zero row, kept zero


def rwpe_blocks(A: torch.Tensor, n: torch.Tensor, k: int) -> dict:
    """Every candidate block for a padded batch, computed once.

    A: (B, N, N) float64 adjacency, A[b,i,j] = 1 for i->j, zero outside n[b].
    Returns name -> (B, N, k), zero on padded rows.

    The undirected block needs only ONE power series: for a symmetric A the in-
    and out-degrees coincide, so M_bwd = M_fwd exactly and a backward series would
    be a bit-identical copy.
    """
    B, N, _ = A.shape
    dev = A.device
    real = (torch.arange(N, device=dev).unsqueeze(0) < n.unsqueeze(1))

    A_sink = A.clone()
    # utils/rrwp.py:38 — self-loop every real node with out-degree 0.
    sinks = (A.sum(2) == 0) & real
    eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
    A_sink[sinks.unsqueeze(-1) & eye] = 1.0

    A_sym = ((A + A.transpose(1, 2)) > 0).to(A.dtype)

    blocks = {
        "fwd":       _return_probs(_transition(A), k),
        "bwd":       _return_probs(_transition(A.transpose(1, 2)), k),
        "fwd_sink":  _return_probs(_transition(A_sink), k),
        "bwd_sink":  _return_probs(_transition(A_sink.transpose(1, 2)), k),
        "undir":     _return_probs(_transition(A_sym), k),
    }
    m = real.unsqueeze(-1)
    return {name: b * m for name, b in blocks.items()}


# ── metrics ───────────────────────────────────────────────────────────────────

def effective_rank(X: np.ndarray):
    """PCA components needed for 90 / 99% of the variance of the pooled features."""
    if X.shape[0] < 2 or not np.isfinite(X).all():
        return None
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    if (s ** 2).sum() <= 0:
        return {"pc90": 0, "pc99": 0, "dims": int(X.shape[1]), "note": "zero variance"}
    frac = np.cumsum(s ** 2) / (s ** 2).sum()
    return {
        "pc90": int(np.searchsorted(frac, 0.90) + 1),
        "pc99": int(np.searchsorted(frac, 0.99) + 1),
        "dims": int(X.shape[1]),
    }


def degree_r2(X: np.ndarray, deg: np.ndarray):
    """Fraction of the encoding's variance that is a pure function of (in, out) degree.

    Group nodes by their exact degree pair and take between-group / total sum of
    squares. This is the R^2 of the BEST POSSIBLE function of the degree pair, not
    of a particular regression — so a high value cannot be blamed on model choice.
    """
    if X.shape[0] < 2:
        return None
    total = float(((X - X.mean(0, keepdims=True)) ** 2).sum())
    if total <= 0:
        return {"r2": None, "note": "zero variance", "groups": 0}
    keys = defaultdict(list)
    for i, d in enumerate(map(tuple, deg)):
        keys[d].append(i)
    mu = X.mean(0, keepdims=True)
    between = sum(len(m) * float(((X[m].mean(0, keepdims=True) - mu) ** 2).sum())
                  for m in keys.values())
    return {"r2": between / total, "groups": len(keys), "n": int(X.shape[0])}


def degree_basis(D: np.ndarray) -> np.ndarray:
    """A FAIR degree baseline: in/out degree plus the transforms a random-walk
    diagonal actually contains.

    The first pass used a linear probe on raw (in, out) and it was too weak to be
    a reference: (M^t)_ii is built from RECIPROCAL degrees, and the single most
    predictive fact in a Levi graph — "this node has no out-edges", i.e. it is a
    tail entity — is a threshold, not a monotone direction. A feature set that
    beats raw-degree while scoring degree_r2 = 0.999 has not found new
    information; it has found a better parameterization of the same information,
    and only this basis can tell the two apart.
    """
    in_d, out_d = D[:, 0], D[:, 1]
    return np.stack([
        in_d, out_d,
        np.log1p(in_d), np.log1p(out_d),
        1.0 / (1.0 + in_d), 1.0 / (1.0 + out_d),
        (in_d == 0).astype(float), (out_d == 0).astype(float),
    ], axis=1)


def auc(scores: np.ndarray, labels: np.ndarray):
    """Rank AUC with ties averaged. labels in {0,1}."""
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def logistic_auc(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Cross-validated AUC of a logistic probe. Held out, so it measures signal, not fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if y.sum() < 20 or (1 - y).sum() < 20:
        return None
    scores = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        scores[te] = clf.decision_function(sc.transform(X[te]))
    return auc(scores, y)


# ── dataset loading ───────────────────────────────────────────────────────────

def load_graphs(gtds_dir: str):
    """Graphs only — `graphs.pkl` is the nx topology, no need to memory-map the
    Arrow feature table (which carries the N^2 SPD column)."""
    with open(os.path.join(gtds_dir, "graphs.pkl"), "rb") as f:
        return pickle.load(f)


def answer_labels(G):
    """1 for nodes whose text is a gold answer, else 0. None when the graph carries
    no gold list (non-KGQA datasets) or none of them landed in G.

    `process_dataset.py:370` stashes `gold_answers` (already restricted to answers
    grounded in G by `present_answer_texts`) on the graph, and every node carries
    `text`, so the match is the same one the evaluator scores on.
    """
    gold = G.graph.get("gold_answers")
    if not gold:
        return None
    gold = {str(g).strip().lower() for g in gold}
    lab = np.zeros(G.number_of_nodes(), dtype=np.int64)
    for i, (_, data) in enumerate(G.nodes(data=True)):
        if str(data.get("text", "")).strip().lower() in gold:
            lab[i] = 1
    return lab if lab.sum() else None


def to_batch(graphs, dev):
    """Pad a chunk of nx graphs into (A, n)."""
    ns = [g.number_of_nodes() for g in graphs]
    N = max(ns)
    A = torch.zeros(len(graphs), N, N, dtype=torch.float64, device=dev)
    for b, g in enumerate(graphs):
        pos = {u: i for i, u in enumerate(g.nodes())}
        for u, v in g.edges():
            A[b, pos[u], pos[v]] = 1.0
    return A, torch.tensor(ns, device=dev)


def is_bipartite(A_row: torch.Tensor, n: int) -> bool:
    """Two-colour the symmetrized graph. Bipartite => no odd closed walks => every
    odd-t entry of the undirected block is identically zero."""
    S = ((A_row[:n, :n] + A_row[:n, :n].T) > 0).cpu().numpy()
    colour = -np.ones(n, dtype=np.int8)
    for root in range(n):
        if colour[root] >= 0:
            continue
        colour[root] = 0
        stack = [root]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(S[u]):
                if colour[v] < 0:
                    colour[v] = 1 - colour[u]
                    stack.append(v)
                elif colour[v] == colour[u]:
                    return False
    return True


# ── the run ───────────────────────────────────────────────────────────────────

def collect(graphs, k, chunk_elems, want_answers, dev):
    """One pass over the dataset: every block, plus degrees, labels and topology stats."""
    per_set = {name: [] for name in FEATURE_SETS}
    degs, labels, node_counts = [], [], []
    n_bipartite = 0

    sizes = np.array([g.number_of_nodes() for g in graphs])
    order = np.argsort(sizes)                                   # pad-efficient chunks
    start = 0
    while start < len(order):
        # Size the chunk by padded ELEMENTS, not graph count: WebQSP's cap is 512
        # nodes, and a fixed count would swing peak memory by ~70x across the
        # size-sorted order. The chunk pads to its LAST (largest) member, which is
        # what the count has to be solved against.
        N = int(sizes[order[start]])
        size = max(1, min(256, int(chunk_elems / max(N * N, 1))))
        N = int(sizes[order[min(start + size, len(order)) - 1]])
        size = max(1, min(256, int(chunk_elems / max(N * N, 1))))
        sel = [graphs[i] for i in order[start:start + size]]
        start += size

        A, n = to_batch(sel, dev)
        blocks = rwpe_blocks(A, n, k)
        real = (torch.arange(A.shape[1], device=dev).unsqueeze(0) < n.unsqueeze(1))
        flat = real.flatten()

        for name, parts in FEATURE_SETS.items():
            X = torch.cat([blocks[p] for p in parts], dim=2)
            per_set[name].append(X.flatten(0, 1)[flat].cpu().numpy())

        out_d, in_d = A.sum(2), A.sum(1)
        degs.append(torch.stack([in_d, out_d], -1).flatten(0, 1)[flat].cpu().numpy())
        node_counts.extend(n.cpu().tolist())
        for b, g in enumerate(sel):
            n_bipartite += int(is_bipartite(A[b], g.number_of_nodes()))
            if want_answers:
                lab = answer_labels(g)
                labels.append(np.full(g.number_of_nodes(), -1, dtype=np.int64)
                              if lab is None else lab)

    return (
        {name: np.concatenate(v) for name, v in per_set.items()},
        np.concatenate(degs),
        np.concatenate(labels) if labels else None,
        {"graphs": len(graphs),
         "nodes_per_graph_median": float(np.median(node_counts)),
         "bipartite_graph_frac": n_bipartite / max(len(graphs), 1)},
    )


def analyse(name, X, D, y, k, topo, rng):
    n_nodes = len(X)
    nz = np.abs(X).max(1) > EPS
    half = X.shape[1] // 2

    res = {
        "feature_set": name,
        "dims": int(X.shape[1]),
        "k": k,
        "nodes": n_nodes,
        "zero_row_frac": float((~nz).mean()),
        "live_t_frac": (np.abs(X) > EPS).mean(0).round(6).tolist(),
        **topo,
    }
    # fwd == bwd only means anything for the two directed sets, and only among rows
    # that are alive at all (an all-zero row trivially matches itself).
    if name.startswith("dir") and "+" not in name and nz.any():
        d = np.abs(X[nz][:, :half] - X[nz][:, half:]).max(1)
        res["fwd_eq_bwd_frac_of_nonzero"] = float((d <= EPS).mean())

    sub = rng.choice(n_nodes, size=min(200_000, n_nodes), replace=False)
    res["effective_rank"] = effective_rank(X[sub])
    res["degree_r2"] = degree_r2(X[sub], D[sub])

    if y is not None:
        mask = y >= 0
        Xa, ya, Da = X[mask], y[mask], D[mask].astype(float)
        s = rng.choice(len(ya), size=min(200_000, len(ya)), replace=False)
        res["answer_nodes"] = int(ya.sum())
        res["answer_candidates"] = int(len(ya))
        res["answer_zero_row_frac"] = float(
            (np.abs(Xa[ya == 1]).max(1) <= EPS).mean()) if ya.sum() else None
        Ba = degree_basis(Da)
        res["answer_auc_rwpe"] = logistic_auc(Xa[s], ya[s])
        res["answer_auc_degree_raw"] = logistic_auc(Da[s], ya[s])
        res["answer_auc_degree_basis"] = logistic_auc(Ba[s], ya[s])
        # The single bit `out_degree == 0` — "this node is a tail entity". No
        # probe, no fitting: the raw indicator scored as a ranking.
        res["answer_auc_is_sink"] = auc((Da[:, 1] == 0).astype(float), ya)
        res["answer_auc_rwpe_plus_degree"] = logistic_auc(
            np.concatenate([Xa[s], Ba[s]], 1), ya[s])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K_STEPS)
    ap.add_argument("--chunk-elems", type=float, default=4e6,
                    help="padded N*N elements per chunk (memory knob)")
    ap.add_argument("--max-graphs", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", nargs="+", default=None,
                    help="dataset names to run (default: all)")
    ap.add_argument("--out", default="feature_diagnostic.json")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    dev = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    rng = np.random.default_rng(0)

    targets = [("webqsp", "train", os.path.join(WEBQSP, "train.gtds"), True),
               ("webqsp", "test", os.path.join(WEBQSP, "test.gtds"), True)]
    for task in ("node_degree", "shortest_path", "edge_count"):
        targets.append((f"graphqa_{task}", "train",
                        os.path.join(GRAPHQA, task, "train.gtds"), False))

    if args.only:
        targets = [t for t in targets if t[0] in set(args.only)]

    all_res = []
    for name, split, path, want_answers in targets:
        if not os.path.exists(path):
            print(f"[skip] {name}/{split}: {path} not found", flush=True)
            continue
        t0 = time.time()
        graphs = load_graphs(path)
        if args.max_graphs:
            graphs = graphs[:args.max_graphs]
        print(f"[{name}/{split}] {len(graphs)} graphs loaded in {time.time()-t0:.1f}s",
              flush=True)

        t1 = time.time()
        sets, D, y, topo = collect(graphs, args.k, args.chunk_elems, want_answers, dev)
        print(f"[{name}/{split}] features in {time.time()-t1:.1f}s", flush=True)
        del graphs

        for fs, X in sets.items():
            r = analyse(fs, X, D, y, args.k, topo, rng)
            r.update(dataset=name, split=split)
            all_res.append(r)
            print(json.dumps(r), flush=True)

    out = os.path.join(RESULTS, args.out)
    with open(out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nwrote {out}")

    print(f"\n{'dataset/split':27s} {'feature set':10s} {'dims':>4s} {'zero':>6s} "
          f"{'ansZero':>7s} {'degR2':>6s} {'pc90':>4s} {'ansAUC':>7s} {'degAUC':>7s} "
          f"{'sinkAUC':>7s}")
    for r in all_res:
        er = r.get("effective_rank") or {}
        dr = r.get("degree_r2") or {}
        f2 = lambda v: ("%.3f" % v) if isinstance(v, float) else "-"
        print(f"{r['dataset']+'/'+r['split']:27s} {r['feature_set']:10s} "
              f"{r['dims']:4d} {r['zero_row_frac']:6.3f} "
              f"{f2(r.get('answer_zero_row_frac')):>7s} "
              f"{f2(dr.get('r2')):>6s} {str(er.get('pc90','-')):>4s} "
              f"{f2(r.get('answer_auc_rwpe')):>7s} "
              f"{f2(r.get('answer_auc_degree_basis')):>7s} "
              f"{f2(r.get('answer_auc_is_sink')):>7s}")


if __name__ == "__main__":
    main()
