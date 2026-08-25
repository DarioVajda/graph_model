"""Landmark anchor coordinates — the feature `LandmarkBias` consumes.

See ``src/models/biases/LANDMARK_BIAS.md``. For a graph with anchors
``A = (a_1..a_k)`` every node ``i`` gets three coordinate rows:

    channel 0  D_out[i,j] = SPD(i, a_j)        directed, forward
    channel 1  D_in [i,j] = SPD(a_j, i)        directed, backward
    channel 2  D_und[i,j] = SPD_und(i, a_j)    undirected skeleton

clipped to ``d_max`` and symbolised as ``{0..d_max}``, ``UNREACH = d_max+1``,
``PAD = d_max+2``. Stored flat as ``(N*3*k,)`` int16 per graph.

Two decisions are load-bearing and both are measured, not assumed
(`src/experiments/bias_experiments/landmark/README.md`):

* **The rule is `degree`.** It beat betweenness on 3 of 4 metrics at every k and
  costs O(1) instead of O(NM); PageRank and FPS are eliminated by mechanism
  (PageRank mass lands on sinks, whose D_in column is dead; FPS seeks the
  periphery, which in a sparse directed graph reaches nothing — 45% of nodes got
  a fully dead row).
* **Channel 2 is not optional on a KG.** 94.4% of WebQSP node pairs have no
  directed path, and gold-answer pairs are *less* directed-visible than average
  (0.034 vs 0.052), so without the undirected block the bias is identically zero
  at init on the pairs the task turns on.

**Anchors are emitted round-robin across weakly connected components** (components
by size desc, within a component by degree desc). That ordering is what lets a
*prefix* of the stored anchors act as a smaller-k selection while keeping every
component covered — so `landmark_k_collate` sweeps k without touching the dataset
cache key, the same trick `magnetic_m_collate` plays for M.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


WL_ROUNDS = 3


def _struct_key(A_csr, n):
    """Permutation-invariant tie-break key per node, from directed Weisfeiler-Lehman
    colour refinement.

    Why refinement and not just degree: the anchor SET must be a function of the
    graph up to isomorphism or Property 1 fails, and (in-deg, out-deg) alone leaves
    large ties on a KG — every degree-1 leaf looks alike. WL separates any two
    nodes a 1-WL test can, which covers essentially all real ties here.

    What it cannot do is separate nodes in the same automorphism orbit; there the
    key is genuinely equal and the index breaks the tie. That residual is real and
    documented rather than hidden: choosing k of n interchangeable nodes requires
    breaking a symmetry, and no deterministic rule can do it equivariantly. In
    practice the datasets are stored in RCM order, itself a deterministic function
    of the topology, so the index is not an arbitrary serialization artifact.
    """
    ind = np.asarray(A_csr.sum(0)).ravel()
    outd = np.asarray(A_csr.sum(1)).ravel()
    deg = ind + outd
    succ = A_csr.tocsr()
    pred = A_csr.T.tocsr()

    colour = [hash((float(ind[i]), float(outd[i]))) for i in range(n)]
    for _ in range(WL_ROUNDS):
        nxt = []
        for i in range(n):
            s = sorted(colour[j] for j in succ.indices[succ.indptr[i]:succ.indptr[i + 1]])
            p = sorted(colour[j] for j in pred.indices[pred.indptr[i]:pred.indptr[i + 1]])
            nxt.append(hash((colour[i], tuple(s), tuple(p))))
        colour = nxt

    # Rank the colours so the key is a small int independent of hash randomization
    # across processes (PYTHONHASHSEED); only the induced PARTITION is used.
    ranks = {c: r for r, c in enumerate(sorted(set(colour)))}
    keys = [(float(ind[i]), float(outd[i]), ranks[colour[i]]) for i in range(n)]
    return keys, deg


def select_anchors(A_csr, n, k):
    """Degree anchors, component-stratified, emitted round-robin across components.

    Returns an int array of length ``min(k, n)``.
    """
    ncomp, comps = connected_components(A_csr, directed=True, connection="weak")
    keys, deg = _struct_key(A_csr, n)
    sizes = np.bincount(comps, minlength=ncomp)
    sig = {c: tuple(sorted(deg[comps == c].tolist())) for c in range(ncomp)}
    order = sorted(range(ncomp), key=lambda c: (-sizes[c], sig[c]))

    # Each component's own priority list: degree desc, structural tie-break.
    ranked = {c: sorted(np.flatnonzero(comps == c).tolist(),
                        key=lambda i: (-deg[i], keys[i], i)) for c in order}
    out, exhausted = [], False
    while len(out) < min(k, n) and not exhausted:
        exhausted = True
        for c in order:
            if ranked[c]:
                out.append(ranked[c].pop(0))
                exhausted = False
                if len(out) >= min(k, n):
                    break
    return np.array(out, dtype=np.int64)


def landmark_coords(edges, n: int, k: int = 32, d_max: int = 8) -> np.ndarray:
    """``(n, 3, k)`` int16 anchor coordinates for one graph.

    Anchor slots beyond ``min(k, n)`` are PAD for every node, which is what lets
    ``LandmarkBias`` recover ``k_val`` from any real node's row.
    """
    unreach, pad = d_max + 1, d_max + 2
    if n == 0:
        return np.zeros((0, 3, k), dtype=np.int16)
    if edges:
        r, c = zip(*edges)
    else:
        r, c = (), ()
    A = csr_matrix((np.ones(len(r)), (list(r), list(c))), shape=(n, n))

    anchors = select_anchors(A, n, k)
    out = np.full((n, 3, k), pad, dtype=np.int16)
    if len(anchors) == 0:
        return out

    # Only k source/target BFS runs are needed, not the full APSP: `indices=`
    # restricts Dijkstra to the anchor rows. O(k(N+E)), not O(N^2).
    d_from_anchor = shortest_path(A, method="D", unweighted=True, directed=True,
                                  indices=anchors)                    # (k', n)
    d_to_anchor = shortest_path(A.T.tocsr(), method="D", unweighted=True,
                                directed=True, indices=anchors)       # (k', n)
    d_und = shortest_path(A, method="D", unweighted=True, directed=False,
                          indices=anchors)                            # (k', n)

    def sym(M):
        M = M.T                                                       # (n, k')
        s = np.where(np.isfinite(M), np.minimum(M, d_max), unreach)
        return s.astype(np.int16)

    ka = len(anchors)
    out[:, 0, :ka] = sym(d_to_anchor)     # i -> a_j
    out[:, 1, :ka] = sym(d_from_anchor)   # a_j -> i
    out[:, 2, :ka] = sym(d_und)
    return out
