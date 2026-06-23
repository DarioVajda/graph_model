"""
Synthetic graph-batch generator for the flex-attention benchmarks.

Why synthetic (and what is / isn't real)
-----------------------------------------
Attention latency and memory depend on tensor *shapes* and on the *sparsity
structure* of the attention mask — not on the numerical values of the spectral
features. So this generator:

  * builds a **real graph topology** (random-geometric by default, which has
    recoverable spatial locality so the RCM-vs-random ordering axis actually
    bites), and computes the **real K-hop mask and SPD** from it — these drive
    the sparsity / density story we are measuring;
  * **synthesizes the magnetic-Laplacian eigenvectors** (``magnetic_V`` /
    ``magnetic_lambdas``) as random tensors of the correct shape. A true
    magnetic eigendecomposition at N=2048 would dominate setup time for zero
    benchmarking value, and the ``eager`` and ``flex`` paths are always fed
    the *identical* ``node_bias`` tensor, so output parity is unaffected.

The packed batch is produced by the real :class:`GraphCollatorV2`, so token
packing, position resets, the prompt-node-last convention, and feature padding
are bit-for-bit what the training pipeline would see.

Two entry points:
  * :func:`make_model_batch`  — a HF-ready batch dict for the full-model bench.
  * :func:`make_attention_inputs` — q/k/v + structural tensors + a precomputed
    per-layer ``node_bias`` for the attention-only (isolation) bench.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Make `src...` importable when run as a script or in a fresh subprocess.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.text_graph_collator_v2 import GraphCollatorV2, _single_k_hop_mask  # noqa: E402
from src.models.bias import GraphAttentionBias  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GraphSpec:
    """One point in the benchmark sweep (a *target*; sizes are jittered around it)."""

    n_nodes: int                       # target number of *prefix* nodes
    tokens_per_node: int               # target tokens per prefix node
    prompt_tokens: int = 128           # target tokens on the prompt node
    k_hop: int = 0                     # 0 disables the hard K-hop gate
    k_hop_directed: bool = False
    ordering: str = "rcm"              # 'rcm' | 'random' | 'identity'
    directed: bool = False             # directed topology (for K-hop direction)
    avg_degree: float = 6.0            # target mean degree of the generated graph
    magnetic_m: int = 32               # truncated eigenvector count M
    laplacian_dim: int = 0             # 0 = omit (we benchmark spd + magnetic)
    jitter: float = 0.2                # ± fractional jitter on the size targets
    vocab_size: int = 128256           # Llama-3.2 vocab
    seed: int = 0

    def estimate_tokens(self, batch_size: int) -> int:
        """Upper-ish estimate of total packed tokens, for the OOM pre-filter."""
        per_graph = self.n_nodes * self.tokens_per_node + self.prompt_tokens
        return int(round(per_graph * batch_size * (1.0 + self.jitter)))


# ── Graph topology ───────────────────────────────────────────────────────────

def _generate_graph(n: int, avg_degree: float, directed: bool, rng: np.random.Generator) -> nx.Graph:
    """A random-geometric graph (spatial locality) with ~``avg_degree`` mean degree.

    Random-geometric is chosen deliberately: nodes near each other in the unit
    square are connected, so a bandwidth-reducing relabel (RCM) can recover that
    locality as contiguity in node-index order — which is exactly what lets flex's
    block mask skip blocks. An Erdos-Renyi graph would have no locality to recover.
    """
    if n <= 1:
        G = nx.empty_graph(max(n, 1), create_using=nx.DiGraph if directed else nx.Graph)
        return G
    # radius for target average degree in a unit square: deg ≈ n * pi * r^2
    radius = float(np.sqrt(avg_degree / (np.pi * max(n, 1))))
    seed = int(rng.integers(0, 2**31 - 1))
    G = nx.random_geometric_graph(n, radius, seed=seed)
    if directed:
        # Orient each undirected edge by node index → a DAG-like directed graph.
        D = nx.DiGraph()
        D.add_nodes_from(G.nodes())
        for u, v in G.edges():
            D.add_edge(min(u, v), max(u, v))
        return D
    return G


def _ordering_permutation(G: nx.Graph, mode: str, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return (perm, wall_time_seconds).

    ``perm[i]`` is the original node id that should be placed at packed position
    ``i``. We *measure* the reorder wall-time so the 'is RCM cost negligible?'
    claim is backed by numbers rather than asserted.
    """
    n = G.number_of_nodes()
    t0 = time.perf_counter()
    if mode == "identity":
        perm = np.arange(n)
    elif mode == "random":
        perm = rng.permutation(n)
    elif mode == "rcm":
        # RCM on the symmetrised adjacency (bandwidth-reducing order).
        UG = G.to_undirected() if G.is_directed() else G
        A = nx.to_scipy_sparse_array(UG, nodelist=range(n), format="csr", dtype=np.int8)
        A = csr_matrix(A)
        perm = np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True))
    else:
        raise ValueError(f"unknown ordering mode {mode!r}")
    dt = time.perf_counter() - t0
    return perm.astype(np.int64), dt


# ── Per-graph feature synthesis ──────────────────────────────────────────────

def _spd_matrix(G: nx.Graph, n: int, cutoff: Optional[int] = None) -> torch.Tensor:
    """All-pairs shortest-path distances (BFS), unreachable → n (a large value)."""
    dist = torch.full((n, n), n, dtype=torch.int32)
    dist.fill_diagonal_(0)
    for src, targets in nx.all_pairs_shortest_path_length(G, cutoff=cutoff):
        for tgt, d in targets.items():
            dist[src, tgt] = d
    return dist


def _build_items(spec: GraphSpec, batch_size: int) -> tuple[list[dict], dict]:
    """Build a list of synthetic ``TextGraph`` dicts plus per-batch metadata."""
    rng = np.random.default_rng(spec.seed)
    items: list[dict] = []
    reorder_times: list[float] = []

    def _jit(target: int) -> int:
        lo, hi = (1.0 - spec.jitter) * target, (1.0 + spec.jitter) * target
        return max(1, int(round(rng.uniform(lo, hi))))

    for _ in range(batch_size):
        n = _jit(spec.n_nodes)
        G = _generate_graph(n, spec.avg_degree, spec.directed, rng)
        n = G.number_of_nodes()

        perm, dt = _ordering_permutation(G, spec.ordering, rng)
        reorder_times.append(dt)
        # Relabel so that node-index order == the chosen packing order.
        new_label = {int(old): i for i, old in enumerate(perm)}
        G = nx.relabel_nodes(G, new_label, copy=True)

        prompt_node = int(rng.integers(0, n))
        # Per-node token counts (prompt node ~prompt_tokens, others ~tokens_per_node).
        lengths = [_jit(spec.tokens_per_node) for _ in range(n)]
        lengths[prompt_node] = _jit(spec.prompt_tokens)
        input_ids = [
            rng.integers(0, spec.vocab_size, size=L, dtype=np.int64).tolist()
            for L in lengths
        ]

        item: dict = {
            "input_ids": input_ids,
            "edges": list(G.edges()),
            "num_nodes": n,
            "prompt_node": prompt_node,
            "shortest_path_dists": _spd_matrix(G, n),
        }
        # Synthetic magnetic eigenvectors (shape-faithful; values irrelevant to timing).
        m_eff = min(spec.magnetic_m, n) if spec.magnetic_m > 0 else n
        item["magnetic_V"] = torch.randn(n, m_eff, 2)
        item["magnetic_lambdas"] = torch.sort(torch.rand(m_eff)).values
        if spec.laplacian_dim > 0:
            item["laplacian_coordinates"] = torch.randn(n, spec.laplacian_dim)
        items.append(item)

    meta = {
        "reorder_time_mean_s": float(np.mean(reorder_times)),
        "reorder_time_max_s": float(np.max(reorder_times)),
        "num_nodes_actual": [it["num_nodes"] for it in items],
    }
    return items, meta


def _collate(spec: GraphSpec, items: list[dict]) -> dict:
    collator = GraphCollatorV2(
        pad_token_id=0,
        k_hop=spec.k_hop,
        k_hop_directed=spec.k_hop_directed,
        magnetic_m=spec.magnetic_m,
    )
    return collator(items)


# ── Public entry points ──────────────────────────────────────────────────────

def make_model_batch(
    spec: GraphSpec,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict, dict]:
    """A HuggingFace-ready batch dict for the full-model benchmark (+ metadata).

    Float feature tensors are cast to ``dtype``; integer/bool tensors are left
    as-is. ``labels`` are added (the prompt span) so the model computes a loss
    and we get a real backward pass.
    """
    items, meta = _build_items(spec, batch_size)
    # Attach labels = the prompt node's own tokens (teacher-forced LM loss).
    for it in items:
        it["labels"] = torch.tensor(it["input_ids"][it["prompt_node"]], dtype=torch.long)
    batch = _collate(spec, items)

    out: dict = {}
    for k, v in batch.items():
        if v is None:
            continue
        if torch.is_floating_point(v):
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device=device)
    meta["total_tokens"] = int(batch["input_ids"].shape[0] * batch["input_ids"].shape[1])
    meta["seq_len"] = int(batch["input_ids"].shape[1])
    meta["max_num_nodes"] = int(batch["num_nodes"].max().item())
    return out, meta


def make_attention_inputs(
    spec: GraphSpec,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    bias_config: Optional[object] = None,
) -> tuple[dict, dict]:
    """Inputs for the attention-only (isolation) benchmark.

    Returns a dict with random q/k/v ``(B, H, L, d)`` / ``(B, Hkv, L, d)``, the
    structural tensors (``node_ids``, ``prompt_node``, ``pad_mask``,
    ``k_hop_mask``, ``num_nodes``), and a precomputed per-layer ``node_bias``
    ``(B, H, N, N)`` built by the real :class:`GraphAttentionBias` from the
    synthetic spd + magnetic features. The same ``node_bias`` is consumed by
    both the dense (``eager``) and ``flex`` paths, guaranteeing parity.
    """
    items, meta = _build_items(spec, batch_size)
    batch = _collate(spec, items)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    B = batch["input_ids"].shape[0]
    L = batch["input_ids"].shape[1]
    N = int(batch["num_nodes"].max().item())

    q = torch.randn(B, num_heads, L, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, num_kv_heads, L, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, num_kv_heads, L, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Build the node-level soft bias with the real module (spd + magnetic by default).
    if bias_config is None:
        bias_config = _default_bias_config(spec)
    # Compute node_bias in fp32 then cast: it's a precomputed *input* to the
    # timed attention region, so its compute dtype doesn't affect isolation
    # timings, and fp32 avoids the bias module's internal mixed-dtype arithmetic.
    bias_mod = GraphAttentionBias(
        num_heads=num_heads, head_dim=head_dim, layer_idx=0,
        bias_config=bias_config, k_hop=0,
    ).to(device=device, dtype=torch.float32)

    magnetic = None
    if batch.get("magnetic_V") is not None:
        magnetic = (
            batch["magnetic_V"].float(),
            batch["magnetic_lambdas"].float(),
        )
    node_bias = bias_mod(
        dtype=torch.float32, device=device,
        num_nodes=batch["num_nodes"],
        spd=batch.get("shortest_path_dists"),
        laplacian=batch.get("laplacian_coordinates"),
        magnetic=magnetic,
    )  # (B, H, N, N) or None
    if node_bias is not None:
        node_bias = node_bias.to(dtype)

    pad_mask = batch["attention_mask"]
    out = {
        "query": q, "key": k, "value": v,
        "node_ids": batch["node_ids"],
        "prompt_node": batch["prompt_node"],
        "pad_mask": pad_mask,
        "k_hop_mask": batch.get("k_hop_mask"),
        "num_nodes": batch["num_nodes"],
        "node_bias": node_bias,
        "q_len": L, "kv_len": L, "N": N,
    }
    meta["total_tokens"] = B * L
    meta["seq_len"] = L
    meta["max_num_nodes"] = N
    return out, meta


def _default_bias_config(spec: GraphSpec):
    """A minimal config object enabling spd + magnetic (the biases we must keep)."""
    class _Cfg:
        spd = True
        max_spd = 32
        laplacian = bool(spec.laplacian_dim > 0)
        rwse = False
        rrwp = False
        magnetic = True
        magnetic_dim = 32
        magnetic_q = 0.25
    return _Cfg()


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = GraphSpec(n_nodes=32, tokens_per_node=8, k_hop=2, ordering="rcm")
    print("estimate_tokens:", spec.estimate_tokens(batch_size=2))

    batch, meta = make_model_batch(spec, batch_size=2, device=dev, dtype=torch.bfloat16)
    print("\n[make_model_batch] keys:", sorted(batch.keys()))
    print("  input_ids:", tuple(batch["input_ids"].shape), "| seq_len:", meta["seq_len"],
          "| max_nodes:", meta["max_num_nodes"], "| reorder_ms:", round(meta["reorder_time_mean_s"]*1e3, 3))

    ai, meta2 = make_attention_inputs(
        spec, batch_size=2, num_heads=32, num_kv_heads=8, head_dim=64, device=dev,
    )
    nb = ai["node_bias"]
    print("\n[make_attention_inputs] q:", tuple(ai["query"].shape),
          "| node_bias:", None if nb is None else tuple(nb.shape),
          "| k_hop_mask:", None if ai["k_hop_mask"] is None else tuple(ai["k_hop_mask"].shape))
    print("  OK")
