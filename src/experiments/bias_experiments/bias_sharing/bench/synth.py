"""
Synthetic WebQSP-shaped batches at node counts WebQSP itself cannot reach.

`002_webqsp_g_sweep` caps graphs at 512 nodes, and the magnetic bias is
``(B, H, N, N)`` in **nodes** — so every sweep in this package measured the
sharing knob in a regime where the thing being shared is small. This module
builds batches with the same *token* profile as WebQSP at N ∈ {512 … 4096}, so
`speed.py` can price the knob where PLAN §D5 says it actually hurts.

What is faithful, and what is not
---------------------------------
**Faithful, because it drives cost:**

* per-node token counts, sampled i.i.d. from WebQSP's empirical histogram
  (`token_stats.py`) rather than from a jittered constant — the distribution is
  right-skewed (mean 3.0, sd 2.3, tail past 100) and its *variance* sets how many
  tokens a given node count packs into;
* the prompt node's token count, from its own empirical histogram;
* the eigenvector truncation ``M = min(N, magnetic_m)`` — the contracted
  dimension of the bias einsums;
* packing, position resets, prompt-node-last, bucket padding — the real
  :class:`GraphCollatorV2`, configured exactly as `kgqa/train.py` configures it
  for flex.

**Not faithful, because it does not:**

* **Topology is a uniform random attachment tree.** At ``k_hop=0`` (002's
  setting) the block mask is causal + bidirectional-prefix + padding only — the
  graph never enters it — and the SPD bias is a lookup whose cost is independent
  of the values looked up. Structure is therefore free to be arbitrary, which is
  what the request assumed. It would *not* be free at ``k_hop > 0``.
* **Eigenvectors are random.** ``magnetic_V`` / ``magnetic_lambdas`` are shape-
  faithful noise. A real magnetic eigendecomposition at N=4096 would dominate
  setup for no timing consequence: the einsums touch every element regardless.
* **Token ids are random.** They index an embedding table; only the count matters.
* **Every graph has exactly N nodes**, not a distribution around it. N is the
  independent variable of the study, so it is held exact rather than sampled.

:func:`verify_against_webqsp` re-measures the generated batch and reports the
drift against the fixture, so the "matches WebQSP closely" claim is checked at
run time rather than asserted here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .....utils import GraphCollatorV2

STATS_PATH = os.path.join(os.path.dirname(__file__), "webqsp_token_stats.json")

# 002_webqsp_g_sweep's data-side settings, which the model config must match.
WEBQSP_MAGNETIC_M = 128
WEBQSP_MAX_SPD = 64
LLAMA_VOCAB = 128256


@dataclass
class SynthSpec:
    """One point of the study: N nodes, WebQSP's token profile, batch size B."""

    n_nodes: int
    batch_size: int = 1
    magnetic_m: int = WEBQSP_MAGNETIC_M
    max_spd: int = WEBQSP_MAX_SPD
    k_hop: int = 0
    seed: int = 0
    vocab_size: int = LLAMA_VOCAB
    stats_path: str = STATS_PATH
    # Filled in by build_batch so results carry the achieved shape.
    meta: dict = field(default_factory=dict)


def _load_hist(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with open(path) as f:
        stats = json.load(f)

    def unpack(block):
        keys = np.array([int(k) for k in block["hist"]], dtype=np.int64)
        counts = np.array(list(block["hist"].values()), dtype=np.float64)
        return keys, counts / counts.sum()

    pre_v, pre_p = unpack(stats["prefix_node_tokens"])
    pro_v, pro_p = unpack(stats["prompt_node_tokens"])
    return pre_v, pre_p, pro_v, pro_p, stats


def _random_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Uniform random attachment: node i (>0) links to a uniform j < i.

    Gives O(log n) depth, so SPD values stay in the range a KG subgraph produces
    and the ``max_spd`` clamp behaves as it does on real data.
    """
    if n <= 1:
        return []
    parents = np.floor(rng.random(n - 1) * np.arange(1, n)).astype(np.int64)
    return [(int(parents[i - 1]), i) for i in range(1, n)]


def _spd_matrix(n: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """All-pairs BFS distances as int32 ``(n, n)``, via scipy's C implementation.

    networkx's pure-Python all-pairs walk is minutes at n=4096; this is seconds.
    """
    if n <= 1:
        return torch.zeros((n, n), dtype=torch.int32)
    rows = np.array([e[0] for e in edges], dtype=np.int64)
    cols = np.array([e[1] for e in edges], dtype=np.int64)
    data = np.ones(rows.size, dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    dist = shortest_path(adj, method="D", unweighted=True, directed=False)
    # A tree is connected, so no infinities; clip defensively anyway.
    return torch.from_numpy(np.nan_to_num(dist, posinf=n).astype(np.int32))


def build_items(spec: SynthSpec) -> list[dict]:
    """``batch_size`` TextGraph dicts, ready for :class:`GraphCollatorV2`."""
    rng = np.random.default_rng(spec.seed)
    pre_v, pre_p, pro_v, pro_p, _ = _load_hist(spec.stats_path)
    n = spec.n_nodes
    items = []

    for _ in range(spec.batch_size):
        edges = _random_tree_edges(n, rng)
        prompt_node = int(rng.integers(0, n))

        lengths = rng.choice(pre_v, size=n, p=pre_p).astype(np.int64)
        lengths[prompt_node] = int(rng.choice(pro_v, p=pro_p))
        input_ids = [
            rng.integers(0, spec.vocab_size, size=int(L), dtype=np.int64).tolist()
            for L in lengths
        ]

        m_eff = min(spec.magnetic_m, n) if spec.magnetic_m > 0 else n
        items.append({
            "input_ids": input_ids,
            "edges": edges,
            "num_nodes": n,
            "prompt_node": prompt_node,
            "shortest_path_dists": _spd_matrix(n, edges),
            "magnetic_V": torch.randn(n, m_eff, 2),
            "magnetic_lambdas": torch.sort(torch.rand(m_eff)).values,
            # Teacher-forced loss on the prompt node's own tokens, as in kgqa.
            "labels": torch.tensor(input_ids[prompt_node], dtype=torch.long),
        })
    return items


def build_batch(
    spec: SynthSpec,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    pad_to_block: bool = True,
) -> tuple[dict, dict]:
    """Collate a synthetic batch onto ``device``; returns ``(batch, meta)``.

    ``pad_to_block`` mirrors ``kgqa/train.py``: True for the flex backend (both L
    and N are bucket-padded), False otherwise. The plain-LLM baseline is fed the
    **same** collated tensors, so its sequence length is identical by construction
    rather than by a second, independently padded build.
    """
    items = build_items(spec)
    collator = GraphCollatorV2(
        pad_token_id=0, k_hop=spec.k_hop, magnetic_m=spec.magnetic_m,
        pad_to_block=pad_to_block, max_spd=spec.max_spd,
    )
    batch = collator(items)

    out = {}
    for key, value in batch.items():
        if value is None:
            continue
        out[key] = (value.to(device=device, dtype=dtype)
                    if torch.is_floating_point(value) else value.to(device))

    meta = {
        "n_nodes": spec.n_nodes,
        "batch_size": spec.batch_size,
        "seq_len": int(batch["input_ids"].shape[1]),
        "node_slots": int(batch["magnetic_V"].shape[1]) if "magnetic_V" in batch else None,
        "magnetic_m": int(batch["magnetic_V"].shape[2]) if "magnetic_V" in batch else None,
        "real_tokens": int(batch["attention_mask"].sum().item()),
        "tokens_per_node": float(batch["attention_mask"].sum().item()
                                 / (spec.n_nodes * spec.batch_size)),
    }
    spec.meta = meta
    return out, meta


# ── fidelity check ────────────────────────────────────────────────────────────

def verify_against_webqsp(spec: SynthSpec, n_graphs: int = 8) -> dict:
    """Re-measure generated per-node token counts against the WebQSP fixture.

    Returns both sides plus the drift, so `speed.py` can print it and the README
    can quote a checked number instead of a claim.
    """
    probe = SynthSpec(n_nodes=spec.n_nodes, batch_size=n_graphs,
                      magnetic_m=spec.magnetic_m, seed=spec.seed + 9999,
                      stats_path=spec.stats_path)
    lengths = []
    for item in build_items(probe):
        per_node = [len(x) for x in item["input_ids"]]
        lengths.extend(v for i, v in enumerate(per_node) if i != item["prompt_node"])

    got = np.array(lengths)
    _, _, pro_v, pro_p, stats = _load_hist(spec.stats_path)
    # One prompt node per graph means only `n_graphs` draws, far too few to
    # summarize a distribution whose mean (23.9) sits three times its median (8).
    # Sample the prompt histogram directly so the reported comparison reflects the
    # sampler rather than the probe size.
    prompts = np.random.default_rng(spec.seed).choice(pro_v, size=20000, p=pro_p)
    ref = stats["prefix_node_tokens"]["summary"]
    measured = {
        "n": int(got.size), "mean": float(got.mean()), "std": float(got.std()),
        "median": float(np.median(got)), "p90": float(np.percentile(got, 90)),
        "p99": float(np.percentile(got, 99)), "max": int(got.max()),
    }
    return {
        "webqsp": {k: ref[k] for k in measured},
        "synthetic": measured,
        "drift": {k: measured[k] - ref[k] for k in ("mean", "std", "median", "p90")},
        "prompt_mean": {"webqsp": stats["prompt_node_tokens"]["summary"]["mean"],
                        "synthetic": float(np.mean(prompts))},
    }


if __name__ == "__main__":
    for n in (512, 1024, 2048, 4096):
        spec = SynthSpec(n_nodes=n, batch_size=1, seed=0)
        batch, meta = build_batch(spec, torch.device("cpu"))
        print(f"N={n:5d}  L={meta['seq_len']:6d}  real_tokens={meta['real_tokens']:6d}  "
              f"tok/node={meta['tokens_per_node']:.3f}  "
              f"node_slots={meta['node_slots']}  M={meta['magnetic_m']}")
    v = verify_against_webqsp(SynthSpec(n_nodes=1024), n_graphs=4)
    print("\nfidelity (prefix-node tokens):")
    print("  webqsp   ", {k: round(x, 3) for k, x in v["webqsp"].items()})
    print("  synthetic", {k: round(x, 3) for k, x in v["synthetic"].items()})
