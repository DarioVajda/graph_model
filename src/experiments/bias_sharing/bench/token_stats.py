"""
Extract WebQSP's tokens-per-node distribution into a small JSON fixture.

The §6 benchmark needs synthetic graphs an order of magnitude larger than WebQSP
(1024–4096 nodes vs its 512 cap) whose *token* profile is nonetheless WebQSP's,
because packed sequence length — not node count — is what the attention stack
actually costs. Matching the mean alone would be wrong: WebQSP's per-node lengths
are strongly right-skewed (mean 3.0, median 3, but a tail out past 100 tokens),
and a ±20%-jittered constant would produce a sequence of the right length with
the wrong length *variance* between nodes.

So the benchmark samples node lengths i.i.d. from the empirical histogram written
here, rather than from a parametric stand-in.

Reads the same cache `002_webqsp_g_sweep` trained on, via the Arrow feature table
only (`input_ids` per node) — `graphs.pkl` holds topology we do not need and is
3 GB. Cheap enough for a login node; no model is loaded.

    python3 -m src.experiments.bias_sharing.bench.token_stats

Splits: `dev` + `test` (1874 graphs, ~220k prefix nodes). `train` is excluded on
purpose — it stores 8 augmentation versions of each graph, which would weight the
histogram by version count rather than by graph.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
from datasets import load_from_disk

# The cache `002_webqsp_g_sweep` reads: 029's `isolated` arm, levi construction.
WEBQSP_CACHE = (
    "src/experiments/kgqa/processed_datasets/"
    "sr-webqsp_meta-llama-Llama-3.2-1B_vlast_1_cap512_nmax50_ver8_spd64_"
    "magq0.25m128_len1024_rcm1_seed42_dfv3_qnisolated"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "webqsp_token_stats.json")
SPLITS = ("dev", "test")


def _summary(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def collect(cache: str, splits=SPLITS) -> dict:
    prefix_lengths: Counter = Counter()   # tokens per non-prompt node
    prompt_lengths: Counter = Counter()   # tokens on the prompt node
    node_counts: list[int] = []
    packed: list[int] = []

    for split in splits:
        ds = load_from_disk(os.path.join(cache, f"{split}.gtds", "features"))
        for row in ds:
            per_node = [len(x) for x in row["input_ids"]]
            prompt = row["prompt_node"]
            node_counts.append(len(per_node))
            packed.append(sum(per_node))
            for i, length in enumerate(per_node):
                (prompt_lengths if i == prompt else prefix_lengths)[length] += 1

    prefix = np.repeat(*zip(*sorted(prefix_lengths.items())))
    prompt = np.repeat(*zip(*sorted(prompt_lengths.items())))
    return {
        "source": {"cache": cache, "splits": list(splits), "graphs": len(node_counts)},
        # Histograms are the fixture the sampler reads; the summaries exist so a
        # synthetic batch can be checked against the real distribution it claims
        # to match without re-opening the 3 GB cache.
        "prefix_node_tokens": {"hist": dict(sorted(prefix_lengths.items())),
                               "summary": _summary(prefix)},
        "prompt_node_tokens": {"hist": dict(sorted(prompt_lengths.items())),
                               "summary": _summary(prompt)},
        "nodes_per_graph": _summary(np.array(node_counts)),
        "packed_tokens_per_graph": _summary(np.array(packed)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=WEBQSP_CACHE)
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args(argv)

    stats = collect(args.cache)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1, sort_keys=False)

    pn = stats["prefix_node_tokens"]["summary"]
    print(f"wrote {args.out}")
    print(f"  graphs                 {stats['source']['graphs']}")
    print(f"  prefix-node tokens     mean {pn['mean']:.3f}  sd {pn['std']:.3f}  "
          f"median {pn['median']:.0f}  p90 {pn['p90']:.0f}  max {pn['max']}  (n={pn['n']})")
    print(f"  prompt-node tokens     mean {stats['prompt_node_tokens']['summary']['mean']:.2f}")
    print(f"  nodes/graph            mean {stats['nodes_per_graph']['mean']:.1f}  "
          f"max {stats['nodes_per_graph']['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
