"""Subgraph sizes from the two retrievers the results table compares.

The ceiling numbers in ``rog_pool_ceiling`` say what each input condition makes
*reachable*; they say nothing about what it costs to read. This measures the
other half: how big the graph each extractor hands its reader actually is.

Both sides are the raw retriever output — SR's ``subgraph.tuples`` and the
``graph`` field of ``rmanluo/RoG-{webqsp,cwq}`` — so the comparison is like for
like. Our own node cap is applied downstream at build time and is not included
here; ``analyse_dataset.analyse_built_split`` reports the post-cap sizes the
model is trained on.

Nodes are counted as distinct tuple endpoints, which is the same set the
ceilings intersect gold answers against, so a size row and a ceiling row always
describe the same graph.

    python -m src.experiments.kgqa.analysis.subgraph_sizes --datasets webqsp cwq
"""

import argparse
import statistics

from ..sr_records import load_sr_records
from .rog_pool_ceiling import DATASETS, ROG_SPLIT


def _stats(values):
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "mean": sum(values) / n,
        "median": statistics.median(values),
        "p95": values[min(n - 1, int(0.95 * n))],
        "max": values[-1],
    }


def _summarize(per_q):
    """[(nodes, triples, relations), ...] -> {field: stats}."""
    return {
        "nodes": _stats([q[0] for q in per_q]),
        "triples": _stats([q[1] for q in per_q]),
        "relations": _stats([q[2] for q in per_q]),
    }


def _sizes(tuples):
    nodes, rels = set(), set()
    for h, r, t in tuples:
        nodes.add(h)
        nodes.add(t)
        rels.add(r)
    return len(nodes), len(tuples), len(rels)


def sr_sizes(dataset, split):
    return _summarize([
        _sizes(rec["subgraph"]["tuples"])
        for rec in load_sr_records(dataset, split)
    ])


def rog_sizes(dataset, split):
    from datasets import load_dataset

    ds = load_dataset(DATASETS[dataset], split=ROG_SPLIT[split])
    return _summarize([_sizes(rec["graph"]) for rec in ds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["webqsp", "cwq"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    for dataset in args.datasets:
        print(f"\n[sizes] {dataset}/{args.split}")
        for label, fn in (("SR (raw)", sr_sizes), ("RoG 2-hop pool", rog_sizes)):
            res = fn(dataset, args.split)
            n = res["nodes"]["n"]
            print(f"  {label:<16} n={n}")
            for field in ("nodes", "triples", "relations"):
                s = res[field]
                print(f"    {field:<10} mean {s['mean']:8.1f}  median {s['median']:7.1f}"
                      f"  p95 {s['p95']:7.0f}  max {s['max']:7.0f}")


if __name__ == "__main__":
    main()
