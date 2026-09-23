"""Answer-coverage ceiling of the RoG/GNN-RAG 2-hop candidate pool.

The SR ceilings in the README bound *our* input condition. The baselines that
run their own retrieval are bounded by something else, and none of their papers
publishes it — GNN-RAG reports no subgraph answer coverage in either the ACL
version or the arXiv one. This measures it from the data instead, over the
``graph`` field of ``rmanluo/RoG-{webqsp,cwq}``: the 2-hop neighbourhoods RoG
and GNN-RAG retrieve from.

Mirrors ``analyse_dataset._ceilings`` exactly so the numbers are comparable to
the SR tables:

  hits1_ceiling — fraction of questions with >= 1 gold present (bounds Hits@1)
  recall_macro  — mean over questions of present/total
  f1_macro      — mean of 2R/(1+R) (perfect precision => P=1); the reported metric

Two things this is not. It measures the POOL those methods retrieve from, not
their retrievers' output, so it is an upper bound on their ceiling rather than
their ceiling — their GNN selects a subset of it. And the matching key differs
from the SR analysis: RoG ships name-resolved triples, so gold answer strings
are intersected with node strings, where the SR path intersects mids. Both ask
"is the gold entity a node of the retrieved graph"; golds that are unnamed mids
never match on either side, deflating both equally.

    python -m src.experiments.kgqa.analysis.rog_pool_ceiling --datasets webqsp cwq
"""

import argparse

DATASETS = {"webqsp": "rmanluo/RoG-webqsp", "cwq": "rmanluo/RoG-cwq"}
# RoG names HF splits train/validation/test; our splits are train/dev/test.
ROG_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


def ceilings(per_q):
    """Coverage ceilings from [(n_present, n_gold), ...]; see module docstring."""
    n = len(per_q)
    recalls = [p / t for p, t in per_q]
    return {
        "n_questions": n,
        "hits1_ceiling": sum(p >= 1 for p, _ in per_q) / n,
        "recall_macro": sum(recalls) / n,
        "f1_macro": sum(2 * r / (1 + r) for r in recalls) / n,
    }


def pool_presence(dataset, split):
    """{question id -> (n_gold_present, n_gold)} over the RoG 2-hop pool."""
    from datasets import load_dataset

    ds = load_dataset(DATASETS[dataset], split=ROG_SPLIT[split])
    out = {}
    for rec in ds:
        golds = {a for a in rec["answer"] if a}
        if not golds:
            continue
        nodes = set()
        for tri in rec["graph"]:
            nodes.add(tri[0])
            nodes.add(tri[2])
        out[rec["id"]] = (len(golds & nodes), len(golds))
    return out


def analyse(dataset, split):
    per_id = pool_presence(dataset, split)
    out = ceilings(list(per_id.values()))
    out["n_empty_graph"] = sum(1 for p, _ in per_id.values() if p == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["webqsp", "cwq"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    for dataset in args.datasets:
        r = analyse(dataset, args.split)
        print(f"\n[rog-pool] {dataset}/{args.split}  n={r['n_questions']}")
        print(f"  Hits@1 ceiling : {r['hits1_ceiling'] * 100:.1f}")
        print(f"  Recall (macro) : {r['recall_macro'] * 100:.1f}")
        print(f"  F1 (macro)     : {r['f1_macro'] * 100:.1f}")


if __name__ == "__main__":
    main()
