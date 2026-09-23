"""Per-question agreement between the SR subgraph and the RoG 2-hop pool.

The CWQ ceilings from ``rog_pool_ceiling`` land within 0.03 points of the SR
ceilings (80.69 vs 80.66 Hits@1), which is the kind of coincidence that is
usually a shared data source or a bug. It is neither: SR-CWQ comes from SR's own
retriever over an int-coded Freebase cache (see ``sr_records``), not from RoG's
2-hop neighbourhoods. This checks that directly by asking whether the same
*questions* are covered rather than how many.

The WebQSP run is the control — there the two disagree strongly and in one
direction (106 RoG-only against 32 SR-only), which is what shows the comparison
is not structurally forced to agree.

    python -m src.experiments.kgqa.analysis.ceiling_agreement --dataset cwq
"""

import argparse

from ..sr_records import load_sr_records
from .rog_pool_ceiling import pool_presence


def sr_presence(dataset, split):
    """{question id -> (n_gold_present, n_gold)} over the raw SR subgraph."""
    out = {}
    for rec in load_sr_records(dataset, split):
        golds = {a["kb_id"] for a in rec.get("answers", []) if a.get("kb_id")}
        if not golds:
            continue
        nodes = set()
        for h, _, t in rec["subgraph"]["tuples"]:
            nodes.add(h)
            nodes.add(t)
        out[rec["id"]] = (len(golds & nodes), len(golds))
    return out


def compare(dataset, split):
    sr = sr_presence(dataset, split)
    rog = pool_presence(dataset, split)
    common = sorted(set(sr) & set(rog))

    counts = {"both": 0, "sr_only": 0, "rog_only": 0, "neither": 0}
    deltas = []
    for qid in common:
        s, r = sr[qid][0] >= 1, rog[qid][0] >= 1
        counts["both" if s and r else
               "sr_only" if s else
               "rog_only" if r else "neither"] += 1
        deltas.append(sr[qid][0] / sr[qid][1] - rog[qid][0] / rog[qid][1])

    n = len(common)
    return {
        "n_common": n,
        "n_sr": len(sr),
        "n_rog": len(rog),
        **counts,
        "sr_hits1": (counts["both"] + counts["sr_only"]) / n,
        "rog_hits1": (counts["both"] + counts["rog_only"]) / n,
        "union_hits1": (n - counts["neither"]) / n,
        "mean_recall_delta": sum(deltas) / len(deltas),
        "n_recall_differs": sum(1 for d in deltas if d != 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["webqsp", "cwq"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    r = compare(args.dataset, args.split)
    n = r["n_common"]
    print(f"\n[agree] {args.dataset}/{args.split}: "
          f"SR={r['n_sr']} RoG={r['n_rog']} common={n}")
    for key, label in (("both", "covered by both"),
                       ("sr_only", "SR only (RoG pool misses)"),
                       ("rog_only", "RoG only (SR misses)"),
                       ("neither", "neither")):
        print(f"  {label:<28}: {r[key]:5d} ({r[key] / n * 100:.1f}%)")
    print(f"  => SR hits1 {r['sr_hits1'] * 100:.2f}  "
          f"RoG hits1 {r['rog_hits1'] * 100:.2f}  "
          f"union {r['union_hits1'] * 100:.2f}")
    print(f"  mean per-q recall delta (SR - RoG): "
          f"{r['mean_recall_delta'] * 100:+.2f} pts; "
          f"questions where they differ: {r['n_recall_differs']}")


if __name__ == "__main__":
    main()
