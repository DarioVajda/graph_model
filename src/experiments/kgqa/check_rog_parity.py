"""
E1.2 — eval-parity check: our gold answer lists vs RoG's, by question id.

Our evaluator scores generations against ``full_gold_texts(record)`` (the
gold list stored as ``graph['gold_answers']``). GNN-RAG's numbers score against
the ``answer`` lists of ``rmanluo/RoG-{webqsp,cwq}``. Benchmark comparability
requires the two lists to be THE SAME per question — this script verifies that
by question id and reports every divergence.

The WebQSP run of this check (2026-07-05, pre-script) matched 1628/1628 test
questions exactly; this scripted version reproduces it and extends it to CWQ.

Comparison is set-based (order carries no meaning for either side) over the
raw strings both evaluators match against. Also pins the eval denominator:
every RoG question present / absent in the SR records, and vice versa.

    python -m src.experiments.kgqa.check_rog_parity --dataset cwq --split test
"""

import argparse
from collections import Counter

from .process_dataset import full_gold_texts
from .sr_records import load_sr_records

ROG_DATASET = {"webqsp": "rmanluo/RoG-webqsp", "cwq": "rmanluo/RoG-cwq"}
# RoG names HF splits train/validation/test; our SR splits are train/dev/test.
ROG_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


def rog_answers(dataset, split):
    """{question id -> RoG's raw `answer` list} for one split."""
    from datasets import load_dataset
    ds = load_dataset(ROG_DATASET[dataset], split=ROG_SPLIT[split])
    return {r["id"]: list(r["answer"]) for r in ds}


def check(dataset, split, show=10):
    ours = {r["id"]: full_gold_texts(r) for r in load_sr_records(dataset, split)
            if r.get("answers")}
    theirs = rog_answers(dataset, split)

    only_ours = sorted(set(ours) - set(theirs))
    only_theirs = sorted(set(theirs) - set(ours))
    common = set(ours) & set(theirs)

    exact, diffs = 0, []
    for qid in common:
        a, b = Counter(ours[qid]), Counter(theirs[qid])
        if a == b:
            exact += 1
        else:
            diffs.append((qid, sorted((a - b).elements()), sorted((b - a).elements())))

    print(f"[parity] {dataset}/{split}: ours={len(ours)} answered SR records, "
          f"RoG={len(theirs)} rows, common={len(common)}")
    print(f"[parity] gold lists identical (as multisets): {exact}/{len(common)}")
    if only_ours:
        print(f"[parity] {len(only_ours)} ids only in SR (e.g. {only_ours[:3]})")
    if only_theirs:
        print(f"[parity] {len(only_theirs)} ids only in RoG (e.g. {only_theirs[:3]})")
    for qid, extra_ours, extra_rog in diffs[:show]:
        print(f"  {qid}: ours-only={extra_ours}  rog-only={extra_rog}")
    if len(diffs) > show:
        print(f"  ... and {len(diffs) - show} more differing questions")
    return exact, len(common), diffs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset", choices=tuple(ROG_DATASET), required=True)
    p.add_argument("--split", choices=tuple(ROG_SPLIT), default="test")
    args = p.parse_args(argv)
    check(args.dataset, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
