"""How much of the training mixture `max_length` truncates, per arm.

A flat arm is a single node, so `max_length` caps its WHOLE prompt and cuts from
the right — taking the trailing answer with it. `render` then supervises the
prompt node's last token, which on a truncated row is a mid-molecule token, so
the row trains the model to answer at a position where the molecule has not
finished. The graph arm is immune: the cap is per node, and an atom text is a
handful of tokens.

Evaluation can exclude those rows and does (`tools/notation/probe.py`). Training
cannot. This measures what training actually swallowed, which decides whether the
defect is worth a rebuild or is a disclosure and nothing more.

The census needs no tokenizer: a built source stores its already-truncated
`input_ids`, so a node is truncated exactly when it sits at the cap.

    src/generalist/tools/launch/run_py.sh src/generalist/tools/checks/truncation_census.py
    src/generalist/tools/launch/run_py.sh src/generalist/tools/checks/truncation_census.py --split test

Weighting matters as much as the rate. A 3 % rate on a source holding 2 % of the
mixture is not the same defect as a 3 % rate on one holding 20 %, so the per-arm
bottom line is weighted by each task's mixture share.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def census(config, adapter_config, split, arms, cap):
    """`(task, arm) -> (truncated, rows)` over every task the mixture trains on."""
    from src.generalist.adapters.molecules import load, splits_for

    out = {}
    for entry in config.mixture_entries():
        task = entry["name"]
        name = task.split("/", 1)[1]
        if split not in splits_for(name):
            continue
        for arm in arms:
            try:
                source = load(task, split, arm, config=adapter_config, check_keys=0)
            except Exception as exc:                    # an unbuilt arm is not a failure
                print(f"  {task}/{arm}: skipped ({type(exc).__name__})", file=sys.stderr)
                continue
            hit = rows = 0
            for item in source.dataset:
                rows += 1
                if any(len(ids) >= cap for ids in item["input_ids"]):
                    hit += 1
            out[(task, arm)] = (hit, rows)
            del source
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-length", type=int, default=None,
                    help="the cap to count against (default: the config's).")
    ap.add_argument("--out", default=None, help="write the census as JSON.")
    args = ap.parse_args(argv)

    from src.generalist.config import ARMS, RunConfig

    config = RunConfig(run_name="truncation_census", mixture="molecule_generalist",
                       validators="default").validate()
    adapter_config = config.adapter_config()
    cap = args.max_length or config.max_length
    weights = {e["name"]: e["weight"] for e in config.mixture_entries()}

    counts = census(config, adapter_config, args.split, ARMS, cap)
    tasks = sorted({task for task, _ in counts})

    print(f"\nbuild {adapter_config.build_version()}  split {args.split}  cap {cap}\n")
    head = f"{'task':<26}{'share':>7}" + "".join(f"{a:>16}" for a in ARMS)
    print(head)
    print("-" * len(head))
    for task in tasks:
        line = f"{task:<26}{weights.get(task, 0.0):>6.1%} "
        for arm in ARMS:
            hit, rows = counts.get((task, arm), (0, 0))
            line += f"{hit:>7}/{rows:<8}" if rows else f"{'--':>16}"
        print(line)

    print("-" * len(head))
    # The number that decides anything: the share of an arm's training draw whose
    # supervised token is wrong, weighted by how often the mixture asks for it.
    line = f"{'weighted share of draw':<26}{'':>7}"
    for arm in ARMS:
        share = sum(weights.get(task, 0.0) * (counts[(task, arm)][0] / counts[(task, arm)][1])
                    for task in tasks
                    if counts.get((task, arm), (0, 0))[1])
        line += f"{share:>15.3%} "
    print(line + "\n")

    if args.out:
        payload = {"build_version": adapter_config.build_version(), "split": args.split,
                   "cap": cap,
                   "counts": {f"{t}|{a}": v for (t, a), v in counts.items()}}
        # The job runs on a compute node, which cannot see a caller's node-local
        # scratch: an unwritable --out must not throw away a census that has
        # already printed.
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
