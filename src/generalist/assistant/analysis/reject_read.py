"""Print the accept pass's rejections, stratified by reason, for hand reading.

This exists because of §9.4's first rule: **a filter's rejection log is evidence
about the filter before it is evidence about the writer.** Five times now a
filter in this section has been discarding correct rows, and the only symptom
each time was a yield that looked plausible. A rejection reason that is a large
share of the log gets read before it is believed.

    src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.analysis.reject_read \
        --accepted .../v5/accepted --reason statement_dropped --n 12
"""

import argparse
import collections
import json
import os
import random
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True,
                        help="the accept pass's output directory")
    parser.add_argument("--reason", default=None,
                        help="one reason, or all of them if omitted")
    parser.add_argument("--n", type=int, default=8, help="rows per reason")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = []
    with open(os.path.join(args.accepted, "rejected.jsonl")) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    by_reason = collections.defaultdict(list)
    for row in rows:
        by_reason[row["reason"]].append(row)
    rng = random.Random(args.seed)

    wanted = [args.reason] if args.reason else sorted(
        by_reason, key=lambda r: -len(by_reason[r]))
    for reason in wanted:
        sample = by_reason.get(reason, [])
        if not sample:
            print(f"no rows rejected as {reason!r}")
            continue
        rng.shuffle(sample)
        print("=" * 78)
        print(f"{reason}  —  {len(by_reason[reason])} rows")
        print("=" * 78)
        for row in sample[:args.n]:
            print(f"\n--- {row['id']}  {row['cell']}")
            print(f"  detail: {row['detail']}")
            print(f"  Q     : {row['turn']}")
            print(f"  A     : {row['reply']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
