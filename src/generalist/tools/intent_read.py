"""Print rendered intents, one per cell, for reading by hand.

The renderer is the only place a claim can enter an answer, so its output is read
before anything is written from it. One example per (task, twist) by default,
which is short enough to read in full.

With `--asks` and `--voiced` it overlays what the writer returned beside the
brief it was given, which is how a writer pass is read before it is trusted at
scale. The four renderer defects the first smoke build carried were all found
this way and none of them by a test.
"""

import argparse
import collections
import glob
import json
import os
import random


def _load_jsonl(directory: str, field: str) -> dict:
    out = {}
    if not directory:
        return out
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if field in row:
                        out[row["id"]] = row[field]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", required=True)
    parser.add_argument("--asks", default=None,
                        help="the ask pass's output, to show beside the brief")
    parser.add_argument("--voiced", default=None,
                        help="the voice pass's output, to show beside the render")
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--role", default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    turns = _load_jsonl(args.asks, "turn")
    replies = _load_jsonl(args.voiced, "reply")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.batches, "*.json"))):
        if os.path.basename(path) == "manifest.json":
            continue
        with open(path) as handle:
            payload = json.load(handle)
        if payload.get("role") != args.role:
            continue
        rows.extend(payload["examples"])

    random.Random(args.seed).shuffle(rows)
    seen = collections.Counter()
    for row in rows:
        intent, rendered = row["intent"], row["render"]
        cell = (intent["task"], intent["twist"])
        if seen[cell] >= args.per_cell:
            continue
        if turns and row["id"] not in turns:
            continue
        seen[cell] += 1
        print("=" * 78)
        print(f"{row['id']}  task={intent['task']}  twist={intent['twist']}")
        print(f"  situation : {intent['situation']['persona']}, "
              f"{intent['situation']['context']}")
        print(f"  style     : {intent['style']['register']} / "
              f"{intent['style']['length']} / {intent['style']['format']}")
        print("  facts     :")
        for fact in intent["facts"]:
            print(f"     - {fact['text']}  [{fact['kind']}={fact['value']}]")
        if intent.get("spare_family"):
            print(f"  not on the sheet : {intent['spare_family']}")
        print("  ask       :", json.dumps(rendered["ask"], sort_keys=True))
        print("  statements:")
        for statement in rendered["statements"]:
            print(f"     - {statement}")
        if rendered.get("skeleton"):
            print("  skeleton  :", json.dumps(rendered["skeleton"],
                                              sort_keys=True))
        print("  turns     :")
        for turn in rendered["turns"]:
            body = turn["text"] if turn["text"] is not None else "<written>"
            print(f"     {turn['role']:>9}: {body}")
        if row["id"] in turns:
            print("  WRITTEN   :", turns[row["id"]])
        if row["id"] in replies:
            print("  VOICED    :", replies[row["id"]])
    print(f"\n{len(seen)} (task, twist) cells shown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
