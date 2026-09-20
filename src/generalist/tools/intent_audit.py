"""The hand audit of the composed set, and the interval it reports.

Step 7 of §9.4's order of work, and the number the design is judged on. The
`verify` rate is not it: `verify` checks that the statements survived the
re-voicing, and the statements are computed, so a set can verify at 1.000 and
still be full of questions that do not fit their answers. Only reading does that,
and reading only counts if what was read was chosen before it was read.

    --mode sheet   draw 100 composed rows, stratified by twist, task and format
    --mode score   read the labels back and report the defect rate with an
                   interval

**The draw is even over the twists, and the set is not.** A twist is a small
share of the set — `needs_clarification` is about one row in a hundred — so a
draw that mirrored the population would reach it once, and a defect that is
universal within a twist would read as one unlucky row. Every twist therefore
gets about a quarter of the budget, and each drawn row carries the share of the
population its cell stands for. `--mode score` prints both rates: the rate over
the draw answers "is any cell broken", the population-weighted rate answers
"what fraction of the training data is bad". Quote the one that matches the
question being asked.

A row is labelled `defect: true` if a careful reader would not want a model
trained on it, and `class` names why in one of the closed set below. Anything
that does not fit gets `class: other` and a note — a class list that never grows
is a class list nobody is really using.

    src/generalist/tools/run_py.sh -m src.generalist.tools.intent_audit \
        --mode sheet --composed .../v5/composed --out .../v5/audit --n 100
    src/generalist/tools/run_py.sh -m src.generalist.tools.intent_audit \
        --mode score --labels .../v5/audit/labels.jsonl --out .../v5/audit

The interval is Wilson at 95 %, which is the right one at n=100 near the
boundaries — a normal approximation on 2/100 gives a lower bound below zero.
"""

import argparse
import collections
import json
import math
import os
import random
import sys

#: What a defect can be. The first five are the shapes the v1 hand reviews found;
#: the last two are what the rendered design could still get wrong.
CLASSES = (
    "answer_wrong",        # the reply states something the statements do not
    "question_mismatch",   # the reply does not answer what was asked
    "question_leaks",      # the question contains its own answer
    "format_wrong",        # the brief's format was not met
    "unnatural",           # nobody would type this question
    "connective_false",    # a "therefore"/"whereas" nothing licenses
    "shot_leaks",          # a demonstration answers the target
    "other",
)


def _load(path: str) -> list:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _wilson(successes: int, n: int, z: float = 1.96):
    """A 95 % interval that stays inside [0, 1] at n=100 and small counts."""
    if not n:
        return (float("nan"), float("nan"))
    phat = successes / n
    denominator = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) \
        / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _sheet(args) -> int:
    rows = []
    for name in ("train", "test"):
        path = os.path.join(args.composed, f"{name}.jsonl")
        if os.path.exists(path):
            rows.extend(_load(path))
    if not rows:
        raise SystemExit(f"no train.jsonl or test.jsonl under {args.composed}")

    # Stratified on the twist as well as the task and the format, and the twist
    # is the axis that matters most. A twist is a small share of the set —
    # `needs_clarification` is about one row in a hundred — so a draw stratified
    # only on (task, format) reaches it once or twice by luck, and a defect that
    # is *universal within a twist* reads as one unlucky row rather than as a
    # whole cell being broken. That is not hypothetical: the exchange never
    # reaching the question node was 100 % of `needs_clarification` and would
    # have shown up here as a single odd row.
    #
    # The cost is that the draw is no longer a sample of the set, so the headline
    # rate is a per-cell average and `_score` reports the population-weighted
    # rate beside it.
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[(row.get("twist", "none"), row["task"],
                 row["brief"]["format"])].append(row)
    rng = random.Random(args.seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Two levels, twist outermost, because a flat round-robin over the cells in
    # sorted order would hand the whole budget to whichever twist sorts first.
    # Each round gives every twist one row, and inside a twist the round-robin
    # runs over its (task, format) cells in a shuffled order, so the budget is
    # split about evenly over the twists and spread over the tasks within each.
    by_twist = collections.defaultdict(list)
    for key in buckets:
        by_twist[key[0]].append(key)
    for keys in by_twist.values():
        rng.shuffle(keys)
    twists = sorted(by_twist)

    chosen, cursor, drawn = [], {t: 0 for t in twists}, collections.Counter()
    exhausted = set()
    while len(chosen) < args.n and len(exhausted) < len(twists):
        for twist in twists:
            if len(chosen) >= args.n or twist in exhausted:
                continue
            keys = by_twist[twist]
            for _ in range(len(keys)):
                key = keys[cursor[twist] % len(keys)]
                cursor[twist] += 1
                depth = cursor[twist] // len(keys)
                if depth - 1 < len(buckets[key]):
                    chosen.append(buckets[key][depth - 1])
                    drawn[twist] += 1
                    break
            else:
                exhausted.add(twist)

    # What each drawn row stands for in the set as a whole. The draw is even over
    # the twists and the set is not, so a row from a rare twist speaks for far
    # fewer rows than one from `none`; `_score` uses this for the second rate it
    # prints, and the two rates answering differently is itself the finding.
    population = collections.Counter(row.get("twist", "none") for row in rows)
    weight = {t: (population[t] / drawn[t]) / len(rows) if drawn[t] else 0.0
              for t in twists}

    os.makedirs(args.out, exist_ok=True)
    lines = []
    for i, row in enumerate(chosen, 1):
        lines.append(f"--- {i:03d}  {row['id']}  {row['cell']}  "
                     f"twist={row.get('twist', 'none')}  role={row['role']} ---")
        lines.append(f"BRIEF : {row['brief']['register']}; "
                     f"{row['brief']['length']}; {row['brief']['format']}")
        for shot in row.get("shots") or []:
            lines.append(f"  shot Q: {shot['question']}")
            lines.append(f"  shot A: {shot['answer']}")
        if row.get("pointer"):
            lines.append(f"POINTER: {row['pointer']}")
        lines.append(f"Q     : {row['question']}")
        for turn in row.get("turns") or []:
            if turn["text"] is not None and turn["role"] == "assistant" \
                    and turn["text"] != row["rendered_reply"]:
                lines.append(f"  (assistant asked: {turn['text']})")
        lines.append(f"A     : {row['answer']}")
        lines.append("STATEMENTS:")
        lines += [f"  - {s}" for s in row["statements"]]
        lines.append("")
    with open(os.path.join(args.out, "sheet.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    blank = os.path.join(args.out, "labels.jsonl")
    if not os.path.exists(blank):
        with open(blank, "w") as handle:
            for row in chosen:
                twist = row.get("twist", "none")
                handle.write(json.dumps(
                    {"id": row["id"], "task": row["task"], "twist": twist,
                     "format": row["brief"]["format"],
                     "weight": round(weight[twist], 6), "defect": None,
                     "class": "", "note": ""}, sort_keys=True) + "\n")
    print(f"{len(chosen)} rows over {len(buckets)} (twist, task, format) cells")
    for twist in twists:
        print(f"  {twist:20} {drawn[twist]:3} drawn of {population[twist]:5} "
              f"in the set")
    print(f"  sheet : {os.path.join(args.out, 'sheet.txt')}")
    print(f"  labels: {blank}")
    return 0


def _score(args) -> int:
    labels = [r for r in _load(args.labels) if r.get("defect") is not None]
    if not labels:
        raise SystemExit("no row is labelled yet")
    bad = sum(1 for r in labels if r["defect"])
    low, high = _wilson(bad, len(labels))
    print(f"{bad}/{len(labels)} rows defective in the draw — "
          f"{bad / len(labels):.3f} [{low:.3f}, {high:.3f}] (Wilson 95 %)")

    # The draw over-samples the rare twists on purpose, so the line above is the
    # rate over the cells and not over the set. This is the rate over the set:
    # every row counted as the share of the population its cell stands for. It
    # is the number to quote for "what fraction of the training data is bad", and
    # the line above is the one to quote for "is any cell broken".
    total_weight = sum(r.get("weight", 0.0) for r in labels)
    weighted = sum(r.get("weight", 0.0) for r in labels if r["defect"])
    if total_weight:
        print(f"{'':>{len(str(bad)) + len(str(len(labels))) + 1}} "
              f"population-weighted — {weighted / total_weight:.3f}")
    print()

    by_class = collections.Counter(r.get("class") or "unclassified"
                                   for r in labels if r["defect"])
    if by_class:
        print("by class:")
        for name, n in by_class.most_common():
            print(f"  {name:20} {n}")
    unknown = sorted(set(by_class) - set(CLASSES) - {"unclassified"})
    if unknown:
        print(f"\nclasses not in CLASSES: {unknown} — add them or fix the label")

    # Per twist, and with its own interval, because "is this cell broken" is a
    # question about the cell and an aggregate cannot answer it: a twist that is
    # 1 % of the set can be defective in every row without moving the total.
    print("\nby twist:")
    for twist in sorted({r.get("twist", "none") for r in labels}):
        rows = [r for r in labels if r.get("twist", "none") == twist]
        n_bad = sum(1 for r in rows if r["defect"])
        lo, hi = _wilson(n_bad, len(rows))
        print(f"  {twist:20} {n_bad:3}/{len(rows):3}  "
              f"[{lo:.2f}, {hi:.2f}]")

    print("\nby task:")
    for task in sorted({r["task"] for r in labels}):
        rows = [r for r in labels if r["task"] == task]
        n_bad = sum(1 for r in rows if r["defect"])
        print(f"  {task:14} {n_bad:3}/{len(rows):3}")
    print("\nby format:")
    for fmt in sorted({r["format"] for r in labels}):
        rows = [r for r in labels if r["format"] == fmt]
        n_bad = sum(1 for r in rows if r["defect"])
        print(f"  {fmt:34} {n_bad:3}/{len(rows):3}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "audit.json"), "w") as handle:
            by_twist = {}
            for twist in sorted({r.get("twist", "none") for r in labels}):
                rows = [r for r in labels if r.get("twist", "none") == twist]
                by_twist[twist] = {"n": len(rows),
                                   "defects": sum(1 for r in rows if r["defect"])}
            json.dump({"n": len(labels), "defects": bad,
                       "rate": bad / len(labels),
                       "wilson95": [low, high],
                       "weighted_rate": (weighted / total_weight
                                         if total_weight else None),
                       "by_twist": by_twist,
                       "by_class": dict(by_class)}, handle, indent=1,
                      sort_keys=True)
    print("\nDefects, for the write-up:")
    for row in labels:
        if row["defect"]:
            print(f"  {row['id']}  {row.get('twist', 'none'):20} "
                  f"{row.get('class', ''):18} {row.get('note', '')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("sheet", "score"))
    parser.add_argument("--composed")
    parser.add_argument("--labels")
    parser.add_argument("--out")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "sheet":
        if not args.composed or not args.out:
            raise SystemExit("--mode sheet needs --composed and --out")
        return _sheet(args)
    if not args.labels:
        raise SystemExit("--mode score needs --labels")
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
