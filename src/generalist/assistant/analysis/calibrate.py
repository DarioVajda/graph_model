"""Measure the judge before its verdicts count. Two modes, in order.

Step 5 of §9.4's order of work, second half. An unmeasured filter is the mistake
this section has made four times — a filter's rejection log is evidence about the
filter before it is evidence about the writer — so the judge does not gate
anything until it has been scored against rows read by hand.

    --mode sheet   draw a stratified sample and write it out for reading
    --mode score   read the labels back and report precision and recall

The sheet is deliberately printed as prose rather than as a form: the point is
to read the reply against the statements and decide, and a layout that lets you
skim to a checkbox defeats that. The label file is separate from the sheet, so
the judge's own verdict is never in front of the reader while they label — it is
in the sheet only under `--show-verdict`, which is for auditing a disagreement
after the fact, not for labelling.

    src/generalist/tools/run_py.sh -m src.generalist.assistant.analysis.calibrate \
        --mode sheet --batches .../v5 --asks .../v5/ask --voiced .../v5/voice \
        --judged .../v5/judged --out .../v5/calibration --n 100

    src/generalist/tools/run_py.sh -m src.generalist.assistant.analysis.calibrate \
        --mode score --judged .../v5/judged \
        --labels .../v5/calibration/labels.jsonl

**Which way round precision and recall run.** The judge is a *filter*, so the
event it is detecting is a defect. Precision is the share of rows it refused
that were genuinely defective — the cost of a low number is correct rows thrown
away, which is the failure this pipeline keeps having. Recall is the share of
genuinely defective rows it caught, and the cost of a low number is defects in
the shipped set. Both are reported per axis, because a judge can be excellent at
"preserved" and useless at "added".
"""

import argparse
import collections
import glob
import json
import os
import random
import sys

AXES = ("responsive", "preserved", "added")


def _load_jsonl(directory_or_file: str) -> dict:
    paths = ([directory_or_file] if os.path.isfile(directory_or_file)
             else sorted(glob.glob(os.path.join(directory_or_file, "*.jsonl"))))
    out = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    return out


def _sheet(args) -> int:
    turns = _load_jsonl(args.asks)
    replies = _load_jsonl(args.voiced)
    verdicts = _load_jsonl(args.judged) if args.judged else {}

    examples = {}
    for path in sorted(glob.glob(os.path.join(args.batches, "*batch-*.json"))):
        with open(path) as handle:
            for example in json.load(handle)["examples"]:
                if example["id"] in turns and example["id"] in replies:
                    examples[example["id"]] = example
    if not examples:
        raise SystemExit("nothing to calibrate on — no row has both a turn and "
                         "a reply")

    # Stratified by (task, format): a judge measured only on the commonest cell
    # is a judge measured on `report` + prose and nothing else.
    buckets = collections.defaultdict(list)
    for identifier, example in examples.items():
        buckets[(example["intent"]["task"],
                 example["intent"]["style"]["format"])].append(identifier)
    rng = random.Random(args.seed)
    for ids in buckets.values():
        rng.shuffle(ids)
    chosen, cursor = [], 0
    keys = sorted(buckets)
    while len(chosen) < args.n and cursor < max(len(v) for v in buckets.values()):
        for key in keys:
            if cursor < len(buckets[key]) and len(chosen) < args.n:
                chosen.append(buckets[key][cursor])
        cursor += 1

    os.makedirs(args.out, exist_ok=True)
    lines = []
    for i, identifier in enumerate(chosen, 1):
        example = examples[identifier]
        rendered = example["render"]
        intent = example["intent"]
        lines.append(f"--- {i:03d}  {identifier}  {example['cell']} ---")
        lines.append(f"SITUATION : {intent['situation']['persona']} — "
                     f"{intent['situation']['context']}")
        lines.append(f"STYLE     : {intent['style']['register']}; "
                     f"{intent['style']['length']}; {intent['style']['format']}")
        lines.append("")
        lines.append(f"TURN      : {turns[identifier]['turn']}")
        lines.append("")
        lines.append("STATEMENTS:")
        lines += [f"  - {s}" for s in rendered["statements"]]
        lines.append("")
        lines.append(f"RENDERED  : {rendered['reply']}")
        lines.append("")
        lines.append(f"REPLY     : {replies[identifier]['reply']}")
        if args.show_verdict and identifier in verdicts:
            verdict = verdicts[identifier]
            lines.append("")
            lines.append("JUDGE     : " + ", ".join(
                f"{axis}={verdict.get(axis)}" for axis in AXES)
                + f"  ({verdict.get('note', '')})")
        lines.append("")
    with open(os.path.join(args.out, "sheet.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    blank = os.path.join(args.out, "labels.jsonl")
    if not os.path.exists(blank):
        with open(blank, "w") as handle:
            for identifier in chosen:
                handle.write(json.dumps(
                    {"id": identifier, "responsive": None, "preserved": None,
                     "added": None, "note": ""}, sort_keys=True) + "\n")
    print(f"{len(chosen)} rows over {len(buckets)} (task, format) cells")
    print(f"  sheet : {os.path.join(args.out, 'sheet.txt')}")
    print(f"  labels: {blank}")
    return 0


def _score(args) -> int:
    verdicts = _load_jsonl(args.judged)
    labels = _load_jsonl(args.labels)
    paired = [(labels[i], verdicts[i]) for i in sorted(labels)
              if i in verdicts and all(labels[i].get(a) is not None
                                       for a in AXES)]
    if not paired:
        raise SystemExit("no labelled row has a verdict — label the sheet first")

    print(f"{len(paired)} labelled rows with a verdict "
          f"({len(labels) - len(paired)} unlabelled or unjudged)\n")
    print(f"{'axis':12} {'agree':>7} {'prec':>7} {'rec':>7} {'TP':>4} {'FP':>4} "
          f"{'FN':>4} {'TN':>5}   (the event is a DEFECT)")
    report = {}
    for axis in AXES:
        # `added` reads the other way round: True is the defect there, where for
        # `responsive` and `preserved` the defect is False.
        defect = (lambda v: bool(v)) if axis == "added" else (lambda v: not v)
        tp = fp = fn = tn = 0
        for label, verdict in paired:
            hand, machine = defect(label[axis]), defect(verdict.get(axis))
            tp += hand and machine
            fp += (not hand) and machine
            fn += hand and (not machine)
            tn += (not hand) and (not machine)
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        agree = (tp + tn) / len(paired)
        report[axis] = {"precision": precision, "recall": recall,
                        "agreement": agree, "tp": tp, "fp": fp, "fn": fn,
                        "tn": tn}
        print(f"{axis:12} {agree:7.3f} {precision:7.3f} {recall:7.3f} "
              f"{tp:4} {fp:4} {fn:4} {tn:5}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "calibration.json"), "w") as handle:
            json.dump({"n": len(paired), "axes": report}, handle, indent=1,
                      sort_keys=True)

    print("\nDisagreements, for reading:")
    for label, verdict in paired:
        differ = [a for a in AXES if bool(label[a]) != bool(verdict.get(a))]
        if differ:
            print(f"  {label['id']}  {','.join(differ):24} "
                  f"judge: {verdict.get('note', '')[:70]}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("sheet", "score"))
    parser.add_argument("--batches")
    parser.add_argument("--asks")
    parser.add_argument("--voiced")
    parser.add_argument("--judged")
    parser.add_argument("--labels")
    parser.add_argument("--out")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-verdict", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "sheet":
        for name in ("batches", "asks", "voiced", "out"):
            if not getattr(args, name):
                raise SystemExit(f"--mode sheet needs --{name}")
        return _sheet(args)
    for name in ("judged", "labels"):
        if not getattr(args, name):
            raise SystemExit(f"--mode score needs --{name}")
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
