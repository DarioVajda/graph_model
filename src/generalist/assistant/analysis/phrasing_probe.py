"""Does the assistant answer "does it contain X?" or only the wording it was trained on?

The thirty-question case study found one stable failure: asked whether a molecule
contains a functional group that it does contain, every assistant checkpoint
says no, through better data, 2x and 4x budget. Thirty questions leave four or
five gradeable presence cases per checkpoint, so one flip is noise and no
intervention can be read off them. This is the same question asked about a
thousand times, balanced yes/no, graded against RDKit.

Each question is one molecule, one group and one **wording family**. Two of the
families are the wordings the composed set itself uses ("Does this molecule
contain …", "I believe this molecule contains …"); the rest never occur in it.
That split is the point of the probe:

* fails on the unseen wordings only -> the model learned the set's phrasing, and
  widening the phrasing is the fix;
* fails on the seen wordings too -> phrasing is not the cause, and widening it
  would be aimed at the wrong thing.

Molecules come from both roles. Test-role molecules are the measurement;
train-role ones are a control for whether the model can answer about a molecule
it was trained on.

    # 1. write the questions (CPU)
    src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.analysis.phrasing_probe \\
        make --out src/generalist/results/assistant/phrasing_probe
    # 2. generate, one job per checkpoint, through case_study
    GPU=1 src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.analysis.case_study \\
        --cases .../phrasing_probe/cases.jsonl --checkpoint new_1x_s0=<ckpt> \\
        --config ... --cell ... --out .../phrasing_probe/new_1x_s0
    # 3. grade every run directory under the probe
    python -m src.generalist.assistant.analysis.phrasing_probe \\
        grade --out src/generalist/results/assistant/phrasing_probe

Grading reads the reply's claim about presence, not its wording: a leading
yes/no (or correct/incorrect for a premise) when there is one, otherwise the
sentence that names the group, negated or not. A reply that makes no readable
claim is counted as unclear and reported, never silently scored.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics
import sys

from ..facts import article

COMPOSED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "assistant", "final", "composed")

#: ``(name, template, inverted, seen)``. ``inverted`` means a "yes" asserts the
#: group is absent; ``seen`` means the composed set asks this way.
FAMILIES = (
    ("seen_contain", "Does this molecule contain {a} {g}?", False, True),
    ("seen_premise", "I believe this molecule contains {a} {g}. Is that correct?",
     False, True),
    ("is_there", "Is there {a} {g} anywhere in this structure?", False, False),
    ("have", "Does this compound have {a} {g}?", False, False),
    ("casual", "does it have {a} {g} in it", False, False),
    ("embedded", "I'm going through a list of compounds and need to flag some "
                 "of them. Is {a} {g} present in this one?", False, False),
    ("free_of", "Is this molecule free of {g}s?", True, False),
)

#: Wordings kept out of any widened training set, so a gain on them is a gain on
#: wordings training never saw: two plain questions, a claim of presence, a
#: claim of absence (``inverted``: agreeing means absent) and two negated
#: questions. Asked only under ``make --held-out``, so the original probe's
#: cases stay as they were.
#:
#: Tried and dropped (2026-10-03): 1,800 added training rows in many wordings,
#: with claims to correct and negated questions, moved the 2x anneal from 0.64
#: to 0.65 here, made wrong-claim correction worse (0.37 -> 0.15) and cost 1-2.5
#: AUC points on the property tasks, while ``fg_presence`` itself stayed at
#: 0.99. The model knows the groups; widening the wording does not carry that
#: into conversation.
HELD_OUT_FAMILIES = (
    ("detect", "Do you detect {a} {g} in this molecule?", False, False),
    ("found", "Can {a} {g} be found in this compound?", False, False),
    ("claim_present", "Correct me if I'm wrong, but this molecule contains "
                      "{a} {g}, doesn't it?", False, False),
    ("claim_absent", "This compound doesn't contain {a} {g}, correct?", True, False),
    ("clear_of", "Is this compound clear of {gp}?", True, False),
    ("none_appears", "Is it the case that no {g} appears in this molecule?",
     True, False),
)

PLURAL = {
    "carboxylic acid": "carboxylic acids", "hydroxyl group": "hydroxyl groups",
    "primary amine": "primary amines", "ketone": "ketones", "ether": "ethers",
    "nitro group": "nitro groups", "amide": "amides",
    "sulfonamide": "sulfonamides", "nitrile": "nitriles", "halogen": "halogens",
}

#: How a reply names each group. `\bamides?` cannot match inside "sulfonamide".
GROUP_WORDS = {
    "carboxylic acid": r"carboxyl",
    "hydroxyl group": r"hydroxyl|alcohol|\boh\b",
    "primary amine": r"amine",
    "ketone": r"ketone",
    "ether": r"\bethers?\b",
    "nitro group": r"\bnitro",
    "amide": r"\bamides?\b",
    "sulfonamide": r"sulfonamide",
    "nitrile": r"nitrile",
    "halogen": r"halogen|fluor|chlor|brom|iod",
}

YES_WORDS = {"yes", "yeah", "yep", "correct", "true", "right", "indeed"}
NO_WORDS = {"no", "nope", "incorrect", "false", "wrong"}
NEGATION = re.compile(r"\b(no|not|without|lacks?|lacking|free of|absent|none|"
                      r"neither|nor|cannot)\b|n't")
ASSERTION = re.compile(r"contain|\bhas\b|\bhave\b|present|includ|there (is|are)|"
                       r"features?|bears?|\bgot\b|'s an?\b")


def _molecules(role: str) -> list:
    path = os.path.join(COMPOSED, f"{role}.jsonl")
    seen, out = set(), []
    with open(path) as handle:
        for line in handle:
            smiles = json.loads(line)["smiles"]
            if smiles not in seen:
                seen.add(smiles)
                out.append(smiles)
    return out


def make(args) -> int:
    from rdkit import Chem

    from ....experiments.molecules.tasks import _SMARTS, FUNCTIONAL_GROUPS

    families = FAMILIES + (HELD_OUT_FAMILIES if args.held_out else ())
    rng = random.Random(args.seed)
    cases = []
    for role, n in (("test", args.n_test), ("train", args.n_train)):
        pool = _molecules(role)
        rng.shuffle(pool)
        taken = 0
        for smiles in pool:
            if taken == n:
                break
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            present = [g for g in FUNCTIONAL_GROUPS
                       if mol.HasSubstructMatch(_SMARTS[g])]
            absent = [g for g in FUNCTIONAL_GROUPS if g not in present]
            if not present or not absent:
                continue
            taken += 1
            for family, template, inverted, seen in families:
                for group, truth in ((rng.choice(present), True),
                                     (rng.choice(absent), False)):
                    question = template.format(a=article(group), g=group,
                                               gp=PLURAL[group])
                    cases.append({
                        "id": f"pp-{len(cases):05d}", "smiles": smiles,
                        "atoms": [], "question": question,
                        "probe": family, "expect": "yes" if truth != inverted else "no",
                        "group": group, "present": truth, "family": family,
                        "inverted": inverted, "seen": seen, "role": role})
        if taken < n:
            raise SystemExit(f"only {taken} usable {role}-role molecules, wanted {n}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "cases.jsonl")
    with open(path, "w") as handle:
        for row in cases:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} questions over {args.n_test} test-role and "
          f"{args.n_train} train-role molecules to {path}")
    return 0


def claimed_presence(reply: str, group: str, inverted: bool):
    """True / False for what the reply says about the group, None if unreadable."""
    low = (reply or "").strip().lower()
    names = re.compile(GROUP_WORDS[group])
    # "No hydroxyl group." is a statement of absence, not an answer word.
    lead = re.match(r"\W*no\s+(\w+(?:\s\w+)?)", low)
    if lead and names.search(lead.group(1)):
        return False
    first = re.match(r"\W*([a-z]+)", low)
    word = first.group(1) if first else ""
    if word in YES_WORDS or word in NO_WORDS:
        said_yes = word in YES_WORDS
        return (not said_yes) if inverted else said_yes
    for sentence in re.split(r"(?<=[.!?;\n])\s*", low):
        if not names.search(sentence):
            continue
        if NEGATION.search(sentence):
            return False
        if ASSERTION.search(sentence):
            return True
    return None


def _runs(out: str, cases: dict) -> dict:
    """``{label: {case id: reply}}`` from every case_study output under ``out``."""
    replies = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(out, "*", "case_study.jsonl"))):
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] not in cases:
                    continue
                for label, text in row["replies"].items():
                    replies[label][row["id"]] = text
    return replies


def _arm(label: str) -> str:
    return re.sub(r"_s\d+$", "", label)


def grade(args) -> int:
    with open(os.path.join(args.out, "cases.jsonl")) as handle:
        cases = {row["id"]: row for row in map(json.loads, handle)}
    replies = _runs(args.out, cases)
    if not replies:
        raise SystemExit(f"no case_study.jsonl under {args.out}/*/")

    graded = []
    for label, by_id in replies.items():
        for case_id, reply in by_id.items():
            case = cases[case_id]
            claim = claimed_presence(reply, case["group"], case["inverted"])
            graded.append({**case, "label": label, "arm": _arm(label),
                           "reply": reply, "claim": claim,
                           "right": claim is not None and claim == case["present"]})
    with open(os.path.join(args.out, "graded.jsonl"), "w") as handle:
        for row in graded:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    arms = sorted({row["arm"] for row in graded}, key=lambda a: (
        min(i for i, row in enumerate(graded) if row["arm"] == a)))
    lines = [f"# Phrasing probe", "",
             f"{len(cases)} questions per checkpoint; accuracy is the mean over "
             "seeds, ± the spread between the best and worst seed. `unclear` "
             "replies count as wrong and are reported separately.", ""]

    def cell(rows_by_label, keep):
        scores = []
        for rows in rows_by_label.values():
            kept = [r for r in rows if keep(r)]
            if kept:
                scores.append(sum(r["right"] for r in kept) / len(kept))
        if not scores:
            return "--"
        mean = statistics.fmean(scores)
        spread = (max(scores) - min(scores)) / 2 if len(scores) > 1 else 0.0
        return f"{mean:.2f}±{spread:.2f}"

    slices = (
        ("all", lambda r: True),
        ("seen, truth yes", lambda r: r["seen"] and r["present"]),
        ("seen, truth no", lambda r: r["seen"] and not r["present"]),
        ("unseen, truth yes", lambda r: not r["seen"] and r["present"]),
        ("unseen, truth no", lambda r: not r["seen"] and not r["present"]),
        ("test-role, truth yes", lambda r: r["role"] == "test" and r["present"]),
        ("train-role, truth yes", lambda r: r["role"] == "train" and r["present"]),
    )
    lines.append("| arm | seeds | " + " | ".join(s for s, _ in slices)
                 + " | says absent | unclear |")
    lines.append("|---|--:|" + "---|" * (len(slices) + 2))
    by_arm = {}
    for arm in arms:
        rows_by_label = collections.defaultdict(list)
        for row in graded:
            if row["arm"] == arm:
                rows_by_label[row["label"]].append(row)
        by_arm[arm] = rows_by_label
        everything = [r for rows in rows_by_label.values() for r in rows]
        absent = sum(r["claim"] is False for r in everything) / len(everything)
        unclear = sum(r["claim"] is None for r in everything) / len(everything)
        lines.append(f"| {arm} | {len(rows_by_label)} | "
                     + " | ".join(cell(rows_by_label, keep) for _, keep in slices)
                     + f" | {absent:.2f} | {unclear:.2f} |")

    asked = {row["family"] for row in graded}
    families = [f for f, *_ in FAMILIES + HELD_OUT_FAMILIES if f in asked]
    lines += ["", "## By wording family, truth yes / truth no", ""]
    lines.append("| arm | " + " | ".join(families) + " |")
    lines.append("|---|" + "---|" * len(families))
    for arm in arms:
        cells = []
        for family in families:
            yes = cell(by_arm[arm], lambda r, f=family: r["family"] == f and r["present"])
            no = cell(by_arm[arm], lambda r, f=family: r["family"] == f and not r["present"])
            cells.append(f"{yes} / {no}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += ["", "## Unclear replies, a sample", ""]
    unclear_rows = [r for r in graded if r["claim"] is None]
    for row in random.Random(0).sample(unclear_rows, min(12, len(unclear_rows))):
        lines.append(f"* `{row['label']}` {row['question']!r} -> {row['reply'][:160]!r}")

    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w") as handle:
        handle.write(report)
    print(report)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--n-test", type=int, default=50)
    m.add_argument("--n-train", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--held-out", action="store_true",
                   help="also ask HELD_OUT_FAMILIES")
    g = sub.add_parser("grade")
    g.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return make(args) if args.mode == "make" else grade(args)


if __name__ == "__main__":
    sys.exit(main())
