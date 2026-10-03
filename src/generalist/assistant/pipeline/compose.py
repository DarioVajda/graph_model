"""Attach few-shot demonstrations to an accepted assistant set (§9.4).

Step 6 of §9.4's order of work, second half, after `accept.py`. It reads
an accepted set and gives a share of its rows worked examples drawn **from the
set itself**, so that every demonstration has an RDKit fact sheet behind it and
has already passed every filter in the accept pass.

    RUNMOD=src.generalist.assistant.pipeline.compose src/generalist/tools/launch/run_cli.sh \
        --accepted src/generalist/results/assistant/final/accepted \
        --out src/generalist/results/assistant/final/composed

This is the whole reason the axis is back. §9.4 withdrew few-shot because a
demonstration written by the writer is about a molecule nothing computed facts
for, and so cannot be verified — and the first batch produced a 4-atom smallest
ring in a molecule with no rings at all. Composing demonstrations out of
already-accepted rows removes the writer from the demonstration entirely: the
writer never sees the shots, and the shots are rows that were already going into
the set on their own account.

**Demonstrations are drawn, not written, and the draw has rules.** A
demonstration may not be the target, may not be the same molecule as the target,
and may not state one of the target's own facts — that last one is the copy
shortcut, and it is the only way a demonstration can make an example answerable
without reading the graph. Within what is left the selector prefers a row with
the target's format and a shared fact family, because a demonstration exists to
show the shape of the wanted answer.

**Test-role targets take train-role demonstrations only.** Two test rows in one
context would stop being two independent measurements: the model would see one
test item's answer while being scored on another.

**Demonstration polarity is drawn 50/50, independently of the target's.** This
one is not obvious and the first composed set got it wrong. Refusing a
demonstration that states the target's own value stops a demonstration being
*copied*, but it does not stop the demonstration set being *informative*: the
first set produced a yes-answer target with four yes-answer demonstrations
behind it, because the pool skews yes and nothing in the selector looked at
polarity. A model can answer that row from the demonstrations alone. Nor is the
fix to prefer the opposite polarity — done systematically, that is the same
shortcut inverted, and the model learns to answer against the demonstrations.
The only arrangement that carries no information is a coin flip that does not
look at the target, so that is what each slot asks for, and the format and
family preference operates inside whatever the flip allows.

The output is the accepted JSONL with two fields added — ``shots``, the drawn
demonstrations, and ``pointer``, the sentence that refers to them. The ``question``
and ``answer`` fields are untouched, so a composed set verifies exactly as the
accepted set it came from does.

The pointer sentences, the copy rule's test for a stated fact, and the verifier
for rows built before rendered statements come from the domain (`--domain`,
molecules by default); the draw is the same for every domain.
"""

import argparse
import json
import os
import random
import sys

from ..domain import get_domain

#: How many candidates are scored per target. The pool is thousands of rows and
#: the preference only needs a good match rather than the best one, so this is
#: sampled rather than swept — scoring every pool row against every target is
#: quadratic and buys a demonstration nobody can tell apart from this one.
CANDIDATE_SAMPLE = 80


def _take(ranked, shots, used, want: str):
    """The best remaining candidate with polarity ``want``, or None.

    ``want`` empty means any. Order is `shot_candidates`' ranking, so the
    polarity constraint narrows the field without giving up the format and
    family preference inside it.
    """
    from ..shots import SHOT_REUSE_CEILING, fact_polarity

    for candidate in ranked:
        if any(s["id"] == candidate["id"] for s in shots):
            continue
        if any(s["key"] == candidate["key"] for s in shots):
            continue
        if used.get(candidate["id"], 0) >= SHOT_REUSE_CEILING:
            continue
        if want and fact_polarity(candidate["facts"]) != want:
            continue
        return candidate
    return None


def _agreement(splits, fact_polarity) -> dict:
    """How often a demonstration's polarity matched its target's, at chance or not.

    Counted only where both sides *have* a polarity — a count fact has none, and
    scoring those as disagreement would bury the signal under the majority of
    the set.
    """
    agree = total = 0
    for rows in splits.values():
        for row in rows:
            target = fact_polarity(row["facts"])
            if not target:
                continue
            for shot in row.get("shots") or []:
                if not shot.get("polarity"):
                    continue
                total += 1
                agree += shot["polarity"] == target
    return {"pairs": total, "agreed": agree,
            "rate": round(agree / total, 4) if total else None}


def _reverify(row, domain=None) -> bool:
    """Does this row still pass the check that admitted it?"""
    domain = get_domain(domain)
    if "statements" in row:
        from ..intents import TERSE_FORMATS
        from .accept import (dropped_statements, format_met,
                             states_the_gloss, states_the_verdict)

        fmt = (row.get("brief") or {}).get("format", "prose")
        if dropped_statements(row, row["answer"], fmt, domain):
            return False
        if not format_met(row["answer"], fmt, row.get("skeleton")):
            return False
        # The same blind spot the accept pass had, twice over: a `decide` reply
        # owes an answer to the constraint and an `explain` reply owes its
        # general sentence, and neither is one of the statements, so counting
        # statements cannot see either go missing.
        if not states_the_gloss(row, row["answer"], domain):
            return False
        if row.get("task") == "decide" and fmt not in TERSE_FORMATS:
            return states_the_verdict(row, row["answer"], fmt)
        return True
    return domain.legacy_verify(row["answer"], row["facts"],
                                row["brief"])["passed"]


def _load(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", required=True,
                        help="directory holding train.jsonl and test.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot-fraction", type=float, default=None,
                        help="overrides shots.SHOT_FRACTION")
    parser.add_argument("--domain", default=None,
                        help="the assistant domain the set was built for "
                             "(default: molecules)")
    args = parser.parse_args(argv)
    domain = get_domain(args.domain)

    from ..shots import (SHOT_FRACTION, draw_shot_count, fact_polarity,
                         question_text, shot_candidates, shot_pointer)

    fraction = args.shot_fraction if args.shot_fraction is not None else SHOT_FRACTION
    rng = random.Random(args.seed)

    splits = {}
    for name in ("train", "test"):
        path = os.path.join(args.accepted, f"{name}.jsonl")
        splits[name] = _load(path) if os.path.exists(path) else []
    print(f"{len(splits['train'])} train rows, {len(splits['test'])} test rows")

    # Demonstrations always come from the train split, for both splits. For a
    # test target that is the independence rule above; for a train target it is
    # simply where the rows are.
    pool = splits["train"]
    used = {}
    polarity_drawn = {}
    os.makedirs(args.out, exist_ok=True)

    stats = {"train": {}, "test": {}}
    for name, rows in splits.items():
        counts = {}
        for row in rows:
            wanted = draw_shot_count(rng, fraction)
            shots = []
            if wanted:
                candidates = rng.sample(pool, min(CANDIDATE_SAMPLE, len(pool)))
                ranked = shot_candidates(row, candidates, domain)
                # The polarity each slot is *asked* for, drawn 50/50 and
                # independently of the target's own. See `_wanted_polarity`.
                target_polarity = fact_polarity(row["facts"])
                for slot in range(wanted):
                    want = (rng.choice(("yes", "no")) if target_polarity else "")
                    picked = _take(ranked, shots, used, want)
                    if picked is None:
                        picked = _take(ranked, shots, used, "")
                    if picked is None:
                        break
                    # `question_text` and not the bare field: a demonstration
                    # drawn from a `needs_clarification` row is an exchange, and
                    # showing only its opening turn would put a question with no
                    # answer in front of the target, answered.
                    shots.append({"id": picked["id"], "key": picked["key"],
                                  "question": question_text(picked),
                                  "answer": picked["answer"],
                                  "polarity": fact_polarity(picked["facts"])})
                    used[picked["id"]] = used.get(picked["id"], 0) + 1
                    polarity_drawn[want or "n/a"] = \
                        polarity_drawn.get(want or "n/a", 0) + 1
            row["shots"] = shots
            row["pointer"] = shot_pointer(rng, domain) if shots else ""
            counts[len(shots)] = counts.get(len(shots), 0) + 1
        stats[name] = counts
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    # Composition must not disturb what was verified. It only ever adds fields,
    # so this should be every row every time — which is exactly why it is worth
    # asserting rather than assuming: a future change that folds the pointer
    # into `question`, or rewrites an answer, would show up here as a drop.
    #
    # Which verifier that is depends on where the row came from, and the row
    # says: a rendered row carries the `statements` its reply was built from, and
    # re-reading it with the composition verifier would be checking it against a
    # standard it was never held to.
    reverified = {name: sum(1 for row in rows if _reverify(row, domain))
                  for name, rows in splits.items()}
    summary = {"reverified": reverified,
               "shots_by_count": stats,
               "shot_fraction_asked": fraction,
               "rows_used_as_demonstrations": len(used),
               "max_reuse": max(used.values()) if used else 0,
               "polarity_asked": polarity_drawn,
               # The measurement the balance rule exists for: how often a
               # demonstration ended up agreeing with the target it stands in
               # front of. On a yes/no target this should sit at chance, and a
               # number far from it means the pool could not supply the
               # polarity the flip asked for.
               "polarity_agreement": _agreement(splits, fact_polarity)}
    for name, counts in stats.items():
        total = sum(counts.values())
        with_shots = total - counts.get(0, 0)
        summary[f"{name}_with_shots"] = with_shots
        summary[f"{name}_share"] = round(with_shots / total, 4) if total else 0.0
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
