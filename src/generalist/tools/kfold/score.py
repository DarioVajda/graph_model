"""Score `KFOLD_TRANSFER.md`: fix the owed thresholds, then read the crossings.

    src/generalist/tools/kfold/score.py
    src/generalist/tools/kfold/score.py --json out.json

Run it whenever; it reports what is missing rather than failing, so it is
useful while the study is still running.

**Why this is a separate pass and not something the fork does.** Eleven of the
sixteen tasks have no specialist, so their threshold is a reading off runs that
had not finished when the legs were submitted (`KFOLD_TRANSFER.md` §6a). Those
legs carry an `anchor` instead of a `value`, run their full budget, record every
evaluation and report a null crossing. The curve is the durable artefact; the
crossing is a reading off it, and this takes the reading.

**One anchor, for every task in the table.** The threshold is 95 % of the mean
test score of the annealed trunks that **did** train the task. The folds
partition the tasks, so every task trains in three of the four mixtures and the
anchor costs no extra run. The claim is "95 % of what a generalist with the task
in its mixture reaches". The annealed model is the right one to read — a trunk
stops mid-stable-phase by construction and is not the model any comparison
should use — even though the legs fork from the trunk checkpoint, because the
anchor is a level, not a starting point.

**BACE, BBBP and HIV take that same anchor, not their specialists'.** They are
the only three tasks with a specialist, and their legs were submitted carrying
95 % of the specialist *test* score. Reading them that way puts three rows of a
fourteen-row table on a strictly harder bar than the other eleven, and a ratio
measured against one bar does not compare with a ratio measured against another.
The specialist number is better used as the side comparison it actually is: how
far below a dedicated model the generalist level sits, which is the one thing
the other eleven tasks cannot say. It rides along in the `spec95` column and
anchors nothing. The threshold baked into those forks is simply ignored — a
crossing is a reading off a recorded curve, so the bar can change after the run.

**The mean is over folds, not over runs.** Fold A has three seeds and B, C and D
one each, so averaging every cell alike would hand fold A three fifths of the
weight in every non-A anchor and tilt it toward fold A's mixture. Each fold's
cells are averaged first, then those fold means.

**`bond_path` and `longest_chain` are out of the headline.** Held out of every
mixture, they have no trunk that trained them and so no anchor of this kind;
their bar is the from-base leg's own final score, which answers "how much sooner
does the trunk reach what the backbone ends at" rather than "how much sooner
does it reach competence". A second convention costs more to explain than two
rows are worth. They still run and still score, printed below the table and
labelled `self`, and they stay out of anything quoted as the study's result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)

FRACTION = 0.95


def read_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_history(out_dir, leg) -> dict:
    """A leg rebuilt from its own `history.jsonl`, for a fork that never finished.

    `result.json` is written once, after every leg has run, so a fork killed in
    its second leg leaves the first leg's finished curve nowhere else. `_run_leg`
    appends each evaluation as it is taken; this reads those back in the shape
    `result.json` would have given, so a walled job still scores.

    A kill can leave the last line half-written, so a line that does not parse
    ends the read rather than failing it — what came before is intact.
    """
    path = os.path.join(out_dir, leg, "history.jsonl")
    history = []
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    break
                history.append([int(row["step"]), row.get("metrics") or {}])
    except OSError:
        return {}
    if not history:
        return {}
    return {"history": history, "metrics": history[-1][1], "recovered": True}


def anneal_metrics(cell_config) -> dict:
    """The annealed model's flat metric dict, or {} if the anneal has not run."""
    path = os.path.join(f"{cell_config.run_dir()}-anneal-"
                        f"{cell_config.max_steps}", "result.json")
    result = read_json(path) or {}
    for leg in (result.get("legs") or {}).values():
        if leg.get("start") == "anneal" or leg.get("leg") == "anneal":
            return leg.get("metrics") or {}
    return {}


def crossing(history, key, value, consecutive=3, direction="max"):
    """First step of the first run of `consecutive` evaluations that meet it.

    The same rule `fork.steps_to_target` applies, restated here because this
    reads a threshold the fork did not have. A step missing the metric breaks
    the run rather than being skipped: with a coarse grid, treating an absent
    value as neutral would let two crossings an hour apart count as persistence.
    """
    want_max = direction == "max"
    start, streak = None, 0
    for step, metrics in sorted(history, key=lambda pair: pair[0]):
        got = (metrics or {}).get(key)
        met = got is not None and (
            (float(got) >= value) if want_max else (float(got) <= value))
        if not met:
            start, streak = None, 0
            continue
        if streak == 0:
            start = int(step)
        streak += 1
        if streak >= consecutive:
            return start
    return None


def area(history, key):
    """Mean metric over the budget — time-to-threshold's companion number.

    Time-to-threshold is sensitive to an arbitrary threshold, which is exactly
    why the transfer literature pairs it with an area. Trapezoid over the
    evaluated steps, divided by the span, so it reads on the metric's own scale.
    """
    points = [(int(s), float((m or {}).get(key)))
              for s, m in sorted(history, key=lambda pair: pair[0])
              if (m or {}).get(key) is not None]
    if len(points) < 2:
        return None
    total = 0.0
    for (s0, v0), (s1, v1) in zip(points, points[1:]):
        total += (s1 - s0) * (v0 + v1) / 2.0
    return total / (points[-1][0] - points[0][0])


def median(values):
    """Middle value of whatever is not None, or None if nothing is.

    Fold A's three seeds are the only replication in the study (§7), so a task
    that ran more than one is summarised by the middle seed rather than by
    whichever cell name sorts first. One seed crossing a step early is exactly
    the noise three seeds exist to expose, and a mean would let it move the
    headline; the median does not.
    """
    kept = sorted(value for value in values if value is not None)
    if not kept:
        return None
    middle = len(kept) // 2
    if len(kept) % 2:
        return kept[middle]
    return (kept[middle - 1] + kept[middle]) / 2.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", default=None,
                        help="also write the table here, for the write-up.")
    args = parser.parse_args(argv)

    from src.generalist.config import RunConfig, TRANSFER_FOLDS, config_cells
    from src.generalist.tools.kfold.adapt_all import (
        EVERY_FOLD, FOLD_CONFIG, fork_dir, legs)

    # Every fold's cells, and the annealed metrics of each.
    cells_by_fold, metrics_by_fold = {}, {}
    for fold, path in FOLD_CONFIG.items():
        cells_by_fold[fold] = {name: RunConfig(**values)
                               for name, values in config_cells(path).items()}
        metrics_by_fold[fold] = {
            name: anneal_metrics(config)
            for name, config in cells_by_fold[fold].items()}

    def cross_fold_anchor(task, metric_key):
        """Mean over the *folds* that trained the task, seeds averaged within.

        Fold-then-seed, not a flat mean over cells. Fold A runs three seeds and
        the others one apiece, so a flat mean would give fold A three fifths of
        every non-A anchor and pull the threshold toward fold A's mixture — a
        property of how the pilot was sized, not of the task.
        """
        per_fold = []
        for fold, tasks in TRANSFER_FOLDS.items():
            if task in tasks:
                continue                      # this fold held it out
            seeds = [float(metrics[metric_key])
                     for metrics in metrics_by_fold[fold].values()
                     if metrics.get(metric_key) is not None]
            if seeds:
                per_fold.append(sum(seeds) / len(seeds))
        if not per_fold:
            return None, 0
        return sum(per_fold) / len(per_fold), len(per_fold)

    rows, appendix, missing = [], [], []
    for fold, task, _config, _metric, _value, starts in legs():
        cells = cells_by_fold[fold]

        # **Every cell of the fold, not the first one.** Fold A runs three seeds
        # and they are the study's only error bar (§7); reading `sorted(cells)[0]`
        # scores s0 and discards s1 and s2 without saying so, which turns the one
        # measurement of between-run variance into a single number that looks
        # exactly like the unreplicated folds.
        per_seed, plan, why = [], None, None
        for cell in sorted(cells):
            out_dir = fork_dir(cells[cell], task)
            cell_plan = read_json(os.path.join(out_dir, "fork.json"))
            if not cell_plan:
                why = why or "has not run"
                continue
            plan = plan or cell_plan
            result = read_json(os.path.join(out_dir, "result.json"))
            recovered = None
            if not result:
                # No `result.json` means the fork did not reach the end of its
                # second leg. Whatever curve the legs did record is on disk.
                partial = {name: read_history(out_dir, name)
                           for name in ("parent", "base")}
                partial = {k: v for k, v in partial.items() if v}
                if not partial:
                    why = why or "has not run"
                    continue
                result, recovered = {"legs": partial}, sorted(partial)
            per_seed.append((cell, result, recovered))

        if not per_seed:
            missing.append((fold, task, why or "has not run"))
            continue

        target = plan.get("target") or {}
        key = target.get("metric")
        consecutive = int(target.get("consecutive", 1))
        direction = target.get("direction", "max")

        # The specialist threshold the fork was submitted with, where one exists.
        # It no longer sets the bar — it is the side comparison that says how far
        # below a dedicated model the generalist level sits.
        specialist = (None if target.get("value") is None
                      else float(target["value"]))

        seed_legs = []
        for cell, result, recovered in per_seed:
            legs_json = result.get("legs") or {}
            parent = (legs_json.get("parent") or {})
            base = (legs_json.get("base") or {})

            # The base leg is shared: bond_path and longest_chain run one off
            # their owning fold and the parent leg alone off the other three.
            if not base and task in EVERY_FOLD:
                owner = next(f for f, ts in TRANSFER_FOLDS.items()
                             if task in ts)
                shared = read_json(os.path.join(
                    fork_dir(sorted(cells_by_fold[owner].values(),
                                    key=lambda c: c.run_name)[0], task),
                    "result.json")) or {}
                base = ((shared.get("legs") or {}).get("base") or {})
            seed_legs.append((cell, parent, base, recovered))

        if task in EVERY_FOLD:
            finals = [float((base.get("metrics") or {})[key])
                      for _cell, _parent, base, _rec in seed_legs
                      if (base.get("metrics") or {}).get(key) is not None]
            threshold = (FRACTION * (sum(finals) / len(finals))
                         if finals else None)
            kind, n = "self", len(finals)
        else:
            anchor, n = cross_fold_anchor(task, key)
            threshold = None if anchor is None else FRACTION * anchor
            kind = "cross-fold"

        if threshold is None:
            missing.append((fold, task, f"{kind} anchor not available yet"))
            continue

        seeds = []
        for cell, parent, base, recovered in seed_legs:
            parent_steps = crossing(parent.get("history") or [], key, threshold,
                                    consecutive, direction)
            base_steps = crossing(base.get("history") or [], key, threshold,
                                  consecutive, direction)
            seeds.append({
                "cell": cell,
                "parent_steps": parent_steps, "base_steps": base_steps,
                "ratio": (base_steps / parent_steps
                          if parent_steps and base_steps else None),
                "parent_area": area(parent.get("history") or [], key),
                "base_area": area(base.get("history") or [], key),
                "recovered": recovered})

        ratio = median([seed["ratio"] for seed in seeds])
        row = {
            "fold": fold, "task": task, "metric": key,
            "threshold": round(threshold, 4), "anchor": kind, "anchor_n": n,
            "spec95": None if specialist is None else round(specialist, 4),
            "seeds": seeds,
            "recovered": sorted({name for seed in seeds
                                 for name in (seed["recovered"] or ())}) or None,
            "parent_steps": median([seed["parent_steps"] for seed in seeds]),
            "base_steps": median([seed["base_steps"] for seed in seeds]),
            "ratio": None if ratio is None else round(ratio, 2),
            "parent_area": median([seed["parent_area"] for seed in seeds]),
            "base_area": median([seed["base_area"] for seed in seeds]),
            "starts": starts or "parent,base"}
        (appendix if task in EVERY_FOLD else rows).append(row)

    def spread(seeds, field):
        """``lo-hi`` across the seeds, or "" when there is nothing to show.

        Printed next to the median so a one-seed fold and a three-seed fold are
        never mistaken for each other, and so a wide spread is visible in the
        table rather than only in the JSON.
        """
        values = [seed[field] for seed in seeds if seed[field] is not None]
        if len(values) < 2 or min(values) == max(values):
            return ""
        return f"{min(values):g}-{max(values):g}"

    def table(title, entries, note):
        print(f"{title}\n{'fold':4s} {'task':22s} {'n':>2s} {'sd':>2s} "
              f"{'thresh':>7s} {'spec95':>7s} {'parent':>7s} {'base':>7s} "
              f"{'ratio':>6s}  {'spread (parent | base)':22s}  metric")
        for row in entries:
            spec = ("—" if row["spec95"] is None else f"{row['spec95']:.4f}")
            mark = ("  [partial: no result.json, read from "
                    f"{'+'.join(row['recovered'])}]" if row["recovered"] else "")
            seeds = row["seeds"]
            both = (f"{spread(seeds, 'parent_steps') or '·'} | "
                    f"{spread(seeds, 'base_steps') or '·'}"
                    if len(seeds) > 1 else "")
            print(f"{row['fold']:4s} {row['task']:22s} {row['anchor_n']:2d} "
                  f"{len(seeds):2d} {row['threshold']:7.4f} {spec:>7s} "
                  f"{str(row['parent_steps']):>7s} "
                  f"{str(row['base_steps']):>7s} {str(row['ratio']):>6s}  "
                  f"{both:22s}  {row['metric']}{mark}")
        if not entries:
            print("  (nothing scorable yet)")
        print(f"  {note}\n")

    table("== the study ==", rows,
          "thresh = 95 % of the mean annealed score over the `n` folds that "
          "trained the task.\n  spec95 = 95 % of the specialist, where one "
          "exists; reported for comparison, anchors nothing.\n  sd = seeds "
          "scored; parent/base are the median over them and `spread` their "
          "range.")
    table("== held out of every mixture (not part of the headline) ==", appendix,
          "thresh = 95 % of the from-base leg's own final score. A different "
          "question; quote it as one.")

    for fold, task, why in missing:
        print(f"[wait] fold {fold} / {task}: {why}")
    print(f"\n{len(rows)} scored, {len(appendix)} in the appendix, "
          f"{len(missing)} waiting")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"fraction": FRACTION, "rows": rows,
                       "appendix": appendix,
                       "missing": [{"fold": f, "task": t, "why": w}
                                   for f, t, w in missing]}, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
