"""Declare the intents, render their replies, and write the batches.

Step 3 of §9.4's order of work. The subjects and their fact sheets come from the
domain's source (`Domain.source`; for molecules, `molecules/pool.py`, which
samples the §3 partition stratified by source and heavy-atom count). What this
file adds is what happens after the sheet: instead of drawing a style brief and
handing a model a subset of facts to write about, it declares an **intent** and
**renders the reply**, and the writer is left with nothing to be wrong about.

    src/generalist/tools/run_py.sh -m src.generalist.assistant.pipeline.build \
        --config src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc \
        --out src/generalist/results/assistant/v5 --n-train 11000 --n-test 900

`--domain` picks the domain (molecules by default), and the manifest records it.

**The build fails loudly.** A sheet too thin to support any task is dropped and
counted; a coverage cell that comes out empty is a build failure, not a quiet
shortfall. A bank that silently loses a family looks exactly like a bank that
never had one, which is the lesson the harvester's two silent gate bugs left.

**Write more than the target.** The accept pass takes a share of what is written,
so `--n-train` is the number of *intents*, not the number of accepted rows. At
the measured yields, ~11,000 intents land the 9,600 the fork arithmetic wants.

**A top-up build needs `--id-prefix`.** Ids restart at `train-00000` every build
and batch files at `train-batch-0000.json`, so a second build that tops a short
set up would collide with the first in both — silently, because the ask, voice
and judge passes join on the id and the accept pass would read one row's reply
against another row's statements. The prefix goes on the id *and* the batch
filename, which is what makes the two builds mergeable: copy the batches and
their pass outputs into one directory and accept over the union, so the dedup
pool is the whole set rather than each half separately.
"""

import argparse
import collections
import json
import os
import random

from ..domain import get_domain

BATCH_SIZE = 20

#: Below this many intents in a split, an empty coverage cell is reported but not
#: refused. The rarest task is drawn at 0.08 and the rarest twist inside it far
#: below that, so a 40-intent smoke split misses a cell often enough that failing
#: on it would only teach me to pass a flag.
TASK_FLOOR = 400


def _intent_for(entry, source, situations, rng, role, domain):
    """One declared intent over one subject, with its rendered reply."""
    from ..intents import sample_intent

    drawn = source.sheet(entry, rng)
    if drawn is None:
        return None
    subject_id, subject, sheet = drawn
    drawn = sample_intent(subject_id, subject, sheet, situations, rng, role,
                          domain)
    if drawn is None:
        return None
    intent, rendered = drawn
    return {
        **source.fields(entry),
        "sheet": [f.to_json() for f in sheet],
        "intent": intent.to_json(),
        "render": rendered.to_json(),
        "cell": intent.cell(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell", default=None)
    parser.add_argument("--out", required=True, help="directory for the batches")
    parser.add_argument("--n-train", type=int, default=11000)
    parser.add_argument("--n-test", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-prefix", default="",
                        help="prefix for ids and batch filenames, so a top-up "
                             "build can be merged with an earlier one")
    parser.add_argument("--domain", default=None,
                        help="the assistant domain to build for (default: "
                             "molecules)")
    args = parser.parse_args(argv)

    from ...config import RunConfig, load_config_file
    from ..intents import load_situations
    from ..render import TASKS

    domain = get_domain(args.domain)
    config = RunConfig(**load_config_file(args.config, args.cell)).validate()
    rng = random.Random(args.seed)
    situations = load_situations(domain=domain)

    source = domain.source(config)
    by_role = source.by_role(("train", "test"))
    print(f"pool: {len(by_role['train'])} train-role, {len(by_role['test'])} "
          f"test-role subjects; {len(situations)} situations", flush=True)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"seed": args.seed, "config": os.path.abspath(args.config),
                "cell": args.cell, "id_prefix": args.id_prefix,
                "domain": domain.name, "situations": len(situations),
                "batches": [], "counts": {}, "dropped": {}, "coverage": {}}

    for role, wanted in (("train", args.n_train), ("test", args.n_test)):
        chosen = source.sample(by_role[role], wanted, rng)
        examples, dropped = [], 0
        for entry in chosen:
            built = _intent_for(entry, source, situations, rng, role, domain)
            if built is None:
                dropped += 1
                continue
            examples.append(built)
        for i, example in enumerate(examples):
            example["id"] = f"{args.id_prefix}{role}-{i:05d}"

        batches = [examples[i:i + args.batch_size]
                   for i in range(0, len(examples), args.batch_size)]
        for j, batch in enumerate(batches):
            name = f"{args.id_prefix}{role}-batch-{j:04d}.json"
            with open(os.path.join(args.out, name), "w") as handle:
                json.dump({"role": role, "batch": j, "examples": batch}, handle,
                          indent=1, sort_keys=True)
            manifest["batches"].append({"path": name, "role": role,
                                        "n": len(batch)})
        manifest["counts"][role] = len(examples)
        manifest["dropped"][role] = dropped

        tasks = collections.Counter(e["intent"]["task"] for e in examples)
        twists = collections.Counter(e["intent"]["twist"] for e in examples)
        formats = collections.Counter(e["intent"]["style"]["format"]
                                      for e in examples)
        cells = collections.Counter(e["cell"] for e in examples)
        manifest["coverage"][role] = {"task": dict(tasks), "twist": dict(twists),
                                      "format": dict(formats),
                                      "cells": len(cells)}
        print(f"\n{role}: {len(examples)} intents in {len(batches)} batches, "
              f"{dropped} subjects dropped for a sheet too thin")
        print("  task  :", dict(tasks.most_common()))
        print("  twist :", dict(twists.most_common()))
        print("  format:", dict(formats.most_common()))
        print(f"  cells : {len(cells)} non-empty")

        # A cell that comes out empty is either a sheet that never supports it or
        # a renderer that accepts a twist and ignores it — `fill_record` and
        # `decide` have each shipped the second, and a bank that silently loses a
        # family looks exactly like a bank that never had one. So empties are
        # always *printed*, and they fail the build only where the split is big
        # enough that chance cannot explain them: the rarest task is drawn at
        # 0.08, so 40 test-role intents miss one about one time in thirty.
        loud = len(examples) >= TASK_FLOOR
        declared = {(task, twist) for task, (_fn, _arity, allowed)
                    in TASKS.items() for twist in allowed}
        drawn = {(e["intent"]["task"], e["intent"]["twist"]) for e in examples}
        empty = sorted(declared - drawn)
        missing = [t for t in TASKS if not tasks.get(t)]
        for label, values in (("TASKS", missing), ("CELLS", empty)):
            if values:
                print(f"  EMPTY {label}: {values}")
        if loud and (missing or empty):
            raise SystemExit(
                f"{role}: {len(missing)} task(s) and {len(empty)} declared "
                f"(task, twist) cell(s) came out empty over {len(examples)} "
                f"intents, which is too many for chance — a cell the build "
                f"cannot ground is a build failure")

    with open(os.path.join(args.out, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
