"""Declare the intents, render their replies, and write the batches.

Step 3 of §9.4's order of work. Molecules and their fact sheets come from the
§3 partition, sampled stratified by source and heavy-atom count — the roles and
the label splits are the part of the pipeline that was never in question, and
`_sources_and_labels`, `_captions` and `_stratified` below are that sampling
unchanged from the build this design replaced. What is new is what happens after
the sheet: instead of drawing a style brief and handing a model a subset of facts
to write about, this declares an **intent** and **renders the reply**, and the
writer is left with nothing to be wrong about.

    src/generalist/tools/run_py.sh -m src.generalist.tools.intent_build \
        --config src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc \
        --out src/generalist/results/assistant/v5 --n-train 11000 --n-test 900

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

BATCH_SIZE = 20

#: Heavy-atom bands the pool is stratified over, alongside the source corpus.
SIZE_BUCKETS = ((0, 20), (21, 28), (29, 38), (39, 10000))

#: Below this many intents in a split, an empty coverage cell is reported but not
#: refused. The rarest task is drawn at 0.08 and the rarest twist inside it far
#: below that, so a 40-intent smoke split misses a cell often enough that failing
#: on it would only teach me to pass a flag.
TASK_FLOOR = 400


def _bucket(n: int) -> str:
    for low, high in SIZE_BUCKETS:
        if low <= n <= high:
            return f"{low}-{high}"
    return "other"


def _sources_and_labels(config, roles_wanted):
    """``(molecules, labels)`` — the pool by role with its source, and Tier-B labels.

    One pass over every corpus, because both halves come from the same records:
    `load_tier_b` gives the molecule and the corpus it came from, and
    `build_tier_b_examples` gives the labelled (molecule, endpoint) pairs the
    trunk actually trained on.
    """
    from ..adapters.molecules import (TIER_B_CORPORA, _endpoint_of_question,
                                      partition, partition_key)
    from ...experiments.molecules.data import load_tier_b
    from ...experiments.molecules.tier_b import build_tier_b_examples

    print("partition...", flush=True)
    part = partition(config)
    molecules, seen = [], set()
    for name in config.pool:
        print(f"pool {name}...", flush=True)
        records, _spec, _dropped = load_tier_b(name)
        for record in records:
            key = partition_key(record["mol"])
            if key in seen:
                continue
            seen.add(key)
            role = part.role(key)
            if role in roles_wanted:
                molecules.append({"mol": record["mol"], "key": key,
                                  "source": name, "role": role})

    labels = {}
    for corpus in TIER_B_CORPORA:
        print(f"labels {corpus}...", flush=True)
        splits, _stats = build_tier_b_examples(corpus)
        # The endpoint is recovered from the question the trunk was trained on,
        # so the fact sheet names the assay in the same words the trunk saw.
        endpoints = _endpoint_of_question(corpus)
        for split, examples in splits.items():
            for mol, question, answer in examples:
                key = partition_key(mol)
                labels.setdefault(key, []).append({
                    "corpus": corpus,
                    "endpoint": endpoints.get(question, ""),
                    "split": split,
                    "label": answer.strip().lower().startswith("yes")})
    return molecules, labels


def _captions(config) -> dict:
    """``key -> ChEBI-20 caption``, over every split."""
    from ..adapters.molecules import load_chebi

    chebi, _stats = load_chebi(config)
    out = {}
    for split, records in chebi.items():
        for record in records:
            out.setdefault(record["key"], (record["text"], split))
    return out


def _stratified(molecules, n, rng):
    """``n`` molecules, spread over (source, size bucket) in the pool's proportions.

    Largest-remainder rather than rounding each cell independently: rounding
    gives a total that misses ``n`` by however many cells there are, and the
    shortfall would land wherever the loop happened to stop.
    """
    cells = {}
    for entry in molecules:
        heavy = entry["mol"].GetNumHeavyAtoms()
        entry["heavy_atoms"] = heavy
        cells.setdefault((entry["source"], _bucket(heavy)), []).append(entry)

    total = sum(len(v) for v in cells.values())
    if total < n:
        raise SystemExit(f"the pool holds {total} molecules and {n} were asked for")
    exact = {cell: n * len(v) / total for cell, v in cells.items()}
    take = {cell: min(int(value), len(cells[cell]))
            for cell, value in exact.items()}
    short = n - sum(take.values())
    order = sorted(cells, key=lambda c: exact[c] - take[c], reverse=True)
    i = 0
    while short > 0:
        cell = order[i % len(order)]
        if take[cell] < len(cells[cell]):
            take[cell] += 1
            short -= 1
        i += 1

    out = []
    for cell, count in take.items():
        out.extend(rng.sample(cells[cell], count))
    rng.shuffle(out)
    return out


def _intent_for(entry, labels, captions, situations, rng):
    """One declared intent over one molecule, with its rendered reply."""
    from ..assistant import fact_sheet
    from ..intents import sample_intent

    tier_b = []
    for record in labels.get(entry["key"], []):
        if entry["role"] == "train" and record["split"] != "train":
            continue
        tier_b.append((record["corpus"], record["endpoint"], record["label"]))
    caption, caption_split = captions.get(entry["key"], ("", ""))
    if entry["role"] == "train" and caption_split != "train":
        caption = ""

    sheet = fact_sheet(entry["mol"], rng=rng, tier_b=tier_b, caption=caption)
    if not sheet:
        return None
    from ..assistant import canonical_smiles

    drawn = sample_intent(entry["key"], canonical_smiles(entry["mol"]), sheet,
                          situations, rng, entry["role"])
    if drawn is None:
        return None
    intent, rendered = drawn
    return {
        "key": entry["key"],
        "role": entry["role"],
        "source": entry["source"],
        "heavy_atoms": entry["heavy_atoms"],
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
    args = parser.parse_args(argv)

    from ..config import RunConfig, load_config_file
    from ..intents import load_situations

    config = RunConfig(**load_config_file(args.config, args.cell)).validate()
    adapter_config = config.adapter_config()
    rng = random.Random(args.seed)
    situations = load_situations()

    molecules, labels = _sources_and_labels(adapter_config, {"train", "test"})
    captions = _captions(adapter_config)
    by_role = {"train": [m for m in molecules if m["role"] == "train"],
               "test": [m for m in molecules if m["role"] == "test"]}
    print(f"pool: {len(by_role['train'])} train-role, {len(by_role['test'])} "
          f"test-role molecules; {len(situations)} situations", flush=True)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"seed": args.seed, "config": os.path.abspath(args.config),
                "cell": args.cell, "id_prefix": args.id_prefix,
                "situations": len(situations),
                "batches": [], "counts": {}, "dropped": {}, "coverage": {}}

    for role, wanted in (("train", args.n_train), ("test", args.n_test)):
        chosen = _stratified(by_role[role], wanted, rng)
        examples, dropped = [], 0
        for entry in chosen:
            built = _intent_for(entry, labels, captions, situations, rng)
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
              f"{dropped} molecules dropped for a sheet too thin")
        print("  task  :", dict(tasks.most_common()))
        print("  twist :", dict(twists.most_common()))
        print("  format:", dict(formats.most_common()))
        print(f"  cells : {len(cells)} non-empty")

        from ..render import TASKS

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
