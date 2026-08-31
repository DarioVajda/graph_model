"""Re-score Tier-A runs on molecules the model did NOT see in training (PLAN.md §3.2.10).

WHY THIS EXISTS. Tier A draws molecules **with replacement** from the ~3552-molecule
bace+bbbp pool until it has train+val+test examples, then `prepare_dataset` slices that
list positionally. Nothing makes one slice's molecules disjoint from another's. For a
*molecule-level* family the example is a deterministic function of the molecule, so a
molecule recurring across the boundary is an exact duplicate — same graph, same question,
same answer — and memorising it answers the test item. Measured duplicate rates reach
**73.6%**, and measured accuracy on the duplicate subset is 1.0000 in almost every run.

Tier B is unaffected: its scaffold split is molecule-disjoint by construction
(`tier_b.py`) and by test (`test_no_scaffold_spans_two_splits`).

NO GPU REQUIRED. `per_example/*.jsonl` already records `i` (index within the test split)
and `correct` for every test item, and generation is deterministic given `data_seed`, so
each row maps back to its molecule by replaying the draw.

VALIDATION, and it is not optional: each run's recomputed overall accuracy must reproduce
the `test_accuracy` that run recorded. A mismatch means the index mapping is wrong and
every derived number is meaningless, so such runs are reported and skipped rather than
quietly included.

    python3 -m src.experiments.molecules.duplicate_analysis 008_m2_screen_graph 009_m2_screen_flat

Note the `results/` directories carry PRE-RENUMBERING names (PLAN.md, Configs): the sweep
written by config `006_recipe` is on disk as `009_recipe`. Map them by name, not number.
"""

from __future__ import annotations

import json
import os
import random
import sys

from rdkit import Chem

from .data import load_tier_b
from .tasks import TASK_GENERATORS

#: Tier A's configured split sizes (RunConfig defaults; every sweep in §3.2 used these).
TRAIN, VAL, TEST = 4000, 500, 1000

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

_pools: dict = {}
_cache: dict = {}

#: What every §3.2 sweep sets. NOT `RunConfig.pool`'s default, which is five corpora
#: (`hiv, bace, bbbp, tox21, lipo`) — a replay against the wrong pool draws different
#: molecules and reports duplicate rates for a dataset nobody trained on, silently.
#: `rescore_sweep` therefore reads each run's recorded `pool` rather than assuming.
SWEEP_POOL = ("bace", "bbbp")


def molecule_pool(sources=SWEEP_POOL):
    sources = tuple(sources)
    if sources not in _pools:
        mols = []
        for name in sources:
            mols.extend(r["mol"] for r in load_tier_b(name)[0])
        _pools[sources] = mols
    return _pools[sources]


def replay_stream(task, data_seed=0, pool_sources=SWEEP_POOL):
    """The full generated stream, `[(smiles, question, answer), ...]`, in order.

    Replays `generate_examples` exactly, so `items[:TRAIN]` is the train split and
    `items[TRAIN + VAL:]` the test split. Exposed separately from `replay` so a test can
    compare the whole stream's answer distribution against a cached `.gtds` sidecar --
    which is what proves the replay describes the data the runs actually used.
    """
    key = ("stream", task, data_seed, tuple(pool_sources))
    if key in _cache:
        return _cache[key]

    pool = molecule_pool(pool_sources)
    generator = TASK_GENERATORS[task]
    rng = random.Random(data_seed)
    items = []
    while len(items) < TRAIN + VAL + TEST:
        mol = pool[rng.randrange(len(pool))]
        made = generator(mol, rng)
        if made is None:
            continue
        question, answer, _named = made
        items.append((Chem.MolToSmiles(mol, canonical=True), question, answer))

    _cache[key] = items
    return items


def replay(task, data_seed=0, pool_sources=SWEEP_POOL):
    """``(train_mols, train_exact, test_items)`` for one family."""
    key = (task, data_seed, tuple(pool_sources))
    if key in _cache:
        return _cache[key]

    items = replay_stream(task, data_seed, pool_sources)
    train = items[:TRAIN]
    out = ({s for s, _q, _a in train},
           {(s, q) for s, q, _a in train},
           items[TRAIN + VAL:])
    _cache[key] = out
    return out


def rescore_sweep(sweep, results_root=RESULTS):
    """Per-run overall / duplicate / novel accuracy for one sweep directory."""
    runs_path = os.path.join(results_root, sweep, "runs.jsonl")
    per_dir = os.path.join(results_root, sweep, "per_example")
    if not (os.path.exists(runs_path) and os.path.isdir(per_dir)):
        return []

    records = {}
    for line in open(runs_path):
        rec = json.loads(line)
        records[rec["run_name"]] = rec

    rows = []
    for fname in sorted(os.listdir(per_dir)):
        if not fname.endswith(".jsonl"):
            continue
        rec = records.get(fname[: -len(".jsonl")])
        if rec is None or rec.get("task") not in TASK_GENERATORS:
            continue

        per = [json.loads(l) for l in open(os.path.join(per_dir, fname))]
        if len(per) != TEST:
            continue

        overall = sum(r["correct"] for r in per) / len(per)
        recorded = rec.get("test_accuracy")
        if recorded is None or abs(overall - recorded) > 2e-3:
            print(f"  !! {rec['run_name']}: recomputed {overall:.4f} != recorded "
                  f"{recorded} — index mapping unverified, SKIPPING", file=sys.stderr)
            continue

        recorded_pool = rec.get("pool") or SWEEP_POOL
        if isinstance(recorded_pool, str):
            recorded_pool = tuple(p for p in recorded_pool.split(",") if p)
        train_mols, train_exact, test_items = replay(
            rec["task"], rec.get("data_seed", 0), tuple(recorded_pool))
        # TWO notions of novelty, and the difference matters (PLAN.md §3.2.10):
        #
        #   exact   — the (molecule, question) pair was in train. For a molecule-level
        #             family the question is fixed, so this IS molecule novelty. For an
        #             atom-level family the named atom varies, so excluding only exact
        #             duplicates still leaves molecules the model trained on under a
        #             different question. That is the WEAKER measure.
        #   unseen  — the molecule never appeared in train under ANY question. The
        #             strict measure, and the one to quote for a generalisation claim.
        dup, novel, unseen = [], [], []
        for r in per:
            smiles, question, _answer = test_items[r["i"]]
            (dup if (smiles, question) in train_exact else novel).append(r["correct"])
            if smiles not in train_mols:
                unseen.append(r["correct"])
        seen = 1 - len(unseen) / len(per)

        rows.append({
            "run": rec["run_name"], "sweep": sweep, "task": rec["task"],
            "arm": rec.get("arm"), "bias": rec.get("bias"), "seed": rec.get("seed"),
            "stereo_tags": rec.get("stereo_tags"),
            "overall": overall,
            "exact_dup_frac": len(dup) / len(per),
            "mol_seen_frac": seen,
            "acc_dup": (sum(dup) / len(dup)) if dup else None,
            "acc_novel": (sum(novel) / len(novel)) if novel else None,
            "n_novel": len(novel),
            "acc_unseen_mol": (sum(unseen) / len(unseen)) if unseen else None,
            "n_unseen_mol": len(unseen),
        })
    return rows


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2

    rows = []
    for sweep in argv:
        rows.extend(rescore_sweep(sweep))

    header = (f"{'task':20s}{'arm':6s}{'bias':14s}{'st':4s}{'sd':4s}"
              f"{'overall':>8s}{'exact%':>8s}{'novel':>8s}{'n':>6s}"
              f"{'UNSEEN':>8s}{'n':>6s}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["task"], r["arm"] or "", str(r["bias"]),
                                         str(r["seed"]))):
        fmt = lambda v: "   n/a" if v is None else f"{v:.4f}"       # noqa: E731
        print(f"{r['task']:20s}{r['arm'] or '':6s}{str(r['bias']):14s}"
              f"{str(r['stereo_tags'])[:4]:4s}{str(r['seed']):4s}"
              f"{r['overall']:>8.4f}{r['exact_dup_frac']:>8.1%}"
              f"{fmt(r['acc_novel']):>8s}{r['n_novel']:>6d}"
              f"{fmt(r['acc_unseen_mol']):>8s}{r['n_unseen_mol']:>6d}")
    print("\n`novel` excludes exact (molecule, question) duplicates only; `UNSEEN` excludes"
          "\nevery molecule that appeared in train under any question -- quote UNSEEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
