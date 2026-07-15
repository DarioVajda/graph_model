"""Build naming v2: extend entities_names.json with Freebase-native aliases.

Motivation (A1 audit, results/error_analysis/audit_pipeline_losses.py,
2026-07-07): 79 test questions lose their gold inside OUR pipeline, 76 of them
because the gold mid is missing from entities_names.json — the node gets the
"unnamed entity" fallback text and is then collapsed by ``_collapse_cvts`` as a
presumed CVT mediator. Naming the mid fixes both the text and the collapse.

Sources, merged in priority order (first writer wins):
  1. the existing entities_names.json  (GNN-RAG's dump, 560k entries)
  2. ``type.object.name`` triples found inside the raw SR subgraphs themselves
     (case-preserved, Freebase-native, tiny)
  3. FB5M.name.txt — the SimpleQuestions FB5M name dump (~4M Freebase
     ``type.object.name`` entries, lowercased). Freebase-NATIVE names: on the
     61 A1 target mids it scores 59/61 with ZERO normalized mismatches vs the
     RoG gold texts. (The Freebase→Wikidata label mapping was evaluated and
     REJECTED: 54/61, 9 style mismatches — "Victoria" vs gold "Queen
     Victoria" — which normalized-substring scoring cannot absorb.)
     Download: https://raw.githubusercontent.com/castorini/BuboQA-data/master/FB5M.name.txt.bz2
     -> place at data/FB5M.name.txt.bz2 (git-ignored, like all data).

New entries are RESTRICTED to mids that actually appear in the sr-webqsp /
sr-cwq subgraphs (keeps the file small; FB5M alone would add ~4M rows).
Mids named by no source keep the "unnamed entity" fallback — for true CVT
mediators that is correct behavior.

The previous file is preserved as entities_names.v1.json (both git-ignored).

Usage (repo root, .venv):  python -m src.experiments.kgqa.build_entities_names_v2
"""

import bz2
import json
import os
import re
import shutil

from ..sr_records import CWQ_ID2MID_TABLE, SPLITS, load_sr_records, split_path

# the kgqa experiment dir (one level up from analysis/): data files live there
EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES_PATH = os.path.join(EXPERIMENT_DIR, "entities_names.json")
V1_BACKUP = os.path.join(EXPERIMENT_DIR, "entities_names.v1.json")
FB5M_BZ2 = os.path.join(EXPERIMENT_DIR, "data", "FB5M.name.txt.bz2")

_FB5M_LINE = re.compile(r'^<fb:([^>]+)>\t<fb:type\.object\.name>\t"(.*)"\t?\.$')


def iter_sr_records():
    """Normalized records of every available dataset/split (mids everywhere).

    CWQ needs the int->mid decode table on top of its raw splits; without it the
    dataset is skipped with a warning (a webqsp-only clone still reproduces —
    it then just rebuilds the webqsp-restricted dictionary).
    """
    for dataset in ("webqsp", "cwq"):
        missing = [p for s in SPLITS if not os.path.exists(p := split_path(dataset, s))]
        if dataset == "cwq" and not missing and not os.path.exists(CWQ_ID2MID_TABLE):
            missing = [CWQ_ID2MID_TABLE]
        if missing:
            print(f"[names-v2] WARN: {dataset} skipped ({missing[0]} missing)")
            continue
        for split in SPLITS:
            yield from load_sr_records(dataset, split)


def main():
    if not os.path.exists(V1_BACKUP):
        shutil.copy2(NAMES_PATH, V1_BACKUP)
        print(f"[names-v2] backed up v1 -> {V1_BACKUP}")

    # Seed from the CURRENT file (the v1 seed on a fresh clone): rebuilds are
    # append-only, so every already-assigned name — and with it the node text of
    # every existing naming-v2 cache — stays bit-identical. Only never-named
    # mids (e.g. CWQ's, once its splits + decode table are present) gain entries.
    names = json.load(open(NAMES_PATH))
    n_v1 = len(names)
    print(f"[names-v2] seed entries: {n_v1}")

    # pass 1: collect every mid in the SR subgraphs + harvest in-data name triples
    mids, harvest = set(), {}
    for rec in iter_sr_records():
        for h, r, t in rec["subgraph"]["tuples"]:
            for x in (h, t):
                if isinstance(x, str) and x.startswith(("m.", "g.")):
                    mids.add(x)
            if (r == "type.object.name" and isinstance(h, str)
                    and isinstance(t, str) and t and h not in harvest):
                harvest[h] = t
    print(f"[names-v2] SR mids: {len(mids)}; in-data name triples: {len(harvest)}")

    added_harvest = 0
    for mid, name in harvest.items():
        if mid in mids and mid not in names:
            names[mid] = name
            added_harvest += 1

    # pass 2: FB5M names for the remaining unnamed SR mids
    added_fb5m = 0
    with bz2.open(FB5M_BZ2, "rt") as f:
        for line in f:
            m = _FB5M_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            mid, name = m.group(1), m.group(2)
            if name and mid in mids and mid not in names:
                names[mid] = name
                added_fb5m += 1

    still_unnamed = sum(1 for m in mids if m not in names)
    print(f"[names-v2] added: {added_harvest} in-data + {added_fb5m} FB5M "
          f"-> {len(names)} total ({still_unnamed} SR mids stay unnamed — CVTs etc.)")

    with open(NAMES_PATH, "w") as f:
        json.dump(names, f)
    print(f"[names-v2] wrote {NAMES_PATH}")


if __name__ == "__main__":
    main()
