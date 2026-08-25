"""Add the `landmark` column to EXISTING `.gtds` caches, in place.

    python -m src.experiments.bias_experiments.landmark.add_landmark_column \
        --roots <gtds_parent_dir> [--k 32] [--d_max 8]

Why in place rather than a dataset rebuild: the WebQSP cache is 3.5 GB and its
directory name IS `data_config_key`, so adding a field to the builder would fork
the cache and re-run retrieval, tokenization, SPD and the magnetic eigendecomp —
hours of work to obtain a column that costs O(k(N+E)) per graph. The landmark
coordinates are a pure function of the stored topology (`graphs.pkl`), so they can
be appended to the Arrow table of a cache that already exists. Runs that do not
enable `landmark` never read the column.

Idempotent: an existing `landmark` column is dropped and rewritten, so re-running
with a different `--k` upgrades a cache rather than corrupting it. Because anchors
are stored round-robin across components, a prefix slice at collate serves every
k' <= k, so the built value only needs to be the LARGEST k the sweep uses.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from tqdm import tqdm

from src.utils.landmark import landmark_coords
from src.utils.text_graph_dataset import TextGraphDataset


def augment(path: str, k: int, d_max: int, force: bool) -> str:
    ds = TextGraphDataset.load(path)
    cols = ds._hf_dataset.column_names
    meta = {}
    meta_path = os.path.join(TextGraphDataset.gtds_path(path), "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    if "landmark" in cols and not force:
        if meta.get("landmark_k") == k and meta.get("landmark_d_max") == d_max:
            return f"skip (already k={k}, d_max={d_max})"

    t0 = time.time()
    rows = []
    for g in tqdm(ds.graphs, desc=os.path.basename(path)):
        n = g.number_of_nodes()
        pos = {u: i for i, u in enumerate(g.nodes())}
        edges = [(pos[u], pos[v]) for u, v in g.edges()]
        rows.append(landmark_coords(edges, n, k=k, d_max=d_max).reshape(-1).tolist())

    if "landmark" in cols:
        ds._hf_dataset = ds._hf_dataset.remove_columns("landmark")
    ds._hf_dataset = ds._hf_dataset.add_column("landmark", rows)
    ds.save(path)

    # save() rewrites metadata.json from scratch, so the landmark keys go in
    # AFTER it — reading the file it just wrote, not the one from before.
    meta_path = os.path.join(TextGraphDataset.gtds_path(path), "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update(landmark_k=k, landmark_d_max=d_max)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return f"ok ({len(rows)} graphs, {time.time()-t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="directories holding *.gtds (searched one level deep too)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--d-max", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="split basenames to process (e.g. dev test train)")
    args = ap.parse_args()

    targets = []
    for root in args.roots:
        targets += sorted(glob.glob(os.path.join(root, "*.gtds")))
        targets += sorted(glob.glob(os.path.join(root, "*", "*.gtds")))
    if args.only:
        keep = set(args.only)
        targets = [t for t in targets
                   if os.path.basename(t)[:-len(".gtds")] in keep]
    if not targets:
        raise SystemExit(f"no .gtds found under {args.roots}")

    print(f"{len(targets)} splits to augment (k={args.k}, d_max={args.d_max})\n")
    for t in targets:
        base = t[:-len(".gtds")] if t.endswith(".gtds") else t
        print(f"[{os.path.relpath(t)}] {augment(base, args.k, args.d_max, args.force)}",
              flush=True)


if __name__ == "__main__":
    main()
