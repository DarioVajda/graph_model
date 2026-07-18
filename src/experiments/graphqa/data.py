"""
Dataset construction and loading for the GraphQA experiment.

``run_data_prep_mode(cfg)`` builds this config's splits and caches them under
``cfg.dataset_dir()``; ``load_data(cfg)`` returns ``(train, val, test)`` for training.
Both are idempotent — a built split is reused, never rebuilt.

Splits follow the benchmark's own: GraphQA ships 1000 train / 500 validation / 500 test
per task, and all nine reported tasks have all three. The model trains on train only,
the checkpoint is selected on validation, and test is scored once at the end. Two
non-reported tasks (``disconnected_nodes``, ``node_classification``) ship no validation
file; those fall back to carving ``cfg.val_fraction`` off the end of train.
"""

import os

from ...utils import TextGraphDataset
from .process_dataset import build_split, has_raw_split

SPLITS = ("train", "validation", "test")


def split_path(cfg, split):
    """Where a built split lives (``TextGraphDataset`` appends the .gtds suffix)."""
    return os.path.join(cfg.dataset_dir(), split)


def _is_built(cfg, split):
    return os.path.exists(TextGraphDataset.gtds_path(split_path(cfg, split)))


def required_splits(cfg):
    """The built splits this config needs.

    Derived from the config alone (``has_official_val``), NOT from what happens to be
    on disk: the raw download is gitignored, so a filesystem-derived answer would
    silently shrink to nothing on a fresh checkout and turn a clear "run data prep"
    error into an obscure one from the loader.
    """
    return list(SPLITS) if cfg.has_official_val() else ["train", "test"]


def run_data_prep_mode(cfg):
    """Build and cache every split this config needs. Idempotent."""
    from transformers import AutoTokenizer  # deferred: keeps --help / --init fast

    print(f"[data_prep] {cfg.graph_type}/{cfg.task} -> {cfg.dataset_dir()}")
    if not cfg.uses_default_cache():
        print("[data_prep] non-default feature settings -> tagged cache dir "
              "(the untagged dir holds the datasets built with the defaults)")

    wanted = required_splits(cfg)
    missing = [s for s in wanted if not _is_built(cfg, s)]
    if not missing:
        print(f"[data_prep] already built: {wanted} — nothing to do.")
        return

    # Cross-check the config's expectation against the raw download before doing any
    # work, so a wrong TASKS_WITHOUT_OFFICIAL_VAL entry fails here with a clear
    # message rather than midway through as a missing-file traceback.
    absent = [s for s in missing if not has_raw_split(cfg.task, s)]
    if absent:
        raise FileNotFoundError(
            f"Raw GraphQA json missing for {cfg.task} split(s) {absent}. Download the "
            f"dataset first (python3 -m src.experiments.graphqa.download); if the "
            f"benchmark genuinely ships no {absent} for this task, add it to "
            f"TASKS_WITHOUT_OFFICIAL_VAL in config.py.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    os.makedirs(cfg.dataset_dir(), exist_ok=True)
    for split in missing:
        ds = build_split(cfg, split, tokenizer)
        ds.save(split_path(cfg, split))
        print(f"[data_prep] built {split}: {len(ds)} graphs -> {split_path(cfg, split)}")

    if "validation" not in wanted:
        print(f"[data_prep] note: {cfg.task} ships no validation split; training will "
              f"carve {cfg.val_fraction:.0%} off the end of train instead.")


def load_data(cfg):
    """Return ``(train, val, test)`` for this config, building nothing.

    Raises if a split is missing rather than silently building it: data prep is a
    separate (cheap, CPU-only) sweep mode, and a training job discovering it must
    build 11 tasks of features on a GPU node is a mistake worth failing loudly on.
    """
    for split in required_splits(cfg):
        if not _is_built(cfg, split):
            raise FileNotFoundError(
                f"Split {split!r} is not built for {cfg.graph_type}/{cfg.task} at "
                f"{split_path(cfg, split)}. Run data prep first:\n"
                f"  python3 -m src.experiments.graphqa --mode data_prep "
                f"--task {cfg.task} --graph-type {cfg.graph_type}")

    train = TextGraphDataset.load(split_path(cfg, "train"))
    test = TextGraphDataset.load(split_path(cfg, "test"))

    if cfg.has_official_val():
        val = TextGraphDataset.load(split_path(cfg, "validation"))
    else:
        # No official validation split for this task: carve the tail off train. The
        # raw examples are not ordered by graph size, so a tail slice is
        # distributionally comparable to the head.
        cut = int((1.0 - cfg.val_fraction) * len(train))
        train, val = train[:cut], train[cut:]

    print(f"[data] {cfg.graph_type}/{cfg.task}: "
          f"train={len(train)} val={len(val)} test={len(test)} "
          f"({'official' if cfg.has_official_val() else f'carved {cfg.val_fraction:.0%} of train'} val)")
    return train, val, test
