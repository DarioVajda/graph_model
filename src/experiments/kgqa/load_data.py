"""
Thin loader for the KGQA experiment (multi-benchmark; E2.3 of TODO_cwq).

The heavy lifting (parsing, Levi construction, CVT collapse, naming,
verbalization, feature computation) lives in ``process_dataset.py``, which caches
each (dataset × data config) to a config-keyed ``.gtds`` directory. This module
resolves those directories (``cfg.data_config_key(dataset)``) and loads:

  * train — the per-dataset train splits, concatenated (plain concat; each split
    carries its own ``versions`` augmentation as built). A single train dataset
    is returned bare, exactly as before.
  * eval/test — one dataset per entry, ``{name: TextGraphDataset}``: eval
    datasets are ALWAYS scored separately, never pooled.

Build the caches first (per data config):
    python -m src.experiments.kgqa --mode data_prep <flags>
"""

import os

from torch.utils.data import ConcatDataset

from ...utils import TextGraphDataset
from .process_dataset import OUTPUT_ROOT


def _split_path(cfg, dataset, split):
    out_dir = os.path.join(OUTPUT_ROOT, cfg.for_dataset(dataset).data_config_key(dataset))
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"No processed KGQA dataset at {out_dir}.\n"
            f"Build it first:  python -m src.experiments.kgqa --mode data_prep "
            f"--dataset {dataset} --rel-mode {cfg.rel_mode} "
            f"--max-nodes {cfg.resolved_max_nodes(dataset)} "
            f"--n-max {cfg.resolved_n_max(dataset)} "
            f"--versions {cfg.resolved_versions(dataset)} "
            f"--magnetic-m {cfg.magnetic_m} --max-spd {cfg.max_spd} "
            f"--data-seed {cfg.data_seed}"
        )
    return os.path.join(out_dir, split)


def load_data(cfg):
    """Return ``(train, eval_sets, test_sets)`` for ``cfg``'s dataset roles.

    ``train`` is one TextGraphDataset (single train dataset — the historical
    path, byte-identical) or a ``ConcatDataset`` over them (mixed training).
    ``eval_sets`` / ``test_sets`` map dataset name -> TextGraphDataset, in
    ``cfg.eval_datasets`` order.
    """
    trains = [TextGraphDataset.load(_split_path(cfg, ds, "train"))
              for ds in cfg.train_datasets]
    train = trains[0] if len(trains) == 1 else ConcatDataset(trains)
    eval_sets = {ds: TextGraphDataset.load(_split_path(cfg, ds, "dev"))
                 for ds in cfg.eval_datasets}
    test_sets = {ds: TextGraphDataset.load(_split_path(cfg, ds, "test"))
                 for ds in cfg.eval_datasets}
    return train, eval_sets, test_sets
