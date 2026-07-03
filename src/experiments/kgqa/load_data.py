"""
Thin loader for the KGQA (SR-WebQSP) experiment.

The heavy lifting (parsing, Levi construction, CVT collapse, naming,
verbalization, feature computation) lives in ``process_dataset.py``, which caches
each split to a config-keyed ``.gtds`` directory. This module just resolves that
directory (from the ``RunConfig``) and loads the splits, mapping ``dev`` -> eval.

Build the cache first (per data config):
    python -m src.experiments.kgqa --mode data_prep <flags>
"""

import os

from ...utils import TextGraphDataset
from .process_dataset import OUTPUT_ROOT


def load_data(cfg):
    """Return (train, eval, test) TextGraphDatasets for ``cfg``'s data config.

    The data-affecting fields of ``cfg`` (rel_mode, max_nodes, magnetic_m, …)
    resolve the cache directory via ``cfg.data_config_key()``, so a run always
    loads exactly the dataset its config describes.
    """
    out_dir = os.path.join(OUTPUT_ROOT, cfg.data_config_key())
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"No processed KGQA dataset at {out_dir}.\n"
            f"Build it first:  python -m src.experiments.kgqa --mode data_prep "
            f"--rel-mode {cfg.rel_mode} --max-nodes {cfg.max_nodes} --n-max {cfg.n_max} "
            f"--versions {cfg.versions} --magnetic-m {cfg.magnetic_m} --max-spd {cfg.max_spd} "
            f"--data-seed {cfg.data_seed}"
        )

    train = TextGraphDataset.load(os.path.join(out_dir, "train"))
    eval_ = TextGraphDataset.load(os.path.join(out_dir, "dev"))
    test = TextGraphDataset.load(os.path.join(out_dir, "test"))
    return train, eval_, test
