"""
Thin loader for the KGQA (SR-WebQSP) experiment.

The heavy lifting (parsing, Levi construction, CVT collapse, naming,
verbalization, feature computation) lives in ``process_dataset.py``, which caches
each split to a config-keyed ``.gtds`` directory. This module just resolves that
directory and loads the splits, mapping ``dev`` -> eval.

Run ``python -m src.experiments.kgqa.process_dataset`` once (per config) first.
"""

import os
from types import SimpleNamespace

from ...utils import TextGraphDataset
from .process_dataset import DEFAULTS, OUTPUT_ROOT, config_key


def _resolve_dir(overrides):
    cfg = SimpleNamespace(**{**DEFAULTS, **overrides})
    return os.path.join(OUTPUT_ROOT, config_key(cfg))


def _cli_hint(overrides):
    return " ".join(f"--{k} {v}" for k, v in overrides.items())


def load_data(tokenizer=None, **overrides):
    """Return (train, eval, test) TextGraphDatasets for the given config.

    ``tokenizer`` is accepted for signature-compatibility with the other
    experiments but is unused — the cached data is already tokenized. Any
    ``overrides`` (e.g. ``rel_mode='last_2'``, ``max_nodes=256``) must match the
    config the cache was built with, since they determine the cache directory.
    """
    out_dir = _resolve_dir(overrides)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"No processed KGQA dataset at {out_dir}.\n"
            f"Build it first:  python -m src.experiments.kgqa.process_dataset "
            f"{_cli_hint(overrides)}".rstrip()
        )

    train = TextGraphDataset.load(os.path.join(out_dir, "train"))
    eval_ = TextGraphDataset.load(os.path.join(out_dir, "dev"))
    test = TextGraphDataset.load(os.path.join(out_dir, "test"))
    return train, eval_, test
