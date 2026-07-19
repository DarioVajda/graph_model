"""
JSONL split loading for the flat-LLM baseline.

Split out of the old package-level ``data_load.py``, whose graph half now lives in
``our_tests/data.py``. The two halves never shared anything but a dispatch flag.
"""

import json
import os


def load_text_dataset_split(path, split):
    file = os.path.join(path, f"{split}.jsonl")
    if not os.path.exists(file):
        raise ValueError(f"No dataset file found for split '{split}' at path '{file}'")

    dataset = []
    with open(file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    print(f"Loaded text dataset for split '{split}' from file '{file}' "
          f"with a total of {len(dataset)} examples.")
    return dataset


def load_dataset(path, type="text"):
    """Return ``(train, val, test)`` JSONL splits.

    ``type`` is accepted for call-site compatibility with the old shared loader; only
    ``"text"`` is meaningful here (graph loading moved to ``our_tests/data.py``).
    """
    if type != "text":
        raise ValueError(
            f"load_dataset(type={type!r}) is not supported here — this module loads the "
            f"baselines' JSONL splits. For .gtds graph datasets use "
            f"src.experiments.our_tests.data.load_data(cfg).")
    return (load_text_dataset_split(path, split="train"),
            load_text_dataset_split(path, split="val"),
            load_text_dataset_split(path, split="test"))
