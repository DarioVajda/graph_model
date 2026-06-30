"""
Durable per-run results logging.

Each training run (one ``impl``/``k_hop``/``seed``) and each benchmark run (one
``impl``/``k_hop``/``size``) appends a single JSON line — its hyperparameters plus
its results — to the ``results/`` directory alongside this experiment package
(``train_runs.jsonl`` / ``benchmarks.jsonl``). Appending per run (not per sweep)
means a crashed sweep still leaves every completed run on disk, and the files
accrete across invocations for later analysis.
"""

import json
import os
import time

# Anchored to this package's directory (not the CWD) so results always land in
# src/experiments/expressiveness/results/ regardless of where the run is launched.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRAIN_RESULTS_PATH = os.path.join(RESULTS_DIR, "train_runs.jsonl")
BENCH_RESULTS_PATH = os.path.join(RESULTS_DIR, "benchmarks.jsonl")


def append_jsonl(path, record):
    """Append ``record`` (a dict) as one timestamped JSON line to ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
