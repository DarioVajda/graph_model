"""
Minimal JSONL append used by the experiment's train record.

Package-private on purpose: experiments stay independent, so each one carries its own
tiny copy rather than sharing a utility (same pattern as ``graphqa`` / ``probes``).

This replaces the old ``run_metadata_graph.json`` bookkeeping, which recorded
hyperparameters but no metrics — so a run's *result* lived only in wandb or a log. One
record per run, hyperparameters and metrics together, is what makes the tables
reproducible from the results directory alone.
"""

import json
import os
import time


def append_jsonl(path, record):
    """Append ``record`` (a dict) as one timestamped JSON line to ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
