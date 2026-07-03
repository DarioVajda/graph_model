"""
Minimal JSONL append used by the KGQA training record.

Package-private (each experiment keeps its own copy so the experiments stay
independent) — same pattern as ``expressiveness/_io.py``.
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
