"""Shared pytest configuration.

Tests import `src.*` and `sweep` as top-level packages, so the repo root must be
importable regardless of the directory pytest is invoked from.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
