"""Generalist GTLM — the continuously-trained, multi-domain graph LLM.

Orchestration only: task registry, unified example schema, mixture sampling, the
trunk/fork lifecycle, forgetting control and the evaluation suites that gate new
data. Architecture lives in ``src/models``; per-domain data pipelines live in
``src/experiments/<domain>``.

Not exported from the ``gtlm`` wheel (same treatment as ``experiments``) — run
from the repo root as ``python -m src.generalist``. See PLAN.md.
"""
