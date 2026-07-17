"""Tests for sweep report aggregation (sweep/report.py)."""

import json

from sweep import report


def _write_runs(tmp_path, runs):
    with open(tmp_path / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")


# ── split_constant_columns ───────────────────────────────────────────────────
def test_constant_columns_are_split_out():
    runs = [{"sweep_run": "0000", "lr": 1e-4, "model": "llama", "f1": 0.5},
            {"sweep_run": "0001", "lr": 3e-4, "model": "llama", "f1": 0.6}]
    varying, constants = report.split_constant_columns(runs)
    assert varying == ["sweep_run", "lr", "f1"]
    assert constants == {"model": "llama"}


def test_single_run_keeps_all_columns():
    runs = [{"sweep_run": "0000", "lr": 1e-4, "f1": 0.5}]
    varying, constants = report.split_constant_columns(runs)
    assert varying == ["sweep_run", "lr", "f1"]
    assert constants == {}


def test_column_missing_from_some_runs_stays_in_table():
    runs = [{"sweep_run": "0000", "extra": 1},
            {"sweep_run": "0001", "extra": 1},
            {"sweep_run": "0002"}]
    varying, constants = report.split_constant_columns(runs)
    assert "extra" in varying
    assert constants == {}


# ── write_report ─────────────────────────────────────────────────────────────
def test_report_rows_sorted_by_sweep_run(tmp_path):
    # runs.jsonl is in completion order (parallel jobs finish shuffled)
    _write_runs(tmp_path, [{"sweep_run": "0002", "lr": 1, "f1": 0.1},
                           {"sweep_run": "0000", "lr": 2, "f1": 0.2},
                           {"sweep_run": "0001", "lr": 3, "f1": 0.3}])
    text = report.write_report(str(tmp_path))
    rows = [l for l in text.splitlines() if l.startswith("| 000")]
    assert [r.split(" | ")[0].lstrip("| ") for r in rows] == ["0000", "0001", "0002"]


def test_report_shared_section_and_narrow_table(tmp_path):
    _write_runs(tmp_path, [{"sweep_run": "0000", "model": "llama", "lr": 1, "f1": 0.1},
                           {"sweep_run": "0001", "model": "llama", "lr": 2, "f1": 0.2}])
    text = report.write_report(str(tmp_path))
    assert "- `model` = llama" in text      # constant -> shared section
    header = next(l for l in text.splitlines() if l.startswith("| sweep_run"))
    assert "model" not in header            # ... and out of the table
    assert "lr" in header and "f1" in header


def test_markdown_table_shape_and_alignment():
    rows = [{"sweep_run": "0000", "f1": 0.5, "note": "a|b"},
            {"sweep_run": "0001", "f1": 0.6, "note": "c"}]
    lines = report.format_markdown_table(rows).splitlines()
    assert lines[0] == "| sweep_run | f1 | note |"
    assert lines[1] == "| --- | ---: | --- |"      # numeric col right-aligned
    assert len(lines) == 4
    assert "a\\|b" in lines[2]                     # pipes escaped inside cells


def test_report_empty_sweep(tmp_path):
    text = report.write_report(str(tmp_path))
    assert "0 run(s)" in text and "(no runs)" in text
