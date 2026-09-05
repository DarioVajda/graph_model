"""`HostMemProbe` — the host-RAM reading, pinned so it cannot silently return zeros.

Three host-RAM OOM kills in this project (molecules PLAN.md §8.4.9, kgqa
TODO_cwq.md, kgqa README.md) were each answered by raising `--mem` rather than by a
measurement, because there was nothing that measured host memory. The probe is that
instrument, and an instrument nobody exercises is how `_per_example` went 0-for-the-
whole-campaign while looking like an absent nicety. Two things have to hold or the
readings mean nothing:

  * descendant RSS is actually summed — a parent-only number understates the
    dataloader-worker fan-out, which is the whole mechanism under suspicion;
  * everything degrades to `None`/`False` off `/proc` instead of raising, because
    this runs inside the training loop of jobs that cost GPU-hours.
"""

import json
import multiprocessing as mp
import time

import pytest

from src.experiments.expressiveness.training.instrumentation import (
    HostMemProbe, StepMemCallback, _descendant_pids, _status_kb,
)


class _State:
    """The two fields the callback reads off `TrainerState`."""

    def __init__(self, step=0, epoch=0.0):
        self.global_step = step
        self.epoch = epoch


def _touch(mb, hold):
    """Fault in ~`mb` MiB of private anonymous pages, then idle."""
    blob = bytearray(mb * 1024 * 1024)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    time.sleep(hold)


@pytest.fixture
def probe():
    p = HostMemProbe()
    if not p.available:
        pytest.skip("/proc/<pid>/status is not readable on this host")
    return p


def test_the_parent_reading_is_a_real_resident_size(probe):
    s = probe.sample()
    # A Python interpreter with pytest loaded is tens of MB; 1 MB would mean the
    # kB/GB conversion is wrong, and 100 GB would mean it is wrong the other way.
    assert 0.005 < s["self_rss_gb"] < 100.0
    assert s["self_hwm_gb"] >= s["self_rss_gb"]


def test_children_are_summed_into_the_tree_total(probe):
    """The fan-out is the point: a parent-only number misses it entirely."""
    kids = [mp.Process(target=_touch, args=(120, 8)) for _ in range(3)]
    for k in kids:
        k.start()
    try:
        # Give them time to fault the pages in; poll rather than sleep a fixed
        # amount so a loaded test machine does not turn this into a flake.
        for _ in range(80):
            s = probe.sample()
            if s["children_rss_gb"] > 0.25:
                break
            time.sleep(0.1)
        assert s["n_children"] >= 3
        assert s["children_rss_gb"] > 0.25, s
        assert s["tree_rss_gb"] == pytest.approx(s["self_rss_gb"] + s["children_rss_gb"])
        assert s["tree_rss_gb"] > s["self_rss_gb"]
    finally:
        for k in kids:
            k.join()


def test_grandchildren_count_too():
    """Workers can spawn their own helpers; the walk is recursive, not one level."""
    assert _descendant_pids(1)              # init's descendants are everything
    assert _descendant_pids(2 ** 30) == []  # a pid that cannot exist


def test_a_dead_process_reads_as_none_rather_than_raising():
    assert _status_kb(2 ** 30) is None


def test_missing_cgroup_files_are_none_not_an_exception(probe, monkeypatch):
    monkeypatch.setattr(probe, "_cgroup",
                        {k: ["/nonexistent/memory.file"] for k in ("current", "peak", "limit")})
    s = probe.sample()
    assert s["cgroup_gb"] is None and s["cgroup_peak_gb"] is None
    # The RSS half of the reading must survive a missing cgroup — the two sources
    # are independent and one going away must not take the other with it.
    assert s["self_rss_gb"] is not None


def test_the_anon_versus_page_cache_split_is_read_back(probe, tmp_path):
    """A peak made of `anon` and one made of `file` need different fixes: only the
    first is unreclaimable, and `memory.current` alone cannot tell them apart."""
    f = tmp_path / "memory.stat"
    f.write_text("anon 2147483648\nfile 1073741824\nnot_a_key 5\nfile_mapped 536870912\n")
    probe._cgroup = dict(probe._cgroup, stat=[str(f)])
    stat = probe.cgroup_stat_gb()
    assert stat["cg_anon_gb"] == 2.0
    assert stat["cg_file_gb"] == 1.0
    assert stat["cg_file_mapped_gb"] == 0.5
    assert "cg_not_a_key_gb" not in stat
    assert probe.sample()["cg_anon_gb"] == 2.0


def test_an_unlimited_cgroup_reads_as_none_not_as_the_string_max(probe, tmp_path):
    """cgroup v2 writes the literal `max` for "no limit"; `int()` would raise."""
    f = tmp_path / "memory.max"
    f.write_text("max\n")
    probe._cgroup = {"current": [str(f)], "peak": [str(f)], "limit": [str(f)]}
    assert probe.sample()["cgroup_limit_gb"] is None


# ── the callback around it ───────────────────────────────────────────────────

def test_the_trace_records_the_shape_and_the_summary_records_the_peak(tmp_path):
    """The stride must not swallow the named events: a load-time plateau and a
    late creep are only distinguishable if `data_loaded` and the evals are in the
    trace whatever the sampling interval is."""
    trace = tmp_path / "host_mem" / "run.jsonl"
    cb = StepMemCallback(trace_path=str(trace), host_every_steps=10)
    if not cb.host.available:
        pytest.skip("/proc/<pid>/status is not readable on this host")
    cb.mark("data_loaded")
    state = _State()
    cb.on_train_begin(None, state, None)
    for i in range(1, 26):
        state.global_step = i
        cb._t0 = 0.0                       # stand in for on_step_begin's clock
        cb.on_step_end(None, state, None)
    cb.on_evaluate(None, state, None)
    cb.on_train_end(None, state, None)

    lines = [json.loads(x) for x in trace.read_text().splitlines()]
    events = [l["event"] for l in lines]
    assert events[0] == "data_loaded"
    assert "train_begin" in events and "evaluate_end" in events and "train_end" in events
    # Step 1 always, then the stride: 10 and 20. The first is what pins the
    # pre-training level when a run has no `data_loaded` mark of its own.
    assert [l["step"] for l in lines if l["event"] == "step"] == [1, 10, 20]

    out = cb.summary()
    assert out["host_mem_samples"] == len(lines)
    assert out["host_mem_trace"] == str(trace)
    assert out["host_rss_self_peak_gb"] > 0
    assert out["host_rss_train_begin_gb"] > 0
    assert out["n_steps"] == 25
    # Tree >= self holds *within a sample* and nowhere else. `self_hwm_gb` is
    # `VmHWM`, a kernel high-water mark over the whole process lifetime, while
    # `tree_rss_gb` is the current sum at sample time — so a process that peaked
    # before the trace opened reports a self peak above every tree sample, which
    # is the probe working, not failing. An earlier draft asserted it across the
    # summary and passed only because a bare module run happened to peak inside
    # the trace; under the whole suite it failed at 1.079 vs 1.174 GB.
    for line in lines:
        assert line["tree_rss_gb"] >= line["self_rss_gb"]


def test_an_unwritable_trace_path_never_takes_the_run_down(tmp_path):
    """This is measurement *about* a run that has already cost GPU-hours; the same
    contract `_per_example` carries. A failed trace degrades to nulls, never to a
    crash at the last line of a six-hour job."""
    cb = StepMemCallback(trace_path=str(tmp_path / "nope" / "run.jsonl"))
    if not cb.host.available:
        pytest.skip("/proc/<pid>/status is not readable on this host")
    (tmp_path / "nope").chmod(0o500)   # the constructor already created it
    try:
        assert cb.mark("data_loaded") is not None      # sampling still works
        assert cb.summary()["host_mem_trace"] is None  # the trace does not
    finally:
        (tmp_path / "nope").chmod(0o700)


def test_it_degrades_to_nulls_where_proc_is_not_readable(tmp_path, monkeypatch):
    cb = StepMemCallback(trace_path=str(tmp_path / "run.jsonl"))
    cb.host.available = False
    assert cb.mark("data_loaded") is None
    out = cb.summary()
    assert out["host_mem_samples"] == 0
    assert out["host_rss_self_peak_gb"] is None
    assert out["host_rss_train_begin_gb"] is None
