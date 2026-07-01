"""Tests for the generic sweep runner's config expansion (sweep/expand.py)."""

import pytest

from sweep import expand as sweep


# ── JSONC parsing ────────────────────────────────────────────────────────────
def test_loads_strips_comments_and_trailing_commas():
    cfg = sweep.loads(
        """
        {
          // a line comment
          "seeds": [0, 1],   /* block comment */
          "bias_lr": 1e-3,   // trailing comma below is allowed
        }
        """
    )
    assert cfg == {"seeds": [0, 1], "bias_lr": 1e-3}


def test_loads_preserves_comment_markers_inside_strings():
    cfg = sweep.loads('{"path": "a//b", "url": "http://x/*y*/z"}')
    assert cfg == {"path": "a//b", "url": "http://x/*y*/z"}


def test_loads_rejects_non_object():
    with pytest.raises(sweep.SweepError):
        sweep.loads("[1, 2, 3]")


# ── meta split ───────────────────────────────────────────────────────────────
def test_split_meta_separates_reserved_keys():
    meta, params = sweep.split_meta(
        {"name": "run1", "execution": {"mode": "local"}, "seeds": [0, 1]})
    assert meta == {"name": "run1", "execution": {"mode": "local"}}
    assert params == {"seeds": [0, 1]}


# ── expansion: fixed + scalar axes (cartesian) ───────────────────────────────
def test_fixed_scalar_in_every_run():
    runs = sweep.expand({"bias_lr": 1e-3, "magnetic_m": 128})
    assert runs == [{"bias_lr": 1e-3, "magnetic_m": 128}]


def test_cartesian_product_of_scalar_axes():
    runs = sweep.expand({"seeds": [0, 1], "k_hops": [0, 1], "bias_lr": 1e-3})
    assert len(runs) == 4
    assert {(r["seeds"], r["k_hops"]) for r in runs} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(r["bias_lr"] == 1e-3 for r in runs)


def test_list_of_lists_is_a_list_valued_axis():
    runs = sweep.expand({"len_buckets": [[640, 768], [512]]})
    assert sorted((r["len_buckets"] for r in runs), key=len) == [[512], [640, 768]]


# ── expansion: bundles flatten and vary together ─────────────────────────────
def test_bundle_keys_flatten_and_label_disappears():
    runs = sweep.expand({
        "size_profile": [
            {"num_nodes": 500, "batch_size": 4},
            {"num_nodes": 2000, "batch_size": 2},
        ],
    })
    assert len(runs) == 2
    assert all("size_profile" not in r for r in runs)
    assert {(r["num_nodes"], r["batch_size"]) for r in runs} == {(500, 4), (2000, 2)}


def test_bundle_params_stay_coupled_not_crossed():
    # num_nodes/batch_size must vary TOGETHER: 500<->4 and 2000<->2, never 500<->2.
    runs = sweep.expand({
        "seeds": [0, 1],
        "size_profile": [
            {"num_nodes": 500, "batch_size": 4},
            {"num_nodes": 2000, "batch_size": 2},
        ],
    })
    assert len(runs) == 4  # 2 seeds x 2 profiles
    for r in runs:
        assert (r["num_nodes"], r["batch_size"]) in {(500, 4), (2000, 2)}


def test_bundle_list_valued_member_taken_literally():
    # A list inside a bundle object is ONE value, not a nested axis.
    runs = sweep.expand({
        "size_profile": [{"num_nodes": 500, "len_buckets": [640, 768]}],
    })
    assert runs == [{"num_nodes": 500, "len_buckets": [640, 768]}]


def test_multiple_bundles_multiply():
    runs = sweep.expand({
        "size_profile": [{"num_nodes": 500}, {"num_nodes": 1000}],
        "lr_profile": [{"lr": 1e-4}, {"lr": 1e-5}],
    })
    assert len(runs) == 4
    assert {(r["num_nodes"], r["lr"]) for r in runs} == {
        (500, 1e-4), (500, 1e-5), (1000, 1e-4), (1000, 1e-5)}


# ── collisions ───────────────────────────────────────────────────────────────
def test_collision_top_level_and_bundle_raises():
    with pytest.raises(sweep.SweepError):
        sweep.expand({
            "batch_size": 8,
            "size_profile": [{"num_nodes": 500, "batch_size": 4}],
        })


def test_collision_between_two_bundles_raises():
    with pytest.raises(sweep.SweepError):
        sweep.expand({
            "a": [{"x": 1}],
            "b": [{"x": 2}],
        })


def test_mixed_bundle_and_scalars_in_one_list_raises():
    with pytest.raises(sweep.SweepError):
        sweep.expand({"bad": [{"x": 1}, 2]})


# ── arbitrary pass-through (schema-less) ─────────────────────────────────────
def test_arbitrary_keys_pass_through():
    runs = sweep.expand({"some_future_param": "hello", "nested": {"a": 1}})
    assert runs == [{"some_future_param": "hello", "nested": {"a": 1}}]
