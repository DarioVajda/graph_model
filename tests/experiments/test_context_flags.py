"""Pin the context experiment's sweep contract, its grid arithmetic and its cache identity.

Three things are guarded here, all of which fail silently rather than loudly if broken.

**The flag round-trip.** A resolved-config dict rendered to CLI flags by the sweep runner
must parse back into an equivalent ``RunConfig``. The grid axes are the fragile part: they
are tuples of ints rendered as comma-joined strings, so a type slip turns
``node_counts=[8, 16]`` into the string ``"8,16"`` and the grid silently becomes one cell.

**The grid arithmetic.** ``train_cells()`` decides what the model is trained on and
``cell_length()`` is the x-axis of the published figure; if either drifts, the heatmap is
annotated with lengths the model never saw.

**The cache key.** ``data_config_key()`` decides which built dataset a run loads. A
training-only knob leaking in makes every seed rebuild an identical dataset; a construction
knob leaking out makes two different constructions collide on one cache.
"""

import pytest

from sweep.execute import render_flags
from src.experiments.context.__main__ import build_parser, config_from_args
from src.experiments.context.config import MODES as MODES_TUPLE, RunConfig


def _roundtrip(params):
    argv = render_flags(params)
    args = build_parser().parse_args(argv)
    return config_from_args(args)


# ── the sweep contract ────────────────────────────────────────────────────────

def test_defaults_roundtrip():
    assert _roundtrip({}) == RunConfig().validate()


def test_grid_axes_roundtrip_as_int_tuples():
    """A list value renders to `--node-counts 8,16,32`; it must parse back to ints."""
    cfg = _roundtrip({"node_counts": [8, 16, 32], "token_counts": [32, 64]})
    assert cfg.node_counts == (8, 16, 32)
    assert cfg.token_counts == (32, 64)
    assert all(isinstance(v, int) for v in cfg.node_counts + cfg.token_counts)


def test_grid_axes_also_roundtrip_from_the_string_form():
    """The form the sweep CONFIGS must use.

    ``sweep.expand`` reads a JSON list as a sweep AXIS, so a config written as
    ``"node_counts": [8, 16]`` silently becomes two runs with one-value grids —
    each of which trains on a different dataset and reports a different cache key.
    The configs therefore pass the axes as comma-joined strings, which expand as
    scalars; this pins that they still parse into the grid.
    """
    cfg = _roundtrip({"node_counts": "8,16,32,64,128", "token_counts": "32,64,128,256,512"})
    assert cfg.node_counts == (8, 16, 32, 64, 128)
    assert cfg.token_counts == (32, 64, 128, 256, 512)
    assert cfg.data_config_key() == RunConfig().data_config_key()


SHIPPED_CONFIGS = ["001_data_prep", "002_smoke", "003_train_16k", "004_grid",
                   "005_train_32k_diag", "006_flat_zeroshot", "007_flat_train",
                   "008_chain_data", "009_chain_graph", "010_chain_flat",
                   "011_small_graph", "012_small_flat",
                   "013_chain_long_graph", "014_chain_long_flat",
                   "015_hard8k_graph", "016_hard8k_flat",
                   "017_decoy_graph", "018_decoy_flat",
                   "019_fan1_n32_graph", "020_fan1_n32_flat",
                   "021_mainsweep_graph", "022_mainsweep_flat", "023_mainsweep_grid", "024_mainsweep_calibrate",
                   "025_mainsweep_calibrate_default",
                   "026_mainsweep_graph_seed2", "027_mainsweep_flat_seed2"]


def test_every_mode_in_MODES_is_accepted_by_the_cli():
    """The parser's --mode choices must come from config.MODES, not a copy of it.

    Regression: --mode choices were a hardcoded literal, so adding "flat_grid" /
    "flat_train" to MODES left the CLI rejecting them. Both flat sweeps reached
    the cluster and died in ~45 s on argparse (jobs 119421, 119422).
    """
    from src.experiments.context.config import MODES
    for mode in MODES:
        args = build_parser().parse_args(["--mode", mode])
        assert args.mode == mode


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_shipped_config_renders_to_flags_the_cli_accepts(name):
    """Every shipped config must survive the exact path the sweep runner takes.

    expand -> render_flags -> argparse -> RunConfig.validate(). Checking the
    config parses as JSONC is not enough: the failure above was downstream of
    that, in argparse.
    """
    import os

    from sweep.expand import load_and_expand
    from sweep.execute import render_flags
    from src.experiments.context.__main__ import config_from_args

    _meta, runs = load_and_expand(
        os.path.join("src/experiments/context/configs", f"{name}.jsonc"))
    for run in runs:
        cfg = config_from_args(build_parser().parse_args(render_flags(run)))
        assert cfg.mode in MODES_TUPLE


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_shipped_configs_keep_the_grid_axes_scalar(name):
    """Guard the config files themselves against the list-vs-axis trap.

    Caught in practice: `002_smoke` with list axes submitted 4 runs instead of 1.
    """
    import os

    from sweep.expand import load_and_expand

    path = os.path.join("src/experiments/context/configs", f"{name}.jsonc")
    _meta, runs = load_and_expand(path)
    for run in runs:
        assert isinstance(run["node_counts"], str), f"{name}: node_counts expanded into an axis"
        assert isinstance(run["token_counts"], str), f"{name}: token_counts expanded into an axis"
    # Every run must still describe the SAME grid it was written with.
    assert len({r["node_counts"] for r in runs}) == 1


def test_train_sweep_shape_roundtrips():
    """The shape `003_train_16k` actually uses: seeds crossed with a fixed recipe."""
    params = {
        "mode": "train", "seed": 1, "max_train_len": 16384, "n_train": 4000,
        "lora_r": 64, "k_hop": 0, "graph_attn_impl": "flex", "dtype": "bf16",
        "spd": True, "magnetic": True, "rrwp": False, "magnetic_m": 128,
    }
    cfg = _roundtrip(params)
    assert (cfg.mode, cfg.seed, cfg.max_train_len) == ("train", 1, 16384)
    assert cfg.bias_params() == {"spd": True, "max_spd": 8, "magnetic": True,
                                 "magnetic_dim": 128, "magnetic_q": 0.25}


def test_grid_mode_roundtrips():
    cfg = _roundtrip({"mode": "grid", "checkpoint_path": "./checkpoints/context/x"})
    assert cfg.mode == "grid" and cfg.checkpoint_path.endswith("/x")


def test_boolean_flags_can_be_negated():
    """`rrwp` is off by default, so the negation of an ON default is what needs cover."""
    assert _roundtrip({"magnetic": False, "spd": True}).magnetic is False
    assert _roundtrip({"rrwp": True}).rrwp is True


# ── grid arithmetic ───────────────────────────────────────────────────────────

def test_cell_length_counts_content_nodes_only():
    """N counts QUESTION + PROMPT too, so a cell is (N-2)*T tokens of content."""
    cfg = RunConfig()
    assert cfg.cell_length(128, 512) == 126 * 512 + 64
    assert cfg.cell_length(8, 32) == 6 * 32 + 64


def test_train_cells_excludes_exactly_the_three_over_cap_cells():
    """At the locked 16k cap, 22 of 25 cells are in the training distribution."""
    cfg = RunConfig().validate()
    over = set(cfg.cells()) - set(cfg.train_cells())
    assert over == {(64, 512), (128, 256), (128, 512)}


def test_len_buckets_are_block_aligned_and_cover_every_train_cell():
    """The collator REJECTS a bucket that is not a multiple of the flex block size."""
    cfg = RunConfig().validate()
    buckets = cfg.len_buckets()
    assert all(b % 128 == 0 for b in buckets)
    for (n, t) in cfg.train_cells():
        assert any(b >= cfg.cell_length(n, t) for b in buckets)


def test_grid_len_buckets_cover_the_extrapolation_cells():
    cfg = RunConfig().validate()
    buckets = cfg.grid_len_buckets()
    assert max(buckets) >= cfg.cell_length(128, 512)
    assert all(b % 128 == 0 for b in buckets)


# ── cache identity ────────────────────────────────────────────────────────────

def test_training_knobs_stay_out_of_the_cache_key():
    base = RunConfig().data_config_key()
    for knob in ({"seed": 7}, {"lr": 1e-3}, {"num_epochs": 9}, {"batch_size": 4},
                 {"k_hop": 2}, {"lora_r": 8}, {"graph_attn_impl": "eager"}):
        assert RunConfig(**knob).data_config_key() == base, knob


def test_grid_shard_filters_stay_out_of_the_cache_key():
    """`only_cells` / `only_hops` pick among built splits; they never change bytes."""
    base = RunConfig(hop_counts=(1, 2, 3, 4)).data_config_key()
    for knob in ({"only_cells": "128x512"}, {"only_hops": "1,3"}):
        assert RunConfig(hop_counts=(1, 2, 3, 4), **knob).data_config_key() == base, knob


def test_selected_hops_filters_the_built_mixture():
    cfg = RunConfig(hop_counts=(1, 2, 3, 4))
    assert cfg.selected_hops() == (1, 2, 3, 4)
    assert RunConfig(hop_counts=(1, 2, 3, 4), only_hops="3,1").selected_hops() == (1, 3)
    assert RunConfig(hop_counts=(1, 2, 3, 4), only_hops=" 2 ").selected_hops() == (2,)


def test_selected_cells_filters_the_grid():
    cfg = RunConfig(only_cells="128x512,64x256")
    assert cfg.selected_cells() == [(64, 256), (128, 512)]


def test_only_hops_rejects_a_k_the_build_does_not_contain():
    with pytest.raises(ValueError, match="only_hops names k=5"):
        RunConfig(hop_counts=(1, 2, 3, 4), only_hops="5").validate()
    with pytest.raises(ValueError, match="not an integer"):
        RunConfig(hop_counts=(1, 2, 3, 4), only_hops="1x2").validate()


def test_shard_filters_roundtrip_through_the_sweep_runner():
    cfg = _roundtrip({"only_cells": "128x512", "only_hops": "1,3",
                      "hop_counts": [1, 2, 3, 4], "mode": "grid",
                      "checkpoint_path": "./ckpt"})
    assert cfg.only_cells == "128x512"
    assert cfg.selected_hops() == (1, 3)
    assert cfg.selected_cells() == [(128, 512)]


def test_construction_knobs_change_the_cache_key():
    base = RunConfig().data_config_key()
    for knob in ({"node_counts": (8, 16)}, {"token_counts": (32,)}, {"max_train_len": 32768},
                 {"n_train": 10}, {"n_test": 10}, {"code_len": 4}, {"data_seed": 1},
                 {"magnetic_m": 64}, {"max_spd": 16}, {"rrwp": True},
                 {"data_format_version": 99}):
        assert RunConfig(**knob).data_config_key() != base, knob


# ── validation ────────────────────────────────────────────────────────────────

def test_grid_mode_requires_a_checkpoint():
    with pytest.raises(ValueError, match="checkpoint-path"):
        RunConfig(mode="grid").validate()


def test_cap_below_the_smallest_cell_is_rejected():
    with pytest.raises(ValueError, match="admits no cell"):
        RunConfig(max_train_len=8).validate()


def test_truncated_magnetic_basis_is_rejected():
    """magnetic_m < max(N) would silently drop eigenvectors on the biggest graphs."""
    with pytest.raises(ValueError, match="truncates the eigenbasis"):
        RunConfig(magnetic_m=64).validate()
