"""
Pin the probes experiment's argparse to the sweep runner's flag-rendering contract.

The runner (``sweep.execute.render_flags``) turns a resolved parameter dict into
CLI flags; the experiment's ``build_parser`` must map them 1:1 back to values.
If a future edit renames a field, drops ``BooleanOptionalAction``, or forgets a
flag, the round-trip below breaks — before a sweep silently mis-runs.

Also pins the bias-arm validation (the suite's free-form arm axis) and the
per-task node-range defaults against the README spec.
"""

import dataclasses

import pytest

from sweep.execute import render_flags
from src.experiments.probes.__main__ import build_parser, config_from_args
from src.experiments.probes.config import RunConfig, TASK_NODE_RANGES


# A representative resolved run: every bool flipped off its default, list flags,
# and None-valued knobs (rendered as omitted -> parser default None).
REPRESENTATIVE = {
    "mode": "train",
    "task": "local_hop", "bias": "spd+magnetic_shared", "k_hop": 3, "seed": 2,
    "k_hop_directed": False,
    "model_name": "meta-llama/Llama-3.2-1B",
    "impl": "v2-eager", "flex_compile_mode": "default",
    "max_spd": 16, "magnetic_dim": 64, "magnetic_q": 0.5, "magnetic_m": 128,
    "train_size": 100, "val_size": 20, "test_size": 20, "data_seed": 7,
    "min_nodes": 50, "max_nodes": 80, "ordering": "original",
    "len_buckets": [640, 1280], "node_buckets": None,
    "p_bidirectional": 0.3, "num_colors": 6, "hop_radius": 2,
    "lora": False, "lora_r": 16, "lora_dropout": 0.1,
    "lr": 2e-5, "bias_lr": 5e-3, "num_epochs": 2, "batch_size": 1,
    "accumulation_steps": 2, "eval_steps": 10, "max_steps": 4,
    "num_workers": 2, "gradient_checkpointing": True,
    "measure_density": False, "density_sample_graphs": 4, "density_sample_batches": 2,
    "wandb_project": None,
}


def test_render_flags_roundtrips_through_parser():
    args = build_parser().parse_args(render_flags(REPRESENTATIVE))
    for key, value in REPRESENTATIVE.items():
        assert getattr(args, key) == value, f"{key}: {getattr(args, key)!r} != {value!r}"


def test_roundtrip_builds_valid_runconfig():
    args = build_parser().parse_args(render_flags(REPRESENTATIVE))
    cfg = config_from_args(args)          # includes .validate()
    assert cfg.task == "local_hop"
    assert cfg.bias_tokens() == ["spd", "magnetic_shared"]
    assert cfg.needs_spd() and cfg.needs_magnetic()
    assert cfg.model_bias_config() == {
        "spd": True, "magnetic_shared": True,
        "max_spd": 16, "magnetic_dim": 64, "magnetic_q": 0.5,
    }
    assert cfg.node_range() == (50, 80)   # explicit override wins
    assert cfg.lora_config() is None      # --no-lora


def test_every_runconfig_field_has_a_flag():
    """No RunConfig knob is unreachable from the CLI (else a sweep can't set it)."""
    dests = {a.dest for a in build_parser()._actions}
    for f in dataclasses.fields(RunConfig):
        assert f.name in dests, f"RunConfig.{f.name} has no corresponding CLI flag"


def test_defaults_match_runconfig():
    """Parser defaults are pulled from RunConfig, so an omitted sweep key = the dataclass default."""
    args = build_parser().parse_args([])
    d = RunConfig()
    for f in dataclasses.fields(RunConfig):
        got, want = getattr(args, f.name), getattr(d, f.name)
        if isinstance(want, tuple):
            got = tuple(got)
        assert got == want, f"default {f.name}: {got!r} != {want!r}"


def test_task_node_ranges_follow_readme():
    """The per-task N ranges are part of the agreed spec (README)."""
    assert TASK_NODE_RANGES == {
        "direction": (100, 400),
        "substructure": (10, 50),
        "local_hop": (100, 300),
        "text_path": (25, 40),
    }
    for task, (lo, hi) in TASK_NODE_RANGES.items():
        assert RunConfig(task=task).node_range() == (lo, hi)


@pytest.mark.parametrize("bad_bias", [
    "spd+magnetic+magnetic_shared",   # magnetic variants are mutually exclusive
    "rrwp",                           # registered but not wired in this experiment
    "gibberish",                      # not in BIAS_TYPES at all
    "spd+spd",                        # duplicate token
    "",                               # empty arm (must be 'none')
])
def test_invalid_bias_arms_rejected(bad_bias):
    with pytest.raises(ValueError):
        RunConfig(bias=bad_bias).validate()


def test_valid_bias_arms_accepted():
    for arm in ("none", "spd", "magnetic", "spd+magnetic",
                "magnetic_shared", "spd+magnetic_shared"):
        cfg = RunConfig(bias=arm).validate()
        if arm == "none":
            assert cfg.bias_tokens() == [] and cfg.model_bias_config() == {}


@pytest.mark.parametrize("r", [4, 8, 16, 32])
def test_lora_alpha_is_always_twice_r(r):
    """alpha is derived as 2*r, so bumping r can't silently change the scale."""
    cfg = RunConfig(lora=True, lora_r=r).validate()
    assert cfg.lora_config()["lora_alpha"] == 2 * r
    assert RunConfig(lora=False, lora_r=r).lora_config() is None


def test_local_hop_radius_must_fit_inside_mask():
    with pytest.raises(ValueError):
        RunConfig(task="local_hop", k_hop=2, hop_radius=2).validate()
    RunConfig(task="local_hop", k_hop=3, hop_radius=2).validate()  # README setting
