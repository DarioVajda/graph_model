"""
Pin the graphqa experiment's sweep contract: a resolved-config dict rendered to CLI
flags by the sweep runner must parse back into an equivalent RunConfig. This guards
against flag edits (underscore vs hyphen, store_true vs BooleanOptionalAction, nargs
vs comma-list) that would silently break `python3 -m sweep ...graphqa ...`.
"""

import pytest

from sweep.execute import render_flags
from src.experiments.graphqa.__main__ import build_parser, config_from_args
from src.experiments.graphqa.config import RunConfig


def _roundtrip(params):
    argv = render_flags(params)
    args = build_parser().parse_args(argv)
    return config_from_args(args)


def test_defaults_roundtrip():
    """Rendering an empty override set parses to the dataclass defaults."""
    assert _roundtrip({}) == RunConfig().validate()


def test_ablation_bundle_roundtrips():
    """The shape the ablation config actually uses: a graph_type + bias-arm bundle."""
    params = {
        "task": "shortest_path", "graph_type": "incidence",
        "spd": True, "rrwp": False, "magnetic": True,
        "seed": 43, "lora": True, "lora_r": 16,
        "impl": "v2-eager", "dtype": "fp32", "k_hop": 0,
        "lr": 3e-5, "bias_lr": 5e-3, "num_epochs": 20,
        "wandb_project": None,
    }
    cfg = _roundtrip(params)
    assert cfg.task == "shortest_path" and cfg.graph_type == "incidence"
    assert cfg.spd is True and cfg.rrwp is False and cfg.magnetic is True
    assert cfg.arm() == "no-rrwp"
    assert cfg.seed == 43 and cfg.lora_r == 16
    assert cfg.impl == "v2-eager" and cfg.dtype == "fp32"
    assert cfg.lr == 3e-5 and cfg.bias_lr == 5e-3
    assert cfg.wandb_project is None


@pytest.mark.parametrize("feature", ["spd", "rrwp", "magnetic", "lora"])
def test_false_bools_render_negative_flags(feature):
    """A False bool must render as --no-x, not vanish (which would keep the default)."""
    assert f"--no-{feature}" in render_flags({feature: False})
    assert getattr(_roundtrip({feature: False}), feature) is False


def test_data_prep_mode_roundtrips():
    cfg = _roundtrip({"mode": "data_prep", "task": "node_count", "graph_type": "standard"})
    assert cfg.mode == "data_prep" and cfg.task == "node_count"


def test_unwired_features_rejected():
    """laplacian/rwse are in the schema but data prep never computes them."""
    with pytest.raises(ValueError, match="not wired"):
        _roundtrip({"laplacian": True})
    with pytest.raises(ValueError, match="not wired"):
        _roundtrip({"rwse": True})


def test_v0_impl_is_gone():
    """The legacy v0 path was removed; asking for it must fail loudly, not fall back."""
    with pytest.raises(SystemExit):
        _roundtrip({"impl": "v0-eager"})


def test_unknown_key_fails_fast():
    """An unknown config key becomes an unknown flag and must fail the run."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(render_flags({"not_a_real_knob": 1}))
