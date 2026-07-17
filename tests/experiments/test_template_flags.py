"""
Pin the template experiment's sweep contract: a resolved-config dict rendered to
CLI flags by the sweep runner must parse back into an equivalent RunConfig. This
guards against flag edits (underscore vs hyphen, store_true vs BooleanOptionalAction,
nargs vs comma-list) that would silently break `python3 -m sweep ...template ...`.
"""

from sweep.execute import render_flags
from src.experiments.template.__main__ import build_parser, config_from_args
from src.experiments.template.config import RunConfig


def _roundtrip(params):
    argv = render_flags(params)
    args = build_parser().parse_args(argv)
    return config_from_args(args)


def test_defaults_roundtrip():
    """Rendering an empty override set parses to the dataclass defaults."""
    cfg = _roundtrip({})
    assert cfg == RunConfig().validate()


def test_representative_config_roundtrips():
    params = {
        "k_hop": 2,
        "seed": 1,
        "lora": True,
        "lora_r": 8,
        "spd": True,
        "rrwp": False,
        "magnetic": True,
        "num_samples": 120,
        "num_epochs": 3,
        "lr": 1e-4,
        "bias_lr": 5e-3,
        "active_params": ["graph_bias", "embed"],
        "wandb_project": None,
    }
    cfg = _roundtrip(params)
    assert cfg.k_hop == 2 and cfg.seed == 1
    assert cfg.lora is True and cfg.lora_r == 8
    assert cfg.rrwp is False and cfg.spd is True and cfg.magnetic is True
    assert cfg.num_samples == 120 and cfg.num_epochs == 3
    assert cfg.lr == 1e-4 and cfg.bias_lr == 5e-3
    assert cfg.active_params == ("graph_bias", "embed")
    assert cfg.wandb_project is None


def test_no_lora_renders_negative_flag():
    """A False bool must render as --no-lora, not vanish (which would keep the default)."""
    assert "--no-lora" in render_flags({"lora": False})
    cfg = _roundtrip({"lora": False})
    assert cfg.lora is False and cfg.lora_config() is None
