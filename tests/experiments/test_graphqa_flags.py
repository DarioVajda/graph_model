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


# ── every field must actually be forwarded, not just parsed ───────────────────

# Fields whose value cannot be perturbed generically, with the reason.
_UNPERTURBABLE = {
    "mode": "changes which entry point runs",
    "model_name": "a different backbone changes the valid impl set",
    "impl": "constrained by the backbone",
    "graph_type": "covered explicitly above",
    "task": "covered explicitly above",
    "dtype": "covered explicitly above",
    "laplacian": "unwired -> validate() rejects it (tested above)",
    "rwse": "unwired -> validate() rejects it (tested above)",
    "val_fraction": "must stay in (0, 1)",
    "max_steps": "-1 is the sentinel; +1 gives 0, a different sentinel",
}

# Overrides a field needs alongside it to stay valid.
_COMPANIONS = {
    "magnetic_linear": {"magnetic": False},
    # Each decoupled head REPLACES the magnetic placement rather than modifying
    # it, so it has to be tested with the default magnetic arm switched off.
    "magnetic_magnitude": {"magnetic": False},
    "magnetic_hybrid": {"magnetic": False},
    "magnetic_groups": {"magnetic": True},
    # spd defaults ON here, and bias_self_node refuses to combine with it (SPDBias
    # has no self-distance row, so the flag would cover only some active biases).
    "bias_self_node": {"spd": False},
}


def _perturb(name, value):
    """A value different from the default, or None if there isn't an obvious one."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 2
    if isinstance(value, str):
        return None
    if value is None:
        return "GraphLLM" if name == "wandb_project" else None
    return None


@pytest.mark.parametrize("field", [f for f in RunConfig.__dataclass_fields__])
def test_every_field_is_forwarded_from_cli(field):
    """A flag that parses but is dropped by ``config_from_args`` is invisible.

    That exact bug shipped once: --magnetic-linear and --magnetic-m-collate were
    added to the parser but not to the RunConfig(...) call, so 45 submitted jobs
    silently trained the DEFAULT arm — the linear runs came out byte-identical to
    the no-bias floor, which reads as "linearization destroys the task" rather
    than as a wiring fault. Enumerate the fields so a new knob cannot repeat it.
    """
    if field in _UNPERTURBABLE:
        pytest.skip(_UNPERTURBABLE[field])
    default = getattr(RunConfig(), field)
    alt = _perturb(field, default)
    if alt is None:
        pytest.skip(f"no generic perturbation for {field}={default!r}")
    parser_dests = {a.dest for a in build_parser()._actions}
    if field not in parser_dests:
        pytest.skip(f"{field} has no CLI flag")
    cfg = _roundtrip({field: alt, **_COMPANIONS.get(field, {})})
    assert getattr(cfg, field) == alt, (
        f"--{field.replace('_', '-')} parses but config_from_args does not forward it")
