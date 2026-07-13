"""
Pin the KGQA experiment's argparse to the sweep runner's flag-rendering contract.

The runner (``sweep.execute.render_flags``) turns a resolved parameter dict into
CLI flags; the experiment's ``build_parser`` must map them 1:1 back to values.
If a future edit renames a field, drops ``BooleanOptionalAction``, or forgets a
flag, the round-trip below breaks — before a sweep silently mis-runs.
"""

import dataclasses

import pytest

from sweep.execute import render_flags
from src.experiments.kgqa.__main__ import build_parser, config_from_args
from src.experiments.kgqa.config import RunConfig


# A representative resolved run: every bool flipped off its default, a multi-value
# list flag, and None-valued knobs (rendered as omitted -> parser default None).
REPRESENTATIVE = {
    "mode": "train",
    # dataset roles + per-dataset knobs (E2): lists round-trip comma-joined,
    # dict-valued knobs as "k=v" pairs.
    "train_datasets": ["webqsp", "cwq"], "eval_datasets": ["webqsp", "cwq"],
    "selection_dataset": "cwq", "log_strict_metrics": True,
    "rel_mode": "last_2", "max_nodes": {"webqsp": 256, "cwq": 1024},
    "n_max": 15, "versions": {"webqsp": 4, "cwq": 1},
    "max_length": 512, "rcm": False, "data_seed": 7, "use_gpu": False,
    "analyse_dataset": True,
    "model_name": "meta-llama/Llama-3.2-1B",
    "spd": True, "max_spd": 32, "magnetic": False, "magnetic_dim": 64,
    "magnetic_q": 0.5, "magnetic_m": 0,
    "num_epochs": 2, "batch_size": 1, "accumulation_steps": 2,
    "lr": 1e-4, "bias_lr": 2e-3, "eval_steps": 50, "max_steps": 4, "seed": 3,
    "lora_r": 8, "k_hop": 0, "k_hop_directed": True,
    "bias_weight_decay": 0.1, "lora_dropout": 0.1,
    "graph_attn_impl": "eager", "dtype": "fp32", "gradient_checkpointing": False,
    "active_params": ["graph_bias", "embeddings"], "num_workers": 2,
    "gen_max_new_tokens": 64, "gen_max_samples": None, "gen_eval_samples": 32,
    "wandb_project": None,
}


def test_render_flags_roundtrips_through_parser():
    args = build_parser().parse_args(render_flags(REPRESENTATIVE))
    for key, value in REPRESENTATIVE.items():
        assert getattr(args, key) == value, f"{key}: {getattr(args, key)!r} != {value!r}"


def test_roundtrip_builds_valid_runconfig():
    args = build_parser().parse_args(render_flags(REPRESENTATIVE))
    cfg = config_from_args(args)          # includes .validate()
    assert cfg.rel_mode == "last_2"
    assert cfg.k_hop_directed is True
    assert cfg.gradient_checkpointing is False
    assert cfg.active_params == ("graph_bias", "embeddings")
    assert cfg.gen_max_samples is None


def test_every_runconfig_field_has_a_flag():
    """No RunConfig knob is unreachable from the CLI (else a sweep can't set it)."""
    dests = {a.dest for a in build_parser()._actions}
    for f in dataclasses.fields(RunConfig):
        assert f.name in dests, f"RunConfig.{f.name} has no corresponding CLI flag"


@pytest.mark.parametrize("bad", [
    {"bias_weight_decay": -0.1},
    {"lora_dropout": 1.0},
])
def test_validate_rejects_bad_regularization_values(bad):
    with pytest.raises(ValueError):
        RunConfig(**bad).validate()


def test_defaults_match_runconfig():
    """Parser defaults are pulled from RunConfig, so an omitted sweep key = the dataclass default."""
    args = build_parser().parse_args([])
    d = RunConfig()
    for f in dataclasses.fields(RunConfig):
        got, want = getattr(args, f.name), getattr(d, f.name)
        # active_params default is a list on the parser vs tuple on the dataclass.
        if isinstance(want, tuple):
            got = tuple(got)
        assert got == want, f"default {f.name}: {got!r} != {want!r}"
