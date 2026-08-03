"""``--magnetic-groups`` must survive the CLI -> RunConfig -> model-config path.

This exists because it silently did not. The flag was added to all three
argparse builders but never passed into the ``RunConfig(...)`` construction, so
every arm of a six-arm sweep ran the default (legacy per-layer) path and
recorded ``magnetic_groups: 0``. Nothing errored; the sweep just measured one
configuration six times.

An argparse flag with no constructor wiring is invisible to every other test in
the repo, so pin the whole chain per experiment:

    argv -> parse_args -> RunConfig -> bias_params() -> what the model builds
"""

import pytest

from src.experiments.context.__main__ import build_parser as context_parser
from src.experiments.graphqa.__main__ import build_parser as graphqa_parser
from src.experiments.kgqa.__main__ import build_parser as kgqa_parser

# (name, parser factory, minimal argv that makes the config valid)
EXPERIMENTS = [
    ("graphqa", graphqa_parser, ["--mode", "train", "--task", "node_degree"]),
    ("kgqa", kgqa_parser, ["--mode", "train", "--dataset", "webqsp"]),
    ("context", context_parser, ["--mode", "train"]),
]


def _config_for(parser_factory, argv):
    """argv -> the experiment's RunConfig, via its own parser + builder."""
    parser = parser_factory()
    args = parser.parse_args(argv)
    # Each __main__ builds its RunConfig in a module-level helper; find it the
    # same way the module itself does, by importing and calling build_config.
    return args


@pytest.mark.parametrize("name,parser_factory,base_argv", EXPERIMENTS)
@pytest.mark.parametrize("groups", [0, 1, 2, 4, 8, 16])
def test_flag_parses(name, parser_factory, base_argv, groups):
    args = _config_for(parser_factory, base_argv + ["--magnetic-groups", str(groups)])
    assert args.magnetic_groups == groups


@pytest.mark.parametrize("name,parser_factory,base_argv", EXPERIMENTS)
@pytest.mark.parametrize("groups", [0, 1, 4, 16])
def test_flag_reaches_run_config_and_model(name, parser_factory, base_argv, groups):
    """The bug: the flag parsed fine and was then dropped on the floor."""
    import importlib
    mod = importlib.import_module(f"src.experiments.{name}.__main__")
    parser = parser_factory()
    args = parser.parse_args(base_argv + ["--magnetic-groups", str(groups)])
    cfg = mod.config_from_args(args)

    assert cfg.magnetic_groups == groups, (
        f"{name}: --magnetic-groups {groups} did not reach RunConfig "
        f"(got {cfg.magnetic_groups}) — the flag is parsed but not wired into "
        "the RunConfig(...) construction.")

    bias = cfg.bias_params()
    if groups:
        assert bias.get("magnetic_groups") == groups
        assert "magnetic" not in bias, (
            "magnetic_groups must SUPERSEDE the per-layer flag model-side, or the "
            "model builds both a per-layer and a grouped magnetic bias.")
    else:
        assert bias.get("magnetic") is True
        assert "magnetic_groups" not in bias


@pytest.mark.parametrize("name,parser_factory,base_argv", EXPERIMENTS)
def test_default_is_legacy_path(name, parser_factory, base_argv):
    """Omitting the flag must leave every existing config's behaviour untouched."""
    import importlib
    mod = importlib.import_module(f"src.experiments.{name}.__main__")
    cfg = mod.config_from_args(parser_factory().parse_args(base_argv))
    assert cfg.magnetic_groups == 0
    assert cfg.bias_params().get("magnetic") is True
