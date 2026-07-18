"""
Pin the tag_benchmarks experiment's sweep contract: a resolved-config dict rendered to
CLI flags by the sweep runner must parse back into an equivalent RunConfig. This guards
against flag edits (underscore vs hyphen, store_true vs BooleanOptionalAction) that
would silently break `python3 -m sweep ...tag_benchmarks ...`.

Also pins `dataset_dir()` against the directory names the existing on-disk cache uses:
those names predate the refactor, so a change here silently orphans built datasets
(up to 90,941 ogbn-arxiv train subgraphs) rather than failing.
"""

import os

import pytest

from sweep.execute import render_flags
from src.experiments.tag_benchmarks.__main__ import build_parser, config_from_args
from src.experiments.tag_benchmarks.config import RunConfig, DATASETS_DIR


def _roundtrip(params):
    argv = render_flags(params)
    args = build_parser().parse_args(argv)
    return config_from_args(args)


def test_defaults_roundtrip():
    """Rendering an empty override set parses to the dataclass defaults."""
    assert _roundtrip({}) == RunConfig().validate()


def test_dataset_bundle_roundtrips():
    """The shape the sweep template actually uses: a dataset bundle + a bias arm."""
    params = {
        "dataset": "pubmed", "max_neighbors": 60, "text_mapping": "target_abstract",
        "samples_per_node": 2, "spd": True, "rrwp": False, "magnetic": True,
        "seed": 43, "impl": "v2-flex", "dtype": "bf16",
    }
    cfg = _roundtrip(params)
    assert cfg.dataset == "pubmed"
    assert cfg.samples_per_node == 2
    assert (cfg.spd, cfg.rrwp, cfg.magnetic) == (True, False, True)
    assert cfg.arm() == "no-rrwp"


def test_bools_render_as_paired_flags():
    """Bias flags must use BooleanOptionalAction: the runner renders --no-x for false."""
    assert "--no-magnetic" in render_flags({"magnetic": False})
    assert "--magnetic" in render_flags({"magnetic": True})
    assert _roundtrip({"magnetic": False}).magnetic is False
    assert _roundtrip({"magnetic": True}).magnetic is True


def test_parameterized_mapping_roundtrips():
    """text_mapping_param survives the render->parse trip and reaches the dir name."""
    cfg = _roundtrip({"dataset": "cora", "text_mapping": "random_abstracts",
                      "text_mapping_param": 0.2, "max_neighbors": 20})
    assert cfg.text_mapping_param == 0.2
    assert cfg.mapping_name() == "random_abstracts_p0.2"


def test_null_renders_as_omitted():
    """A null in the config means 'unset', not the string 'None'."""
    assert render_flags({"test_subsample": None}) == []
    assert _roundtrip({"test_subsample": None}).test_subsample is None


# ── validate() draws the experiment's line ────────────────────────────────────

def test_unwired_bias_features_are_rejected():
    """laplacian/rwse were enabled by default pre-refactor but data prep never produced
    their features, making them a silent no-op. They must fail loudly now."""
    with pytest.raises(ValueError, match="not wired"):
        RunConfig(laplacian=True).validate()
    with pytest.raises(ValueError, match="not wired"):
        RunConfig(rwse=True).validate()


def test_mapping_must_match_dataset_text_attributes():
    """reddit nodes carry raw post text; the citation graphs carry (title, abstract)."""
    with pytest.raises(ValueError, match="not available for dataset"):
        RunConfig(dataset="cora", text_mapping="full_text").validate()
    with pytest.raises(ValueError, match="not available for dataset"):
        RunConfig(dataset="reddit", text_mapping="target_abstract").validate()


def test_parameterized_mappings_require_their_param():
    with pytest.raises(ValueError, match="requires --text-mapping-param"):
        RunConfig(dataset="cora", text_mapping="random_abstracts").validate()
    with pytest.raises(ValueError, match="takes no parameter"):
        RunConfig(dataset="cora", text_mapping="target_abstract",
                  text_mapping_param=0.5).validate()


def test_data_prep_requires_explicit_samples_per_node():
    """It is baked into the built dataset but absent from its directory name, so there
    is no value to infer — see config._CACHE_DEFAULTS."""
    with pytest.raises(ValueError, match="requires an explicit --samples-per-node"):
        RunConfig(mode="data_prep", samples_per_node=None).validate()
    RunConfig(mode="data_prep", samples_per_node=4).validate()   # explicit is fine


# ── cache identity ────────────────────────────────────────────────────────────

# (config kwargs) -> the directory name the pre-refactor pipeline built.
HISTORICAL_DIRS = [
    (dict(dataset="cora", max_neighbors=60, text_mapping="target_abstract"),
     "cora_hops2_neighbors60_target_abstract"),
    (dict(dataset="cora", max_neighbors=111, text_mapping="all_titles"),
     "cora_hops2_neighbors111_all_titles"),
    (dict(dataset="cora", max_neighbors=15, text_mapping="neighbor_abstracts"),
     "cora_hops2_neighbors15_neighbor_abstracts"),
    (dict(dataset="cora", max_neighbors=30, text_mapping="random_abstracts",
          text_mapping_param=0.5),
     "cora_hops2_neighbors30_random_abstracts_p0.5"),
    (dict(dataset="ogbn-arxiv", max_neighbors=20, text_mapping="random_abstracts",
          text_mapping_param=0.2),
     "ogbn-arxiv_hops2_neighbors20_random_abstracts_p0.2"),
    (dict(dataset="pubmed", max_neighbors=60, text_mapping="target_abstract"),
     "pubmed_hops2_neighbors60_target_abstract"),
    (dict(dataset="reddit", max_neighbors=15, text_mapping="full_text"),
     "reddit_hops2_neighbors15_full_text"),
    (dict(dataset="reddit", max_neighbors=30, text_mapping="truncated_text",
          text_mapping_param=128),
     "reddit_hops2_neighbors30_truncated_text_128"),
    (dict(dataset="reddit", max_neighbors=60, text_mapping="more_target_text"),
     "reddit_hops2_neighbors60_more_target_text"),
]


@pytest.mark.parametrize("kwargs,expected", HISTORICAL_DIRS)
def test_dataset_dir_matches_historical_names(kwargs, expected):
    """A default-feature config resolves to the directory the old pipeline wrote."""
    cfg = RunConfig(**kwargs).validate()
    assert cfg.dataset_dir() == os.path.join(DATASETS_DIR, expected)


def test_samples_per_node_does_not_change_the_dir():
    """It is not part of the name (the old scheme never encoded it), so it must not
    tag the path — the caches disagree on it and would all be orphaned."""
    a = RunConfig(dataset="cora", max_neighbors=60, samples_per_node=16).validate()
    b = RunConfig(dataset="cora", max_neighbors=60, samples_per_node=1).validate()
    assert a.dataset_dir() == b.dataset_dir()


def test_non_default_features_get_a_tagged_sibling():
    """Changing a feature-generation knob must not silently reuse the default cache."""
    default = RunConfig(dataset="cora", max_neighbors=60).validate()
    tweaked = RunConfig(dataset="cora", max_neighbors=60, magnetic_q=0.1).validate()
    assert default.uses_default_cache()
    assert not tweaked.uses_default_cache()
    assert tweaked.dataset_dir() != default.dataset_dir()
    assert tweaked.dataset_dir().startswith(default.dataset_dir() + "__")


def test_model_side_knobs_do_not_tag_the_cache():
    """Bias on/off, lr, seed etc. don't affect the built data, so arms share a cache."""
    base = RunConfig(dataset="cora", max_neighbors=60).validate()
    arm = RunConfig(dataset="cora", max_neighbors=60, magnetic=False, lr=1e-5,
                    seed=7, k_hop=2, max_spd=32).validate()
    assert arm.dataset_dir() == base.dataset_dir()
