"""Pin the relbench experiment's sweep contract and its cache identity.

Two things are guarded here, both of which fail silently rather than loudly if broken.

**The flag round-trip.** A resolved-config dict rendered to CLI flags by the sweep runner
must parse back into an equivalent ``RunConfig``. This experiment generates its parser from
the dataclass fields, so the usual drift (a knob added to the config but not the parser)
cannot happen -- but the rendering conventions still can (underscore vs hyphen,
``store_true`` vs ``BooleanOptionalAction``), and a break means
``python3 -m sweep src.experiments.relbench ...`` quietly runs the wrong configuration.

**The cache key.** ``data_config_key()`` decides which built dataset a run loads. If a
training-only knob leaks into it, every ablation arm silently rebuilds its own copy of an
identical dataset; if a construction knob leaks *out*, two different constructions collide
on one cache and the second run trains on the first one's data. Neither raises.
"""

import pytest

from sweep.execute import render_flags
from src.experiments.relbench.__main__ import build_parser, config_from_args
from src.experiments.relbench.config import RunConfig


def _roundtrip(params):
    argv = render_flags(params)
    args = build_parser().parse_args(argv)
    return config_from_args(args)


# -- the sweep contract -------------------------------------------------------

def test_defaults_roundtrip():
    assert _roundtrip({}) == RunConfig().validate()


def test_headline_grid_roundtrips():
    """The shape `005_f1_headline` actually uses: an arm axis crossed with seeds."""
    params = {
        "dataset": "rel-trial", "task": "study-outcome", "max_nodes": 24,
        "max_value_chars": 1200, "neighbor_sampling": "recent", "collapse_links": True,
        "arm_name": "graph", "spd": True, "magnetic": True, "rrwp": False,
        "seed": 43, "lr": 3e-4, "impl": "v2-flex", "dtype": "bf16",
    }
    cfg = _roundtrip(params)
    assert (cfg.dataset, cfg.task) == ("rel-trial", "study-outcome")
    assert (cfg.max_nodes, cfg.max_value_chars) == (24, 1200)
    assert cfg.arm() == "base"


def test_flat_control_roundtrips():
    params = {"arm_name": "flat", "spd": False, "magnetic": False, "rrwp": False,
              "max_nodes": 64, "seed": 44}
    cfg = _roundtrip(params)
    assert cfg.is_flat() and cfg.arm() == "flat"


def test_boolean_flags_can_be_negated():
    """`collapse_links` must be switchable off -- it is an ablation, not a constant."""
    assert _roundtrip({"collapse_links": False}).collapse_links is False
    assert _roundtrip({"collapse_links": True}).collapse_links is True


def test_none_valued_flags_survive():
    assert _roundtrip({"relation_cap": 8}).relation_cap == 8
    assert _roundtrip({}).relation_cap is None


# -- validation ---------------------------------------------------------------

def test_flat_arm_rejects_structural_bias():
    """A flat control with a bias on is not a control; it must fail before the GPU."""
    with pytest.raises(ValueError, match="flat control"):
        RunConfig(arm_name="flat", spd=True).validate()


def test_unimplemented_policies_are_rejected_not_ignored():
    """PLAN.md 5.4 names five policies; three need code that does not exist yet."""
    for policy in ("recent_plus_strided", "paper_match", "mixed"):
        with pytest.raises(ValueError, match="not implemented"):
            RunConfig(neighbor_sampling=policy).validate()


def test_planned_axes_are_rejected_not_ignored():
    with pytest.raises(ValueError, match="not implemented"):
        RunConfig(aggregates="seed").validate()
    with pytest.raises(ValueError, match="not implemented"):
        RunConfig(label_history=16).validate()


def test_subsample_knobs_are_rejected_not_ignored():
    """They were dead flags. A dead `test_subsample` is worse than a missing one: it reads
    as "the test split was subsampled" in a config someone later compares to a baseline."""
    with pytest.raises(ValueError, match="not implemented"):
        RunConfig(val_subsample=100).validate()
    with pytest.raises(ValueError, match="not implemented"):
        RunConfig(test_subsample=100).validate()


def test_unwired_bias_features_are_rejected():
    with pytest.raises(ValueError, match="never produced by data prep"):
        RunConfig(laplacian=True).validate()


# -- cache identity -----------------------------------------------------------

def test_training_knobs_do_not_change_the_cache():
    """Ablation arms must share one built dataset, or every arm pays a rebuild."""
    base = RunConfig().validate()
    for knob, value in (("seed", 99), ("lr", 1e-5), ("num_epochs", 20), ("lora_r", 64),
                        ("spd", False), ("magnetic", False), ("k_hop", 2),
                        ("batch_size", 4), ("eval_steps", 10), ("max_steps", 8)):
        variant = RunConfig(**{knob: value}).validate()
        assert variant.data_config_key() == base.data_config_key(), (
            f"{knob} changed the cache key but is a training-only knob")


@pytest.mark.parametrize("knob,value", [
    ("dataset", "rel-trial"), ("task", "driver-top3"), ("max_nodes", 128),
    ("neighbor_sampling", "uniform"), ("collapse_links", False), ("sibling_fanout", 4),
    ("text_mode", "schema_node"), ("time_encoding", "absolute"), ("anonymize", "entities"),
    ("max_value_chars", 1200), ("max_node_chars", 4000), ("null_threshold", 0.5),
    ("question_node", "off"), ("max_length", 8192), ("magnetic_q", 0.5),
    ("data_seed", 7), ("samples_per_node", 4),
    # These stride the build (data.py). Omitting them once let a 201-of-11,411 smoke cache
    # satisfy `_is_built()` for a full-scale run.
    ("max_train_samples", 200), ("max_val_samples", 100),
])
def test_construction_knobs_change_the_cache(knob, value):
    """Otherwise two different constructions collide and the second run trains on the
    first one's data -- silently, since nothing compares the cache to the request."""
    base = RunConfig().validate()
    variant = RunConfig(**{knob: value}).validate()
    assert variant.data_config_key() != base.data_config_key(), (
        f"{knob} changes the built bytes but not the cache key")


def test_rrwp_only_enters_the_key_when_enabled():
    """kgqa's convention: caches stay valid across arms that leave RRWP off."""
    assert RunConfig(rrwp=False).validate().data_config_key() == \
        RunConfig().validate().data_config_key()
    assert RunConfig(rrwp=True).validate().data_config_key() != \
        RunConfig().validate().data_config_key()


def test_bias_arms_share_one_built_dataset():
    """They differ only in which biases the model reads, and the biases are computed at
    build time for every arm. Three directories would mean three identical builds -- and
    `dataset_dir()` embedding `arm()` is exactly how that happened."""
    graph = RunConfig().validate()
    for knob in ("spd", "magnetic"):
        variant = RunConfig(**{knob: False}).validate()
        assert variant.dataset_dir() == graph.dataset_dir(), (
            f"the {variant.arm()!r} arm does not share the graph arm's cache directory")


def test_flat_and_graph_arms_get_separate_caches():
    """Same sampled rows, different serialization -- they cannot share a directory."""
    graph = RunConfig().validate()
    flat = RunConfig(arm_name="flat", spd=False, magnetic=False).validate()
    assert graph.data_config_key() != flat.data_config_key()
    assert graph.dataset_dir() != flat.dataset_dir()
