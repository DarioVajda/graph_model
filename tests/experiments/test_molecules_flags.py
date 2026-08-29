"""
Pin the molecules experiment's argparse to the sweep runner's flag contract, and
pin the two guardrails that are only worth anything if they are enforced in code:
the held-out task refusal and the flat-arm bias refusal.

The plumbing smoke-run item in `src/generalist/PLAN.md` §10 exists because a
silently-unwired flag (the `--magnetic-groups` class of bug) makes an entire
campaign uninterpretable. These tests are the cheap version of that check.
"""

import pytest

pytest.importorskip("rdkit")

from sweep.execute import render_flags  # noqa: E402

from src.experiments.molecules.__main__ import build_parser, config_from_args  # noqa: E402
from src.experiments.molecules.config import RunConfig  # noqa: E402
from src.experiments.molecules.data import HELD_OUT_TIER_A_TASKS  # noqa: E402
from src.experiments.molecules.tasks import assert_tier_a_wired  # noqa: E402


# Every bool flipped off its default, list flags populated, None knobs omitted.
REPRESENTATIVE = {
    "mode": "train",
    "task": "ring_size", "arm": "graph", "encoding": "terse_levi",
    "stereo_tags": False, "bias": "spd+magnetic_shared",
    "k_hop": 3, "k_hop_directed": True, "seed": 2, "held_out_eval": False,
    "question_node": "off",
    "model_name": "meta-llama/Llama-3.2-1B",
    "impl": "v2-eager", "flex_compile_mode": "default",
    "max_spd": 16, "magnetic_dim": 64, "magnetic_q": 0.5, "magnetic_m": 128,
    "pool": ["bace", "bbbp"],
    "train_size": 100, "val_size": 20, "test_size": 20, "data_seed": 7,
    "max_train_examples": 256, "max_eval_examples": 64,
    "ordering": "original",
    "len_buckets": [640, 1280], "node_buckets": None,
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
    cfg = config_from_args(build_parser().parse_args(render_flags(REPRESENTATIVE)))
    assert cfg.task == "ring_size"
    assert cfg.encoding == "terse_levi"
    assert cfg.stereo_tags is False
    assert cfg.bias_tokens() == ["spd", "magnetic_shared"]
    assert cfg.model_bias_config() == {
        "spd": True, "magnetic_shared": True,
        "max_spd": 16, "magnetic_dim": 64, "magnetic_q": 0.5,
    }


def test_every_config_field_has_a_flag():
    """No RunConfig field may be settable only in code — that is how a sweep axis
    silently does nothing."""
    import dataclasses

    parser_dests = {a.dest for a in build_parser()._actions}
    for f in dataclasses.fields(RunConfig):
        assert f.name in parser_dests, f"RunConfig.{f.name} has no CLI flag"


def test_flat_arm_refuses_a_bias():
    """A single-node graph has no structure for a bias to read (Property 2).

    Allowing it would put a bias name in the run record for a run where the bias
    provably did nothing.
    """
    with pytest.raises(ValueError, match="single-node graph"):
        RunConfig(arm="flat", bias="spd+magnetic").validate()
    RunConfig(arm="flat", bias="none").validate()      # the supported combination


def test_rejected_encoding_is_not_selectable():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--encoding", "terse_atom_only"])


def test_held_out_task_refuses_to_build_a_training_split():
    """PLAN.md §4.1's held-out declaration, enforced rather than remembered.

    Covers BOTH declarations — the Tier-A family (`bond_path`) and the Tier-B
    dataset (`clintox`). A declaration enforced for one tier and not the other is
    the failure mode this test exists to prevent.
    """
    from src.experiments.molecules.data import HELD_OUT_DATASETS
    from src.experiments.molecules.dataset import load_data

    declared = tuple(HELD_OUT_TIER_A_TASKS) + tuple(HELD_OUT_DATASETS)
    assert declared, "the held-out declaration is empty"
    for task in declared:
        cfg = RunConfig(task=task, train_size=4, val_size=2, test_size=2).validate()
        with pytest.raises(ValueError, match="permanently held out"):
            load_data(cfg)


def test_held_out_eval_flag_is_the_only_way_through():
    """The escape hatch exists, is explicit, and is off by default."""
    assert RunConfig().held_out_eval is False
    cfg = RunConfig(task="bond_path", held_out_eval=True).validate()
    assert cfg.held_out_eval is True


def test_tier_a_tasks_are_all_wired():
    assert assert_tier_a_wired()


def test_no_config_makes_pool_a_sweep_axis():
    """`pool` is a list-VALUED parameter, not a sweep axis.

    The runner cannot tell the two apart: `"pool": ["bace", "bbbp"]` silently
    expands into two runs of one corpus each rather than one run over both, and
    the run records look entirely plausible. Configs must write it as a comma
    string. Caught for real while building 003; pinned here so it cannot recur.
    """
    import glob
    import os

    from sweep.expand import load_config

    configs = glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "src", "experiments", "molecules", "configs", "*.jsonc"))
    assert configs, "no molecules sweep configs found"
    for path in configs:
        cfg = load_config(path)
        pool = cfg.get("pool")
        assert not isinstance(pool, list), (
            f"{os.path.basename(path)} sets pool={pool!r} as a list, which the "
            "sweep runner reads as an AXIS. Write it as a comma string.")


def test_every_config_selects_the_job_array_path():
    """`max_concurrent` must be SET, because it is not only a throttle.

    Setting it at all is what selects the job-array path (`--array=0-(K-1)%N`);
    omitting it submits K independent sbatch jobs instead (`sweep/execute.py:441`).
    The array gives one job id — which is what lets a mistake be fixed in place on
    every queued task at once (`scontrol update jobid=<id> ...`) rather than looped
    over K ids — plus the `array_map.tsv` needed to read results back.

    The house convention is `max_concurrent == number of runs`: array, no effective
    throttle. A value BELOW the run count is legitimate (an idle cluster, where a
    large array would claim the whole partition), so this test does not require
    equality — only that the field is present at all.
    """
    import glob
    import os

    from sweep.expand import load_config

    configs = glob.glob(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "src", "experiments", "molecules", "configs", "*.jsonc"))
    assert configs, "no molecules sweep configs found"
    for path in configs:
        sb = load_config(path).get("execution", {}).get("sbatch", {})
        if not sb:
            continue
        assert sb.get("max_concurrent"), (
            f"{os.path.basename(path)} does not set max_concurrent, so its runs go "
            "out as independent jobs rather than one array. Set it to the run count.")


def test_node_position_mode_is_unwired_here():
    """`spd_depth` is not an axis of this experiment, by decision (2026-08-29).

    It has two measurements and no positive result: kgqa E3 scored 0.6412 vs
    0.7351 for the `reset` default (−9.4 F1), and the molecules canary put it at
    0.998 vs 1.000. An unwired knob that still *accepts* a value is the worse
    failure — a sweep could set it, the run record would report it, and nothing
    would happen. So RunConfig must reject the field outright.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(RunConfig)}
    assert "node_position_mode" not in fields
    with pytest.raises(TypeError):
        RunConfig(node_position_mode="spd_depth")
    assert "--node-position-mode" not in {
        s for a in build_parser()._actions for s in a.option_strings}


def test_question_node_defaults_to_the_prefix_and_is_in_the_cache_key():
    """The question goes in the PREFIX so the graph can attend to it.

    With `question_node="off"` the question sits inside the prompt node, and every
    atom and bond node must then encode the molecule question-agnostically. That
    is the layout `probes` uses and is not the default here.

    It also changes the graph, so it must be part of the dataset cache key —
    getting that wrong silently reuses a dataset built under the other layout.
    """
    from src.experiments.molecules.dataset import dataset_path

    assert RunConfig().question_node == "on"
    assert dataset_path(RunConfig().validate()) != \
        dataset_path(RunConfig(question_node="off").validate())


def test_isolated_is_not_a_silent_synonym_for_on():
    """graphqa/kgqa spell this value "isolated"; here it is "on", and only "on".

    Accepting both would put two spellings of one arm into the run records, which
    is exactly the kind of thing that turns a later group-by into a wrong table.
    """
    with pytest.raises(ValueError, match="question_node must be one of"):
        RunConfig(question_node="isolated").validate()
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--question-node", "isolated"])


def test_default_question_node_leaves_the_cache_path_untagged():
    """The 2026-08-29 "isolated" -> "on" rename must not orphan built datasets.

    `dataset_path` tags the key only when `question_node` differs from the default,
    so a pure rename of the default leaves every existing `.gtds` path valid. If
    someone later tags the default too, this test fails and says why.
    """
    from src.experiments.molecules.dataset import dataset_path

    assert "_qon" not in dataset_path(RunConfig().validate())
    assert "_qoff" in dataset_path(RunConfig(question_node="off").validate())


def test_tier_routing_and_regression_refusal():
    """One task axis spans both tiers; the tier decides the selection metric."""
    from src.experiments.molecules.dataset import tier_of
    from src.experiments.molecules.train import TIER_METRIC

    assert tier_of("ring_membership") == "A"
    assert tier_of("bace") == "B"
    assert RunConfig(task="bace").validate().tier() == "B"

    # Selecting Tier B on exact match instead of AUROC would optimise a different
    # quantity than the one reported — the loss-vs-metric defect, again.
    assert TIER_METRIC["A"] != TIER_METRIC["B"]
    assert TIER_METRIC["B"] == "eval_roc_auc"

    # Regression sets have no yes/no readout yet, and must say so rather than
    # producing a nonsense binary label.
    with pytest.raises(ValueError, match="regression set"):
        RunConfig(task="esol").validate()


def test_spd_matrix_accepts_both_stored_shapes():
    """`shortest_path_dists` is written flat and read back nested.

    Deriving n from `len(row)` gives `sqrt(61) = 8` on a nested 61x61 row and blows
    up on reshape — and does so ONLY on the graph arm, because a single-node flat
    example has one element in either shape and looks correct. That is exactly how
    it reached the cluster: nine graph jobs failed, nine flat jobs passed.
    """
    import numpy as np

    from src.experiments.molecules.analysis import (
        as_spd_matrix, clamped_fraction, geometry_of,
    )

    m = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.int32)
    for stored in (m.tolist(), m.flatten().tolist()):        # nested and flat
        assert as_spd_matrix(stored).shape == (3, 3)
        assert geometry_of(stored)["diameter"] == 2
        assert geometry_of(stored)["n_nodes"] == 3

    # A flat-arm example: one node, one element, no distances at all.
    assert as_spd_matrix([[0]]).shape == (1, 1)
    assert geometry_of([[0]])["diameter"] == 0

    # The clamp counts pairs at or beyond max_spd; below it, nothing is folded.
    assert clamped_fraction(stored, max_spd=2) == pytest.approx(1 / 3)   # the two 2s
    assert clamped_fraction(stored, max_spd=64) == 0.0

    with pytest.raises(ValueError, match="not a square"):
        as_spd_matrix([1, 2, 3])


def test_convergence_detector_separates_interrupted_from_converged():
    """A score is only a ceiling if the curve stopped rising before the budget did.

    These are the real 004 validation curves for `fg_count` (base rate 0.760). The
    graph arm sits AT the base rate for five evals, departs, and is still climbing
    when cosine anneals the LR to min; the flat arm is flat from eval 9. Reading
    the two final numbers as a like-for-like comparison compares a converged run
    against an interrupted one.
    """
    from src.experiments.molecules.train import _convergence

    m = "eval_em_accuracy"
    graph = [0.740, 0.742, 0.742, 0.752, 0.748, 0.778,
             0.800, 0.784, 0.798, 0.816, 0.820, 0.814]
    flat = [0.742, 0.766, 0.778, 0.778, 0.808, 0.828,
            0.832, 0.840, 0.850, 0.852, 0.852, 0.852]

    assert _convergence([{m: v} for v in graph], m)["still_improving"] is True
    assert _convergence([{m: v} for v in flat], m)["still_improving"] is False
    # Too few evals to judge -> None, never a bare False that reads as "converged".
    assert _convergence([{m: 0.5}], m)["still_improving"] is None


def test_base_rate_is_recorded_so_learnability_is_checkable():
    """PLAN.md §3.2.4 criterion 3 needs the majority-class rate in the record.

    `fg_count`'s base rate is 0.760: an arm reporting 0.74 has learned nothing, yet
    reads as a respectable score beside a task whose base rate is 0.285.
    """
    from src.experiments.molecules.train import _answer_stats

    stats = _answer_stats({"answers": {" 0": 76, " 1": 20, " 2": 4}})
    assert stats["base_rate"] == pytest.approx(0.76)
    assert stats["n_classes"] == 3
    assert list(stats["answer_distribution"]) == [" 0", " 1", " 2"]   # sorted by count
    assert _answer_stats({})["base_rate"] is None


def test_margin_readout_matches_the_relbench_contract():
    """The Tier-B readout must be the same quantity relbench computes."""
    import numpy as np
    import torch

    from src.experiments.molecules.evaluate import (
        make_margin_metrics, make_margin_preprocessor, tied_pair_fraction,
    )

    yes_id, no_id = 7566, 2360           # " Yes" / " No" under the Llama-3 tokenizer
    pre = make_margin_preprocessor(yes_id, no_id)

    vocab = 8000
    logits = torch.zeros(2, 3, vocab)
    logits[0, 1, yes_id], logits[0, 1, no_id] = 3.0, 1.0      # positive, margin +2
    logits[1, 1, yes_id], logits[1, 1, no_id] = 0.5, 2.5      # negative, margin -2
    labels = torch.full((2, 3), -100)
    labels[0, 2], labels[1, 2] = yes_id, no_id

    out = pre(logits, labels).numpy()
    assert out[0, 0] - out[0, 1] == pytest.approx(2.0)
    assert out[1, 0] - out[1, 1] == pytest.approx(-2.0)

    metrics = make_margin_metrics(yes_id)(( out, ))
    assert metrics["roc_auc"] == pytest.approx(1.0)
    # sigmoid applied before thresholding: a +2 margin must land above 0.5.
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["tied_pair_fraction"] == pytest.approx(0.0)
    # ... and a fully tied split must report it rather than looking like 0.5 skill.
    assert tied_pair_fraction(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == 1.0


def test_both_arms_share_an_identical_prompt_tail():
    """The scored position must be preceded by identical text in both arms.

    The supervised token is the last token of the prompt node. If the graph arm
    ends "A:  Yes" and the flat arm ends "A: Yes", the two arms differ at the only
    position that is scored — an uncontrolled difference in the control itself.
    """
    from rdkit import Chem

    from src.experiments.molecules.config import RunConfig
    from src.experiments.molecules.dataset import build_flat_example, build_graph_example

    cfg = RunConfig(task="ring_membership").validate()
    mol = Chem.MolFromSmiles("Oc1ccccc1")
    question, answer = "Question: is atom 2 part of a ring?", " Yes"

    graph = build_graph_example(mol, question, answer, [1], cfg)
    graph_tail = graph.nodes[graph.graph["prompt_node"]]["text"]

    flat_cfg = RunConfig(task="ring_membership", arm="flat", bias="none").validate()
    flat = build_flat_example(mol, question, answer, flat_cfg)
    flat_text = flat.nodes[flat.graph["prompt_node"]]["text"]

    assert graph_tail == "\nA: Yes"
    assert flat_text.endswith(graph_tail), (
        f"arms disagree on the prompt tail:\n  graph {graph_tail!r}\n"
        f"  flat  ...{flat_text[-16:]!r}")
