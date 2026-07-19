"""
Pin the our_tests experiment's sweep contract and its question_node feature.

Two things are guarded here:

1. The sweep contract — a resolved-config dict rendered to CLI flags by the sweep
   runner must parse back into an equivalent RunConfig.
2. The question_node port (TODO.md §2). The load-bearing invariant is that the node the
   answer is generated in is NEVER empty before the answer, in either mode, and that
   `off` stays byte-identical to the historical single-prompt-node layout — otherwise
   the correctness re-run silently measures two changes instead of one.
"""

import pytest

from sweep.execute import render_flags
from src.experiments.our_tests.__main__ import build_parser, config_from_args
from src.experiments.our_tests.config import RunConfig
from src.experiments.our_tests.prompt import (ANSWER_PREFIX, GetGraphLabels, node_texts)


def _roundtrip(params):
    argv = render_flags(params)
    args = build_parser().parse_args(argv)
    return config_from_args(args)


def _expand_config(name):
    """Every run of a sweep config, as parsed RunConfigs."""
    import os
    from sweep.expand import load_config, expand

    path = os.path.join("src", "experiments", "our_tests", "configs", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    for run in expand(load_config(path)):
        yield _roundtrip({k: v for k, v in run.items()
                          if k not in ("name", "results_dir", "execution")})


# --------------------------------------------------------------------------- #
# sweep contract
# --------------------------------------------------------------------------- #

def test_defaults_roundtrip():
    """Rendering an empty override set parses to the dataclass defaults."""
    assert _roundtrip({}) == RunConfig().validate()


def test_probe_arm_roundtrips():
    """The shape the question_node probe config uses."""
    params = {
        "task": "family", "question_node": "isolated",
        "spd": True, "rrwp": True, "magnetic": True,
        "seed": 43, "lora": True, "lora_r": 32,
        "lr": 5e-5, "bias_lr": 1e-2, "num_epochs": 10,
        "wandb_project": None,
    }
    cfg = _roundtrip(params)
    assert cfg.task == "family" and cfg.question_node == "isolated"
    assert cfg.arm() == "base" and cfg.seed == 43


def test_ablation_arm_roundtrips():
    cfg = _roundtrip({"task": "kg_qa", "spd": False, "rrwp": True, "magnetic": True})
    assert cfg.arm() == "no-spd" and cfg.task == "kg_qa"


def test_bias_flags_do_not_change_the_dataset():
    """Ablation arms must share one built dataset — the bias flags are model-side."""
    base = _roundtrip({"task": "family"})
    no_spd = _roundtrip({"task": "family", "spd": False})
    assert base.dataset_dir() == no_spd.dataset_dir()


def test_question_node_forks_the_dataset_cache():
    """question_node IS a data-prep knob, so it must not reuse the `off` cache."""
    off = _roundtrip({"task": "family", "question_node": "off"})
    isolated = _roundtrip({"task": "family", "question_node": "isolated"})
    assert off.dataset_dir() != isolated.dataset_dir()
    # `off` at default settings resolves onto the historical (already-built) dataset
    assert off.dataset_dir().endswith("family_tree_graph_dataset")


def test_unwired_features_are_rejected():
    with pytest.raises(ValueError, match="not wired"):
        _roundtrip({"laplacian": True})


def test_null_question_node_is_rejected_with_a_hint():
    """"off" is the single spelling of disabled; None must not slip through as valid."""
    with pytest.raises(ValueError, match="off"):
        RunConfig(question_node=None).validate()


# --------------------------------------------------------------------------- #
# question_node: node texts
# --------------------------------------------------------------------------- #

def test_off_is_byte_identical_to_the_historical_layout():
    """The legacy prep wrote exactly f"Q: {q}\\nA: {a}" into one node."""
    q, a = "Who is Ada's mother?", "Grace Hopper"
    question_text, prompt_text = node_texts(q, a, question_node="off")
    assert question_text is None
    assert prompt_text == f"Q: {q}\nA: {a}"


def test_isolated_splits_question_out_but_keeps_the_answer_anchor():
    q, a = "Who is Ada's mother?", "Grace Hopper"
    question_text, prompt_text = node_texts(q, a, question_node="isolated")
    assert question_text == f"Q: {q}"
    # THE invariant: the target node still has a non-empty prefix before the answer.
    assert prompt_text.startswith(ANSWER_PREFIX)
    assert prompt_text == f"{ANSWER_PREFIX} {a}"


def test_isolated_never_leaves_the_target_node_empty_even_with_an_empty_answer():
    _, prompt_text = node_texts("Q?", "", question_node="isolated")
    assert prompt_text.startswith(ANSWER_PREFIX)


def test_both_modes_supervise_the_same_answer_text():
    """Only the question's visibility changes between modes, not the target span."""
    q, a = "Who?", "Ada"
    _, off_prompt = node_texts(q, a, "off")
    _, iso_prompt = node_texts(q, a, "isolated")
    assert off_prompt.endswith(f"{ANSWER_PREFIX} {a}")
    assert iso_prompt == f"{ANSWER_PREFIX} {a}"


def test_unknown_question_node_mode_raises():
    with pytest.raises(ValueError, match="Unknown question_node"):
        node_texts("q", "a", question_node="topics")


# --------------------------------------------------------------------------- #
# question_node: label masking
# --------------------------------------------------------------------------- #

# "A:" -> [32, 25] under the Llama-3.2 tokenizer; the historical code hardcoded this.
DELIM = [32, 25]


def _example(prompt_ids):
    return {"prompt_node": 0, "input_ids": [prompt_ids]}


def test_mask_supervises_only_tokens_after_the_delimiter():
    # "Q: ... A: <answer>"  ->  [1, 2, 32, 25, 7, 8]
    labels = GetGraphLabels(DELIM, occurrence="last")(_example([1, 2, 32, 25, 7, 8]))
    assert labels == [-100, -100, -100, -100, 7, 8]


def test_last_occurrence_wins_for_off():
    """A question containing "A:" must not steal the delimiter from the real one."""
    ids = [32, 25, 5, 32, 25, 9]      # "A:" appears inside the question AND as delimiter
    labels = GetGraphLabels(DELIM, occurrence="last")(_example(ids))
    assert labels == [-100, -100, -100, -100, -100, 9]


def test_first_occurrence_wins_for_isolated():
    """The isolated prompt node STARTS with the delimiter; an answer containing "A:"
    must not shift the boundary into the answer."""
    ids = [32, 25, 5, 32, 25, 9]      # "A: <answer containing A:>"
    labels = GetGraphLabels(DELIM, occurrence="first")(_example(ids))
    assert labels == [-100, -100, 5, 32, 25, 9]


def test_missing_delimiter_raises():
    with pytest.raises(ValueError, match="Could not find the answer delimiter"):
        GetGraphLabels(DELIM)(_example([1, 2, 3]))


def test_empty_answer_span_raises():
    """The delimiter with nothing after it means no supervision and no generation
    anchor — exactly the degenerate case question_node must never produce."""
    with pytest.raises(ValueError, match="answer span is empty"):
        GetGraphLabels(DELIM)(_example([1, 32, 25]))


def test_bad_occurrence_policy_raises():
    with pytest.raises(ValueError, match="occurrence"):
        GetGraphLabels(DELIM, occurrence="middle")


# --------------------------------------------------------------------------- #
# the delimiter is derived, not hardcoded
# --------------------------------------------------------------------------- #

def test_off_matches_the_prompt_text_in_the_paper_dataset():
    """The strongest form of the byte-identity check: compare against the dataset the
    published numbers were actually produced from, not a synthetic string.

    Skips when the dataset is absent — it is gitignored and large, so a fresh checkout
    legitimately lacks it.
    """
    import os
    import pickle

    from src.experiments.our_tests.config import FAMILY_DIR

    pkl = os.path.join(FAMILY_DIR, "val.gtds", "graphs.pkl")
    if not os.path.exists(pkl):
        pytest.skip(f"paper dataset not present at {pkl}")

    with open(pkl, "rb") as f:
        graphs = pickle.load(f)

    for graph in graphs[:25]:
        prompt_node = graph.graph["prompt_node"]
        text = graph.nodes[prompt_node]["text"]
        # the historical build had no separate question node ...
        assert graph.graph.get("question_node") is None
        # ... and its single prompt node is exactly what node_texts("off") emits
        question, answer = text.split("\n", 1)
        question = question[len("Q: "):]
        answer = answer[len("A: "):]
        assert node_texts(question, answer, "off") == (None, text)


def test_tokenizer_derived_delimiter_matches_the_historical_literal():
    """The legacy prep hardcoded [32, 25] for "A:". Deriving it from the tokenizer must
    reproduce that exactly, or rebuilt datasets would mask a different span."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception as exc:                        # no HF cache / no auth on this box
        pytest.skip(f"tokenizer unavailable: {exc}")
    assert tok.encode(ANSWER_PREFIX, add_special_tokens=False) == DELIM


# --------------------------------------------------------------------------- #
# per-task learning rates
# --------------------------------------------------------------------------- #

def test_paper_learning_rates_differ_per_task():
    """family and kg_qa have different published LRs; the dataclass can only default
    to one of them, so the other must be set explicitly."""
    from src.experiments.our_tests.config import PAPER_LEARNING_RATES
    assert PAPER_LEARNING_RATES["family"] == {"lr": 5e-5, "bias_lr": 1e-2}
    assert PAPER_LEARNING_RATES["kg_qa"] == {"lr": 5e-4, "bias_lr": 3e-2}


def test_defaults_are_the_family_recipe():
    assert RunConfig(task="family").validate().uses_paper_learning_rates() is True


def test_kgqa_at_default_lrs_is_flagged_as_off_recipe():
    """The actual bug this guards: a kg_qa run that forgets to override the LRs
    trains at a tenth of the published lr, and nothing in the output says so."""
    cfg = RunConfig(task="kg_qa").validate()          # defaults = family's LRs
    assert cfg.uses_paper_learning_rates() is False


def test_kgqa_with_its_own_recipe_is_accepted():
    cfg = RunConfig(task="kg_qa", lr=5e-4, bias_lr=3e-2).validate()
    assert cfg.uses_paper_learning_rates() is True


def test_ablation_config_pairs_each_task_with_its_learning_rates():
    """004 sweeps both tasks, so `task` must carry its own recipe rather than be a
    bare axis over one shared setting. The ablation inherits each task's winning
    probe settings — learning rates AND question_node — which are NOT the published
    LRs; pinning them here keeps that an explicit choice.

    family won at `off` (003: 87.9 vs 67.7), kg_qa at `isolated` (003: 70.4 vs 67.3).
    """
    seen = {}
    for cfg in _expand_config("004_ablation.jsonc"):
        seen.setdefault(cfg.task, set()).add(
            (cfg.lr, cfg.bias_lr, cfg.question_node))
    assert seen == {
        "family": {(1e-4, 1e-2, "off")},
        "kg_qa": {(1e-4, 1e-2, "isolated")},
    }


def test_ablation_covers_each_bias_exactly_once():
    """Three arms, each dropping ONE bias — no all-on cell (the probes measured it)
    and no all-off cell (not part of the study)."""
    arms = {cfg.arm() for cfg in _expand_config("004_ablation.jsonc")}
    assert arms == {"no-spd", "no-rrwp", "no-magnetic"}


def test_probe_config_pins_its_deliberate_lr_deviation():
    """003 runs family at 1e-4 instead of the paper's 5e-5 — an intentional choice,
    pinned here so it stays a choice rather than drifting into an unnoticed default.
    kg_qa keeps its published recipe, and both bias LRs are the published ones."""
    seen = {}
    for cfg in _expand_config("003_question_node.jsonc"):
        seen.setdefault(cfg.task, set()).add((cfg.lr, cfg.bias_lr))
    assert seen == {"family": {(1e-4, 1e-2)}, "kg_qa": {(5e-4, 3e-2)}}
