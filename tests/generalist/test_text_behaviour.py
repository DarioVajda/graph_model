"""`text_behaviour` — the probe set, the items it builds, and the property it rests on.

The validator's whole claim is that **adapter-off is the base model exactly** on a
text prompt, which holds only because a text prompt is a single-node graph and
every structural bias vanishes on one node. That is an architectural fact, not a
convention, so it is pinned here rather than asserted in a docstring: if a bias
ever contributes on one node, these tests fail and the control this validator
reports against has quietly stopped being a control.
"""

import json
import os

import pytest
import torch

from src.experiments.molecules.data import prompt_format
from src.generalist.evaluate import get as get_validator
from src.generalist.evaluate.text_behaviour import (
    ADAPTER_STATES, CONDITIONS, PROBES_PATH, build_items, load_probes,
    peft_model, probe_texts, register_metrics,
)

INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
BASE = "meta-llama/Llama-3.2-1B"


# ─────────────────────────────────────────────────────────────────────────────
# The property the control rests on
# ─────────────────────────────────────────────────────────────────────────────

def test_spd_bias_is_exactly_zero_on_a_single_node():
    """`SPDBias` contributes nothing when the only distance is a node to itself.

    The lookup is masked by ``(spd > 0)``, so the self-distance never reaches the
    table — which is what makes a text prompt reach the model as base weights
    plus the adapter and nothing else.
    """
    from src.models.bias import SPDBias

    class Config:
        max_spd = 32

    bias = SPDBias(num_heads=4, head_dim=8, bias_config=Config())
    torch.nn.init.normal_(bias.weights)              # a trained table, not zeros
    out = bias(dtype=torch.float32, device="cpu",
               spd=torch.zeros(1, 1, 1, dtype=torch.long))
    assert out.shape == (1, 4, 1, 1)
    assert torch.count_nonzero(out) == 0


def test_the_magnetic_diagonal_is_masked_so_one_node_carries_no_bias():
    """`finalize_node_bias` zeroes ``i == j``, which on one node is the whole matrix.

    ``bias_self_node`` defaults False and every config in the campaign leaves it
    there; the True branch is checked too, so a future run that turns it on finds
    this test naming what it costs rather than silently invalidating the control.
    """
    from src.models.bias import finalize_node_bias

    b = torch.ones(1, 1, 1, 4)                        # (B, N, N, H), one node
    assert torch.count_nonzero(finalize_node_bias(b, "cpu", False)) == 0
    assert torch.count_nonzero(finalize_node_bias(b, "cpu", True)) == 4


# ─────────────────────────────────────────────────────────────────────────────
# The probe set
# ─────────────────────────────────────────────────────────────────────────────

def test_the_shipped_probe_set_loads_and_its_ids_are_unique():
    probes = load_probes()
    ids = [p["id"] for p in probes["prompts"]]
    assert len(ids) == len(set(ids)), "a duplicate id would double-count a prompt"
    assert probes.get("system"), "the system condition has no system turn to write"
    assert len(ids) >= 24, "too few prompts for a rate to mean anything"


def test_a_probe_set_missing_a_field_is_refused_by_name(tmp_path):
    import json

    path = tmp_path / "probes.json"
    path.write_text(json.dumps({"prompts": [{"id": "a", "category": "b"}]}))
    with pytest.raises(ValueError, match="text"):
        load_probes(str(path))


def test_a_probe_set_with_no_prompts_is_refused(tmp_path):
    import json

    path = tmp_path / "probes.json"
    path.write_text(json.dumps({"prompts": []}))
    with pytest.raises(ValueError, match="no prompts"):
        load_probes(str(path))


# ─────────────────────────────────────────────────────────────────────────────
# The prompts as the run's own format writes them
# ─────────────────────────────────────────────────────────────────────────────

def test_the_chat_prompt_opens_the_assistant_turn_and_stops_there():
    """Generation has to start exactly where an answer would, and no earlier."""
    probes = load_probes()
    fmt = prompt_format(None, INSTRUCT)
    text = probe_texts(probes, fmt, "plain")[0][2]

    assert text.startswith("<|start_header_id|>user<|end_header_id|>")
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "<|eot_id|>" in text
    # One user turn and one assistant turn: a second eot would mean the assistant
    # turn was closed before the model ever wrote into it.
    assert text.count("<|eot_id|>") == 1
    assert "system" not in text


def test_the_system_condition_prepends_a_turn_no_training_example_ever_had():
    probes = load_probes()
    fmt = prompt_format(None, INSTRUCT)
    plain = dict((i, t) for i, _c, t in probe_texts(probes, fmt, "plain"))
    system = dict((i, t) for i, _c, t in probe_texts(probes, fmt, "system"))

    assert set(plain) == set(system)
    for key, text in system.items():
        assert text.startswith("<|start_header_id|>system<|end_header_id|>")
        assert text.endswith(plain[key]), (
            "the system condition must differ from `plain` by a prefix and "
            "nothing else, or the two are not comparable")
        # The stock template's date block is what the build omits for
        # reproducibility; the probe omits it for the same reason.
        assert "Cutting Knowledge" not in text


def test_a_plain_format_has_no_system_turn_to_write_and_says_so_by_being_empty():
    probes = load_probes()
    fmt = prompt_format(None, BASE)
    assert fmt.style == "plain"
    assert probe_texts(probes, fmt, "plain")
    assert probe_texts(probes, fmt, "system") == []


# ─────────────────────────────────────────────────────────────────────────────
# The items
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(INSTRUCT)
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f"instruct tokenizer not available: {exc}")


@pytest.fixture(scope="module")
def texts():
    return probe_texts(load_probes(), prompt_format(None, INSTRUCT), "plain")[:3]


def test_a_probe_item_is_a_one_node_graph_whose_only_distance_is_zero(tokenizer, texts):
    """The data-side half of the property the two bias tests pin."""
    items = build_items(texts, tokenizer, {"max_length": 512})
    for item in items:
        assert int(item["num_nodes"]) == 1
        assert int(item["prompt_node"]) == 0
        assert len(item["input_ids"]) == 1
        assert torch.as_tensor(item["shortest_path_dists"]).tolist() == [[0]]


def test_a_generation_item_supervises_nothing(tokenizer, texts):
    """Nothing to score: the answer does not exist yet, the model writes it."""
    for item in build_items(texts, tokenizer, {"max_length": 512}):
        assert {int(v) for v in item["labels"]} == {-100}


def test_the_supervised_span_is_exactly_the_continuation(tokenizer, texts):
    """The divergence pass scores the backbone's own words and not one token more.

    A span off by one would put the prompt's last token into the NLL, where it is
    both easy and identical under both adapter states — which would dilute the
    very difference the number exists to report.
    """
    answers = [" Weather is short term; climate is the long-run average.",
               " Because they lack vitamin C.",
               " It moves heat from inside to outside."]
    items = build_items(texts, tokenizer, {"max_length": 512}, answers=answers)
    for item, answer in zip(items, answers):
        span = [int(t) for t, label in zip(item["input_ids"][0], item["labels"])
                if int(label) != -100]
        assert tokenizer.decode(span) == answer


def test_build_items_does_not_depend_on_the_order_its_labels_are_computed_in(
        tokenizer, texts):
    """Reversing the inputs reverses the outputs and changes nothing else.

    `compute_labels` hands the callable one example and no index, so a label
    function keyed on call order would pass the test above and silently mislabel
    a shuffled build.
    """
    answers = [" first answer.", " second answer.", " third answer."]
    forward = build_items(texts, tokenizer, {"max_length": 512}, answers=answers)
    backward = build_items(list(reversed(texts)), tokenizer, {"max_length": 512},
                           answers=list(reversed(answers)))

    def span(item):
        return tokenizer.decode(
            [int(t) for t, label in zip(item["input_ids"][0], item["labels"])
             if int(label) != -100])

    assert [span(i) for i in forward] == answers
    assert [span(i) for i in backward] == list(reversed(answers))


# ─────────────────────────────────────────────────────────────────────────────
# The metrics and the wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_register_metrics_counts_stopping_and_collapse(tokenizer):
    # `generate` has already cut the stop token off, so a row is the answer's
    # tokens and a flag saying whether it ended or hit the cap.
    rows = [
        ([1, 2, 3], True),                          # stopped, three tokens
        ([4], True),                                # stopped, one token
        ([5, 6, 7, 8], False),                      # ran to the cap
        ([], True),                                 # stopped immediately
    ]
    out = register_metrics(rows, tokenizer)
    assert out["n"] == 4
    assert out["stop_rate"] == 0.75
    assert out["single_token_rate"] == 0.5          # the 1-token row and the empty
    assert out["new_tokens_mean"] == 2.0
    assert out["empty_rate"] >= 0.25


def test_register_metrics_on_nothing_returns_nothing(tokenizer):
    assert register_metrics([], tokenizer) == {}


@pytest.mark.parametrize("text", [
    "The molecule is a benzene derivative.",
    "It has a role as a non-polar solvent.",
    "This is functionally related to an arachidonic acid.",
    "It is a conjugate base of a phosphate.",
    "The simplest member of the class of benzenes.",
])
def test_a_chebi_caption_is_recognised_whatever_the_question_was(text):
    """The failure this exists to name: fluent, terminated, and not an answer."""
    from src.generalist.evaluate.text_behaviour import caption_shaped

    assert caption_shaped(text)


@pytest.mark.parametrize("text", [
    "The greenhouse effect is a natural process that traps heat.",
    "Chirality means a molecule cannot be superimposed on its mirror image.",
    "A catalyst lowers the activation energy of a reaction.",
    "",
])
def test_an_ordinary_answer_is_not_counted_as_a_caption(text):
    """Including answers that are *about* molecules — the detector reads style.

    A chemistry question answered in chemistry words is the probe set working as
    intended, not a collapse, and counting it would make the rate meaningless on
    the twelve chemistry prompts.
    """
    from src.generalist.evaluate.text_behaviour import caption_shaped

    assert not caption_shaped(text)


def test_every_leaf_the_validator_emits_is_declared():
    """The runner drops a validator that returns an undeclared key (D7.1).

    Both metric families feed the same key space, so the check is that their
    leaves are exactly what `keys` promises — the failure it prevents is a whole
    firing silently producing no metrics.
    """
    validator = get_validator("text_behaviour")()
    emitted = {"n", "new_tokens_mean", "chars_mean", "stop_rate",
               "single_token_rate", "empty_rate", "caption_rate",
               "predictions_path", "base_continuation_nll", "kl_mean",
               "applicable", "reason"}
    assert validator.keys() == emitted


def test_the_predictions_dump_keeps_one_row_per_prompt(tmp_path, tokenizer):
    """A mean over 48 prompts is not diagnosable without the 48 rows.

    The specific thing it settles: whether a small gap between two columns is
    every prompt shifting a little or one prompt diverging completely.
    """
    from src.generalist.evaluate.text_behaviour import write_predictions

    texts = [("a.01", "general", "Q1"), ("a.02", "explain", "Q2")]
    rows = [([9906, 1917], True), ([], False)]
    path = str(tmp_path / "nested" / "plain-on.jsonl")
    write_predictions(path, texts, rows, tokenizer)

    lines = [json.loads(line) for line in open(path)]
    assert [row["id"] for row in lines] == ["a.01", "a.02"]
    assert [row["stopped"] for row in lines] == [True, False]
    assert [row["new_tokens"] for row in lines] == [2, 0]
    assert lines[0]["prompt"] == "Q1" and lines[1]["text"] == ""


def test_a_run_without_an_adapter_declares_only_the_status_keys():
    """`keys(ctx)` narrows, because `run` narrows.

    Declaring numbers a firing will not return is the one thing the runner
    treats as a validator failure, and it drops every metric beside it.
    """
    from src.generalist.evaluate import EvalContext

    validator = get_validator("text_behaviour")()

    class Bare:
        pass

    narrow = validator.keys(EvalContext(step=1, model=Bare()))
    assert narrow == {"applicable", "reason"}
    assert narrow < validator.keys()


def test_the_divergence_pass_reports_all_three_keys_with_nothing_to_score():
    """`n` 0 says nothing was measured; an empty dict would drop the firing."""
    from src.generalist.evaluate.text_behaviour import divergence

    class NoBatches:
        def disable_adapter(self):                             # pragma: no cover
            raise AssertionError("no batch should reach the model")

    out = divergence(NoBatches(), NoBatches(), collator=None, items=[],
                     device=None, batch_tokens=1024)
    assert out == {"n": 0.0, "base_continuation_nll": 0.0, "kl_mean": 0.0}


def test_a_model_without_an_adapter_is_reported_not_crashed():
    """A run with no LoRA has no base model to compare against.

    ``applicable`` 0 with a reason is the honest answer there; raising would put
    a red status on a run that is simply outside this validator's scope.
    """
    class Bare:
        pass

    assert peft_model(Bare()) is None

    class Wrapped:
        def __init__(self, inner):
            self.module = inner

    class Adapted:
        def disable_adapter(self):
            raise AssertionError("not called here")

    inner = Adapted()
    assert peft_model(Wrapped(Wrapped(inner))) is inner


def test_the_text_validator_set_exists_and_holds_only_this_validator():
    """`eval` mode reaches the validator through this preset — see `TEXT_VALIDATORS`."""
    from src.generalist.config import TEXT_VALIDATORS, VALIDATOR_SETS

    assert VALIDATOR_SETS["text"] == TEXT_VALIDATORS
    assert [spec["name"] for spec in TEXT_VALIDATORS] == ["text_behaviour"]


def test_adding_the_text_set_left_the_spent_presets_alone():
    """The hash guard, stated as the rule rather than as six digests.

    `test_the_instruct_campaign_still_resolves_to_the_runs_that_were_measured`
    checks the consequence; this checks the cause, so a failure says *what* was
    done rather than only that something moved.
    """
    from src.generalist.config import DEFAULT_VALIDATORS, VALIDATOR_SETS

    for preset in ("default", "smoke", "shakedown", "notation", "g2s_specialist"):
        names = [spec["name"] for spec in VALIDATOR_SETS[preset]]
        assert "text_behaviour" not in names, (
            f"{preset!r} is a spent preset: `validator_specs` is inside "
            "`config_hash`, so adding to it renames every run that used it")
    assert "text_behaviour" not in [s["name"] for s in DEFAULT_VALIDATORS]


def test_the_probe_set_ships_beside_the_module_that_reads_it():
    assert os.path.exists(PROBES_PATH)
    assert CONDITIONS == ("plain", "system")
    assert ADAPTER_STATES == ("on", "off")
