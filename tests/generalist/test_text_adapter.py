"""
The text adapter (`src/generalist/adapters/text.py`) and what wiring it in must
not move.

Three kinds of claim:

* **Pass semantics.** Pass *p* serves answer ``(p · w + j) mod m``, so nothing
  repeats inside ``m / w`` passes and a repeat, when a run does draw one, is
  counted. Unusable answers (unterminated, empty, a special token) never become
  targets.
* **The built item.** A single node holding the user turn and the assistant
  turn, the supervised span ending in ``<|eot_id|>``, and an example that would
  exceed ``max_length`` dropped and counted rather than truncated.
* **Molecules-only runs are untouched.** No ``text/`` task in their registry and
  no text field in their hash; `test_cli.py` pins the `008` digests themselves.

The replay directory is a hand-written fixture of four prompts, which is enough
to see every rotation by eye.
"""

import json
import os

import pytest

from src.generalist.adapters import AdapterError, adapter_for
from src.generalist.adapters import text as T
from src.generalist.config import (
    MIXTURES, REPLAY_SHARE, REPLAY_SHARE_LOW, RunConfig,
    molecule_generalist_mixture,
)
from src.generalist.registry import TEXT_PREFIX

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

#: id -> (role, weight, answers). ``None`` marks an unterminated answer, and
#: ``SPECIAL`` one that spells a special token in plain text.
SPECIAL = "Sure <|eot_id|> and more"
PROMPTS = {
    "a": ("train", 1, ["A zero.", "A one.", "A two."]),
    "b": ("train", 2, ["B zero.", "B one.", None, "B three.", "B four.", "B five."]),
    "c": ("train", 1, [SPECIAL, "C one.", "C two."]),
    "v": ("val", 1, ["V zero."]),
}


def _write_replay(root, version="t1", model=MODEL, long_answer=None):
    version_dir = os.path.join(root, version)
    answers = os.path.join(version_dir, "answers")
    os.makedirs(answers)
    with open(os.path.join(version_dir, "prompts.jsonl"), "w") as fh:
        for pid, (role, weight, found) in PROMPTS.items():
            fh.write(json.dumps({
                "id": pid, "source": "fixture", "sub": "x", "text": f"Question {pid}?",
                "prompt_tokens": 4, "chem": weight == 2, "chem_tier": "",
                "code": False, "role": role, "samples": len(found),
                "weight": weight}) + "\n")
    with open(os.path.join(answers, "generation.json"), "w") as fh:
        json.dump({"model_name": model, "prompt_style": "chat"}, fh)
    with open(os.path.join(answers, "shard00.jsonl"), "w") as fh:
        for pid, (_role, _weight, found) in PROMPTS.items():
            for j, answer in enumerate(found):
                if pid == "a" and j == 2 and long_answer:
                    answer = long_answer
                fh.write(json.dumps({
                    "id": pid, "sample": j, "text": answer or "cut off mid",
                    "terminated": answer is not None, "has_special": False}) + "\n")
    return version_dir


def _config(tmp_path, **kwargs):
    root = str(tmp_path / "replay")
    if not os.path.exists(os.path.join(root, kwargs.get("replay_version", "t1"))):
        _write_replay(root, kwargs.get("replay_version", "t1"),
                      long_answer=kwargs.pop("long_answer", None))
    kwargs.pop("long_answer", None)
    return T.TextAdapterConfig(model_name=MODEL, replay_root=root,
                               cache_root=str(tmp_path / "cache"),
                               replay_version=kwargs.pop("replay_version", "t1"),
                               **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Pass semantics, without a tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def test_unusable_answers_are_never_targets(tmp_path):
    answers, drops = T.read_answers(_config(tmp_path))
    assert drops == {"unterminated": 1, "special_token": 1}
    assert answers["b"] == ["B zero.", "B one.", "B three.", "B four.", "B five."]
    assert answers["c"] == ["C one.", "C two."]


def test_passes_rotate_answers_and_count_the_repeats(tmp_path):
    config = _config(tmp_path)
    prompts = T.read_prompts(config)
    answers, _ = T.read_answers(config)

    def served(pass_id):
        draws, stats = T._draws(prompts, answers, "train", pass_id)
        return [text for _p, text, _i in draws], stats

    p0, s0 = served(0)
    p1, s1 = served(1)
    p2, s2 = served(2)
    assert p0 == ["A zero.", "B zero.", "B one.", "C one."]
    assert p1 == ["A one.", "B three.", "B four.", "C two."]
    # `a` has three answers, `b` five usable at weight 2, `c` two.
    assert p2 == ["A two.", "B five.", "B zero.", "C one."]
    assert (s0["repeats"], s1["repeats"], s2["repeats"]) == (0, 0, 2)
    # The val split takes one answer per prompt, whatever the prompt's weight.
    val, _ = T._draws(prompts, answers, "val", 0)
    assert [text for _p, text, _i in val] == ["V zero."]


def test_the_partition_is_the_selection_roles(tmp_path):
    part = T.partition(_config(tmp_path))
    assert part.is_role("a", "train") and part.is_role("v", "val")
    assert not part.is_role("v", "train")


def test_a_set_generated_by_another_backbone_is_refused(tmp_path):
    config = _config(tmp_path)
    config.model_name = "meta-llama/Llama-3.2-3B-Instruct"
    with pytest.raises(T.TextBuildError, match="generated by"):
        config.validate()


def test_the_build_version_moves_with_the_answers(tmp_path):
    config = _config(tmp_path)
    before = config.build_version()
    path = os.path.join(config.answers_dir(), "shard00.jsonl")
    with open(path, "a") as fh:
        fh.write(json.dumps({"id": "a", "sample": 3, "text": "A three.",
                             "terminated": True, "has_special": False}) + "\n")
    assert config.build_version() != before


def test_an_ungenerated_set_registers_unbuilt_and_refuses_to_build(tmp_path):
    config = T.TextAdapterConfig(model_name=MODEL, replay_root=str(tmp_path),
                                 cache_root=str(tmp_path / "cache"))
    spec = T.task_specs(config)[T.REPLAY_NAME]
    assert spec.mean_tokens is None and spec.build_version == "ungenerated"
    with pytest.raises(T.TextBuildError, match="have not been generated"):
        T.build(config)


def test_prefix_dispatch():
    assert adapter_for("mol/bace") == "molecules"
    assert adapter_for(f"{TEXT_PREFIX}replay") == "text"
    with pytest.raises(AdapterError):
        adapter_for("graphqa/edge_count")


# ─────────────────────────────────────────────────────────────────────────────
# The built item
# ─────────────────────────────────────────────────────────────────────────────

def test_build_and_load_a_replay_pass(tmp_path):
    from transformers import AutoTokenizer

    long_answer = " ".join(["word"] * 200)
    config = _config(tmp_path, max_length=128, long_answer=long_answer)
    manifest = T.build(config, arms=("graph",), passes=3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    splits = manifest["tasks"][T.REPLAY_NAME]["splits"]
    # Pass 2 serves `a`'s third answer, which is 200 words: dropped, not cut.
    assert splits["train.p2"]["over_length"] == 1
    assert splits["train.p0"]["over_length"] == 0
    assert splits["val.p0"]["emitted"] == 1
    arm = manifest["tasks"][T.REPLAY_NAME]["arms"]["graph"]
    assert arm["train_size"] == 4 and arm["mean_tokens"] > 0

    source = T.load(T.REPLAY_NAME, "train", "graph", pass_id=0, config=config)
    assert len(source) == 4
    nodes, tokens = source.lengths()
    assert nodes == [1, 1, 1, 1]
    for i in range(len(source)):
        item = source[i]
        example = source.example(i)
        assert item["num_nodes"] == 1 and example.split == "train"
        labels = list(item["labels"])
        ids = list(item["input_ids"][0])
        assert len(labels) == len(ids) == tokens[i]
        assert labels[-1] == eot, "the stop token is supervised"
        span = [t for t in labels if t != -100]
        decoded = tokenizer.decode(span)
        assert decoded.endswith(example.answer + "<|eot_id|>")
        assert "Question" not in decoded, "the user turn is not supervised"
        text = item["text"][0]
        assert text.startswith("<|start_header_id|>user<|end_header_id|>\n\n")
        assert "<|begin_of_text|>" not in text

    spec = T.task_specs(config)[T.REPLAY_NAME]
    assert spec.kind == "generator" and spec.cap_per_pass == 4
    assert spec.eval_splits == ()

    with pytest.raises(T.TextBuildError, match="has not been built"):
        T.load(T.REPLAY_NAME, "train", "graph", pass_id=3, config=config)


# ─────────────────────────────────────────────────────────────────────────────
# What wiring it in must not move
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mixture, share", [
    ("molecule_generalist_replay", REPLAY_SHARE),
    ("molecule_generalist_replay08", REPLAY_SHARE_LOW),
])
def test_the_replay_mixture_scales_every_molecule_weight_alike(mixture, share):
    base = {e["name"]: e["weight"] for e in molecule_generalist_mixture()}
    replay = {e["name"]: e for e in MIXTURES[mixture]}
    assert replay.pop("text/replay")["weight"] == share
    assert set(replay) == set(base)
    for name, entry in replay.items():
        assert entry["weight"] == pytest.approx(base[name] * (1 - share))
    total = sum(base.values())
    assert sum(e["weight"] for e in replay.values()) + share == \
        pytest.approx(total)


def test_a_molecules_only_config_hashes_no_text_field():
    config = RunConfig()
    assert not config.has_text_tasks()
    payload = config.hash_payload()
    assert "text_max_length" not in payload and "replay_version" not in payload
    moved = RunConfig(text_max_length=2048, replay_version="v9")
    assert moved.config_hash() == config.config_hash()


def test_a_replay_config_hashes_the_text_fields():
    config = RunConfig(mixture="molecule_generalist_replay")
    assert config.has_text_tasks()
    assert RunConfig(mixture="molecule_generalist_replay",
                     text_max_length=2048).config_hash() != config.config_hash()


def test_a_molecules_only_registry_holds_no_text_task():
    from src.generalist import wiring

    registry, _ = wiring.build_registry(RunConfig())
    assert not [name for name in registry.names() if name.startswith(TEXT_PREFIX)]
