"""`mol/assistant`: the registry entry, the draw, and the two traps it sits on.

§9.4's task is the one corpus here whose rows are **composed offline** rather
than generated from a molecule pool, and that makes two things testable that
nothing else in the adapter needs:

* the composed set lives under an uncommitted ``results/``, so the path to it is
  a config field — and a config field is exactly what `build_version` hashes, so
  the test that matters is that adding it did **not** move the hash;
* a ``text`` answer is stored as ``answer[1:]``, so the draw has to hand over the
  leading space `_draw_chebi` does. It is invisible on a long reply (the first
  character is simply eaten) and fatal on the 51 one-character answers in the
  set, which become the empty string the schema then refuses.
"""

import json
import os

import pytest

from src.generalist.adapters import molecules as M
from src.generalist.tools.assistant_score import render_of

# ─────────────────────────────────────────────────────────────────────────────
# A composed set, in the shape `assistant_compose.py` writes
# ─────────────────────────────────────────────────────────────────────────────

#: Benzene and toluene: two molecules whose canonical SMILES parse, which is all
#: `_draw_assistant` asks of a key.
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"


def _row(row_id, answer, **overrides):
    row = {
        "id": row_id, "key": BENZENE, "smiles": BENZENE, "answer": answer,
        "answers": ["yes"], "brief": {"format": "prose", "length": "short",
                                      "register": "terse"},
        "cell": "report/none/prose", "facts": [
            {"family": "ring_membership", "kind": "yesno", "value": "yes",
             "text": "Atom 1 (C) is in a ring.", "atoms": [1]}],
        "gloss": None, "pointer": "", "question": "Is atom 1 in a ring?",
        "rendered_reply": "Atom 1 (C) is in a ring.", "role": "train",
        "shots": [], "situation": "curator", "skeleton": None,
        "source": "hiv", "statements": ["Atom 1 (C) is in a ring."],
        "task": "report", "turns": [{"role": "person", "text": None},
                                    {"role": "assistant",
                                     "text": "Atom 1 (C) is in a ring."}],
        "twist": "none", "verdict": None, "writer": "gemma-4-31B-it",
    }
    row.update(overrides)
    return row


@pytest.fixture
def composed(tmp_path):
    """A two-row train split and a one-row test split on disk."""
    directory = tmp_path / "composed"
    directory.mkdir()
    rows = [
        _row("train-00000", "Yes, atom 1 (C) is in a ring."),
        # The one-character answer. This is the row that failed the real build.
        _row("train-00001", "6", facts=[
            {"family": "ring_size", "kind": "count", "value": "6",
             "text": "The smallest ring containing atom 1 (C) has 6 atoms.",
             "atoms": [1]}],
             statements=["The smallest ring containing atom 1 (C) has 6 atoms."]),
    ]
    with open(directory / "train.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with open(directory / "test.jsonl", "w") as handle:
        handle.write(json.dumps(_row("test-00000", "No.", key=TOLUENE,
                                     smiles=TOLUENE, role="test")) + "\n")
    return str(directory)


@pytest.fixture
def config(composed):
    return M.MoleculeAdapterConfig(assistant_dir=composed)


# ─────────────────────────────────────────────────────────────────────────────
# The hash. This is the test the whole design decision rests on.
# ─────────────────────────────────────────────────────────────────────────────

def test_assistant_dir_does_not_move_the_build_version(composed):
    """A new config field would otherwise orphan every artifact in the repo.

    `build_dir` is `build_version`, so a field that entered the hash would send
    every existing config to an empty directory and put the campaign's
    checkpoints out of step with the registry it recorded. `assistant_dir` is a
    *location*, like `cache_root` and `chebi_dir`, and is popped for their reason.
    """
    default = M.MoleculeAdapterConfig()
    moved = M.MoleculeAdapterConfig(assistant_dir=composed)
    assert default.build_version() == moved.build_version()
    assert default.build_dir() == moved.build_dir()


def test_a_field_that_does_change_the_bytes_still_moves_it():
    """The control for the test above: the pop is narrow, not a blanket."""
    default = M.MoleculeAdapterConfig()
    other = M.MoleculeAdapterConfig(encoding="levi")
    if other.encoding != default.encoding:
        assert other.build_version() != default.build_version()


# ─────────────────────────────────────────────────────────────────────────────
# The draw
# ─────────────────────────────────────────────────────────────────────────────

def test_the_answer_carries_the_leading_space(config):
    """`_materialise` stores `answer[1:]` for a ``text`` kind.

    Without the space every reply in the set silently loses its first character.
    """
    draws, _stats = M._draw_assistant(config, "train")
    for _mol, _question, answer, _named, _key, _meta in draws:
        assert answer.startswith(" ")


def test_a_one_character_answer_survives_the_slice(config):
    """The 51 rows that turned the omission above into a hard failure."""
    draws, _stats = M._draw_assistant(config, "train")
    answers = [answer for _m, _q, answer, _n, _k, _meta in draws]
    short = [a for a in answers if a.strip() == "6"]
    assert short, "the fixture's one-character row is missing"
    assert short[0][1:] == "6"


def test_the_draw_is_the_adapters_six_tuple(config):
    draws, _stats = M._draw_assistant(config, "train")
    assert len(draws) == 2
    for draw in draws:
        assert len(draw) == 6
    _mol, question, _answer, named, key, meta = draws[0]
    assert key == BENZENE
    assert "ring" in question
    # Every fact is atom-scoped, and the sheet is 1-based against RDKit's 0-based.
    assert named == [0]
    assert meta["id"] == "train-00000"


def test_the_shots_ride_in_meta_as_json(config, composed):
    """They have to survive the sidecar, which is JSON — so no live molecules."""
    with open(os.path.join(composed, "train.jsonl")) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows[0]["shots"] = [{"key": TOLUENE, "question": "Is atom 2 in a ring?",
                         "answer": "Yes.", "polarity": "yes", "id": "x"}]
    with open(os.path.join(composed, "train.jsonl"), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    draws, stats = M._draw_assistant(config, "train")
    meta = draws[0][5]
    assert meta["shots"] == [{"key": TOLUENE, "question": "Is atom 2 in a ring?",
                              "answer": "Yes."}]
    json.dumps(meta)          # the sidecar writes this; it must be serialisable
    assert stats["by_shot_count"] == {"0": 1, "1": 1}


def test_meta_carries_everything_the_scorer_needs(config):
    """The artifact has to describe itself — see `_draw_assistant`'s docstring."""
    draws, _stats = M._draw_assistant(config, "train")
    meta = draws[0][5]
    for field in ("statements", "answers", "verdict", "gloss", "skeleton",
                  "turns", "rendered_reply", "facts", "twist", "task", "brief"):
        assert field in meta, field


def test_a_missing_composed_set_names_the_pipeline(tmp_path):
    config = M.MoleculeAdapterConfig(assistant_dir=str(tmp_path / "nothing"))
    with pytest.raises(M.AdapterBuildError) as caught:
        M._draw_assistant(config, "train")
    assert "intent_pipeline" in str(caught.value)


def test_the_digest_is_recorded_because_it_reaches_no_hash(config):
    draws, stats = M._draw_assistant(config, "train")
    assert stats["assistant_digest"]
    assert stats["n"] == len(draws)


# ─────────────────────────────────────────────────────────────────────────────
# The registry entry
# ─────────────────────────────────────────────────────────────────────────────

def test_the_task_has_no_val_split():
    """`assistant_compose.py` writes two files, and naming a third would fail at
    load rather than at resolve."""
    assert M.splits_for(M.ASSISTANT_TASK) == ("train", "test")
    spec = M.task_specs(M.MoleculeAdapterConfig())[f"mol/{M.ASSISTANT_TASK}"]
    assert spec.eval_splits == ("test",)


def test_the_task_is_registered_but_not_in_the_default_build():
    """A default data_prep must not depend on an uncommitted GPU pipeline."""
    config = M.MoleculeAdapterConfig()
    assert f"mol/{M.ASSISTANT_TASK}" in M.task_specs(config)
    assert M.ASSISTANT_TASK not in M.all_tasks(config)


def test_the_spec_is_a_free_text_corpus():
    spec = M.task_specs(M.MoleculeAdapterConfig())[f"mol/{M.ASSISTANT_TASK}"]
    assert spec.kind == "corpus"
    assert spec.answer_kind == "text"
    assert spec.max_new_tokens == 160
    # Routed by the row's own question, never by a template: the whole point is
    # that the person's turn is different every time.
    assert spec.question_template is None


# ─────────────────────────────────────────────────────────────────────────────
# The flat arm
# ─────────────────────────────────────────────────────────────────────────────

def test_the_flat_arm_refuses_rather_than_dropping_the_demonstrations(config):
    """Silently building without the shots would answer a different question."""
    draws, _stats = M._draw_assistant(config, "train")
    with pytest.raises(M.AdapterBuildError) as caught:
        M._graphs_for(config, M.ASSISTANT_TASK, "flat", draws, 0, "text")
    assert "flat" in str(caught.value)


# ─────────────────────────────────────────────────────────────────────────────
# The render dict the offline scorer rebuilds
# ─────────────────────────────────────────────────────────────────────────────

def test_render_of_recovers_the_two_keys_no_composed_row_stores():
    """`ask` is the one field the composed row drops, and the judge reads two
    keys off it. Both are recoverable, and a drift here would quietly stop the
    judge being told a decline was wanted — its largest error on the build."""
    plain = render_of({"statements": ["a"], "twist": "none", "verdict": None})
    assert plain["ask"]["answerable"] is True
    assert plain["ask"]["constraint"] is False

    refusing = render_of({"statements": ["a"], "twist": "unanswerable",
                          "verdict": None})
    assert refusing["ask"]["answerable"] is False

    deciding = render_of({"statements": ["a"], "twist": "none", "verdict": "no"})
    assert deciding["ask"]["constraint"] is True
    assert deciding["verdict"] == "no"


def test_render_of_inverts_the_composed_rows_renaming(config):
    """`intent_accept` wrote the row as a renaming of the render dict."""
    draws, _stats = M._draw_assistant(config, "train")
    render = render_of(draws[0][5])
    assert render["statements"] == ["Atom 1 (C) is in a ring."]
    assert render["reply"] == "Atom 1 (C) is in a ring."
    assert render["turns"]
