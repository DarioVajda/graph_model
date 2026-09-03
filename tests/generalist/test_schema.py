"""
T1 — the schema (`src/generalist/schema.py`, DESIGN.md §D1).

Three things are pinned here:

* the validator accepts a well-formed example of every answer kind, and rejects
  each malformable field with a message that *names* it (D1.3 — an adapter bug
  has to be readable off the exception, not off a diff of two graphs);
* ``render`` masks exactly the answer span, and does so with the same convention
  `molecules/dataset.py::get_prompt_node_labels` has been training under — that
  function is imported and compared against, so a drift in either is a failure
  here rather than a silent metric shift;
* the token layout is frozen by a golden sequence tied to ``SCHEMA_VERSION``.

The tokenizer is a character-level fake rather than a real one. That is not a
shortcut: the properties under test are the *layout* (which node, which span,
which mask value), and a fake makes the golden readable and keeps the file off
the HF cache. The one tokenizer behaviour that matters and that a char-level
model cannot show — a merge across the answer boundary — gets its own fake.
"""

import pytest

from src.generalist.registry import TaskSpec
from src.generalist.schema import (
    ANSWER_KINDS,
    ANSWER_PREFIX,
    SCHEMA_VERSION,
    SIDECAR_KEY,
    Example,
    SchemaError,
    render,
    validate,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fakes and fixtures
# ─────────────────────────────────────────────────────────────────────────────


class CharTokenizer:
    """One token per character, id = ``ord(c)``.

    Prefix-consistent by construction, which is the ordinary case: tokenizing a
    string's prefix gives a prefix of the string's tokens.
    """

    def __call__(self, texts, padding=False, truncation=True, max_length=512,
                 add_special_tokens=False):
        assert add_special_tokens is False, "the schema must not add specials"
        return {"input_ids": [[ord(c) for c in t][:max_length] for t in texts]}


class MergingTokenizer(CharTokenizer):
    """Character-level, except that ``": "`` is one token (id 1000).

    Stands in for the real thing's behaviour at the answer boundary: the prompt
    prefix ends in ``":"`` and every answer starts with a space, so a real BPE
    merges the two into a token that belongs to both sides.
    """

    MERGE, MERGE_ID = ": ", 1000

    def __call__(self, texts, padding=False, truncation=True, max_length=512,
                 add_special_tokens=False):
        out = []
        for text in texts:
            ids, i = [], 0
            while i < len(text):
                if text[i:i + 2] == self.MERGE:
                    ids.append(self.MERGE_ID)
                    i += 2
                else:
                    ids.append(ord(text[i]))
                    i += 1
            out.append(ids[:max_length])
        return {"input_ids": out}


QUESTION = "Question: how many rings does this molecule have?"


def graph_item(question, answer, arm="graph", ds_label="mol/ring_count"):
    """A ``TextGraphDataset`` item of the shape the molecules package builds.

    Graph arm: atom nodes, an edge-free question node, the prompt node last
    (`data.py::attach_question` + ``relabel_for_dataset``). Flat arm: one node
    holding question + SMILES + answer (`dataset.py::build_flat_example`).
    ``ds_label`` is present because a real item always carries it.
    """
    if arm == "graph":
        texts = ["carbon aromatic ring deg3 H1", "carbon deg2 H2", "single bond",
                 question, f"{ANSWER_PREFIX}{answer}"]
        return {
            "text": texts, "num_nodes": 5, "prompt_node": 4, "question_node": 3,
            "edges": [(0, 2), (2, 0), (1, 2), (2, 1), (4, 0), (4, 1)],
            "ds_label": ds_label,
        }
    text = f"{question}\nSMILES: c1ccccc1O{ANSWER_PREFIX}{answer}"
    return {"text": [text], "num_nodes": 1, "prompt_node": 0, "edges": [],
            "ds_label": ds_label}


def spec_for(kind, name="mol/ring_count", held_out=False):
    return TaskSpec(name=name, domain="molecules", adapter="molecules",
                    kind="generator", answer_kind=kind, held_out=held_out,
                    mean_tokens=64.0)


#: One well-formed example per answer kind, with the task it belongs to.
#: ``" 3"`` and ``" Yes"`` carry their own leading space, as `tasks.py` emits
#: them. A ``smiles`` answer carries none: D1.3 requires it to equal its own
#: RDKit canonicalization, and a leading space is not part of that string — so
#: the whitespace, if any, belongs to the adapter's answer prefix.
KIND_CASES = {
    "token": (" 3", "mol/ring_count"),
    "yesno": (" Yes", "mol/bace"),
    "text": (" The molecule is a phenol found in coal tar.", "mol/chebi20"),
    "smiles": ("Oc1ccccc1", "mol/g2s"),
}


def example_of(kind, arm="graph", **overrides):
    """A well-formed example, with ``overrides`` applied.

    The graph is rebuilt from the overridden ``answer`` so that changing the
    answer exercises the *kind's* check rather than tripping the prompt-tail
    check first. ``question`` deliberately does not flow into the graph: an
    example whose metadata disagrees with its question node is one of the things
    the validator exists to catch.
    """
    answer, name = KIND_CASES[kind]
    answer = overrides.get("answer", answer)
    name = overrides.get("task", name)
    fields = dict(
        task=name, domain="molecules", split="train", arm=arm,
        graph=graph_item(QUESTION, answer, arm=arm, ds_label=name),
        question=QUESTION,
        answer=answer, answer_kind=kind, key="Oc1ccccc1", meta={"source": "bbbp"},
    )
    fields.update(overrides)
    return Example(**fields)


# ─────────────────────────────────────────────────────────────────────────────
# The validator accepts what it should
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ANSWER_KINDS)
@pytest.mark.parametrize("arm", ["graph", "flat"])
def test_validate_accepts_every_kind_in_both_arms(kind, arm):
    example = example_of(kind, arm=arm)
    validate(example, spec_for(kind, name=example.task))


def test_validate_accepts_held_out_split_only_on_a_held_out_task():
    example = example_of("token", split="held_out", task="mol/bond_path")
    validate(example, spec_for("token", name="mol/bond_path", held_out=True))


def test_a_smiles_answer_is_the_canonical_string_and_nothing_else():
    """Not even a leading space: the target has to be a function of the molecule.

    The graph-to-SMILES metrics are validity, round-trip match and canonical
    exact match (`MOLECULE_GENERALIST.md` §5); a target carrying decoration the
    canonicalization does not produce would make the third of those a comparison
    against one arbitrary spelling. Whitespace belongs to the answer prefix.
    """
    validate(example_of("smiles"), spec_for("smiles", name="mol/g2s"))
    with pytest.raises(SchemaError, match="answer:"):
        validate(example_of("smiles", answer=" Oc1ccccc1"),
                 spec_for("smiles", name="mol/g2s"))


# ─────────────────────────────────────────────────────────────────────────────
# ... and rejects what it should, naming the field
# ─────────────────────────────────────────────────────────────────────────────


def _malformed():
    """``(field named in the message, the broken example, its spec)`` cases."""
    cases = []

    def add(field, example, spec=None):
        cases.append((field, example, spec or spec_for(example.answer_kind,
                                                       name=example.task)))

    add("task", example_of("token", task="mol/somethingelse"),
        spec_for("token", name="mol/ring_count"))
    add("domain", example_of("token", domain="kgqa"))
    add("split", example_of("token", split="validation"))
    add("split", example_of("token", split="held_out"))
    add("split", Example(**{**example_of("token").__dict__, "split": "train",
                           "task": "mol/bond_path"}),
        spec_for("token", name="mol/bond_path", held_out=True))
    add("arm", example_of("token", arm="hybrid"))
    add("answer_kind", example_of("token", answer_kind="text"),
        spec_for("token"))
    add("answer_kind", example_of("token", answer_kind="rhyme"),
        spec_for("token"))
    add("key", example_of("token", key=""))
    add("meta", example_of("token", meta=["source"]))
    add("question", example_of("token", question="Question: what colour is it?"))

    # The graph disagreeing with the metadata, in each of the ways it can.
    broken = example_of("token")
    broken.graph = dict(broken.graph, prompt_node=99)
    add("graph", broken)

    broken = example_of("token")
    broken.graph = dict(broken.graph, num_nodes=4)
    add("graph", broken)

    broken = example_of("token")
    broken.graph = dict(broken.graph, text=["only one node"], num_nodes=1,
                        prompt_node=0, question_node=None)
    add("question", broken)

    broken = example_of("token")
    texts = list(broken.graph["text"])
    texts[-1] = f"{ANSWER_PREFIX} 7"
    broken.graph = dict(broken.graph, text=texts)
    add("answer", broken)

    # Kind-specific answer checks.
    add("answer", example_of("yesno", answer=" yes"),
        spec_for("yesno", name="mol/bace"))
    add("answer", example_of("smiles", answer="Oc1ccccc1)("),
        spec_for("smiles", name="mol/g2s"))
    add("answer", example_of("smiles", answer="OCC"),
        spec_for("smiles", name="mol/g2s"))
    return cases


MALFORMED = _malformed()


@pytest.mark.parametrize(
    "field,example,spec", MALFORMED,
    ids=[f"{i}-{case[0]}" for i, case in enumerate(MALFORMED)])
def test_validate_rejects_and_names_the_field(field, example, spec):
    with pytest.raises(SchemaError) as excinfo:
        validate(example, spec)
    message = str(excinfo.value)
    assert message.startswith(f"{field}:"), message


def test_yesno_words_come_from_the_module_that_scores_them():
    """`molecules/evaluate.py` owns the two label words; a copy would drift."""
    from src.experiments.molecules.evaluate import NO_WORD, YES_WORD

    validate(example_of("yesno"), spec_for("yesno", name="mol/bace"))
    validate(example_of("yesno", answer=YES_WORD),
             spec_for("yesno", name="mol/bace"))
    with pytest.raises(SchemaError, match="answer:"):
        validate(example_of("yesno", answer=" Maybe"),
                 spec_for("yesno", name="mol/bace"))
    # An adapter with different words says so explicitly.
    validate(example_of("yesno", answer=" true"),
             spec_for("yesno", name="mol/bace"), yes_no_words=(" true", " false"))
    assert (YES_WORD, NO_WORD) == (" Yes", " No")


# ─────────────────────────────────────────────────────────────────────────────
# render
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["token", "yesno"])
def test_render_supervises_only_the_last_prompt_token(kind):
    example = example_of(kind)
    rendered = render(example, CharTokenizer())
    prompt_ids = rendered.input_ids[rendered.prompt_node]

    assert rendered.prompt_node == example.graph["prompt_node"]
    assert rendered.answer_start == len(prompt_ids) - 1
    assert rendered.labels[:-1] == [-100] * (len(prompt_ids) - 1)
    assert rendered.labels[-1] == prompt_ids[-1]


@pytest.mark.parametrize("kind", ["token", "yesno"])
def test_render_matches_the_molecules_label_function(kind):
    """The convention `render` claims to mirror, compared against the original."""
    from src.experiments.molecules.dataset import get_prompt_node_labels

    for arm in ("graph", "flat"):
        example = example_of(kind, arm=arm)
        rendered = render(example, CharTokenizer())
        existing = get_prompt_node_labels({
            "input_ids": rendered.input_ids,
            "prompt_node": rendered.prompt_node,
        })
        assert rendered.labels == existing


@pytest.mark.parametrize("kind", ["text", "smiles"])
def test_render_supervises_the_whole_multi_token_answer(kind):
    example = example_of(kind)
    rendered = render(example, CharTokenizer())
    prompt_ids = rendered.input_ids[rendered.prompt_node]
    prompt_text = example.graph["text"][rendered.prompt_node]

    # Char-level: the span starts exactly where the answer starts in the string.
    assert rendered.answer_start == len(prompt_text) - len(example.answer)
    assert rendered.labels[:rendered.answer_start] == [-100] * rendered.answer_start
    assert rendered.labels[rendered.answer_start:] == prompt_ids[rendered.answer_start:]
    assert [chr(i) for i in rendered.labels[rendered.answer_start:]] == \
        list(example.answer)


def test_render_widens_the_span_over_a_merged_boundary_token():
    """A token spanning the prefix/answer boundary is supervised, not dropped.

    The caption's prompt node is ``"\\nA: The molecule …"``, and ``": "`` is one
    token, so the answer's first character lives inside a token the prefix also
    claims. Supervising it is what makes the answer emittable; skipping it would
    leave the leading space unsupervised and the model unable to start.
    """
    example = example_of("text")
    rendered = render(example, MergingTokenizer())
    prompt_ids = rendered.input_ids[rendered.prompt_node]

    # "\n", "A", ": " -> the merged token is at index 2 and is the span's start.
    assert prompt_ids[:3] == [ord("\n"), ord("A"), MergingTokenizer.MERGE_ID]
    assert rendered.answer_start == 2
    assert rendered.labels[:2] == [-100, -100]
    assert rendered.labels[2] == MergingTokenizer.MERGE_ID


def test_render_refuses_an_answer_that_is_not_the_prompt_tail():
    example = example_of("text")
    texts = list(example.graph["text"])
    texts[-1] = f"{ANSWER_PREFIX} a different caption"
    example.graph = dict(example.graph, text=texts)
    with pytest.raises(SchemaError, match="answer:"):
        render(example, CharTokenizer())


def test_render_tokenizes_every_node_in_order():
    example = example_of("token")
    rendered = render(example, CharTokenizer())
    assert len(rendered.input_ids) == example.graph["num_nodes"]
    for text, ids in zip(example.graph["text"], rendered.input_ids):
        assert ids == [ord(c) for c in text]


# ─────────────────────────────────────────────────────────────────────────────
# The golden layout — this is what SCHEMA_VERSION names
# ─────────────────────────────────────────────────────────────────────────────


def test_golden_token_layout_pins_the_format_version():
    """Change the layout, change :data:`SCHEMA_VERSION`, change this golden.

    A resume across a schema version is refused (D5.4) precisely because the
    mask below would move; this test is the thing that makes the version mean
    something. Node texts are single characters so the expected ids are readable:
    ``ord('C')=67``, ``ord('O')=79``, ``ord('Q')=81``, ``ord('?')=63``,
    ``'\\nA: 3'`` -> ``[10, 65, 58, 32, 51]``.
    """
    assert SCHEMA_VERSION == "1"

    example = Example(
        task="mol/ring_count", domain="molecules", split="train", arm="graph",
        graph={"text": ["C", "O", "Q?", "\nA: 3"], "num_nodes": 4,
               "prompt_node": 3, "question_node": 2, "edges": [(0, 1), (1, 0)],
               "ds_label": "mol/ring_count"},
        question="Q?", answer=" 3", answer_kind="token", key="CO", meta={},
    )
    rendered = render(example, CharTokenizer())

    assert rendered.input_ids == [[67], [79], [81, 63], [10, 65, 58, 32, 51]]
    assert rendered.labels == [-100, -100, -100, -100, 51]
    assert rendered.answer_start == 4
    assert rendered.prompt_node == 3


# ─────────────────────────────────────────────────────────────────────────────
# to_item / from_item
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ANSWER_KINDS)
@pytest.mark.parametrize("arm", ["graph", "flat"])
def test_to_item_from_item_round_trip(kind, arm):
    example = example_of(kind, arm=arm)
    spec = spec_for(kind, name=example.task)
    item = example.to_item()
    assert Example.from_item(item, spec) == example


def test_to_item_leaves_the_graph_collatable():
    """The item is the graph dict plus a sidecar; the collator reads named keys."""
    example = example_of("yesno")
    item = example.to_item()
    for key in ("text", "num_nodes", "prompt_node", "edges"):
        assert item[key] == example.graph[key]
    assert item["ds_label"] == example.task
    assert item[SIDECAR_KEY]["schema_version"] == SCHEMA_VERSION
    assert Example.from_item(item, spec_for("yesno", name="mol/bace")).graph \
        == example.graph


def test_from_item_derives_what_it_can_from_a_bare_item():
    """An item an existing builder wrote carries no sidecar.

    The graph arm's question comes off the question node verbatim; the flat arm
    is one node with no question node, so the question has to be supplied — the
    schema will not guess where the question ends and the SMILES begins.
    """
    spec = spec_for("token")
    for arm, extra in (("graph", {}), ("flat", {"question": QUESTION})):
        item = graph_item(QUESTION, " 3", arm=arm)
        example = Example.from_item(item, spec, split="val", key="Oc1ccccc1",
                                    **extra)
        assert example.task == spec.name
        assert example.domain == spec.domain
        assert example.answer_kind == spec.answer_kind
        assert example.arm == arm
        assert example.question == QUESTION
        assert example.answer == " 3"
        assert example.split == "val"
        validate(example, spec)


def test_from_item_names_what_it_cannot_derive():
    item = graph_item(QUESTION, " 3")
    with pytest.raises(SchemaError, match="split:"):
        Example.from_item(item, spec_for("token"), key="Oc1ccccc1")
    with pytest.raises(SchemaError, match="key:"):
        Example.from_item(item, spec_for("token"), split="train")
    with pytest.raises(SchemaError, match="question:"):
        Example.from_item(graph_item(QUESTION, " 3", arm="flat"),
                          spec_for("token"), split="train", key="Oc1ccccc1")


def test_from_item_overrides_beat_the_sidecar():
    example = example_of("token")
    spec = spec_for("token")
    rebuilt = Example.from_item(example.to_item(), spec, split="test")
    assert rebuilt.split == "test"
    assert rebuilt.answer == example.answer
