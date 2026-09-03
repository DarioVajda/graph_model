"""
D1 — the one example format, and the one place text becomes tokens.

Every training and evaluation item, from every adapter, is an :class:`Example`.
The point of the file is that there is exactly *one* answer to "how does an
answer become a supervised span", version-stamped as :data:`SCHEMA_VERSION`,
rather than one per experiment. Formatting was the single biggest lever found on
KGQA (format v3, +4.6 F1); eight copies of it would make that lever unmeasurable.

**The graph is not a new format.** ``Example.graph`` *is* the dict a
``TextGraphDataset`` item carries — ``text`` (one string per node), ``num_nodes``,
``prompt_node``, ``edges``, and whatever feature columns the build computed
(``input_ids``, ``labels``, ``shortest_path_dists``, ``magnetic_V``, …). So an
Example goes into ``GraphCollatorV2`` unchanged: :meth:`Example.to_item` returns
that dict plus a sidecar of the schema-level fields, and the collator reads named
keys only, so the sidecar is inert to it. Nothing downstream needs new collator
code, and :meth:`Example.from_item` reads an item that the molecules package (or
any other existing builder) produced.

**What ``render`` mirrors.** ``molecules/dataset.py::get_prompt_node_labels``:
labels are aligned to the *prompt node's* token list — not to the packed batch
sequence — with everything outside the answer span set to ``-100``. That is the
shape ``GraphCollatorV2`` asserts on (``lab.shape[0] != p['prompt_len']`` is a
hard error there), and it is the same contract `expressiveness` and `probes` use.
For ``token`` and ``yesno`` the span is the final token of the prompt node, which
is byte-for-byte what `get_prompt_node_labels` does and what `tasks.py`
guarantees is the whole answer. For the multi-token kinds (``text``, ``smiles``)
the span is found by tokenizing the prompt node's text with the answer removed
and taking the common prefix — see :func:`render`.

**No chat template.** The molecules runs are on base ``meta-llama/Llama-3.2-1B``
(`molecules/config.py::MODEL_NAME`), and a chat template on base weights is a
format the model has never seen. D3's "instruct weights + chat template, both or
neither" therefore resolves to *neither* here. When an instruct backbone lands,
the template goes in this function and the version bumps — that is the whole
reason the version exists.

Free of torch at import time: this module is imported by everything else in the
harness, including the CPU-only ``validate`` mode. RDKit and the molecules
label words are imported lazily, inside the checks that need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Bumped whenever the token layout of a rendered example changes. Recorded in
#: every checkpoint's ``state.json``; a resume across a change is refused (D5.4),
#: because a different mask is a silent metric shift rather than a crash.
SCHEMA_VERSION = "1"

#: D1.1. The kind decides the loss span, the scorer and whether generation runs.
#: A task has exactly one kind.
ANSWER_KINDS = ("token", "yesno", "text", "smiles")

#: The kinds whose answer is a single token by construction, so the supervised
#: span is the prompt node's last token (`tasks.py`: every Tier-A answer is
#: ` Yes`/` No` or a numeral).
SINGLE_TOKEN_KINDS = ("token", "yesno")

SPLITS = ("train", "val", "test", "held_out")
ARMS = ("graph", "flat")

#: The prompt node's answer prefix, as `molecules/data.py::attach_question` writes
#: it: no trailing space, because every answer carries its own leading space and
#: the two arms have to agree byte-for-byte on the tokens before the scored one.
ANSWER_PREFIX = "\nA:"

#: Key under which :meth:`Example.to_item` stows the schema-level fields inside
#: the graph item. Leading underscore so it cannot collide with a
#: ``TextGraphDataset`` feature column.
SIDECAR_KEY = "_schema"


class SchemaError(ValueError):
    """An example that violates D1. The message always names the field."""


# ─────────────────────────────────────────────────────────────────────────────
# The example
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Example:
    """One item, from any adapter, in any arm.

    ``graph`` is the ``TextGraphDataset`` item dict (see the module docstring).
    ``question`` and ``answer`` are the text the adapter emitted; both are also
    present inside the graph (the question in its own prefix node when
    ``question_node`` is on, the answer at the tail of the prompt node), and
    :func:`validate` checks they agree — a mismatch means the graph and the
    metadata drifted, which would score one thing while training another.

    ``key`` is the partition key: the stereo-free canonical SMILES for molecules
    (`MOLECULE_GENERALIST.md` §3), opaque to everything but the adapter that
    made it. ``meta`` is adapter-owned and must stay JSON-serialisable, because
    it travels into the per-example report.
    """

    task: str
    domain: str
    split: str
    arm: str
    graph: dict
    question: str
    answer: str
    answer_kind: str
    key: str
    meta: dict = field(default_factory=dict)

    # ── conversion ───────────────────────────────────────────────────────────

    def to_item(self) -> dict:
        """The ``TextGraphDataset`` item this example collates as.

        A shallow copy of ``graph`` with ``ds_label`` set to the task (that is
        the column ``TextGraphDataset`` already uses to keep an item's origin
        through a merge) and the schema fields under :data:`SIDECAR_KEY`.
        ``GraphCollatorV2`` reads named keys, so the sidecar costs it nothing
        and buys a lossless round-trip.
        """
        item = dict(self.graph)
        item["ds_label"] = self.task
        item[SIDECAR_KEY] = {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "domain": self.domain,
            "split": self.split,
            "arm": self.arm,
            "question": self.question,
            "answer": self.answer,
            "answer_kind": self.answer_kind,
            "key": self.key,
            "meta": dict(self.meta),
        }
        return item

    @classmethod
    def from_item(cls, item: dict, spec, **overrides) -> "Example":
        """Rebuild an Example from a graph item.

        Two callers, two paths:

        * an item this schema wrote — the sidecar is present and is used
          verbatim, so ``from_item(e.to_item(), spec) == e``;
        * an item an existing builder wrote (the molecules package, a ``.gtds``
          loaded off disk) — there is no sidecar, so ``task``/``domain``/
          ``answer_kind`` come from ``spec``, ``question`` and ``answer`` are
          read back out of the graph, ``arm`` is inferred from the node count
          (the flat arm is a single-node graph by construction,
          `molecules/dataset.py`), and anything still missing must be passed as
          a keyword — ``split`` and ``key`` always are, because neither is
          recoverable from the graph.

        ``overrides`` win over both, so an adapter can correct an inference.
        """
        if not isinstance(item, dict):
            raise SchemaError("graph: item must be a dict")

        side = item.get(SIDECAR_KEY)
        if side is not None:
            fields = dict(side)
            fields.pop("schema_version", None)
        else:
            fields = {
                "task": getattr(spec, "name", None),
                "domain": getattr(spec, "domain", None),
                "answer_kind": getattr(spec, "answer_kind", None),
                "arm": "flat" if _num_nodes(item) == 1 else "graph",
                "question": _question_from_item(item),
                "answer": _answer_from_item(item),
                "meta": {},
            }
        fields.update(overrides)

        graph = {k: v for k, v in item.items() if k != SIDECAR_KEY}
        for name in ("task", "domain", "split", "arm", "question", "answer",
                     "answer_kind", "key"):
            if fields.get(name) is None:
                raise SchemaError(
                    f"{name}: not in the item's sidecar, not derivable from the "
                    f"graph, and not passed as a keyword")
        return cls(
            task=fields["task"], domain=fields["domain"], split=fields["split"],
            arm=fields["arm"], graph=graph, question=fields["question"],
            answer=fields["answer"], answer_kind=fields["answer_kind"],
            key=fields["key"], meta=dict(fields.get("meta") or {}),
        )


@dataclass
class Rendered:
    """What :func:`render` returns.

    ``input_ids`` is per node, in node order, exactly as
    ``TextGraphDataset.tokenize`` stores the column. ``labels`` is aligned to the
    *prompt node's* tokens (the collator's contract, see the module docstring),
    ``-100`` outside the answer span. ``answer_start`` indexes into the prompt
    node's tokens.
    """

    input_ids: list[list[int]]
    labels: list[int]
    answer_start: int
    prompt_node: int


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render(example: Example, tokenizer, max_length: int = 512) -> Rendered:
    """Text -> tokens + the supervised span. D1.2: the schema's job, not the adapter's.

    The tokenizer call is the same one ``TextGraphDataset.tokenize`` makes
    (``add_special_tokens=False``, per-node truncation at ``max_length``, no
    EOS), so a dataset built through that path and an example rendered here
    produce identical ``input_ids``.

    The answer span:

    * ``token`` / ``yesno`` — the prompt node's **last** token. This is
      `get_prompt_node_labels` verbatim. `tasks.py` emits single-token answers
      and asserts they tokenize to at most two tokens with the answer in the
      last, so the last token is exactly the answer and nothing else.
    * ``text`` / ``smiles`` — the prompt node's text with the answer suffix
      removed is tokenized on its own, and the span starts at the first position
      where that tokenization and the full one disagree. When the tokenizer
      merges across the boundary (a prefix character and the answer's first
      character landing in one token) the span *widens* by that token rather
      than narrowing: the model must emit the merged token to emit the answer,
      so supervising it is correct, and dropping it would leave the answer's
      first character unsupervised.
    """
    kind = example.answer_kind
    if kind not in ANSWER_KINDS:
        raise SchemaError(f"answer_kind: {kind!r} is not one of {ANSWER_KINDS}")

    texts = _texts(example.graph)
    prompt_node = _prompt_node(example.graph, len(texts))
    input_ids = _tokenize(tokenizer, texts, max_length)
    prompt_ids = input_ids[prompt_node]
    if not prompt_ids:
        raise SchemaError("graph: the prompt node tokenizes to nothing")

    if kind in SINGLE_TOKEN_KINDS:
        answer_start = len(prompt_ids) - 1
    else:
        prompt_text = texts[prompt_node]
        if not example.answer:
            raise SchemaError("answer: empty, so there is no span to supervise")
        if not prompt_text.endswith(example.answer):
            raise SchemaError(
                f"answer: {example.answer!r} is not the tail of the prompt node's "
                f"text {prompt_text!r}; the supervised span cannot be located")
        prefix = prompt_text[: len(prompt_text) - len(example.answer)]
        prefix_ids = _tokenize(tokenizer, [prefix], max_length)[0] if prefix else []
        answer_start = 0
        for a, b in zip(prefix_ids, prompt_ids):
            if a != b:
                break
            answer_start += 1
        answer_start = min(answer_start, len(prompt_ids) - 1)

    labels = [-100] * len(prompt_ids)
    labels[answer_start:] = prompt_ids[answer_start:]
    return Rendered(input_ids=input_ids, labels=labels,
                    answer_start=answer_start, prompt_node=prompt_node)


# ─────────────────────────────────────────────────────────────────────────────
# D1.3 — the validator
# ─────────────────────────────────────────────────────────────────────────────

def validate(example: Example, spec, yes_no_words=None) -> None:
    """Raise :class:`SchemaError` naming the first field that is wrong.

    Runs on every item at adapter build time and on a sample at load time: an
    adapter that emits an invalid item fails the *build*, not the run. The graph
    itself is ``TextGraphDataset``'s to validate; what is checked here is the
    agreement between the graph and the metadata beside it, which nothing else
    looks at.

    ``yes_no_words`` defaults to the two label words the margin readout scores
    with (``molecules/evaluate.py``'s ``YES_WORD`` / ``NO_WORD``), imported
    lazily so this module stays torch-free at import. A ``yesno`` answer that is
    not one of them would be silently unscoreable: the margin is read at the
    answer position for those two token ids and nothing else.
    """
    if not isinstance(example, Example):
        raise SchemaError("example: not an Example")

    for name in ("task", "domain", "split", "arm", "question", "answer",
                 "answer_kind", "key"):
        value = getattr(example, name)
        if not isinstance(value, str) or not value:
            raise SchemaError(f"{name}: must be a non-empty string, got {value!r}")
    if not isinstance(example.meta, dict):
        raise SchemaError(f"meta: must be a dict, got {type(example.meta).__name__}")

    if example.task != getattr(spec, "name", example.task):
        raise SchemaError(
            f"task: {example.task!r} does not match the spec's {spec.name!r}")
    if example.domain != getattr(spec, "domain", example.domain):
        raise SchemaError(
            f"domain: {example.domain!r} does not match the spec's {spec.domain!r}")
    if example.arm not in ARMS:
        raise SchemaError(f"arm: {example.arm!r} is not one of {ARMS}")

    if example.split not in SPLITS:
        raise SchemaError(f"split: {example.split!r} is not one of {SPLITS}")
    if getattr(spec, "held_out", False):
        if example.split != "held_out":
            raise SchemaError(
                f"split: {spec.name!r} is held out and admits only 'held_out', "
                f"got {example.split!r}")
    elif example.split == "held_out":
        raise SchemaError(
            f"split: 'held_out' on {example.task!r}, which is not a held-out task")

    if example.answer_kind not in ANSWER_KINDS:
        raise SchemaError(
            f"answer_kind: {example.answer_kind!r} is not one of {ANSWER_KINDS}")
    spec_kind = getattr(spec, "answer_kind", example.answer_kind)
    if example.answer_kind != spec_kind:
        raise SchemaError(
            f"answer_kind: {example.answer_kind!r} does not match the spec's "
            f"{spec_kind!r}")

    _validate_graph(example)

    if example.answer_kind == "yesno":
        words = tuple(yes_no_words) if yes_no_words is not None else _label_words()
        if example.answer not in words:
            raise SchemaError(
                f"answer: {example.answer!r} is not one of the two label words "
                f"{words}; the logit-margin readout scores those two ids only")
    elif example.answer_kind == "smiles":
        _validate_smiles(example.answer)


def _validate_graph(example: Example) -> None:
    """The graph/metadata agreement checks: shape, prompt node, question node."""
    graph = example.graph
    if not isinstance(graph, dict):
        raise SchemaError(f"graph: must be a dict, got {type(graph).__name__}")

    texts = _texts(graph)
    prompt_node = _prompt_node(graph, len(texts))

    num_nodes = graph.get("num_nodes")
    if num_nodes is not None and int(num_nodes) != len(texts):
        raise SchemaError(
            f"graph: num_nodes is {num_nodes} but 'text' has {len(texts)} entries")

    question_node = graph.get("question_node")
    if question_node is not None and question_node != -1:
        if not isinstance(question_node, int) or not 0 <= question_node < len(texts):
            raise SchemaError(
                f"graph: question_node {question_node!r} is not a node index in "
                f"[0, {len(texts)})")
        if texts[question_node] != example.question:
            raise SchemaError(
                f"question: the question node holds {texts[question_node]!r}, not "
                f"{example.question!r}")
    elif example.question not in texts[prompt_node]:
        # No question node (the flat arm is one node; `question_node: off` folds
        # the question into the prompt), so the question has to be in the prompt.
        raise SchemaError(
            f"question: {example.question!r} does not appear in the prompt node's "
            f"text {texts[prompt_node]!r}")

    if not texts[prompt_node].endswith(example.answer):
        raise SchemaError(
            f"answer: {example.answer!r} is not the tail of the prompt node's "
            f"text {texts[prompt_node]!r}")


def _validate_smiles(answer: str) -> None:
    """A ``smiles`` answer parses and is already its own canonicalization.

    Stereo-free canonical form (``isomericSmiles=False``), because the graph
    carries parity words without the neighbour ordering that would give them
    meaning, so stereo cannot be a target for either arm
    (`MOLECULE_GENERALIST.md` §5). A target that is *not* canonical would make
    the exact-match metric a comparison against one arbitrary spelling out of
    many.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(answer)
    if mol is None:
        raise SchemaError(f"answer: {answer!r} does not parse under RDKit")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if answer != canonical:
        raise SchemaError(
            f"answer: {answer!r} is not its own canonicalization "
            f"(RDKit canonical, stereo-free, is {canonical!r})")


def _label_words() -> tuple:
    """The two ``yesno`` label words, from the module that scores them.

    Imported lazily: `molecules/evaluate.py` pulls torch and sklearn, and this
    module has to import on the login node. One source rather than a second
    literal here — a drifted copy would put examples in the mixture that the
    margin readout cannot see.
    """
    from ..experiments.molecules.evaluate import NO_WORD, YES_WORD

    return (YES_WORD, NO_WORD)


# ─────────────────────────────────────────────────────────────────────────────
# Item helpers
# ─────────────────────────────────────────────────────────────────────────────

def _texts(graph: dict) -> list:
    texts = graph.get("text")
    if not isinstance(texts, (list, tuple)) or not texts:
        raise SchemaError("graph: 'text' must be a non-empty list of node strings")
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise SchemaError(f"graph: node {i}'s text is {type(t).__name__}, not str")
    return list(texts)


def _prompt_node(graph: dict, n_nodes: int) -> int:
    prompt_node = graph.get("prompt_node")
    if not isinstance(prompt_node, int) or isinstance(prompt_node, bool):
        raise SchemaError(
            f"graph: prompt_node must be an int, got {prompt_node!r}")
    if not 0 <= prompt_node < n_nodes:
        raise SchemaError(
            f"graph: prompt_node {prompt_node} is not a node index in "
            f"[0, {n_nodes})")
    return prompt_node


def _num_nodes(item: dict) -> int:
    n = item.get("num_nodes")
    if n is not None:
        return int(n)
    texts = item.get("text")
    return len(texts) if isinstance(texts, (list, tuple)) else 0


def _question_from_item(item: dict):
    """The question node's text, or ``None`` when the graph has no question node."""
    texts = item.get("text")
    q = item.get("question_node")
    if isinstance(texts, (list, tuple)) and isinstance(q, int) and 0 <= q < len(texts):
        return texts[q]
    return None


def _answer_from_item(item: dict):
    """The prompt node's tail after the last :data:`ANSWER_PREFIX`, or ``None``."""
    texts = item.get("text")
    p = item.get("prompt_node")
    if not (isinstance(texts, (list, tuple)) and isinstance(p, int)
            and 0 <= p < len(texts)):
        return None
    text = texts[p]
    idx = text.rfind(ANSWER_PREFIX)
    if idx < 0:
        return None
    return text[idx + len(ANSWER_PREFIX):]


def _tokenize(tokenizer, texts, max_length: int) -> list:
    """``TextGraphDataset.tokenize``'s call, so both paths give the same ids."""
    enc = tokenizer(list(texts), padding=False, truncation=True,
                    max_length=max_length, add_special_tokens=False)
    ids = enc["input_ids"]
    return [list(seq) for seq in ids]
