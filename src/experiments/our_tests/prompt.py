"""
Prompt-node construction and answer-span label masking, shared by both tasks.

``family_tree_prep.py`` and ``data_prep.py`` build different graphs but attach the
question/answer identically, and each carried its own copy of the label masker. Both
now come from here, so the ``question_node`` feature is implemented once.

THE TARGET-NODE INVARIANT (TODO.md §2)
--------------------------------------
The node the answer is generated in must never be empty before the answer. If it were,
the first answer token would have no in-node predecessor to be predicted from, leaving
it ambiguous which token conditions generation. ``node_texts`` therefore always leaves
``ANSWER_PREFIX`` in the prompt node as that anchor — in both modes. The supervised span
is byte-identical between modes; only the *visibility* of the question changes, which is
what makes the off/isolated comparison a clean probe of one thing.
"""

# The answer delimiter. The supervised span starts just past this marker. Tokenized via
# the tokenizer rather than hardcoded as ids (the historical code used the literal
# ``[32, 25]``): ids are model-specific, and a silently wrong literal masks the wrong
# span rather than failing.
ANSWER_PREFIX = "A:"


def node_texts(question, answer, question_node="off"):
    """Return ``(question_node_text | None, prompt_node_text)``.

    ``question_node``:

    * ``"off"``      — one node carrying ``"Q: {q}\\nA: {a}"``. Byte-identical to the
      historical layout, so datasets rebuilt in this mode match the ones the paper's
      numbers came from.
    * ``"isolated"`` — the question moves to its own (edge-free) node; the prompt node
      keeps ``"A: {a}"``. Graph tokens can then attend to the question through the
      bidirectional prefix mask.

    Unlike graphqa's port of this feature — which has only a pre-joined
    ``task_description`` and must ``partition`` it back apart — both call sites here
    hold ``question`` and ``answer`` separately, so the split is done by construction.
    That removes the failure mode where a question or answer containing ``"A:"`` splits
    at the wrong place.
    """
    if question_node == "off":
        return None, f"Q: {question}\n{ANSWER_PREFIX} {answer}"
    if question_node == "isolated":
        return f"Q: {question}", f"{ANSWER_PREFIX} {answer}"
    raise ValueError(f"Unknown question_node {question_node!r} (expected 'off' or 'isolated').")


class GetGraphLabels:
    """Mask every token up to and including ``question_end`` to -100.

    ``occurrence`` picks which match delimits the answer when ``question_end`` appears
    more than once in the prompt node:

    * ``"last"``  — the historical behaviour, and correct for ``question_node="off"``:
      the node holds ``"Q: {q}\\nA: {a}"``, so a question containing "A:" would
      otherwise win over the real delimiter.
    * ``"first"`` — correct for ``question_node="isolated"``: the node *starts* with the
      delimiter, so the first match is the right one by construction, and an answer
      containing "A:" must not be able to shift the boundary into the answer.

    Getting this wrong is silent — it mis-masks rather than raising — so the mode picks
    the policy explicitly instead of defaulting.
    """

    def __init__(self, question_end, tokenizer=None, occurrence="last"):
        if question_end is None:
            raise ValueError(
                "question_end cannot be None; it is the token-id sequence delimiting "
                "the answer span (encode ANSWER_PREFIX with the run's tokenizer).")
        if occurrence not in ("first", "last"):
            raise ValueError(f"occurrence must be 'first' or 'last', got {occurrence!r}.")
        self.question_end = list(question_end)
        self.tokenizer = tokenizer
        self.occurrence = occurrence

    def __call__(self, example):
        prompt_node = example.get("prompt_node", None)
        input_ids = example["input_ids"][prompt_node]
        labels = list(input_ids)
        n = len(self.question_end)

        end_index = None
        for i in range(len(input_ids) - n + 1):
            if list(input_ids[i:i + n]) == self.question_end:
                end_index = i + n - 1
                if self.occurrence == "first":
                    break

        if end_index is None:
            if self.tokenizer is not None:
                for token_id in input_ids:
                    print(f"{token_id}: {self.tokenizer.decode([token_id]).replace(' ', '_')}")
            raise ValueError(
                f"Could not find the answer delimiter {self.question_end} in the prompt "
                f"node's input_ids: {input_ids}")

        # The delimiter must leave at least one supervised token behind it, otherwise
        # this graph would contribute no training signal and (at generation time) the
        # answer would start from an empty node — the invariant this module exists to
        # keep. Failing here is far cheaper than discovering it as a silently degenerate
        # training run.
        if end_index >= len(labels) - 1:
            raise ValueError(
                f"The answer span is empty: the delimiter {self.question_end} ends at "
                f"index {end_index} of {len(labels)} tokens, leaving nothing supervised.")

        for i in range(end_index + 1):
            labels[i] = -100
        return labels


def label_masker(tokenizer, question_node="off"):
    """Build the label masker for a run, with the delimiter derived from ``tokenizer``."""
    question_end = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    occurrence = "first" if question_node == "isolated" else "last"
    return GetGraphLabels(question_end=question_end, tokenizer=tokenizer,
                          occurrence=occurrence)
