"""Few-shot demonstrations, drawn from the accepted set itself (§9.4).

The axis §9.4 withdrew, reinstated on the one condition that withdrawal named.
It was dropped because "a worked example is about a *different* molecule,
nothing computes a fact sheet for that molecule, and so no part of the
demonstration can be verified" — and what came back proved the point, a 4-atom
smallest ring in a molecule with no rings.

That objection is about where the demonstration comes from, not about few-shot.
A demonstration drawn from an **already-accepted row** has a fact sheet,
computed for a real subject from the same pool, and it has already been through
every filter in `pipeline/accept.py`. So the shots are composed after
acceptance, out of the set itself, and every word of every demonstration is
verified to the same standard as the answer it precedes.

The demonstration subjects go into the graph as their own **disconnected
components** — which is also the only honest encoding, since they are different
subjects and an edge between them would be a claim. The prompt node carries a
directed edge to each demonstration node, so the pointer the question makes in
words ("worked examples are attached") is a real edge the structural bias can
read. That puts the target's nodes at distance 1 from the prompt and a
demonstration's at distance 2, behind their own node, so "which subject is the
question about" is answerable from the SPD row rather than only from the text.
How a domain builds those components is its `example_builder`'s business; the
draw below is the same for every domain.
"""

from __future__ import annotations

import random

from .domain import get_domain
from .facts import as_facts, fact_field

#: The share of accepted examples that get demonstrations.
SHOT_FRACTION = 0.22

#: How many demonstrations one example carries, when it carries any. Weighted
#: low: the point is to teach the shape of an answer, and a fourth demonstration
#: costs four more subjects' worth of context for very little of that.
SHOT_COUNTS = (1, 1, 1, 2, 2, 3, 4)

#: How many times one accepted row may serve as somebody else's demonstration.
#: Without a ceiling the selector's preference for a matching family and format
#: concentrates on whichever rows are easiest to match, and a handful of answers
#: would be shown to the model hundreds of times.
SHOT_REUSE_CEILING = 6


def draw_shot_count(rng: random.Random, fraction: float = None) -> int:
    """How many demonstrations this example gets. 0 for most of them."""
    share = SHOT_FRACTION if fraction is None else fraction
    if rng.random() >= share:
        return 0
    return rng.choice(SHOT_COUNTS)


def shot_pointer(rng: random.Random, domain=None) -> str:
    """The line prefixed to a question whose example carries demonstrations.

    Drawn from the domain's `shot_pointers`, several of them for the reason
    every other axis here has several: §9.4 asks for no template recognisable
    across the set, and a fixed preamble on a fifth of the examples is the most
    recognisable template there could be.
    """
    return rng.choice(get_domain(domain).shot_pointers)


def shot_text(question: str, answer: str, index: int, domain=None) -> str:
    """One demonstration, as the text of its node in the graph.

    Numbered, because the demonstrations are a *set* of components and the
    numbering is the only thing that lets an answer refer to one of them. The
    Q/A shape is deliberately not the prompt format the example itself uses:
    a demonstration is quoted material, and formatting it as a second turn
    would give the graph two assistant turns and no way to tell which one is
    being asked for.
    """
    return (f"Example {index + 1} ({get_domain(domain).shot_label})\n"
            f"Q: {question}\nA: {answer}")


def demo_leaks_target(demo_answer: str, target_facts, domain=None) -> bool:
    """Does a demonstration's answer already state one of the target's facts?

    The copy shortcut, and the one way a demonstration can make an example
    easier than the example it demonstrates: a demonstration answering "3" to a
    ring-count question, in front of a target whose ring count is also 3,
    teaches that the answer is whatever the last example answered. The subjects
    differ and the demonstration is true of its own, so nothing here is *wrong*
    — it is just a row the model can get right without reading the graph, which
    is the same defect as a question that states its own answer.
    """
    facts = as_facts(target_facts)
    if not facts:
        return False
    return get_domain(domain).states_any(demo_answer, facts)


def shot_candidates(target, pool, domain=None) -> list:
    """`pool` rows that may demonstrate for `target`, best match first.

    Three hard rules and one preference. A demonstration may not be the target,
    may not be the same subject under a different id (the partition key is the
    identity that matters — two rows about one subject would put the target's
    own structure in its context twice), and may not state one of the target's
    facts. The preference is for a row that shares the target's *format* first
    and a fact *family* second, because a demonstration exists to show the shape
    of the wanted answer, and a JSON demonstration in front of a one-word
    question shows the wrong shape.
    """
    domain = get_domain(domain)
    target_families = {f["family"] for f in target["facts"]}
    target_format = (target.get("brief") or {}).get("format", "")
    out = []
    for row in pool:
        if row["id"] == target["id"] or row["key"] == target["key"]:
            continue
        if demo_leaks_target(row["answer"], target["facts"], domain):
            continue
        families = {f["family"] for f in row["facts"]}
        row_format = (row.get("brief") or {}).get("format", "")
        score = (2 if row_format == target_format else 0) + \
                (1 if families & target_families else 0)
        out.append((score, row))
    out.sort(key=lambda pair: -pair[0])
    return [row for _score, row in out]


def fact_polarity(facts) -> str:
    """``"yes"``, ``"no"``, or ``""`` when a row's yes/no facts do not agree.

    The handle the polarity balance in `pipeline/compose.py` is drawn on. It
    reads the *facts*, not the answer: the facts carry the canonical value
    already, and asking the text would mean re-deriving through the negation
    machinery something that was computed two stages earlier.
    """
    values = {fact_field(f, "value") for f in facts
              if fact_field(f, "kind") == "yesno"}
    return values.pop() if len(values) == 1 else ""


def question_text(row) -> str:
    """The text of a composed row's question node: the pointer, then the exchange.

    The pointer is stored beside the question rather than folded into it so that
    ``question`` stays exactly the string the accept pass verified. A composed
    set therefore re-verifies as the accepted set it came from, and the sentence
    that refers to the attached components is recoverable as its own field. The
    same holds for the turns below: they are assembled here, at the consumer, and
    nothing on the row is rewritten.

    **A `needs_clarification` row is an exchange, and the exchange is the
    question.** Its `question` field holds only the person's opening turn, which
    is underspecified on purpose — "Is that atom part of a ring?" — and the
    clarifying turns that resolve it live in `turns`. Reading `question` alone
    asks the model something that has no answer while showing it one that names
    a part the question never did, which teaches exactly the guess the twist
    exists to teach against. The middle turns are everything between the opening
    and the reply, so a single-turn row is unchanged and needs no transcript
    framing around it.
    """
    pointer = (row.get("pointer") or "").strip()
    question = row["question"]
    middle = (row.get("turns") or [])[1:-1]
    if middle:
        lines = [f"You: {question}"]
        lines += [f"{'Assistant' if t['role'] == 'assistant' else 'You'}: "
                  f"{t['text']}" for t in middle]
        question = "\n".join(lines)
    return f"{pointer}\n{question}" if pointer else question
