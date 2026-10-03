"""Check what the two writer passes returned, filter it, and ship the set.

Step 6 of §9.4's order of work, first half. This is a much smaller program than
the accept pass it replaced, and the reason is the whole point of the redesign:
the old pass had to read a free composition and decide whether it was grounded,
which is an open problem; this one has to decide whether a *transformation*
preserved its input, which is a closed one.

Three checks, in this order:

1. **The person's turn names the ask's anchors and states none of the rendered
   statements.** The writer of the turn was never shown the answer, so a leak here
   is coincidence rather than behaviour — but a question that contains its answer
   is still not an example worth training on. This one check replaces five.
2. **The re-voiced reply preserves the statement set.** Both sides are text the
   pipeline produced, so the failure mode is a dropped or merged clause rather
   than a claim to adjudicate. Under a terse format a statement is preserved by
   its *value* (the number, the yes/no) rather than by its sentence, which is what
   `TERSE_FORMATS` is for.
3. **The format brief is met** — JSON parses and carries the declared skeleton's
   keys, the list has at least two items, the one-word answer is one word.

Then, unchanged from the old pass because they were never the problem: the
held-out-family filter, length bounds, 4-gram deduplication, and a ceiling on how
many rows may share one style brief. With `--judged`, a row the judge marked
unresponsive, unpreserved or additive is refused too, and refused *last*, so the
judge's marginal catch over the patterns is visible in the rejection table.

    src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.pipeline.accept \
        --batches .../v5 --asks .../v5/ask --voiced .../v5/voice \
        --judged .../v5/judged --out .../v5/accepted

**Read the rejections, not just the count.** Five times now a filter here has
been discarding correct rows, and every time the only symptom was a yield that
looked plausible. `analysis/reject_read.py` prints a stratified sample for exactly
that reason.

What is particular to a domain comes from its `Domain` (`--domain`, molecules by
default): how a fact names a part, so an index is required as an anchor and not
read as a value; the domain's own stop words; the families that may not pivot a
claim; the held-out language; and `turn_checks`, the checks on the person's turn
that only a domain can make. The examples in the comments here are molecule rows
because that is where every one of these checks was learned.
"""

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

from ..domain import get_domain

#: In characters. The ceiling is what `mol/assistant` can reproduce at
#: `max_new_tokens 160`; the floor does not apply to a terse format, where "4" is
#: the exactly correct answer and the old pass was throwing it away.
TURN_BOUNDS = (12, 600)
REPLY_BOUNDS = (2, 900)
TERSE_REPLY_FLOOR = 1

#: Jaccard over overlapping word 4-grams of the *person's turn only*, above which
#: a row is a duplicate of one already accepted. Settled at 0.85 on the full
#: build: the similarity histogram has an empty band from 0.80 to 1.00, so
#: anything in it is a coincidence of phrasing rather than a repeat, and reading
#: the 58 near-neighbours the move admits found 53 with different answers. The
#: earlier 0.6 was a guess, and it cost 2,300 rows.
DEDUP_MAX = 0.85
BRIEF_CEILING = 0.05

#: Words too common to carry a statement's identity. A statement is preserved if
#: its content words survive; these are not content. A domain adds its own
#: (`Domain.stop_words` — "atom", "molecule" and the like for molecules).
_STOP = frozenset((
    "the", "this", "that", "it", "its", "has", "have", "is",
    "are", "was", "were", "of", "in", "on", "at", "to", "a", "an", "and", "or",
    "with", "for", "from", "contains", "containing",
    "part", "there", "any", "one", "which", "what",
    # Negation is polarity, not content. Counting it as content let "not" carry
    # a statement's identity, so the clause a statement matched was chosen by
    # whether it said "not" rather than by what it was about.
    "not", "isn", "aren", "doesn", "don", "nor", "none",
    # Connectives. `_CLAUSE` splits on ", " before it reaches " whereas ", so a
    # clause often begins with the connective that introduced it. Counted as
    # content, "whereas" alone kept "whereas atom 21 (O)" from merging into the
    # predicate that follows it, and the subject was then read as unnegated.
    "but", "whereas", "while", "however", "though", "although", "yet", "so",
    "then", "also", "well",
))

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NEGATION = re.compile(
    r"\b(?:not|no|nope|nah|none|never|neither|nor|without|lacks|lacking|absent|"
    r"false|isn|isn't|doesn|doesn't|aren|aren't|don|don't|inactive|zero|n/a|"
    r"negative|unavailable|excluded|non-?[a-z]+)\b")
_AFFIRMATION = re.compile(
    r"\b(?:yes|yep|yeah|yup|true|active|affirmative|correct|indeed)\b")

#: Numbers a re-voicing may spell out. "Two of the molecule's rings exhibit
#: aromaticity" preserves "2 of its rings are aromatic." and the first build
#: refused it.
_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}

#: Where a clause ends, for matching a statement to the part of the reply that
#: answers it. Polarity read over a whole reply is the bug that made one "not"
#: negate every fact in a compound answer.
_CLAUSE = re.compile(r"(?<=[.!?;:])\s+|\n+")

#: A comma or a conjunction is only *sometimes* a clause boundary, so it is
#: applied separately and only where what follows it predicates something. The
#: members of a coordination — "the NR AR LBD, SR p53, or NR AR assays", "the SR
#: ARE and SR ATAD5 assays" — are not clauses, and splitting there scattered one
#: statement's content words over three fragments, none of which then carried
#: enough of it to qualify or any of its polarity.
_SOFT = re.compile(
    r",\s+|\s+(?:and|or|but|whereas|while|however|though|although|yet)\s+",
    re.IGNORECASE)
_PREDICATION = re.compile(
    r"\b(?:is|are|was|were|isn|aren|be|been|being|has|have|had|does|do|did|"
    r"doesn|don|not|no|none|contains|contain|shows|show|lacks|lack|sits|sit|"
    r"belongs|belong|forms|form|measures|measure|carries|carry|yes|true|false|"
    # "yes" was here and its colloquial forms were not, so "Yep, atom 37 (C)
    # isn't part of an ether." had no predicate in its first piece, merged
    # forward, and the answer stopped being a clause of its own.
    r"yep|yup|yeah|nope|nah|correct|right|"
    r"inactive|active|null|unknown)\b", re.IGNORECASE)

#: A list item's marker, for splitting a positional reply.
_LIST_ITEM = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+")


_STOP_BY_DOMAIN: dict = {}


def _stop_words(domain) -> frozenset:
    domain = get_domain(domain)
    words = _STOP_BY_DOMAIN.get(domain.name)
    if words is None:
        words = _STOP_BY_DOMAIN[domain.name] = _STOP | frozenset(domain.stop_words)
    return words


def _content(text: str, domain=None) -> set:
    """Content words, stemmed crudely — "rings" and "ring" are the same claim."""
    stop = _stop_words(domain)
    words = _WORD.findall((text or "").lower())
    out = set()
    for word in words:
        if len(word) < 3 or word in stop:
            continue
        out.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return out


def _normalised(text: str) -> str:
    """One sentence reduced to what it asserts, for comparing two of them.

    Case, the full stop and the run of whitespace are all the difference between
    a claim as the ask writes it and the same claim as a statement writes it, so
    they all go. Nothing else does: this is an identity test between two
    renderings of the same sentence, not a similarity measure.
    """
    return " ".join((text or "").lower().replace(".", " ").split())


def _numbers(text: str) -> list:
    return _NUMBER.findall(text or "")


def _negated(text: str) -> bool:
    return _NEGATION.search((text or "").lower()) is not None


def carries_token(text: str, token: str, domain=None) -> bool:
    """Does this span of reply carry the answer the renderer computed?

    `token` is what `render.answer_token` returned: a count's number, or "yes" /
    "no". Reading the answer out of the *sentence* instead — which is what the
    first build did — turned an assay called `SR ATAD5` into the number 5, an
    atom index into a value, and refused both "Two" for 2 and "rings" for "ring".
    """
    lowered = (text or "").lower()
    if token in ("yes", "no"):
        negative = _negated(lowered)
        positive = _AFFIRMATION.search(lowered) is not None
        if token == "no":
            return negative
        # An affirmative word, or a plain restatement with no negation in it.
        return positive or not negative
    stripped = get_domain(domain).strip_anchors(lowered)
    word = _NUMBER_WORDS.get(token)
    if (re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", stripped) is not None
            or (word is not None
                and re.search(rf"\b{word}\b", stripped) is not None)):
        return True
    # A count of zero is written as an absence, not as a digit: the sheet itself
    # says "Atom 1 (N) is in no ring" for a `ring_size` of 0, and a reply that
    # says "None" or "in no ring" has stated it.
    return token == "0" and _negated(stripped)


#: A short prose reply is a style miss, not a dropped statement. "No activity."
#: answers "It shows no activity against HIV replication." and carries one of its
#: five content words, so the content test alone refuses it.
SHORT_REPLY = 80


def _clauses(reply: str) -> list:
    """The reply split into clauses, with subject fragments merged forward.

    Splitting "atom 21 (O) and atom 3 (C) are not" on "and" leaves "atom 21 (O)"
    as a clause with no predicate, and reading its polarity says the statement
    was not negated. A clause whose only content is an atom reference belongs to
    the clause that follows it.
    """
    out = []
    for hard in _CLAUSE.split(reply or ""):
        if not hard or not hard.strip():
            continue
        pieces = [p.strip() for p in _SOFT.split(hard) if p and p.strip()]
        # Whether a *later* piece in this sentence predicates anything. It is
        # what tells a subject waiting for its verb from the tail of a list:
        # "whereas atom 21 (O)" is followed by "atom 3 (C) are not", so it
        # belongs forward; "SR ATAD5 assays." ends its sentence, so it belongs
        # back with "The molecule is inactive in the SR ARE".
        later, seen = [False] * len(pieces), False
        for i in range(len(pieces) - 1, -1, -1):
            later[i] = seen
            seen = seen or bool(_PREDICATION.search(pieces[i]))

        local, carry = [], ""
        for i, piece in enumerate(pieces):
            if not _PREDICATION.search(piece):
                if later[i]:
                    carry = f"{carry} {piece}".strip()
                elif local:
                    local[-1] = f"{local[-1]}, {piece}"
                else:
                    local.append(f"{carry} {piece}".strip())
                    carry = ""
                continue
            local.append(f"{carry} {piece}".strip() if carry else piece)
            carry = ""
        if carry:
            if local:
                local[-1] = f"{local[-1]} {carry}"
            else:
                local.append(carry)
        out += local
    return out or ([reply.strip()] if (reply or "").strip() else [])


def _threshold(wanted: set, fraction: float = 0.7) -> int:
    """How many of a statement's content words a span has to carry to be about it."""
    return max(1, int(fraction * len(wanted))) if wanted else 0


def _covers(statement: str, reply: str, fraction: float = 0.7,
            domain=None) -> bool:
    """Does the reply as a whole still say what this statement says?

    Used only where the statement has no answer token, which is to say where
    there is no polarity to mis-read and so no reason to narrow to a clause. A
    `caption` statement is a paragraph, and a re-voicing splits and reorders it
    across several sentences: no single clause carries 70% of a paragraph, and
    reading it clause-wise threw away every correctly re-voiced description.
    """
    wanted = _content(statement, domain)
    if not wanted:
        return True
    return len(wanted & _content(reply, domain)) >= _threshold(wanted, fraction)


def candidate_clauses(statement: str, reply: str, domain=None) -> list:
    """The parts of the reply that could be answering this statement.

    A clause qualifies on content words or on the statement's **part index**
    (`Domain.anchor_index`). Both are needed and neither is enough alone: an
    elided answer carries only the atom ("whereas atom 9 (C) is not"), and a
    reply that speaks of the atom by pronoun carries only the words ("the
    smallest ring containing it has 5 atoms").
    """
    wanted = _content(statement, domain)
    index = get_domain(domain).anchor_index(statement.lower())
    need = _threshold(wanted)
    out = []
    for clause in _clauses(reply):
        anchored = bool(index and re.search(
            rf"(?<!\d){index}(?!\d)", clause))
        if anchored or (wanted and len(wanted & _content(clause, domain)) >= need):
            out.append(clause)
    if not out and not wanted and not index:
        return [reply]
    return out


def statement_survives(statement: str, reply: str, terse: bool,
                       token=None, domain=None) -> bool:
    """Is this rendered statement still asserted in this span of reply?

    Under a terse format the span *is* the answer, so only the token has to
    survive. Under prose the reply has to show it is addressing the statement at
    all, and then carry its answer:

    * a **yes/no** is read on the clauses that address the statement, never over
      the whole reply — one "not" in a compound answer used to negate every fact
      in it;
    * a **count** is read over the whole reply, because a number is unambiguous
      wherever it sits and the clause it lands in often speaks of the atom by
      pronoun.
    """
    if terse:
        return token is None or carries_token(reply, token, domain)
    if token is None:
        # Nothing scalar to preserve and no polarity to mis-read, so the span is
        # the whole reply. See `_covers`.
        return _covers(statement, reply, domain=domain)
    candidates = candidate_clauses(statement, reply, domain)
    if not candidates:
        # A reply short enough to be a bare answer has not dropped anything; it
        # has answered tersely under a prose brief, which is a style miss.
        return (len(reply or "") <= SHORT_REPLY
                and carries_token(reply, token, domain))
    if token in ("yes", "no"):
        return any(carries_token(clause, token, domain) for clause in candidates)
    return carries_token(reply, token, domain)


def dropped_statements(rendered, reply: str, fmt: str, domain=None) -> list:
    """The rendered statements this reply no longer makes.

    Three readings, because the three families of format say the same thing in
    three shapes and checking all of them as prose is what made a third of the
    first build's rejections false positives:

    * **terse** — the whole reply is one value. `check_claim` and `decide` answer
      about the *claim*, so what a one-word reply there has to carry is the
      verdict and not the facts behind it.
    * **positional** — the n-th list item answers the n-th statement, in a value
      rather than a sentence: "1. No\\n2. No" states both statements it was
      given, and reading it as prose finds neither.
    * **prose** — the statement's own words, with its polarity read on the clause
      that carries them.
    """
    from ..intents import POSITIONAL_FORMATS, TERSE_FORMATS

    statements = rendered["statements"]
    answers = rendered.get("answers") or [None] * len(statements)
    verdict = rendered.get("verdict")

    if fmt in TERSE_FORMATS:
        if verdict is not None and fmt == "one word":
            # A one-word answer to a false premise may be the verdict ("No") or
            # the true value ("Four"); both answer the question that was asked.
            # Only a *count* is allowed to stand in for the verdict, though: a
            # yes/no fact behind a yes/no verdict is not a second reading of the
            # question, it is the verdict with its polarity flipped.
            if carries_token(reply, verdict, domain):
                return []
            tokens = [t for t in answers if t and t not in ("yes", "no")]
            return [] if tokens and any(carries_token(reply, t, domain)
                                        for t in tokens) else statements[:1]
        wanted = [(s, t) for s, t in zip(statements, answers) if t]
        return [s for s, token in wanted
                if not carries_token(reply, token, domain)]

    if fmt in POSITIONAL_FORMATS:
        items = [i.strip() for i in _LIST_ITEM.split(reply or "") if i.strip()]
        if len(items) < len(statements):
            # Fewer items than statements usually means the *question* bundled
            # two asks into one — "is it in a ring, and if so what size" —
            # and the writer answered them in one item: "2. Yes, 6". The
            # content is all there, so it is read as prose rather than called a
            # drop; every answer token still has to survive.
            return [s for i, s in enumerate(statements)
                    if not statement_survives(s, reply, terse=False,
                                              token=answers[i], domain=domain)]
        # Matched as an assignment rather than by index. The writer of the
        # question is asked to keep the order it was given, but it sometimes
        # reorders, and a reply that answers the question it was actually asked
        # in the order it was actually asked is not a dropped statement.
        free = list(range(len(items)))
        out = []
        for i, statement in enumerate(statements):
            hit = next((j for j in free
                        if statement_survives(statement, items[j],
                                              terse=bool(answers[i]),
                                              token=answers[i],
                                              domain=domain)), None)
            if hit is None:
                out.append(statement)
            else:
                free.remove(hit)
        return out

    return [s for i, s in enumerate(statements)
            if not statement_survives(s, reply, terse=False, token=answers[i],
                                      domain=domain)]


#: A reply that declines. Written from the wordings the writer actually reaches
#: for across every refusal in a build — "I don't have", "unavailable", "No NMR
#: spectrum available", "I don't know", and the JSON `null` — because this is the
#: one check whose two errors are not comparable. A false rejection costs one row
#: of eight hundred; a false acceptance puts an invented melting point in the
#: training set, which is the exact behaviour the `unanswerable` twist exists to
#: teach against. So it is drawn wide and measured, not drawn tight.
_DECLINE = re.compile(
    r"\b(?:do(?:es)?\s+not\s+have|don'?t\s+have|doesn'?t\s+have|"
    r"do(?:es)?\s+not\s+know|don'?t\s+know|doesn'?t\s+know|"
    r"unavailable|not\s+available|no\s+[a-z ]{0,25}available|"
    r"no\s+data|no\s+information|cannot|can'?t|unable|"
    r"isn'?t\s+something|not\s+something|outside|beyond|lacks?|"
    r"null|unknown|not\s+determined|not\s+reported|no\s+record)\b",
    re.IGNORECASE)


def declines(reply: str) -> bool:
    """Does this reply say it cannot answer something?"""
    return _DECLINE.search(reply or "") is not None


#: A reply that opens or closes on the answer word itself — "Yes.", "Nope,",
#: "* Yes", "1. Yes". The leading `\W*` is what lets a list marker or a bullet
#: sit in front of it. Anchored, because the same words appear inside a reply as
#: answers to the *facts*, and only the ends of a reply belong to the decision.
_BARE_VERDICT = re.compile(
    r"^[\W\d]*(?:yes|yeah|yep|yup|no|nope|nah|correct|true|false)\b\s*"
    r"(?:[.,!;:]|$)", re.IGNORECASE)

#: Where a decision stops and its reason begins. The reason restates the facts,
#: and the facts carry their own polarity: "it meets the constraint because atom
#: 11 (C) is not part of an ether" decides yes and reads as no.
_REASON = re.compile(r"\b(?:because|since|given|due to|as the|in that)\b",
                     re.IGNORECASE)

#: What separates a decision from the facts stated alongside it. A negation on
#: the far side of one of these is about a fact, not about the decision.
_CONNECTIVE = re.compile(
    r"\b(?:so|and|but|therefore|thus|hence|however|although|though|while|"
    r"whereas)\b|[,;]", re.IGNORECASE)

#: Words that name the decision itself rather than any of the facts behind it.
#: Deliberately short: a clause has to be *about* the verdict before its polarity
#: is read as the verdict, and no fact sentence talks about constraints.
_VERDICT_WORDS = re.compile(
    r"\b(?:qualif\w*|meets?|met|satisf\w*|criteri\w*|constraints?|"
    r"requirements?|eligible|suitable|fits?)\b", re.IGNORECASE)


def states_the_verdict(rendered, reply: str, fmt: str) -> bool:
    """Does a `decide` reply answer the decision it was asked for?

    A `decide` turn ends in a question none of the statements answers — "does
    this one meet that?" — and the rendered reply answers it in a sentence of
    its own. Three of the smoke build's numbered-list replies stated both facts
    and dropped that sentence, and the positional reading called them complete
    because every statement was present: the check counts statements, and the
    verdict is not one.

    The rule is not "find the verdict clause". Four rewrites went that way and
    each one picked the wrong clause on some real reply — a fact answered in two
    words ("no ether"), a list item that opens with "No sulfonamide", the reason
    trailing the decision, the correction owed to a false premise. What survives
    is weaker and holds: **collect every span that bears on the decision and ask
    whether one of them agrees with the verdict.** A reply that decides
    correctly has such a span; one that never decided has none; and one that
    decided the other way has spans that all disagree.
    """
    from ..intents import POSITIONAL_FORMATS

    verdict = rendered.get("verdict")
    if verdict is None:
        return True
    spare = True
    if fmt in POSITIONAL_FORMATS:
        # Counted in *lines*, not in list items: "1. 6\n2. 6\nYes." answers two
        # statements and then decides on a line of its own, and that line
        # carries no list marker, so splitting on markers found two items, saw
        # no room for a verdict, and called the decision missing. Without a
        # spare line the ends belong to the statements, not to the decision.
        lines = [l for l in (reply or "").splitlines() if l.strip()]
        spare = len(lines) > len(rendered["statements"])
    # A bare list marker splits off as a clause of its own — "1. Yes" comes back
    # as ["1.", "Yes"] — so the ends are taken from the clauses carrying words.
    clauses = [c for c in _clauses(reply) if re.search(r"[A-Za-z]", c)]
    if not clauses:
        return False

    spans = []
    if spare:
        spans += [clauses[0], clauses[-1]]
    for i, clause in enumerate(clauses):
        match = _VERDICT_WORDS.search(clause)
        if not match:
            continue
        # The decision word, what immediately precedes it, and no further than
        # its reason. The prefix matters because the writer negates in front of
        # the word — "that one doesn't qualify" — and it is cut at the nearest
        # connective because a negation on the far side of one belongs to a
        # fact: "It contains no nitro group, ... so it meets the constraint."
        tail = _CONNECTIVE.split(clause[:match.start()])[-1]
        window = tail + _REASON.split(clause[match.start():])[0]
        spans.append(window)
        if i + 1 < len(clauses):
            spans.append(f"{window} {clauses[i + 1]}")
    return any(_decides(span) == verdict for span in spans)


def _decides(span: str):
    """The decision this span states, or None if it states none.

    A negation anywhere in the span carries it, because the writer negates the
    decision rather than restating it ("it does not qualify", "Constraint met:
    No"). Failing that, an affirmation or the decision word standing unnegated
    is a yes — "* Qualifies" is an answer even with no yes in it. Anything else
    is not evidence either way, which matters: treating "no signal" as a yes is
    what let "No, that one doesn't qualify" pass as a yes verdict.
    """
    if not (_BARE_VERDICT.match(span) or _VERDICT_WORDS.search(span)):
        return None
    lowered = span.lower()
    if _NEGATION.search(lowered):
        return "no"
    if _AFFIRMATION.search(lowered) or _VERDICT_WORDS.search(span):
        return "yes"
    return None


#: How much of the gloss's vocabulary a reply has to carry. Looser than the 0.7 a
#: statement is held to, and measured rather than guessed: swept over 670 replies
#: written before the gloss was named and the re-voiced replies written after, the
#: share admitted goes 0.63, 0.27, 0.085, 0.079, 0.073 as the fraction rises from
#: 0.3 to 0.7, while the share of *told* writers admitted stays at 1.000 until 0.7,
#: where it falls to 0.962. So the separation is flat across 0.5-0.6 and the only
#: thing 0.7 buys is a rejected row that had in fact kept the gloss, compressed —
#: "Generally, an atom's part of a functional group if it's in the group's
#: substructure." 0.5 is the loose edge of that plateau, which is the right side to
#: sit on: a gloss has no polarity to invert and the writer is re-voicing a fixed
#: sentence, so a loose check admits a vague paraphrase while a tight one throws
#: away a correct row.
GLOSS_COVERAGE = 0.5


def states_the_gloss(rendered, reply: str, domain=None) -> bool:
    """Does an `explain` reply still carry the general explanation it owes?

    Read with `_covers` rather than clause-wise, for the reason `_covers` exists:
    a gloss has no polarity to mis-read and no atom to anchor on, and a
    re-voicing spreads it over a sentence of its own words — "an aromatic ring
    is a ring system whose electrons are delocalised" for "In general, an
    aromatic ring is a cyclic system with delocalised pi electrons". What has to
    survive is the concept, not the phrasing.

    Terse formats have no room for it and never draw it: `explain` is not in
    `ONE_WORD_TASKS` and renders no skeleton, so there is no exemption to write.
    """
    gloss = rendered.get("gloss")
    return not gloss or _covers(gloss, reply, GLOSS_COVERAGE, domain)


def premise_is_not_false(rendered) -> bool:
    """Does a `false_premise` ask assert something the statements confirm?

    It should be impossible, and it was not. `render._negate` on a count swaps
    the fact's own digit into the fact's own sentence, and a `ring_size` of 0 is
    spelled "atom 24 (O) is in no ring" with no digit in it — so the substitution
    was a silent no-op and the claim came out as the statement verbatim. The
    reply then told a person their correct belief was wrong, which is the worst
    row this pipeline can produce: confidently, computedly false.

    An identity test and not a similarity one. The claim and the statement are
    two renderings of the same sentence when this fires, differing only in case
    and the full stop, so anything looser would start rejecting a `false_premise`
    that happens to share its wording with a second fact on the sheet.
    """
    if rendered.get("ask", {}).get("claim_is_true") is not False:
        return False
    claim = _normalised(rendered["ask"].get("claim") or "")
    return bool(claim) and any(claim == _normalised(statement)
                               for statement in rendered.get("statements", []))


def pivot_is_unassertable(intent, domain=None) -> bool:
    """Is this row's claim or constraint the whole subject written out?

    A `check_claim` or `decide` pivot goes into the person's turn verbatim, so it
    has to be something a person could assert and the reply could settle in a few
    words. A molecule's canonical SMILES is neither, and a row built on one puts
    its own answer in its question. `Domain.unpivotable` is the list; this is the
    same rule applied to rows built before that list existed.
    """
    facts = intent.get("facts") or []
    return (intent.get("task") in ("check_claim", "decide") and bool(facts)
            and facts[0].get("family") in get_domain(domain).unpivotable)


#: Tasks whose person states the answer *by design*: `check_claim` puts the claim
#: in the person's mouth and `decide` puts the constraint there. Testing those for
#: a leak is testing the renderer's own design and rejecting it.
_STATES_BY_DESIGN = frozenset(("check_claim", "decide"))

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _leaked_statements(rendered, turn: str, task: str, fmt: str = "",
                       domain=None) -> list:
    """Statements the person's turn gives away, which should be none.

    Narrow to the point of being almost trivial, and deliberately so. The writer
    of the turn was never shown the reply, so this is a guard against coincidence
    rather than against a behaviour — and the question and the answer share every
    content word *by construction*, because `ask_phrase` is the statement with
    its polarity removed. Any check that reads words therefore rejects every
    well-formed question in the set, which is what the first build did: all five
    `turn_states_the_answer` rejections read by hand were list-format questions
    enumerating the atoms they were asking about.

    So only a **count** leaks here: the writer was never told the number, so a
    number that matches the answer is a leak or a coincidence and neither belongs
    in the set. A yes/no leak is an assertion rather than a word, and that is the
    judge's to catch.

    `check_claim` and `decide` state the value in the person's mouth by design
    and are exempt.
    """
    if task in _STATES_BY_DESIGN:
        return []
    # The format request is part of the brief, not something the writer knew:
    # "Please answer in one word" leaked the count 1 through the number word
    # "one", and a rendered JSON schema leaked the atom index and the size
    # through its own key, `ring_size_atom_5`.
    stripped = " " + (turn or "")
    stripped = re.sub(r"\{[^{}]*\}", " ", stripped)
    for phrase in (fmt, "one word"):
        if phrase:
            stripped = re.sub(re.escape(phrase), " ", stripped,
                              flags=re.IGNORECASE)
    # Part indices and list markers are not answers; leaving them in made
    # "1. atom 5 (C)" leak the count 1 and the atom 5.
    domain = get_domain(domain)
    stripped = _LIST_ITEM.sub(" ", stripped)
    stripped = domain.strip_anchors(stripped)
    if domain.anchor_key_re is not None:
        stripped = domain.anchor_key_re.sub(" ", stripped)
    out = []
    for statement, token in zip(rendered["statements"],
                                rendered.get("answers")
                                or [None] * len(rendered["statements"])):
        if token is None or token in ("yes", "no"):
            continue
        if carries_token(stripped, token, domain):
            out.append(statement)
    return out


def format_met(reply: str, fmt: str, skeleton) -> bool:
    """The declared format, checked against the text that came back."""
    if fmt in ("prose", "lead with the answer, then the detail"):
        return True
    if fmt == "one word":
        return len(reply.strip().strip(".").split()) == 1
    if fmt.startswith("JSON"):
        span = _json_span(reply)
        try:
            payload = json.loads(span)
        except (ValueError, TypeError):
            return False
        return not skeleton or set(skeleton) <= set(payload or {})
    if fmt == "a numbered list":
        return len(re.findall(r"(?:^|\s)\d+[.)]\s+", reply)) >= 2
    if fmt == "a bulleted list":
        return len(re.findall(r"(?:^|\n)\s*[-*•]\s+", reply)) >= 2
    return True


def read_order(paths) -> list:
    """Batch files in reading order: the test-role ones first, then sorted.

    Deduplication is greedy and first-wins against a growing pool, so whichever
    split is read first keeps its rows and the other sheds the collisions. The
    two splits are not equally replaceable: the molecule pool holds 44,088
    train-role against 5,503 test-role, and the test split is the measurement.
    In plain sorted order a top-up build's `b-train-*` batches come before
    `test-*` and take the slots, which cost the test split 94 rows against its
    target while the train split sat 700 over its own.

    Rejecting a test row that repeats a train row is still right — it is leakage
    on the question text. It is only the order that decides which side pays for
    it, and the side with the surplus should.
    """
    return sorted(paths, key=lambda p: ("test-" not in os.path.basename(p), p))


def _json_span(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def _load_jsonl(directory: str, field: str) -> dict:
    out = {}
    if not directory:
        return out
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if field in row:
                        out[row["id"]] = row
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", required=True)
    parser.add_argument("--asks", required=True)
    parser.add_argument("--voiced", required=True)
    parser.add_argument("--judged", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dedup-max", type=float, default=DEDUP_MAX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default=None,
                        help="the assistant domain the batches were built for "
                             "(default: molecules)")
    args = parser.parse_args(argv)
    domain = get_domain(args.domain)

    from ..facts import four_grams, jaccard
    from ..intents import TERSE_FORMATS

    turns = _load_jsonl(args.asks, "turn")
    replies = _load_jsonl(args.voiced, "reply")
    verdicts = _load_jsonl(args.judged, "preserved") if args.judged else {}

    os.makedirs(args.out, exist_ok=True)
    rejects = collections.Counter()
    reject_rows = []
    accepted = []
    seen_grams = []
    written = 0

    def reject(example, reason, detail=""):
        rejects[reason] += 1
        reject_rows.append({"id": example["id"], "reason": reason,
                            "detail": detail, "cell": example["cell"],
                            "turn": turns.get(example["id"], {}).get("turn", ""),
                            "reply": replies.get(example["id"], {}).get("reply", "")})

    for path in read_order(glob.glob(os.path.join(args.batches,
                                                  "*batch-*.json"))):
        with open(path) as handle:
            batch = json.load(handle)
        for example in batch["examples"]:
            identifier = example["id"]
            if identifier not in turns or identifier not in replies:
                rejects["not_written"] += 1
                continue
            written += 1
            turn = turns[identifier]["turn"]
            reply = replies[identifier]["reply"]
            rendered = example["render"]
            fmt = example["intent"]["style"]["format"]
            terse = fmt in TERSE_FORMATS

            # 1 — the turn names what was asked for and gives nothing away
            missing = [a for a in rendered["ask"].get("anchors", [])
                       if not _anchor_named(a, turn, domain)]
            if missing and not rendered["ask"].get("underspecified"):
                reject(example, "anchor_missing", "; ".join(missing))
                continue
            if rendered["ask"].get("underspecified") and rendered["ask"].get("anchor") \
                    and _anchor_named(rendered["ask"]["anchor"], turn, domain):
                reject(example, "clarification_not_ambiguous",
                       rendered["ask"]["anchor"])
                continue
            leaked = _leaked_statements(rendered, turn,
                                        example["intent"]["task"], fmt, domain)
            if leaked:
                reject(example, "turn_states_the_answer", leaked[0])
                continue
            licensed = " ".join([rendered["ask"].get("claim") or ""]
                                + rendered["statements"] + [rendered["reply"]])
            failed = None
            for reason, check in domain.turn_checks:
                detail = check(turn, example["intent"], licensed)
                if detail is not None:
                    failed = (reason, detail)
                    break
            if failed:
                reject(example, *failed)
                continue
            # The sheet writes a count as "1 stereocenter(s)", and both the
            # `false_premise` claim and the draft reply are the sheet's own
            # sentences handed to the writer, so the spelling survives into the
            # person's mouth — "I believe 1 stereocenter(s) have a defined
            # configuration" — and into the assistant's: "it contains 3
            # amide(s)". Nobody writes either. It is 0.6 % of turns and 0.6 % of
            # replies, cheaper to drop than to re-render; the renderer should
            # stop producing the spelling (§9.4).
            if "(s)" in turn or "(s)" in reply:
                reject(example, "reads_like_a_data_sheet", "(s)")
                continue

            # 2 — the re-voicing preserved the statement set
            dropped = dropped_statements(rendered, reply, fmt, domain)
            if dropped:
                reject(example, "statement_dropped", dropped[0])
                continue
            # The verdict is not a statement, so the check above cannot miss it
            # and cannot see it either. A terse reply *is* the verdict and is
            # already checked as one.
            if example["intent"]["task"] == "decide" and not terse \
                    and not states_the_verdict(rendered, reply, fmt):
                reject(example, "verdict_dropped", rendered.get("verdict") or "")
                continue
            # Nor is the refusal a statement. `answerable: False` reached the
            # voice prompt and the judge prompt and stopped there, so nothing
            # verified that a row built to decline actually declined — and a row
            # that answers instead teaches the model to invent the property the
            # twist exists to refuse.
            if rendered["ask"].get("answerable") is False and not declines(reply):
                reject(example, "refusal_dropped",
                       rendered["ask"].get("missing") or "")
                continue
            # And neither is the gloss. It is the only thing separating `explain`
            # from `report`, and without this check 92% of explain rows shipped
            # without one.
            if not states_the_gloss(rendered, reply, domain):
                reject(example, "gloss_dropped", rendered.get("gloss", "")[:40])
                continue
            # A `decide` ask asks two questions with two different answers — what
            # the value is and whether it meets the constraint — and one word can
            # carry only one of them. The render gives the verdict, so the word
            # is the answer to the question that was not asked first, and where
            # the constraint runs against the value it reads as its opposite:
            # "whether atom 6 (C) is part of a halogen" answered "Yes" over a
            # statement saying it is not. `decide` is off `ONE_WORD_TASKS` now,
            # and this stops the rows drawn before it was.
            if fmt == "one word" and example["intent"]["task"] == "decide":
                reject(example, "one_word_over_two_questions", "")
                continue
            # A false premise that is not false. `_negate` on a count swaps the
            # digit in the fact's own sentence, and `ring_size` 0 reads "atom 24
            # (O) is in no ring" with no digit in it, so the claim came out as
            # the statement verbatim and the reply then told a person their
            # correct belief was wrong. The renderer no longer produces it; these
            # are the rows that were built before it stopped.
            if premise_is_not_false(rendered):
                reject(example, "premise_is_not_false",
                       (rendered["ask"].get("claim") or "")[:40])
                continue
            # A claim or a constraint that is the whole subject written out.
            # "I believe its canonical SMILES is COc1ccccc1N1C(=O)..." puts the
            # answer in the question and the reply copies it back, so the row
            # needs no graph. `Domain.unpivotable` stops the draw.
            if pivot_is_unassertable(example["intent"], domain):
                reject(example, "pivot_is_the_structure",
                       example["intent"]["facts"][0]["family"])
                continue

            # 3 — the format brief
            if not format_met(reply, fmt, rendered.get("skeleton")):
                reject(example, "format", fmt)
                continue

            if domain.mentions_held_out(turn) or domain.mentions_held_out(reply):
                reject(example, "held_out_language")
                continue

            low, high = TURN_BOUNDS
            if not low <= len(turn) <= high:
                reject(example, "turn_length", str(len(turn)))
                continue
            low, high = REPLY_BOUNDS
            if terse:
                low = TERSE_REPLY_FLOOR
            if not low <= len(reply) <= high:
                reject(example, "reply_length", str(len(reply)))
                continue

            grams = four_grams(turn)
            if any(jaccard(grams, other) > args.dedup_max
                   for other in seen_grams):
                reject(example, "duplicate")
                continue

            if verdicts:
                verdict = verdicts.get(identifier)
                if verdict is None:
                    reject(example, "unjudged")
                    continue
                if not verdict.get("responsive"):
                    reject(example, "judge_unresponsive", verdict.get("note", ""))
                    continue
                if not verdict.get("preserved"):
                    reject(example, "judge_dropped", verdict.get("note", ""))
                    continue
                if verdict.get("added"):
                    reject(example, "judge_added", verdict.get("note", ""))
                    continue

            seen_grams.append(grams)
            accepted.append({
                "id": identifier, "key": example["key"], "role": example["role"],
                "source": example["source"], "cell": example["cell"],
                "smiles": example["intent"]["smiles"],
                "task": example["intent"]["task"],
                "twist": example["intent"]["twist"],
                # `brief` and not `style`: `compose.py` ranks candidate
                # demonstrations on `brief["format"]`, and a JSON demonstration
                # in front of a one-word question shows the wrong shape.
                "brief": example["intent"]["style"],
                "situation": example["intent"]["situation"]["id"],
                "facts": example["intent"]["facts"],
                "statements": rendered["statements"],
                # `answers`, `verdict` and `gloss` travel with the row because
                # they are what preservation is checked against, and the composed
                # set is re-checked after the demonstrations are attached.
                "answers": rendered.get("answers"),
                "verdict": rendered.get("verdict"),
                "gloss": rendered.get("gloss"),
                "skeleton": rendered.get("skeleton"),
                "rendered_reply": rendered["reply"],
                "turns": rendered["turns"],
                "question": turn, "answer": reply,
                "writer": replies[identifier].get("writer", ""),
            })

    accepted = _apply_brief_ceiling(accepted, args.seed, rejects)

    by_role = collections.defaultdict(list)
    for row in accepted:
        by_role[row["role"]].append(row)
    for role, rows in sorted(by_role.items()):
        with open(os.path.join(args.out, f"{role}.jsonl"), "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(args.out, "rejected.jsonl"), "w") as handle:
        for row in reject_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "written": written, "accepted": len(accepted),
        "yield": round(len(accepted) / max(written, 1), 4),
        "rejected": dict(rejects.most_common()),
        "by_role": {r: len(v) for r, v in sorted(by_role.items())},
        "by_task": dict(collections.Counter(r["task"] for r in accepted)),
        "by_twist": dict(collections.Counter(r["twist"] for r in accepted)),
        "by_format": dict(collections.Counter(r["brief"]["format"]
                                              for r in accepted)),
        "judged": bool(verdicts),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)

    print(f"{len(accepted)}/{written} accepted ({summary['yield']:.3f})")
    for reason, n in rejects.most_common():
        print(f"  {reason:28} {n:6}  {n / max(written, 1):.3f}")
    print("\nby task  :", summary["by_task"])
    print("by twist :", summary["by_twist"])
    print("by format:", summary["by_format"])
    return 0


def _anchor_named(anchor: str, turn: str, domain=None) -> bool:
    """Is this anchor named in the turn — by index for a part, by word else?

    "atom 14 (C)" may reasonably be typed "atom 14", "C14" or "the carbon at
    position 14", so what is required is the index, not the rendering. A
    qualifier anchor (a functional group) is required by its name.
    """
    lowered = (turn or "").lower()
    number = get_domain(domain).anchor_index(anchor.lower())
    if number:
        # `\b14\b` does not match "C14": 'c' and '1' are both word characters, so
        # there is no boundary between them, and the commonest way a chemist
        # writes an atom would have been refused as a missing anchor.
        return re.search(rf"(?<!\d){number}(?!\d)", lowered) is not None
    return all(word in lowered for word in _content(anchor, domain))


def _apply_brief_ceiling(accepted, seed, rejects):
    """No style brief may own more than `BRIEF_CEILING` of the shipped set."""
    if not accepted:
        return accepted
    rng = random.Random(seed)
    order = list(accepted)
    rng.shuffle(order)
    cap = max(1, int(BRIEF_CEILING * len(order)))
    kept, counts = [], collections.Counter()
    for row in order:
        brief = json.dumps(row["brief"], sort_keys=True)
        if counts[brief] >= cap:
            rejects["brief_ceiling"] += 1
            continue
        counts[brief] += 1
        kept.append(row)
    kept.sort(key=lambda r: r["id"])
    return kept


if __name__ == "__main__":
    sys.exit(main())
