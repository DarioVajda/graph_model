"""
Caption metrics for ChEBI-20 (`MOLECULE_GENERALIST.md` §6): BLEU-2/4, ROUGE-L, METEOR.

**Why these are implemented here rather than imported.** The environment has no
`nltk`, no `rouge_score`, no `sacrebleu` and no `evaluate` — checked, not assumed.
The three metrics are a few dozen lines each, they are fully specified in their
papers, and a new dependency on a training image that is otherwise pinned costs
more than the code does. So they are written out, with the definition each one
follows named in its docstring, and pinned by hand-computed cases in
`tests/generalist/test_validators.py`.

**What is not the standard implementation, and how much it matters.** METEOR's
published definition runs three matcher stages in order — exact, Porter stem,
WordNet synonym — and only the first is available without a stemmer and a
lexicon. :func:`meteor` therefore implements the exact-match stage alone, which
is a *lower bound* on the full score. That is fine for the comparison this
project makes (arm 2 against arm 1, both scored the same way) and it is not
comparable to a published MolT5 METEOR; every Tier-C number carries that
disclosure alongside the templated-caption caveat `molecules/PLAN.md` §1 already
attaches to it.

The tokenization is one regex — words and standalone punctuation, lowercased —
rather than Moses. Chemical captions are plain English sentences with formulae in
them ("The molecule is a dicarboxylic acid ..."), so the difference from a Moses
tokenizer is small, but it is a difference, and it is the reason these numbers
belong beside our own and not in a leaderboard row.

No torch, no numpy: pure Python over token lists.
"""

from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["tokenize", "bleu", "rouge_l", "meteor", "caption_metrics"]

#: Words (letters, digits, and the ``-``/``+``/``,`` inside a chemical name kept
#: attached: ``2,3-dihydroxy`` is one token, not five) or a single other
#: non-space character as its own token.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-+,'][a-z0-9]+)*|[^\sa-z0-9]")


def tokenize(text: str) -> list:
    """Lowercase, then words and standalone punctuation. See the module docstring."""
    return _TOKEN.findall((text or "").lower())


# ─────────────────────────────────────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(hypotheses, references, max_n: int = 4) -> float:
    """Corpus BLEU (Papineni et al. 2002), single reference per hypothesis.

    Clipped n-gram counts summed over the corpus, the geometric mean of the
    resulting precisions, times the brevity penalty ``exp(1 - r/c)`` on the
    corpus totals. Corpus-level rather than a mean of sentence BLEUs, which is
    the definition and also the only one that behaves on short captions: a
    sentence with no 4-gram match scores 0 and would drag a mean to nearly zero
    however good the rest are.

    A precision of zero at any order makes the whole score zero, as in the
    original — no smoothing. With ``max_n=2`` this is BLEU-2, with 4 BLEU-4.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"bleu: {len(hypotheses)} hypotheses against {len(references)} "
            "references; they are paired")
    if not hypotheses:
        return 0.0

    matches = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        h = hyp if isinstance(hyp, list) else tokenize(hyp)
        r = ref if isinstance(ref, list) else tokenize(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            h_counts, r_counts = _ngrams(h, n), _ngrams(r, n)
            totals[n] += max(len(h) - n + 1, 0)
            matches[n] += sum(min(c, r_counts[g]) for g, c in h_counts.items())

    log_p = 0.0
    for n in range(1, max_n + 1):
        if totals[n] == 0 or matches[n] == 0:
            return 0.0
        log_p += math.log(matches[n] / totals[n]) / max_n
    penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return float(penalty * math.exp(log_p))


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def _lcs(a, b) -> int:
    """Length of the longest common subsequence. O(len(a) x len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(hypotheses, references) -> float:
    """Mean sentence-level ROUGE-L F1 (Lin 2004), ``beta = 1``.

    Sentence-level and averaged, which is what `rouge_score` reports and what the
    molecule-captioning literature quotes. ``beta = 1`` — the original weights
    recall by ``beta``, and the F-measure with ``beta = 1`` is the variant every
    comparable number uses.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"rouge_l: {len(hypotheses)} hypotheses against {len(references)} "
            "references; they are paired")
    if not hypotheses:
        return 0.0

    scores = []
    for hyp, ref in zip(hypotheses, references):
        h = hyp if isinstance(hyp, list) else tokenize(hyp)
        r = ref if isinstance(ref, list) else tokenize(ref)
        if not h or not r:
            scores.append(0.0)
            continue
        lcs = _lcs(h, r)
        if lcs == 0:
            scores.append(0.0)
            continue
        precision, recall = lcs / len(h), lcs / len(r)
        scores.append(2 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores))


# ─────────────────────────────────────────────────────────────────────────────
# METEOR
# ─────────────────────────────────────────────────────────────────────────────

#: Banerjee & Lavie's tuned parameters: the F-mean's recall weight, the chunk
#: penalty's exponent and its coefficient.
METEOR_ALPHA, METEOR_BETA, METEOR_GAMMA = 0.9, 3.0, 0.5


def _align(hyp, ref) -> list:
    """Greedy exact-match alignment: ``[(hyp index, ref index), ...]``, hyp order.

    Each hypothesis token takes the earliest still-unused identical reference
    token. This is the exact-match stage of METEOR's matcher and is what `nltk`'s
    ``_match_enums`` does; the stem and synonym stages need a stemmer and WordNet
    and are absent here (module docstring).
    """
    used = set()
    by_token: dict = {}
    for j, token in enumerate(ref):
        by_token.setdefault(token, []).append(j)
    alignment = []
    for i, token in enumerate(hyp):
        for j in by_token.get(token, ()):
            if j not in used:
                used.add(j)
                alignment.append((i, j))
                break
    return alignment


def _chunks(alignment) -> int:
    """Contiguous runs in *both* sentences — METEOR's fragmentation count."""
    if not alignment:
        return 0
    count = 1
    for (i0, j0), (i1, j1) in zip(alignment, alignment[1:]):
        if i1 != i0 + 1 or j1 != j0 + 1:
            count += 1
    return count


def meteor(hypotheses, references) -> float:
    """Mean sentence-level METEOR, exact-match stage only (module docstring).

    ``F_mean = P R / (alpha P + (1 - alpha) R)``, penalty
    ``gamma (chunks / matches) ** beta``, score ``F_mean (1 - penalty)``.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"meteor: {len(hypotheses)} hypotheses against {len(references)} "
            "references; they are paired")
    if not hypotheses:
        return 0.0

    scores = []
    for hyp, ref in zip(hypotheses, references):
        h = hyp if isinstance(hyp, list) else tokenize(hyp)
        r = ref if isinstance(ref, list) else tokenize(ref)
        alignment = _align(h, r)
        m = len(alignment)
        if not m:
            scores.append(0.0)
            continue
        precision, recall = m / len(h), m / len(r)
        f_mean = (precision * recall
                  / (METEOR_ALPHA * precision + (1 - METEOR_ALPHA) * recall))
        penalty = METEOR_GAMMA * (_chunks(alignment) / m) ** METEOR_BETA
        scores.append(f_mean * (1 - penalty))
    return float(sum(scores) / len(scores))


def caption_metrics(predictions, targets) -> dict:
    """The four D1.1 caption numbers plus the example count.

    Tokenizes once and reuses the token lists, because ROUGE-L is quadratic in
    the caption length and ChEBI captions run to a hundred tokens.
    """
    predictions, targets = list(predictions), list(targets)
    if len(predictions) != len(targets):
        raise ValueError(
            f"caption_metrics: {len(predictions)} predictions against "
            f"{len(targets)} targets; they are paired")
    if not predictions:
        return {"bleu2": 0.0, "bleu4": 0.0, "rouge_l": 0.0, "meteor": 0.0, "n": 0}

    hyp = [tokenize(p) for p in predictions]
    ref = [tokenize(t) for t in targets]
    return {
        "bleu2": bleu(hyp, ref, max_n=2),
        "bleu4": bleu(hyp, ref, max_n=4),
        "rouge_l": rouge_l(hyp, ref),
        "meteor": meteor(hyp, ref),
        "n": len(hyp),
    }
