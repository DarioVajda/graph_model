"""
Caption metrics for ChEBI-20 — re-exported from the molecules package.

The implementation moved to `src/experiments/molecules/captions.py`. ChEBI-20 is
Tier C of the molecules benchmark suite (`molecules/PLAN.md` §1) and its
specialist trains in that package, so the metric belongs on the same layer as the
data. The generalist adapter already depends on `experiments.molecules`, never
the other way round, so importing down is the direction the layering allows.

Nothing about the definitions changed: this module exists so that
`evaluate/scorers.py` and every test that imports `caption_metrics` from here
keep working, and so there is exactly one implementation to pin.

**What these are and are not.** BLEU and ROUGE-L follow their papers; METEOR is
the exact-match stage only, so it is a lower bound on the published definition.
These numbers are comparable between our own arms and **not** comparable to a
MolT5 number — for that, `molecules/chebi_lit_metrics.py` rescores dumped
predictions under the reference implementation's protocol.
"""

from ...experiments.molecules.captions import (  # noqa: F401
    bleu,
    caption_metrics,
    meteor,
    rouge_l,
    tokenize,
)

__all__ = ["tokenize", "bleu", "rouge_l", "meteor", "caption_metrics"]
