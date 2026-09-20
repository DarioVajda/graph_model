"""
Per-example loss normalisation: what it changes, and what it must not.

`PLAN.md` §9 — every instrument gets a test that fails when the instrument is
wrong. The instrument here decides how much of the gradient a long caption gets,
and the failure it guards is silent: a wrong normalisation trains a perfectly
ordinary-looking model that writes at the wrong length.

The quantity is `generalist/mixture.py::MixtureLoss`'s `per_example` — each
example's summed token loss divided by its OWN supervised span, then meaned over
examples, so a 400-token caption and a 40-token one contribute equally.
"""

import pytest
import torch

from src.experiments.molecules.loss import (
    LOSS_NORMS, PerExampleLossTrainer, trainer_class_for)
from src.utils import GraphTrainerV2


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Model:
    """Returns fixed logits; the trainer supplies the normalisation under test."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, **kwargs):
        return _Out(self._logits)


def _loss(logits, labels):
    trainer = PerExampleLossTrainer.__new__(PerExampleLossTrainer)
    return trainer.compute_loss(_Model(logits), {"labels": labels})


def _per_token(logits, labels):
    """HF's own quantity, for comparison: one mean over every supervised token."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=-100)


def test_equal_spans_make_the_two_normalisations_agree():
    """With every example the same length there is nothing to re-weight."""
    torch.manual_seed(0)
    logits = torch.randn(4, 9, 7)
    labels = torch.randint(0, 7, (4, 9))

    assert _loss(logits, labels).item() == pytest.approx(
        _per_token(logits, labels).item(), rel=1e-5)


def test_THE_POINT_a_long_example_stops_dominating():
    """One long, badly-predicted example against three short, well-predicted ones.

    Under `per_token` the long example owns most of the supervised positions and
    therefore most of the loss. Under `per_example` it is one voice in four.
    """
    torch.manual_seed(0)
    V, L = 7, 21
    logits = torch.zeros(4, L, V)
    labels = torch.full((4, L), -100)
    # three short examples, 2 supervised tokens each, confidently correct
    for i in range(3):
        labels[i, 1:3] = 0
        logits[i, 0:2, 0] = 10.0
    # one long example, 19 supervised tokens, confidently WRONG
    labels[3, 1:20] = 0
    logits[3, 0:19, 1] = 10.0

    per_ex = _loss(logits, labels).item()
    per_tok = _per_token(logits, labels).item()

    assert per_tok > per_ex, (
        "per_token must weight the long wrong example more heavily")
    # It is one of four examples, so it carries about a quarter of the loss,
    # against roughly 19/25 of the supervised tokens.
    assert per_ex == pytest.approx(10.0 / 4, rel=0.05)


def test_an_example_with_no_supervised_token_is_not_counted():
    """It must contribute nothing, and must not divide by zero getting there."""
    torch.manual_seed(0)
    V, L = 5, 7
    logits = torch.zeros(2, L, V)
    labels = torch.full((2, L), -100)
    labels[0, 1:4] = 0
    logits[0, 0:3, 1] = 10.0          # wrong, so a clearly nonzero loss

    loss = _loss(logits, labels)

    assert torch.isfinite(loss)
    # Only row 0 is counted, so the mean over examples equals row 0's own loss.
    solo_logits, solo_labels = logits[:1], labels[:1]
    assert loss.item() == pytest.approx(_loss(solo_logits, solo_labels).item(),
                                        rel=1e-5)


def test_the_label_shift_matches_the_codebase_convention():
    """Logit at t predicts token t+1, as `shift_logits_for_metrics` records.

    A trainer that shifted the other way would score every position against its
    neighbour and still return a plausible number.
    """
    V, L = 6, 5
    logits = torch.zeros(1, L, V)
    labels = torch.full((1, L), -100)
    labels[0, 3] = 2
    logits[0, 2, 2] = 20.0           # position 2 predicts label at 3 -> ~0 loss

    assert _loss(logits, labels).item() == pytest.approx(0.0, abs=1e-4)

    off_by_one = torch.zeros(1, L, V)
    off_by_one[0, 3, 2] = 20.0       # the wrong position
    assert _loss(off_by_one, labels).item() > 1.0


def test_the_selector_refuses_an_unknown_normalisation():
    """A typo must not silently fall back to HF's default under a false record."""
    assert trainer_class_for("per_example") is PerExampleLossTrainer
    assert trainer_class_for("per_token") is GraphTrainerV2
    with pytest.raises(ValueError, match="loss_norm"):
        trainer_class_for("per_exmaple")


def test_the_default_is_still_per_token():
    """Every pre-2026-09-19 result was trained under it; flipping the default
    would re-interpret them without re-running them."""
    from src.experiments.molecules.config import RunConfig

    assert RunConfig(task="chebi20", arm="graph").loss_norm == "per_token"
    assert LOSS_NORMS[0] == "per_token"
