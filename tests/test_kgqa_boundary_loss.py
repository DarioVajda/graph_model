"""
Pin the D4 boundary-token loss re-weighting (evaluate.boundary_weighted_loss).

Contract: weight 1.0 reproduces plain masked-CE exactly; weights are
renormalized so the mean weight over supervised tokens is 1 (a uniform loss
landscape keeps the SAME total loss, only per-token emphasis shifts); -100
positions never contribute; num_items accounting mirrors HF (sum/num_items).
"""

import torch
import pytest

from src.experiments.kgqa.evaluate import boundary_weighted_loss
from src.experiments.kgqa.config import RunConfig

NL = 198
V = 300


def _case(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 9, V, generator=g)
    labels = torch.tensor([
        [-100, -100, 5, NL, 7, NL, 9, 2, -100],
        [-100, 4, NL, 6, 2, -100, -100, -100, -100],
    ])
    return logits, labels


def _plain_ce(logits, labels):
    sl = logits.float()[..., :-1, :].contiguous().view(-1, V)
    st = labels[..., 1:].contiguous().view(-1)
    return torch.nn.functional.cross_entropy(sl, st, ignore_index=-100)


def test_weight_one_equals_plain_ce():
    logits, labels = _case()
    got = boundary_weighted_loss(logits, labels, 1.0, NL)
    assert torch.allclose(got, _plain_ce(logits, labels), atol=1e-6)


def test_uniform_losses_unchanged_by_weighting():
    # If every supervised token has the same CE, renormalization must make the
    # weighted loss equal the unweighted one for ANY weight.
    logits = torch.zeros(1, 8, V)                     # uniform logits -> equal CE
    labels = torch.tensor([[-100, 3, NL, 4, NL, 5, 2, -100]])
    base = boundary_weighted_loss(logits, labels, 1.0, NL)
    for w in (2.0, 4.0, 10.0):
        got = boundary_weighted_loss(logits, labels, w, NL)
        assert torch.allclose(got, base, atol=1e-6), w


def test_weighting_shifts_emphasis_toward_boundaries():
    logits, labels = _case()
    sl = logits.float()[..., :-1, :].contiguous().view(-1, V)
    st = labels[..., 1:].contiguous().view(-1)
    per_tok = torch.nn.functional.cross_entropy(sl, st, ignore_index=-100,
                                                reduction="none")
    mask = st != -100
    nl_mean = per_tok[(st == NL)].mean()
    other_mean = per_tok[mask & (st != NL)].mean()
    got = boundary_weighted_loss(logits, labels, 4.0, NL)
    base = boundary_weighted_loss(logits, labels, 1.0, NL)
    # moving weight toward boundary tokens moves the loss toward their mean CE
    if nl_mean > other_mean:
        assert got > base
    else:
        assert got < base


def test_num_items_accounting():
    logits, labels = _case()
    n_sup = int((labels[..., 1:] != -100).sum())
    per_mean = boundary_weighted_loss(logits, labels, 4.0, NL)
    summed = boundary_weighted_loss(logits, labels, 4.0, NL, num_items_in_batch=n_sup)
    assert torch.allclose(per_mean, summed, atol=1e-6)
    half = boundary_weighted_loss(logits, labels, 4.0, NL, num_items_in_batch=2 * n_sup)
    assert torch.allclose(half * 2, summed, atol=1e-6)


def test_trainer_class_keeps_its_methods():
    # regression: a module-level helper once landed mid-class and silently
    # detached evaluate/set_gen_max_samples (KeyError 'eval_f1' at selection)
    from src.experiments.kgqa.evaluate import KGQAGraphTrainer, PerDatasetEvalMixin
    for m in ("evaluate", "set_gen_max_samples", "compute_loss"):
        assert callable(getattr(KGQAGraphTrainer, m, None)), m
    # evaluate lives in the shared mixin since the multidataset refactor;
    # pin it there so a mid-class helper can't detach it again unnoticed
    assert "evaluate" in PerDatasetEvalMixin.__dict__
    for m in ("set_gen_max_samples", "compute_loss"):
        assert m in KGQAGraphTrainer.__dict__, m


def test_config_validation():
    with pytest.raises(ValueError):
        RunConfig(boundary_loss_weight=0.0).validate()
    with pytest.raises(ValueError):
        RunConfig(boundary_loss_weight=4.0, data_format_version=2).validate()
    RunConfig(boundary_loss_weight=4.0, data_format_version=3).validate()
