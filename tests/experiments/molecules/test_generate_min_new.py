"""
The stop token is forbidden for the first few steps, and the defect if it is not.

`PLAN.md` §9: every instrument gets a test that fails when the instrument is
wrong, including one that reproduces the defect.

The defect, 2026-09-18: greedy decoding was free to choose EOS at the very first
generated position, which decodes under ``skip_special_tokens=True`` to the empty
string and scores as a total miss. `043` (instruct, WSD) lost 189 / 375 / 788
captions of 3,261 that way across three seeds, while every base-backbone cell lost
none — the chat template's
``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` followed straight by
``<|eot_id|>`` is the instruct-tuned spelling of an empty turn, and the base
backbone carries no such prior.

It read as a *quality* result rather than a *stopping* one, which is what made it
dangerous: `043` came in at 0.3434 ±0.0584 and the obvious story — "the instruct
backbone is worse" — was one I had already half-written. The tell was the seed
spread, thirty times the base control's, over eval_loss curves that agreed to
three decimal places. One position in ~50 moves eval_loss by ~0.005, so the
upstream metric could not see it.
"""

import pytest

from src.experiments.molecules import generate as G


class _Tok:
    eos_token_id = 128009
    pad_token_id = None

    def decode(self, ids, skip_special_tokens=False):
        return "caption"


class _Recorder:
    """Stands in for the model: records the kwargs `generate` was called with."""

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        import torch

        self.kwargs = kwargs
        n = kwargs["input_ids"].shape[0]
        return torch.zeros((n, kwargs["input_ids"].shape[1] + 3), dtype=torch.long)


def _packed(rows):
    import torch

    return {"input_ids": torch.zeros((len(rows), 4), dtype=torch.long)}


def test_the_stop_token_is_forbidden_for_the_first_steps_by_default():
    model = _Recorder()
    out = []

    G._run_batch(model, _Tok(), _packed, [{}, {}], None, 0, 256, out)

    assert model.kwargs["min_new_tokens"] == G.DEFAULT_MIN_NEW_TOKENS
    assert G.DEFAULT_MIN_NEW_TOKENS >= 1, \
        "zero would re-admit the empty caption this constraint exists to prevent"


def test_THE_DEFECT_min_new_tokens_zero_lets_a_caption_be_empty():
    """With the constraint off, an immediate EOS is a legal greedy choice.

    This is the state `043` was scored in.
    """
    model = _Recorder()
    out = []

    G._run_batch(model, _Tok(), _packed, [{}], None, 0, 256, out,
                 min_new_tokens=0)

    assert model.kwargs["min_new_tokens"] == 0


def test_the_constraint_is_not_conditional_on_the_backbone():
    """It applies to every arm, including the ones that do not need it.

    A constraint switched on only for the arm it rescues is not a measurement —
    it is a thumb on the scale. The base cells emit no empty captions, so this
    costs them nothing, and that is the point.
    """
    import inspect

    sig = inspect.signature(G.generate_captions)
    default = sig.parameters["min_new_tokens"].default

    assert default == G.DEFAULT_MIN_NEW_TOKENS
    src = inspect.getsource(G._run_batch)
    assert "model_name" not in src and "instruct" not in src.lower(), \
        "generation must not branch on which backbone it is scoring"


@pytest.mark.parametrize("n_rows", [1, 4])
def test_it_is_threaded_through_every_batch(n_rows):
    model = _Recorder()
    out = []

    G._run_batch(model, _Tok(), _packed, [{}] * n_rows, None, 0, 256, out, 7)

    assert model.kwargs["min_new_tokens"] == 7
    assert len(out) == n_rows
