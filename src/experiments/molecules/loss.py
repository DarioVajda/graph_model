"""
Per-example loss normalisation for Tier C, as the generalist harness does it.

HF's causal-LM loss is a mean over every supervised token in the micro-batch, so
an example contributes in proportion to its length. For a classification tier
that is one token per example and the distinction is empty. For captions it is
not: ChEBI answers run from a handful of tokens to 391, so a per-token mean lets
the long captions set the gradient, and the model is pushed toward their length.

The measured shape of that, over the 2x2 (2026-09-18) --- reference captions
average 43.9 words:

===================================== ========== ============ ==============
 run                                   4-gram rep  mean words   runaway >=200
===================================== ========== ============ ==============
 `probes/010` instruct, per_example         0.009         40.2       0 / 0 / 0
 `041` base WSD, per_token                  0.017         43.2      12 / 21 / 4
 `043` instruct WSD, per_token              0.035         48.2     126 / 34 / 40
===================================== ========== ============ ==============

The per-token runs write long and repeat; the per-example run writes *short* of
the reference and never runs away, on the same backbone that produces 126
runaways here. That is the hypothesis this module exists to test, and it is only
a hypothesis: `010` differs from `043` in six other ways at once (§6d.3).

``per_example`` divides each example's summed token loss by its own supervised
span and then means over examples, so every example contributes one unit
regardless of length. This is `generalist/mixture.py::MixtureLoss`'s definition,
reproduced rather than imported --- that class carries a task-id table and a
mixture-level DDP rescale this package has no use for.

**No DDP rescale here, and that is deliberate.** `MixtureLoss` multiplies by the
world size because it normalises by the *global* example count itself, so DDP's
gradient averaging would otherwise divide twice. This trainer returns a mean over
its own rank's examples and lets DDP average across ranks, which is the same
quantity when the ranks carry equal micro-batches --- and HF builds them that way.
"""

from __future__ import annotations

import torch

from ...utils import GraphTrainerV2

#: The default is `per_token`, which is HF's own behaviour and what every result
#: in this package before 2026-09-19 was trained under. Changing the default
#: would silently re-interpret those runs.
LOSS_NORMS = ("per_token", "per_example")


class PerExampleLossTrainer(GraphTrainerV2):
    """`GraphTrainerV2` with the loss meaned over examples, not over tokens.

    Only `compute_loss` differs; the bias-LR parameter groups, the
    `bias_parameters.pt` sidecar and everything else are inherited unchanged, so a
    run under this trainer differs from one under `GraphTrainerV2` in exactly the
    normalisation.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs,
                                        return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        logits = outputs.logits

        # Labels are NOT pre-shifted in this codebase -- the model shifts
        # internally, and `shift_logits_for_metrics` records the convention as
        # "logit at t predicts token t+1". Reproducing it here rather than
        # trusting `outputs.loss`, which is the per-token quantity being replaced.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).to(token_loss.dtype)
        spans = mask.sum(dim=-1)
        # An example with no supervised token contributes nothing, and must not
        # divide by zero on the way there.
        per_example = (token_loss * mask).sum(dim=-1) / spans.clamp(min=1.0)

        counted = (spans > 0).to(token_loss.dtype)
        loss = (per_example * counted).sum() / counted.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


def trainer_class_for(loss_norm):
    """The Trainer class implementing ``loss_norm``.

    Unknown values raise rather than falling back, because a typo that silently
    selected HF's default would produce a run whose record claims a normalisation
    it did not use --- and nothing downstream could tell.
    """
    if loss_norm not in LOSS_NORMS:
        raise ValueError(
            f"loss_norm must be one of {LOSS_NORMS}, got {loss_norm!r}")
    return PerExampleLossTrainer if loss_norm == "per_example" else GraphTrainerV2
