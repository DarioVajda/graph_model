"""
D7.3 — the validators the first build ships with.

============  ==============================================================
`in_mixture`  per task, on the splits its spec declares: the metric for its
              answer kind, per endpoint where the corpus has several
`held_out`    the same scorers, zero-shot, on the tasks that never train
`bias_norm`   L2 over the graph-bias channel — the resume fingerprint, and
              the number `feedback-verify-nulls-are-real` asks for
`grad_share`  measured per-task gradient share against the configured weight
`base_exact`  adapters off, logits equal to the base model's (Property 2)
`perm_spread` the flat twin's AUROC spread over randomized SMILES; the graph
              arm's over node relabelings, which must be ~0
`leakage`     the negative control — `stereo_assigned` with the parity
              channel closed, read against the split's own base rate
`throughput`  wall-clock s/it, peak GB, tokens/s
`per_example` the molecules per-example error / geometry report
============  ==============================================================

Every one of them is stdlib at import time — torch, numpy, sklearn and RDKit are
imported inside ``run`` — because this module is imported by
`evaluate/__init__.py`, which `validate` mode imports on the login node.

**The context's open field.** Three validators need something structural about
the run that is not a model, a dataset or a step. They read it from
``ctx.config`` under the constants named below rather than growing an
`EvalContext` field each, since the field list is what the trainer is wired to
and one validator's private requirement is not worth widening it. The constants
exist so that the trainer wires to a name rather than to a string literal.
"""

from __future__ import annotations

from . import BaseValidator, EvalError, EvalNeedsError, register

__all__ = [
    "ACTIVE_PARAMS", "GRAD_SHARE_COUNTS_FN", "GRAD_SHARE_LOSS_FN",
    "GRAD_SHARE_TASKS", "LEAKAGE_TASK", "MAX_LENGTH", "MAX_SPD",
    "UNCONDITIONAL_FORWARD", "BaseExact", "BiasNorm", "GradShare", "HeldOut",
    "InMixture", "Leakage", "PerExample", "PermSpread", "Throughput",
]

# ── the ctx.config keys the built-ins read ───────────────────────────────────
#: Parameter-name substrings that select the graph-bias channel, as
#: `molecules/train.py`'s ``ACTIVE_PARAMS`` and `checkpoint.bias_norm` use them.
ACTIVE_PARAMS = "active_params"
#: ``(task) -> scalar tensor``: that task's rows of the *current* micro-batch,
#: contributing to the loss exactly as `MixtureLoss` summed them. Only the
#: trainer can build this — it is the one thing here that needs to feed the model
#: mid-step — so the trainer installs it just before it fires the validators and
#: removes it after. Drawing a batch from ``train_sampler`` instead is not an
#: option: `draw_step` is a cursor, and advancing it inside an evaluation would
#: desynchronise the run it is measuring.
GRAD_SHARE_LOSS_FN = "grad_share_loss_fn"
#: ``() -> {task: examples}``: how many rows each task had in the micro-batch the
#: loss closure measures. Installed by the same trainer, and read for the same
#: reason the closure is: a share measured on four examples and a share measured
#: on forty are different claims, and only the trainer knows which one this is.
GRAD_SHARE_COUNTS_FN = "grad_share_counts_fn"
#: Optional task order for the above; defaults to the mixture's tasks.
GRAD_SHARE_TASKS = "grad_share_tasks"
#: The SPD clamp, for the geometry columns of the per-example report.
MAX_SPD = "max_spd"
#: Per-node truncation length, for a validator that re-renders an item it has
#: re-written. Read from the run rather than defaulted, so a run at a different
#: length does not have its re-written rows truncated somewhere else than its
#: built ones.
MAX_LENGTH = "max_length"
#: Set by a run whose forward pass is altered whatever the adapter context —
#: D4 arm B/C, or a bias family that is not row-constant on a single-node graph.
#: `base_exact` reports itself inapplicable rather than reporting a failure.
UNCONDITIONAL_FORWARD = "unconditional_forward_change"


def _sources(ctx) -> dict:
    return ctx.eval_sets or {}


def _spec(ctx, task):
    if ctx.registry is None:
        raise EvalNeedsError(
            f"{task}: no registry in the context, so the answer kind that "
            "decides the scorer is unknown")
    return ctx.registry.get(task)


def _is_held_out(ctx, task, spec) -> bool:
    from ..registry import is_held_out

    return bool(is_held_out(spec)) or "held_out" in ctx.sources(task)


# ─────────────────────────────────────────────────────────────────────────────
# in_mixture / held_out — one implementation, two task filters
# ─────────────────────────────────────────────────────────────────────────────

class _ScoringValidator(BaseValidator):
    """Shared body of `in_mixture` and `held_out` (D7.3: "same scorers").

    The two differ only in which tasks and which splits they look at, so they
    share :mod:`scorers` and this class. A second copy of the yes/no readout is
    exactly the drift `molecules/PLAN.md` warns about — the margin, its sigmoid
    and its tie diagnostics have one implementation in the repo and this is not
    a second one.
    """

    needs = frozenset({"model", "tokenizer", "collator", "eval_sets", "registry"})
    protocol_version = "1"

    #: Splits this validator is allowed to touch.
    splits: tuple = ()
    #: Whether the spec's ``eval_splits`` narrows that further. A held-out task
    #: has one split and no choice about it; an in-mixture task's spec says which
    #: of val / test the run scores.
    respect_eval_splits: bool = False

    def targets(self, ctx) -> list:
        """``[(task, split, source, spec), ...]`` this validator will score."""
        out = []
        for task in sorted(_sources(ctx)):
            spec = _spec(ctx, task)
            if not self.selects(ctx, task, spec):
                continue
            for split, source in sorted(ctx.sources(task).items()):
                if split not in self.splits:
                    continue
                if self.respect_eval_splits and split not in spec.eval_splits:
                    continue
                out.append((task, split, source, spec))
        return out

    def selects(self, ctx, task, spec) -> bool:
        raise NotImplementedError

    def keys(self, ctx=None) -> set:
        from .scorers import METRIC_KEYS

        if ctx is None:
            return {k for keys in METRIC_KEYS.values() for k in keys}
        return {k for _t, _s, _src, spec in self.targets(ctx)
                for k in METRIC_KEYS[spec.answer_kind]}

    def run(self, ctx) -> dict:
        from .scorers import DEFAULT_BATCH_SIZE, DEFAULT_BATCH_TOKENS, score_source

        max_samples = self.option("max_samples")
        batch_size = int(self.option("batch_size", DEFAULT_BATCH_SIZE))
        batch_tokens = int(self.option("batch_tokens", DEFAULT_BATCH_TOKENS))
        out = {}
        for task, split, source, spec in self.targets(ctx):
            scored = score_source(
                ctx.model, ctx.tokenizer, ctx.collator, source, spec,
                device=ctx.device, max_samples=max_samples, batch_size=batch_size,
                per_endpoint=bool(self.option("per_endpoint", True)),
                batch_tokens=batch_tokens)
            for key, value in scored.items():
                out[f"{task}/{split}/{key}"] = value
        return out


@register
class InMixture(_ScoringValidator):
    """Every task the run trains on, on the splits its spec declares.

    Teacher-forced for ``token`` and ``yesno``, generative for ``text`` and
    ``smiles`` (D1.1). ``yesno`` tasks carry ``tied_pair_fraction``,
    ``n_distinct`` and ``pos_rate`` in every record from the first run rather
    than retrofitted after a suspicious number — see `scorers` for why the first
    two are meaningless apart. Tox21 and SIDER additionally report per endpoint,
    which `meta["endpoint"]` carries.

    Both ``val`` and ``test`` are scored, and neither is a selection signal: the
    harness refuses a selection on ``test`` outright (D7.4) and a training run
    does not select at all. Test is scored here because the arm-2-minus-arm-1
    comparison is a *reported* number and the curve it sits on is worth having.
    """

    name = "in_mixture"
    cadence = "steps:500"
    splits = ("val", "test")
    respect_eval_splits = True

    def selects(self, ctx, task, spec) -> bool:
        return not _is_held_out(ctx, task, spec)


@register
class HeldOut(_ScoringValidator):
    """Zero-shot on the tasks that never train (`MOLECULE_GENERALIST.md` §4).

    ``bond_path``, ``longest_chain`` and ClinTox. The registry refuses to put any
    of them in a mixture, so nothing here can be contaminated by construction;
    what this measures is transfer, and it is the number the `adapt` fork's
    steps-to-target is compared against.
    """

    name = "held_out"
    cadence = "milestone"
    splits = ("held_out",)

    def selects(self, ctx, task, spec) -> bool:
        return _is_held_out(ctx, task, spec)


# ─────────────────────────────────────────────────────────────────────────────
# bias_norm
# ─────────────────────────────────────────────────────────────────────────────

@register
class BiasNorm(BaseValidator):
    """L2 over the graph-bias channel. One number, two jobs.

    It is the resume fingerprint `checkpoint.verify` checks the restored
    ``bias_parameters.pt`` against (the 2026-07-17 reload bug is why the tensors
    live beside the adapter and why the pairing is verified), and it is the
    number `feedback-verify-nulls-are-real` asks for before any null is believed:
    a zero-init module that never left its init has not produced a null result,
    it has produced no result.

    ``present`` is 0 on the flat arm, where there is no bias channel to measure.
    Reporting ``l2 = 0`` there would be indistinguishable from a graph arm whose
    bias never moved, which is the exact confusion this validator exists to
    prevent.
    """

    name = "bias_norm"
    cadence = "steps:500"
    needs = frozenset({"model"})
    protocol_version = "1"

    def keys(self, ctx=None) -> set:
        return {"l2", "present"}

    def run(self, ctx) -> dict:
        from ..checkpoint import bias_norm

        active = self.option("active_params") or (ctx.config or {}).get(
            ACTIVE_PARAMS, ["graph_bias"])
        norm = bias_norm(ctx.model, active)
        if norm is None:
            return {"l2": float("nan"), "present": 0.0}
        return {"l2": float(norm), "present": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# grad_share
# ─────────────────────────────────────────────────────────────────────────────

@register
class GradShare(BaseValidator):
    """Measured per-task gradient share against the configured weight (D4.3).

    The claim two-level normalisation makes is that a task's share of the
    gradient equals its share of the examples. This is the instrument that checks
    it, and it is on by default.

    **What the number is.** One optimizer step — the one the trainer last ran —
    and the L2 norm of the gradient each task's rows of it produce, summed over
    the step's micro-batches and normalised to sum to one over the tasks that
    *were* in it. The sample is the step rather than a micro-batch because a
    micro-batch is four examples of the step's twenty-five: the share swings by a
    factor of five between firings and a task with three tenths of the mixture is
    missing from it often enough to report ``nan`` about a third of the time,
    which leaves neither of D4.3's claims readable. Three things still follow, and
    reading the metric without them will mislead:

    * A task the step did not sample reports ``nan``, not zero, and is left out of
      ``max_abs_error``. Zero would read as "this task contributes nothing to the
      gradient" when what happened is that it was not drawn. ``n_measured`` says
      how many of the ``n_tasks`` were actually in the step.
    * ``step_share`` is the task's realised *example* share of the step. The
      configured weight is an expectation; the realised counts are a draw from it,
      and comparing ``share`` against ``weight`` without ``step_share`` beside it
      confounds "the sampler drew a different mix" with "this task's gradients are
      larger".
    * A share of norms is not a share of a sum — ``‖g_a‖ + ‖g_b‖ ≠ ‖g_a + g_b‖``
      — and per-example gradient magnitudes differ between tasks even when the
      example counts are exactly right. So ``abs_error`` is a *diagnostic*, not a
      conservation law with a tight tolerance: it catches a task that is absent
      from the gradient or dominating it, which is what D4.3 wants it for.

    It costs one forward and one backward per (task, micro-batch) pair the task
    appears in, which is why it runs on a cadence and not every step. The per-task
    loss closure comes from the trainer through ``ctx.config[GRAD_SHARE_LOSS_FN]``;
    see that constant for why it cannot be reconstructed here.
    """

    name = "grad_share"
    cadence = "steps:200"
    needs = frozenset({"model", "mixture"})
    protocol_version = "3"

    def keys(self, ctx=None) -> set:
        keys = {"share", "weight", "abs_error", "max_abs_error", "n_tasks",
                "n_measured"}
        # The example counts are the trainer's to supply and a context without
        # them gets none of the leaves that come from them — declared against
        # this context, not against every context this validator could have.
        if ctx is None or (ctx.config or {}).get(GRAD_SHARE_COUNTS_FN) is not None:
            keys |= {"step_share", "examples"}
        return keys

    def run(self, ctx) -> dict:
        from ..mixture import measure_grad_share

        config = ctx.config or {}
        loss_fn = config.get(GRAD_SHARE_LOSS_FN)
        if loss_fn is None:
            raise EvalNeedsError(
                "grad_share: the trainer must install a per-task loss closure at "
                f"config[{GRAD_SHARE_LOSS_FN!r}] before firing this validator — "
                "only it can feed the model the current micro-batch, and drawing "
                "one from the sampler would advance the cursor of the run being "
                "measured.")
        tasks = list(config.get(GRAD_SHARE_TASKS) or ctx.mixture.shares)
        measured = measure_grad_share(
            ctx.model, loss_fn, tasks,
            params=config.get("grad_share_params"))

        counts_fn = config.get(GRAD_SHARE_COUNTS_FN)
        counts = dict(counts_fn() or {}) if counts_fn is not None else {}
        drawn = float(sum(counts.values()))

        shares = dict(ctx.mixture.shares)
        nan = float("nan")
        out, worst = {}, 0.0
        for task in tasks:
            share = float(measured[task]) if task in measured else nan
            weight = float(shares.get(task, nan))
            error = abs(share - weight)
            out[f"{task}/share"] = share
            out[f"{task}/weight"] = weight
            out[f"{task}/abs_error"] = error
            if counts_fn is not None:
                n = float(counts.get(task, 0.0))
                out[f"{task}/examples"] = n
                out[f"{task}/step_share"] = (n / drawn) if drawn else nan
            if error == error:            # nan-safe: absent task, or unknown weight
                worst = max(worst, error)
        out["max_abs_error"] = worst
        out["n_tasks"] = float(len(tasks))
        out["n_measured"] = float(len(measured))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# base_exact
# ─────────────────────────────────────────────────────────────────────────────

#: Bit-identical is what "never written to" means, and it is what a frozen tensor
#: gives. The tolerance is not zero only because a checkpoint can be reloaded in a
#: different dtype than it was saved in; it is far below any training step.
BASE_EXACT_TOL = 1e-6

#: Name fragments belonging to what this project *adds*. A parameter carrying one
#: is not part of the backbone and has no counterpart in the base model, so it is
#: not compared — everything else must match.
BASE_EXACT_ADDED = ("lora_", "graph_bias", "modules_to_save")


def _backbone_name(name: str) -> str:
    """A trained model's parameter name as the base model spells it.

    PEFT wraps the model twice (``base_model.model.…``) and inserts
    ``base_layer`` in front of every module it adapted. Neither is a different
    tensor — they are the same storage under a longer path — so the comparison
    strips them rather than reporting every adapted projection as missing.
    """
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(".base_layer.", ".")


@register
class BaseExact(BaseValidator):
    """The backbone is still base Llama, tensor for tensor (Property 2).

    Compares every backbone parameter of the trained model against a freshly
    loaded base model and reports the largest absolute difference. What it proves
    is that the backbone weights were never written to: everything this project
    adds is additive and removable, which is what makes "the graph channel caused
    this" a statement about the channel rather than about a differently-trained
    Llama. The LoRA tensors, the bias channel and anything else this project adds
    have no counterpart in the base model and are skipped by name; PEFT's
    ``base_model.model.`` prefix and its ``base_layer`` indirection are paths to
    the *same* storage, not different tensors, so they are stripped rather than
    reported as missing.

    **Weights, not logits.** The design asked for a logit comparison on a plain
    text batch, and on this architecture there is no such batch: `GTLMLlama`'s
    forward requires ``node_ids`` and ``prompt_node``, and disabling the LoRA
    adapter does not disable the graph bias, which is a separate module and is
    added to the attention logits whatever graph it is handed. So a logit
    comparison either cannot run or is answering a different question. The weight
    comparison is the property itself and is exact rather than within a bf16
    tolerance — a forward pass was only ever the instrument for reading it.

    **When it is meaningless, it says so instead of failing.** A run that unfreezes
    ``W_q``/``W_k`` or fine-tunes the backbone (D4 arm B/C, deferred) moves the
    backbone and *should* — the property does not hold and its failure is not a
    defect. ``applicable`` is 0 in that case with ``reason`` naming it, and a
    ``within_tolerance`` of 0 alongside ``applicable = 0`` is the expected
    reading, not an alarm. Detection is automatic (a trainable parameter that is
    neither LoRA nor a declared bias tensor) and can also be declared through
    ``config[UNCONDITIONAL_FORWARD]``.
    """

    name = "base_exact"
    cadence = "milestone"
    protocol_version = "2"

    @property
    def needs(self) -> frozenset:
        # A caller may hand the base model in directly (a test, or a run that has
        # one resident already); only otherwise is the name required.
        base = frozenset({"model"})
        return base if self.option("base_model") is not None else base | {"base_model"}

    def keys(self, ctx=None) -> set:
        return {"max_abs_diff", "within_tolerance", "applicable", "reason",
                "n_tensors", "n_unmatched"}

    def _applicability(self, ctx) -> tuple:
        config = ctx.config or {}
        declared = config.get(UNCONDITIONAL_FORWARD)
        if declared:
            return 0.0, f"the run declares an unconditional forward change: {declared}"
        active = config.get(ACTIVE_PARAMS, ["graph_bias"])
        stray = [name for name, param in ctx.model.named_parameters()
                 if param.requires_grad and "lora_" not in name
                 and not any(a in name for a in active)]
        if stray:
            return 0.0, ("trainable parameters outside the LoRA adapter and the "
                         f"bias channel ({stray[:3]}); the backbone moves, so the "
                         "base model is not recoverable and this check does not apply")
        return 1.0, ""

    def run(self, ctx) -> dict:
        import torch

        applicable, reason = self._applicability(ctx)
        base = self.option("base_model")
        if base is None:
            from transformers import AutoModelForCausalLM

            base = AutoModelForCausalLM.from_pretrained(
                ctx.base_model_name, torch_dtype=getattr(ctx.model, "dtype", None))

        theirs = dict(base.named_parameters())
        added = tuple(self.option("added_fragments") or BASE_EXACT_ADDED)

        worst, compared, unmatched = 0.0, 0, 0
        with torch.no_grad():
            for name, param in ctx.model.named_parameters():
                if any(fragment in name for fragment in added):
                    continue
                other = theirs.get(_backbone_name(name))
                if other is None:
                    # A backbone tensor the base model does not have is a
                    # structural difference, not a weight that moved: counted and
                    # reported, so a silently renamed module cannot pass by
                    # comparing nothing.
                    unmatched += 1
                    continue
                if other.shape != param.shape:
                    raise EvalError(
                        f"base_exact: {name} is {tuple(param.shape)} in the trained "
                        f"model and {tuple(other.shape)} in the base model; the two "
                        "are not the same architecture")
                diff = (param.detach().float().cpu()
                        - other.detach().float().cpu()).abs().max()
                worst = max(worst, float(diff))
                compared += 1

        if not compared:
            raise EvalError(
                "base_exact: no backbone parameter matched a base-model parameter, "
                "so nothing was checked. The name mapping is wrong, not the model.")

        tol = float(self.option("tolerance", BASE_EXACT_TOL))
        return {"max_abs_diff": worst,
                "within_tolerance": float(worst <= tol and unmatched == 0),
                "applicable": applicable,
                "reason": reason or "adapters are additive and removable",
                "n_tensors": float(compared),
                "n_unmatched": float(unmatched)}


# ─────────────────────────────────────────────────────────────────────────────
# perm_spread
# ─────────────────────────────────────────────────────────────────────────────

#: Property 1 was verified to 2.77e-5 on this architecture, so the graph arm's
#: spread over relabelings is asserted at this and read as a bug above it.
#:
#: That verification was on a quantity that is not quantized. The margin is: it
#: is a difference of two bf16 logits, and on the smoke run's own rows it takes
#: 21 distinct values over 152 examples with a minimum gap of exactly 0.125. A
#: 1e-4 assertion on a quantity whose smallest nonzero value is 0.125 cannot be
#: met by an equivariant model — it can only be met by luck, when every
#: permutation happens to land on the same grid point. So the comparison is
#: against ``max(tolerance, margin_quantum)``, with the quantum measured from
#: the margins in hand rather than assumed, and reported beside the spread so
#: the assumption is visible instead of buried in a threshold.
PERM_TOL = 1e-4


@register
class PermSpread(BaseValidator):
    """The atom-order-invariance measurement (`molecules/PLAN.md` §6).

    Ten re-writings of the same molecule per test example, scored the way the
    AUROC is scored, and the spread across them reported. On the **flat** arm the
    re-writing is a randomized SMILES: the same molecule from a different
    starting atom is a different token string, and the spread is the whole
    claim — the graph arm answers a question about a molecule, the flat arm about
    a string that happens to denote one. On the **graph** arm there is no input
    order to randomize, so the re-writing is a relabeling of the nodes, which is
    the same permutation acting on the input GTLM actually consumes; Property 1
    makes the spread zero and this asserts it — against a floor the same run
    measures, because bf16 in batches does not reproduce an exact zero and the
    amount by which it misses grows with the margins. See :meth:`_control_spread`.

    **Stratified by symmetry class, or the effect is understated.** Benzene has
    one atom symmetry class, so every traversal yields the same string and the
    flat arm is invariant for free; molecules like that dilute the spread toward
    zero. ``Chem.CanonicalRankAtoms(mol, breakTies=False)`` gives the classes and
    the ``asymmetric`` stratum is the one the claim rests on.
    """

    name = "perm_spread"
    cadence = "end"
    needs = frozenset({"model", "tokenizer", "collator", "eval_sets", "registry"})
    protocol_version = "2"

    def _tasks(self, ctx) -> list:
        out = []
        for task in sorted(_sources(ctx)):
            spec = _spec(ctx, task)
            if spec.answer_kind != "yesno":
                continue                    # the spread is a spread *of AUROC*
            source = ctx.sources(task).get(self.option("split", "test"))
            if source is not None and len(source):
                out.append((task, source, spec))
        return out

    def keys(self, ctx=None) -> set:
        declared = {"auroc_mean", "auroc_spread", "auroc_std", "n_molecules",
                    "n_permutations", "margin_spread_max", "margin_quantum"}
        # `within_tolerance` asserts Property 1 and Property 1 is a graph-arm
        # property; see :meth:`run`. `margin_control_max` is the floor it is
        # asserted against and is measured in the same place. Context-aware
        # because the declaration has to match what the arm actually produces.
        if ctx is None or getattr(ctx, "arm", None) == "graph":
            declared |= {"within_tolerance", "margin_control_max"}
        return declared

    def run(self, ctx) -> dict:
        import numpy as np

        from ...experiments.molecules.evaluate import (
            answer_token_ids, make_margin_preprocessor,
        )
        from .scorers import eval_indices, teacher_forced

        n_perms = int(self.option("n_permutations", 10))
        cap = self.option("n_molecules", 200)
        tol = float(self.option("tolerance", PERM_TOL))
        yes_id, no_id = answer_token_ids(ctx.tokenizer)
        preprocess = make_margin_preprocessor(yes_id, no_id)

        out = {}
        for task, source, _task_spec in self._tasks(ctx):
            arm = ctx.arm or getattr(source, "arm", "flat")
            indices = eval_indices(len(source), cap)
            views = [_PermutedSource(source, arm, p, ctx.tokenizer, indices)
                     for p in range(n_perms)]

            margins, true_ids = [], None
            for view in views:
                preds, _labels = teacher_forced(
                    ctx.model, ctx.collator, view, range(len(view)),
                    device=ctx.device,
                    batch_size=int(self.option("batch_size", 8)),
                    preprocess=preprocess)
                margins.append(preds[:, 0] - preds[:, 1])
                if true_ids is None:
                    # Column 2 is the target token id, gathered in the same pass
                    # as the score so the two cannot drift apart. It is the same
                    # for every permutation — only the input was re-written.
                    true_ids = preds[:, 2]
            margins = np.asarray(margins, dtype=np.float64)      # (perms, molecules)
            y_true = (np.asarray(true_ids, dtype=np.float64) == yes_id).astype(float)

            # The per-molecule spread of the raw margin, before any metric: this
            # is the quantity Property 1 makes zero on the graph arm, and it is
            # asserted rather than an AUROC because an AUROC can be identical
            # across permutations that moved every score.
            spread = (float((margins.max(axis=0) - margins.min(axis=0)).max())
                      if margins.size else 0.0)
            quantum = _margin_quantum(margins)
            out[f"{task}/n_permutations"] = float(n_perms)
            out[f"{task}/n_molecules"] = float(margins.shape[1])
            out[f"{task}/margin_spread_max"] = spread
            out[f"{task}/margin_quantum"] = quantum
            if arm == "graph":
                # A graph-arm assertion only. Property 1 says the spread is zero
                # there, so a 0/1 flag is the right shape for it. On the flat arm
                # the re-writing is a different *string* for the same molecule and
                # a nonzero spread is the finding rather than a failure — the
                # first flat cross-check read `margin_spread_max` 31.625 beside
                # `within_tolerance` 0.0, which is the arm behaving exactly as
                # described and reads at a glance as a broken run. The spread and
                # the quantum are reported on both arms; the verdict is not.
                control = self._control_spread(ctx, source, arm, indices,
                                               preprocess)
                out[f"{task}/margin_control_max"] = control
                out[f"{task}/within_tolerance"] = float(
                    spread <= max(tol, quantum, control))

            classes = _symmetry_classes(source, indices)
            strata = {"all": np.ones(len(indices), dtype=bool),
                      "asymmetric": np.asarray([c > 1 for c in classes]),
                      "symmetric": np.asarray([c <= 1 for c in classes])}
            for stratum, mask in strata.items():
                out.update({f"{task}/{stratum}/{k}": v for k, v in
                            _auroc_spread(margins, y_true, mask).items()})
        return out

    def _control_spread(self, ctx, source, arm, indices, preprocess) -> float:
        """The same measurement with nothing permuted — the instrument's own floor.

        Property 1 is a statement about the function the model computes, and the
        margin is not that function: it is that function evaluated in bf16, in
        batches, on a kernel whose reductions are not associative. Re-ordering the
        molecules changes each one's padding and its neighbours in the reduction,
        and the margin moves — with no relabelling anywhere, on inputs that are
        bit-identical. That movement is the floor below which the permuted spread
        cannot say anything, and it is not a constant: on the BACE cross-check it
        was 0.375 at the end of the stable phase and 0.750 after the anneal, which
        tracks the margins doubling from 2.5 to 6.0 rather than any property of
        the model.

        A fixed absolute tolerance therefore cannot be right at both ends of a
        run, and picking a relative one from two measurements would be fitting the
        threshold to the data it has to judge. Measuring it is the alternative:
        this runs ``n_control`` extra passes over the *unpermuted* molecules in
        shuffled batch order, and the assertion becomes "relabelling moves the
        margin no more than re-batching does", which is Property 1 in a form the
        hardware can actually satisfy. Set ``n_control: 0`` to fall back to the
        quantum alone.
        """
        import numpy as np

        from .scorers import teacher_forced

        n_control = int(self.option("n_control", 3))
        if n_control < 2:
            return 0.0
        rng = np.random.default_rng(0)
        rows = []
        for k in range(n_control):
            order = rng.permutation(len(indices)) if k else np.arange(len(indices))
            view = _PermutedSource(source, arm, 0, ctx.tokenizer,
                                   [indices[j] for j in order])
            preds, _labels = teacher_forced(
                ctx.model, ctx.collator, view, range(len(view)),
                device=ctx.device, batch_size=int(self.option("batch_size", 8)),
                preprocess=preprocess)
            row = np.empty(len(indices), dtype=np.float64)
            row[order] = preds[:, 0] - preds[:, 1]
            rows.append(row)
        rows = np.asarray(rows, dtype=np.float64)
        return float((rows.max(axis=0) - rows.min(axis=0)).max()) if rows.size else 0.0


def _margin_quantum(margins) -> float:
    """The margin's resolution, measured: the smallest gap between two of them.

    The margin is a difference of two bf16 logits and so lives on a grid. A
    spread no wider than one step of that grid is not a spread this instrument
    can distinguish from rounding, whatever the configured tolerance says — so
    the grid is measured from the margins in hand rather than assumed from the
    dtype, which would need the logits' magnitudes and not just their difference.

    Zero when every margin is identical, which is the case where the spread is
    zero anyway and the tolerance decides on its own.
    """
    import numpy as np

    if margins.size == 0:
        return 0.0
    distinct = np.unique(margins)
    if distinct.size < 2:
        return 0.0
    return float(np.diff(distinct).min())


def _auroc_spread(margins, y_true, mask) -> dict:
    """AUROC per permutation over a stratum, and the spread across them."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    nan = float("nan")
    if margins.size == 0 or not mask.any():
        return {"auroc_mean": nan, "auroc_spread": nan, "auroc_std": nan,
                "n_molecules": 0.0}
    y = y_true[mask]
    if len(np.unique(y)) < 2:
        # One class in the stratum: AUROC is undefined and a spread of an
        # undefined quantity is not zero, it is absent.
        return {"auroc_mean": nan, "auroc_spread": nan, "auroc_std": nan,
                "n_molecules": float(mask.sum())}
    aurocs = [float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-row[mask]))))
              for row in margins]
    return {"auroc_mean": float(np.mean(aurocs)),
            "auroc_spread": float(max(aurocs) - min(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "n_molecules": float(mask.sum())}


def _symmetry_classes(source, indices) -> list:
    """Distinct atom symmetry classes per selected molecule, from the partition key.

    The key is the stereo-free canonical SMILES (`MOLECULE_GENERALIST.md` §3), so
    it is always present and always parses — no need to dig the input string back
    out of the prompt, which differs between the two arms.
    """
    from rdkit import Chem, RDLogger

    from ..schema import SIDECAR_KEY

    RDLogger.DisableLog("rdApp.*")
    out = []
    for i in indices:
        key = (source[i].get(SIDECAR_KEY) or {}).get("key", "")
        mol = Chem.MolFromSmiles(key) if key else None
        if mol is None:
            out.append(1)
            continue
        out.append(len(set(Chem.CanonicalRankAtoms(mol, breakTies=False))))
    return out


class _PermutedSource:
    """One re-writing of every selected molecule, as a ``TaskSource``.

    Flat arm: the prompt's SMILES is replaced by a seeded randomized SMILES of
    the same molecule and the prompt node is re-tokenized through
    `schema.render`, so the label convention is the schema's and not a second
    copy of it. Permutation 0 is the item as built, which anchors the spread to
    the number the ordinary evaluation reports.

    Graph arm: the node ordering is permuted. Every node-indexed column is
    permuted with it, and a column this does not know how to permute is an error
    rather than a pass-through — a silently mis-permuted feature would produce a
    fake nonzero spread, which is the one failure mode that would look like a
    real violation of Property 1.
    """

    def __init__(self, source, arm, perm_id, tokenizer, indices):
        self._source = source
        self._arm = arm
        self._perm = int(perm_id)
        self._tokenizer = tokenizer
        self._indices = list(indices)
        self.task = getattr(source, "task", "?")
        self.split = getattr(source, "split", "?")
        self.arm = arm
        self.pass_id = getattr(source, "pass_id", 0)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict:
        item = self._source[self._indices[i]]
        if self._perm == 0:
            return item
        if self._arm == "flat":
            return _rewritten_flat_item(item, self._tokenizer, self._perm)
        return _relabelled_graph_item(item, self._perm)


#: The flat prompt's SMILES sits between these, as `dataset.build_flat_example`
#: writes it: ``"{question}\nSMILES: {smiles}\nA:{answer}"``.
SMILES_MARKER = "\nSMILES: "


def _rewritten_flat_item(item, tokenizer, perm_id) -> dict:
    """The same molecule, written from a different starting atom."""
    from rdkit import Chem, RDLogger

    from ...experiments.molecules.data import flat_serialize
    from ..schema import SIDECAR_KEY, Example, render

    RDLogger.DisableLog("rdApp.*")
    side = item.get(SIDECAR_KEY) or {}
    prompt_node = int(item["prompt_node"])
    text = item["text"][prompt_node]
    start = text.find(SMILES_MARKER)
    if start < 0:
        raise EvalError(
            f"perm_spread: the flat prompt {text[:60]!r} carries no "
            f"{SMILES_MARKER!r}; there is no SMILES to re-write")
    start += len(SMILES_MARKER)
    end = text.find("\n", start)
    mol = Chem.MolFromSmiles(text[start:end])
    if mol is None:
        raise EvalError(
            f"perm_spread: the flat prompt's SMILES {text[start:end]!r} does not "
            "parse, so it cannot be re-written")
    rewritten = flat_serialize(mol, canonical=False,
                               seed=perm_id * 7919 + len(text))

    new_text = text[:start] + rewritten + text[end:]
    out = dict(item)
    out["text"] = [new_text if n == prompt_node else t
                   for n, t in enumerate(item["text"])]
    stub = Example(task=side.get("task", "_"), domain=side.get("domain", "_"),
                   split=side.get("split", "test"), arm="flat",
                   graph={"text": [new_text], "prompt_node": 0, "num_nodes": 1},
                   question=side.get("question", "_"),
                   answer=side.get("answer", ""),
                   answer_kind=side.get("answer_kind", "yesno"),
                   key=side.get("key", "_"))
    rendered = render(stub, tokenizer)
    out["input_ids"] = [list(rendered.input_ids[0])]
    # A tensor, as `TextGraphDataset.__getitem__` hands one over: the collator
    # asserts on its shape and would not accept a list.
    out["labels"] = _as_labels(item.get("labels"), rendered.labels)
    return out


def _as_labels(existing, labels):
    import torch

    if hasattr(existing, "shape"):
        return torch.tensor(labels, dtype=existing.dtype)
    return torch.tensor(labels, dtype=torch.long)


#: Columns whose meaning is not "one row per node": left alone by a relabeling.
#: ``labels`` is aligned to the prompt node's *tokens*, and its length can equal
#: the node count by coincidence, so it is excluded by name and never by shape.
#:
#: ``magnetic_lambdas`` is the one that cost a reading. It is ``(M,)``, indexed
#: by *eigenvector*, and ``M`` is the node count whenever the spectrum is not
#: truncated — so the shape rule below caught it and permuted it, pairing every
#: eigenvector with another one's eigenvalue. The graph arm then reported a
#: ``margin_spread_max`` of 0.75 where Property 1 says zero, which is exactly the
#: "reads as a violation rather than as the bug it is" this function was written
#: to prevent. A spectrum belongs to the graph; relabelling its nodes does not
#: reorder it. ``magnetic_V`` is ``(N, M, 2)`` and *is* node-indexed, but only on
#: axis 0, which is the axis the shape rule permutes — so it is correct there.
_NODE_INVARIANT = ("num_nodes", "labels", "ds_label", "edge_attr",
                   "magnetic_lambdas")

#: Columns indexed by node on their first *two* axes.
_PAIRWISE = ("shortest_path_dists", "rrwp")


def _relabelled_graph_item(item, perm_id) -> dict:
    """The same graph with its nodes listed in a different order.

    Every column is placed by name or by shape, and a column that matches
    neither is refused. The alternative — passing an unrecognised column through
    untouched — leaves a feature indexed by the old node order beside a graph in
    the new one, and the resulting nonzero spread would read as a violation of
    Property 1 rather than as the bug it is.
    """
    import random

    from ..schema import SIDECAR_KEY

    n = int(item["num_nodes"])
    order = list(range(n))
    random.Random(f"perm:{perm_id}:{n}").shuffle(order)
    where = {old: new for new, old in enumerate(order)}   # old index -> new index

    out = {}
    for key, value in item.items():
        if key in (SIDECAR_KEY,) + _NODE_INVARIANT or value is None:
            out[key] = value
        elif key in ("prompt_node", "question_node"):
            out[key] = where[int(value)] if 0 <= int(value) < n else int(value)
        elif key == "edges":
            out[key] = [(where[int(u)], where[int(v)]) for u, v in value]
        elif key == "original_ids":
            out[key] = {original: where[int(i)] for original, i in value.items()}
        elif key in _PAIRWISE:
            out[key] = _permute_pairwise(value, order, n)
        elif hasattr(value, "shape") or isinstance(value, (list, tuple)):
            length = value.shape[0] if hasattr(value, "shape") else len(value)
            if length != n:
                raise EvalError(
                    f"perm_spread: column {key!r} has {length} rows on a "
                    f"{n}-node graph, and this relabeling does not know what they "
                    "mean. Teach it, or exclude the column — a silently "
                    "mis-permuted feature reads as a violation of Property 1.")
            out[key] = (value[order] if hasattr(value, "shape")
                        else [value[old] for old in order])
        else:
            out[key] = value                              # scalars, strings
    return out


def _permute_pairwise(value, order, n):
    """Relabel a column indexed by node on both of its first two axes.

    ``shortest_path_dists`` arrives as an ``(n, n)`` tensor off the dataset and
    as a flat or nested list from a hand-built item; the return keeps whichever
    shape it was given, because the collator reads a tensor and
    `analysis.as_spd_matrix` reads either.
    """
    import numpy as np

    if hasattr(value, "shape"):
        return value[order][:, order]
    matrix = np.asarray(value)
    if matrix.ndim == 1:
        matrix = matrix.reshape(n, n)
        return matrix[np.ix_(order, order)].reshape(-1).tolist()
    return matrix[np.ix_(order, order)].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# throughput
# ─────────────────────────────────────────────────────────────────────────────

@register
class Throughput(BaseValidator):
    """Wall-clock seconds per iteration, peak GB, tokens/s.

    **Wall clock, between two firings, divided by the steps between them** — not
    a mean of per-step millisecond timers (`feedback-throughput-metric`). The two
    disagree by everything that is not inside the timed region: the dataloader,
    the evaluation passes, checkpoint writes, the DDP barrier. It is the wall
    clock that decides what a run costs, so it is the wall clock that is
    reported.

    The first firing has nothing to measure against and reports ``nan``; that is
    also true after a resume, since the instrument's state lives in the process
    and not in the checkpoint.

    **Host RAM is reported beside the GPU peak, because it is the one that kills
    runs here.** `molecules/PLAN.md` §8.4.9 lost 11.8 GPU-h to a host OOM, and the
    reason it took two months to explain is that nothing in this repo measured
    host memory during a run — every number was a post-mortem ``sacct`` MaxRSS,
    which on this cluster is the cgroup peak. ``host_peak_gb`` is that same
    quantity, live, so a ``--mem`` request can be read off a run instead of
    estimated from one. ``host_anon_gb`` is beside it because only the anonymous
    half is unreclaimable: a cgroup total dominated by page cache is not pressure,
    and reading the total alone is how 43 GB of reclaimable file cache gets
    mistaken for a memory requirement.
    """

    name = "throughput"
    cadence = "steps:50"
    protocol_version = "2"

    def __init__(self, cadence=None, **options):
        super().__init__(cadence, **options)
        self._last = None                     # (monotonic seconds, step)
        self._host = None

    def keys(self, ctx=None) -> set:
        return {"s_per_it", "tokens_per_s", "peak_gb", "steps_measured", "wall_s",
                "host_gb", "host_peak_gb", "host_anon_gb", "host_limit_gb"}

    def run(self, ctx) -> dict:
        import time

        import torch

        now, step = time.monotonic(), int(ctx.step)
        nan = float("nan")
        s_per_it = wall = nan
        steps = 0.0
        if self._last is not None:
            wall = now - self._last[0]
            steps = float(step - self._last[1])
            s_per_it = wall / steps if steps > 0 else nan
        self._last = (now, step)

        peak = nan
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

        tokens_per_step = getattr(ctx.mixture, "tokens_per_step", None)
        tokens_per_s = (tokens_per_step / s_per_it
                        if tokens_per_step and s_per_it == s_per_it and s_per_it > 0
                        else nan)
        return {"s_per_it": float(s_per_it), "tokens_per_s": float(tokens_per_s),
                "peak_gb": float(peak), "steps_measured": steps,
                "wall_s": float(wall), **self._host_gb()}

    def _host_gb(self) -> dict:
        """Cgroup usage, peak, anon and limit — ``nan`` wherever the kernel is silent."""
        from ...experiments.expressiveness.training.instrumentation import HostMemProbe

        nan = float("nan")
        if self._host is None:
            self._host = HostMemProbe()
        if not self._host.available:
            return {"host_gb": nan, "host_peak_gb": nan, "host_anon_gb": nan,
                    "host_limit_gb": nan}
        stat = self._host.cgroup_stat_gb()
        return {
            "host_gb": float(self._host.cgroup_gb("current") or nan),
            "host_peak_gb": float(self._host.cgroup_gb("peak") or nan),
            "host_anon_gb": float(stat.get("cg_anon_gb", nan)),
            "host_limit_gb": float(self._host.cgroup_gb("limit") or nan),
        }


# ─────────────────────────────────────────────────────────────────────────────
# per_example
# ─────────────────────────────────────────────────────────────────────────────

@register
class PerExample(BaseValidator):
    """The molecules per-example error / geometry report on ``test``.

    Wraps `molecules/analysis.py`'s ``write_per_example_report`` — the report
    that answers "is a mistake explained by the molecule's width, or by the
    ``max_spd`` clamp", and whose Tier-B path was untested for a whole campaign
    because it was wrapped in a try/except that made every failure look like an
    absent nicety. Here the failure is an entry in the validator status list, and
    the report writes one JSON line per example into the scratch directory.

    That function wants a trainer, and there is no trainer in an `EvalContext`;
    it wants ``.predict`` and nothing else, so it gets a shim that runs the same
    forward with the same ``preprocess_logits_for_metrics`` the tier would use.

    **The whole split, always.** Every other scoring validator takes a
    ``max_samples`` and this one refuses it, for two reasons. The row's ``i`` is
    its join key back into the split, and over a subsample it would silently
    mean a position in the subsample instead. And the summary is a
    *distribution* — ``diameter_p90``, ``examples_touching_clamp`` — which is
    the quantity the report exists to state precisely; estimating it from 32
    rows would answer the ``max_spd`` question with noise. It fires once, at the
    end of a run, so the cost is one pass and it buys the whole file.
    """

    name = "per_example"
    cadence = "end"
    REJECTED_OPTIONS = {
        "max_samples": "the report is over the whole split — a row's 'i' is its "
                       "index into that split, and the clamp and diameter "
                       "summaries are distributions, not estimates",
    }
    needs = frozenset({"model", "tokenizer", "collator", "eval_sets", "registry",
                       "scratch_dir"})
    protocol_version = "1"

    SUMMARY_KEYS = ("per_example_path", "per_example_accuracy",
                    "per_example_roc_auc", "diameter_p50", "diameter_p90",
                    "diameter_max", "examples_touching_clamp",
                    "mean_clamped_fraction")

    def keys(self, ctx=None) -> set:
        return set(self.SUMMARY_KEYS)

    def _tasks(self, ctx) -> list:
        split = self.option("split", "test")
        out = []
        for task in sorted(_sources(ctx)):
            spec = _spec(ctx, task)
            if spec.answer_kind not in ("token", "yesno"):
                continue     # the report reads a token span or a margin, not text
            source = ctx.sources(task).get(split)
            if source is not None and len(source):
                out.append((task, source, spec))
        return out

    def run(self, ctx) -> dict:
        import os

        from ...experiments.molecules.analysis import write_per_example_report
        from ...experiments.molecules.evaluate import answer_token_ids

        out_dir = os.path.join(ctx.scratch_dir, "per_example")
        os.makedirs(out_dir, exist_ok=True)
        yes_id, no_id = answer_token_ids(ctx.tokenizer)

        out = {}
        for task, source, spec in self._tasks(ctx):
            tier = "B" if spec.answer_kind == "yesno" else "A"
            shim = _PredictShim(ctx, tier, yes_id, no_id,
                                batch_size=int(self.option("batch_size", 8)))
            path = os.path.join(
                out_dir, f"{task.replace('/', '_')}-step{int(ctx.step)}.jsonl")
            summary = write_per_example_report(
                shim, source, _AnalysisConfig(tier, (ctx.config or {}).get(MAX_SPD, 32)),
                path, yes_id=yes_id if tier == "B" else None)
            for key in self.SUMMARY_KEYS:
                value = summary.get(key)
                if key == "per_example_path":
                    out[f"{task}/{key}"] = str(value or "")
                else:
                    out[f"{task}/{key}"] = (float("nan") if value is None
                                            else float(value))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# leakage — the negative control the suite lost
# ─────────────────────────────────────────────────────────────────────────────

#: The family the detector runs on. `stereo_assigned` asks how many stereocentres
#: have a *defined* configuration, and that answer lives in the parity tag and
#: nowhere else in a plain atom-bond graph (`molecules/PLAN.md` §1) — which is
#: what makes it the one task whose input channel can be closed cleanly and the
#: score read against a floor.
LEAKAGE_TASK = "mol/stereo_assigned"


def _parity_pairs() -> set:
    """The two-word phrases `data.atom_text` and `data.bond_text` append.

    Read off the renderer's own tables rather than spelled out here. A word added
    to either table has to reach the stripper, and a stripper that quietly misses
    one reports "at chance" while the channel is still open — which is the one
    failure of this instrument that looks like a pass.
    """
    from ...experiments.molecules.data import _BOND_STEREO_WORDS, _CHIRAL_WORDS

    return ({("chiral", word) for word in _CHIRAL_WORDS.values()} |
            {("stereo", word) for word in _BOND_STEREO_WORDS.values()})


def _without_pairs(text: str, pairs: set) -> str:
    """Drop each two-word phrase from space-joined node text.

    Word-level rather than a substring replace: ``atom_text`` joins its parts
    with single spaces, so a phrase is always two whole words, and matching on
    substrings would make ``chiral cw`` a candidate inside ``chiral ccw``.
    """
    words = text.split(" ")
    out, i = [], 0
    while i < len(words):
        if i + 1 < len(words) and (words[i], words[i + 1]) in pairs:
            i += 2
            continue
        out.append(words[i])
        i += 1
    return " ".join(out)


def _restereo_item(item, arm, keep_stereo, tokenizer, pairs, max_length):
    """One item with the stereo channel open or closed. ``(item, changed)``.

    **Graph arm.** The parity words come out of every node's text except the
    prompt node's, which carries the question and no molecule. Keeping them is
    the item exactly as built, so the tagged view costs nothing.

    **Flat arm.** ``stereo_tags`` is not a flat-arm knob — `flat_serialize` never
    consulted it, which is why `016` ran no flat ``off`` cell — so the channel is
    closed by re-serialising the molecule without stereochemistry. Both views are
    then re-serialised canonically, the tagged one included: canonicalisation is
    itself a re-writing that moves the flat arm's score (that is `perm_spread`'s
    whole finding), so comparing a canonical stripped string against the built
    string would confound the two changes. The consequence is that this
    validator's tagged number is its own baseline and is *not* the number
    `in_mixture` reports for the same split.
    """
    from ..schema import SIDECAR_KEY, Example, render

    side = item.get(SIDECAR_KEY) or {}
    prompt_node = int(item["prompt_node"])

    if arm == "graph":
        if keep_stereo:
            return item, 0
        texts = list(item["text"])
        new_texts = [t if n == prompt_node else _without_pairs(t, pairs)
                     for n, t in enumerate(texts)]
        changed = int(any(a != b for a, b in zip(texts, new_texts)))
        if not changed:
            return item, 0
    else:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        text = item["text"][prompt_node]
        start = text.find(SMILES_MARKER)
        if start < 0:
            raise EvalError(
                f"leakage: the flat prompt {text[:60]!r} carries no "
                f"{SMILES_MARKER!r}; there is no SMILES to re-write")
        start += len(SMILES_MARKER)
        end = text.find("\n", start)
        mol = Chem.MolFromSmiles(text[start:end])
        if mol is None:
            raise EvalError(
                f"leakage: the flat prompt's SMILES {text[start:end]!r} does not "
                "parse, so its stereochemistry cannot be removed")
        with_stereo = Chem.MolToSmiles(mol, isomericSmiles=True)
        without = Chem.MolToSmiles(mol, isomericSmiles=False)
        changed = int(with_stereo != without)
        written = with_stereo if keep_stereo else without
        new_text = text[:start] + written + text[end:]
        new_texts = [new_text if n == prompt_node else t
                     for n, t in enumerate(item["text"])]

    out = dict(item)
    out["text"] = new_texts
    stub = Example(task=side.get("task", "_"), domain=side.get("domain", "_"),
                   split=side.get("split", "test"), arm=arm,
                   graph={"text": new_texts, "prompt_node": prompt_node,
                          "num_nodes": len(new_texts)},
                   question=side.get("question", "_"),
                   answer=side.get("answer", ""),
                   answer_kind=side.get("answer_kind", "token"),
                   key=side.get("key", "_"))
    rendered = render(stub, tokenizer, max_length=max_length)
    out["input_ids"] = [list(ids) for ids in rendered.input_ids]
    out["labels"] = _as_labels(item.get("labels"), rendered.labels)
    return out, changed


class _StereoSource:
    """One view of a split, with the stereo channel open or closed.

    Counts the rows it actually re-wrote (``changed``) as a set of indices rather
    than a running total, so a row fetched twice cannot inflate it. That count is
    what makes the "at chance" reading falsifiable: on a split with any non-zero
    answer some molecule carries an assigned centre, so a strip that changed
    nothing has failed to find the channel rather than found it closed.
    """

    def __init__(self, source, arm, keep_stereo, tokenizer, indices, pairs,
                 max_length):
        self._source = source
        self._arm = arm
        self._keep = bool(keep_stereo)
        self._tokenizer = tokenizer
        self._indices = list(indices)
        self._pairs = pairs
        self._max_length = int(max_length)
        self.changed = set()
        self.task = getattr(source, "task", "?")
        self.split = getattr(source, "split", "?")
        self.arm = arm
        self.pass_id = getattr(source, "pass_id", 0)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict:
        item, changed = _restereo_item(
            self._source[self._indices[i]], self._arm, self._keep,
            self._tokenizer, self._pairs, self._max_length)
        if changed:
            self.changed.add(i)
        return item


@register
class Leakage(BaseValidator):
    """The negative control: `stereo_assigned` with the parity channel closed.

    §1 designates this family as the suite's leakage detector — strip the parity
    tag and the answer becomes unknowable, so anything materially above the base
    rate is information arriving by a route the experiment did not intend. It is
    what caught the split defect of `molecules/PLAN.md` §3.2.10, where up to 73.6 %
    of test examples were exact duplicates of training items, and the suite has had
    no working version of it since `014`'s pool made the family single-answer.

    **This is not `016`, and the difference matters when reading the number.**
    `016` trains a second model with ``stereo_tags: off`` and compares two runs;
    this closes the channel at *evaluation* on the one model the run trains, which
    costs two scoring passes instead of a training run. The two catch the same
    thing — a memorised molecule is answered from memory whether or not its parity
    words are present, so contamination keeps the stripped score high — but they
    are not the same measurement, and a model trained with the tags on can also
    lose accuracy simply because its input moved. That asymmetry is in the safe
    direction: it can only push the stripped score *down*, toward the floor, so a
    stripped score above the line is still evidence and a stripped score at the
    line is weaker evidence of cleanliness than `016`'s would be.

    **The floor is measured, never assumed.** `base` is the majority-class share
    of the rows actually scored, `sigma` its sampling error at that n, and the
    verdict is ``stripped <= base + sigmas * sigma``. `016` pre-registered 0.774
    against a 0.732 base on the `bace,bbbp,tox21,lipo` pool; the generalist draws
    from the whole train-role partition, so the base rate is a different number and
    is computed here rather than inherited — a score without its own floor is what
    §9's defect table already has one entry for.

    **A single-answer split reports `void`, not a pass.** That is the `014` state:
    the family scores 1.000 in both arms and the off/on contrast has nothing to sit
    on. Reporting it as a pass would retire the detector exactly when it has
    stopped working.
    """

    name = "leakage"
    cadence = "milestone"
    needs = frozenset({"model", "tokenizer", "collator", "eval_sets", "registry"})
    protocol_version = "1"

    def keys(self, ctx=None) -> set:
        return {"tagged", "stripped", "gap", "base", "sigma", "line", "passed",
                "void", "n", "n_stripped"}

    def run(self, ctx) -> dict:
        from collections import Counter

        from ..schema import SIDECAR_KEY
        from .scorers import eval_indices, score_source

        task = str(self.option("task", LEAKAGE_TASK))
        split = str(self.option("split", "test"))
        source = ctx.sources(task).get(split)
        if source is None or not len(source):
            # The family is not in this run's mixture — which is a decision the
            # mixture is entitled to make, and not this validator's to report on.
            return {}

        spec = _spec(ctx, task)
        if spec.answer_kind != "token":
            raise EvalError(
                f"leakage: {task} is scored as {spec.answer_kind!r}; the detector "
                "reads an exact-match accuracy against a majority-class floor and "
                "has no reading for that answer kind")

        arm = ctx.arm or getattr(source, "arm", "flat")
        indices = eval_indices(len(source), self.option("max_samples", 500))
        max_length = int((ctx.config or {}).get(MAX_LENGTH, 512))
        pairs = _parity_pairs()

        answers = [(source[i].get(SIDECAR_KEY) or {}).get("answer")
                   for i in indices]
        counts = Counter(answers)
        n = len(indices)
        base = max(counts.values()) / n if n else float("nan")
        void = float(len(counts) <= 1)

        views = {keep: _StereoSource(source, arm, keep, ctx.tokenizer, indices,
                                     pairs, max_length)
                 for keep in (True, False)}
        scored = {keep: score_source(ctx.model, ctx.tokenizer, ctx.collator, view,
                                     spec, device=ctx.device, max_samples=None,
                                     batch_size=int(self.option("batch_size", 8)))
                  for keep, view in views.items()}

        n_stripped = len(views[False].changed)
        if not void and not n_stripped:
            raise EvalError(
                f"leakage: closing the stereo channel changed none of the {n} "
                f"{task} rows on the {arm} arm, yet the split has "
                f"{len(counts)} distinct answers — so some molecule carries an "
                "assigned centre and the strip failed to find it. The verdict "
                "would read as 'at chance' while the channel was still open.")

        tagged = float(scored[True]["em_accuracy"])
        stripped = float(scored[False]["em_accuracy"])
        sigma = (base * (1.0 - base) / n) ** 0.5 if n else float("nan")
        line = base + float(self.option("sigmas", 3.0)) * sigma
        return {
            f"{task}/tagged": tagged,
            f"{task}/stripped": stripped,
            f"{task}/gap": tagged - stripped,
            f"{task}/base": float(base),
            f"{task}/sigma": float(sigma),
            f"{task}/line": float(line),
            f"{task}/passed": float("nan") if void else float(stripped <= line),
            f"{task}/void": void,
            f"{task}/n": float(n),
            f"{task}/n_stripped": float(n_stripped),
        }


class _AnalysisConfig:
    """The two fields `analysis.write_per_example_report` reads off a run config."""

    def __init__(self, tier: str, max_spd: int):
        self._tier = tier
        self.max_spd = int(max_spd)

    def tier(self) -> str:
        return self._tier


class _Predictions:
    """``transformers``' ``PredictionOutput``, as much of it as the report reads."""

    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids
        self.metrics = {}


class _PredictShim:
    """``trainer.predict`` over a ``TaskSource``, with the tier's own unpacking.

    Tier A reduces the logits to argmax predictions aligned to the label
    positions; Tier B reduces them to ``(logit_yes, logit_no, true_token_id)``.
    Reading Tier B's array with Tier A's unpacking is exactly the defect that
    made this report fail 0-for-29 across the campaign, so the two paths are
    chosen here by the tier and never by chance.
    """

    def __init__(self, ctx, tier, yes_id, no_id, batch_size=8):
        self._ctx = ctx
        self._tier = tier
        self._yes_id, self._no_id = yes_id, no_id
        self._batch_size = batch_size

    def predict(self, dataset, metric_key_prefix=None):
        from ...experiments.molecules.evaluate import make_margin_preprocessor
        from ...utils import shift_logits_for_metrics

        from .scorers import teacher_forced

        preprocess = (make_margin_preprocessor(self._yes_id, self._no_id)
                      if self._tier == "B" else shift_logits_for_metrics)
        preds, labels = teacher_forced(
            self._ctx.model, self._ctx.collator, dataset, range(len(dataset)),
            device=self._ctx.device, batch_size=self._batch_size,
            preprocess=preprocess)
        if self._tier == "A":
            preds = preds.astype("int64")
        return _Predictions(preds, labels.astype("int64"))
