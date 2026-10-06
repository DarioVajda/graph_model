"""
D8.2 — one ``RunConfig``, one place.

Every knob a generalist run reads is a field here with a default, and
:meth:`RunConfig.validate` rejects the combinations that cannot work *before*
anything is built or allocated. The layout follows `molecules/config.py`, which
is the house shape for a one-run config: dataclass fields grouped by what they
describe, derived helpers at the bottom, and a ``validate`` that refuses rather
than warns.

Three things are specific to this config and worth stating once:

* **The mixture is named, not inlined.** ``mixture`` is the key of a preset in
  :data:`MIXTURES` and ``task_weights`` overrides individual weights as
  ``"mol/bace=0.03,mol/hiv=0.12"``. The reason is the sweep runner: a list of
  objects in a sweep config is a *bundle* (`sweep/README.md`), so a literal
  ``tasks: [{...}, {...}]`` would silently become a sweep axis. Naming the
  preset keeps every config key a scalar, which is what makes the whole config
  sweepable, and puts the seventeen weights somewhere they can carry the
  paragraph of `MOLECULE_GENERALIST.md` §2 that justifies them. The resolved
  entries — not the preset's name — are what the config hash and the registry
  see, so an override moves the hash.
* **``validators`` is named the same way**, for the same reason.
* **The budget is not a knob.** ``max_steps`` is 0 by default and the step count
  comes from ``registry.resolve``: three passes of the finite sources at their
  share fixes the number of examples (`MOLECULE_GENERALIST.md` §2). A non-zero
  ``max_steps`` overrides it and is for smoke runs, where a step count is the
  point.

**The config hash** (``state.json``, D8.2) is over this object with ``run_name``,
``output_dir``, ``results_dir`` and the Slurm fields excluded. Those change
between two jobs of the same run — a chain's second chunk, a re-submission on a
different partition — and a resume that read them as a discontinuity would
append a re-warm for a change in nothing. It is taken over the *resolved*
mixture and validator lists rather than over the preset names and the
``task_weights`` string, so a mixture written two ways hashes once and a weight
that actually moves moves it.

No torch and no transformers at import: ``validate`` mode resolves a whole
config on a login node.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, fields

MODEL_NAME = "meta-llama/Llama-3.2-1B"

#: Parameter-name substrings that select the graph-bias channel. The same list
#: `molecules/train.py` trains and `checkpoint.bias_norm` fingerprints.
ACTIVE_PARAMS = ("graph_bias",)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(_HERE, "results")
CONFIGS_DIR = os.path.join(_HERE, "configs")

#: The two directories that hold runnable configs, split by what a file is for
#: rather than how big it is — `configs/README.md` states the rule. `runs/` is
#: the campaign, reproduced by naming a file; `probes/` is everything that
#: answered a question once. `forks/` is deliberately not here: a fork overlay
#: is not a `RunConfig` and does not resolve as one.
RUNS_DIR = os.path.join(CONFIGS_DIR, "runs")
PROBES_DIR = os.path.join(CONFIGS_DIR, "probes")


def runnable_configs(configs_dir: str = CONFIGS_DIR) -> list:
    """Every shipped config that is meant to resolve as a ``RunConfig``.

    Discovery rather than a hand-kept list, so a config added to either
    directory is covered by `test_shipped_configs_validate` without anyone
    remembering to register it. ``forks/`` is excluded by construction.
    """
    out = []
    for sub in ("runs", "probes"):
        directory = os.path.join(configs_dir, sub)
        if not os.path.isdir(directory):
            continue
        out.extend(os.path.join(directory, name)
                   for name in os.listdir(directory)
                   if name.endswith((".json", ".jsonc")))
    return sorted(out)

#: Bias tokens whose dataset features the molecules adapter actually produces.
#: A strict subset of `src/models/bias.py`'s ``BIAS_TYPES``, so checking against
#: it is the tighter of the two checks — and it needs no torch import, which is
#: what keeps ``validate`` mode light.
WIRED_TOKENS = ("spd", "magnetic", "magnetic_shared")

#: Re-exported from `schema.py`, which owns them: an arm is a property of the
#: example format, and a config that accepts one the schema rejects fails after
#: the dataset is built rather than before.
from .schema import ARMS, FLAT_ARMS  # noqa: E402  (re-export, kept beside its use)

LOSS_NORMS = ("per_example", "per_token")

#: Fields excluded from the config hash: two jobs of one run differ in these.
UNHASHED_FIELDS = ("run_name", "output_dir", "results_dir")

#: Fields the *resolved* view in :meth:`RunConfig.to_dict` supersedes, and which
#: the hash therefore reads from that view instead. Keeping both would make a
#: no-op ``task_weights`` override — a task's own weight written out explicitly —
#: read as a different run, and two spellings of one mixture are one mixture.
DERIVED_FIELDS = ("mixture", "task_weights", "task_passes", "validators")


class ConfigError(ValueError):
    """A run configuration that cannot produce a defensible number."""


# ─────────────────────────────────────────────────────────────────────────────
# The mixture presets
# ─────────────────────────────────────────────────────────────────────────────

#: Molecule counts per Tier-B source, from `MOLECULE_GENERALIST.md` §1's table.
#: Only their *ratios* matter — they set the within-block temperature weighting —
#: so the round numbers of that table are used rather than a count that would
#: drift with a re-download.
TIER_B_SIZES = {"bace": 1_500, "bbbp": 2_000, "hiv": 41_000,
                "tox21": 78_000, "sider": 39_000}

#: `MOLECULE_GENERALIST.md` §2, block shares.
BLOCK_SHARES = {"tier_b": 0.40, "tier_a": 0.25, "chebi": 0.20, "g2s": 0.15}

#: The nine Tier-A families that train (the adapter's ``TIER_A_TRAIN_TASKS``,
#: restated here so a mixture can be printed without importing RDKit).
TIER_A_FAMILIES = (
    "ring_membership", "aromatic_ring", "ring_size", "ring_count",
    "fg_presence", "fg_count", "fg_atom_membership",
    "stereo_potential", "stereo_assigned",
)

#: The three the molecules package holds out of *every* mixture
#: (``HELD_OUT_TIER_A_TASKS`` + ``HELD_OUT_DATASETS``), restated here for the same
#: reason as `TIER_A_FAMILIES`: so a mixture can be built without importing RDKit.
#:
#: They appear in `TRANSFER_FOLDS` because the fold table is the study's task
#: partition, and a fold naming one of them is a no-op on the mixture — they were
#: never in it. `KFOLD_TRANSFER.md` §3 records what that costs: every fold's trunk
#: lacks all three, so they are a constant across the four trunks rather than a
#: per-fold difference, and the study reads them as a cross-trunk control.
HELD_OUT_EVERYWHERE = ("bond_path", "longest_chain", "clintox")

#: Finite sources get at most six passes.
#:
#: §2 wrote three, and three is what the budget rule turns into a problem: the
#: budget is ``min over finite corpora of (passes x train_size) / share``, and
#: within-block weight goes as ``size ** 0.5`` while the cap goes as ``size``, so
#: ``available / share`` scales as ``size ** 0.5`` and **the smallest corpus
#: always binds**. At three passes BBBP — 1,244 training molecules, 2.35 % of the
#: run — set the length of the whole campaign at 2,799 steps, and the large
#: corpora were nowhere near their own caps: HIV saw 0.52 epochs and Tox21 0.43.
#: A dataset should not shorten training for every other task merely by being
#: small.
#:
#: Six doubles the budget to 5,599 steps and takes HIV to 1.04 epochs and Tox21
#: to 0.86, with BBBP at exactly its cap. Raising BBBP alone would not have done
#: it — BACE simply inherits the binding role at 3.00 epochs and the budget moves
#: 12 % — so the cap moves for the finite corpora as a set.
#:
#: Six is a ceiling and was never the fix. The fix is on the sampling side and is
#: now implemented: ``budget_scale`` sets the run length and `registry.resolve`
#: **down-weights a corpus that cannot sustain its share for that long**, within
#: its block, instead of stopping the whole run when the smallest one runs out.
#: At ``budget_scale`` 1.0 the water-filling is a no-op and the shares below are
#: exactly what they have always been, which is what keeps every result measured
#: so far comparable. Six remains the per-corpus repeat ceiling and is overridable
#: per task (``task_passes``), which is how a budget the mixture cannot otherwise
#: sustain gets bought — explicitly, in the config, rather than silently.
CORPUS_PASSES = 6

#: Blocks, for the water-filling. A block is a design statement about what the
#: model should be — 40 % property prediction, 25 % structure, 20 % captioning,
#: 15 % generation — and it must NOT drift with the compute budget. So a clamped
#: corpus gives its share to its own block and to nowhere else: BACE and BBBP
#: hand theirs to HIV/Tox21/SIDER, and Tier B still holds 40 %.
#:
#: A block of one has nobody to redistribute to, so ChEBI-20 can only shrink and
#: spill the remainder across the mixture. That is allowed, and bounded: no block
#: may fall below ``registry.BLOCK_SHARE_FLOOR`` of its design share, which lets
#: ChEBI carry the mixture to 2.41x. BACE's own 1 % floor binds first at 2.28x,
#: so on this mixture a per-task floor always fires before the block floor does.
BLOCKS = {"tier_b": "tier_b", "tier_a": "tier_a", "chebi": "chebi", "g2s": "g2s"}


#: `KFOLD_TRANSFER.md` §3 — the four leave-cluster-out folds. A fold names the
#: tasks its trunk must **not** see, so `mol/<name>` for each of these is dropped
#: from that fold's mixture and the share is redistributed inside its own block.
#:
#: Grouping is by family and not at random, and the reason is the whole design:
#: hold out `ring_size` while `ring_count` and `ring_membership` stay in and the
#: measurement is "a near-twin was still in the trunk", which is both weaker and
#: unanswerable. A fold removes a capability whole.
#:
#: `bond_path`, `longest_chain` and `clintox` are listed in the folds they belong
#: to even though `molecules/data.py` holds them out of every mixture already.
#: Listing them costs nothing — dropping a task the mixture never had is a no-op —
#: and it keeps the fold table readable as the study's task partition rather than
#: as a diff against whatever the adapter happens to hold out this month.
TRANSFER_FOLDS = {
    "A": ("ring_membership", "aromatic_ring", "ring_size", "ring_count",
          "bond_path", "longest_chain"),
    "B": ("fg_presence", "fg_count", "fg_atom_membership",
          "stereo_potential", "stereo_assigned"),
    "C": ("bace", "bbbp", "hiv", "tox21", "sider", "clintox"),
    "D": ("g2s", "chebi20"),
}


def molecule_generalist_mixture(exclude: tuple = ()) -> tuple:
    """`MOLECULE_GENERALIST.md` §2's mixture, computed from its own rule.

    Tier B is weighted by ``size ** 0.5`` within its block, which is where §2's
    "roughly BACE 5 %, BBBP 6 %, HIV 27 %, Tox21 37 %, SIDER 26 %" comes from.
    The rule is written out rather than the five percentages, so that changing a
    source's size or adding one produces the weights the document describes
    instead of a table that has quietly stopped matching it.

    Weights are absolute example shares; ``registry.resolve`` normalises them,
    so they are readable as fractions of the run and still survive a task being
    dropped from a config.

    ``exclude`` names bare tasks to leave out — one `KFOLD_TRANSFER.md` fold. The
    share is redistributed **inside the removed task's own block**, which is what
    `BLOCKS` means: a block is a design statement about what the model should be
    (40 % property prediction, 25 % structure, 20 % captioning, 15 % generation)
    and it must not drift because a fold took a task out of it. So the per-family
    Tier-A weight is recomputed over the survivors rather than left at 1/9, and
    Tier B's ``size ** 0.5`` rule is renormalised over the corpora that remain.

    **A fold that empties a whole block cannot hold that block's share**, because
    there is nobody inside it to redistribute to. Fold C removes all of Tier B
    and fold D removes ChEBI and g2s outright, so those two trunks come out with
    the remaining blocks proportionally enlarged — 25/20/15 becomes ~42/33/25 on
    C — and they are structurally different models rather than "the generalist
    minus a cluster". That is stated in `KFOLD_TRANSFER.md` §3 and it is why C and
    D are never pooled with A and B.

    With ``exclude`` empty this returns exactly what it always returned, so
    `008`'s ``mixture_hash`` does not move.
    """
    exclude = tuple(exclude)
    known = (tuple(TIER_B_SIZES) + TIER_A_FAMILIES + ("chebi20", "g2s")
             + tuple(HELD_OUT_EVERYWHERE))
    unknown = sorted(set(exclude) - set(known))
    if unknown:
        raise ConfigError(
            f"molecule_generalist_mixture: exclude names {unknown}, which is not "
            f"a task of this mixture (have {sorted(known)}). A fold that names a "
            "task nobody trains is a typo, not a no-op.")

    entries = []

    sizes = {n: s for n, s in TIER_B_SIZES.items() if n not in exclude}
    if sizes:
        root = {name: math.sqrt(size) for name, size in sizes.items()}
        total = sum(root.values())
        for name in sorted(sizes):
            entries.append({"name": f"mol/{name}",
                            "weight": BLOCK_SHARES["tier_b"] * root[name] / total,
                            "passes": CORPUS_PASSES, "block": BLOCKS["tier_b"],
                            # BACE and BBBP are two of the five sets this campaign
                            # reports, and they are also the two the water-filling
                            # thins first. A floor makes "the model barely trained on
                            # a benchmark it is scored on" a refusal rather than a
                            # number nobody looked at.
                            "floor": 0.01 if name in ("bace", "bbbp") else None})

    families = tuple(n for n in TIER_A_FAMILIES if n not in exclude)
    if families:
        per_family = BLOCK_SHARES["tier_a"] / len(families)
        for name in families:
            entries.append({"name": f"mol/{name}", "weight": per_family,
                            "block": BLOCKS["tier_a"]})

    if "chebi20" not in exclude:
        entries.append({"name": "mol/chebi20", "weight": BLOCK_SHARES["chebi"],
                        "passes": CORPUS_PASSES, "block": BLOCKS["chebi"]})
    if "g2s" not in exclude:
        entries.append({"name": "mol/g2s", "weight": BLOCK_SHARES["g2s"],
                        "block": BLOCKS["g2s"]})
    if not entries:
        raise ConfigError("molecule_generalist_mixture: exclude empties the "
                          "mixture; there is nothing left to train on.")
    return tuple(entries)


#: `MOLECULE_GENERALIST.md` §9.1: the replay share, pre-registered.
REPLAY_SHARE = 0.15

#: The second screen's share. At 0.15 the captions went away outright —
#: `caption_rate` 0.139 -> 0.000 with `kl_mean` halved — and the molecule suite
#: paid for it: the five-set property mean fell 9 seed-sd and g2s `exact_match`
#: 0.074. The overshoot on the text side is what makes a smaller share worth
#: measuring, and 0.08 gives the molecule gradient back about half of what 0.15
#: took from it.
REPLAY_SHARE_LOW = 0.08


def molecule_generalist_replay_mixture(share: float = REPLAY_SHARE) -> tuple:
    """§2's mixture at (1 - share), plus `text/replay` as a block of its own.

    Every molecule weight is scaled rather than one block being cut, so the four
    blocks keep the ratios §2 justifies and the molecule gradient loses exactly
    the replay share and nothing else.

    **The ``budget_scale`` that reproduces `008` is 2.0 x (1 - share)**, so 1.7
    at 0.15 and 1.84 at 0.08, not 2.0. The scale multiplies the mixture's own
    feasible budget, ``min(available / share)``, and scaling every corpus share
    by (1 - share) raises that base by 1 / (1 - share). At 2.0 the budget would
    be 2.35x §2's instead of 2x, BACE would be thinned to 0.0097 and refused at
    its 0.01 floor. At the scaled value the corpora are asked for exactly what
    `008` asked of them, so BACE and BBBP sit at the same pass caps.

    ``floor`` stays at 0.01 for the same reason: at the scaled budget the thinned
    corpora land on `008`'s absolute shares, not (1 - share) of them.
    """
    entries = [dict(e, weight=e["weight"] * (1.0 - share))
               for e in molecule_generalist_mixture()]
    entries.append({"name": "text/replay", "weight": share, "block": "replay"})
    return tuple(entries)


#: The smoke mixture: three maximally different tasks (D8/T10). ``mol/bace`` is
#: ``yesno`` and a corpus, ``mol/ring_size`` is ``token`` and a generator,
#: ``mol/g2s`` is ``smiles`` and a generator — so one 200-step run exercises the
#: teacher-forced margin readout, the teacher-forced exact match, generation,
#: both task kinds and both pass disciplines.
SMOKE_MIXTURE = (
    {"name": "mol/bace", "weight": 0.4, "passes": CORPUS_PASSES},
    {"name": "mol/ring_size", "weight": 0.3},
    {"name": "mol/g2s", "weight": 0.3},
)

#: The cross-check mixture: BACE alone, for 40 passes over its 1208 training
#: molecules — the specialist cell's budget, expressed as the harness expresses
#: budgets. `MOLECULE_GENERALIST.md`'s checklist asks for one specialist cell
#: trained *through this harness* as a single-task mixture, because until that
#: number lands where the molecules trainer's does, arm 2 minus arm 1 is a
#: difference between two trainers and not transfer.
#:
#: BACE is the cell to use: it is the smallest Tier-B corpus (0.75 h a run), it
#: is `yesno`, so the readout is the AUROC the campaign is scored on, and the
#: reference exists on both arms at exactly this recipe —
#: `026_lr3e4_lora_screen` seed 0, `rich_levi`, `question_node on`, `max_spd`
#: 32, `lora_r` 16, `lr` 3e-4: graph 0.8220, flat 0.8598.
CROSS_CHECK_MIXTURE = (
    {"name": "mol/bace", "weight": 1.0, "passes": 40},
)

#: The graph-to-SMILES specialist: `mol/g2s` and nothing else.
#:
#: §8 measured the graph arm at `exact_match` **0.0000** — 0 of 1500 attempts
#: over three seeds — and doubling the horizon left it at exactly 0.0000 while
#: validity fell 0.056 -> 0.040. Two readings survive that: the arm cannot
#: serialize a graph at 1B, or a 15 % share of a sixteen-task mixture is not
#: enough of the gradient to learn a generative task with. A specialist
#: separates them, because it is the second reading taken to its limit — one
#: task, the whole budget, the whole schedule.
#:
#: It is a mixture of one generator, so it has no finite source to set the
#: budget from and `max_steps` is not optional here: `registry.resolve` refuses a
#: mixture with no corpus in it unless a step count already bounds the run.
G2S_SPECIALIST_MIXTURE = (
    {"name": "mol/g2s", "weight": 1.0},
)

#: The ChEBI-20 specialist: `mol/chebi20` and nothing else.
#:
#: Captioning is the one molecule benchmark with a published ladder that the
#: campaign has never run as a specialist — `molecules/PLAN.md` §1 deferred Tier C
#: to the generalist, and the generalist only ever gave it a 20 % share of a
#: sixteen-task mixture. A specialist gives the task the whole budget, which is
#: what a number quoted against MolT5 has to be.
#:
#: ``passes`` is 15 rather than `CORPUS_PASSES`. Six is the ceiling that keeps one
#: small corpus from setting the length of a *mixed* run; on a mixture of one
#: there is nothing to protect, and six passes over 26k captions is less exposure
#: than the generalist's 20 % share bought over its own trunk. Fifteen is about
#: 2.7x that share, and `max_steps` bounds the run in any case.
CHEBI_SPECIALIST_MIXTURE = (
    {"name": "mol/chebi20", "weight": 1.0, "passes": 15},
)

#: The smoke mixture plus the two tasks the smoke run never reached: the
#: `admit` fork's candidate and the only ``text`` task in the campaign.
#:
#: It is never trained. It exists so `eval` mode can score the smoke checkpoint
#: on both of them, which settles two separate things at once:
#:
#:   * ChEBI is the campaign's only ``answer_kind: "text"`` task, so until it is
#:     scored once against a real model the caption path of `score_source` — the
#:     dispatch, the 256-token generation, `captions.caption_metrics` — has run
#:     only in unit tests on hand-written strings.
#:   * an admission verdict compares the child against the *parent*, and
#:     `check_admission` is honestly undecided without a parent number. This is
#:     where the parent's number for the candidate comes from; a fork whose
#:     `baseline_metrics` were left empty would exercise the undecided branch
#:     and nothing else.
#:
#: `mol/ring_count` is the candidate because it is a Tier-A generator: cheap to
#: build, ``token``-scored, not in the parent mixture and not held out — the
#: three things `_plan_admit` requires of a candidate.
SMOKE_PROBE_MIXTURE = SMOKE_MIXTURE + (
    {"name": "mol/ring_count", "weight": 0.2},
    {"name": "mol/chebi20", "weight": 0.2, "passes": CORPUS_PASSES},
)

#: The graph-domain smoke: one task from each of the six graph domains
#: (`GRAPH_GENERALIST.md` §2), which between them cover both task kinds and the
#: three answer kinds those domains add to the trunk — ``span`` (GraphQA, TAG),
#: ``yesno`` (probes, expressiveness, ``our_tests``) and ``entities`` (KGQA). It
#: is the mixture the adapters are verified on and is never a result: build it
#: with a small ``adapter_options[...]["limit"]`` so the build takes minutes.
#: `data_prep` also builds each domain's held-out tasks, as it does for molecules.
GRAPH_SMOKE_MIXTURE = (
    {"name": "graphqa/node_count", "weight": 0.2, "passes": CORPUS_PASSES},
    {"name": "probes/local_hop", "weight": 0.2},
    {"name": "expressiveness/hard", "weight": 0.1},
    {"name": "our_tests/kg_qa", "weight": 0.2},
    {"name": "kgqa/webqsp", "weight": 0.15, "passes": CORPUS_PASSES},
    {"name": "tag/cora", "weight": 0.15, "passes": CORPUS_PASSES},
)

MIXTURES = {
    "molecule_generalist": molecule_generalist_mixture(),
    # `KFOLD_TRANSFER.md` — one trunk mixture per fold, each missing that fold's
    # cluster. Named `fold_<id>` so a run directory and a `mixture_hash` both say
    # which trunk they came from without consulting the table.
    **{f"molecule_generalist_fold_{fold}":
       molecule_generalist_mixture(exclude=tasks)
       for fold, tasks in TRANSFER_FOLDS.items()},
    "molecule_generalist_replay": molecule_generalist_replay_mixture(),
    "molecule_generalist_replay08": molecule_generalist_replay_mixture(REPLAY_SHARE_LOW),
    "smoke": SMOKE_MIXTURE,
    "smoke_probe": SMOKE_PROBE_MIXTURE,
    "cross_check": CROSS_CHECK_MIXTURE,
    "g2s_specialist": G2S_SPECIALIST_MIXTURE,
    "chebi_specialist": CHEBI_SPECIALIST_MIXTURE,
    "graph_smoke": GRAPH_SMOKE_MIXTURE,
}


# ─────────────────────────────────────────────────────────────────────────────
# The validator presets
# ─────────────────────────────────────────────────────────────────────────────

#: D7.1's list, with two costs settled at config time rather than left at the
#: library defaults.
#:
#: ``in_mixture`` runs on ``milestone`` with ``max_samples: 500`` instead of
#: ``steps:500`` over the whole split. Uncapped it generates 3.3k ChEBI captions
#: and 1k SMILES strings every firing, which on a ~4k-step run is a large
#: fraction of the run spent measuring it. The *reportable* numbers are never
#: these — they come from the anneal fork's end-of-leg pass and from ``eval``
#: mode, both of which score the whole split.
#:
#: The cadence was ``steps:1000`` and is now ``milestone``, on the measurement
#: the comment above used to promise. **One firing costs over an hour**: the
#: arm-2 flat cells stalled at step 1000 for 65 minutes at `max_samples: 500`,
#: with `AveCPU` tracking wall clock the whole way, so that is work and not a
#: hang. At `steps:1000` over a 5,599-step run that is five firings — around six
#: hours of measurement against 1.6 hours of training on the flat arm, and worse
#: on the graph arm, where every row is 3.5x longer. Generation is what costs:
#: 500 ChEBI captions at 256 new tokens, on two splits, and sixteen tasks behind
#: them.
#:
#: ``milestone`` puts it on the same two firings as ``held_out``, ``base_exact``
#: and ``leakage``, so the whole expensive half of the suite fires together and
#: a run is measured twice rather than five times. What that costs is the
#: resolution of a *diagnostic* curve — `in_mixture` carries no ``end`` cadence,
#: so it was never the source of a reported number. What it buys is a campaign
#: that finishes inside its chunk instead of spilling across three.
#:
#: **The hour is now eight minutes, and the cadence should go back on the next
#: campaign.** The firing was slow because generation ran one example per
#: `generate` call, and it ran one at a time because a right-padded batch cannot
#: be continued (`MOLECULE_GENERALIST.md` §6). Batched, left-padded and on the
#: flex prefill it is 8.5x faster on the graph arm and 11.2x on the flat one,
#: and sharded across four ranks another 3.4x on top — so the 133-minute flat
#: firing measured here is about 12 minutes on one card. At that price
#: ``steps:1000`` costs a run under an hour of measurement and gives back the
#: five-point curve. It is **not** changed now: ``validator_specs`` is inside
#: ``config_hash``, so editing this tuple while the six arm-2 cells are live
#: would refuse their own resume. Change it with the next campaign's configs.
DEFAULT_VALIDATORS = (
    {"name": "in_mixture", "cadence": "milestone", "max_samples": 500},
    {"name": "held_out", "cadence": "milestone", "max_samples": 500},
    {"name": "bias_norm", "cadence": "steps:500"},
    {"name": "grad_share", "cadence": "steps:200"},
    {"name": "base_exact", "cadence": "milestone"},
    {"name": "perm_spread", "cadence": "end"},
    # Two teacher-forced passes over a capped `stereo_assigned` split, so it costs
    # about what one `held_out` firing does and runs on the same cadence. It is on
    # by default because the campaign's most expensive defect (§3.2.10) was found
    # by this control firing and nothing else, and the suite has been without it
    # since `014`.
    {"name": "leakage", "cadence": "milestone", "max_samples": 500},
    {"name": "throughput", "cadence": "steps:50"},
    {"name": "per_example", "cadence": "end"},
)

#: The smoke set. The same validators — T10 asserts that *every* validator ran —
#: at cadences a 200-step run reaches, and with sample caps that keep the
#: generative ones to seconds.
SMOKE_VALIDATORS = (
    {"name": "in_mixture", "cadence": "steps:100", "max_samples": 32},
    {"name": "held_out", "cadence": "milestone", "max_samples": 32},
    {"name": "bias_norm", "cadence": "steps:50"},
    {"name": "grad_share", "cadence": "steps:50"},
    {"name": "base_exact", "cadence": "milestone"},
    {"name": "perm_spread", "cadence": "end", "n_molecules": 8,
     "n_permutations": 4},
    # At 32 rows the verdict is unreadable — the line sits three sampling sigmas
    # out and sigma is 0.08 there — so what the smoke exercises is the path, not
    # the reading. The smoke mixture also does not carry `stereo_assigned`, in
    # which case the validator reports nothing at all, which is the third branch.
    {"name": "leakage", "cadence": "milestone", "max_samples": 32},
    {"name": "throughput", "cadence": "steps:25"},
    # No cap: `per_example` reports the whole split by construction and refuses
    # a `max_samples`. On the smoke mixture that is 152 bace + 1000 ring_size
    # rows, about a minute, and it is the file the `max_spd` question is
    # answered from.
    {"name": "per_example", "cadence": "end"},
)

#: The shakedown set: the default validators at **production sample counts**, on
#: cadences a few-hundred-step run reaches. The smoke set answers "did every
#: validator run"; this one answers "what does a firing cost", which the smoke
#: cannot, because a 32-sample generative pass is not a 500-sample one and
#: `in_mixture` never fires inside a short run at `steps:1000`. That number is
#: what decides whether the D7 cadences are affordable over a 2,799-step run, and
#: it is not derivable from anything already measured.
SHAKEDOWN_VALIDATORS = (
    {"name": "in_mixture", "cadence": "steps:100", "max_samples": 500},
    {"name": "held_out", "cadence": "milestone", "max_samples": 500},
    {"name": "bias_norm", "cadence": "steps:50"},
    {"name": "grad_share", "cadence": "steps:50"},
    {"name": "base_exact", "cadence": "milestone"},
    {"name": "perm_spread", "cadence": "end"},
    {"name": "leakage", "cadence": "milestone", "max_samples": 500},
    {"name": "throughput", "cadence": "steps:25"},
    {"name": "per_example", "cadence": "end"},
)

#: The default set minus `perm_spread`, for the canonical-only notation arms of
#: `MOLECULE_GENERALIST.md` §8.3 (`flat_selfies`, `flat_inchi`).
#:
#: **Dropped because it cannot be measured there, not because it is inconvenient.**
#: Property 1 is read as the spread of the margin across *re-orderings* of the
#: same molecule, and SELFIES and InChI are canonical by construction: there is no
#: re-ordered form of either, so there is nothing to sweep. The validator refuses
#: rather than returning a spread of zero — a zero would read as the tightest
#: possible Property-1 pass on an arm that never had the property — and this set
#: keeps that refusal from firing once per run. `leakage` stays: stripping
#: stereochemistry *is* expressible in every notation, so the campaign's leakage
#: detector keeps working here (`evaluate/builtin.py::_write_notation`).
NOTATION_VALIDATORS = tuple(
    spec for spec in DEFAULT_VALIDATORS if spec["name"] != "perm_spread")

#: The `g2s_specialist` set: what is left of the suite when the mixture is one
#: generative task.
#:
#: Six of the nine defaults have nothing to measure here and are dropped for that
#: reason rather than for cost. `base_exact` and `perm_spread` read a ``token``
#: or ``yesno`` margin and g2s is ``smiles``; `per_example` skips every other
#: kind by construction; `grad_share` compares a task's share of the gradient
#: against its configured weight, which on a mixture of one is 1.0 against 1.0;
#: `held_out` and `leakage` want tasks a g2s-only build never materialises, so
#: leaving them on would ask the run to build sixteen sources it does not train.
#:
#: `in_mixture` goes back to a step cadence, which the default set gave up when a
#: firing cost over an hour. It is affordable again — batched left-padded
#: generation is 8.5x faster on the graph arm and there is one task behind it
#: rather than sixteen — and the curve is the point of the run: whether validity
#: moves off the floor at all, and when. ``steps:1000`` rather than ``steps:500``
#: because g2s generates 256 new tokens a row and 500 rows a split, which is the
#: expensive half of what used to cost an hour; eleven firings over the trunk is
#: a readable curve at a few per cent of the run.
G2S_SPECIALIST_VALIDATORS = (
    {"name": "in_mixture", "cadence": "steps:1000", "max_samples": 500},
    {"name": "bias_norm", "cadence": "steps:500"},
    {"name": "throughput", "cadence": "steps:50"},
)

#: The `chebi_specialist` set. Same three survivors as `g2s_specialist`, and for
#: the same reasons: `base_exact` and `perm_spread` read a ``token`` or ``yesno``
#: margin and a caption is ``text``; `per_example` skips every other kind;
#: `grad_share` on a mixture of one compares 1.0 against 1.0; `held_out` and
#: `leakage` want tasks a ChEBI-only build never materialises.
#:
#: ``max_samples`` is 200 at ``steps:2000``, which is a *curve* and not a result.
#: The reported caption numbers come from `experiments/molecules/chebi_score.py` over the whole
#: 3,300-molecule split and are rescored offline under the published metric
#: protocol (`experiments/molecules/chebi_lit_metrics.py`) — a 500-row in-training firing was the
#: instrument this section exists to stop quoting. Keeping the cap low matters
#: here because a firing generates captions at 256 new tokens on two splits, which
#: is the expensive half of the suite.
CHEBI_SPECIALIST_VALIDATORS = (
    {"name": "in_mixture", "cadence": "steps:2000", "max_samples": 200},
    {"name": "bias_norm", "cadence": "steps:500"},
    {"name": "throughput", "cadence": "steps:50"},
)

#: `text_behaviour` alone, for `eval` mode over a checkpoint that is already
#: trained — the six instruct cells of `MOLECULE_GENERALIST.md` §7, which were trained before this
#: validator existed.
#:
#: **A separate set rather than an entry in `DEFAULT_VALIDATORS`, and that is not
#: a stylistic choice.** ``validator_specs`` is inside ``config_hash``
#: (`hash_payload` keeps the resolved list, not the preset name), so adding a
#: validator to the default set renames every run that used it and refuses their
#: own resume. A new key adds a set without touching the ones already spent.
#: `eval` mode reaches it through the generated ``--validators`` flag, which
#: overrides the field for that one job; the record it writes carries both the
#: config's hash and the checkpoint's, so the override is visible rather than
#: silent.
#:
#: Fold `text_behaviour` into the default set with the **next** campaign's
#: configs, the way `in_mixture`'s cadence is waiting to be — same rule, same
#: reason.
TEXT_VALIDATORS = (
    {"name": "text_behaviour", "cadence": "manual"},
)

VALIDATOR_SETS = {
    "default": DEFAULT_VALIDATORS,
    "smoke": SMOKE_VALIDATORS,
    "shakedown": SHAKEDOWN_VALIDATORS,
    "notation": NOTATION_VALIDATORS,
    "g2s_specialist": G2S_SPECIALIST_VALIDATORS,
    "chebi_specialist": CHEBI_SPECIALIST_VALIDATORS,
    "text": TEXT_VALIDATORS,
    "none": (),
}


# ─────────────────────────────────────────────────────────────────────────────
# RunConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """Every knob a generalist run reads. One knob, one place."""

    # ── identity and where it writes ─────────────────────────────────────────
    run_name: str = "molecule_generalist"
    #: Empty means ``<results_dir>/runs/<run_name>``; see :meth:`run_dir`.
    output_dir: str = ""
    results_dir: str = DEFAULT_RESULTS_DIR

    # ── arm, backbone and bias architecture ──────────────────────────────────
    #: ``graph`` is the ``rich_levi`` molecule graph; ``flat`` is the SMILES
    #: single-node twin, on which every graph bias vanishes (Property 2).
    arm: str = "graph"
    model_name: str = MODEL_NAME
    impl: str = "v2-flex"
    flex_compile_mode: str = "max-autotune-no-cudagraphs"
    bias: str = "spd+magnetic"
    max_spd: int = 32
    magnetic_dim: int = 32
    magnetic_q: float = 0.25
    magnetic_m: int = 0
    k_hop: int = 0
    k_hop_directed: bool = False
    lora: bool = True
    #: `MOLECULE_GENERALIST.md` §6: r16, the r32 axis is closed.
    lora_r: int = 16
    #: The molecules value, so arm 1 and arm 2 match. The trunk's 0.15 is not
    #: used here — a different regulariser would be an uncontrolled difference
    #: in exactly the comparison this campaign exists to make.
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False

    # ── data: the embedded molecules adapter config (D3) ─────────────────────
    encoding: str = "rich_levi"
    stereo_tags: bool = True
    question_node: str = "on"
    ordering: str = "rcm"
    max_length: int = 512
    #: How a turn is spelled: ``"plain"``, ``"chat"``, or ``None``/``"auto"`` for
    #: D3's pairing — chat iff the backbone is an Instruct variant. Naming it
    #: explicitly is the weights-vs-formatting control arm.
    prompt_style: str = None
    #: Supervise a stop token at the end of a generative answer. See
    #: `adapters.molecules.GENERATIVE_ANSWER_KINDS` for what its absence did to
    #: the graph arm's graph-to-SMILES column; the configs that reproduce a
    #: number measured before 2026-09-10 pin it to ``false``.
    answer_eos: bool = True
    tier_a_cap_per_pass: int = 4000
    tier_a_val_size: int = 500
    tier_a_test_size: int = 1000
    g2s_cap_per_pass: int = 4000
    g2s_val_size: int = 500
    g2s_test_size: int = 1000
    held_out_size: int = 1000
    chebi_heavy_atom_cap: int = 64
    chebi_allow_disconnected: bool = False
    #: Empty means the adapter's own default (``results/data``).
    cache_root: str = ""
    data_seed: int = 0
    #: Generator passes ``data_prep`` materialises. 0 means "as many as the
    #: resolved mixture will consume", which ``validate`` prints per task.
    generator_passes: int = 0

    # ── data: the text adapter (`adapters/text.py`) ──────────────────────────
    #: Only read when the mixture names a ``text/`` task, and only hashed then,
    #: so every molecules-only config keeps its hash.
    #:
    #: The node length for ``text/`` tasks. A prompt of up to 256 tokens and an
    #: answer of up to 768 do not fit the molecule nodes' 512, and ``max_length``
    #: is inside every molecule build hash, so the text task has its own.
    text_max_length: int = 1024
    #: The versioned prompts-and-answers directory under ``results/replay``.
    replay_version: str = "v1"

    # ── data: the graph domains (`adapters/_graph.py`) ───────────────────────
    #: ``{domain: {field: value}}`` overrides on a graph domain's adapter config —
    #: ``{"kgqa": {"strict_cross_dataset": true}, "probes": {"limit": 64}}``. Each
    #: domain carries its own node length and sizes (`adapters/<domain>.py`), so
    #: an empty dict is that domain's specialist settings. Hashed only for the
    #: domains the mixture names, so a molecules-only config keeps its hash.
    adapter_options: dict = None

    # ── mixture (D2, D4) ─────────────────────────────────────────────────────
    mixture: str = "molecule_generalist"
    #: ``"mol/bace=0.03,mol/hiv=0.12"`` — per-task weight overrides on the preset.
    task_weights: str = ""
    #: ``"mol/chebi20=7"`` — per-task repeat-cap overrides on the preset's
    #: ``CORPUS_PASSES``. This is how a budget the mixture cannot otherwise
    #: sustain gets bought: `resolve` refuses a ``budget_scale`` whose binding
    #: block is a single corpus, and names it, and raising that corpus's passes
    #: here is the explicit decision to repeat it more. In the config, in the
    #: hash, in the record — never inferred.
    task_passes: str = ""
    #: How much longer than the mixture's own feasible budget to train, as a
    #: multiple. 1.0 is that budget exactly and makes the water-filling a no-op,
    #: so every share is the preset's and every result measured so far stays
    #: comparable. Above 1.0, a corpus that cannot sustain its share for that long
    #: is down-weighted *within its block* rather than being allowed to end the
    #: run — which is the point: 1,244 BBBP molecules should not decide how long
    #: Tox21 trains for.
    budget_scale: float = 1.0
    #: D4.4: the effective batch, in tokens. ``batch_size`` is derived from it,
    #: never configured. The value is chosen from the smoke run's measured s/it
    #: (DESIGN.md §10), not from a round number.
    tokens_per_step: int = 16384
    loss_norm: str = "per_example"
    #: D2.2's floor: a task worth less than one example per this many steps is
    #: refused rather than left silently absent from the gradient. 0 disables it,
    #: which only a short smoke run has any business doing.
    min_examples_per: int = 1000
    #: 0 means the budget rule of `MOLECULE_GENERALIST.md` §2 sets the horizon.
    max_steps: int = 0

    # ── schedule (D5.2) ──────────────────────────────────────────────────────
    lr: float = 3e-4
    bias_lr: float = 1e-2
    #: Where an anneal fork decays to. §7: "decays to lr/10".
    lr_min: float = 3e-5
    warmup_steps: int = 200
    #: The re-warm a discontinuous resume appends (D5.4). Explicit rather than
    #: "the warmup length" so a chunk boundary's cost is a decision.
    rewarm_steps: int = 200
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # ── batching ─────────────────────────────────────────────────────────────
    #: Micro-batches per optimizer step. With ``tokens_per_step`` and the world
    #: size this fixes the per-micro-batch token budget (D4.4). Ignored when
    #: ``micro_batch_tokens`` is set, which derives it.
    accumulation_steps: int = 8
    #: Padded tokens one micro-batch may hold on one rank — a memory budget in
    #: the units the collator builds (`GRAPH_GENERALIST.md` §3). Set, it replaces
    #: ``accumulation_steps``: the sampler derives the accumulation that keeps
    #: the heaviest step inside it (`MixtureSampler.derive_accumulation_steps`).
    #: 0 keeps the configured accumulation, as every run before it did. Batching
    #: only regroups a step, so neither budget is hashed unless set.
    micro_batch_tokens: int = 0
    #: The second budget, ``rows x padded nodes²`` per micro-batch per rank — what
    #: the dense pair bias holds. 0 means no pair budget.
    micro_batch_node_pairs: int = 0
    #: Let a corpus retire part-way through the run instead of refusing to start
    #: (`MixtureSampler.check_supply`). A smoke run budgeted by a step count wants
    #: it; a trunk or a fork that runs out of a corpus is a different experiment.
    allow_exhaustion: bool = False

    # ── checkpointing (D5.3) ─────────────────────────────────────────────────
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10

    # ── evaluation (D7) ──────────────────────────────────────────────────────
    validators: str = "default"
    #: Fire the ``milestone`` validators every this many steps. 0 means never
    #: during training — the milestone set then runs only from a fork's end or
    #: from ``eval`` mode.
    milestone_steps: int = 0
    #: ``eval`` mode's override for the scoring validators' ``max_samples``.
    #: 0 leaves each validator's own option alone.
    eval_max_samples: int = 0
    #: D7.4: a training run does not select. The field exists so that a config
    #: that tries to is refused by name rather than ignored.
    selection: dict = None

    # ── seeds and tracking ───────────────────────────────────────────────────
    seed: int = 0
    wandb_project: str = None

    # ── Slurm (excluded from the config hash) ────────────────────────────────
    #: How the run is submitted. Recorded because a run record that cannot say
    #: what hardware produced it is missing the one thing a throughput number
    #: means anything against; excluded from the hash because a second chunk on
    #: a different partition is the same run.
    partition: str = "frida"
    account: str = "povejmo"
    gpus: str = "B200"
    gpus_per_config: int = 1
    cpus: int = 16
    mem: str = "128G"
    #: One chunk's walltime, sized to the *window* rather than the workload
    #: (`feedback-fit-jobs-to-window`).
    chunk_time: str = "24:00:00"
    #: Chunks the chain submits. Chunk 1 is ``train``, the rest ``resume``.
    chunks: int = 1
    chain_dependency: str = "afterany"
    container: str = ("/shared/workspace/povejmo/containers/"
                      "transformers_deepspeed_latest.sqsh")
    #: A compile cache shared across the chain's chunks
    #: (`project-ddp-flex-bucketing`). Empty means per-job.
    inductor_cache: str = ""

    #: Every field name above that the config hash ignores.
    SLURM_FIELDS = ("partition", "account", "gpus", "gpus_per_config", "cpus",
                    "mem", "chunk_time", "chunks", "chain_dependency",
                    "container", "inductor_cache")

    # ── derived: paths ───────────────────────────────────────────────────────

    def run_dir(self) -> str:
        """Where checkpoints land. ``output_dir`` wins; otherwise derived."""
        if self.output_dir:
            return os.path.abspath(self.output_dir)
        return os.path.abspath(os.path.join(self.results_dir, "runs", self.run_name))

    def lineage_dir(self) -> str:
        """Where ``lineage.json`` lives — one file per results tree, not per run."""
        return os.path.abspath(self.results_dir)

    def runs_jsonl(self) -> str:
        return os.path.join(self.lineage_dir(), "runs.jsonl")

    # ── derived: model ───────────────────────────────────────────────────────

    def bias_tokens(self) -> list:
        if self.bias.strip() == "none":
            return []
        return [t.strip() for t in self.bias.split("+") if t.strip()]

    def needs_spd(self) -> bool:
        return "spd" in self.bias_tokens()

    def needs_magnetic(self) -> bool:
        return bool({"magnetic", "magnetic_shared"} & set(self.bias_tokens()))

    def lora_config(self):
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout}

    def model_bias_config(self) -> dict:
        cfg = {}
        for token in self.bias_tokens():
            cfg[token] = True
        if self.needs_spd():
            cfg["max_spd"] = self.max_spd
        if self.needs_magnetic():
            cfg.update(magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        return cfg

    # ── derived: the mixture and the validators ──────────────────────────────

    def weight_overrides(self) -> dict:
        """``task_weights`` parsed. Raises on anything that is not ``name=float``."""
        out = {}
        for chunk in (self.task_weights or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, sep, raw = chunk.partition("=")
            if not sep or not name.strip():
                raise ConfigError(
                    f"task_weights: {chunk!r} is not 'name=weight'; the whole "
                    "field is a comma-joined list of those")
            try:
                out[name.strip()] = float(raw)
            except ValueError:
                raise ConfigError(
                    f"task_weights: {name.strip()} has weight {raw!r}, which is "
                    "not a number") from None
        return out

    def pass_overrides(self) -> dict:
        """``task_passes`` parsed. Raises on anything that is not ``name=int``."""
        out = {}
        for chunk in (self.task_passes or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            name, sep, raw = chunk.partition("=")
            if not sep or not name.strip():
                raise ConfigError(
                    f"task_passes: {chunk!r} is not 'name=passes'; the whole "
                    "field is a comma-joined list of those")
            try:
                passes = int(raw)
            except ValueError:
                raise ConfigError(
                    f"task_passes: {name.strip()} has passes {raw!r}, which is "
                    "not an integer") from None
            if passes < 1:
                raise ConfigError(
                    f"task_passes: {name.strip()} has passes {passes}; a corpus "
                    "is seen at least once")
            out[name.strip()] = passes
        return out

    def mixture_entries(self) -> tuple:
        """The preset with ``task_weights`` and ``task_passes`` applied.

        An override for a task the preset does not contain is an error: silently
        adding a task would put it in the gradient without it appearing in the
        document that justifies the mixture, and silently ignoring the override
        would leave a config saying something the run does not do.
        """
        try:
            preset = MIXTURES[self.mixture]
        except KeyError:
            raise ConfigError(
                f"mixture: {self.mixture!r} is not a preset (have "
                f"{sorted(MIXTURES)}). Presets live in config.py so the weights "
                "sit beside the paragraph that justifies them.") from None
        entries = [dict(e) for e in preset]
        names = {e["name"] for e in entries}
        overrides = self.weight_overrides()
        passes = self.pass_overrides()
        for field, given in (("task_weights", overrides), ("task_passes", passes)):
            unknown = sorted(set(given) - names)
            if unknown:
                raise ConfigError(
                    f"{field} names {unknown}, which the {self.mixture!r} mixture "
                    f"does not contain (it has {sorted(names)}). Add the task to "
                    "the preset if it belongs in the run.")
        for entry in entries:
            if entry["name"] in overrides:
                entry["weight"] = overrides[entry["name"]]
            if entry["name"] in passes:
                if "passes" not in entry:
                    raise ConfigError(
                        f"task_passes names {entry['name']}, which is a generator "
                        "in this mixture — it draws a fresh pass every time and "
                        "has no repeat cap to raise (D4.2).")
                entry["passes"] = passes[entry["name"]]
        return tuple(entries)

    def validator_specs(self) -> tuple:
        """The validator list, with ``eval_max_samples`` applied if it is set."""
        try:
            specs = VALIDATOR_SETS[self.validators]
        except KeyError:
            raise ConfigError(
                f"validators: {self.validators!r} is not a preset (have "
                f"{sorted(VALIDATOR_SETS)})") from None
        out = []
        for spec in specs:
            spec = dict(spec)
            if self.eval_max_samples and "max_samples" in spec:
                spec["max_samples"] = int(self.eval_max_samples)
            out.append(spec)
        return tuple(out)

    # ── derived: the adapter config (D3) ─────────────────────────────────────

    def adapter_config(self):
        """The embedded :class:`MoleculeAdapterConfig`.

        Imported here rather than at module scope: it pulls RDKit through its
        own ``validate``, and this module is imported by ``__main__`` before a
        mode is even chosen.
        """
        from .adapters.molecules import DEFAULT_CACHE_ROOT, MoleculeAdapterConfig

        return MoleculeAdapterConfig(
            encoding=self.encoding, stereo_tags=self.stereo_tags,
            question_node=self.question_node, ordering=self.ordering,
            magnetic_q=self.magnetic_q, magnetic_m=self.magnetic_m,
            max_spd=self.max_spd, model_name=self.model_name,
            max_length=self.max_length, answer_eos=self.answer_eos,
            prompt_style=self.prompt_style,
            tier_a_cap_per_pass=self.tier_a_cap_per_pass,
            tier_a_val_size=self.tier_a_val_size,
            tier_a_test_size=self.tier_a_test_size,
            g2s_cap_per_pass=self.g2s_cap_per_pass,
            g2s_val_size=self.g2s_val_size, g2s_test_size=self.g2s_test_size,
            held_out_size=self.held_out_size,
            chebi_heavy_atom_cap=self.chebi_heavy_atom_cap,
            chebi_allow_disconnected=self.chebi_allow_disconnected,
            data_seed=self.data_seed,
            cache_root=self.cache_root or DEFAULT_CACHE_ROOT,
        )

    def has_text_tasks(self) -> bool:
        """Whether the mixture names a ``text/`` task, which is what brings the
        text adapter into the registry, the build and the hash."""
        from .registry import TEXT_PREFIX

        return any(e["name"].startswith(TEXT_PREFIX) for e in self.mixture_entries())

    def text_adapter_config(self):
        """The :class:`TextAdapterConfig` for this run's backbone and format."""
        from .adapters.text import DEFAULT_CACHE_ROOT, TextAdapterConfig

        return TextAdapterConfig(
            model_name=self.model_name, prompt_style=self.prompt_style,
            max_length=self.text_max_length, magnetic_q=self.magnetic_q,
            magnetic_m=self.magnetic_m, replay_version=self.replay_version,
            cache_root=self.cache_root or DEFAULT_CACHE_ROOT)

    def graph_domains(self, extra_tasks=()) -> tuple:
        """The graph domains the mixture (or ``extra_tasks``) names, sorted."""
        from .adapters import GRAPH_DOMAINS

        names = [e["name"] for e in self.mixture_entries()] + list(extra_tasks)
        return tuple(sorted(d for d in GRAPH_DOMAINS
                            if any(n.startswith(f"{d}/") for n in names)))

    def domain_adapter_config(self, domain: str):
        """The adapter config for one graph domain.

        The run's backbone, format, magnetic settings, ordering, stop-token rule
        and seed, over the domain's own defaults, then ``adapter_options[domain]``.
        ``max_length`` is not taken from the run: it is the molecule node length,
        and each domain carries the one its specialist used.
        """
        from dataclasses import fields as dc_fields

        from .adapters import get_adapter
        from .adapters._graph import DEFAULT_CACHE_ROOT

        module = get_adapter(domain)
        cls = module.DOMAIN_SPEC.config_class
        options = dict((self.adapter_options or {}).get(domain) or {})
        known = {f.name for f in dc_fields(cls)}
        unknown = sorted(set(options) - known)
        if unknown:
            raise ConfigError(
                f"adapter_options[{domain!r}]: {unknown} are not fields of "
                f"{cls.__name__} (have {sorted(known)})")
        values = dict(model_name=self.model_name, prompt_style=self.prompt_style,
                      magnetic_q=self.magnetic_q, magnetic_m=self.magnetic_m,
                      ordering=self.ordering, answer_eos=self.answer_eos,
                      data_seed=self.data_seed,
                      cache_root=self.cache_root or DEFAULT_CACHE_ROOT)
        values.update(options)
        return cls(**values)

    # ── derived: the schedule ────────────────────────────────────────────────

    def decay_min_factor(self) -> float:
        """Where an anneal decays to, as a factor on ``lr`` (the schedule's unit)."""
        return float(self.lr_min) / float(self.lr)

    # ── the hash (D8.2) ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """A JSON-serialisable view, with the derived mixture spelled out.

        The *resolved* entries rather than the preset name, because two configs
        naming the same preset with different ``task_weights`` are two different
        runs and the hash has to say so.
        """
        out = asdict(self)
        out["mixture_entries"] = [dict(e) for e in self.mixture_entries()]
        out["validator_specs"] = [dict(s) for s in self.validator_specs()]
        return out

    def hash_payload(self) -> dict:
        """:meth:`to_dict` minus the fields two jobs of one run may differ in.

        Also minus the three fields the resolved entries supersede: the hash is
        over what the run *does*, so a mixture written two ways hashes once.
        """
        drop = (set(UNHASHED_FIELDS) | set(self.SLURM_FIELDS)
                | set(DERIVED_FIELDS))
        payload = {k: v for k, v in self.to_dict().items() if k not in drop}
        # `block` and `floor` describe how the water-filling REALLOCATES a share
        # it has to take away, so they are inert unless something is clamped, and
        # nothing is clamped below `budget_scale` 1.0 — which is itself hashed. A
        # run that draws different data therefore still hashes differently, and
        # annotating the preset does not retroactively make every finished run
        # look like a different one.
        payload["mixture_entries"] = [
            {k: v for k, v in entry.items() if k not in ("block", "floor")}
            for entry in payload.get("mixture_entries", [])
        ]
        # Same rule, same reason: at 1.0 the water-filling clamps nothing and the
        # run draws exactly what a config written before the field existed drew.
        # Hashing it unconditionally would say two identical runs are different.
        if payload.get("budget_scale") == 1.0:
            payload.pop("budget_scale")
        # And again: ``answer_eos: false`` is what every run before 2026-09-10
        # trained under, so a config that pins it draws byte-identical data to
        # one written before the field existed. Hashing it would rename six
        # finished cells and refuse their own resume.
        if payload.get("answer_eos") is False:
            payload.pop("answer_eos")
        # The padded budgets regroup a step and draw nothing differently; unset,
        # they are every run before they existed. `allow_exhaustion` only decides
        # whether a run that would retire a corpus starts at all.
        for name in ("micro_batch_tokens", "micro_batch_node_pairs"):
            if not payload.get(name):
                payload.pop(name, None)
        payload.pop("allow_exhaustion", None)
        # The text adapter's knobs change nothing a run without a `text/` task
        # draws, so they are hashed only when one is in the mixture.
        if not self.has_text_tasks():
            payload.pop("text_max_length", None)
            payload.pop("replay_version", None)
        # Same rule for the graph domains: only the options of a domain this
        # mixture draws from change what it draws.
        domains = self.graph_domains()
        options = {d: o for d, o in (payload.pop("adapter_options", None) or {}).items()
                   if d in domains and o}
        if options:
            payload["adapter_options"] = options
        # Same rule as `MoleculeAdapterConfig.build_version`: hash the *resolved*
        # prompt style, and only when it is not the plain one every run before
        # this field existed used.
        from .adapters.molecules import resolved_prompt_style

        style = resolved_prompt_style(self)
        payload.pop("prompt_style", None)
        if style != "plain":
            payload["prompt_style"] = style
        return payload

    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.hash_payload(), sort_keys=True,
                       separators=(",", ":"), default=str).encode()).hexdigest()

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self) -> "RunConfig":
        """Refuse, before any data is built or any GPU is allocated.

        Everything here is checkable without torch, without a built dataset and
        without the raw CSVs, so a config is checkable at the moment it is
        written rather than at the moment a job starts.
        """
        from .evaluate import build_validators, check_selection

        if self.arm not in ARMS:
            raise ConfigError(f"arm: {self.arm!r} is not one of {ARMS}")
        if self.loss_norm not in LOSS_NORMS:
            raise ConfigError(
                f"loss_norm: {self.loss_norm!r} is not one of {LOSS_NORMS}")

        # Property 2: the flat arm is a single-node graph, where every structural
        # bias is identically zero. Letting a bias arm ride along would advertise
        # a comparison that is not happening.
        tokens = self.bias_tokens()
        if self.arm in FLAT_ARMS and self.bias.strip() != "none":
            raise ConfigError(
                f"the flat arms are single-node graphs, where every graph bias "
                f"vanishes by construction (Property 2). Use bias 'none' on "
                f"{self.arm!r} so the run record cannot imply a bias was in play.")
        if self.bias.strip() != "none" and not tokens:
            raise ConfigError(
                f"bias: {self.bias!r} is empty; use 'none' for the no-bias arm")
        if len(tokens) != len(set(tokens)):
            raise ConfigError(f"bias: duplicate token in {self.bias!r}")
        for token in tokens:
            if token not in WIRED_TOKENS:
                raise ConfigError(
                    f"bias: {token!r} is not one of the wired tokens "
                    f"{WIRED_TOKENS}; the molecules adapter computes features "
                    "for those only")
        if "magnetic" in tokens and "magnetic_shared" in tokens:
            raise ConfigError("bias: pick one of 'magnetic' / 'magnetic_shared'")

        if self.impl not in ("v2-flex", "v2-eager"):
            raise ConfigError(f"impl: {self.impl!r} is not 'v2-flex' or 'v2-eager'")
        if self.lora and self.lora_r < 1:
            raise ConfigError(f"lora_r: must be >= 1, got {self.lora_r}")

        if self.tokens_per_step < 1:
            raise ConfigError(
                f"tokens_per_step: must be a positive int, got "
                f"{self.tokens_per_step}. It is the effective batch (D4.4) and "
                "the batch size is derived from it.")
        if self.accumulation_steps < 1:
            raise ConfigError(
                f"accumulation_steps: must be >= 1, got {self.accumulation_steps}")
        if self.max_steps < 0:
            raise ConfigError(
                f"max_steps: must be >= 0 (0 = the mixture's own budget), got "
                f"{self.max_steps}")
        if not math.isfinite(self.budget_scale) or self.budget_scale <= 0:
            raise ConfigError(
                f"budget_scale: must be a positive finite number (1.0 = the "
                f"mixture's own feasible budget), got {self.budget_scale!r}")
        self.pass_overrides()               # parses, or raises here rather than at resolve
        if self.min_examples_per < 0:
            raise ConfigError("min_examples_per: must be >= 0")
        if self.micro_batch_tokens < 0 or self.micro_batch_node_pairs < 0:
            raise ConfigError(
                "micro_batch_tokens and micro_batch_node_pairs: must be >= 0 "
                "(0 = unset)")
        if self.micro_batch_node_pairs and not self.micro_batch_tokens:
            raise ConfigError(
                "micro_batch_node_pairs needs micro_batch_tokens: the pair budget "
                "sizes a derived accumulation, and without micro_batch_tokens the "
                "accumulation is configured")
        if self.generator_passes < 0:
            raise ConfigError("generator_passes: must be >= 0")

        if not (self.lr > 0 and self.bias_lr > 0):
            raise ConfigError(
                f"lr and bias_lr must both be positive, got {self.lr} and "
                f"{self.bias_lr}")
        if not 0 < self.lr_min < self.lr:
            raise ConfigError(
                f"lr_min: must satisfy 0 < lr_min < lr, got {self.lr_min} against "
                f"lr {self.lr}. It is where an anneal fork lands, so a value at or "
                "above lr would make the anneal a warm-up.")
        if self.warmup_steps < 0:
            raise ConfigError("warmup_steps: must be >= 0")
        if self.rewarm_steps < 1:
            raise ConfigError(
                "rewarm_steps: must be >= 1. A discontinuous resume needs a "
                "re-warm length (D5.2) and a schedule with neither this nor a "
                "warmup segment is an error rather than a guess.")

        if self.save_steps < 1:
            raise ConfigError("save_steps: must be >= 1")
        if self.save_total_limit < 1:
            raise ConfigError(
                "save_total_limit: must be >= 1; a chain resumes from the last "
                "complete checkpoint and keeping none would end the run")
        if self.logging_steps < 1:
            raise ConfigError("logging_steps: must be >= 1")
        if self.milestone_steps < 0:
            raise ConfigError("milestone_steps: must be >= 0 (0 = never)")
        if self.chunks < 1:
            raise ConfigError("chunks: must be >= 1")
        if self.seed < 0 or self.data_seed < 0:
            raise ConfigError("seed and data_seed must both be >= 0")

        # D7.4. A training run does not select, so `selection` set at all is the
        # error — check_selection refuses it by name and says why.
        check_selection(self.selection, mode="train")

        # Both resolve their presets and raise on a typo; `build_validators`
        # additionally rejects an unknown validator name and a bad cadence here,
        # on the login node, rather than at step 500 of a GPU job.
        self.mixture_entries()
        build_validators(self.validator_specs())

        # The embedded adapter config's own checks (encoding, question_node, the
        # source names). Needs RDKit but no torch and no built data.
        self.adapter_config().validate()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Loading a config file
# ─────────────────────────────────────────────────────────────────────────────

#: Keys a sweep config carries for the runner rather than for the run.
RESERVED_KEYS = ("name", "execution", "chain")

#: ``execution.sbatch`` key -> ``RunConfig`` field. The sweep runner owns that
#: block's vocabulary (`sweep/README.md`) and the chain script needs the same
#: numbers, so it is read *into* the config rather than read a second time from
#: the file. One consequence worth stating: the Slurm fields a run record shows
#: are then the ones the job actually asked for.
SBATCH_TO_FIELD = {
    "partition": "partition", "account": "account", "cpus": "cpus",
    "mem": "mem", "time": "chunk_time", "container": "container",
    "inductor_cache": "inductor_cache", "gpus_per_config": "gpus_per_config",
}

_FIELD_NAMES = frozenset(f.name for f in fields(RunConfig))


def _fields_from(path: str, run: dict, meta: dict) -> dict:
    """One expanded run, as a dict of ``RunConfig`` field values.

    Reserved runner keys are dropped; ``name`` becomes ``run_name``, which is the
    one place the two vocabularies differ. The reserved ``execution.sbatch`` and
    ``chain`` blocks are folded onto the Slurm fields, so a config says how it is
    submitted once. An explicit field wins over the block, which is what makes an
    override possible without editing what the sweep runner reads.

    A key that is neither reserved nor a field is an error. A config with a typo
    in it is otherwise a job that runs to completion with a default nobody chose.
    """
    out = {}
    if "name" in meta:
        out["run_name"] = meta["name"]
    if "results_dir" in meta:
        out["results_dir"] = meta["results_dir"]

    for key, value in run.items():
        if key == "name":
            # Only reachable from inside a bundle; the top-level `name` is the
            # file's and arrives above. Mapped rather than dropped, so a cell
            # that names itself is not silently ignored.
            out["run_name"] = value
        elif key in RESERVED_KEYS:
            continue
        elif key in _FIELD_NAMES:
            out[key] = value
        else:
            raise ConfigError(
                f"{path}: {key!r} is not a RunConfig field and is not one of the "
                f"reserved keys {RESERVED_KEYS}. Fields are "
                f"{sorted(_FIELD_NAMES)}.")

    sbatch = ((meta.get("execution") or {}).get("sbatch") or {})
    for key, field_name in SBATCH_TO_FIELD.items():
        if key in sbatch:
            out.setdefault(field_name, sbatch[key])
    if "gpus" in sbatch:
        # A list names several acceptable node features; the constraint that
        # renders from it is `|`-joined (`sweep/README.md`).
        gpus = sbatch["gpus"]
        out.setdefault("gpus", "|".join(str(g) for g in gpus)
                       if isinstance(gpus, list) else str(gpus))
    chain = run.get("chain") or {}
    if "chunks" in chain:
        out.setdefault("chunks", int(chain["chunks"]))
    if "dependency" in chain:
        out.setdefault("chain_dependency", str(chain["dependency"]))
    return out


def config_cells(path: str) -> dict:
    """Every run a ``.jsonc`` config resolves to, as ``cell name -> field values``.

    The sweep runner's own loader is reused (`sweep/expand.py`), so a file that
    ``python -m sweep`` accepts and a file that ``--config`` accepts are the same
    file — and a file that expands to several runs there expands to the same
    several here. A list value is an axis and a list of objects is a bundle of
    keys that vary together; a file with neither resolves to exactly one cell,
    which is every probe config and was every run config before the arm-2
    campaign was merged into one.

    **A cell's name is its ``run_name``, plus ``_s<seed>`` when the file sweeps
    the seed.** That rule is not a convenience: a campaign's cells have to be
    addressable one at a time — a chain is submitted per cell, and so is an
    anneal fork — and the suffix is the convention every run directory on disk
    already follows. Names must come out distinct, because two cells sharing one
    would share an output directory and quietly overwrite each other.
    """
    from sweep.expand import SweepError, expand, load_config, split_meta

    raw = load_config(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a config must be a JSON object")
    meta, sweep = split_meta(raw)
    try:
        runs = expand(sweep)
    except SweepError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    seeds = {run.get("seed") for run in runs}
    suffix_seed = len(runs) > 1 and len(seeds) > 1

    cells = {}
    for run in runs:
        values = _fields_from(path, run, meta)
        name = values.get("run_name")
        if not name:
            raise ConfigError(f"{path}: a cell has no name; give the file a "
                              f"'name', or every bundle object a 'run_name'.")
        if suffix_seed:
            name = f"{name}_s{values.get('seed')}"
        values["run_name"] = name
        if name in cells:
            raise ConfigError(
                f"{path}: two cells both resolve to the name {name!r}, so they "
                f"would share an output directory. Give the axis that separates "
                f"them a distinct 'run_name'.")
        cells[name] = values
    return cells


def load_config_file(path: str, cell: str = None) -> dict:
    """A ``.jsonc`` config as a dict of ``RunConfig`` field values.

    ``cell`` names which of a multi-cell config's runs to resolve. A file that
    holds exactly one run needs no name and refuses one that does not match; a
    file that holds several refuses to guess, because picking a cell for the
    caller is picking which run the numbers came from.
    """
    cells = config_cells(path)
    if cell is None:
        if len(cells) == 1:
            return next(iter(cells.values()))
        raise ConfigError(
            f"{path} resolves to {len(cells)} cells; name one with --cell. "
            f"Cells: {', '.join(sorted(cells))}")
    if cell not in cells:
        raise ConfigError(
            f"{path} has no cell {cell!r}. Cells: {', '.join(sorted(cells))}")
    return cells[cell]


def shell_assignments(config: "RunConfig") -> str:
    """The chain script's view of a config, as ``GEN_<KEY>='value'`` lines.

    A shell script needs a dozen values out of a JSONC file, and every way of
    getting them in bash alone is a way of getting them slightly wrong. This is
    the one place the two languages meet, and it is single-quoted so nothing in
    a config can become a command.
    """
    values = {
        "RUN_NAME": config.run_name,
        "RUN_DIR": config.run_dir(),
        "RESULTS_DIR": config.lineage_dir(),
        "CONFIG_HASH": config.config_hash(),
        "PARTITION": config.partition,
        "ACCOUNT": config.account,
        "GPUS": config.gpus,
        "GPUS_PER_CONFIG": config.gpus_per_config,
        "CPUS": config.cpus,
        "MEM": config.mem,
        "TIME": config.chunk_time,
        "CHUNKS": config.chunks,
        "DEPENDENCY": config.chain_dependency,
        "CONTAINER": config.container,
        "INDUCTOR_CACHE": config.inductor_cache,
    }
    lines = []
    for key, value in values.items():
        text = str(value).replace("'", "'\"'\"'")
        lines.append(f"GEN_{key}='{text}'")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# --init
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE = """\
{
  // ─────────────────────────────────────────────────────────────────────────
  // Generalist run config (JSONC: // comments and trailing commas allowed).
  //
  //   python3 -m src.generalist validate  --config <this file>
  //   python3 -m src.generalist data_prep --config <this file>
  //   python3 -m src.generalist train     --config <this file>
  //   src/generalist/tools/launch/chain.sh <this file> # chunked, on Slurm
  //
  // Every key is a RunConfig field (src/generalist/config.py) and every value
  // is a scalar, so the same file is a sweep config:
  //   python3 -m sweep src.generalist <this file>
  // A list value makes a key a sweep AXIS. The mixture and the validator set
  // are named presets rather than inline lists for exactly that reason — a list
  // of objects is a sweep bundle, not a mixture.
  // ─────────────────────────────────────────────────────────────────────────

  "name": "%(name)s",
  "results_dir": "src/generalist/results",

  "execution": {
    "mode": "sbatch",
    "sbatch": {
      "granularity": "per_config",
      "max_concurrent": 1,
      "partition": "frida",
      "account": "povejmo",
      "gpus": "B200",
      "gpus_per_config": 1,
      "cpus": 16,
      "mem": "128G",
      "time": "24:00:00",
      "inductor_cache": ".inductor_cache/generalist",
      "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
    }
  },

  // ── what the run is ───────────────────────────────────────────────────────
  "arm": "graph",                    // "graph" | "flat" (flat needs bias "none")
  "mixture": "molecule_generalist",  // a preset in config.py MIXTURES
  "task_weights": "",                // "mol/bace=0.03,mol/hiv=0.12"
  "validators": "default",           // a preset in config.py VALIDATOR_SETS

  // ── model and bias ────────────────────────────────────────────────────────
  "model_name": "meta-llama/Llama-3.2-1B",
  "impl": "v2-flex",
  "bias": "spd+magnetic",
  "max_spd": 32,
  "lora_r": 16,
  "lora_dropout": 0.05,

  // ── data ──────────────────────────────────────────────────────────────────
  "encoding": "rich_levi",
  "stereo_tags": true,
  "question_node": "on",
  "data_seed": 0,

  // ── mixture and schedule ──────────────────────────────────────────────────
  "tokens_per_step": 16384,
  "accumulation_steps": 8,
  "lr": 3e-4,
  "bias_lr": 1e-2,
  "lr_min": 3e-5,
  "warmup_steps": 200,
  "rewarm_steps": 200,
  "weight_decay": 0.1,

  // ── checkpointing and logging ─────────────────────────────────────────────
  "save_steps": 500,
  "save_total_limit": 3,
  "logging_steps": 10,
  "milestone_steps": 2000,

  "seed": 0,
  "wandb_project": null
}
"""


def write_template(name: str, configs_dir: str = PROBES_DIR) -> str:
    """``--init <name>``: a sweep config under ``configs/probes/``.

    Probes rather than runs, because a config that does not exist yet has not
    produced a number anyone quotes; a file earns its way into ``runs/`` by
    becoming the campaign, and moving it there is a deliberate act.
    """
    if not (name.endswith(".json") or name.endswith(".jsonc")):
        name += ".jsonc"
    os.makedirs(configs_dir, exist_ok=True)
    path = os.path.join(configs_dir, name)
    stem = os.path.basename(name).rsplit(".", 1)[0]
    with open(path, "w") as f:
        f.write(TEMPLATE % {"name": stem})
    return path
