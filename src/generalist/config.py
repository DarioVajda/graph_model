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

ARMS = ("graph", "flat")
LOSS_NORMS = ("per_example", "per_token")

#: Fields excluded from the config hash: two jobs of one run differ in these.
UNHASHED_FIELDS = ("run_name", "output_dir", "results_dir")

#: Fields the *resolved* view in :meth:`RunConfig.to_dict` supersedes, and which
#: the hash therefore reads from that view instead. Keeping both would make a
#: no-op ``task_weights`` override — a task's own weight written out explicitly —
#: read as a different run, and two spellings of one mixture are one mixture.
DERIVED_FIELDS = ("mixture", "task_weights", "validators")


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
#: This is a ceiling, not the fix. The right correction is on the sampling side:
#: down-weight a small corpus so it is drawn less often instead of capping the
#: run when it runs out. That changes the mixture shares every result so far was
#: measured under, so it waits for the next campaign rather than landing between
#: arm 1 and arm 2. Worth doing before the larger generalists.
CORPUS_PASSES = 6


def molecule_generalist_mixture() -> tuple:
    """`MOLECULE_GENERALIST.md` §2's mixture, computed from its own rule.

    Tier B is weighted by ``size ** 0.5`` within its block, which is where §2's
    "roughly BACE 5 %, BBBP 6 %, HIV 27 %, Tox21 37 %, SIDER 26 %" comes from.
    The rule is written out rather than the five percentages, so that changing a
    source's size or adding one produces the weights the document describes
    instead of a table that has quietly stopped matching it.

    Weights are absolute example shares; ``registry.resolve`` normalises them,
    so they are readable as fractions of the run and still survive a task being
    dropped from a config.
    """
    entries = []

    root = {name: math.sqrt(size) for name, size in TIER_B_SIZES.items()}
    total = sum(root.values())
    for name in sorted(TIER_B_SIZES):
        entries.append({"name": f"mol/{name}",
                        "weight": BLOCK_SHARES["tier_b"] * root[name] / total,
                        "passes": CORPUS_PASSES})

    per_family = BLOCK_SHARES["tier_a"] / len(TIER_A_FAMILIES)
    for name in TIER_A_FAMILIES:
        entries.append({"name": f"mol/{name}", "weight": per_family})

    entries.append({"name": "mol/chebi20", "weight": BLOCK_SHARES["chebi"],
                    "passes": CORPUS_PASSES})
    entries.append({"name": "mol/g2s", "weight": BLOCK_SHARES["g2s"]})
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

MIXTURES = {
    "molecule_generalist": molecule_generalist_mixture(),
    "smoke": SMOKE_MIXTURE,
    "smoke_probe": SMOKE_PROBE_MIXTURE,
    "cross_check": CROSS_CHECK_MIXTURE,
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

VALIDATOR_SETS = {
    "default": DEFAULT_VALIDATORS,
    "smoke": SMOKE_VALIDATORS,
    "shakedown": SHAKEDOWN_VALIDATORS,
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
    #: `MOLECULE_GENERALIST.md` §7: r16, the r32 axis is closed.
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

    # ── mixture (D2, D4) ─────────────────────────────────────────────────────
    mixture: str = "molecule_generalist"
    #: ``"mol/bace=0.03,mol/hiv=0.12"`` — per-task weight overrides on the preset.
    task_weights: str = ""
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
    #: size this fixes the per-micro-batch token budget (D4.4).
    accumulation_steps: int = 8

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

    def mixture_entries(self) -> tuple:
        """The preset with ``task_weights`` applied — what the registry resolves.

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
        unknown = sorted(set(overrides) - names)
        if unknown:
            raise ConfigError(
                f"task_weights names {unknown}, which the {self.mixture!r} mixture "
                f"does not contain (it has {sorted(names)}). Add the task to the "
                "preset if it belongs in the run.")
        for entry in entries:
            if entry["name"] in overrides:
                entry["weight"] = overrides[entry["name"]]
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
            max_length=self.max_length,
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
        return {k: v for k, v in self.to_dict().items() if k not in drop}

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
        if self.arm == "flat" and self.bias.strip() != "none":
            raise ConfigError(
                "the flat arm is a single-node graph, where every graph bias "
                "vanishes by construction (Property 2). Use bias 'none' on the "
                "flat arm so the run record cannot imply a bias was in play.")
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
        if self.min_examples_per < 0:
            raise ConfigError("min_examples_per: must be >= 0")
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


def load_config_file(path: str) -> dict:
    """A ``.jsonc`` config as a dict of ``RunConfig`` field values.

    The sweep runner's own loader is reused (`sweep/expand.py`), so a file that
    ``python -m sweep`` accepts and a file that ``--config`` accepts are the same
    file. Reserved runner keys are dropped; ``name`` becomes ``run_name``, which
    is the one place the two vocabularies differ.

    The reserved ``execution.sbatch`` and ``chain`` blocks are folded onto the
    Slurm fields, so a config says how it is submitted once. An explicit
    top-level field wins over the block, which is what makes an override
    possible without editing what the sweep runner reads.

    A key that is neither reserved nor a field is an error. A sweep config with a
    typo in it is otherwise a job that runs to completion with a default nobody
    chose.
    """
    from sweep.expand import load_config

    raw = load_config(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a config must be a JSON object")

    out = {}
    for key, value in raw.items():
        if key == "name":
            out["run_name"] = value
        elif key == "results_dir":
            out["results_dir"] = value
        elif key in RESERVED_KEYS:
            continue
        elif key in _FIELD_NAMES:
            out[key] = value
        else:
            raise ConfigError(
                f"{path}: {key!r} is not a RunConfig field and is not one of the "
                f"reserved keys {RESERVED_KEYS}. Fields are "
                f"{sorted(_FIELD_NAMES)}.")

    sbatch = ((raw.get("execution") or {}).get("sbatch") or {})
    for key, field_name in SBATCH_TO_FIELD.items():
        if key in sbatch:
            out.setdefault(field_name, sbatch[key])
    if "gpus" in sbatch:
        # A list names several acceptable node features; the constraint that
        # renders from it is `|`-joined (`sweep/README.md`).
        gpus = sbatch["gpus"]
        out.setdefault("gpus", "|".join(str(g) for g in gpus)
                       if isinstance(gpus, list) else str(gpus))
    chain = raw.get("chain") or {}
    if "chunks" in chain:
        out.setdefault("chunks", int(chain["chunks"]))
    if "dependency" in chain:
        out.setdefault("chain_dependency", str(chain["dependency"]))
    return out


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
  //   src/generalist/tools/chain.sh <this file>        # chunked, on Slurm
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
