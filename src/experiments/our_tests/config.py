"""
Shared configuration for the our_tests experiment (Family Tree + Knowledge-Graph QA).

One ``RunConfig`` dataclass holds EVERY knob the entry points read; ``__main__.py``
builds a ``RunConfig`` from argparse and stays a thin dispatcher, so a value lives in
exactly one place. This mirrors ``src/experiments/graphqa`` — see that module for the
reference implementation of the same contract.

Two synthetic tasks, each one graph per question with the answer supervised by exact
match:

* ``family``  — a generated family tree; questions ask about relatives/attributes.
* ``kg_qa``   — a generated knowledge graph (30-50 entities); multi-hop lookups.

Both are built as **incidence** (Levi) graphs: one node per entity AND one per
relation. Unlike graphqa there is no second encoding to compare, so ``graph_type`` is
not a config axis here.

PROTOCOL NOTE. The defaults below reproduce the recipe the paper's numbers were
produced with (recorded in ``run_metadata_graph.json``, entry ``1B_graph_family_lora_v5``:
lr 5e-5, bias_lr 1e-2, 10 epochs, LoRA r32). They are deliberately NOT changed to match
graphqa's refactored protocol: the point of the re-run (TODO.md §3) is to isolate the
graph-bias reload fix, so anything else that would move the numbers is held fixed.
"""

import os
from dataclasses import dataclass


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "our_tests"

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# The two datasets keep their historical on-disk locations: both are large, gitignored,
# and expensive to rebuild (generation + a magnetic eigendecomposition per graph), so a
# default config must resolve onto what is already there rather than regenerate it.
FAMILY_DIR = os.path.join(EXPERIMENT_DIR, "family_tree_graph_dataset")
KGQA_DIR = os.path.join(EXPERIMENT_DIR, "graph_datasets")

TASKS = ("family", "kg_qa")
IMPLS = ("v2-eager", "v2-flex")
DTYPES = ("fp32", "bf16")

# Where the question text lives (ported from graphqa / kgqa, same semantics).
#
# "off"      — the historical layout: ONE prompt node holding "Q: {q}\nA: {a}", so graph
#              tokens never see the question. Byte-identical to pre-feature builds.
# "isolated" — the question moves into its own QUESTION *prefix* node with NO edges; the
#              bidirectional prefix mask then lets every graph token attend to it
#              (question-conditioned graph encoding). The prompt node keeps the "A:"
#              anchor onward.
#
# CRITICAL (TODO.md §2): the prompt node must never be left empty before the answer —
# the first answer token would then have no in-node predecessor to be predicted from.
# Splitting on ANSWER_PREFIX guarantees the prompt node retains "A:" as that anchor, so
# the supervised span is byte-identical between the two modes and only the *visibility*
# of the question changes. That is what makes this a clean probe.
QUESTION_NODES = ("off", "isolated")

# Graph-bias features whose data + model modules are both wired here.
# The published per-task learning rates (paper table "Hyperparameter configuration
# used for training GTLM on our synthetic Family Tree and Knowledge Graph QA
# datasets"). Kept here as data so a run can be checked against them rather than
# relying on whoever writes the sweep config to remember that they differ.
PAPER_LEARNING_RATES = {
    "family": {"lr": 5e-5, "bias_lr": 1e-2},
    "kg_qa": {"lr": 5e-4, "bias_lr": 3e-2},
}

WIRED_FEATURES = ("spd", "rrwp", "magnetic")
# Exposed in the historical BIAS_PARAMS dict but never produced by data prep.
UNWIRED_FEATURES = ("laplacian", "rwse")

# Feature-generation settings the on-disk datasets were built with. A config matching
# these resolves to the historical, untagged cache path (see ``dataset_dir``).
_CACHE_DEFAULTS = dict(
    model_name=MODEL_NAME, max_length=32_768,
    magnetic_q=0.25, max_rw_steps=16,
    question_node="off",
)


@dataclass
class RunConfig:
    """Every knob the train / data_prep entry points read (built from argparse)."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"                          # "train" | "data_prep"
    task: str = "family"                         # "family" | "kg_qa"
    question_node: str = "off"                   # "off" | "isolated" (see QUESTION_NODES)

    # ── model ──────────────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    # v2-eager is pinned numerically equivalent to the legacy v0 path these datasets
    # were trained with (tests/models/test_modeling_gtlm_llama_v2.py), at fp32.
    impl: str = "v2-eager"
    flex_compile_mode: str = "max-autotune-no-cudagraphs"
    dtype: str = "fp32"                          # the dtype the v0<->v2 parity is proven at

    # ── k-hop attention gate ───────────────────────────────────────────────────
    k_hop: int = 0                               # 0 disables the mask (the historical setting)
    k_hop_directed: bool = False

    # ── graph-bias features (the ablation axes) ────────────────────────────────
    # The datasets always carry all three, so the flags are NOT part of the cache
    # identity and every arm shares one built dataset.
    spd: bool = True
    max_spd: int = 8
    rrwp: bool = True
    max_rw_steps: int = 16
    magnetic: bool = True
    magnetic_dim: int = 32                       # model bias-MLP hidden width
    magnetic_q: float = 0.25                     # magnetic-Laplacian charge (data-prep knob)
    magnetic_m: int = 0                          # eigenvectors kept (0 -> all N)
    laplacian: bool = False                      # unwired (see UNWIRED_FEATURES)
    rwse: bool = False                           # unwired

    # ── dataset ────────────────────────────────────────────────────────────────
    max_length: int = 32_768                     # per-node tokenization cap (historical)

    # ── data generation (only read in data_prep mode) ──────────────────────────
    # Both generators seed a module-level RNG at import; `gen_seed` makes that explicit
    # and reproducible instead of order-dependent.
    gen_seed: int = 42
    # family
    family_train: int = 3500
    family_val: int = 200
    family_test: int = 1000
    family_generations: int = 3
    family_marriage_prob: float = 0.7
    family_child_prob: float = 0.75
    # kg_qa
    kgqa_train: int = 500
    kgqa_val: int = 30
    kgqa_test: int = 150
    kgqa_min_nodes: int = 30
    kgqa_max_nodes: int = 50

    # ── LoRA (alpha is always 2*r, derived — never an independent knob) ────────
    lora: bool = True
    lora_r: int = 32                             # paper recipe: r32 / alpha64
    lora_dropout: float = 0.05

    # ── training schedule (the paper recipe; see module docstring) ─────────────
    num_epochs: int = 10
    batch_size: int = 4
    accumulation_steps: int = 4
    # LEARNING RATES ARE TASK-DEPENDENT. The paper uses a different pair per task:
    #
    #     task     lr       bias_lr
    #     family   5e-5     1e-2      <- the defaults below
    #     kg_qa    5e-4     3e-2
    #
    # These defaults are FAMILY's. A kg_qa run must set both explicitly (the sweep
    # configs do); leaving them at the defaults trains kg_qa with a 10x too small lr
    # and a 3x too small bias_lr, which is not the published recipe. See
    # PAPER_LEARNING_RATES below and `validate()`, which refuses the mismatch.
    lr: float = 5e-5                             # base / LoRA learning rate
    bias_lr: float = 1e-2                        # graph-bias learning rate
    eval_steps: int = 40                         # val eval + checkpoint cadence
    max_steps: int = -1                          # >0 caps optimizer steps (smoke tests)
    seed: int = 42
    num_workers: int = 0
    gradient_checkpointing: bool = True          # historical default (large graphs)
    include_f1: bool = False

    # ── data prep ──────────────────────────────────────────────────────────────
    # True matches the historical builds: the legacy prep scripts called
    # compute_*() bare, and those default to use_gpu=True. Feature values can differ
    # in the last bits between CPU and GPU eigendecomposition, so this stays True to
    # keep rebuilt datasets comparable with the existing ones. Data prep therefore
    # wants a GPU node.
    use_gpu: bool = True

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None                    # None -> no tracking

    # ── derived helpers ────────────────────────────────────────────────────────

    def torch_dtype(self):
        import torch
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.dtype]

    def backend(self):
        """The graph-attention backend ('eager' | 'flex') from ``impl``."""
        return self.impl.split("-", 1)[1]

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout}

    def bias_params(self):
        """The graph-bias flag/architecture dict passed to the model config.

        Only enabled, wired features contribute keys. ``magnetic_q`` rides along as
        provenance (no bias module reads it — the charge is consumed when the
        eigenvectors are computed during data prep).
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim,
                       magnetic_q=self.magnetic_q)
        return cfg

    def arm(self):
        """Short ablation-arm label ('base' | 'no-spd' | ...) for records + tables."""
        off = [f for f in WIRED_FEATURES if not getattr(self, f)]
        if not off:
            return "base"
        return "no-" + "+".join(off)

    def uses_default_cache(self):
        """True when this config's feature settings match the built-on-disk datasets."""
        return all(getattr(self, k) == v for k, v in _CACHE_DEFAULTS.items())

    def _base_dataset_dir(self):
        """The historical on-disk location for this task."""
        if self.task == "family":
            return FAMILY_DIR
        return os.path.join(KGQA_DIR, f"dataset_{self.kgqa_min_nodes}-{self.kgqa_max_nodes}")

    def dataset_dir(self):
        """Directory holding this config's built splits.

        Feature-generation knobs are part of the cache identity: changing ``magnetic_q``
        (say), or moving the question into its own node, must not silently reuse
        datasets built with the old value. A config matching ``_CACHE_DEFAULTS``
        resolves to the historical untagged path, so the existing (large, gitignored)
        datasets keep working untouched; anything else gets a tagged sibling.
        """
        base = self._base_dataset_dir()
        if self.uses_default_cache():
            return base
        tags = [f"q{self.magnetic_q}", f"rw{self.max_rw_steps}", f"len{self.max_length}"]
        if self.question_node != "off":
            tags.append(f"qn-{self.question_node}")
        if self.model_name != MODEL_NAME:
            tags.append(self.model_name.split("/")[-1])
        return f"{base}__{'_'.join(tags)}"

    def uses_paper_learning_rates(self):
        """True when lr/bias_lr match the published pair for this task."""
        expected = PAPER_LEARNING_RATES.get(self.task)
        if expected is None:
            return False
        return self.lr == expected["lr"] and self.bias_lr == expected["bias_lr"]

    def run_name(self):
        """Default (standalone) run name; the sweep runner overrides this."""
        lora_tag = f"_lora{self.lora_r}" if self.lora else ""
        khop_tag = f"_k{self.k_hop}" if self.k_hop else ""
        qn_tag = f"_qn-{self.question_node}" if self.question_node != "off" else ""
        return (f"{EXPERIMENT_NAME}_{self.arm()}_{self.task}"
                f"{qn_tag}{lora_tag}{khop_tag}_s{self.seed}")

    def validate(self):
        """Reject settings this experiment does not support, with a clear message.

        The sweep runner is experiment-agnostic and passes any key through; this is
        where the experiment draws its own line (fail fast, before any GPU work).
        """
        if self.mode not in ("train", "data_prep"):
            raise ValueError(f"Unknown mode {self.mode!r} (expected 'train' or 'data_prep').")
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r} (expected one of {TASKS}).")
        if self.question_node not in QUESTION_NODES:
            hint = ' (use "off", not None/null, to disable)' if self.question_node is None else ""
            raise ValueError(
                f"Unknown question_node {self.question_node!r} "
                f"(expected one of {QUESTION_NODES}){hint}.")
        if self.impl not in IMPLS:
            raise ValueError(f"Unknown impl {self.impl!r} (expected one of {IMPLS}).")
        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r} (expected one of {DTYPES}).")

        enabled_unwired = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if enabled_unwired:
            raise ValueError(
                f"Bias feature(s) {enabled_unwired} are exposed in the config but not "
                f"wired in this experiment (data prep computes only {WIRED_FEATURES}). "
                f"Disable them or extend the prep modules to produce their features.")

        # The two tasks have DIFFERENT published learning rates, and the dataclass can
        # only default to one of them (family's). Running kg_qa without overriding both
        # silently trains it at 1/10th the published lr — a mistake that is invisible in
        # the results and easy to repeat. Warn rather than raise: deliberate LR
        # experiments are legitimate, but they should be deliberate.
        if self.mode == "train":
            expected = PAPER_LEARNING_RATES.get(self.task)
            if expected and not self.uses_paper_learning_rates():
                print(f"[config] WARNING: {self.task} learning rates deviate from the "
                      f"published recipe (lr={expected['lr']:g}, "
                      f"bias_lr={expected['bias_lr']:g}); this run uses "
                      f"lr={self.lr:g}, bias_lr={self.bias_lr:g}. Intentional?")

        if self.lora_r < 0:
            raise ValueError("lora_r must be >= 0.")
        if self.k_hop < 0:
            raise ValueError("k_hop must be >= 0 (0 disables the mask).")
        if self.kgqa_min_nodes > self.kgqa_max_nodes:
            raise ValueError("kgqa_min_nodes must be <= kgqa_max_nodes.")
        return self
