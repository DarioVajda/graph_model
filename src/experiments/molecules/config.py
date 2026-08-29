"""
Configuration for the molecules experiment (PLAN.md).

One ``RunConfig`` holds every knob; ``__main__.py`` builds it from argparse and
stays a thin dispatcher. One run = one (task, arm, encoding, bias, seed).

The two axes that make this experiment what it is:

* ``arm`` — ``"graph"`` (atoms + Levi bond nodes) vs ``"flat"`` (a single-node
  graph holding the SMILES, which by Property 2 *is* the base LLM). The flat arm
  is the matched control; the external anchors in PLAN.md §2 are not.
* ``encoding`` — the three usable cells of PLAN.md §3.2. ``terse_atom_only`` is
  rejected by construction, not scored badly, and `data.py` raises on it.
"""

from dataclasses import dataclass, field

from .data import ENCODINGS, QUESTION_NODE_MODES
from .dataset import ALL_TASKS, ARMS, DEFAULT_POOL, tier_of
from .tier_b import REGRESSION_TASKS

MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "molecules"

#: Bias tokens whose dataset features `dataset.py` produces.
WIRED_TOKENS = ("spd", "magnetic", "magnetic_shared")


@dataclass
class RunConfig:
    """Every knob the train / data_prep entry points read."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"                     # "train" | "data_prep" | "eval"
    # `--mode eval` only: a trained checkpoint directory to re-score and analyse
    # per-example. Empty in every training run.
    checkpoint: str = ""
    task: str = "ring_membership"           # a Tier-A generator or a Tier-B corpus
    arm: str = "graph"                      # "graph" | "flat"
    encoding: str = "rich_levi"             # PLAN.md §3.2 (graph arm only)
    stereo_tags: bool = True                # parity tag in atom text (never the CIP label)
    bias: str = "spd+magnetic"              # '+'-joined arm string, or "none"
    k_hop: int = 0                          # D3: 0 everywhere
    k_hop_directed: bool = False            # molecules are undirected

    # Where the question lives — see QUESTION_NODE_MODES in data.py. "on" (the
    # default) puts it in its own edge-free PREFIX node, so every atom and bond
    # node attends to it. This is settled, not an open axis. Changes the graph, so
    # it is part of the dataset cache key.
    question_node: str = "on"

    # NOTE: there is deliberately no `node_position_mode` here. GTLM supports
    # "spd_depth" (a node's tokens start at STRIDE * depth from the prompt node
    # rather than at 0) and `GraphCollatorV2` still implements it, but it is
    # UNWIRED in this experiment by decision, 2026-08-29. kgqa measured it at
    # 0.6412 +- 0.0037 F1 against 0.7351 +- 0.0076 for the "reset" default — a
    # 9.4-point regression (kgqa/README.md E3) — and the molecules canary put it
    # marginally lower again (0.998 vs 1.000) with less bias movement. Two
    # measurements, no positive result, so it is not an axis worth the surface
    # area here. `build_collator` is the shared helper again as a result.
    # See PLAN.md §3.2.3.

    # Held-out enforcement (PLAN.md §4.1). Must be set explicitly to build a
    # held-out task at all, and even then only for evaluation.
    held_out_eval: bool = False

    # ── model / bias architecture ──────────────────────────────────────────────
    model_name: str = MODEL_NAME
    impl: str = "v2-flex"                   # "v2-flex" | "v2-eager"
    flex_compile_mode: str = "max-autotune-no-cudagraphs"
    # Deliberately still 32, pending evidence. The Levi transform doubles every
    # distance (measured Levi diameter over 147 bace+bbbp molecules: median 28,
    # p90 38, max 66), so 32 clamps the far half of ~24.5% of molecules. That makes
    # 64 a plausible fix, but "plausible" is how `spd_depth` got built — the
    # per-example error-vs-diameter measurement decides it first.
    max_spd: int = 32
    magnetic_dim: int = 32
    magnetic_q: float = 0.25
    magnetic_m: int = 0

    # ── dataset ────────────────────────────────────────────────────────────────
    pool: tuple = DEFAULT_POOL              # Tier A only: which corpora supply molecules
    train_size: int = 4000                  # Tier A only (Tier B's sizes are the split's)
    val_size: int = 500
    test_size: int = 1000
    # Tier B only: 0 = use the whole scaffold split. Caps subsample randomly under
    # `data_seed`, never by slicing (see `prepare_tier_b_graphs`).
    max_train_examples: int = 0
    max_eval_examples: int = 0
    data_seed: int = 0
    ordering: str = "rcm"
    len_buckets: tuple = None
    node_buckets: tuple = None

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora: bool = True
    lora_r: int = 8
    lora_dropout: float = 0.05

    # ── training schedule ──────────────────────────────────────────────────────
    lr: float = 1e-5
    bias_lr: float = 1e-3
    num_epochs: int = 20
    batch_size: int = 4
    accumulation_steps: int = 8
    eval_steps: int = 100
    max_steps: int = -1
    seed: int = 0
    num_workers: int = 4
    gradient_checkpointing: bool = False

    # ── measurement ────────────────────────────────────────────────────────────
    measure_density: bool = True
    density_sample_graphs: int = 16
    density_sample_batches: int = 8

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None

    # ── derived helpers ────────────────────────────────────────────────────────

    def bias_tokens(self):
        if self.bias.strip() == "none":
            return []
        return [t.strip() for t in self.bias.split("+") if t.strip()]

    def needs_spd(self):
        return "spd" in self.bias_tokens()

    def needs_magnetic(self):
        return bool({"magnetic", "magnetic_shared"} & set(self.bias_tokens()))

    def lora_config(self):
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout}

    def model_bias_config(self):
        cfg = {}
        for token in self.bias_tokens():
            cfg[token] = True
        if self.needs_spd():
            cfg["max_spd"] = self.max_spd
        if self.needs_magnetic():
            cfg.update(magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        return cfg

    def tier(self):
        return tier_of(self.task)

    def run_name(self):
        parts = [EXPERIMENT_NAME, self.task, self.arm]
        if self.arm == "graph":
            parts += [self.encoding, self.bias]
        return "_".join(parts + [f"s{self.seed}"])

    def validate(self):
        """Reject unsupported combinations before any GPU work."""
        if self.mode not in ("train", "data_prep", "eval"):
            raise ValueError(f"Unknown mode {self.mode!r}.")
        if self.mode == "eval" and not self.checkpoint:
            raise ValueError("mode 'eval' requires --checkpoint.")
        if self.task not in ALL_TASKS:
            raise ValueError(f"Unknown task {self.task!r} (expected one of {ALL_TASKS}).")
        if self.task in REGRESSION_TASKS:
            raise ValueError(
                f"{self.task!r} is a regression set; the yes/no margin readout does "
                "not apply to it. The numeric_text path is not built yet (PLAN.md §7.2).")
        if self.arm not in ARMS:
            raise ValueError(f"Unknown arm {self.arm!r} (expected one of {ARMS}).")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"Unknown encoding {self.encoding!r} (expected {ENCODINGS}); "
                             "'terse_atom_only' is rejected by construction, see PLAN.md §3.2.")

        # The flat arm is a single-node graph, so every structural bias is
        # identically zero on it (Property 2). Letting a bias arm ride along would
        # advertise a comparison that is not happening.
        if self.arm == "flat" and self.bias.strip() != "none":
            raise ValueError(
                "The flat arm is a single-node graph, where every graph bias "
                "vanishes by construction (Property 2). Use --bias none for the "
                "flat arm so the run record cannot imply a bias was in play.")

        tokens = self.bias_tokens()
        if self.bias.strip() != "none" and not tokens:
            raise ValueError(f"Empty bias arm {self.bias!r} (use 'none' for the no-bias arm).")
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Duplicate token in bias arm {self.bias!r}.")
        from ...models.bias import BIAS_TYPES     # deferred: keeps config import light
        registry = {cls.config_key for cls in BIAS_TYPES}
        for token in tokens:
            if token not in registry:
                raise ValueError(
                    f"Bias token {token!r} is not registered in src/models/bias.py "
                    f"BIAS_TYPES ({sorted(registry)}).")
            if token not in WIRED_TOKENS:
                raise ValueError(
                    f"Bias token {token!r} is registered but not wired here "
                    f"(dataset.py computes features for {WIRED_TOKENS} only).")
        if "magnetic" in tokens and "magnetic_shared" in tokens:
            raise ValueError("Pick one of 'magnetic' / 'magnetic_shared' per arm.")

        if self.question_node not in QUESTION_NODE_MODES:
            raise ValueError(f"question_node must be one of {QUESTION_NODE_MODES}, "
                             f"got {self.question_node!r}.")
        if self.impl not in ("v2-flex", "v2-eager"):
            raise ValueError(f"impl must be 'v2-flex' or 'v2-eager', got {self.impl!r}.")
        if min(self.train_size, self.val_size, self.test_size) < 1:
            raise ValueError("train/val/test sizes must all be >= 1.")
        if not self.pool:
            raise ValueError("pool must name at least one Tier-B corpus.")
        return self
