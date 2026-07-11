"""
Shared configuration for the probe suite.

One ``RunConfig`` dataclass holds EVERY knob the entry point reads; ``__main__.py``
builds a ``RunConfig`` from argparse and stays a thin dispatcher, so a value lives
in exactly one place.

The suite (see README.md, the agreed spec) trains one (task, bias arm, k_hop,
seed) configuration per run. ``task`` selects the synthetic probe (``data.py``);
``bias`` is a free-form '+'-joined arm string (e.g. ``"spd+magnetic_shared"``)
whose tokens are resolved against the bias registry in ``src/models/bias.py``
(``BIAS_TYPES``) — any new registered bias variant becomes a valid token with no
suite changes. Only the features ``data.py`` computes are accepted (WIRED_TOKENS).
"""

from dataclasses import dataclass


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "probes"

TASKS = ("direction", "substructure", "local_hop", "text_path")

# Per-task node-count ranges from the README spec, used when min/max_nodes are
# left at None. text_path counts PERSONS (~40-60 tokens each -> L <= ~2.5k).
TASK_NODE_RANGES = {
    "direction":    (100, 400),
    "substructure": (10, 50),
    "local_hop":    (100, 300),
    "text_path":    (25, 40),
}

# Bias tokens whose dataset features data.py produces (spd + magnetic_V/lambdas;
# magnetic_shared consumes the same magnetic features). Registry tokens outside
# this set (laplacian/rwse/rrwp) are rejected by validate().
WIRED_TOKENS = ("spd", "magnetic", "magnetic_shared")

POSSIBLE_LABELS = [" Yes", " No"]


@dataclass
class RunConfig:
    """Every knob the train/data_prep entry points read (built from argparse in ``__main__``)."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"                         # "train" | "data_prep"
    task: str = "direction"                     # which probe (see TASKS)
    # '+'-joined bias arm string: "none" | tokens from BIAS_TYPES config_keys,
    # e.g. "spd", "magnetic", "spd+magnetic", "magnetic_shared", ...
    bias: str = "spd"
    k_hop: int = 0                              # 0 disables the k-hop mask
    # Probe graphs are directed (substructure is stored as a symmetric digraph),
    # so the k-hop mask follows edge directions by default.
    k_hop_directed: bool = True

    # ── model / bias architecture ──────────────────────────────────────────────
    model_name: str = MODEL_NAME
    impl: str = "v2-flex"                       # "v2-flex" | "v2-eager" (debug)
    flex_compile_mode: str = "max-autotune-no-cudagraphs"   # | "default"
    max_spd: int = 32
    magnetic_dim: int = 32
    magnetic_q: float = 0.25
    magnetic_m: int = 0                         # eigenvectors kept (0 -> all N)

    # ── dataset (shared) ───────────────────────────────────────────────────────
    train_size: int = 4000
    val_size: int = 500
    test_size: int = 1000
    data_seed: int = 0                          # decoupled from the training seed
    # None -> the task's README range (TASK_NODE_RANGES).
    min_nodes: int = None
    max_nodes: int = None
    ordering: str = "rcm"                       # node ordering ("rcm" | "original")
    # Flex L/N bucketing ladders (None -> GraphCollatorV2's coarse defaults).
    len_buckets: tuple = None
    node_buckets: tuple = None

    # ── task-specific generator knobs ──────────────────────────────────────────
    # direction / local_hop / text_path: probability an undirected skeleton edge
    # stays bidirectional (the rest are oriented one way, 50/50 per direction).
    p_bidirectional: float = 0.2
    num_colors: int = 10                        # local_hop color vocabulary
    hop_radius: int = 2                         # local_hop question radius (< k_hop when k_hop > 0)

    # ── LoRA (README recipe: r=8, dropout=0.05; alpha is always 2*r, derived) ─
    lora: bool = True
    lora_r: int = 8
    lora_dropout: float = 0.05

    # ── training schedule (README recipe) ─────────────────────────────────────
    lr: float = 1e-5
    bias_lr: float = 1e-3
    num_epochs: int = 25
    batch_size: int = 4
    accumulation_steps: int = 8
    eval_steps: int = 100                       # val eval + checkpoint cadence
    max_steps: int = -1                         # >0 caps optimizer steps (smoke tests)
    seed: int = 0
    num_workers: int = 4
    gradient_checkpointing: bool = False        # backbone activation ckpt (bias has its own)

    # ── density measurement (data property, once per run) ─────────────────────
    measure_density: bool = True
    density_sample_graphs: int = 16
    density_sample_batches: int = 8

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None

    # ── derived helpers ────────────────────────────────────────────────────────

    def bias_tokens(self):
        """The arm's token list; ``"none"`` -> [] (no bias features)."""
        if self.bias.strip() == "none":
            return []
        return [t.strip() for t in self.bias.split("+") if t.strip()]

    def needs_spd(self):
        return "spd" in self.bias_tokens()

    def needs_magnetic(self):
        return bool({"magnetic", "magnetic_shared"} & set(self.bias_tokens()))

    def node_range(self):
        lo, hi = TASK_NODE_RANGES[self.task]
        return (self.min_nodes or lo, self.max_nodes or hi)

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled).

        alpha is always ``2 * r`` (the scaling convention shared across the
        experiments, e.g. kgqa); it is derived, never an independent knob, so
        bumping ``lora_r`` can never silently change the effective LoRA scale.
        """
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2, "lora_dropout": self.lora_dropout}

    def model_bias_config(self):
        """The graph-bias flag/architecture dict passed to the model config.

        Each arm token sets its registry flag; the architecture knobs ride along
        only for the features that use them.
        """
        cfg = {}
        for token in self.bias_tokens():
            cfg[token] = True
        if self.needs_spd():
            cfg["max_spd"] = self.max_spd
        if self.needs_magnetic():
            cfg.update(magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        return cfg

    def run_name(self):
        """Default (standalone) run name; the sweep runner overrides this."""
        return f"{EXPERIMENT_NAME}_{self.task}_{self.bias}_k{self.k_hop}_s{self.seed}"

    def validate(self):
        """Reject settings this experiment does not support, with a clear message.

        The sweep runner is experiment-agnostic and passes any key through; this
        is where the experiment draws its own line (fail fast, before GPU work).
        """
        if self.mode not in ("train", "data_prep"):
            raise ValueError(f"Unknown mode {self.mode!r} (expected 'train' or 'data_prep').")
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r} (expected one of {TASKS}).")

        tokens = self.bias_tokens()
        if self.bias.strip() != "none" and not tokens:
            raise ValueError(f"Empty bias arm {self.bias!r} (use 'none' for the no-bias arm).")
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Duplicate token in bias arm {self.bias!r}.")
        from ...models.bias import BIAS_TYPES  # deferred: keeps config import light
        registry = {cls.config_key for cls in BIAS_TYPES}
        for token in tokens:
            if token not in registry:
                raise ValueError(
                    f"Bias token {token!r} (arm {self.bias!r}) is not registered in "
                    f"src/models/bias.py BIAS_TYPES ({sorted(registry)}).")
            if token not in WIRED_TOKENS:
                raise ValueError(
                    f"Bias token {token!r} is registered but not wired in this experiment "
                    f"(data.py computes features for {WIRED_TOKENS} only).")
        if "magnetic" in tokens and "magnetic_shared" in tokens:
            raise ValueError(
                "Arm combines 'magnetic' and 'magnetic_shared' — they are variants of "
                "the same feature; pick one per arm.")

        if self.impl not in ("v2-flex", "v2-eager"):
            raise ValueError(f"impl must be 'v2-flex' or 'v2-eager', got {self.impl!r}.")
        if min(self.train_size, self.val_size, self.test_size) < 1:
            raise ValueError("train/val/test sizes must all be >= 1.")
        if self.k_hop > 0 and self.task == "local_hop" and self.hop_radius >= self.k_hop:
            raise ValueError(
                f"local_hop needs hop_radius < k_hop (the answer must sit inside the "
                f"mask); got radius={self.hop_radius}, k_hop={self.k_hop}.")
        if not (0.0 <= self.p_bidirectional <= 1.0):
            raise ValueError("p_bidirectional must be in [0, 1].")
        return self
