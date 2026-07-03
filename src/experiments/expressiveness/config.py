"""
Shared configuration for the expressiveness experiment.

Holds the (now fixed) bias/model constants and a single ``RunConfig`` dataclass
carrying every knob the ``train`` / ``bench`` entry points read. ``__main__.py``
builds a ``RunConfig`` from argparse and stays a thin dispatcher.

The graph-aware bias is fixed to **shortest-path distance + magnetic Laplacian**
only (every other feature was dropped from this experiment). The single tunable
spectral knob is ``magnetic_m`` — the number of magnetic-Laplacian eigenvectors
kept (``0`` keeps all ``N``). ``magnetic_dim`` (the bias-MLP hidden width) and
``magnetic_q`` (the charge) are model/architecture constants, not run knobs.
"""

from dataclasses import dataclass, field


MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Fixed bias/model constants (no longer per-run knobs).
MAX_SPD = 32           # shortest-path-distance bucket cap. Must exceed the largest
                       # finite path so the unreachable sentinel (32767 -> top bucket)
                       # is never shared with a far-but-connected pair.
MAGNETIC_DIM = 32      # magnetic-bias MLP hidden width (model architecture)
MAGNETIC_Q = 0.25      # magnetic-Laplacian charge

# Both implementations report prompt-span exact-match accuracy (``em``), so v0 and
# v2 are directly comparable. (v0's em is computed in ``evaluation.py``; v2's by
# ``GraphTrainerV2``.)
ACCURACY_METRIC = {"v0": "eval_em_accuracy", "v2": "eval_em_accuracy"}

POSSIBLE_LABELS = [" Yes", " No"]


# Graph-bias features. ``spd`` + ``magnetic`` are wired end-to-end (data_gen
# computes them; the model builds their modules). ``laplacian`` / ``rwse`` /
# ``rrwp`` are exposed in the config schema for completeness but are NOT wired in
# this experiment — data_gen no longer produces them — so RunConfig.validate()
# rejects them rather than letting a run silently break.
WIRED_FEATURES = ("spd", "magnetic")
UNWIRED_FEATURES = ("laplacian", "rwse", "rrwp")


def tokenized_label_options(tokenizer, labels=POSSIBLE_LABELS):
    """Tokenize the Yes/No answer options (no special tokens) for v0 scoring."""
    return [tokenizer(label, add_special_tokens=False).input_ids for label in labels]


def run_suffix(magnetic_m=0):
    """Build the ``spd(8)+magnetic(...)`` suffix used in run/checkpoint names."""
    return f"spd({MAX_SPD})+magnetic(dim={MAGNETIC_DIM},q={MAGNETIC_Q},m={magnetic_m})"


@dataclass
class RunConfig:
    """Every knob the train/bench entry points read (built from argparse in ``__main__``)."""
    # ── What to run ──────────────────────────────────────────────────────────
    mode: str = "train"                                   # "train" | "bench"
    impls: tuple = ("v0-eager", "v2-eager")
    k_hops: tuple = (0,)
    k_hop_directed: bool = False                          # HARD graphs are bidirectional
    difficulty: str = "HARD"                              # "HARD" | "EASY"
    # None -> no experiment tracking; a string -> the wandb project to report to.
    wandb_project: str = None
    # flex's real operating point (README): autotune gives ~4.7x fwd at k>0 and
    # unlocks 64-wide blocks. Costs ~320s one-time compile per shape (cached on disk).
    # Use "default" for quick iteration (128-blocks only, no autotune).
    flex_compile_mode: str = "max-autotune-no-cudagraphs"  # | "default"

    model_name: str = MODEL_NAME
    # Magnetic-Laplacian eigenvectors kept (0 -> all N). Used by BOTH dataset
    # generation and the collator, so they always agree.
    magnetic_m: int = 0

    # ── graph-bias features ──────────────────────────────────────────────────
    # spd + magnetic are wired end-to-end; laplacian/rwse/rrwp are exposed in the
    # schema but rejected by validate() (data_gen no longer computes them). The
    # numeric knobs default to the (formerly constant) architecture values.
    spd: bool = True
    magnetic: bool = True
    laplacian: bool = False
    rwse: bool = False
    rrwp: bool = False
    max_spd: int = MAX_SPD
    magnetic_dim: int = MAGNETIC_DIM
    magnetic_q: float = MAGNETIC_Q

    # ── graph size (shared by train graph range + bench fixed size) ──────────
    num_nodes: int = 500
    # Optional explicit overrides; when None they are derived from ``num_nodes``
    # as round(0.8*num_nodes) / round(1.2*num_nodes) (see ``__post_init__``).
    min_nodes: int = None
    max_nodes: int = None

    # ── train mode ───────────────────────────────────────────────────────────
    train_dataset_size: int = 2_000
    eval_dataset_size: int = 500
    ordering: str = "rcm"                                 # "rcm" (default, block-locality) | "original" (baseline)
    # Flex L/N bucketing ladders (passed to GraphCollatorV2). None -> the kernel's
    # coarse defaults (multiples of 512 / powers of two). Set tight, anticipated
    # ladders matched to the graph sizes to cut padding waste and the number of
    # distinct compiled shapes. Each len bucket must be a multiple of the block size.
    len_buckets: tuple = None
    node_buckets: tuple = None
    lr: float = 4e-6
    bias_lr: float = 1e-2
    num_epochs: int = 3
    batch_size: int = 4
    accumulation_steps: int = 8
    eval_steps: int = 25
    # DataLoader worker processes. 0 (HF default) builds each batch's O(N^2) spd +
    # magnetic features synchronously on the main process, stalling the GPU between
    # steps; >0 overlaps that construction with compute. With N in the thousands the
    # per-item build is ~seconds, so this matters a lot at large graph sizes.
    num_workers: int = 4
    seeds: tuple = (0,)
    max_steps: int = -1                                   # >0 caps steps (quick tests)

    # ── LoRA (optional backbone adaptation alongside the graph bias) ─────────
    # The token stream carries no edge info (and the model is prefix-permutation-
    # invariant), so LoRA cannot substitute for the bias signal — it only lets the
    # frozen backbone adapt to *reading* the bias. When enabled, the LoRA adapters
    # are the only "base" trainable params, so they train at ``lr`` (set it to a
    # LoRA-appropriate value, e.g. 1e-4); the graph bias still trains at ``bias_lr``.
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # ── density measurement (standalone, on a random subset) ─────────────────
    measure_density: bool = True
    density_sample_graphs: int = 16
    density_sample_batches: int = 8

    # ── bench mode (synthetic large-graph throughput) ────────────────────────
    bench_batch_size: int = 2
    bench_num_warmup: int = 8
    bench_num_iters: int = 8
    bench_num_examples: int = 16

    def __post_init__(self):
        # Derive the node range from ``num_nodes`` unless explicitly overridden.
        if self.min_nodes is None:
            self.min_nodes = round(0.8 * self.num_nodes)
        if self.max_nodes is None:
            self.max_nodes = round(1.2 * self.num_nodes)

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout}

    @property
    def is_easy(self) -> bool:
        return self.difficulty == "EASY"

    def validate(self):
        """Reject settings this experiment does not support, with a clear message.

        The runner is experiment-agnostic and will happily pass any key through;
        this is where the experiment draws its own line. Only spd + magnetic are
        wired (data + model); enabling laplacian/rwse/rrwp would build model
        modules with no matching dataset features, so we fail fast instead.
        """
        enabled_unwired = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if enabled_unwired:
            raise ValueError(
                f"Bias feature(s) {enabled_unwired} are exposed in the config but not "
                f"wired in this experiment (data_gen computes only {WIRED_FEATURES}). "
                f"Disable them or extend data_gen to produce their features.")
        if not (self.spd or self.magnetic):
            raise ValueError("At least one of spd / magnetic must be enabled.")
        return self

    def model_bias_config(self):
        """The graph-bias flag/architecture dict passed to the model config.

        Only enabled, wired features contribute keys, so the model builds modules
        exactly for the features the dataset carries.
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        return cfg
