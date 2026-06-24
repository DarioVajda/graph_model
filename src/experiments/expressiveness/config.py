"""
Shared configuration for the expressiveness experiment.

Holds the bias/model constants (identical across v0 and v2) and a single
``RunConfig`` dataclass carrying every knob the ``train`` / ``bench`` entry
points read, so ``__main__.py`` stays a thin dispatcher.
"""

from dataclasses import dataclass, field


MODEL_NAME = "meta-llama/Llama-3.2-1B"
SPECTRAL_DIMS = 16

# Shared bias configuration (identical across v0 and v2).
BIAS_PARAMS = {
    "spd": True,
    "max_spd": 8,
    "laplacian": False,
    "rwse": False,
    "rrwp": False,                  # excluded: dominant on-disk feature, dropped from this experiment
    "max_rw_steps": 16,
    "magnetic": True,
    "magnetic_dim": 32,
    "magnetic_q": 0.25,
}

# Which eval metric each implementation's trainer reports as its accuracy.
#   v0  scores the Yes/No token directly (classification accuracy)
#   v2  uses prompt-span exact match (em accuracy)
ACCURACY_METRIC = {"v0": "eval_classification_accuracy", "v2": "eval_em_accuracy"}

POSSIBLE_LABELS = [" Yes", " No"]


def tokenized_label_options(tokenizer, labels=POSSIBLE_LABELS):
    """Tokenize the Yes/No answer options (no special tokens) for v0 scoring."""
    return [tokenizer(label, add_special_tokens=False).input_ids for label in labels]


def run_suffix(bias_params=BIAS_PARAMS):
    """Build the ``spd(8)+rrwp(16)+magnetic(...)`` suffix used in run/checkpoint names."""
    return "+".join(
        bias_type
        for bias_type in [
            f"spd({bias_params['max_spd']})", "laplacian", "rwse",
            f"rrwp({bias_params['max_rw_steps']})",
            f"magnetic(dim={bias_params['magnetic_dim']},q={bias_params['magnetic_q']})",
        ]
        if bias_params[bias_type.split('(')[0]]
    )


@dataclass
class RunConfig:
    """Every knob the train/bench entry points read (env-overridable in ``__main__``)."""
    # ── What to run ──────────────────────────────────────────────────────────
    mode: str = "train"                                   # "train" | "bench"
    impls: tuple = ("v0-eager", "v2-eager")
    k_hops: tuple = (0, 1)
    k_hop_directed: bool = False                          # HARD graphs are bidirectional
    difficulty: str = "HARD"
    report_to: str = "none"                               # "none" | "wandb"
    # flex's real operating point (README #4): autotune gives ~4.7x fwd at k>0 and
    # unlocks 64-wide blocks. Costs ~320s one-time compile per shape (cached on disk).
    # Use "default" for quick iteration (128-blocks only, no autotune).
    flex_compile_mode: str = "max-autotune-no-cudagraphs"  # | "default"

    model_name: str = MODEL_NAME
    spectral_dims: int = SPECTRAL_DIMS
    bias_params: dict = field(default_factory=lambda: dict(BIAS_PARAMS))

    # ── train mode ───────────────────────────────────────────────────────────
    train_dataset_size: int = 2_000
    eval_dataset_size: int = 500
    min_nodes: int = 10
    max_nodes: int = 25
    ordering: str = "rcm"                                 # "rcm" (default, block-locality) | "original" (baseline)
    lr: float = 4e-6
    bias_lr: float = 1e-2
    num_epochs: int = 3
    batch_size: int = 4
    accumulation_steps: int = 8
    eval_steps: int = 25
    seeds: tuple = (0, 1, 2)
    max_steps: int = -1                                   # >0 caps steps (quick tests)

    # ── density measurement (standalone, on a random subset) ─────────────────
    measure_density: bool = True
    density_sample_graphs: int = 16
    density_sample_batches: int = 8

    # ── bench mode (synthetic large-graph throughput) ────────────────────────
    bench_sizes: tuple = (500, 1000)
    bench_batch_size: int = 2
    bench_num_warmup: int = 2
    bench_num_iters: int = 5
    bench_num_examples: int = 16

    @property
    def is_easy(self) -> bool:
        return self.difficulty == "EASY"
