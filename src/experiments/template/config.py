"""
Shared configuration for the template experiment.

One ``RunConfig`` dataclass holds EVERY knob the entry point reads; ``__main__.py``
builds a ``RunConfig`` from argparse and stays a thin dispatcher, so a value lives
in exactly one place. Copy this file into a new experiment and edit the fields.

The synthetic task (see ``data.py``) predicts a prompt node's out-degree from its
local graph neighbourhood. Three graph-bias features are wired end-to-end here
(the dataset computes them and the model builds their modules): shortest-path
distance (``spd``), random-walk structural encoding (``rrwp``) and the magnetic
Laplacian (``magnetic``). ``laplacian`` / ``rwse`` are exposed in the schema for
completeness but are NOT wired — ``validate()`` rejects them rather than letting a
run build model modules with no matching dataset features.
"""

from dataclasses import dataclass, field


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "template"

# Graph-bias features whose data + model modules are both wired in this experiment.
WIRED_FEATURES = ("spd", "rrwp", "magnetic")
# Exposed in the config schema but not produced by data.py -> rejected by validate().
UNWIRED_FEATURES = ("laplacian", "rwse")


@dataclass
class RunConfig:
    """Every knob the train entry point reads (built from argparse in ``__main__``)."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"                        # only "train" today; kept for parity

    # ── model + LoRA ───────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    # LoRA lets the frozen backbone adapt to *reading* the graph bias; disabled with
    # --no-lora. When on, the adapters are the only "base" trainable params.
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32                        # convention: 2 * lora_r
    lora_dropout: float = 0.05
    # Substrings of parameter names to unfreeze. "graph_bias" activates every graph
    # -bias module (they all live under graph_bias.bias_modules.*).
    active_params: tuple = ("graph_bias",)

    # ── graph-bias features ────────────────────────────────────────────────────
    spd: bool = True
    max_spd: int = 8                            # shortest-path-distance bucket cap
    rrwp: bool = True
    max_rw_steps: int = 16                      # random-walk structural-encoding steps
    magnetic: bool = True
    magnetic_dim: int = 32                      # magnetic-bias MLP hidden width
    magnetic_q: float = 0.25                    # magnetic-Laplacian charge
    # Magnetic-Laplacian eigenvectors kept by the collator (0 -> keep all).
    magnetic_m: int = 0
    laplacian: bool = False                     # unwired (see UNWIRED_FEATURES)
    rwse: bool = False                          # unwired

    # ── k-hop attention gate ───────────────────────────────────────────────────
    k_hop: int = 2                              # 0 disables the k-hop mask

    # ── synthetic dataset ──────────────────────────────────────────────────────
    num_samples: int = 80                       # split 75 / 12.5 / 12.5 into train/eval/test
    max_length: int = 64                        # per-node tokenization cap
    data_seed: int = 42                         # RNG for graph generation (decoupled from training seed)

    # ── training schedule ──────────────────────────────────────────────────────
    num_epochs: int = 5
    batch_size: int = 2
    accumulation_steps: int = 4
    lr: float = 3e-4                            # base / LoRA learning rate
    bias_lr: float = 5e-3                       # graph-bias learning rate
    eval_steps: int = 20                        # eval + checkpoint cadence
    max_steps: int = -1                         # >0 caps optimizer steps (quick smoke tests)
    seed: int = 42                              # training seed (decoupled from data_seed)
    num_workers: int = 0                        # DataLoader workers (>0 overlaps feature build)
    gradient_checkpointing: bool = True
    include_f1: bool = False                    # also log macro-F1 alongside exact match

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None                   # None -> no tracking; a string -> wandb project

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout}

    def bias_params(self):
        """The graph-bias flag/architecture dict passed to ``init_model``.

        Only enabled, wired features contribute keys, so the model builds modules
        exactly for the features ``data.py`` produces.
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        return cfg

    def run_name(self):
        """Default (standalone) run name; the sweep runner overrides this."""
        lora_tag = f"_lora{self.lora_r}" if self.lora else ""
        return f"{EXPERIMENT_NAME}_khop{self.k_hop}{lora_tag}_s{self.seed}"

    def validate(self):
        """Reject settings this experiment does not support, with a clear message.

        The sweep runner is experiment-agnostic and passes any key through; this is
        where the experiment draws its own line (fail fast, before any GPU work).
        """
        if self.mode != "train":
            raise ValueError(f"Unknown mode {self.mode!r} (only 'train' is supported).")
        enabled_unwired = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if enabled_unwired:
            raise ValueError(
                f"Bias feature(s) {enabled_unwired} are exposed in the config but not "
                f"wired in this experiment (data.py computes only {WIRED_FEATURES}). "
                f"Disable them or extend data.py to produce their features.")
        if not (self.spd or self.rrwp or self.magnetic):
            raise ValueError("At least one graph-bias feature (spd / rrwp / magnetic) must be enabled.")
        if self.lora_r < 0:
            raise ValueError("lora_r must be >= 0.")
        if self.num_samples < 8:
            raise ValueError("num_samples must be >= 8 (the 75/12.5/12.5 split needs a few graphs).")
        return self
