"""
Single source of truth for the KGQA (SR-WebQSP) experiment.

One ``RunConfig`` dataclass carries every knob both entry points read — merging
what used to live in three places: ``process_dataset.DEFAULTS`` (data prep),
``__main__.parse_args`` (training), and the hard-coded ``BIAS_PARAMS`` (model).
``__main__.py`` builds a ``RunConfig`` from argparse and stays a thin dispatcher;
``process_dataset``/``train`` consume the object instead of their own namespaces.

Field groups (mirrors the flag groups in ``__main__.build_parser``):
  * data-prep keys — determine the ``.gtds`` cache directory (``data_config_key``).
  * shared model/bias keys — used by BOTH data prep and the model/collator.
  * train keys — training schedule + graph-attention knobs.
  * generative-eval keys, tracking.

Two knobs that were silently entangled before are now single, explicit fields:
  * ``magnetic_m`` — the number of magnetic-Laplacian eigenvectors, used by BOTH
    data prep (``compute_magnetic_lap(m=...)``) and the collator. Previously the
    collator took ``BIAS_PARAMS['magnetic_dim']`` by coincidence (both were 128).
  * ``max_spd`` — one cutoff used by data prep (SPD bucket cap) and the model
    (``BIAS_PARAMS['max_spd']``); the old ``spd_cutoff`` alias is gone.

The training ``seed`` is decoupled from ``data_seed`` (the augmentation RNG that
is baked into the cache key), so sweeping the training seed no longer forces a
full dataset rebuild per seed.
"""

from dataclasses import dataclass


MODEL_NAME = "meta-llama/Llama-3.2-1B"

REL_MODES = ("last_1", "last_2", "full")
GRAPH_ATTN_IMPLS = ("flex", "eager")
DTYPES = ("bf16", "fp32")

# LoRA target modules for the Llama backbone (fixed architecture choice).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class RunConfig:
    """Every knob the data_prep / train entry points read (built from argparse)."""

    # ── what to run ──────────────────────────────────────────────────────────
    mode: str = "train"                       # "train" | "data_prep"

    # ── data-prep keys (these determine the .gtds cache directory) ───────────
    rel_mode: str = "last_1"                  # relation verbalization: last_1|last_2|full
    max_nodes: int = 512                      # Levi-graph (+prompt) node cap
    n_max: int = 20                           # max answers kept in the training target
    versions: int = 8                         # per-graph answer-order augmentations (train only)
    max_length: int = 1024                    # per-node token cap (kept non-binding)
    rcm: bool = True                          # reverse-Cuthill-McKee node ordering
    data_seed: int = 42                       # augmentation RNG seed (baked into cache key)
    use_gpu: bool = True                      # data-prep only (SPD/magnetic on GPU); train ignores it
    analyse_dataset: bool = False             # data-prep only: coverage-ceiling analysis (not in cache key)

    # ── shared model/bias keys (used by BOTH data prep and the model) ────────
    model_name: str = MODEL_NAME
    spd: bool = True
    max_spd: int = 64                         # data prep: SPD cutoff; model: bucket cap
    magnetic: bool = True
    magnetic_dim: int = 128                   # magnetic-bias MLP hidden width (model architecture)
    magnetic_q: float = 0.25                  # magnetic-Laplacian charge
    magnetic_m: int = 128                     # # magnetic eigenvectors (data prep + collator; 0 = all N)

    # ── train keys ───────────────────────────────────────────────────────────
    num_epochs: int = 5
    batch_size: int = 2
    accumulation_steps: int = 4
    lr: float = 3e-4                          # base LoRA/backbone lr
    bias_lr: float = 5e-3                     # graph-bias param group lr
    eval_steps: int = 100
    max_steps: int = -1                       # >0 caps optimizer steps (quick smoke tests)
    seed: int = 42                            # training seed (decoupled from data_seed)
    lora_r: int = 16                          # LoRA rank (0 disables LoRA)
    k_hop: int = 2                            # k-hop attention gate (0 disables)
    k_hop_directed: bool = False
    graph_attn_impl: str = "flex"             # "flex" | "eager"
    dtype: str = "bf16"                       # "bf16" | "fp32"
    gradient_checkpointing: bool = True
    active_params: tuple = ("graph_bias",)    # trainable param groups besides LoRA
    num_workers: int = 4                      # DataLoader workers (0 = synchronous feature build)

    # ── generative-eval keys ─────────────────────────────────────────────────
    gen_max_new_tokens: int = 128
    gen_max_samples: int = None               # None = full dev set

    # ── tracking ─────────────────────────────────────────────────────────────
    wandb_project: str = None                 # None = no tracking; a string = the wandb project

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def torch_dtype(self):
        import torch
        return torch.bfloat16 if self.dtype == "bf16" else torch.float32

    def bias_params(self):
        """The graph-bias flag/architecture dict (formerly the hard-coded ``BIAS_PARAMS``).

        Only enabled features contribute keys, so the model builds modules exactly
        for the features the dataset carries. ``magnetic_dim`` is the model's MLP
        width — distinct from ``magnetic_m`` (the eigenvector count), which is a
        data/collator knob and is NOT part of this dict.
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        return cfg

    def lora_config(self):
        """PEFT LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if self.lora_r <= 0:
            return None
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_r * 2,
            "target_modules": list(LORA_TARGET_MODULES),
            "lora_dropout": 0.05,
            "bias": "none",
        }

    def data_config_key(self):
        """Cache-directory name, built ONLY from data-affecting fields.

        Training-only knobs (seed, lr, k_hop, …) are deliberately excluded so two
        runs differing only in training config share one built dataset. Uses
        ``data_seed`` (not the training ``seed``) and includes ``max_length``
        (which affects tokenization). A fresh naming scheme — there is no legacy
        ``processed_datasets/`` cache to stay compatible with.
        """
        model = str(self.model_name).replace("/", "-")
        return (f"sr-webqsp_{model}_v{self.rel_mode}_cap{self.max_nodes}_nmax{self.n_max}"
                f"_ver{self.versions}_spd{self.max_spd}_magq{self.magnetic_q}m{self.magnetic_m}"
                f"_len{self.max_length}_rcm{int(self.rcm)}_seed{self.data_seed}")

    def validate(self):
        """Reject accepted-but-unsupported combinations with a clear message.

        The runner is experiment-agnostic and passes any key through; this is where
        the experiment draws its own line (unknown *flags* already fail-fast in
        argparse). Returns ``self`` so ``__main__`` can chain ``.validate()``.
        """
        if self.rel_mode not in REL_MODES:
            raise ValueError(f"rel_mode={self.rel_mode!r} not in {REL_MODES}.")
        if self.graph_attn_impl not in GRAPH_ATTN_IMPLS:
            raise ValueError(f"graph_attn_impl={self.graph_attn_impl!r} not in {GRAPH_ATTN_IMPLS}.")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype={self.dtype!r} not in {DTYPES}.")
        if self.lora_r < 0:
            raise ValueError(f"lora_r must be >= 0 (0 disables LoRA); got {self.lora_r}.")
        if self.n_max < 1:
            raise ValueError(f"n_max must be >= 1; got {self.n_max}.")
        if self.versions < 1:
            raise ValueError(f"versions must be >= 1; got {self.versions}.")
        if not (self.spd or self.magnetic):
            raise ValueError("At least one of spd / magnetic must be enabled.")
        return self
