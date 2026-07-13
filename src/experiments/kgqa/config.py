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

import dataclasses
from dataclasses import dataclass


MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Data-format version, appended to the cache key: bump when the pipeline's
# SEMANTICS change without any config knob changing, so stale caches can't be
# silently reused. v2 = literal-kb_id fallback for answer texts + unanswerable
# dev/test questions kept as empty-target eval rows (benchmark denominators).
# v3 = newline answer separator (commas appear INSIDE 12.7% of test golds,
# making comma-joined sets unlearnable/unparseable; "\n" is collision-free and
# matches GNN-RAG's own generation format) + naming v2 (entities_names.json
# extended with a broader mid->name alias source).
# This module constant is only the DEFAULT for ``RunConfig.data_format_version``
# — the version is a real config knob so sweeps can pin an older format (e.g.
# a dfv2 control arm) against current code.
DATA_FORMAT_VERSION = 3

# Which mid->name dictionary node naming reads. NOT the same axis as
# DATA_FORMAT_VERSION (and the numbers run opposite ways): naming v1 is the 560k
# dict shipped by GNN-RAG, naming v2 the 598.5k Freebase-alias extension built by
# ``build_entities_names_v2.py``. Data-format v3 *bundles* naming v2, so a dfv2
# control arm must pin ``naming_version=1`` to be a true control — it is a real
# config knob (and part of the cache key) precisely so that pinning works
# regardless of what ``entities_names.json`` currently holds on disk.
NAMING_VERSION = 2
ENTITY_NAMES_FILES = {1: "entities_names.v1.json", 2: "entities_names.json"}

DATASETS = ("webqsp", "cwq")   # mirrored in sr_records.DATASETS (no import: keep config dep-free)
REL_MODES = ("last_1", "last_2", "full")
GRAPH_ATTN_IMPLS = ("flex", "eager")
DTYPES = ("bf16", "fp32")
PROMPT_STYLES = ("plain", "chat")

# D3 (design locked 2026-07-08): the chat prompt style wraps the question in the
# Llama-3 chat template INSIDE the single prompt node (graph nodes unchanged).
# The system turn is pinned to a minimal constant — never the template default,
# whose auto-inserted "Today Date" would make cache content depend on build date.
PINNED_SYSTEM_PROMPT = "Answer the question using the provided knowledge graph context."
# The assistant-header token sequence replaces "Answer:" as the label/generation
# boundary (question_end) under the chat style.
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# LoRA target modules for the Llama backbone (fixed architecture choice).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class RunConfig:
    """Every knob the data_prep / train entry points read (built from argparse)."""

    # ── what to run ──────────────────────────────────────────────────────────
    mode: str = "train"                       # "train" | "data_prep"

    # ── dataset roles (E2.1, TODO_cwq) ────────────────────────────────────────
    # Which benchmark(s) to train on and which to evaluate on — each a non-empty
    # subset of DATASETS. Mixed training is a plain concat of the built train
    # splits; eval datasets are ALWAYS scored separately (never pooled), each
    # under its own metric namespace (eval_{ds}_f1, test_{ds}_hits1, ...).
    # The `--dataset X` CLI alias sets both to (X,).
    train_datasets: tuple = ("webqsp",)
    eval_datasets: tuple = ("webqsp",)
    # Checkpoint selection for multi-eval runs: None = mean of the per-dataset
    # dev F1s (logged as eval_sel_f1); a dataset name pins selection to that
    # benchmark alone (must be in eval_datasets).
    selection_dataset: str = None

    # ── data-prep keys (these determine the .gtds cache directory) ───────────
    rel_mode: str = "last_1"                  # relation verbalization: last_1|last_2|full
    # max_nodes / n_max / versions are DATASET-RESOLVABLE (E2.2): a scalar
    # applies to every dataset; a {"webqsp": 512, "cwq": 1024} mapping (CLI:
    # --max-nodes webqsp=512,cwq=1024) resolves per dataset. Cache keys embed
    # the RESOLVED values, so e.g. WebQSP keys are byte-stable regardless of
    # CWQ settings. versions joined the set 2026-07-12: CWQ builds use 1
    # (near-singleton answer sets make order augmentation a no-op) while
    # WebQSP keeps 8.
    max_nodes: "int | dict" = 512             # Levi-graph (+prompt) node cap
    n_max: "int | dict" = 20                  # max answers kept in the training target
    versions: "int | dict" = 8                # per-graph answer-order augmentations (train only)
    max_length: int = 1024                    # per-node token cap (kept non-binding)
    rcm: bool = True                          # reverse-Cuthill-McKee node ordering
    data_seed: int = 42                       # augmentation RNG seed (baked into cache key)
    data_format_version: int = DATA_FORMAT_VERSION   # pipeline-semantics version (cache key + answer separator)
    naming_version: int = NAMING_VERSION      # mid->name dict: 1 = GNN-RAG's 560k, 2 = +Freebase aliases
    prompt_style: str = None                  # "plain" | "chat" | None = auto ("chat" iff Instruct model)
    cvt_collapse: bool = None                 # single-parent CVT mediator collapse. None = arm default:
                                              # True for the graph pipeline (its historical behavior),
                                              # False for the flat serialization (raw triples). The 2x2
                                              # collapse ablation (2026-07-09) sets it explicitly.
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
    seq_len: int = 4096                       # flat (text-serialization) arm only: max prompt+target
                                              # tokens (covers 99.6%+; outliers drop trailing triples)
    boundary_loss_weight: float = 1.0         # D4: loss weight on the "\n" answer-boundary token
                                              # (1.0 = off; renormalized, see KGQAGraphTrainer)
    # Regularization (survivors of the TODO_reg probe campaign; defaults
    # preserve historical behavior — graph-bias params were accidentally
    # undecayed, LoRA dropout was hard-coded 0.05).
    bias_weight_decay: float = 0.0            # AdamW decay on graph-bias MATRICES (see GraphTrainerV2)
    lora_dropout: float = 0.05                # LoRA adapter-input dropout (both arms)

    # ── generative-eval keys ─────────────────────────────────────────────────
    gen_max_new_tokens: int = 128
    gen_max_samples: int = None               # final dev/test scoring cap; None = full split
    gen_eval_samples: int = None              # in-training eval cap (checkpoint selection only);
                                              # None = fall back to gen_max_samples. When it caps
                                              # a split, the subsample is FIXED and seeded
                                              # (comparable across checkpoints), not first-n.
    log_strict_metrics: bool = False          # also compute/log the _strict exact-set variants
                                              # (E3.2: opt-in; the GNN-RAG-verbatim metrics are
                                              # always logged)

    # ── tracking ─────────────────────────────────────────────────────────────
    wandb_project: str = None                 # None = no tracking; a string = the wandb project

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def datasets(self):
        """Every dataset the run references (train ∪ eval), in DATASETS order."""
        referenced = set(self.train_datasets) | set(self.eval_datasets)
        return tuple(d for d in DATASETS if d in referenced)

    def _per_dataset(self, name, dataset):
        """Resolve a dataset-resolvable knob: scalar = every dataset, mapping = per key."""
        value = getattr(self, name)
        if isinstance(value, dict):
            if dataset not in value:
                raise ValueError(f"{name}={value} has no entry for dataset {dataset!r}.")
            return int(value[dataset])
        return value

    def resolved_max_nodes(self, dataset):
        return self._per_dataset("max_nodes", dataset)

    def resolved_n_max(self, dataset):
        return self._per_dataset("n_max", dataset)

    def resolved_versions(self, dataset):
        return self._per_dataset("versions", dataset)

    def for_dataset(self, dataset):
        """A single-dataset VIEW of this config: the dataset-resolvable knobs
        collapsed to that dataset's scalars, roles narrowed to it. The data
        pipeline (process_dataset / flat_data / analyse_dataset) always works on
        one dataset at a time and reads plain-int ``max_nodes``/``n_max``/
        ``versions`` off the view, exactly like the pre-multi-dataset code."""
        if dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}; got {dataset!r}")
        return dataclasses.replace(
            self,
            train_datasets=(dataset,), eval_datasets=(dataset,), selection_dataset=None,
            max_nodes=self.resolved_max_nodes(dataset),
            n_max=self.resolved_n_max(dataset),
            versions=self.resolved_versions(dataset))

    @property
    def answer_sep(self):
        """Separator JOINING answers in the training target (v3: newline)."""
        return "\n" if self.data_format_version >= 3 else ", "

    @property
    def answer_parse_sep(self):
        """Separator the evaluator SPLITS generations on (must mirror ``answer_sep``)."""
        return "\n" if self.data_format_version >= 3 else ","

    @property
    def resolved_prompt_style(self):
        """``prompt_style`` with the auto default applied: "chat" iff the backbone
        is an Instruct variant, unless explicitly overridden (the instruct+plain
        control arm sets ``prompt_style="plain"`` on an Instruct model)."""
        if self.prompt_style is not None:
            return self.prompt_style
        return "chat" if "Instruct" in str(self.model_name) else "plain"

    @property
    def question_end_str(self):
        """The label-mask / generation-cut anchor string (tokenized by callers).

        Plain: the literal "Answer:" (the prompt is "{q}\\nAnswer: ..."); chat:
        the assistant-header sequence — everything up to and including it is
        prompt, everything after is target. ``AnswerLabelMasker`` and
        ``_find_prefix_len`` match arbitrary token subsequences, so only this
        constant changes between the styles."""
        return ASSISTANT_HEADER if self.resolved_prompt_style == "chat" else "Answer:"

    @property
    def entity_names_file(self):
        """Basename of the mid->name dict this config's node naming reads."""
        return ENTITY_NAMES_FILES[self.naming_version]

    def resolved_cvt_collapse(self, arm):
        """``cvt_collapse`` with the arm default applied (``arm``: "graph"|"flat").

        The graph pipeline has always collapsed single-parent CVT mediators; the
        flat serialization was deliberately built on the RAW pre-collapse triples
        (the collapse is our graph processing). The 2x2 collapse ablation crosses
        the knob explicitly; every pre-existing cache keeps its key because only
        non-default values add a suffix."""
        if self.cvt_collapse is not None:
            return self.cvt_collapse
        return arm == "graph"

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
            "lora_dropout": self.lora_dropout,
            "bias": "none",
        }

    def data_config_key(self, dataset):
        """Cache-directory name for ONE dataset, built ONLY from data-affecting fields.

        Training-only knobs (seed, lr, k_hop, …) are deliberately excluded so two
        runs differing only in training config share one built dataset. Uses
        ``data_seed`` (not the training ``seed``) and includes ``max_length``
        (which affects tokenization). Ends in the ``DATA_FORMAT_VERSION`` so
        semantic pipeline changes invalidate old caches.

        Dataset-resolvable knobs are embedded as their RESOLVED values, so one
        dataset's key never depends on another dataset's settings (WebQSP keys
        stay byte-identical to their pre-multi-dataset names).
        """
        model = str(self.model_name).replace("/", "-")
        # prompt_style and naming_version are data-affecting, but each suffix is
        # only added for non-default values so every pre-existing cache keeps its
        # name (plain style, naming v2).
        ps = self.resolved_prompt_style
        # f"sr-{dataset}" keeps every pre-existing key byte-identical (the prefix
        # was the literal "sr-webqsp_" before the dataset knob existed).
        return (f"sr-{dataset}_{model}_v{self.rel_mode}"
                f"_cap{self.resolved_max_nodes(dataset)}_nmax{self.resolved_n_max(dataset)}"
                f"_ver{self.resolved_versions(dataset)}_spd{self.max_spd}"
                f"_magq{self.magnetic_q}m{self.magnetic_m}"
                f"_len{self.max_length}_rcm{int(self.rcm)}_seed{self.data_seed}"
                f"_dfv{self.data_format_version}"
                + ("" if ps == "plain" else f"_ps{ps}")
                + ("" if self.naming_version == NAMING_VERSION else f"_nm{self.naming_version}")
                + ("" if self.resolved_cvt_collapse("graph") else "_nocvt"))

    def validate(self):
        """Reject accepted-but-unsupported combinations with a clear message.

        The runner is experiment-agnostic and passes any key through; this is where
        the experiment draws its own line (unknown *flags* already fail-fast in
        argparse). Returns ``self`` so ``__main__`` can chain ``.validate()``.
        """
        for role in ("train_datasets", "eval_datasets"):
            names = getattr(self, role)
            if not names or not set(names) <= set(DATASETS):
                raise ValueError(f"{role}={names!r} must be a non-empty subset of {DATASETS}.")
            if len(set(names)) != len(names):
                raise ValueError(f"{role}={names!r} contains duplicates.")
        if self.selection_dataset is not None and self.selection_dataset not in self.eval_datasets:
            raise ValueError(
                f"selection_dataset={self.selection_dataset!r} must be one of "
                f"eval_datasets={self.eval_datasets!r} (or None = mean of their dev F1s).")
        for name in ("max_nodes", "n_max", "versions"):
            value = getattr(self, name)
            if isinstance(value, dict) and not set(value) <= set(DATASETS):
                raise ValueError(f"{name}={value} has keys outside {DATASETS}.")
            for ds in self.datasets:
                self._per_dataset(name, ds)   # raises if a referenced dataset is unresolvable
        if self.rel_mode not in REL_MODES:
            raise ValueError(f"rel_mode={self.rel_mode!r} not in {REL_MODES}.")
        if self.graph_attn_impl not in GRAPH_ATTN_IMPLS:
            raise ValueError(f"graph_attn_impl={self.graph_attn_impl!r} not in {GRAPH_ATTN_IMPLS}.")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype={self.dtype!r} not in {DTYPES}.")
        if self.lora_r < 0:
            raise ValueError(f"lora_r must be >= 0 (0 disables LoRA); got {self.lora_r}.")
        for ds in self.datasets:
            if self.resolved_n_max(ds) < 1:
                raise ValueError(f"n_max must be >= 1; got {self.resolved_n_max(ds)} for {ds}.")
            if self.resolved_versions(ds) < 1:
                raise ValueError(f"versions must be >= 1; got {self.resolved_versions(ds)} for {ds}.")
        if self.data_format_version not in (2, 3):
            raise ValueError(
                f"data_format_version must be 2 or 3; got {self.data_format_version}.")
        if self.naming_version not in ENTITY_NAMES_FILES:
            raise ValueError(
                f"naming_version must be one of {sorted(ENTITY_NAMES_FILES)}; "
                f"got {self.naming_version}.")
        if self.prompt_style is not None and self.prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"prompt_style must be one of {PROMPT_STYLES} or None (auto); "
                f"got {self.prompt_style!r}.")
        if self.boundary_loss_weight <= 0:
            raise ValueError(
                f"boundary_loss_weight must be > 0; got {self.boundary_loss_weight}.")
        if self.boundary_loss_weight != 1.0 and self.data_format_version < 3:
            raise ValueError(
                "boundary_loss_weight requires the dfv3 newline separator "
                "(the boundary token is '\\n').")
        if self.bias_weight_decay < 0:
            raise ValueError(f"bias_weight_decay must be >= 0; got {self.bias_weight_decay}.")
        if not (0 <= self.lora_dropout < 1):
            raise ValueError(f"lora_dropout must be in [0, 1); got {self.lora_dropout}.")
        if not (self.spd or self.magnetic):
            raise ValueError("At least one of spd / magnetic must be enabled.")
        return self
