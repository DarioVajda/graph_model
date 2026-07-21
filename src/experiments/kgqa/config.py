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
# Question-node arm (mechanism-3 fix, 2026-07-15): move the question text out of
# the PROMPT node into its own QUESTION prefix node, so the bidirectional-prefix
# mask lets every graph token attend to the question (question-conditioned graph
# encoding — the flat arm already has this, its serialization is question-first).
# The value picks the QUESTION node's DIRECTED OUT-EDGE set (edges feed the
# SPD/magnetic bias features; token-level visibility comes from the mask either
# way): "all" = an edge to every base-graph node; "topics" = edges to the topic
# entities (mirrors PROMPT's own edges); "isolated" = no edges (the pipeline
# already handles disconnected graphs: isolated topic entities occur in
# production). "off" = no QUESTION node at all — the historical
# single-prompt-node format ("{q}\nAnswer:"). Always a string; None/null is
# rejected so "off" is the single spelling of "disabled".
QUESTION_NODE_MODES = ("off", "all", "topics", "isolated")
# E3 (TODO.md, 2026-07-16): how GraphCollatorV2 assigns each node's starting
# RoPE position. "reset" = historical behavior, every node's tokens start at
# local position 0 (zero inter-node relative-position signal survives RoPE —
# see TODO.md's mechanism-1 writeup). "spd_depth" = a node's tokens start at
# STRIDE * depth(node), depth = shortest-path distance from the prompt node
# (clamped to max_spd, the same far/unreachable bucket SPDBias itself clamps
# into); the prompt node's own depth is one past the deepest prefix node, so
# generation-time query positions stay strictly larger than any prefix node's.
# A TRAIN-TIME collator knob only — not in the .gtds cache key (doesn't touch
# cached graph features) — so both modes reuse the same built dataset.
NODE_POSITION_MODES = ("reset", "spd_depth")
# Graph construction (2026-07-19): how the retrieved subgraph becomes nodes.
#   "levi"    — the historical bipartite Levi graph (h -> rel -> t per triple,
#               single-parent CVT collapse; entities/relations are the nodes).
#   "triplet" — one node PER SELECTED TRIPLE, its text the flat arm's
#               "head | relation | tail" line from the SAME select_triples()
#               call the text-serialization control uses (same budget, same
#               triple set, raw/uncollapsed — a 100%-content-fair graph arm);
#               symmetric edges connect triples sharing an entity. PROMPT's
#               topic edges go to the triples CONTAINING a topic entity;
#               targets still come from the collapsed Levi base, so they are
#               byte-identical to both other arms'.
GRAPH_CONSTRUCTIONS = ("levi", "triplet")
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
    question_node: str = "off"                # QUESTION prefix node (see QUESTION_NODE_MODES above):
                                              # "off" = historical "{q}\nAnswer:" prompt node;
                                              # "all"|"topics"|"isolated" = its directed out-edge set.
                                              # Graph arm + plain prompt style only; in the cache key.
    graph_construction: str = "levi"          # subgraph -> node scheme (see GRAPH_CONSTRUCTIONS
                                              # above): "levi" = historical bipartite Levi graph;
                                              # "triplet" = one node per selected raw triple (the
                                              # flat arm's line text/budget), entity-sharing edges.
                                              # Graph arm only; in the cache key.
    flat_shuffle_lines: bool = False          # E1 diagnostic (2026-07-16): scramble each question's
                                              # triple-line order (once per question, RNG-seeded,
                                              # shared across that question's `versions` rows — mirrors
                                              # how the answer-order augmentation is scoped) to test
                                              # whether flat's edge over the graph arm comes from the
                                              # retrieval-order -> RoPE-position correlation. Flat-only
                                              # (`flat_data_prep`/`flat_train`); in the flat cache key.
    use_gpu: bool = True                      # data-prep only (SPD/magnetic on GPU); train ignores it
    analyse_dataset: bool = False             # data-prep only: coverage-ceiling analysis (not in cache key)

    # ── shared model/bias keys (used by BOTH data prep and the model) ────────
    model_name: str = MODEL_NAME
    spd: bool = True
    max_spd: int = 64                         # data prep: SPD cutoff; model: bucket cap
    magnetic: bool = True
    # Once-per-forward variant of magnetic (see MagneticSharedBias in
    # src/models/bias.py): identical math/params, computed once per forward and
    # shared across layers instead of once per layer. Mutually exclusive with
    # `magnetic` in practice (both consume the same data; combining them just
    # double-counts the term) — the bias ablation sweep never sets both True.
    magnetic_shared: bool = False
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
    node_position_mode: str = "reset"         # E3 (see NODE_POSITION_MODES above): "reset" (historical)
                                              # | "spd_depth". Graph arm only; requires spd=True when
                                              # "spd_depth"; NOT in the .gtds cache key (train-time
                                              # collator behavior only).
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
        if self.magnetic_shared:
            cfg.update(magnetic_shared=True, magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
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
                + ("" if self.resolved_cvt_collapse("graph") else "_nocvt")
                + ("" if self.question_node == "off" else f"_qn{self.question_node}")
                + ("" if self.graph_construction == "levi" else f"_gc{self.graph_construction}"))

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
        if self.question_node not in QUESTION_NODE_MODES:
            hint = ' (use "off", not None/null, to disable)' if self.question_node is None else ""
            raise ValueError(
                f"question_node must be one of {QUESTION_NODE_MODES}; "
                f"got {self.question_node!r}{hint}.")
        if self.question_node != "off":
            if self.resolved_prompt_style != "plain":
                raise ValueError(
                    "question_node requires the plain prompt style (the chat template "
                    "embeds the question in the prompt node; extending it is deferred "
                    "until the scale run needs it, like flat+chat).")
            if self.mode.startswith("flat"):
                raise ValueError(
                    "question_node is a graph-arm knob (the flat serialization is "
                    "already question-first); remove it from flat_* runs.")
        if self.graph_construction not in GRAPH_CONSTRUCTIONS:
            raise ValueError(
                f"graph_construction must be one of {GRAPH_CONSTRUCTIONS}; "
                f"got {self.graph_construction!r}.")
        if self.graph_construction == "triplet":
            if self.mode.startswith("flat"):
                raise ValueError(
                    "graph_construction is a graph-arm knob (the flat serialization "
                    "builds no graph); remove it from flat_* runs.")
            if self.cvt_collapse is not None:
                raise ValueError(
                    "cvt_collapse has no meaning under graph_construction='triplet' "
                    "(nodes are the RAW selected triples, exactly the flat "
                    "serialization's lines; targets come from the collapsed Levi "
                    "base like both other arms) — leave it at the arm default (null).")
        if self.flat_shuffle_lines and not self.mode.startswith("flat"):
            raise ValueError(
                "flat_shuffle_lines is a flat-arm-only diagnostic (it scrambles the "
                "flat serialization's triple-line order; the graph arm has no "
                "equivalent linear order to scramble) — remove it from graph runs.")
        if self.node_position_mode not in NODE_POSITION_MODES:
            raise ValueError(
                f"node_position_mode must be one of {NODE_POSITION_MODES}; "
                f"got {self.node_position_mode!r}.")
        if self.node_position_mode == "spd_depth":
            if not self.spd:
                raise ValueError(
                    "node_position_mode='spd_depth' requires spd=True (it derives each "
                    "node's position from shortest_path_dists, the same feature SPDBias "
                    "consumes).")
            if self.mode.startswith("flat"):
                raise ValueError(
                    "node_position_mode is a graph-arm knob (the flat serialization has "
                    "no per-node RoPE reset to fix); remove it from flat_* runs.")
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
        if self.magnetic and self.magnetic_shared:
            raise ValueError(
                "spd/magnetic/magnetic_shared: magnetic and magnetic_shared are two "
                "placements of the same term (per-layer vs. once-per-forward) — enable "
                "at most one. `spd=magnetic=magnetic_shared=False` is the valid "
                "no-graph-bias ('none') arm.")
        return self
