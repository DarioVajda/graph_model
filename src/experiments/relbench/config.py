"""Shared configuration for the RelBench x GTLM experiment (see PLAN.md).

One ``RunConfig`` holds every knob the entry points read, matching the tag_benchmarks and
kgqa shape so the sweep runner's render -> parse -> config round-trip works unchanged.

Two things here differ from the other experiments and are worth knowing before editing:

**No task-specific values.** Adding a `(dataset, task)` pair must require zero code changes:
the question text comes from relbench metadata plus the task's own docstring, the columns
from dtype and null rules, the answer words from `task_type`, and the node budget from the
schema. There is no `INSTRUCTIONS` dict and no `COLUMN_SPEC` dict, deliberately (PLAN.md
5.0 A).

**The cache key is a hash, not a path built from knobs.** Construction has ~15 axes here
against tag_benchmarks' four, so a readable directory name is not achievable; instead
``data_config_key()`` hashes exactly the knobs that change the built bytes, and the
directory carries a readable prefix plus that hash. Training-only knobs (seed, lr, bias
arms) are excluded so all three architecture arms share one built dataset.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from .row_text import TEXT_MODES

MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "relbench"

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EXPERIMENT_DIR, "processed_data")
RAW_DIR = os.path.join(EXPERIMENT_DIR, "raw_data")

# relbench caches downloads here unless told otherwise. Set before any relbench import so
# login node and compute nodes agree on one repo-local, gitignored path.
os.environ.setdefault("RELBENCH_CACHE_DIR", RAW_DIR)

SPLITS = ("train", "val", "test")
IMPLS = ("v2-flex", "v2-eager")
DTYPES = ("bf16", "fp32")
MODES = ("train", "data_prep", "dump")

# PLAN.md 5.4's menu. `recent`/`uniform` map onto PyG's `temporal_strategy`; the rest need
# `sampler_policies.py` and are not implemented yet.
SAMPLING = ("recent", "uniform")
SAMPLING_PLANNED = ("recent_plus_strided", "paper_match", "mixed")

# `TEXT_MODES` is imported from row_text (its single source of truth): `key_value` (control,
# default) labels every field on every row; `schema_node` hoists each table's column list
# into a header node and renders rows positionally; `shortest` hoists a table only when that
# is actually shorter for the rows sampled, so it is never worse than `key_value`.
TIME_ENCODINGS = ("relative", "absolute", "both")
ANONYMIZE = ("none", "entities", "all")
QUESTION_NODES = ("isolated", "seed", "off")
# Where the answer-bearing PROMPT node sits in the topology. kgqa attaches it to the nodes
# being asked about unconditionally (`process_dataset.py:434`); "seed" is that convention,
# with node 0 (the entity row) standing in for kgqa's topic entities. Edges feed the bias,
# never the attention mask, so an attached prompt node is still strictly causal and the
# answer is still unreadable from the position that predicts it.
PROMPT_NODES = ("seed", "isolated")
# Binary tasks read `logit(" yes") - logit(" no")` off the LM head in fp32; regression
# generates the number. Neither adds parameters -- the point is to measure GTLM itself, not
# a probe on top of it, so there is deliberately no MLP-head readout here.
READOUTS = ("logit_margin", "numeric_text")

WIRED_FEATURES = ("spd", "rrwp", "magnetic")
# The default stack. RRWP is wired but off: 13x the storage for -0.8 F1 on WebQSP.
DEFAULT_ON_FEATURES = ("spd", "magnetic")
UNWIRED_FEATURES = ("laplacian", "rwse")

# The supervised span. Masking matches the LAST occurrence, because a row's text can contain
# "Answer:" but the prompt node's instruction always ends with it.
ANSWER_PREFIX = "Answer:"

# Knobs that change the built bytes. Anything not listed is training-only and must not be,
# or ablation arms stop sharing a cache and every arm pays a rebuild.
_CACHE_KEYS = (
    "dataset", "task", "max_nodes", "neighbor_sampling", "collapse_links",
    "sibling_fanout", "parent_fanout", "relation_cap", "label_history", "aggregates",
    "text_mode", "time_encoding", "anonymize", "max_value_chars", "max_node_chars",
    "null_threshold", "question_node", "prompt_node", "model_name", "max_length", "magnetic_q",
    "magnetic_m", "readout", "regression_buckets", "data_seed", "samples_per_node",
    # The smoke-run caps stride the train/val builds (data.py), so they change the built
    # bytes. Leaving them out let a 201-row smoke cache satisfy `_is_built()` for a
    # full-scale run, which then trained on 201 of 11,411 rows and reported it as a real
    # number. Loud failure is impossible here -- strided row_ids are still monotonic, so
    # the ordering assert passes.
    "max_train_samples", "max_val_samples",
)


@dataclass
class RunConfig:
    """Every knob the entry points read (built from argparse by ``__main__``)."""

    mode: str = "train"

    # -- what to predict ----------------------------------------------------
    dataset: str = "rel-f1"
    task: str = "driver-dnf"

    # -- neighborhood construction (cache key) ------------------------------
    # `max_nodes` counts CONTENT rows: pure join rows ride along free, because they carry
    # nothing but the link (PLAN.md 4.2). Sized from the measured degree distribution --
    # 64 for rel-f1 (hop-1 p50 is 53), 24 for rel-trial (p50 11, and its rows are far
    # wider). `analyse_dataset.py` is where those numbers come from; do not guess new ones.
    max_nodes: int = 64
    neighbor_sampling: str = "recent"
    collapse_links: bool = True            # contract contentless join rows into edges
    sibling_fanout: int = 0                # >0 enables PLAN.md 5.4's `include_siblings`
    parent_fanout: int = 1                 # a row has exactly one parent per fkey
    relation_cap: int = None               # per-relation cap handed to PyG (None -> max_nodes)
    label_history: int = 0                 # k past task rows for the same entity
    aggregates: str = "off"                # "off" | "seed" (not implemented yet)

    # -- row -> text (cache key) --------------------------------------------
    text_mode: str = "key_value"
    time_encoding: str = "relative"
    anonymize: str = "none"
    # Per-field character cap. On rel-trial this is the single most consequential knob:
    # `detailed_descriptions` and `criteria` are the free text the whole experiment is
    # about, and at 200 the documents came out 3x smaller than the LLM baseline's smallest
    # configuration (PLAN.md 6.1.1). 1200 is the rel-trial default; rel-f1 never truncates.
    max_value_chars: int = 200
    # Per-NODE character cap, or None for no cap (the default). None, not a number, because
    # a node cap silently overrides the per-field cap: at 600 it truncated 95.5% of rel-trial
    # `studies` rows -- the seed nodes, the ones holding `detailed_descriptions` and
    # `brief_summaries` -- so raising `max_value_chars` behind it changed nothing at all.
    # `max_value_chars` bounds each field and `max_length` bounds the sequence, so an
    # uncapped node is already bounded twice over. rel-f1 never came close (longest row: 179
    # chars), so this only ever bound where it did the most damage.
    max_node_chars: int = None
    null_threshold: float = 0.95

    # -- graph layout (cache key) -------------------------------------------
    # `isolated` won the kgqa ablation outright: the bidirectional prefix mask already
    # exposes the question to every graph token, which is what makes the encoding
    # question-conditioned without an edge.
    question_node: str = "isolated"
    # "seed" attaches the answer node to the entity row (kgqa's convention, and reported
    # there to help slightly); "isolated" leaves it edgeless. This is a construction knob:
    # it changes the SPD and magnetic tensors, so it is in the cache key.
    prompt_node: str = "seed"

    # -- readout ------------------------------------------------------------
    readout: str = "logit_margin"
    regression_buckets: int = 0            # 0 -> numeric_text; >0 -> quantile buckets

    # -- model --------------------------------------------------------------
    model_name: str = MODEL_NAME
    impl: str = "v2-flex"
    flex_compile_mode: str = "max-autotune-no-cudagraphs"
    dtype: str = "bf16"

    k_hop: int = 0
    k_hop_directed: bool = False

    # -- graph-bias features ------------------------------------------------
    # Data prep computes SPD and magnetic always so ablation arms share a cache. RRWP is
    # opt-in: it was 13x the storage for -0.8 F1 on WebQSP.
    spd: bool = True
    max_spd: int = 8
    rrwp: bool = False
    max_rw_steps: int = 16
    magnetic: bool = True
    magnetic_dim: int = 32
    magnetic_q: float = 0.25
    magnetic_m: int = 0
    laplacian: bool = False
    rwse: bool = False

    # -- dataset construction -----------------------------------------------
    max_length: int = 32_768
    samples_per_node: int = 1
    data_seed: int = 42
    use_gpu: bool = True
    # Smoke-run caps: stride the train/val build down to roughly this many rows. Both are
    # in the cache key, because a strided build is a different dataset.
    max_train_samples: int = None
    max_val_samples: int = None
    # Never implemented, and `test_subsample` never should be: `task.evaluate` compares the
    # prediction vector to the full test table positionally, so a subsampled test split
    # produces a number that is not the benchmark's. `validate()` rejects both.
    val_subsample: int = None
    test_subsample: int = None

    # -- arm ----------------------------------------------------------------
    # "graph" is the full stack; "flat" serializes the same sampled rows into one node with
    # no structural bias -- the matched control, and the primary scientific comparison.
    arm_name: str = "graph"

    # -- LoRA ---------------------------------------------------------------
    lora: bool = True
    lora_r: int = 32
    lora_dropout: float = 0.05

    # -- schedule -----------------------------------------------------------
    num_epochs: int = 4
    batch_size: int = 1
    accumulation_steps: int = 32
    lr: float = 3e-4
    bias_lr: float = 5e-2
    eval_steps: int = 100
    max_steps: int = -1
    seed: int = 42
    num_workers: int = 0
    gradient_checkpointing: bool = True

    wandb_project: str = None

    # -- derived ------------------------------------------------------------

    def torch_dtype(self):
        import torch
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.dtype]

    def backend(self):
        return self.impl.split("-", 1)[1]

    def is_flat(self):
        return self.arm_name == "flat"

    def lora_config(self):
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout}

    def bias_params(self):
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim)
        return cfg

    def arm(self):
        """Short ablation-arm label for records and tables.

        Measured against this experiment's default stack (SPD + magnetic), not against every
        wired feature: RRWP is off by default, so treating it as a deviation would label
        every ordinary run `no-rrwp`. Turning it on is the deviation.
        """
        if self.is_flat():
            return "flat"
        off = [f for f in DEFAULT_ON_FEATURES if not getattr(self, f)]
        label = "base" if not off else "no-" + "+".join(off)
        return f"{label}+rrwp" if self.rrwp else label

    def cache_fields(self):
        """Exactly the knobs that change the built bytes."""
        fields = {k: getattr(self, k) for k in _CACHE_KEYS}
        # RRWP only enters the identity when it is on, so caches stay valid across arms that
        # leave it off -- kgqa's convention.
        if self.rrwp:
            fields["rrwp"] = True
            fields["max_rw_steps"] = self.max_rw_steps
        # The flat control is a different serialization of the SAME sampled rows, so it must
        # be a different cache but share every sampling knob.
        fields["arm_kind"] = "flat" if self.is_flat() else "graph"
        return fields

    def data_config_key(self):
        """Short stable hash of the construction, used as the cache directory suffix."""
        blob = json.dumps(self.cache_fields(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def dataset_dir(self):
        """Where this config's built splits live.

        A readable prefix keeps the directory listing navigable; the hash is what actually
        distinguishes configs, because there are far too many construction axes to spell
        out in a path.
        """
        # The prefix must contain only knobs that are IN the cache key. `arm()` is not: the
        # bias arms (base / no-spd+magnetic) share one built dataset by design, and putting
        # `arm()` here gave them the same key but three different directories, so each would
        # rebuild byte-identical data. Only the graph/flat split is a real construction axis.
        prefix = (f"{self.dataset}_{self.task}_n{self.max_nodes}"
                  f"_{self.neighbor_sampling}_{'flat' if self.is_flat() else 'graph'}")
        return os.path.join(DATASETS_DIR, "datasets", f"{prefix}__{self.data_config_key()}")

    def run_name(self):
        lora_tag = f"_lora{self.lora_r}" if self.lora else ""
        return (f"{EXPERIMENT_NAME}_{self.dataset}_{self.task}_{self.arm()}"
                f"_n{self.max_nodes}{lora_tag}_s{self.seed}")

    def validate(self):
        """Reject unsupported settings before any GPU work, with an actionable message."""
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r} (expected one of {MODES}).")
        if self.impl not in IMPLS:
            raise ValueError(f"Unknown impl {self.impl!r} (expected one of {IMPLS}).")
        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r} (expected one of {DTYPES}).")

        if self.neighbor_sampling in SAMPLING_PLANNED:
            raise ValueError(
                f"neighbor_sampling {self.neighbor_sampling!r} is planned (PLAN.md 5.4) but "
                f"not implemented: it needs sampler_policies.py. Available now: {SAMPLING}.")
        if self.neighbor_sampling not in SAMPLING:
            raise ValueError(
                f"Unknown neighbor_sampling {self.neighbor_sampling!r} (expected {SAMPLING}).")

        for name, allowed in (("text_mode", TEXT_MODES), ("time_encoding", TIME_ENCODINGS),
                              ("anonymize", ANONYMIZE), ("question_node", QUESTION_NODES),
                              ("prompt_node", PROMPT_NODES), ("readout", READOUTS)):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"Unknown {name} {getattr(self, name)!r} (expected one of {allowed}).")

        if self.arm_name not in ("graph", "flat"):
            raise ValueError(f"Unknown arm_name {self.arm_name!r} (expected 'graph' or 'flat').")
        if self.is_flat() and any(getattr(self, f) for f in WIRED_FEATURES):
            raise ValueError(
                "The flat control must run with every structural bias off -- it exists to "
                "isolate the bias channel at byte-identical supervision. Pass "
                "--no-spd --no-magnetic (and --no-rrwp if set).")

        enabled_unwired = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if enabled_unwired:
            raise ValueError(
                f"Bias feature(s) {enabled_unwired} are exposed in the config but never "
                f"produced by data prep, so their bias is identically zero -- a silent "
                f"no-op. Disable them, or extend data.py to compute them.")

        if self.aggregates != "off":
            raise ValueError(
                f"aggregates={self.aggregates!r} is planned (PLAN.md 5.4 fix 2) but not "
                f"implemented; only 'off' works today.")
        if self.label_history:
            raise ValueError(
                "label_history is planned (PLAN.md 5.4) but not implemented. Note it is "
                "structurally empty on rel-trial anyway: study-outcome has 1.00 task rows "
                "per entity, so there is no per-entity label history to add (PLAN.md 4.2).")

        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1; got {self.max_nodes}.")
        if self.max_value_chars < 1:
            raise ValueError("max_value_chars must be >= 1.")
        if self.max_node_chars is not None:
            if self.max_node_chars < 1:
                raise ValueError("max_node_chars must be >= 1, or None for no cap.")
            if self.max_node_chars < self.max_value_chars:
                raise ValueError(
                    f"max_node_chars={self.max_node_chars} is below "
                    f"max_value_chars={self.max_value_chars}, which makes the per-field cap "
                    f"unreachable: every node is cut before a single field can use its "
                    f"budget. Raise it, or set it to None for no node cap.")
        if not 0 < self.null_threshold <= 1:
            raise ValueError(f"null_threshold must be in (0, 1]; got {self.null_threshold}.")
        if self.samples_per_node < 1:
            raise ValueError(f"samples_per_node must be >= 1; got {self.samples_per_node}.")
        if self.k_hop < 0:
            raise ValueError("k_hop must be >= 0 (0 disables the mask).")
        if self.k_hop and self.prompt_node == "isolated":
            raise ValueError(
                "k_hop > 0 with prompt_node='isolated' blinds the readout: the K-hop mask "
                "builds the prompt node's row from its own edges "
                "(text_graph_collator_v2._single_k_hop_mask), and an edgeless prompt node "
                "reaches nothing but itself. This is the prompt-node gating that explained "
                "kgqa's k_hop collapse. Use prompt_node='seed', or leave k_hop at 0.")
        if self.lora_r < 0:
            raise ValueError("lora_r must be >= 0.")
        for name in ("max_train_samples", "max_val_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None (no cap); got {value}.")
        if self.val_subsample is not None:
            raise ValueError(
                "val_subsample is not implemented; use max_val_samples, which strides the "
                "val build and is part of the cache key.")
        if self.test_subsample is not None:
            raise ValueError(
                "test_subsample is not implemented, deliberately. `task.evaluate` compares "
                "predictions to the full test table positionally, so a subsampled test "
                "split yields a number that is not comparable to any published baseline.")
        return self
