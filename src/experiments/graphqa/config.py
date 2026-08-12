"""
Shared configuration for the GraphQA experiment.

One ``RunConfig`` dataclass holds EVERY knob the entry points read; ``__main__.py``
builds a ``RunConfig`` from argparse and stays a thin dispatcher, so a value lives
in exactly one place.

GraphQA (`baharef/GraphQA <https://huggingface.co/datasets/baharef/GraphQA>`_) poses
one algorithmic question per graph (count the nodes, find a shortest path, ...). Each
example becomes a text-attributed graph whose prompt node carries the question and
answer; supervision is the answer span only, scored by exact match.

Two graph encodings are compared (``graph_type``): ``standard`` (one node per graph
vertex) and ``incidence`` (the bipartite Levi graph, one node per vertex AND per
edge — hence ~5-10x the nodes). The ablation turns the three graph-bias features
(``spd`` / ``rrwp`` / ``magnetic``) on and off independently.

This experiment runs the **v2** model only (``GTLMLlamaForCausalLM``). The legacy v0
path it used historically is pinned as numerically equivalent to v2-eager at fp32 by
``tests/models/test_modeling_gtlm_llama_v2.py`` (loss + every bias gradient) and
``tests/models/test_v2_ragged_magnetic_padding.py`` (the same, at the wide within-batch
node spread this experiment's incidence graphs produce), so nothing is lost by dropping
it — and k-hop masking is gained for free.
"""

import os
from dataclasses import dataclass


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "graphqa"

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EXPERIMENT_DIR, "processed_datasets")
RAW_DIR = os.path.join(EXPERIMENT_DIR, "hf_dataset")

GRAPH_TYPES = ("standard", "incidence")
IMPLS = ("v2-eager", "v2-flex")
DTYPES = ("fp32", "bf16")

# Backbones this experiment can train, keyed by a substring of ``model_name`` (so the
# backbone is not a second knob that can disagree with the checkpoint). Llama-3 is the
# experiment's model; the other two are positional-encoding probes — evidence that the
# graph bias is not tied to one PE scheme — run on one small task rather than across
# the ablation grid:
#
#   llama    single global RoPE, every layer               (the production backbone)
#   bloom    ALiBi: additive slope bias, no rotation       (modeling_gtlm_bloom.py)
#   gemma-3  layer-heterogeneous RoPE: most layers at a local base
#            (rope_local_base_freq), every 6th at a global one (rope_theta)
#                                                          (modeling_gtlm_gemma3.py)
BACKBONES = {
    "llama": dict(
        classes=("GTLMLlamaConfig", "GTLMLlamaForCausalLM"),
        # PEFT's defaults are model_type-keyed and don't know our gtlm_* types, so the
        # LoRA targets are named per backbone here.
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"),
        impls=IMPLS,
    ),
    "bloom": dict(
        classes=("GTLMBloomConfig", "GTLMBloomForCausalLM"),
        lora_targets=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
        # The BLOOM adapter wires the eager backend only (BLOOM's hand-written
        # attention never dispatches to the registered gtlm_flex function).
        impls=("v2-eager",),
    ),
    "gemma-3": dict(
        classes=("GTLMGemma3Config", "GTLMGemma3ForCausalLM"),
        # Gemma-3 uses Llama's projection names, so the same LoRA targets apply.
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"),
        # Strategy B like Llama (stock attention dispatches through HF's attention
        # interface), so both backends are reachable — unlike the BLOOM probe.
        impls=IMPLS,
    ),
}

# Checkpoints that *look* like a wired backbone but are not, with the reason. Checked
# before the BACKBONES substring match so the failure names the actual obstacle rather
# than "no backbone wires this" (gemma-2) or blowing up later inside from_pretrained
# on a config shape we never handle (multimodal gemma-3).
_MULTIMODAL_GEMMA3 = (
    "Only the text-only gemma-3-1b checkpoints load through GTLMGemma3Config. The 4b "
    "and larger Gemma-3 releases are multimodal: their config nests the text config "
    "under `text_config`, which this experiment does not unwrap."
)
_GEMMA2_SOFTCAP = (
    "Gemma-2 ships attn_logit_softcapping=50.0 and final_logit_softcapping=30.0, which "
    "the shared GTLM stack applies at neither site — training it would silently diverge "
    "from the pretrained backbone (GTLMGemma3ForCausalLM refuses such a config by "
    "design). Gemma-3 sets both to None; use gemma-3-1b-pt, which is also the more "
    "interesting probe — Gemma-2 is plain global RoPE, positionally identical to Llama."
)

UNWIRED_BACKBONES = {
    "gemma-2": _GEMMA2_SOFTCAP,
    "gemma-3-4b": _MULTIMODAL_GEMMA3,
    "gemma-3-12b": _MULTIMODAL_GEMMA3,
    "gemma-3-27b": _MULTIMODAL_GEMMA3,
}

# Where the question text lives (mirrors the kgqa experiment's arm of the same
# name). "off" — the historical layout: the prompt node carries "{question}{answer}"
# together, so graph tokens never see the question. "isolated" — the question moves
# into its own QUESTION *prefix* node with NO edges to the graph; the bidirectional
# prefix mask then lets every graph token attend to it (question-conditioned graph
# encoding), while the prompt node is left holding just the "A:" anchor + answer, so
# the supervised span and generation anchor are byte-identical to "off".
QUESTION_NODES = ("off", "isolated")

# Every task `process_dataset` can build. (`maximum_flow` ships in the raw download
# but is unbuildable — the released files omit the edge capacities the task needs.)
TASKS = (
    "connected_nodes", "disconnected_nodes", "cycle_check", "edge_count",
    "edge_existence", "node_classification", "node_count", "node_degree",
    "reachability", "shortest_path", "triangle_counting",
)

# The nine tasks the paper table reports (`analysis/prep_table.py`). These are exactly
# the tasks that ship an official validation split, so the reported protocol never
# falls back to carving val out of train.
REPORTED_TASKS = (
    "node_count", "edge_count", "cycle_check", "triangle_counting", "node_degree",
    "connected_nodes", "reachability", "edge_existence", "shortest_path",
)

# Tasks whose raw download has no `*_zero_shot_validation.json`; these fall back to
# carving `val_fraction` off the end of train (see `data.py`). None are reported.
TASKS_WITHOUT_OFFICIAL_VAL = ("disconnected_nodes", "node_classification")

# Graph-bias features whose data + model modules are both wired here.
WIRED_FEATURES = ("spd", "rrwp", "magnetic")
# Exposed in the schema but not produced by data prep -> rejected by validate().
UNWIRED_FEATURES = ("laplacian", "rwse")

# The feature-generation settings the on-disk dataset cache was built with. A config
# matching these resolves to the historical, untagged cache path (see `dataset_dir`).
_CACHE_DEFAULTS = dict(
    model_name=MODEL_NAME, max_length=1024,
    magnetic_q=0.25, magnetic_m=0, max_rw_steps=16,
    question_node="off",
)


@dataclass
class RunConfig:
    """Every knob the train / data_prep entry points read (built from argparse)."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"                         # "train" | "data_prep"
    task: str = "shortest_path"                 # which GraphQA task (see TASKS)
    graph_type: str = "standard"                # "standard" | "incidence"
    question_node: str = "off"                  # "off" | "isolated" (see QUESTION_NODES)

    # ── model ──────────────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    # v2-eager is the default because it is the implementation pinned equivalent to
    # the historical v0 runs, and because GraphQA sequences are short (~35 tokens
    # standard / ~150 incidence) — far below the lengths where flex pays off.
    impl: str = "v2-eager"
    flex_compile_mode: str = "max-autotune-no-cudagraphs"
    # fp32 is the dtype the v0<->v2 parity is proven at. bf16 halves memory and is
    # what the flex benchmarks measured, but it is a real numerical change, not a
    # free speedup — opt in deliberately.
    dtype: str = "fp32"

    # ── k-hop attention gate ───────────────────────────────────────────────────
    k_hop: int = 0                              # 0 disables the mask (the historical setting)
    k_hop_directed: bool = False                # GraphQA graphs are symmetric digraphs

    # ── graph-bias features ────────────────────────────────────────────────────
    # The ablation arms flip these; the dataset always carries all three (they are
    # not part of the cache key), so arms share one built dataset.
    spd: bool = True
    max_spd: int = 8                            # model-side distance-bucket clamp
    rrwp: bool = True
    max_rw_steps: int = 16                      # random-walk steps (data + model)
    magnetic: bool = True
    # Layer-sharing granularity for the magnetic bias (see GroupBiasCache in
    # src/models/bias.py). 0 = today's per-layer instance. G >= 1 replaces it with
    # G instances, layer l served by group l*G//num_layers, so G = num_layers is
    # per-layer and G = 1 is one instance for the whole stack. Purely a model-side
    # knob: it does not touch feature generation, the dataset cache key or the arm
    # label, so a G sweep trains on byte-identical data to the baseline.
    magnetic_groups: int = 0
    # Swap the magnetic bias's 2-layer MLP head for a single linear map onto the
    # heads, making the bias a bilinear form in the eigenvectors (see
    # src/models/LINEAR_BIAS.md). Same features, same data, different head — so it
    # is a model-side knob like magnetic_groups and shares the built dataset.
    magnetic_linear: bool = False
    # Decoupled magnetic heads (MagneticMagnitudeBias / MagneticHybridBias;
    # src/models/MIXED_BIAS.md). `magnetic_magnitude` is the per-node magnitude
    # channel alone, `magnetic_hybrid` adds the linear phase channel to it. Same
    # features and same data as `magnetic`, so both are model-side knobs that
    # share the built dataset, exactly like magnetic_linear.
    magnetic_magnitude: bool = False
    magnetic_hybrid: bool = False
    magnetic_magnitude_dim: int = 64            # d_magnitude — appended per head (NOT free)
    magnetic_magnitude_repr_dim: int = 256      # d_magnitude_repr — internal to the MLP (free)
    magnetic_dim: int = 32                      # model bias-MLP hidden width
    magnetic_q: float = 0.25                    # magnetic-Laplacian charge (data)
    magnetic_m: int = 0                         # eigenvectors kept (0 -> all N)
    # Collator-only cap on the eigenvector count, deliberately OUTSIDE the dataset
    # cache identity: it narrows what the collator hands the model, never the
    # build, so an M-sweep reuses one cache instead of rebuilding per value. 0 =
    # no override (fall back to magnetic_m).
    magnetic_m_collate: int = 0
    # Keep the intra-node diagonal b_ii instead of zeroing it (LINEAR_BIAS.md
    # §7.3). Model-side only, like magnetic_groups: same features, same data.
    # Incompatible with spd (the SPD lookup has no self-distance row).
    bias_self_node: bool = False
    laplacian: bool = False                     # unwired (see UNWIRED_FEATURES)
    rwse: bool = False                          # unwired

    # ── dataset ────────────────────────────────────────────────────────────────
    max_length: int = 1024                      # per-node tokenization cap
    # Fallback val size for the tasks with no official validation split; ignored for
    # every reported task.
    val_fraction: float = 0.15

    # ── LoRA (alpha is always 2*r, derived — never an independent knob) ────────
    lora: bool = True
    lora_r: int = 16
    lora_dropout: float = 0.05

    # ── training schedule ──────────────────────────────────────────────────────
    num_epochs: int = 20
    batch_size: int = 4
    accumulation_steps: int = 8
    lr: float = 3e-5                            # base / LoRA learning rate
    bias_lr: float = 5e-3                       # graph-bias learning rate
    eval_steps: int = 20                        # val eval + checkpoint cadence
    max_steps: int = -1                         # >0 caps optimizer steps (smoke tests)
    seed: int = 42
    num_workers: int = 0
    gradient_checkpointing: bool = False
    include_f1: bool = False                    # also log macro-F1 alongside exact match

    # ── data prep ──────────────────────────────────────────────────────────────
    use_gpu: bool = False                       # build SPD/RRWP/magnetic on GPU

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None                   # None -> no tracking

    # ── derived helpers ────────────────────────────────────────────────────────

    def torch_dtype(self):
        import torch
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.dtype]

    def backend(self):
        """The graph-attention backend ('eager' | 'flex') from ``impl``."""
        return self.impl.split("-", 1)[1]

    def backbone(self):
        """Which base-LLM family ``model_name`` names (a key of ``BACKBONES``)."""
        name = self.model_name.lower()
        for key, reason in UNWIRED_BACKBONES.items():
            if key in name:
                raise ValueError(f"model_name={self.model_name!r} is not wired: {reason}")
        for key in BACKBONES:
            if key in name:
                return key
        raise ValueError(
            f"model_name={self.model_name!r} names no backbone this experiment wires "
            f"(expected one of {tuple(BACKBONES)} in the name).")

    def gtlm_classes(self):
        """The ``(config_cls, model_cls)`` pair for this run's backbone."""
        from ... import models
        return tuple(getattr(models, n) for n in BACKBONES[self.backbone()]["classes"])

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout,
                "target_modules": list(BACKBONES[self.backbone()]["lora_targets"])}

    @property
    def uses_magnetic(self) -> bool:
        """True when ANY placement of the magnetic term is active.

        Single source of truth for every collator/data gate. Gating on
        ``self.magnetic`` alone would silently starve the linear, magnitude and
        hybrid arms of the eigenvectors they are built from, and the run would
        train with no magnetic bias while looking perfectly healthy.
        """
        return bool(self.magnetic or self.magnetic_linear
                    or self.magnetic_magnitude or self.magnetic_hybrid)

    @property
    def collate_magnetic_m(self) -> int:
        """Eigenvector count the COLLATOR emits (0 = keep all, as the cache stores).

        Note the asymmetry with ``magnetic_m``: both use 0 for "all", so an
        override of 0 means "no override" rather than "emit nothing".
        """
        if not self.uses_magnetic:
            return 0
        return self.magnetic_m_collate or self.magnetic_m

    def bias_params(self):
        """The graph-bias flag/architecture dict passed to the model config.

        Only enabled, wired features contribute keys, so the model builds modules
        exactly for the features an arm uses. ``magnetic_dim`` is the model's MLP
        width — distinct from ``magnetic_m`` (the eigenvector count), which is a
        data/collator knob and is NOT part of this dict. ``magnetic_q`` rides along
        as provenance (no bias module reads it — the charge is consumed when the
        eigenvectors are computed during data prep), so a saved checkpoint records
        which charge its features were built with.
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.magnetic:
            # magnetic_groups replaces the per-layer instance with G grouped ones;
            # same features, same data, different sharing granularity.
            share = ({"magnetic_groups": self.magnetic_groups} if self.magnetic_groups
                     else {"magnetic": True})
            cfg.update(**share, magnetic_dim=self.magnetic_dim,
                       magnetic_q=self.magnetic_q)
        elif self.magnetic_linear:
            # Same spectral features and the same magnetic_dim/magnetic_q; only the
            # head differs (one Linear onto the heads instead of the 2-layer MLP).
            cfg.update(magnetic_linear=True, magnetic_dim=self.magnetic_dim,
                       magnetic_q=self.magnetic_q)
        elif self.magnetic_magnitude or self.magnetic_hybrid:
            # Same features again, plus the two magnitude-channel widths. One
            # branch for both keys: they build the identical channel and differ
            # only in whether the phase channel rides alongside it.
            key = "magnetic_hybrid" if self.magnetic_hybrid else "magnetic_magnitude"
            cfg.update(**{key: True}, magnetic_dim=self.magnetic_dim,
                       magnetic_q=self.magnetic_q,
                       magnetic_magnitude_dim=self.magnetic_magnitude_dim,
                       magnetic_magnitude_repr_dim=self.magnetic_magnitude_repr_dim)
        if self.bias_self_node:
            # Model-side only; it does not touch feature generation or the cache key.
            cfg.update(bias_self_node=True)
        return cfg

    def arm(self):
        """Short ablation-arm label ('base' | 'no-spd' | ...) for records + tables."""
        # A '+selfnode' suffix, not a separate branch: bias_self_node modifies
        # whichever bias is active rather than replacing it, and appending keeps
        # every existing label byte-identical while it is off (the default).
        self_node = "+selfnode" if self.bias_self_node else ""
        off = [f for f in WIRED_FEATURES if not getattr(self, f)]
        head = ("mag-linear" if self.magnetic_linear else
                "mag-hybrid" if self.magnetic_hybrid else
                "mag-magnitude" if self.magnetic_magnitude else None)
        if head:
            # Without this the non-MLP heads report as 'no-magnetic' — the label of
            # the arm that has NO magnetic term at all, which is the one thing they
            # must never be confused with.
            off = [f for f in off if f != "magnetic"]
            tag = head if not off else head + "+no-" + "+".join(off)
            return tag + self_node
        if not off:
            return "base" + self_node
        return "no-" + "+".join(off) + self_node

    def uses_default_cache(self):
        """True when this config's feature settings match the built-on-disk cache."""
        return all(getattr(self, k) == v for k, v in _CACHE_DEFAULTS.items())

    def dataset_dir(self):
        """Directory holding this config's built splits.

        Feature-generation knobs are part of the cache identity: changing
        ``magnetic_q`` (say) must not silently reuse datasets built with the old
        value. Configs matching ``_CACHE_DEFAULTS`` resolve to the historical
        untagged path so the large existing cache (and ``graphqa_mag_khop``, which
        hardcodes that layout) keeps working; anything else gets a tagged sibling.
        """
        base = f"{DATASETS_DIR}/{self.graph_type}/{self.task}"
        if self.uses_default_cache():
            return base
        tags = [f"q{self.magnetic_q}", f"rw{self.max_rw_steps}", f"len{self.max_length}"]
        if self.magnetic_m:
            tags.append(f"m{self.magnetic_m}")
        if self.question_node != "off":
            tags.append(f"qn-{self.question_node}")
        if self.model_name != MODEL_NAME:
            tags.append(self.model_name.split("/")[-1])
        return f"{base}__{'_'.join(tags)}"

    def has_official_val(self):
        return self.task not in TASKS_WITHOUT_OFFICIAL_VAL

    def run_name(self):
        """Default (standalone) run name; the sweep runner overrides this."""
        lora_tag = f"_lora{self.lora_r}" if self.lora else ""
        khop_tag = f"_k{self.k_hop}" if self.k_hop else ""
        qn_tag = f"_qn-{self.question_node}" if self.question_node != "off" else ""
        return (f"{EXPERIMENT_NAME}_{self.arm()}_{self.task}_{self.graph_type}"
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
        if self.graph_type not in GRAPH_TYPES:
            raise ValueError(f"Unknown graph_type {self.graph_type!r} (expected one of {GRAPH_TYPES}).")
        if self.question_node not in QUESTION_NODES:
            raise ValueError(f"Unknown question_node {self.question_node!r} (expected one of {QUESTION_NODES}).")
        if self.impl not in IMPLS:
            raise ValueError(f"Unknown impl {self.impl!r} (expected one of {IMPLS}).")
        backbone = self.backbone()          # raises on an unwired model_name
        if self.impl not in BACKBONES[backbone]["impls"]:
            raise ValueError(
                f"impl {self.impl!r} is not available on the {backbone!r} backbone "
                f"(it wires {BACKBONES[backbone]['impls']}).")
        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r} (expected one of {DTYPES}).")

        enabled_unwired = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if enabled_unwired:
            raise ValueError(
                f"Bias feature(s) {enabled_unwired} are exposed in the config but not "
                f"wired in this experiment (data prep computes only {WIRED_FEATURES}). "
                f"Disable them or extend process_dataset.py to produce their features.")

        if self.magnetic and self.magnetic_linear:
            raise ValueError(
                "magnetic_linear swaps the HEAD on the magnetic term; it cannot be "
                "combined with --magnetic (that would ask for two magnetic biases). "
                "Use --no-magnetic --magnetic-linear for the linear arm.")
        if self.magnetic_linear and self.magnetic_groups:
            raise ValueError(
                "magnetic_linear and magnetic_groups are two different placements of "
                "the same term; enable at most one.")
        # Same rule for the decoupled heads. Note `magnetic_groups` is NOT a
        # competing head here — at this layer it MODIFIES `magnetic` (bias_params
        # emits magnetic_groups in place of magnetic), so `magnetic=True` +
        # `magnetic_groups=G` is the intended combination and must stay legal.
        placements = {"magnetic": self.magnetic, "magnetic_linear": self.magnetic_linear,
                      "magnetic_magnitude": self.magnetic_magnitude,
                      "magnetic_hybrid": self.magnetic_hybrid,
                      "magnetic_groups": bool(self.magnetic_groups)}
        for name in ("magnetic_magnitude", "magnetic_hybrid"):
            if not placements[name]:
                continue
            clash = [k for k, v in placements.items() if v and k != name]
            if clash:
                raise ValueError(
                    f"{name} is a different HEAD on the same magnetic term, so it "
                    f"cannot combine with {clash} — that would ask for two magnetic "
                    "biases on one feature set. Enable exactly one placement.")
        if self.magnetic_m_collate:
            if not self.uses_magnetic:
                raise ValueError(
                    "magnetic_m_collate is set but no magnetic term is enabled, so it "
                    "would silently do nothing.")
            if self.magnetic_m and self.magnetic_m_collate > self.magnetic_m:
                raise ValueError(
                    f"magnetic_m_collate={self.magnetic_m_collate} exceeds the built "
                    f"magnetic_m={self.magnetic_m}; the collator can only truncate, so "
                    f"this would silently fall back and mislabel the run.")
            if self.magnetic_m_collate < 0:
                raise ValueError("magnetic_m_collate must be >= 0 (0 = no override).")

        if self.bias_self_node:
            if self.spd:
                raise ValueError(
                    "bias_self_node does not cover SPDBias (its lookup has no row for "
                    "self-distance 0), so with spd=True the flag would apply to only "
                    "some of the active biases. Drop spd from this arm — the "
                    "factorization cannot express SPD anyway (LINEAR_BIAS.md §3).")
            if not (self.uses_magnetic or self.rrwp):
                raise ValueError(
                    "bias_self_node is set but no bias with a diagonal is enabled; it "
                    "would silently do nothing.")

        if self.lora_r < 0:
            raise ValueError("lora_r must be >= 0.")
        if self.k_hop < 0:
            raise ValueError("k_hop must be >= 0 (0 disables the mask).")
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1).")
        return self
