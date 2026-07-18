"""
Shared configuration for the TAG (text-attributed graph) benchmarks experiment.

One ``RunConfig`` dataclass holds EVERY knob the entry points read; ``__main__.py``
builds a ``RunConfig`` from argparse and stays a thin dispatcher, so a value lives in
exactly one place. This replaces the old layout, where the dataset choice lived in
``process_data.py``'s ``__main__``, the schedule in ``__main__.parse_args``, and the
bias architecture in a hard-coded ``BIAS_PARAMS`` dict.

The task is node classification on a citation or interaction graph: each node's k-hop
neighbourhood becomes one text-attributed graph, the target node's text carries the
question, and supervision is the answer span only, scored by exact match.

``text_mapping`` selects how much text each node contributes as a function of its
distance from the target (titles only, abstracts near the target, ...) — the axis the
experiment was built to study, and historically the reason for a dozen sibling dataset
directories.

This experiment runs the **v2** model only (``GTLMLlamaForCausalLM``). The legacy v0
path is pinned numerically equivalent to v2-eager at fp32 by
``tests/models/test_modeling_gtlm_llama_v2.py`` (loss + every bias gradient). Note the
historical runs here loaded v0 with ``attn_implementation="sdpa"``, which that test does
not cover — sdpa and eager compute the same function in a different accumulation order,
so expect statistical, not bit-level, agreement with pre-refactor numbers. See README.
"""

import os
from dataclasses import dataclass


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "tag_benchmarks"

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EXPERIMENT_DIR, "processed_data")
RAW_DIR = os.path.join(EXPERIMENT_DIR, "raw_data")

DATASETS = ("cora", "pubmed", "ogbn-arxiv", "reddit")
IMPLS = ("v2-flex", "v2-eager")
DTYPES = ("bf16", "fp32")
SPLITS = ("train", "val", "test")

# Node text is built from (title, abstract) for the citation graphs and from a single
# raw post for reddit, so the two families accept disjoint text mappings.
CITATION_DATASETS = ("cora", "pubmed", "ogbn-arxiv")
REDDIT_DATASETS = ("reddit",)

# Mappings over (title, abs): how much text a node contributes given its hop distance
# from the target. `random_abstracts` takes p (abstract kept with probability
# p**distance); the rest are parameterless.
CITATION_MAPPINGS = ("all_titles", "all_abstracts", "target_abstract",
                     "neighbor_abstracts", "random_abstracts")
# Mappings over reddit's raw post text. `truncated_text` takes a character cap.
REDDIT_MAPPINGS = ("full_text", "truncated_text", "more_target_text")
TEXT_MAPPINGS = CITATION_MAPPINGS + REDDIT_MAPPINGS

# The two mappings that take `text_mapping_param`; it must be None for every other.
PARAMETERIZED_MAPPINGS = ("random_abstracts", "truncated_text")

# Graph-bias features whose data AND model modules are both wired here.
WIRED_FEATURES = ("spd", "rrwp", "magnetic")
# Exposed in the model's bias registry but never produced by data prep. The old
# BIAS_PARAMS enabled both by default, which was silently a no-op: with no features on
# the dataset the collator emits (B, N, 0) tensors, and a cdist over a zero-width
# embedding is identically zero — so these biases contributed nothing but parameters.
# Rejected by validate() rather than silently ignored. (graphqa reached the same
# conclusion about the same two features.)
UNWIRED_FEATURES = ("laplacian", "rwse")

# Per-dataset classification instruction, appended to the target node's text. The
# answer follows ANSWER_PREFIX, which is also the label-mask boundary.
ANSWER_PREFIX = "A: "
INSTRUCTIONS = {
    "cora": (
        "Q: Given this paper citation graph, classify this paper into 7 classes: "
        "Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, "
        "Reinforcement_Learning, Rule_Learning, Theory. Please tell me which class "
        "does this paper belong to?\n" + ANSWER_PREFIX
    ),
    "ogbn-arxiv": (
        "Q: Given this paper citation graph, classify this paper into 40 classes: "
        "cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), "
        "cs.CY(Computers and Society), cs.CR(Cryptography and Security), "
        "cs.DC(Distributed, Parallel, and Cluster Computing), "
        "cs.HC(Human-Computer Interaction), "
        "cs.CE(Computational Engineering, Finance, and Science), "
        "cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), "
        "cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), "
        "cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), "
        "cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), "
        "cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), "
        "cs.ET(Emerging Technologies), cs.SY(Systems and Control), "
        "cs.CG(Computational Geometry), cs.OH(Other Computer Science), "
        "cs.PL(Programming Languages), cs.SE(Software Engineering), "
        "cs.LG(Machine Learning), cs.SD(Sound), "
        "cs.SI(Social and Information Networks), cs.RO(Robotics), "
        "cs.IT(Information Theory), cs.PF(Performance), "
        "cs.CL(Computational Complexity), cs.IR(Information Retrieval), "
        "cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), "
        "cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), "
        "cs.GT(Computer Science and Game Theory), cs.DB(Databases), "
        "cs.DL(Digital Libraries), cs.DM(Discrete Mathematics). Please tell me which "
        "class does this paper belong to?\n" + ANSWER_PREFIX
    ),
    "pubmed": (
        "Q: Given this paper citation graph, classify this paper into 3 classes: "
        "Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus "
        "Type2. Please tell me which class does this paper belong to?\n" + ANSWER_PREFIX
    ),
    "reddit": (
        "Q: Given this user reddit user post interaction graph, classify this reddit "
        "user into 2 classes: Normal Users and Popular Users. Please tell me which "
        "class does this reddit user belong to?\n" + ANSWER_PREFIX
    ),
}

# The feature-generation settings the on-disk dataset cache was built with. A config
# matching these resolves to the historical, untagged directory name (see dataset_dir).
#
# `samples_per_node` is deliberately NOT here, and is not part of the directory name
# either: the pre-refactor naming scheme never encoded it, and the existing caches were
# built with wildly different values (cora/60 = 16, cora/30 = 4, pubmed/60 = 2,
# ogbn-arxiv/60 = 1) that only their metadata.json records. Tagging on it would orphan
# every existing directory. Instead the value is cross-checked against the cache's
# metadata on load (see data.py `_cache_versions`), so the ambiguity fails loudly
# instead of silently handing back the wrong augmentation.
_CACHE_DEFAULTS = dict(
    model_name=MODEL_NAME, max_length=32_768, magnetic_q=0.25, magnetic_m=0,
    max_rw_steps=16, max_train_samples=None, max_val_samples=None,
)


@dataclass
class RunConfig:
    """Every knob the train / data_prep entry points read (built from argparse)."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"                         # "train" | "data_prep"

    # ── which dataset (these determine the cache directory name) ──────────────
    dataset: str = "cora"                       # see DATASETS
    hops: int = 2                               # neighbourhood radius sampled per node
    max_neighbors: int = 60                     # node cap per sampled neighbourhood
    text_mapping: str = "target_abstract"       # see TEXT_MAPPINGS
    # p for `random_abstracts`, character cap for `truncated_text`; None otherwise.
    text_mapping_param: float = None

    # ── model ──────────────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    # v2-flex is the default because TAG sequences are long (529-2757 avg tokens; see
    # processed_data/README.md) — flex captures the small (B,H,N,N) node bias and
    # gathers it inside the kernel instead of materializing the (B,H,L,L) token-level
    # bias, and it is bias-agnostic, so rrwp rides along for free. This matches kgqa,
    # the other long-sequence experiment. v2-eager is the parity/debug path (it is the
    # implementation pinned equivalent to v0), not the fast one.
    impl: str = "v2-flex"
    flex_compile_mode: str = "max-autotune-no-cudagraphs"
    # bf16 matches kgqa and probes, and is what the flex benchmarks measured. graphqa
    # defaults to fp32 instead because that is where v0<->v2 parity is proven and its
    # sequences are far too short for the cost to matter; here it matters.
    dtype: str = "bf16"

    # ── k-hop attention gate ───────────────────────────────────────────────────
    k_hop: int = 0                              # 0 disables the mask (the historical setting)
    k_hop_directed: bool = False                # the sampled neighbourhoods are undirected

    # ── graph-bias features ────────────────────────────────────────────────────
    # Data prep always computes all three, so ablation arms share one built dataset
    # (these are not part of the cache key).
    spd: bool = True
    max_spd: int = 8                            # model-side distance-bucket clamp
    rrwp: bool = True
    max_rw_steps: int = 16                      # random-walk steps (data + model)
    magnetic: bool = True
    magnetic_dim: int = 32                      # model bias-MLP hidden width
    magnetic_q: float = 0.25                    # magnetic-Laplacian charge (data)
    magnetic_m: int = 0                         # eigenvectors kept (0 -> all N)
    laplacian: bool = False                     # unwired (see UNWIRED_FEATURES)
    rwse: bool = False                          # unwired

    # ── dataset construction (data prep) ──────────────────────────────────────
    max_length: int = 32_768                    # per-node tokenization cap
    # Neighbourhood re-samples per train node (stored as the dataset's
    # `per_graph_versions`). NOT part of the cache directory name — see _CACHE_DEFAULTS.
    # None means "adopt whatever the built cache holds", which is the only sane default
    # given the existing caches disagree (16 / 4 / 2 / 1); data_prep requires it to be
    # explicit, and train cross-checks it when set.
    samples_per_node: int = None
    max_train_samples: int = None               # subsample train nodes (None = all)
    max_val_samples: int = None                 # subsample val nodes (None = all)
    data_seed: int = 42                         # neighbourhood-sampling RNG (data prep only)
    use_gpu: bool = True                        # build SPD/RRWP/magnetic on GPU

    # ── evaluation split sizing (train time) ──────────────────────────────────
    # The val split is strided down to at most this many graphs for the in-training
    # evals that drive checkpoint selection — full val is 542-29,799 graphs and is
    # scored every eval_steps. Preserved exactly as the pre-refactor loader did it
    # (contiguous blocks of 50, evenly strided across the split) so selection behavior
    # is unchanged. None = use the whole split.
    val_subsample: int = 500
    # Reddit's test split is 26,811 graphs; capping it trades reported-metric variance
    # for wall clock. None = score the whole split (the historical behavior).
    test_subsample: int = None

    # ── LoRA (alpha is always 2*r, derived — never an independent knob) ────────
    lora: bool = True
    lora_r: int = 32
    lora_dropout: float = 0.05

    # ── training schedule ──────────────────────────────────────────────────────
    num_epochs: int = 10
    batch_size: int = 1
    accumulation_steps: int = 32
    lr: float = 3e-4                            # base / LoRA learning rate
    bias_lr: float = 5e-2                       # graph-bias learning rate
    eval_steps: int = 40                        # val eval + checkpoint cadence
    max_steps: int = -1                         # >0 caps optimizer steps (smoke tests)
    seed: int = 42
    num_workers: int = 0
    gradient_checkpointing: bool = True
    include_f1: bool = True                     # macro-F1 alongside exact match

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None                   # None -> no tracking

    # ── derived helpers ────────────────────────────────────────────────────────

    def torch_dtype(self):
        import torch
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.dtype]

    def backend(self):
        """The graph-attention backend ('eager' | 'flex') from ``impl``."""
        return self.impl.split("-", 1)[1]

    def is_reddit(self):
        return self.dataset in REDDIT_DATASETS

    def allowed_mappings(self):
        """The text mappings this dataset's raw node attributes can support."""
        return REDDIT_MAPPINGS if self.is_reddit() else CITATION_MAPPINGS

    def instruction(self):
        return INSTRUCTIONS[self.dataset]

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout}

    def bias_params(self):
        """The graph-bias flag/architecture dict passed to the model config.

        Only enabled, wired features contribute keys, so the model builds modules
        exactly for the features an arm uses.
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim)
        return cfg

    def arm(self):
        """Short ablation-arm label ('base' | 'no-spd' | ...) for records + tables."""
        off = [f for f in WIRED_FEATURES if not getattr(self, f)]
        if not off:
            return "base"
        return "no-" + "+".join(off)

    def mapping_name(self):
        """The text mapping's directory-name form, e.g. 'random_abstracts_p0.2'.

        Reproduces the pre-refactor ``get_mapping_name`` strings exactly — these are
        the names the existing cache directories carry (see ``dataset_dir``).
        """
        if self.text_mapping == "random_abstracts":
            return f"random_abstracts_p{self.text_mapping_param}"
        if self.text_mapping == "truncated_text":
            return f"truncated_text_{int(self.text_mapping_param)}"
        return self.text_mapping

    def uses_default_cache(self):
        """True when this config's feature settings match the built-on-disk cache."""
        return all(getattr(self, k) == v for k, v in _CACHE_DEFAULTS.items())

    def dataset_dir(self):
        """Directory holding this config's built splits.

        Feature-generation knobs are part of the cache identity: changing ``magnetic_q``
        (say) must not silently reuse datasets built with the old value. Configs
        matching ``_CACHE_DEFAULTS`` resolve to the historical untagged path, so the
        large existing cache (up to 90,941 ogbn-arxiv train subgraphs) keeps working;
        anything else gets a tagged sibling.
        """
        base = os.path.join(
            DATASETS_DIR,
            f"{self.dataset}_hops{self.hops}_neighbors{self.max_neighbors}_{self.mapping_name()}")
        if self.uses_default_cache():
            return base
        tags = [f"q{self.magnetic_q}", f"rw{self.max_rw_steps}", f"len{self.max_length}"]
        if self.magnetic_m:
            tags.append(f"m{self.magnetic_m}")
        if self.max_train_samples is not None:
            tags.append(f"tr{self.max_train_samples}")
        if self.max_val_samples is not None:
            tags.append(f"va{self.max_val_samples}")
        if self.model_name != MODEL_NAME:
            tags.append(self.model_name.split("/")[-1])
        return f"{base}__{'_'.join(tags)}"

    def run_name(self):
        """Default (standalone) run name; the sweep runner overrides this."""
        lora_tag = f"_lora{self.lora_r}" if self.lora else ""
        khop_tag = f"_k{self.k_hop}" if self.k_hop else ""
        return (f"{EXPERIMENT_NAME}_{self.arm()}_{self.dataset}_n{self.max_neighbors}"
                f"_{self.mapping_name()}{lora_tag}{khop_tag}_s{self.seed}")

    def validate(self):
        """Reject settings this experiment does not support, with a clear message.

        The sweep runner is experiment-agnostic and passes any key through; this is
        where the experiment draws its own line (fail fast, before any GPU work).
        """
        if self.mode not in ("train", "data_prep"):
            raise ValueError(f"Unknown mode {self.mode!r} (expected 'train' or 'data_prep').")
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {self.dataset!r} (expected one of {DATASETS}).")
        if self.impl not in IMPLS:
            raise ValueError(f"Unknown impl {self.impl!r} (expected one of {IMPLS}).")
        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r} (expected one of {DTYPES}).")

        if self.text_mapping not in TEXT_MAPPINGS:
            raise ValueError(
                f"Unknown text_mapping {self.text_mapping!r} (expected one of {TEXT_MAPPINGS}).")
        if self.text_mapping not in self.allowed_mappings():
            family = "reddit's raw post text" if self.is_reddit() else "(title, abstract) pairs"
            raise ValueError(
                f"text_mapping {self.text_mapping!r} is not available for dataset "
                f"{self.dataset!r}, whose nodes carry {family}; pick one of "
                f"{self.allowed_mappings()}.")
        if self.text_mapping in PARAMETERIZED_MAPPINGS and self.text_mapping_param is None:
            need = ("a keep-probability p in (0, 1]" if self.text_mapping == "random_abstracts"
                    else "a character cap > 0")
            raise ValueError(
                f"text_mapping {self.text_mapping!r} requires --text-mapping-param ({need}).")
        if self.text_mapping not in PARAMETERIZED_MAPPINGS and self.text_mapping_param is not None:
            raise ValueError(
                f"text_mapping {self.text_mapping!r} takes no parameter, but "
                f"text_mapping_param={self.text_mapping_param} was given.")
        if self.text_mapping == "random_abstracts" and not (0 < self.text_mapping_param <= 1):
            raise ValueError(
                f"random_abstracts takes a probability in (0, 1]; got {self.text_mapping_param}.")
        if self.text_mapping == "truncated_text" and self.text_mapping_param <= 0:
            raise ValueError(
                f"truncated_text takes a character cap > 0; got {self.text_mapping_param}.")

        enabled_unwired = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if enabled_unwired:
            raise ValueError(
                f"Bias feature(s) {enabled_unwired} are exposed in the config but not wired "
                f"in this experiment (data prep computes only {WIRED_FEATURES}). They were "
                f"enabled by default before the refactor, but with no features on the "
                f"dataset their bias is identically zero — a silent no-op. Disable them, or "
                f"extend data.py to call compute_laplacian_coordinates / compute_rwse.")

        if self.hops < 1:
            raise ValueError(f"hops must be >= 1; got {self.hops}.")
        if self.max_neighbors < 2:
            raise ValueError(f"max_neighbors must be >= 2; got {self.max_neighbors}.")
        if self.samples_per_node is not None and self.samples_per_node < 1:
            raise ValueError(f"samples_per_node must be >= 1 or None; got {self.samples_per_node}.")
        if self.mode == "data_prep" and self.samples_per_node is None:
            raise ValueError(
                "data_prep requires an explicit --samples-per-node: it is baked into the "
                "built dataset but not into its directory name, so there is no value to "
                "infer. (The existing caches used 16 for cora/60, 4 for cora/30 and "
                "reddit/15, 2 for pubmed/60, 1 for ogbn-arxiv/60.)")
        if self.lora_r < 0:
            raise ValueError("lora_r must be >= 0.")
        if self.k_hop < 0:
            raise ValueError("k_hop must be >= 0 (0 disables the mask).")
        for name in ("val_subsample", "test_subsample", "max_train_samples", "max_val_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None (no cap); got {value}.")
        return self
