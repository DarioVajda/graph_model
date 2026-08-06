"""
Single source of truth for the context-exhaustion experiment (Needle in a Graph).

One ``RunConfig`` carries every knob all three entry points read (``data_prep``,
``train``, ``grid``); ``__main__.py`` builds it from argparse and stays a thin
dispatcher. Field groups mirror the flag groups in ``__main__.build_parser``.

Two axes define the experiment (see README.md):
  * ``node_counts`` (N) — TOTAL nodes per graph, including the QUESTION node and
    the PROMPT node, so a graph has ``N - 2`` content nodes of which exactly one
    is the gold node (``N - 3`` distractors).
  * ``token_counts`` (T) — tokens per content node, exact (see ``data.py``).

The packed sequence length of a cell is ``(N - 2) * T + |QUESTION| + |PROMPT|``,
which ``cell_length()`` computes with a fixed allowance for the two short nodes.

Data-prep keys determine the ``.gtds`` cache directory (``data_config_key``);
train keys never do, so a seed sweep reuses one built dataset.
"""

from dataclasses import dataclass

from ...models.flex_kernel import align_len


MODEL_NAME = "meta-llama/Llama-3.2-1B"
EXPERIMENT_NAME = "context"

# Data-format version, appended to the cache key: bump when the builder's
# SEMANTICS change without any config knob changing, so stale caches can't be
# silently reused. v1 = initial build (uniform-per-T needle offset, nested node
# subsets, exact token counts, fixed-token-length codes). v2 = fixed-CHARACTER-
# length codes from an interior-digit template (data.CODE_TEMPLATE): v1 filtered
# only on token length, so a 4-char gold code could be a substring of a 5-char
# distractor (ambiguous item), and 1% of codes were all-digit and collided with
# years in the wikitext filler.
DATA_FORMAT_VERSION = 2

# Filler corpus. Parquet-backed; the bare "wikitext" id redirects here.
CORPUS_REPO = "Salesforce/wikitext"
CORPUS_CONFIG = "wikitext-103-raw-v1"

# Reserved token allowance for the two non-content nodes (QUESTION + PROMPT).
# Used only to predict a cell's packed length for the flex bucket ladder; the
# real length is whatever the tokenizer produces and is always <= this bound.
NON_CONTENT_TOKENS = 64

# Graph-bias features whose data + model modules are both wired here.
WIRED_FEATURES = ("spd", "rrwp", "magnetic")

# flat_grid = the zero-shot flat-text arm (README §3.1): same 25 test splits, serialized
# to one ordinary sequence and scored with the PRETRAINED backbone (no checkpoint).
MODES = ("data_prep", "data_merge", "train", "grid", "flat_grid", "flat_train")
GRAPH_ATTN_IMPLS = ("flex", "eager")
DTYPES = ("bf16", "fp32")

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class RunConfig:
    """Every knob the three entry points read (built from argparse in ``__main__``)."""

    # ── what to run ────────────────────────────────────────────────────────────
    mode: str = "train"

    # ── model + LoRA ───────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128                       # convention: 2 * lora_r
    lora_dropout: float = 0.15
    active_params: tuple = ("graph_bias",)

    # ── graph-bias features ────────────────────────────────────────────────────
    # RRWP is off by default: (N, N, K) fp32 is a 5x storage multiplier and on a
    # star its walk profile is near-degenerate (README §A.5).
    spd: bool = True
    max_spd: int = 8                            # nothing on a star exceeds 2
    rrwp: bool = False
    max_rw_steps: int = 16
    magnetic: bool = True
    # Layer-sharing granularity for the magnetic bias (see GroupBiasCache in
    # src/models/bias.py). 0 = today's per-layer instance. G >= 1 replaces it with
    # G instances, layer l served by group l*G//num_layers, so G = num_layers is
    # per-layer and G = 1 is one instance for the whole stack. Purely a model-side
    # knob: it does not enter data_config_key(), so a G sweep reuses one build.
    magnetic_groups: int = 0
    # Linear-head magnetic bias (LinearMagneticBias / src/models/LINEAR_BIAS.md).
    # Consumes exactly the same eigenvector DATA as `magnetic`, so it must NOT
    # change data_config_key() — the arm reuses the existing build — and every
    # gate that keys on `magnetic` must accept it too (see uses_magnetic).
    magnetic_linear: bool = False
    magnetic_dim: int = 128
    magnetic_q: float = 0.25
    magnetic_m: int = 128                       # eigenvectors kept (0 -> all)
    # Collator-only eigenvector truncation for the M-sweep (LINEAR_BIAS.md §2.6).
    # 0 = follow magnetic_m. Deliberately NOT in data_config_key(): eigenvalues
    # come back from `eigh` in ascending order and both the builder and the
    # collator truncate by prefix slice, so slicing a stored-m dataset to M here
    # is bit-identical to having built it at M — and the whole M-grid therefore
    # runs off ONE build instead of one build per M.
    magnetic_m_collate: int = 0
    # Keep the intra-node diagonal b_ii instead of zeroing it (LINEAR_BIAS.md
    # §7.3). Model-side only: it changes what the bias emits, not what the dataset
    # stores, so it must NOT enter data_config_key() — the arm reuses the existing
    # build. Incompatible with spd (the SPD lookup has no self-distance row).
    bias_self_node: bool = False

    # ── attention ──────────────────────────────────────────────────────────────
    # k_hop=0 keeps the prefix dense, which is what the dilution claim is about;
    # k_hop=1 would make a star sparse but changes the measurement (README §A.1).
    k_hop: int = 0
    k_hop_directed: bool = False
    graph_attn_impl: str = "flex"
    dtype: str = "bf16"
    compile_mode: str = "max-autotune-no-cudagraphs"

    # ── the grid ───────────────────────────────────────────────────────────────
    node_counts: tuple = (8, 16, 32, 64, 128)   # N (total nodes, incl. QUESTION + PROMPT)
    token_counts: tuple = (32, 64, 128, 256, 512)  # T (tokens per content node)

    # ── dataset construction ───────────────────────────────────────────────────
    # Training only sees cells whose packed length fits the cap; the grid is
    # evaluated in full, so the over-cap cells are length extrapolation (README §A.6).
    max_train_len: int = 16384
    n_train: int = 4000
    n_dev: int = 200
    n_test: int = 200                           # per grid cell
    # 0 = the original lookup task (QUESTION names the answer node outright).
    # >0 = k-hop pointer chasing (README §A.1): the QUESTION names a START
    # node and the answer is the code `hops` pointer-steps away, so it cannot be
    # reached by matching a name. hops=0 keeps the v2 build byte-identical.
    hops: int = 0
    # Out-edges per content node when hops > 0: 1 real "Continue at" pointer plus
    # (fan_out - 1) explicitly-labelled decoy references. fan_out=1 is the original
    # construction and keeps those builds byte-identical.
    #
    # This exists because at fan_out=1 the content subgraph is FUNCTIONAL (every
    # node has exactly one successor), so the answer is the unique node at
    # SPD == hops from the start and the graph arm can read it off the distance
    # bias without traversing anything — measured, 100% of graphs. Raising fan_out
    # puts ~fan_out**hops nodes at that distance, so structure prunes the candidate
    # set but cannot identify the answer; only the text says which reference is
    # real. Both edge kinds enter the DiGraph identically and every content node
    # has the SAME out-degree, so neither edge type nor degree leaks the chain.
    fan_out: int = 1
    # k as a MIXTURE axis (the main sweep). Empty = use the scalar ``hops`` for every
    # graph, which is what every build so far did and what keeps their bytes and cache
    # keys unchanged. Non-empty = k is drawn per graph (``data.sample_hops``) exactly as
    # (N, T) is drawn by ``sample_cell``, and the QUESTION node states the drawn k — so
    # the model must read the task off the input rather than assume a constant.
    #
    # The gold chain is ``slot_order[:k+1]``, a PREFIX, so one blueprint yields nested
    # chains across k: same start node, deeper answer. Test splits built per (N, T, k)
    # from shared blueprint ids are therefore paired along k as well as along the cell.
    hop_counts: tuple = ()
    code_len: int = 3                           # tokens per access code (exact)
    id_pool: int = 4096                         # distinct node-id strings to draw from
    corpus_tokens: int = 20_000_000             # filler tokens to cache from wikitext
    # ── build sharding (strategy, NOT semantics — deliberately out of the cache key) ──
    # The train split is materialised whole in RAM during `datasets.map`, and the peak
    # scales with graphs x tokens: the N=64 cell hit 61.7 GB at n_train=2,000 and a 96 G
    # request OOM-killed the 8k builds. Splitting the blueprint range across jobs bounds
    # each job's peak and builds them in parallel; `--mode data_merge` concatenates the
    # shards with `TextGraphDataset.__add__`. Shard i covers a CONTIGUOUS blueprint
    # range, so `id_offset` is exact and no two shards draw the same graph.
    train_shards: int = 1
    train_shard: int = -1                       # -1 = not building a shard
    # Restrict the TEST-grid loop to these cells, as "NxT" comma-joined ("128x512,64x256").
    # Empty = every cell. Same role as train_shard for the other half of the build: the
    # test grid is 16 cells x 4 k = 64 splits spanning 960..64,576 tokens, and building
    # them in one job projects to ~33 h -- past any sensible walltime -- while building
    # them per cell is embarrassingly parallel and bounded by the largest cell (~7 h).
    # NOT in the cache key: it changes which splits a job builds, never their bytes.
    only_cells: str = ""
    # Restrict the TEST-grid loop to these k values, comma-joined ("1,3"). Empty = every
    # k in `hops_list()`. The cell axis alone cannot shard the grid finely enough: cost
    # goes as ~L^1.7, so the single 128x512 cell is ~half the whole grid and pins any
    # cell-sharded job to that floor. k splits it further at no cost, because all four k
    # values of a cell have identical packed length.
    # NOT in the cache key (same rule as only_cells) -- `hop_counts` names which splits a
    # build produced and IS in the key; this only picks among them at scoring time.
    only_hops: str = ""
    data_seed: int = 42                         # graph-construction RNG (in the cache key)
    data_format_version: int = DATA_FORMAT_VERSION

    # ── training schedule ──────────────────────────────────────────────────────
    num_epochs: int = 3
    batch_size: int = 1
    accumulation_steps: int = 8
    lr: float = 1e-4
    bias_lr: float = 5e-3
    bias_weight_decay: float = 0.0
    eval_steps: int = 250
    max_steps: int = -1
    seed: int = 42                              # training seed (decoupled from data_seed)
    num_workers: int = 4
    gradient_checkpointing: bool = True

    # ── grid mode ──────────────────────────────────────────────────────────────
    checkpoint_path: str = None                 # required by --mode grid

    # ── tracking ───────────────────────────────────────────────────────────────
    wandb_project: str = None

    # ── derived: the grid ──────────────────────────────────────────────────────

    def cell_length(self, n, t):
        """Upper bound on the packed sequence length of cell ``(n, t)``."""
        return (n - 2) * t + NON_CONTENT_TOKENS

    def cells(self):
        """Every (N, T) cell of the grid, row-major."""
        return [(n, t) for n in self.node_counts for t in self.token_counts]

    def selected_cells(self):
        """The cells the test-grid loop should build — every cell unless filtered."""
        if not self.only_cells:
            return self.cells()
        want = set()
        for spec in filter(None, (c.strip() for c in self.only_cells.split(","))):
            n, _sep, t = spec.partition("x")
            want.add((int(n), int(t)))
        return [c for c in self.cells() if c in want]

    def hops_list(self):
        """The k values this config builds: the mixture if set, else the scalar."""
        return tuple(self.hop_counts) if self.hop_counts else (self.hops,)

    def selected_hops(self):
        """The k values the grid loop should score — every built k unless filtered."""
        if not self.only_hops:
            return self.hops_list()
        want = {int(k.strip()) for k in self.only_hops.split(",") if k.strip()}
        return tuple(k for k in self.hops_list() if k in want)

    def needle_tokens(self):
        """Upper bound on a content node's needle, in tokens.

        Measured on Llama-3.2-1B at ``code_len=3``: the KV sentence is 13 tokens, the
        real pointer 7, each decoy 9 — so fan_out = 1/2/3 gives 20/29/38, which this
        reproduces. Scaled by ``code_len`` so a longer code is accounted for.
        """
        return 17 + self.code_len + 9 * (self.fan_out - 1)

    def train_cells(self):
        """The cells that fit ``max_train_len`` — the training distribution."""
        return [(n, t) for (n, t) in self.cells() if self.cell_length(n, t) <= self.max_train_len]

    def len_buckets(self):
        """Explicit flex length ladder: one 128-aligned entry per trainable cell.

        Batching is cell-homogeneous (see ``train.CellGroupedSampler``), so every
        batch's raw length already sits on a cell length and this ladder adds no
        padding beyond block alignment — while bounding the number of distinct
        compiled shapes to the number of cells (README §A.7). The collator
        REJECTS a bucket that is not a multiple of the block size, hence
        ``align_len``.
        """
        lens = {align_len(self.cell_length(n, t), 128) for (n, t) in self.train_cells()}
        return sorted(lens)

    def grid_len_buckets(self):
        """As ``len_buckets`` but over the FULL grid (used by ``grid`` mode)."""
        lens = {align_len(self.cell_length(n, t), 128) for (n, t) in self.cells()}
        return sorted(lens)

    def node_buckets(self):
        """Explicit flex node ladder: the grid's node counts, floored at 32.

        N drives the ``(B, H, N, N)`` bias shape the compiled kernel guards on.
        """
        return sorted({max(32, n) for n in self.node_counts})

    def n_content_max(self):
        """Content nodes in the largest cell — the size of a graph's blueprint."""
        return max(self.node_counts) - 2

    # ── derived: model wiring ──────────────────────────────────────────────────

    @property
    def torch_dtype(self):
        import torch
        return {"bf16": torch.bfloat16, "fp32": torch.float32}[self.dtype]

    def lora_config(self):
        """LoRA config dict for ``select_active_params`` (``None`` when disabled)."""
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout, "target_modules": LORA_TARGET_MODULES}

    def bias_params(self):
        """The graph-bias flag/architecture dict passed to the model config.

        Only enabled features contribute keys, so the model builds modules exactly
        for the features the dataset carries.
        """
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.uses_magnetic:
            # magnetic_groups replaces the per-layer instance with G grouped ones;
            # same features, same data, different sharing granularity.
            if self.magnetic_linear:
                share = {"magnetic_linear": True}
            elif self.magnetic_groups:
                share = {"magnetic_groups": self.magnetic_groups}
            else:
                share = {"magnetic": True}
            cfg.update(**share, magnetic_dim=self.magnetic_dim, magnetic_q=self.magnetic_q)
        if self.bias_self_node:
            # Model-side only; deliberately absent from data_config_key().
            cfg.update(bias_self_node=True)
        return cfg

    def data_config_key_candidates(self):
        """This config's cache key, then any SUPERSET build that can serve it.

        A built dataset is keyed by which feature COLUMNS it contains. A training
        run that switches a feature off does not need those columns removed — it
        simply never instantiates a module for them (``GraphAttentionBias`` builds
        only enabled types, and an unconsumed column in the batch is inert). So a
        `spd=False` run can read a `spd=True` build, and only the *build* path
        needs the exact key.

        Without this, the Phase 2 arms that drop SPD (LINEAR_BIAS.md §3: SPD has
        no f(i)·g(j) form and so cannot survive the factorization) would each
        demand a fresh multi-hour rebuild of a strict SUBSET of data that already
        exists on disk.

        Ordered most- to least-specific; the loader takes the first that exists.
        """
        seen, out = set(), []
        for spd in ({self.spd, True} if not self.spd else {True}):
            for rrwp in ({self.rrwp, True} if not self.rrwp else {True}):
                for mag in ({self.uses_magnetic, True} if not self.uses_magnetic
                            else {True}):
                    key = self._data_config_key(spd=spd, rrwp=rrwp, magnetic=mag)
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
        # Exact key first, whatever the iteration order above produced.
        exact = self.data_config_key()
        out = [exact] + [k for k in out if k != exact]
        return out

    @property
    def uses_magnetic(self) -> bool:
        """True when the run consumes magnetic eigenvector features.

        Single source of truth for every dataset/collator/model gate. `magnetic`
        and `magnetic_linear` differ only in the HEAD applied to the same
        features, so a gate that checks `magnetic` alone would emit no
        eigenvectors for the linear arm — and the bias would silently return None,
        producing a bias-free run that trains fine and reads as a clean negative.
        """
        return bool(self.magnetic or self.magnetic_linear)

    @property
    def collate_magnetic_m(self) -> int:
        """Eigenvector count the COLLATOR should truncate to (0 = no truncation).

        Distinct from `magnetic_m`, which is a data-build knob in the cache key.
        """
        if not self.uses_magnetic:
            return 0
        return self.magnetic_m_collate or self.magnetic_m

    # ── derived: cache identity ────────────────────────────────────────────────

    def data_config_key(self):
        """Directory name for this config's built `.gtds` tree.

        Everything that changes the BYTES of a built dataset is in here; nothing
        else is (so a training-seed sweep reuses one build).
        """
        return self._data_config_key(spd=self.spd, rrwp=self.rrwp,
                                     magnetic=self.uses_magnetic)

    def _data_config_key(self, *, spd, rrwp, magnetic):
        """``data_config_key`` with the feature flags supplied explicitly.

        Split out so ``data_config_key_candidates`` can name a SUPERSET build
        without mutating the config. Every other component is read from ``self``,
        so the two can never drift.
        """
        grid = f"n{'-'.join(map(str, self.node_counts))}_t{'-'.join(map(str, self.token_counts))}"
        # `uses_magnetic`, not `magnetic`: the linear arm consumes the identical
        # eigenvector bytes, so it must map to the SAME build. Keying on
        # `magnetic` alone would silently fork a second (identical) dataset.
        feats = f"spd{int(spd)}rrwp{int(rrwp)}mag{int(magnetic)}"
        model_tag = self.model_name.split("/")[-1]
        # Appended ONLY when hops > 0, so the already-built lookup datasets keep
        # their existing keys rather than being orphaned by a new field.
        # A mixture supersedes the scalar and gets its own tag, so a k-mixture build
        # can never collide with the single-k builds it is composed of.
        hop_tag = (f"_hm{'-'.join(map(str, self.hop_counts))}" if self.hop_counts
                   else (f"_h{self.hops}" if self.hops else ""))
        # Same rule as hop_tag: absent at the default, so every fan_out=1 build
        # (all of Phase A and the learnability controls) keeps its existing key.
        # Gated on hops as well because fan_out only reaches `realize_chain` — at
        # hops=0 it changes no bytes, so it must not fork the lookup caches.
        fan_tag = (f"_fan{self.fan_out}"
                   if ((self.hops or self.hop_counts) and self.fan_out > 1) else "")
        return (f"{model_tag}{hop_tag}{fan_tag}_{grid}_cap{self.max_train_len}_tr{self.n_train}"
                f"_dev{self.n_dev}_te{self.n_test}_code{self.code_len}_ids{self.id_pool}"
                f"_{feats}_m{self.magnetic_m}_q{self.magnetic_q}_spdcut{self.max_spd}"
                f"_rw{self.max_rw_steps}_seed{self.data_seed}_v{self.data_format_version}")

    def resolved_data_root(self, output_root):
        """Absolute path of the build this config should READ.

        The exact key if it exists, else the first superset build that does (see
        ``data_config_key_candidates``). Raises rather than silently returning a
        non-existent path, so a missing build fails at startup instead of
        mid-training.
        """
        import os
        for key in self.data_config_key_candidates():
            path = os.path.join(output_root, key)
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            "No built dataset for this config. Tried (most specific first):\n  "
            + "\n  ".join(self.data_config_key_candidates())
            + f"\nunder {output_root}. Build it with --mode data_prep.")

    def run_name(self):
        """Default (standalone) run name; the sweep runner overrides this."""
        lora_tag = f"_lora{self.lora_r}" if self.lora else ""
        return f"{EXPERIMENT_NAME}_cap{self.max_train_len}{lora_tag}_s{self.seed}"

    # ── validation ─────────────────────────────────────────────────────────────

    def validate(self):
        """Reject settings this experiment does not support, before any GPU work."""
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r} (expected one of {MODES}).")
        if self.graph_attn_impl not in GRAPH_ATTN_IMPLS:
            raise ValueError(f"graph_attn_impl must be one of {GRAPH_ATTN_IMPLS}.")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}.")
        if min(self.node_counts) < 4:
            raise ValueError("Every N must be >= 4 (QUESTION + PROMPT + gold + >=1 distractor).")
        if not self.train_cells():
            raise ValueError(
                f"max_train_len={self.max_train_len} admits no cell of the grid "
                f"(smallest cell is {self.cell_length(min(self.node_counts), min(self.token_counts))} "
                "tokens). Raise the cap or shrink the grid.")
        for spec in filter(None, (c.strip() for c in self.only_cells.split(","))):
            n, sep, t = spec.partition("x")
            if not (sep and n.isdigit() and t.isdigit()):
                raise ValueError(
                    f"only_cells entry {spec!r} is not of the form NxT (e.g. 128x512).")
            if (int(n), int(t)) not in self.cells():
                raise ValueError(
                    f"only_cells names cell {spec}, which is not in this grid "
                    f"(node_counts={self.node_counts}, token_counts={self.token_counts}).")
        for spec in filter(None, (k.strip() for k in self.only_hops.split(","))):
            if not spec.isdigit():
                raise ValueError(f"only_hops entry {spec!r} is not an integer k.")
            if int(spec) not in self.hops_list():
                raise ValueError(
                    f"only_hops names k={spec}, which this config does not build "
                    f"(hops_list={self.hops_list()}).")
        if self.train_shards < 1:
            raise ValueError("train_shards must be >= 1.")
        if self.train_shard >= self.train_shards:
            raise ValueError(
                f"train_shard={self.train_shard} is out of range for "
                f"train_shards={self.train_shards} (valid: 0..{self.train_shards - 1}).")
        if self.mode == "grid" and not self.checkpoint_path:
            raise ValueError("--mode grid requires --checkpoint-path.")
        if self.code_len < 1:
            raise ValueError("code_len must be >= 1.")
        if self.hops < 0:
            raise ValueError("hops must be >= 0.")
        if any(k < 0 for k in self.hop_counts):
            raise ValueError(f"every hop_counts entry must be >= 0, got {self.hop_counts}.")
        if self.hop_counts and self.hops:
            raise ValueError(
                f"Set hops OR hop_counts, not both (got hops={self.hops}, "
                f"hop_counts={self.hop_counts}). The mixture would silently win and the "
                "scalar would read as documentation of something that never happened.")
        if max(self.hops_list()):
            # The chain occupies hops+1 slots taken from the front of slot_order,
            # so the SMALLEST cell has to hold all of them or that cell's graph
            # cannot be built at all. With a mixture, the DEEPEST k binds.
            smallest = min(self.node_counts) - 2
            deepest = max(self.hops_list())
            if smallest < deepest + 1:
                raise ValueError(
                    f"hops={deepest} needs {deepest + 1} chain nodes, but the smallest "
                    f"cell (N={min(self.node_counts)}) has only {smallest} content nodes. "
                    "Raise min(node_counts) or lower hops.")
            # Each node needs fan_out DISTINCT targets other than itself, so the
            # smallest cell bounds fan_out too.
            if self.fan_out > smallest - 1:
                raise ValueError(
                    f"fan_out={self.fan_out} needs {self.fan_out} distinct targets per node, "
                    f"but the smallest cell (N={min(self.node_counts)}) has only {smallest} "
                    "content nodes. Raise min(node_counts) or lower fan_out.")
        if self.fan_out < 1:
            raise ValueError("fan_out must be >= 1.")
        # The needle must fit the smallest node WITH room left for the offset to vary.
        #
        # The old form of this check (`code_len + 24 > min(token_counts)`) knew only the
        # fan_out=1 needle, so it passed T=32 at fan_out=2 — where the 29-token needle
        # leaves 3 tokens of filler and `max_needle_offset` collapses to 0. Every needle
        # then sits at position 0 of its node: not a low-dilution cell but "node =
        # needle", with the position randomization that every other T has removed. The
        # cell silently stops being on the same axis as the rest of the sweep.
        from .data import SUFFIX_SLACK
        floor = self.needle_tokens() + SUFFIX_SLACK + 1
        if min(self.token_counts) < floor:
            raise ValueError(
                f"token_counts minimum ({min(self.token_counts)}) is below the floor {floor} "
                f"for fan_out={self.fan_out}, code_len={self.code_len}: a {self.needle_tokens()}-"
                f"token needle plus {SUFFIX_SLACK} tokens of slack leaves no room for the "
                "needle offset to vary. Raise token_counts or lower fan_out.")
        if self.uses_magnetic and self.magnetic_m and self.magnetic_m < max(self.node_counts):
            raise ValueError(
                f"magnetic_m={self.magnetic_m} truncates the eigenbasis below the largest "
                f"graph ({max(self.node_counts)} nodes). Set magnetic_m >= max(node_counts) or 0.")
        # magnetic_m_collate is exempt from the check above ON PURPOSE: truncating
        # the eigenbasis is exactly what the M-sweep measures (LINEAR_BIAS.md
        # §2.6). The build stays complete; only what the collator hands the model
        # is narrowed, so one build serves the whole grid.
        if self.magnetic_m_collate:
            if not self.uses_magnetic:
                raise ValueError(
                    f"magnetic_m_collate={self.magnetic_m_collate} set with no magnetic "
                    "bias enabled — it would silently do nothing.")
            if self.magnetic_m and self.magnetic_m_collate > self.magnetic_m:
                raise ValueError(
                    f"magnetic_m_collate={self.magnetic_m_collate} exceeds the built "
                    f"magnetic_m={self.magnetic_m}; the collator can only truncate what "
                    "the dataset stores, so this would silently fall back to the smaller "
                    "value and mislabel the run.")
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
        if self.k_hop < 0:
            raise ValueError("k_hop must be >= 0.")
        return self
