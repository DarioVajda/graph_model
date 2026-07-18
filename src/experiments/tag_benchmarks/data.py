"""
Dataset construction and loading for the TAG benchmarks experiment.

``run_data_prep_mode(cfg)`` samples one text-attributed graph per node of the raw
benchmark and caches the built splits under ``cfg.dataset_dir()``; ``load_data(cfg)``
returns ``(train, val, test)`` for training. Both are idempotent — a built split is
reused, never rebuilt.

Splits are the benchmark's own: every raw dataset ships ``train_mask`` / ``val_mask`` /
``test_mask`` covering every node (verified for cora, pubmed and reddit), and a node's
graph goes to the split its mask names. The model trains on train only, the checkpoint
is selected on val, and test is scored once at the end.

On-disk layout (unchanged from the pre-refactor pipeline, so existing caches load):

    <dataset_dir>/<split>/<first>-<last>.gtds     # chunks of CHUNK_SIZE graphs

Each chunk carries the full feature set (SPD, RRWP, magnetic) regardless of which
biases a training arm enables, so ablation arms share one built dataset.
"""

import os
import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

from ...utils import TextGraphDataset
from .config import ANSWER_PREFIX, RAW_DIR, SPLITS

# Graphs per on-disk chunk. Chunking keeps peak memory bounded while building
# ogbn-arxiv (90,941 train graphs) and is the layout the existing caches use.
CHUNK_SIZE = 10_000


# ── Node text mappings ────────────────────────────────────────────────────────
# Each takes (data, node_idx, distance_from_target) and returns that node's text, so
# how much text a node contributes can depend on how far it sits from the target.

def _title(data, idx, dist):
    return f"{data.title[idx]}"


def _abstract(data, idx, dist):
    return f"{data.title[idx]}\n{data.abs[idx]}"


def _raw_text(data, idx, dist):
    return f"{data.raw_texts[idx]}"


def _truncate(text, max_chars):
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def make_text_mapping(cfg):
    """Build the node-text function this config's ``text_mapping`` names.

    Mirrors the pre-refactor mapping functions exactly (the `get_*` free functions and
    the two parameterized callables), but resolved from config rather than by editing
    `process_data.py`'s __main__ and a `get_mapping_name` reverse lookup.
    """
    name, param = cfg.text_mapping, cfg.text_mapping_param

    if name == "all_titles":
        return _title
    if name == "all_abstracts":
        return _abstract
    if name == "target_abstract":
        # Only the target node spends its abstract; neighbours contribute titles.
        return lambda data, idx, dist: _abstract(data, idx, dist) if dist == 0 else _title(data, idx, dist)
    if name == "neighbor_abstracts":
        return lambda data, idx, dist: _abstract(data, idx, dist) if dist <= 1 else _title(data, idx, dist)
    if name == "random_abstracts":
        # Abstract kept with probability p**dist, so the chance of a full abstract
        # decays with distance from the target.
        def random_abstracts(data, idx, dist):
            if dist == 0 or random.random() < param ** dist:
                return _abstract(data, idx, dist)
            return _title(data, idx, dist)
        return random_abstracts
    if name == "full_text":
        return _raw_text
    if name == "truncated_text":
        return lambda data, idx, dist: _truncate(_raw_text(data, idx, dist), int(param))
    if name == "more_target_text":
        # A character budget that shrinks with distance: 196 / 128 / 64.
        budget = {0: 196, 1: 128}
        return lambda data, idx, dist: _truncate(_raw_text(data, idx, dist), budget.get(dist, 64))

    raise ValueError(f"Unhandled text_mapping {name!r} (config.validate should have caught this).")


# ── Label masking ─────────────────────────────────────────────────────────────

class AnswerLabelMasker:
    """Mask every prompt-node token except the answer, which follows ``ANSWER_PREFIX``.

    Matches the LAST occurrence of the prefix token sequence: the node's own text can
    contain "A:" (an abstract easily does), but the instruction that ends with it is
    always appended last, so the final match is the real answer boundary.
    """

    def __init__(self, question_end):
        if not question_end:
            raise ValueError("question_end must be a non-empty list of token ids.")
        self.question_end = list(question_end)

    def __call__(self, example):
        prompt_node = example.get("prompt_node", None)
        prompt_input_ids = example["input_ids"][prompt_node]
        labels = prompt_input_ids.copy()

        end = len(self.question_end)
        question_end_index = None
        for i in range(len(prompt_input_ids) - end + 1):
            if prompt_input_ids[i:i + end] == self.question_end:
                question_end_index = i + end - 1
        if question_end_index is None:
            raise ValueError(
                f"Could not find the answer-prefix token sequence {self.question_end} in the "
                f"prompt node's input_ids: {prompt_input_ids}")

        for i in range(question_end_index + 1):
            labels[i] = -100
        return labels


def answer_prefix_ids(tokenizer):
    """Token ids of ``ANSWER_PREFIX``, the label-mask boundary.

    Derived from the tokenizer rather than hard-coded (the pre-refactor pipeline pinned
    the literal ``[32, 25]``, which is only correct for the Llama tokenizer).
    """
    return tokenizer.encode(ANSWER_PREFIX.rstrip(), add_special_tokens=False)


# ── Raw graph -> text graph ───────────────────────────────────────────────────

def _to_text_graph(target_node, nodes, edges, label):
    graph = nx.Graph()
    for node_id, text in nodes.items():
        if node_id == target_node:
            text = text + label
        graph.add_node(node_id, text=text)
    for source, target in edges:
        graph.add_edge(source, target)
    graph.graph["prompt_node"] = target_node
    return graph


def _sample_neighborhood(data, node, subset, edge_index, distances, max_nodes, mapping, instruction):
    """One text-attributed graph for ``node``'s neighbourhood.

    When the neighbourhood exceeds ``max_nodes`` it is subsampled with a preference for
    closer nodes: distances get uniform (0, 0.5) noise before sorting, so ties within a
    hop break randomly but a nearer node never loses to a farther one.
    """
    if subset.shape[0] > max_nodes:
        noisy = {n: dist + random.uniform(0, 0.5) for n, dist in distances.items()}
        sorted_nodes = sorted(subset.tolist(), key=lambda x: noisy.get(x, float("inf")))
        kept = sorted(sorted_nodes[:max_nodes])
        subset = torch.tensor(kept, dtype=torch.long)
        edge_index = edge_index[:, (torch.isin(edge_index[0], subset)
                                    & torch.isin(edge_index[1], subset))]

    nodes = {
        idx.item(): mapping(data, idx, distances[idx.item()])
                    + (f"\n{instruction}" if idx.item() == node else "")
        for idx in subset
    }
    edges = edge_index.t().tolist()
    label = data.label_texts[data.y[node]]
    return _to_text_graph(node, nodes, edges, label)


def _node_split(data, node):
    """The split this node belongs to, or None if no mask claims it.

    The pre-refactor loop let `split` fall through from the previous iteration when a
    node matched no mask, silently filing it under whatever split came before. All four
    benchmarks mask every node, so it never fired — but it returns None here and the
    caller counts and skips, rather than leaving a latent mislabeling bug in place.
    """
    for split, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        if mask[node]:
            return split
    return None


# ── Cache paths ───────────────────────────────────────────────────────────────

def split_dir(cfg, split):
    return os.path.join(cfg.dataset_dir(), split)


def _chunk_paths(cfg, split):
    """This split's chunk directories, in build order (`<first>-<last>.gtds`)."""
    path = split_dir(cfg, split)
    if not os.path.isdir(path):
        return []
    chunks = [c for c in os.listdir(path) if c.endswith(".gtds")]
    return [os.path.join(path, c) for c in sorted(chunks, key=lambda x: int(x.split("-")[0]))]


def _is_built(cfg, split):
    return len(_chunk_paths(cfg, split)) > 0


def _cache_versions(cfg, split):
    """The `per_graph_versions` the built split actually carries, from its metadata."""
    import json
    chunks = _chunk_paths(cfg, split)
    if not chunks:
        return None
    meta_path = os.path.join(chunks[0], "metadata.json")
    if not os.path.exists(meta_path):
        return 1        # legacy chunk with no metadata; TextGraphDataset assumes 1 too
    with open(meta_path) as f:
        return json.load(f).get("per_graph_versions", 1)


def _check_samples_per_node(cfg, actual):
    """Fail loudly when the cache's augmentation isn't what this config asked for.

    `samples_per_node` is baked into the built dataset but absent from its directory
    name (the pre-refactor scheme never encoded it), so two configs differing only in
    this knob resolve to the same directory. Nothing else would catch the mismatch.
    """
    if cfg.samples_per_node is None or actual is None or cfg.samples_per_node == actual:
        return
    raise ValueError(
        f"The built train split at {split_dir(cfg, 'train')} was created with "
        f"samples_per_node={actual}, but this config asks for {cfg.samples_per_node}. That "
        f"knob is not part of the directory name, so the cache cannot distinguish them.\n"
        f"Either set --samples-per-node {actual} to use this cache, or move/delete the "
        f"directory and rebuild with --mode data_prep.")


# ── Data prep ─────────────────────────────────────────────────────────────────

def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_chunk(cfg, graphs, path, tokenizer, question_end, versions):
    """Tokenize, compute every structural feature, and write one chunk."""
    dataset = TextGraphDataset(graphs, per_graph_versions=versions)
    dataset.tokenize(tokenizer, max_length=cfg.max_length, add_eos=True)
    dataset.compute_shortest_path_distances(use_gpu=cfg.use_gpu)
    dataset.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps, use_gpu=cfg.use_gpu)
    dataset.compute_magnetic_lap(q=cfg.magnetic_q, use_gpu=cfg.use_gpu, m=cfg.magnetic_m)
    dataset.compute_labels(AnswerLabelMasker(question_end))
    dataset.save(path)


def _stride_for(limit, available):
    """Take every k-th node so `limit` of `available` are kept (1 = keep all)."""
    if limit is None or limit >= available:
        return 1
    return max(1, available // limit)


def load_raw(cfg):
    """The raw PyG benchmark graph for this config's dataset."""
    path = os.path.join(RAW_DIR, cfg.dataset, "processed_data.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset missing at {path}. Download it per the README (the RGLM "
            f"release) and unpack it into {RAW_DIR}/{cfg.dataset}/.")
    return torch.load(path, weights_only=False)


def run_data_prep_mode(cfg):
    """Build and cache every split for this config. Idempotent."""
    from transformers import AutoTokenizer   # deferred: keeps --help / --init fast

    missing = [s for s in SPLITS if not _is_built(cfg, s)]
    if not missing:
        print(f"[data_prep] already built: {list(SPLITS)} at {cfg.dataset_dir()} — nothing to do.")
        _check_samples_per_node(cfg, _cache_versions(cfg, "train"))
        return

    if not cfg.uses_default_cache():
        print("[data_prep] non-default feature settings -> tagged cache dir "
              "(the untagged dir holds the datasets built with the defaults)")
    if _is_built(cfg, "train"):
        _check_samples_per_node(cfg, _cache_versions(cfg, "train"))

    print(f"[data_prep] {cfg.dataset} hops={cfg.hops} neighbors={cfg.max_neighbors} "
          f"mapping={cfg.mapping_name()} -> {cfg.dataset_dir()}")
    print(f"[data_prep] building split(s): {missing}")

    _seed_everything(cfg.data_seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    question_end = answer_prefix_ids(tokenizer)
    print(f"[data_prep] answer-prefix {ANSWER_PREFIX.rstrip()!r} -> token ids {question_end}")

    data = load_raw(cfg)
    mapping = make_text_mapping(cfg)
    instruction = cfg.instruction()

    counts = {s: data[f"{s}_mask"].sum().item() for s in SPLITS}
    print(f"[data_prep] raw nodes: total={data.num_nodes} "
          + " ".join(f"{s}={counts[s]}" for s in SPLITS))

    strides = {
        "train": _stride_for(cfg.max_train_samples, counts["train"]),
        "val": _stride_for(cfg.max_val_samples, counts["val"]),
        "test": 1,
    }
    for split, stride in strides.items():
        if stride > 1:
            print(f"[data_prep] subsampling {split}: every {stride}th node "
                  f"(~{counts[split] // stride} of {counts[split]})")

    pending = {s: [] for s in SPLITS}       # graphs not yet flushed to a chunk
    written = {s: 0 for s in SPLITS}        # graphs already on disk, per split
    seen = {s: 0 for s in SPLITS}           # raw nodes visited, for striding
    unmasked = 0

    def versions_for(split):
        # Only train carries the neighbourhood-resampling augmentation; val/test get
        # one deterministic graph per node.
        return cfg.samples_per_node if split == "train" else 1

    def flush(split):
        first, last = written[split], written[split] + len(pending[split])
        path = os.path.join(split_dir(cfg, split), f"{first}-{last}")
        os.makedirs(split_dir(cfg, split), exist_ok=True)
        print(f"[data_prep] writing {split} chunk {first}-{last} ...")
        _save_chunk(cfg, pending[split], path, tokenizer, question_end, versions_for(split))
        written[split] = last
        pending[split] = []

    for node in range(data.num_nodes):
        split = _node_split(data, node)
        if split is None:
            unmasked += 1
            continue
        if split not in missing:
            continue

        seen[split] += 1
        if seen[split] % strides[split] != 0:
            continue

        if node % 500 == 0:
            print(f"[data_prep] node {node}/{data.num_nodes} "
                  + " ".join(f"{s}={written[s] + len(pending[s])}" for s in missing))

        subset, edge_index, _, _ = k_hop_subgraph(node, cfg.hops, data.edge_index)
        G = nx.Graph()
        G.add_edges_from(edge_index.t().tolist())
        distances = nx.single_source_shortest_path_length(G, node)

        for _ in range(versions_for(split)):
            pending[split].append(_sample_neighborhood(
                data, node, subset, edge_index, distances,
                cfg.max_neighbors, mapping, instruction))

        if len(pending[split]) >= CHUNK_SIZE:
            flush(split)

    for split in missing:
        if pending[split]:
            flush(split)

    if unmasked:
        print(f"[data_prep] WARNING: {unmasked} nodes matched no split mask and were skipped.")
    for split in missing:
        print(f"[data_prep] built {split}: {written[split]} graphs -> {split_dir(cfg, split)}")


# ── Loading ───────────────────────────────────────────────────────────────────

def _load_split(cfg, split):
    """Concatenate this split's chunks into one dataset."""
    dataset = None
    for chunk in _chunk_paths(cfg, split):
        ds = TextGraphDataset.load(chunk)
        ds.assign_label(split)
        dataset = ds if dataset is None else dataset + ds
    return dataset


def _strided_subsample(dataset, max_size):
    """At most ``max_size`` graphs, as contiguous blocks of 50 evenly strided across.

    Preserved verbatim from the pre-refactor loader: a plain head slice would bias the
    sample (nodes are ordered by original graph index, not shuffled), so this walks the
    whole split in blocks. Deterministic — no RNG — so checkpoint selection compares
    like with like across runs.
    """
    if max_size is None or len(dataset) <= max_size:
        return dataset

    block_size = 50
    pieces = max_size // block_size
    stride = (len(dataset) + pieces - 1) // pieces
    out = None
    for i in range(pieces):
        block = dataset[i * stride: i * stride + block_size]
        out = block if out is None else out + block
    return out[:max_size]


def load_data(cfg):
    """Return ``(train, val, test)`` for this config, building nothing.

    Raises if a split is missing rather than silently building it: data prep is a
    separate mode, and a GPU training job discovering it must build 90,941 subgraphs of
    features first is a mistake worth failing loudly on.
    """
    for split in SPLITS:
        if not _is_built(cfg, split):
            raise FileNotFoundError(
                f"Split {split!r} is not built at {split_dir(cfg, split)}. Build it first:\n"
                f"  python3 -m src.experiments.tag_benchmarks --mode data_prep "
                f"--dataset {cfg.dataset} --max-neighbors {cfg.max_neighbors} "
                f"--text-mapping {cfg.text_mapping} --samples-per-node <N>")

    _check_samples_per_node(cfg, _cache_versions(cfg, "train"))

    train = _load_split(cfg, "train")
    val = _strided_subsample(_load_split(cfg, "val"), cfg.val_subsample)
    test = _strided_subsample(_load_split(cfg, "test"), cfg.test_subsample)

    print(f"[data] {cfg.dataset}/{cfg.mapping_name()}: train={len(train)} val={len(val)} "
          f"test={len(test)} (samples_per_node={train.per_graph_versions})")
    return train, val, test
