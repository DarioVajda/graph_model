"""
Build the `.gtds` caches for the context experiment (``--mode data_prep``).

Three kinds of split come out of one config:

    processed_datasets/<data_config_key>/
        train.gtds                 the length-capped (N, T) mixture
        dev.gtds                   same mixture, held out
        test/n{N}_t{T}.gtds        one per grid cell, all 25 built from the SAME
                                   blueprints, so cells are paired (data.py)

Idempotent: a split that already exists on disk is skipped, so re-running after
adding a cell only builds the new one. CPU-only apart from the optional GPU path
inside the feature kernels.

The graph features (SPD, magnetic, optionally RRWP) are computed per split
because they depend on the node subset — a smaller-N cell is a different graph,
not a slice of a bigger one.
"""

import json
import os

from transformers import AutoTokenizer

from ...utils import TextGraphDataset
from .config import EXPERIMENT_NAME
from .data import (
    answer_prefix_len, build_code_pool, build_id_pool, build_split_graphs, load_corpus,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(EXPERIMENT_DIR, "processed_datasets")
RAW_DATA_DIR = os.path.join(EXPERIMENT_DIR, "raw_data")


def split_paths(cfg, root=None):
    """``{split_name: base_path}`` for every split this config defines.

    Pure string building — it never touches the filesystem, so split names can be
    enumerated for a config that was never built (which is all several tests
    want). ``root`` defaults to this config's EXACT key; pass a resolved root
    (see ``load_split``) to read from a superset build instead.
    """
    root = root or os.path.join(OUTPUT_ROOT, cfg.data_config_key())
    paths = {"train": os.path.join(root, "train"), "dev": os.path.join(root, "dev")}
    mixed = bool(cfg.hop_counts)
    for (n, t) in cfg.selected_cells():
        for k in cfg.hops_list():
            name = cell_split_name(n, t, k if mixed else None)
            leaf = f"n{n}_t{t}" + (f"_h{k}" if mixed else "")
            paths[name] = os.path.join(root, "test", leaf)
    return paths


def cell_split_name(n, t, hops=None):
    """Name of one evaluation split.

    ``hops=None`` keeps the historical ``test_n{N}_t{T}`` name, so every single-k
    build on disk resolves unchanged. Under a k mixture the grid is a 3-axis product
    and k has to be in the name, or the four k values of a cell overwrite each other.
    """
    return f"test_n{n}_t{t}" + (f"_h{hops}" if hops is not None else "")


def _finalize(ds, cfg, tokenizer):
    """Compute features + tokenize + label one built dataset, in place."""
    if cfg.spd:
        # cutoff == the model's SPD bucket cap: anything beyond lands in the
        # far/unreachable bucket SPDBias clamps into anyway.
        ds.compute_shortest_path_distances(cutoff=cfg.max_spd)
    if cfg.rrwp:
        ds.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps)
    if cfg.uses_magnetic:
        ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m)
    ds.cast_float_features_to_fp32()

    # Content nodes are built to exactly T tokens, so max_length only has to be a
    # bound that never truncates them (the assertion below catches it if it does).
    ds.tokenize(tokenizer, max_length=max(cfg.token_counts), add_eos=True)

    q_len = answer_prefix_len(tokenizer)

    def get_labels(example):
        import torch
        input_ids = example["input_ids"][example["prompt_node"]]
        labels = torch.tensor(input_ids, dtype=torch.long)
        labels[:q_len] = -100          # supervise the code (+ EOS) only
        return labels

    ds.compute_labels(get_labels)
    return ds


def check_split(ds, cfg, split_name):
    """Assert the build invariants (README §A.4) on every graph of a split.

    These are cheap and they are the only thing standing between this experiment
    and a heatmap of a build bug, so they run on every build rather than only in
    the test suite.
    """
    n_supervised = cfg.code_len + 1        # code tokens + EOS
    for i in range(len(ds)):
        item = ds[i]
        g = ds.graphs[i]
        n, t = g.graph["cell_n"], g.graph["cell_t"]
        pn, qn = g.graph["prompt_node"], g.graph["question_node"]

        if len(item["input_ids"]) != n:
            raise AssertionError(f"{split_name}[{i}]: {len(item['input_ids'])} nodes, expected {n}")
        for node, ids in enumerate(item["input_ids"]):
            if node in (pn, qn):
                continue
            if len(ids) != t:
                raise AssertionError(
                    f"{split_name}[{i}]: content node {node} has {len(ids)} tokens, expected {t}")

        n_lab = int((item["labels"] != -100).sum())
        if n_lab != n_supervised:
            raise AssertionError(
                f"{split_name}[{i}]: {n_lab} supervised positions, expected {n_supervised} "
                f"(code_len={cfg.code_len} + EOS)")

        texts = item["text"]
        gold_code, gold_id = g.graph["gold_code"], g.graph["gold_id"]
        holders = [k for k in range(n) if k not in (pn, qn) and gold_code in texts[k]]
        if len(holders) != 1:
            raise AssertionError(
                f"{split_name}[{i}]: gold code appears in {len(holders)} content nodes, expected 1")

        # Per GRAPH, not per config: with a k mixture cfg.hops is 0 while the
        # graphs are chains, and the lookup-task branch below would then reject
        # every chain graph for the pointers that legitimately name the gold id.
        if g.graph.get("hops", cfg.hops):
            # The whole point of the chain task: the QUESTION must NOT name the
            # answer node, or the traversal is bypassable by a literal match.
            if gold_id in texts[qn]:
                raise AssertionError(
                    f"{split_name}[{i}]: QUESTION names the ANSWER node's id — the "
                    "chain is bypassable by string match")
            start_id = g.graph["start_id"]
            if start_id not in texts[qn]:
                raise AssertionError(f"{split_name}[{i}]: QUESTION does not name the start id")
            # Every chain id must be resolvable to exactly one node that DEFINES it
            # (its own KV sentence); pointers naming it are extra mentions.
            for cid in g.graph["chain_ids"]:
                definers = [k for k in range(n) if k not in (pn, qn)
                            and f"access code for {cid} is" in texts[k]]
                if len(definers) != 1:
                    raise AssertionError(
                        f"{split_name}[{i}]: chain id {cid} is defined by {len(definers)} "
                        "nodes, expected 1")
            # A node whose pointer targets itself would make the chain a no-op.
            if len(set(g.graph["chain_ids"])) != len(g.graph["chain_ids"]):
                raise AssertionError(f"{split_name}[{i}]: chain revisits a node")
        else:
            id_holders = [k for k in range(n) if k not in (pn, qn) and gold_id in texts[k]]
            if len(id_holders) != 1:
                raise AssertionError(
                    f"{split_name}[{i}]: gold id appears in {len(id_holders)} content nodes, "
                    "expected 1")
            if gold_id not in texts[qn]:
                raise AssertionError(f"{split_name}[{i}]: QUESTION node does not name the gold id")


def build_split(cfg, tokenizer, corpus, code_pool, id_pool, split_name, base_path,
                n_graphs, cell=None, hops=None, blueprint_split=None,
                id_offset=0, verbose=True):
    """Build + save one split unless its `.gtds` already exists. Returns the path."""
    built = TextGraphDataset.gtds_path(base_path)
    if os.path.exists(built):
        if verbose:
            print(f"[data_prep] {split_name}: already built ({built}) — skipping")
        return built

    if verbose:
        print(f"[data_prep] {split_name}: building {n_graphs} graphs"
              + (f" at cell N={cell[0]} T={cell[1]}" if cell else " (cell mixture)")
              + (f" hops={hops}" if hops is not None else "")
              + (f" from blueprint {id_offset}" if id_offset else ""))
    graphs = build_split_graphs(
        cfg, tokenizer, corpus, code_pool, id_pool,
        split=blueprint_split or split_name, n_graphs=n_graphs, cell=cell, hops=hops,
        id_offset=id_offset, verbose=verbose)

    ds = TextGraphDataset(graphs, dataset_label=f"{EXPERIMENT_NAME}_{split_name}")
    _finalize(ds, cfg, tokenizer)
    check_split(ds, cfg, split_name)

    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    ds.save(base_path)
    if verbose:
        print(f"[data_prep] {split_name}: saved -> {built}")
    return built


def shard_bounds(cfg, index):
    """``(id_offset, n_graphs)`` of train shard ``index`` — contiguous, non-overlapping.

    Contiguity is what makes sharding safe: ``build_split_graphs`` seeds each graph off
    ``id_offset + i``, so overlapping ranges would build the SAME graphs twice and a gap
    would silently shrink the training set below ``n_train``.
    """
    base, rem = divmod(cfg.n_train, cfg.train_shards)
    start = index * base + min(index, rem)
    return start, base + (1 if index < rem else 0)


def shard_path(cfg, index):
    root = os.path.join(OUTPUT_ROOT, cfg.data_config_key())
    return os.path.join(root, f"train.shard{index}of{cfg.train_shards}")


def run_data_merge_mode(cfg, verbose=True):
    """Concatenate the train shards into the single ``train.gtds`` training expects.

    Separate from data_prep because it is the one step whose peak RAM is NOT bounded by
    sharding: ``TextGraphDataset.__add__`` materialises every graph in one process. It
    is cheap in every other way (no tokenizer, no feature computation, no corpus), so
    giving it its own job lets it be sized independently of the build jobs.
    """
    paths = split_paths(cfg)
    target = TextGraphDataset.gtds_path(paths["train"])
    if os.path.exists(target):
        print(f"[data_merge] {target} already exists — skipping")
        return target

    missing = [i for i in range(cfg.train_shards)
               if not os.path.exists(TextGraphDataset.gtds_path(shard_path(cfg, i)))]
    if missing:
        raise FileNotFoundError(
            f"train shards {missing} of {cfg.train_shards} are not built yet — run "
            f"--mode data_prep --train-shards {cfg.train_shards} --train-shard <i> "
            "for each, then merge.")

    merged = None
    for i in range(cfg.train_shards):
        ds = TextGraphDataset.load(TextGraphDataset.gtds_path(shard_path(cfg, i)))
        merged = ds if merged is None else merged + ds
        if verbose:
            print(f"[data_merge] shard {i}: +{len(ds)} -> {len(merged)} graphs")
    if len(merged) != cfg.n_train:
        raise ValueError(
            f"merged train split has {len(merged)} graphs, expected n_train={cfg.n_train}. "
            "The shard bounds and n_train disagree — do not train on this.")
    merged.assign_label(f"{EXPERIMENT_NAME}_train")
    merged.save(paths["train"])
    print(f"[data_merge] saved -> {target}")
    return target


def run_data_prep_mode(cfg, verbose=True):
    """Build every split this config defines (idempotent).

    With ``--train-shard i`` this builds ONLY that shard of the train split and returns;
    dev and the test grid are built by the same command WITHOUT ``--train-shard``. The
    two are separable because they share nothing but the corpus and the pools, all of
    which are re-derived deterministically from ``data_seed``.
    """
    root = os.path.join(OUTPUT_ROOT, cfg.data_config_key())
    os.makedirs(root, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    corpus = load_corpus(tokenizer, RAW_DATA_DIR, cfg.corpus_tokens, verbose=verbose)
    # One pool size for both: a graph draws ``n_content_max`` ids and the same
    # number of codes, so they need the same headroom over the largest cell.
    code_pool = build_code_pool(tokenizer, cfg.code_len, cfg.id_pool, seed=cfg.data_seed)
    id_pool = build_id_pool(cfg.id_pool)

    paths = split_paths(cfg)

    if cfg.train_shard >= 0:
        offset, size = shard_bounds(cfg, cfg.train_shard)
        build_split(cfg, tokenizer, corpus, code_pool, id_pool,
                    f"train.shard{cfg.train_shard}", shard_path(cfg, cfg.train_shard),
                    size, id_offset=offset, verbose=verbose)
        print(f"[data_prep] shard {cfg.train_shard}/{cfg.train_shards} done "
              f"(blueprints {offset}..{offset + size - 1}); merge with --mode data_merge")
        return root

    if cfg.train_shards == 1:
        build_split(cfg, tokenizer, corpus, code_pool, id_pool, "train", paths["train"],
                    cfg.n_train, verbose=verbose)
    build_split(cfg, tokenizer, corpus, code_pool, id_pool, "dev", paths["dev"],
                cfg.n_dev, verbose=verbose)

    # Every cell reuses the SAME blueprint split ("test") and the same blueprint
    # ids, which is what makes the 25 cells paired rather than 25 independent
    # samples (README §A.3).
    mixed = bool(cfg.hop_counts)
    for (n, t) in cfg.selected_cells():
        for k in cfg.hops_list():
            name = cell_split_name(n, t, k if mixed else None)
            build_split(cfg, tokenizer, corpus, code_pool, id_pool, name, paths[name],
                        cfg.n_test, cell=(n, t), hops=k if mixed else None,
                        blueprint_split="test", verbose=verbose)

    meta = {
        "data_config_key": cfg.data_config_key(),
        "cells": [list(c) for c in cfg.cells()],
        "hops_list": list(cfg.hops_list()),
        "train_cells": [list(c) for c in cfg.train_cells()],
        "cell_lengths": {f"{n}x{t}": cfg.cell_length(n, t) for (n, t) in cfg.cells()},
        "len_buckets": cfg.len_buckets(),
        "grid_len_buckets": cfg.grid_len_buckets(),
        "n_train": cfg.n_train, "n_dev": cfg.n_dev, "n_test": cfg.n_test,
        "max_train_len": cfg.max_train_len, "code_len": cfg.code_len,
        "model_name": cfg.model_name, "data_seed": cfg.data_seed,
        "data_format_version": cfg.data_format_version,
    }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[data_prep] done -> {root}")
    return root


def load_split(cfg, split_name):
    """Load one built split; raises with the build command if it is missing."""
    # Resolve through resolved_data_root so a run that switched a feature OFF can
    # read a build that still has that column: the model never instantiates a
    # module for it, so the extra data is inert. Without this, the Phase 2 arms
    # that drop SPD would each demand a multi-hour rebuild of a strict subset of
    # data already on disk.
    base = split_paths(cfg, root=cfg.resolved_data_root(OUTPUT_ROOT))[split_name]
    built = TextGraphDataset.gtds_path(base)
    if not os.path.exists(built):
        raise FileNotFoundError(
            f"{built} does not exist — run:\n"
            f"  python3 -m src.experiments.{EXPERIMENT_NAME} --mode data_prep")
    return TextGraphDataset.load(built)
