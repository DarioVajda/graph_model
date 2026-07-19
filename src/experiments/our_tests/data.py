"""
Dataset construction and loading for the our_tests experiment.

``run_data_prep_mode(cfg)`` builds this config's splits under ``cfg.dataset_dir()``;
``load_data(cfg)`` returns ``(train, val, test)`` for training. Both are idempotent — a
built split is reused, never rebuilt.

Unlike graphqa (which loads a fixed download) both tasks here are **generated**, so data
prep is genuinely expensive: thousands of graphs, then a magnetic-Laplacian
eigendecomposition per graph. It is a separate sweep mode for that reason, and belongs on
a compute node, not the login node.

The two tasks keep their historical on-disk locations (``config.dataset_dir``), so a
default config resolves onto the existing datasets the paper's numbers came from and
rebuilds nothing.
"""

import os

from ...utils.text_graph_dataset import TextGraphDataset

SPLITS = ("train", "val", "test")


def _split_files(path, split):
    """The ``.gtds`` files making up one split, in order.

    ``train`` may be sharded (``train_0-5000.gtds``, ``train_5000-10000.gtds``, ...) —
    the kg_qa builder writes it that way — so a split is a *list* of files, concatenated
    in shard order.
    """
    if not os.path.isdir(path):
        return []
    files = [f for f in os.listdir(path)
             if f.endswith(".gtds") and (f == f"{split}.gtds" or f.startswith(f"{split}_"))]
    if len(files) > 1:
        # sort by the shard's end index: "train_5000-10000.gtds" -> 10000
        files = sorted(files, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    return files


def _load_split(path, split):
    files = _split_files(path, split)
    if not files:
        raise FileNotFoundError(f"No .gtds files for split {split!r} in {path}")

    dataset = None
    for file in files:
        current = TextGraphDataset.load(os.path.join(path, file))
        dataset = current if dataset is None else dataset + current
    print(f"[data] loaded {split}: {len(dataset)} examples from {len(files)} file(s)")
    return dataset


def _is_built(cfg, split):
    return bool(_split_files(cfg.dataset_dir(), split))


def run_data_prep_mode(cfg):
    """Generate, featurize and cache every split this config needs. Idempotent."""
    from transformers import AutoTokenizer  # deferred: keeps --help / --init fast

    out_dir = cfg.dataset_dir()
    print(f"[data_prep] {cfg.task} (question_node={cfg.question_node}) -> {out_dir}")
    if not cfg.uses_default_cache():
        print("[data_prep] non-default feature settings -> tagged cache dir "
              "(the untagged dir holds the datasets built with the defaults)")

    missing = [s for s in SPLITS if not _is_built(cfg, s)]
    if not missing:
        print(f"[data_prep] already built: {list(SPLITS)} — nothing to do.")
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    os.makedirs(out_dir, exist_ok=True)

    if cfg.task == "family":
        graphs_by_split = _build_family_graphs(cfg)
    else:
        graphs_by_split = _build_kgqa_graphs(cfg)

    from .prompt import label_masker
    masker = label_masker(tokenizer, question_node=cfg.question_node)

    for split in missing:
        graphs = graphs_by_split[split]
        ds = TextGraphDataset(graphs, per_graph_versions=1)
        # add_eos=False is the historical setting for this experiment (graphqa uses
        # True). Held fixed so the re-run isolates the graph-bias reload fix.
        ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=False)
        ds.compute_labels(masker, num_proc=12)
        ds.compute_shortest_path_distances(use_gpu=cfg.use_gpu)
        ds.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps, use_gpu=cfg.use_gpu)
        ds.compute_magnetic_lap(q=cfg.magnetic_q, use_gpu=cfg.use_gpu)
        ds.save(os.path.join(out_dir, split))
        print(f"[data_prep] built {split}: {len(ds)} graphs -> {out_dir}/{split}.gtds")


def _build_family_graphs(cfg):
    """Generate + prepare the family-tree graphs for every split."""
    import random

    from .family_gen import generate_dataset
    from .family_prep import prepare_graph_dataset

    random.seed(cfg.gen_seed)
    raw = generate_dataset(
        n_train=cfg.family_train, n_val=cfg.family_val, n_test=cfg.family_test,
        generations=cfg.family_generations, marriage_prob=cfg.family_marriage_prob,
        child_prob=cfg.family_child_prob, return_dict=True)
    return prepare_graph_dataset(raw, question_node=cfg.question_node)


def _build_kgqa_graphs(cfg):
    """Generate + prepare the knowledge-graph QA graphs for every split."""
    import random

    import numpy as np

    from .kgqa_gen import generate_dataset
    from .kgqa_prep import prepare_dataset

    random.seed(cfg.gen_seed)
    np.random.seed(cfg.gen_seed)
    train_raw, val_raw, test_raw = generate_dataset(
        train_count=cfg.kgqa_train, val_count=cfg.kgqa_val, test_count=cfg.kgqa_test,
        min_nodes=cfg.kgqa_min_nodes, max_nodes=cfg.kgqa_max_nodes)
    # Prepared in the generator's own order (train, val, test) so the shared RNG the
    # question sampler draws from advances exactly as it did historically.
    return {
        "train": prepare_dataset(train_raw, question_node=cfg.question_node),
        "val": prepare_dataset(val_raw, question_node=cfg.question_node),
        "test": prepare_dataset(test_raw, question_node=cfg.question_node),
    }


def load_data(cfg):
    """Return ``(train, val, test)`` for this config, building nothing.

    Raises if a split is missing rather than silently building it: data prep is a
    separate, expensive mode, and a GPU training job discovering it must generate and
    eigendecompose thousands of graphs is a mistake worth failing loudly on.
    """
    out_dir = cfg.dataset_dir()
    for split in SPLITS:
        if not _is_built(cfg, split):
            raise FileNotFoundError(
                f"Split {split!r} is not built for task {cfg.task!r} "
                f"(question_node={cfg.question_node}) at {out_dir}. Build it first:\n"
                f"  python3 -m src.experiments.our_tests --mode data_prep "
                f"--task {cfg.task} --question-node {cfg.question_node}")

    train = _load_split(out_dir, "train")
    val = _load_split(out_dir, "val")
    test = _load_split(out_dir, "test")
    print(f"[data] {cfg.task} (question_node={cfg.question_node}): "
          f"train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test
