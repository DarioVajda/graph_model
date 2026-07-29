"""Build and load the cached `.gtds` splits for the RelBench experiment.

``run_data_prep_mode(cfg)`` turns every row of a relbench task table into one text-attributed
graph and caches the built splits under ``cfg.dataset_dir()``; ``load_data(cfg)`` returns
``(train, val, test)`` and refuses to build, so a GPU job never silently discovers it has
hours of CPU work to do first.

Layout, matching tag_benchmarks so the loaders are interchangeable:

    <dataset_dir>/<split>/<first>-<last>.gtds

**The ordering invariant.** Graphs are built in `task.get_table(split)` row order and the
table row index is stored on every graph, because `task.evaluate` compares predictions to
targets *positionally*. A shuffle of val or test would not raise -- it would quietly produce
a meaningless metric. `load_data` asserts monotonicity for exactly that reason.
"""

import json
import os
import random

import networkx as nx
import numpy as np
import torch

from ...utils import TextGraphDataset
from .config import ANSWER_PREFIX, SPLITS
from .graph_build import build_and_cache, link_tables
from .row_text import RowRenderer
from .sampler import TemporalSampler

# Graphs per on-disk chunk. rel-trial's `site-success` has 151k train rows, so chunking is
# what keeps peak memory bounded during a build.
CHUNK_SIZE = 5_000

QUESTION_NODE_ID = "__question__"
PROMPT_NODE_ID = "__prompt__"


def schema_node_id(table):
    return f"__schema__{table}"


def _hoisted_tables(sampled, renderer, seed_ts, text_mode):
    """Which tables get a header node, and therefore render their rows positionally.

    `key_value` hoists nothing (the control). `schema_node` hoists every table, which is the
    scheme at full strength and loses on a sparse, low-multiplicity schema. `shortest` hoists
    a table only when doing so is actually shorter for the rows this graph sampled -- so it
    is never worse than `key_value` by construction, and needs no threshold to tune.
    """
    if text_mode == "key_value":
        return set()

    rows_by_table = {}
    for table, row, _ in sampled.nodes:
        rows_by_table.setdefault(table, []).append(row)

    if text_mode == "schema_node":
        return set(rows_by_table)
    return {t for t, rows in rows_by_table.items()
            if renderer.cheaper_as_schema_node(t, rows, seed_ts)}


# -- the pieces that must stay task-agnostic (PLAN.md 5.0 A) ------------------

def task_description(task):
    """One line describing the task, from relbench metadata -- never hand-written.

    Prefers the task class's own docstring, but only when the class lives in
    ``relbench.tasks.*``. ``AutoCompleteTask`` is a generic per-column factory whose
    docstring documents its constructor ("Args: dataset: The dataset object...") and would
    otherwise be spliced into the question as if it described the task. All 14 tasks in
    ``relbench/tasks/dbinfer.py`` have no docstring at all, so the fallback is not
    hypothetical (PLAN.md 4.1 finding 4).
    """
    cls = type(task)
    if cls.__module__.startswith("relbench.tasks."):
        doc = " ".join((cls.__dict__.get("__doc__") or "").split())
        if doc:
            return doc

    from relbench.base import TaskType
    window = f"{task.timedelta.days} days"
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return (f"Predict whether {task.target_col} holds for this {task.entity_table} "
                f"row in the next {window}.")
    return (f"Predict {task.target_col} for this {task.entity_table} row over the next "
            f"{window}.")


def question_text(task, description, entity_row, seed_ts):
    """The QUESTION node: what to predict, for whom, as of when."""
    import datetime
    date = datetime.datetime.utcfromtimestamp(seed_ts).strftime("%Y-%m-%d")
    return (f"QUESTION | {description} | prediction date: {date} | "
            f"window: next {task.timedelta.days} days | "
            f"entity: {task.entity_table} #{entity_row}")


def answer_text(task, target):
    """The supervised span. Label words come from `task_type`, never a per-task table."""
    from relbench.base import TaskType
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return f"{ANSWER_PREFIX} {'yes' if int(target) == 1 else 'no'}"
    return f"{ANSWER_PREFIX} {float(target):.2f}"


# -- label masking ------------------------------------------------------------

class AnswerLabelMasker:
    """Supervise only the answer -- everything before ``ANSWER_PREFIX`` is masked to -100.

    Matches the LAST occurrence of the prefix tokens: a sampled row's text can legitimately
    contain "Answer:" (rel-trial's `criteria` and `description` fields are free prose), but
    the prompt node's own instruction always ends with it.
    """

    def __init__(self, prefix_ids):
        if not prefix_ids:
            raise ValueError("prefix_ids must be a non-empty list of token ids.")
        self.prefix_ids = list(prefix_ids)

    def __call__(self, example):
        prompt_node = example.get("prompt_node", None)
        input_ids = example["input_ids"][prompt_node]
        labels = input_ids.copy()

        width = len(self.prefix_ids)
        boundary = None
        for i in range(len(input_ids) - width + 1):
            if input_ids[i:i + width] == self.prefix_ids:
                boundary = i + width - 1
        if boundary is None:
            raise ValueError(
                f"Answer prefix {self.prefix_ids} not found in the prompt node's input_ids. "
                f"The prompt node text should end with {ANSWER_PREFIX!r} + the label.")
        for i in range(boundary + 1):
            labels[i] = -100
        return labels


def answer_prefix_ids(tokenizer):
    return tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)


# -- graph assembly -----------------------------------------------------------

def build_graph(sampled, renderer, seed_ts, question, answer, question_node="isolated",
                prompt_node="seed", text_mode="key_value"):
    """One sampled neighborhood -> one text-attributed `nx.DiGraph`.

    Node 0 is the seed row, prefixed `TARGET`. The QUESTION node is `isolated` by default:
    it carries no edge, because the bidirectional prefix mask already exposes it to every
    graph token, and that is what makes the encoding question-conditioned. The PROMPT node
    holds the answer and is the only supervised node.

    The PROMPT node is attached to node 0 by default -- kgqa's convention, where the prompt
    is wired to the topic entities unconditionally and node 0 is our analogue of those. The
    edge is oriented PROMPT -> seed to match both kgqa and this graph's child -> parent
    convention. It feeds SPD and the magnetic Laplacian only; the attention mask keys off
    ``prompt_node`` identity, not edges, so the answer span stays causally masked either way.
    """
    graph = nx.DiGraph()
    hoisted = _hoisted_tables(sampled, renderer, seed_ts, text_mode)

    for position, (table, row, _) in enumerate(sampled.nodes):
        aligned = table in hoisted
        text = renderer.render(table, row, seed_ts, aligned=aligned)
        if position == 0:
            text = "TARGET " + text
        graph.add_node(position, text=text)
    for i, j in sampled.edges:
        graph.add_edge(i, j)

    # Header nodes are appended after the rows, so node 0 stays the seed. Sequence order does
    # not matter for them: the prefix is bidirectional, so a row sees its header wherever it
    # sits. (The flat arm has no such luxury -- see `build_flat_graph`.)
    for table in sorted(hoisted):
        node = schema_node_id(table)
        graph.add_node(node, text=renderer.header(table))
        for position, (row_table, _, _) in enumerate(sampled.nodes):
            if row_table == table:
                graph.add_edge(position, node)

    if question_node != "off":
        graph.add_node(QUESTION_NODE_ID, text=question)
        if question_node == "seed":
            graph.add_edge(QUESTION_NODE_ID, 0)

    graph.add_node(PROMPT_NODE_ID, text=answer)
    if prompt_node == "seed":
        graph.add_edge(PROMPT_NODE_ID, 0)
    graph.graph["prompt_node"] = PROMPT_NODE_ID
    return graph


def build_flat_graph(sampled, renderer, seed_ts, question, answer, text_mode="key_value"):
    """The matched control: the SAME rows, same order, one node, no topology.

    Byte-identical supervision to the graph arm -- the only difference is that the rows are
    concatenated into a single node instead of carrying the fkey structure in the attention
    bias. This is control #1 and the primary scientific comparison (PLAN.md 8.1).

    **One asymmetry, only under the hoisting modes.** Header lines are emitted *before* the
    rows here, because this node is read causally: a positional row whose header comes later
    in the sequence is unreadable. The graph arm appends its header nodes after the rows
    instead, which costs it nothing because its prefix is bidirectional. So under
    `schema_node`/`shortest` the two arms carry identical *lines* in identical row order, but
    the header lines sit in different places. Under the default `key_value` there are no
    header lines and the arms stay byte-identical.
    """
    hoisted = _hoisted_tables(sampled, renderer, seed_ts, text_mode)
    lines = [renderer.header(table) for table in sorted(hoisted)]
    for position, (table, row, _) in enumerate(sampled.nodes):
        text = renderer.render(table, row, seed_ts, aligned=table in hoisted)
        lines.append(("TARGET " + text) if position == 0 else text)

    graph = nx.DiGraph()
    graph.add_node(PROMPT_NODE_ID, text="\n".join([question] + lines + [answer]))
    graph.graph["prompt_node"] = PROMPT_NODE_ID
    return graph


# -- cache paths --------------------------------------------------------------

def split_dir(cfg, split):
    return os.path.join(cfg.dataset_dir(), split)


def _chunk_paths(cfg, split):
    path = split_dir(cfg, split)
    if not os.path.isdir(path):
        return []
    chunks = [c for c in os.listdir(path) if c.endswith(".gtds")]
    return [os.path.join(path, c) for c in sorted(chunks, key=lambda x: int(x.split("-")[0]))]


def _is_built(cfg, split):
    return len(_chunk_paths(cfg, split)) > 0


def _write_manifest(cfg, counts):
    """Record what this cache is, so a directory of hashes stays readable."""
    path = os.path.join(cfg.dataset_dir(), "manifest.json")
    os.makedirs(cfg.dataset_dir(), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"data_config_key": cfg.data_config_key(),
                   "cache_fields": cfg.cache_fields(),
                   "counts": counts}, fh, indent=2, default=str)


# -- data prep ----------------------------------------------------------------

def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_chunk(cfg, graphs, path, tokenizer, prefix_ids, versions):
    """Tokenize, compute the structural features, mask the labels, write one chunk."""
    dataset = TextGraphDataset(graphs, per_graph_versions=versions)
    dataset.tokenize(tokenizer, max_length=cfg.max_length, add_eos=True)
    if not cfg.is_flat():
        dataset.compute_shortest_path_distances(use_gpu=cfg.use_gpu)
        if cfg.rrwp:
            dataset.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps, use_gpu=cfg.use_gpu)
        dataset.compute_magnetic_lap(q=cfg.magnetic_q, use_gpu=cfg.use_gpu, m=cfg.magnetic_m)
    dataset.compute_labels(AnswerLabelMasker(prefix_ids))
    dataset.save(path)


def _stride_for(limit, available):
    if limit is None or limit >= available:
        return 1
    return max(1, available // limit)


def make_builders(cfg):
    """`(task, sampler, renderer, description)` -- everything a build or a dump needs."""
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    # `get_db()` defaults to `upto_test_timestamp=True` -- rows after `test_timestamp` are
    # never input, even for test seeds in later evaluation windows. That default is exactly
    # what relbench's own `examples/gnn_entity.py` uses, so our input data is the same data
    # RDL and LightGBM get. Do not "fix" the resulting staleness by passing False: it would
    # hand us history no published baseline had.
    db = get_dataset(cfg.dataset).get_db()
    data = build_and_cache(db, cfg.dataset, DATASETS_DIR_GRAPHS, verbose=False)
    # `download=True` verifies the cached task parquets against the RelBench server's
    # hashes. Computing them locally instead is permitted by relbench but leaves row order
    # undefined (`make_table`: "rows need not be ordered deterministically") -- and row
    # order *is* the evaluation index here.
    task = get_task(cfg.dataset, cfg.task, download=True)

    sampler = TemporalSampler(
        data, task.entity_table, max_nodes=cfg.max_nodes,
        strategy="last" if cfg.neighbor_sampling == "recent" else "uniform",
        sibling_fanout=cfg.sibling_fanout, parent_fanout=cfg.parent_fanout,
        relation_cap=cfg.relation_cap, link_tables=link_tables(db),
        collapse_links=cfg.collapse_links)
    renderer = RowRenderer(
        db, text_mode=cfg.text_mode, time_encoding=cfg.time_encoding,
        anonymize=cfg.anonymize, max_value_chars=cfg.max_value_chars,
        max_node_chars=cfg.max_node_chars, null_threshold=cfg.null_threshold)
    return task, sampler, renderer, task_description(task)


DATASETS_DIR_GRAPHS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "processed_data")


def run_data_prep_mode(cfg):
    """Build and cache every split for this config. Idempotent."""
    from transformers import AutoTokenizer

    missing = [s for s in SPLITS if not _is_built(cfg, s)]
    if not missing:
        print(f"[data_prep] already built at {cfg.dataset_dir()} -- nothing to do.")
        return

    print(f"[data_prep] {cfg.dataset}/{cfg.task} arm={cfg.arm()} max_nodes={cfg.max_nodes} "
          f"-> {cfg.dataset_dir()}")
    print(f"[data_prep] building split(s): {missing}")

    _seed_everything(cfg.data_seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    prefix_ids = answer_prefix_ids(tokenizer)
    print(f"[data_prep] answer prefix {ANSWER_PREFIX!r} -> token ids {prefix_ids}")

    task, sampler, renderer, description = make_builders(cfg)
    print(f"[data_prep] question template: {description}")

    counts = {}
    for split in missing:
        table = task.get_table(split, mask_input_cols=False)
        df = table.df
        # Train may be strided down for smoke runs; val and test never are, because their
        # row order is the metric's index.
        stride = _stride_for(cfg.max_train_samples if split == "train" else cfg.max_val_samples,
                             len(df)) if split != "test" else 1
        versions = cfg.samples_per_node if split == "train" else 1

        pending, written = [], 0
        os.makedirs(split_dir(cfg, split), exist_ok=True)

        def flush():
            nonlocal pending, written
            first, last = written, written + len(pending)
            path = os.path.join(split_dir(cfg, split), f"{first}-{last}")
            print(f"[data_prep]   writing {split} chunk {first}-{last} ...")
            _save_chunk(cfg, pending, path, tokenizer, prefix_ids, versions)
            written, pending = last, []

        for i in range(len(df)):
            if stride > 1 and i % stride != 0:
                continue
            row = df.iloc[i]
            seed_ts = int(row[task.time_col].timestamp())
            entity = int(row[task.entity_col])
            question = question_text(task, description, entity, seed_ts)
            answer = answer_text(task, row[task.target_col])

            for version in range(versions):
                sampled = sampler.sample(entity, seed_ts, version=version,
                                         identity=(cfg.dataset, cfg.task, split))
                graph = (build_flat_graph(sampled, renderer, seed_ts, question, answer,
                                          text_mode=cfg.text_mode)
                         if cfg.is_flat() else
                         build_graph(sampled, renderer, seed_ts, question, answer,
                                     question_node=cfg.question_node,
                                     prompt_node=cfg.prompt_node,
                                     text_mode=cfg.text_mode))
                # The table row index travels with the graph: `task.evaluate` compares
                # positionally, so predictions must be re-alignable to this exact order.
                graph.graph["row_id"] = i
                pending.append(graph)

            if len(pending) >= CHUNK_SIZE:
                flush()
            if i % 2000 == 0:
                print(f"[data_prep]   {split} {i}/{len(df)}")

        if pending:
            flush()
        counts[split] = written
        print(f"[data_prep] built {split}: {written} graphs -> {split_dir(cfg, split)}")

    _write_manifest(cfg, counts)


# -- loading ------------------------------------------------------------------

def _load_split(cfg, split):
    dataset = None
    for chunk in _chunk_paths(cfg, split):
        ds = TextGraphDataset.load(chunk)
        ds.assign_label(split)
        dataset = ds if dataset is None else dataset + ds
    return dataset


def load_data(cfg):
    """Return ``(train, val, test)``, building nothing."""
    for split in SPLITS:
        if not _is_built(cfg, split):
            raise FileNotFoundError(
                f"Split {split!r} is not built at {split_dir(cfg, split)}. Build it first:\n"
                f"  python3 -m src.experiments.relbench --mode data_prep "
                f"--dataset {cfg.dataset} --task {cfg.task} --max-nodes {cfg.max_nodes}"
                + (" --arm-name flat --no-spd --no-magnetic" if cfg.is_flat() else ""))

    train = _load_split(cfg, "train")
    val = _load_split(cfg, "val")
    test = _load_split(cfg, "test")

    # `task.evaluate` compares the prediction vector to the target table POSITIONALLY, so a
    # reordering of val or test does not raise -- it silently scores predictions against the
    # wrong rows and reports a plausible-looking number. Check it every load.
    # Test is never strided, so its row_ids must be exactly 0..n-1. Val must be too unless
    # this config asked for a cap. Monotonicity alone would not catch a partial split.
    assert_row_order(test, "test", contiguous=True)
    assert_row_order(val, "val", contiguous=cfg.max_val_samples is None)

    print(f"[data] {cfg.dataset}/{cfg.task} [{cfg.arm()}]: train={len(train)} "
          f"val={len(val)} test={len(test)}")
    return train, val, test


def assert_row_order(dataset, split, contiguous=False):
    """Every graph must sit at the index of the task-table row it was built from.

    ``contiguous`` additionally requires the split to be *complete* -- row_ids exactly
    ``0..n-1``. A strided cache is still monotonic, so without this a 114-of-566 val split
    (or worse, a partial test split) loads and scores without complaint.
    """
    rows = [dataset.graphs[i].graph.get("row_id") for i in range(len(dataset))]
    if any(r is None for r in rows):
        raise ValueError(
            f"{split}: graphs are missing `row_id`. The cache predates the ordering "
            f"invariant and must be rebuilt -- predictions cannot be aligned without it.")
    if rows != sorted(rows):
        raise ValueError(
            f"{split}: graphs are not in task-table order (row_id is not monotonic). "
            f"`task.evaluate` compares positionally, so this would score predictions "
            f"against the wrong targets.")
    if contiguous and rows != list(range(len(rows))):
        raise ValueError(
            f"{split}: cache holds {len(rows)} graphs whose row_ids are not 0..{len(rows)-1}, "
            f"so it is a strided or partial build of a split that must be complete. "
            f"Rebuild it without a sample cap -- a subset of {split} is not comparable to "
            f"any published baseline.")
