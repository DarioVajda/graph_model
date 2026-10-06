"""
D3 for KGQA — ``kgqa/webqsp`` and ``kgqa/cwq``, SR-retrieved Freebase subgraphs.

`GRAPH_GENERALIST.md` §2: both benchmarks, **Levi construction only** (the
triplet graph needs 256–512 GB of RAM to build and still lost to flat). Graphs
are `experiments/kgqa/process_dataset.py::build_base_levi` with the specialist's
settings: ``rel_mode last_1``, single-parent CVT collapse, naming v2
(``entities_names.json``), ``max_nodes`` 512 for WebQSP and 1,024 for CWQ. One
node of that budget goes to the question node, as the specialist's
``question_node: isolated`` arm reserved it. The prompt node points at the topic
entities.

**The target is the graph-present answers**, shuffled and capped at ``n_max``
(20), one per line — the data-format-v3 separator, GNN-RAG's own. Train keeps only
questions with at least one present answer and writes ``versions`` rows per
question (8 for WebQSP, 1 for CWQ), each a fresh answer order; the rows share the
question's key. Dev and test keep every question with a gold answer, because
GNN-RAG's denominators do: an unanswerable row (no gold entity in the subgraph)
carries the first ``n_max`` gold names as its target so the schema has a span to
hold, and ``meta["unanswerable"]`` says so. That row's target is never trained
on (it is an eval split) and is not what it is scored against. It is capped like
every other target because some WebQSP test questions list hundreds of gold
entities, and the uncapped list ran the prompt node past the 1,024-token cap.

**Scoring is ``entities``**: generated, split on ``"\\n"``, and scored against
``meta["gold"]`` — the full gold list, `full_gold_texts` — with GNN-RAG's F1 and
Hits@1 (`experiments/kgqa/evaluate.py`). Generation budget 128 tokens for
WebQSP and 256 for CWQ.

**Keys and the CWQ/WebQSP overlap.** The key is the question id. CWQ was
written by extending WebQSP questions, and its ids carry the seed
(``WebQTrn-1430_ac05…``). Measured over the SR files: **8,581 CWQ train rows
(923 distinct seeds) are built on WebQSP *test* questions**, 1,752 on WebQSP
dev; in the other direction 193 WebQSP train questions seed CWQ test rows and
226 seed CWQ dev rows. Under the default (``strict_cross_dataset: false``) each
benchmark keeps its own official splits, which is what every published number
on either was trained under — but a trunk trained on both sees an extension of
roughly 56 % of WebQSP's test questions, so a WebQSP number from such a trunk
carries that caveat. ``strict_cross_dataset: true`` keys both by the WebQSP seed
and lets the partition's ``test > val > train`` priority remove every crossing,
at the cost of ~37 % of CWQ train (10,333 rows) and up to ~15 % of WebQSP
train (419 questions); the ledger records exactly what moved.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from ._graph import (Draw, GraphAdapterConfig, GraphDomain, TaskInfo,
                     experiment_path, file_digest)
from ._partition import Claim

DOMAIN = "kgqa"
PREFIX = "kgqa/"
ADAPTER_VERSION = "2"

DATASETS = ("webqsp", "cwq")
#: The SR files' split names.
RAW_SPLIT = {"train": "train", "val": "dev", "test": "test"}
MAX_NODES = {"webqsp": 512, "cwq": 1024}
VERSIONS = {"webqsp": 8, "cwq": 1}
MAX_NEW_TOKENS = {"webqsp": 128, "cwq": 256}
ANSWER_SEP = "\n"
MAGNETIC_M_CAP = 128

_ID = re.compile(rb'^\{"id": "([^"]+)"')


@dataclass
class KGQAAdapterConfig(GraphAdapterConfig):
    max_length: int = 1024
    rel_mode: str = "last_1"
    n_max: int = 20
    entity_names_file: str = "entities_names.json"
    strict_cross_dataset: bool = False


def _tasks() -> tuple:
    return tuple(
        TaskInfo(name=ds, answer_kind="entities", kind="corpus", metric="f1",
                 max_new_tokens=MAX_NEW_TOKENS[ds], magnetic_m_cap=MAGNETIC_M_CAP,
                 chunk=2000)
        for ds in DATASETS)


def _raw_path(dataset: str, split: str) -> str:
    return experiment_path("kgqa", "data", f"sr-{dataset}", f"{RAW_SPLIT[split]}.json")


def question_key(config, dataset: str, qid: str) -> str:
    if config.strict_cross_dataset:
        return f"webq:{qid.split('_')[0]}"
    return f"{dataset}:{qid}"


def _ids(dataset: str, split: str) -> list:
    """Question ids in file order, read off each line's prefix — CWQ's train file
    is 570 MB and the partition needs only this."""
    out = []
    with open(_raw_path(dataset, split), "rb") as fh:
        for n, line in enumerate(fh):
            match = _ID.match(line)
            if not match:
                raise ValueError(f"{_raw_path(dataset, split)} line {n}: no leading id")
            out.append(match.group(1).decode())
    return out


def _claims(config, infos) -> list:
    """Both datasets always, whichever is built: under the strict key a CWQ test
    question removes its WebQSP seed from WebQSP train."""
    claims = []
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            keys = {question_key(config, dataset, q) for q in _ids(dataset, split)}
            claims.append(Claim(f"{dataset}/{split}", split, tuple(sorted(keys))))
    return claims


_NAMES: dict = {}


def _entity_names(config) -> dict:
    path = experiment_path("kgqa", config.entity_names_file)
    if path not in _NAMES:
        with open(path) as fh:
            _NAMES[path] = json.load(fh)
    return _NAMES[path]


def _records(dataset: str, split: str):
    """`sr_records.load_sr_records`, streamed: one record decoded at a time."""
    from ...experiments.kgqa import sr_records

    id2mid = sr_records.load_cwq_id2mid() if dataset == "cwq" else None
    with open(_raw_path(dataset, split)) as fh:
        for line in fh:
            record = json.loads(line)
            yield (sr_records.normalize_cwq_record(record, id2mid)
                   if id2mid is not None else record)


def _draws(config, info: TaskInfo, split: str, pass_id: int):
    from ...experiments.kgqa.process_dataset import (build_base_levi,
                                                     full_gold_texts,
                                                     present_answer_texts)

    dataset = info.name
    names = _entity_names(config)
    rng = random.Random(f"{config.data_seed}|{dataset}|{split}|{pass_id}")
    versions = VERSIONS[dataset] if split == "train" else 1
    for record in _records(dataset, split):
        if not record.get("answers"):
            continue
        gold = full_gold_texts(record)
        if not gold:
            continue
        # The question node takes one slot of the node budget.
        base = build_base_levi(record, names, config.rel_mode,
                               MAX_NODES[dataset] - 1, cvt_collapse=True)
        present = present_answer_texts(base, record)
        targets = tuple(tp for tp in record["entities"] if tp in base)
        key = question_key(config, dataset, record["id"])
        meta = {"id": record["id"], "gold": gold, "unanswerable": not present}
        if not present:
            if split == "train":
                continue
            yield Draw(graph=base, question=record["question"],
                       answer=ANSWER_SEP.join(gold[: config.n_max]),
                       targets=targets, key=key,
                       meta=meta)
            continue
        for version in range(versions):
            order = present[:]
            rng.shuffle(order)
            yield Draw(graph=base, question=record["question"],
                       answer=ANSWER_SEP.join(order[: config.n_max]),
                       targets=targets, key=key, meta=dict(meta, version=version))


def _source_digests(config) -> dict:
    out = {f"{ds}/{split}": file_digest(_raw_path(ds, split))
           for ds in DATASETS for split in ("train", "val", "test")}
    out["entity_names"] = file_digest(experiment_path("kgqa", config.entity_names_file))
    out.update({p: file_digest(experiment_path(p)) for p in (
        "kgqa/process_dataset.py", "kgqa/sr_records.py")})
    return out


DOMAIN_SPEC = GraphDomain(
    name=DOMAIN, prefix=PREFIX, adapter_version=ADAPTER_VERSION, tasks=_tasks(),
    config_class=KGQAAdapterConfig, draws=_draws, source_digests=_source_digests,
    claims=_claims)

build = DOMAIN_SPEC.build
load = DOMAIN_SPEC.load
partition = DOMAIN_SPEC.partition
task_specs = DOMAIN_SPEC.task_specs
register = DOMAIN_SPEC.register
