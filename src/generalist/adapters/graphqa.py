"""
D3 for GraphQA — ``graphqa/<task>``, the algorithmic questions of `baharef/GraphQA`.

`GRAPH_GENERALIST.md` §2: the nine reported tasks minus the held-out pair
(``triangle_counting``, ``connected_nodes``), plus ``disconnected_nodes`` and
``node_classification`` as train-only — those two ship no official validation
file, and carving one out of train would put a split in the trunk that the
specialist never measured on. ``maximum_flow`` is unbuildable (the released files
omit the capacities) and is not registered.

Graphs come from `experiments/graphqa/process_dataset.py`: the edge list is
parsed back out of the raw ``question`` text, node text is the vertex number
(``"7 likes golf"`` for node classification), the graph is undirected, stored
symmetric. Prompt edges are that module's `extract_prompt_edges`: the nodes the
question names, or every node for a graph-level question. The ``standard``
encoding, which is the one every specialist row was reported on.

**The key is the graph, not the row.** GraphQA reuses its graphs across tasks
and splits: measured over the zero-shot files, a task's 1,000 train rows hold
~967 distinct graphs, and 38 of them also sit in that task's test split (13 in
val). One graph in two roles is the leak the partition exists to stop, so the key
is the graph's node and edge strings, the claims run across *every* task's
files, and a train row whose graph any test or val file holds is dropped and
counted in the ledger (``test > val > train``).

**The held-out pair claims ``test``, not ``held_out``.** Their ``held_out`` split
is the official test file, and those graphs are other tasks' test graphs too; a
``held_out`` claim would outrank ``test`` and pull them out of every in-mixture
task's test split, shrinking the anchors the trunk is compared on. They are
``test`` role, which keeps them out of every train split all the same.

**Answers are ``span``**: ``"8."``, ``"Yes, there is a cycle."``, ``"0, 1, 3, 5."``,
scored teacher-forced by exact match over the supervised span, which is the
specialist's protocol. The raw answers carry an inconsistent leading space
(``" 14."`` on the count tasks, none elsewhere); it is stripped, and the format's
own space is the only one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ._graph import (Draw, GraphAdapterConfig, GraphDomain, TaskInfo,
                     experiment_path, file_digest, _hash)
from ._partition import Claim

DOMAIN = "graphqa"
PREFIX = "graphqa/"
ADAPTER_VERSION = "1"

IN_MIXTURE = ("node_count", "edge_count", "cycle_check", "node_degree",
              "reachability", "edge_existence", "shortest_path")
TRAIN_ONLY = ("disconnected_nodes", "node_classification")
HELD_OUT = ("triangle_counting", "connected_nodes")

#: The raw files' split names.
RAW_SPLIT = {"train": "train", "val": "validation", "test": "test",
             "held_out": "test"}


@dataclass
class GraphQAAdapterConfig(GraphAdapterConfig):
    max_length: int = 512


def _tasks() -> tuple:
    out = [TaskInfo(name=t, answer_kind="span", kind="corpus", metric="em_accuracy")
           for t in IN_MIXTURE]
    out += [TaskInfo(name=t, answer_kind="span", kind="corpus", metric="em_accuracy",
                     splits=("train",), eval_splits=()) for t in TRAIN_ONLY]
    out += [TaskInfo(name=t, answer_kind="span", kind="corpus", metric="em_accuracy",
                     held_out=True) for t in HELD_OUT]
    return tuple(out)


def _raw_path(task: str, split: str) -> str:
    """`graphqa/process_dataset.py::raw_split_file`, spelled out: that module
    imports torch, and the build version needs this path on the login node."""
    return experiment_path("graphqa", "hf_dataset", task,
                           f"{task}_zero_shot_{RAW_SPLIT[split]}.json")


def _rows(task: str, split: str) -> list:
    with open(_raw_path(task, split)) as fh:
        return json.load(fh)


def row_key(row: dict) -> str:
    """The graph's identity: its node list and its edge list, as the raw text has them."""
    from ...experiments.graphqa.process_dataset import extract_graph_data

    nodes, edges = extract_graph_data(row["question"])
    return _hash({"nodes": sorted(nodes),
                  "edges": sorted(tuple(sorted(e)) for e in edges)})


def _question(row: dict) -> str:
    """The question without the specialist's ``"Q: "`` / ``"\\nA: "`` frame."""
    from ...experiments.graphqa.process_dataset import ANSWER_PREFIX

    head, sep, _tail = row["task_description"].partition(ANSWER_PREFIX)
    if not sep:
        raise ValueError(f"task_description has no {ANSWER_PREFIX!r}: "
                         f"{row['task_description']!r}")
    head = head.strip()
    return head[3:] if head.startswith("Q: ") else head


def _draws(config, info: TaskInfo, split: str, pass_id: int):
    import networkx as nx

    from ...experiments.graphqa.process_dataset import (extract_graph_data,
                                                        extract_node_preferences,
                                                        extract_prompt_edges)

    for index, row in enumerate(_rows(info.name, split)):
        nodes, edges = extract_graph_data(row["question"])
        if len(nodes) != int(row["nnodes"]) or len(edges) != int(row["nedges"]):
            raise ValueError(f"graphqa/{info.name}/{split} row {index}: parsed "
                             f"{len(nodes)} nodes / {len(edges)} edges against "
                             f"{row['nnodes']} / {row['nedges']}")
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        for node in graph.nodes:
            graph.nodes[node]["text"] = f"{node}"
        if info.name == "node_classification":
            for node, preference in extract_node_preferences(row).items():
                if node in graph:
                    graph.nodes[node]["text"] = f"{node} likes {preference}"
        targets = tuple(extract_prompt_edges(row, nodes, None, info.name))
        yield Draw(graph=graph, question=_question(row),
                   answer=str(row["answer"]).strip(), targets=targets,
                   key=row_key(row), meta={"row": index})


def _claims(config, infos) -> list:
    """Every task's every built split, at the role its keys must hold."""
    claims = []
    for info in infos:
        for split in info.built_splits():
            role = _role_of_split(info, split)
            keys = tuple(sorted({row_key(r) for r in _rows(info.name, split)}))
            claims.append(Claim(f"{info.name}/{split}", role, keys))
    return claims


def _role_of_split(info: TaskInfo, split: str) -> str:
    return "test" if split == "held_out" else split


def _source_digests(config) -> dict:
    out = {}
    for info in DOMAIN_SPEC.tasks:
        for split in info.built_splits():
            out[f"{info.name}/{split}"] = file_digest(_raw_path(info.name, split))
    return out


DOMAIN_SPEC = GraphDomain(
    name=DOMAIN, prefix=PREFIX, adapter_version=ADAPTER_VERSION, tasks=_tasks(),
    config_class=GraphQAAdapterConfig, draws=_draws,
    source_digests=_source_digests, claims=_claims, role_of_split=_role_of_split)

build = DOMAIN_SPEC.build
load = DOMAIN_SPEC.load
partition = DOMAIN_SPEC.partition
task_specs = DOMAIN_SPEC.task_specs
register = DOMAIN_SPEC.register
