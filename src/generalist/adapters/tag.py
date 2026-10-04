"""
D3 for the TAG benchmarks — ``tag/<dataset>``, node classification on text graphs.

`GRAPH_GENERALIST.md` §2: cora, ogbn-arxiv and reddit in the mixture; pubmed
held out. One example per node: its ``hops``-hop neighbourhood, capped at
``max_neighbors`` nodes with nearer nodes kept first, sampled by
`experiments/tag_benchmarks/data.py::_sample_neighborhood` under the settings of
the specialist's reproduction (`configs/002_paper_tag_repro.jsonc`): 2 hops;
60 nodes for cora and ogbn-arxiv, 30 for reddit and pubmed; the target's
abstract and its neighbours' titles, except reddit, whose posts are cut to 128
characters. One sample per node (``samples_per_node: 1``) — the trunk sees a
corpus pass by pass, and resampling neighbourhoods inside a pass would be an
augmentation no other domain gets.

**Where the target sits.** The specialist glued the instruction and the label
onto the target node's own text, which made the target the prompt node. Here
the target keeps only its own text, the question is the isolated question node
as in every other domain, and the prompt node points at the target. The target
is one hop from the answer rather than being it; that is the price of one format
across domains, and it is the same price GraphQA's node-level tasks pay.

**Splits are the benchmark's own masks**, keyed ``<dataset>:<node>``. Pubmed's
``held_out`` split is its test mask, the split its specialist number is on.

**Answers are ``span``** — the class name, `label_texts[y]` — scored teacher-
forced by exact match, with macro-F1 beside it as the specialist reported.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._graph import (Draw, GraphAdapterConfig, GraphDomain, TaskInfo,
                     experiment_path, file_digest)
from ._partition import Claim

DOMAIN = "tag"
PREFIX = "tag/"
ADAPTER_VERSION = "1"

IN_MIXTURE = ("cora", "ogbn-arxiv", "reddit")
HELD_OUT = ("pubmed",)

#: ``(max_neighbors, text_mapping, text_mapping_param)`` per dataset.
SETTINGS = {
    "cora": (60, "target_abstract", None),
    "ogbn-arxiv": (60, "target_abstract", None),
    "reddit": (30, "truncated_text", 128),
    "pubmed": (30, "target_abstract", None),
}
HOPS = 2
MASKS = {"train": "train_mask", "val": "val_mask", "test": "test_mask",
         "held_out": "test_mask"}


@dataclass
class TAGAdapterConfig(GraphAdapterConfig):
    max_length: int = 1024


def _tasks() -> tuple:
    out = [TaskInfo(name=ds, answer_kind="span", kind="corpus", metric="em_accuracy",
                    chunk=5000) for ds in IN_MIXTURE]
    out += [TaskInfo(name=ds, answer_kind="span", kind="corpus", metric="em_accuracy",
                     held_out=True, chunk=5000) for ds in HELD_OUT]
    return tuple(out)


def _raw_path(dataset: str) -> str:
    return experiment_path("tag_benchmarks", "raw_data", dataset, "processed_data.pt")


_RAW: dict = {}


class _Inert:
    """Stands in for a `torch_sparse` object while a raw file is unpickled."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        self.state = state


class _NoSparseUnpickler(__import__("pickle").Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "torch_sparse":
            return _Inert
        return super().find_class(module, name)


class _NoSparsePickle:
    """A ``pickle_module`` for `torch.load` that never imports `torch_sparse`.

    ogbn-arxiv's processed file carries an ``adj_t`` SparseTensor beside its
    ``edge_index``, and unpickling it imports `torch_sparse`, whose compiled
    extension refuses to load against a torch built for another CUDA. The sampler
    reads ``edge_index`` only, so ``adj_t`` comes back inert and is dropped.
    """

    Unpickler = _NoSparseUnpickler

    def __getattr__(self, name):
        return getattr(__import__("pickle"), name)


def _raw(dataset: str):
    import torch

    if dataset not in _RAW:
        data = torch.load(_raw_path(dataset), weights_only=False,
                          pickle_module=_NoSparsePickle())
        for key in list(data.keys()):
            if isinstance(data[key], _Inert):
                del data[key]
        _RAW[dataset] = data
    return _RAW[dataset]


def _split_nodes(dataset: str, split: str) -> list:
    mask = getattr(_raw(dataset), MASKS[split])
    return [int(i) for i in mask.nonzero().flatten().tolist()]


def _claims(config, infos) -> list:
    claims = []
    for info in infos:
        splits = ("held_out",) if info.held_out else ("train", "val", "test")
        for split in splits:
            keys = tuple(f"{info.name}:{n}" for n in _split_nodes(info.name, split))
            claims.append(Claim(f"{info.name}/{split}", split, keys))
    return claims


def _question(dataset: str) -> str:
    from ...experiments.tag_benchmarks.config import ANSWER_PREFIX, INSTRUCTIONS

    text = INSTRUCTIONS[dataset]
    if not text.endswith("\n" + ANSWER_PREFIX):
        raise ValueError(f"tag/{dataset}: instruction does not end in the answer prefix")
    text = text[: -len("\n" + ANSWER_PREFIX)]
    return text[3:] if text.startswith("Q: ") else text


def _draws(config, info: TaskInfo, split: str, pass_id: int):
    import networkx as nx
    from torch_geometric.utils import k_hop_subgraph

    from ...experiments.tag_benchmarks.config import RunConfig as TAGConfig
    from ...experiments.tag_benchmarks.data import (_sample_neighborhood,
                                                    make_text_mapping)

    dataset = info.name
    max_neighbors, mapping_name, param = SETTINGS[dataset]
    mapping = make_text_mapping(TAGConfig(dataset=dataset, text_mapping=mapping_name,
                                          text_mapping_param=param))
    data = _raw(dataset)
    question = _question(dataset)
    for node in _split_nodes(dataset, split):
        subset, edge_index, _, _ = k_hop_subgraph(node, HOPS, data.edge_index)
        hop_graph = nx.Graph()
        hop_graph.add_edges_from(edge_index.t().tolist())
        hop_graph.add_node(node)
        distances = nx.single_source_shortest_path_length(hop_graph, node)
        # The specialist's sampler, with no instruction: it then leaves exactly
        # "\n" + label on the target, which comes back off here.
        graph = _sample_neighborhood(data, node, subset, edge_index, distances,
                                     max_neighbors, mapping, "")
        label = data.label_texts[int(data.y[node])]
        tail = "\n" + label
        text = graph.nodes[node]["text"]
        if not text.endswith(tail):
            raise ValueError(f"tag/{dataset}: node {node}'s text does not end in its label")
        graph.nodes[node]["text"] = text[: -len(tail)]
        graph.graph = {}
        yield Draw(graph=graph, question=question, answer=str(label).strip(),
                   targets=(node,), key=f"{dataset}:{node}",
                   meta={"node": node, "graph_nodes": graph.number_of_nodes()})


def _source_digests(config) -> dict:
    out = {ds: file_digest(_raw_path(ds)) for ds in SETTINGS}
    out.update({p: file_digest(experiment_path(p)) for p in (
        "tag_benchmarks/data.py", "tag_benchmarks/config.py")})
    return out


DOMAIN_SPEC = GraphDomain(
    name=DOMAIN, prefix=PREFIX, adapter_version=ADAPTER_VERSION, tasks=_tasks(),
    config_class=TAGAdapterConfig, draws=_draws, source_digests=_source_digests,
    claims=_claims)

build = DOMAIN_SPEC.build
load = DOMAIN_SPEC.load
partition = DOMAIN_SPEC.partition
task_specs = DOMAIN_SPEC.task_specs
register = DOMAIN_SPEC.register
