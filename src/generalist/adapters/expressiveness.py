"""
D3 for expressiveness — ``expressiveness/hard``, connectivity across graph sizes.

`GRAPH_GENERALIST.md` §2: the HARD generator, and the trunk's widest range of
graph sizes. ``generate_hard_graph`` builds 2–10 connected components and asks
whether two nodes share one; answers are balanced. Node text is a spreadsheet
label (``" A"``, ``" AB"``) from `make_node_labels`, the graph is stored
symmetric, and the prompt node points at the two queried nodes —
`expressiveness/data/data_gen.py::prepare_dataset`'s construction, with the
question moved into its own node.

**Sizes are log-uniform over 10–1,000 nodes, not the specialist's 1,600–2,400.**
The trunk wants connectivity at every scale rather than at one large one, and
log-uniform gives every doubling of size the same share — 10–20 nodes as often
as 500–1,000. Uniform would put half the graphs above 500 nodes, and since the
bias is dense in node pairs, those would set both the compute and the peak
memory. At 1,600–2,400 nodes one row's ``(N, N, m)`` magnetic hidden state ran an
80 GB card out of memory on the second step of the smoke run; at 1,000 the worst
row is 5.8x smaller in N², and the mean N² is 37x smaller (1.1e5 against 4.1e6).

**magnetic ``m`` is capped at 128**, as the specialist ran it: at ``m = N`` the
bias einsums are O(N³) a layer and dominate the step, and 128 eigenvectors still
exceed the component count by an order of magnitude.

Train is 1,000 graphs a pass — a generator, so a pass is a fresh draw — with 100
val and 200 test.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._graph import (Draw, GraphAdapterConfig, GraphDomain, TaskInfo, code_digest,
                     graph_key)

DOMAIN = "expressiveness"
PREFIX = "expressiveness/"
ADAPTER_VERSION = "2"

MIN_NODES = 10
MAX_NODES = 1000
MAGNETIC_M_CAP = 128
SIZES = {"train": 1000, "val": 100, "test": 200}


@dataclass
class ExpressivenessAdapterConfig(GraphAdapterConfig):
    max_length: int = 64
    min_nodes: int = MIN_NODES
    max_nodes: int = MAX_NODES


def _tasks() -> tuple:
    return (TaskInfo(name="hard", answer_kind="yesno", kind="generator",
                     metric="roc_auc", sizes=dict(SIZES),
                     magnetic_m_cap=MAGNETIC_M_CAP, chunk=500),)


def log_uniform_size(min_nodes: int, max_nodes: int) -> int:
    """A node count drawn log-uniformly from ``[min_nodes, max_nodes]``, from the
    global ``random`` stream the generator's own draws use."""
    import math
    import random

    if not 4 <= min_nodes <= max_nodes:
        raise ValueError(f"need 4 <= min_nodes <= max_nodes, got {min_nodes}, "
                         f"{max_nodes}: two components of two nodes is the smallest "
                         "graph the generator can build")
    value = math.exp(random.uniform(math.log(min_nodes), math.log(max_nodes + 1)))
    return min(max_nodes, int(value))


def _draws(config, info: TaskInfo, split: str, pass_id: int):
    from ...experiments.expressiveness.data.data_gen import (generate_hard_graph,
                                                             make_node_labels)

    while True:
        graph, x, y, label = generate_hard_graph(
            size=log_uniform_size(config.min_nodes, config.max_nodes), balanced=True)
        for node, text in zip(graph.nodes, make_node_labels(graph.number_of_nodes())):
            graph.nodes[node]["text"] = text
        question = (f"Are the nodes{graph.nodes[x]['text']} and"
                    f"{graph.nodes[y]['text']} connected?")
        yield Draw(graph=graph, question=question,
                   answer=" Yes" if label == 1 else " No", targets=(x, y),
                   key=graph_key(graph, extra=question))


def _source_digests(config) -> dict:
    return code_digest("expressiveness/data/data_gen.py")


DOMAIN_SPEC = GraphDomain(
    name=DOMAIN, prefix=PREFIX, adapter_version=ADAPTER_VERSION, tasks=_tasks(),
    config_class=ExpressivenessAdapterConfig, draws=_draws,
    source_digests=_source_digests)

build = DOMAIN_SPEC.build
load = DOMAIN_SPEC.load
partition = DOMAIN_SPEC.partition
task_specs = DOMAIN_SPEC.task_specs
register = DOMAIN_SPEC.register
