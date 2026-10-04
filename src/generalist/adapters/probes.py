"""
D3 for the probe suite — ``probes/<task>``, balanced yes/no structure probes.

`GRAPH_GENERALIST.md` §2: ``substructure``, ``local_hop`` and ``text_path`` in
the mixture; ``direction`` is held out (`PLAN.md` §3.3). Every graph is drawn by
the specialist's own generator in `experiments/probes/data.py`, at its README
node range and knobs (`experiments/probes/config.py::RunConfig` defaults), so a
trunk row is a specialist row with the question moved into its own node.

The specialist generator writes the question and the answer into one prompt node
(``"Is the node A part of a ring? Yes"``); `_graph.split_prompt` takes that node
back off, keeping its edges as the prompt targets, and the question is the
node's text without the trailing label word. The answer is ``" Yes"`` /
``" No"``, scored by the logit margin like every other ``yesno`` task.

A generator: train pass *p* is a fresh draw. Sizes are the specialist's
4,000 / 500 / 1,000, and the held-out probe draws 1,000.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._graph import (Draw, GraphAdapterConfig, GraphDomain, TaskInfo, code_digest,
                     graph_key, split_prompt)

DOMAIN = "probes"
PREFIX = "probes/"
ADAPTER_VERSION = "1"

IN_MIXTURE = ("substructure", "local_hop", "text_path")
HELD_OUT = ("direction",)

SIZES = {"train": 4000, "val": 500, "test": 1000, "held_out": 1000}

#: text_path bios run ~40-60 tokens a node; nothing here comes near the cap.
MAX_LENGTH = 512


@dataclass
class ProbesAdapterConfig(GraphAdapterConfig):
    max_length: int = MAX_LENGTH


def _tasks() -> tuple:
    out = [TaskInfo(name=t, answer_kind="yesno", kind="generator", metric="roc_auc",
                    sizes=dict(SIZES), chunk=2000) for t in IN_MIXTURE]
    out += [TaskInfo(name=t, answer_kind="yesno", kind="generator", metric="roc_auc",
                     held_out=True, sizes=dict(SIZES), chunk=2000) for t in HELD_OUT]
    return tuple(out)


def _draws(config, info: TaskInfo, split: str, pass_id: int):
    from ...experiments.probes.config import RunConfig as ProbeConfig
    from ...experiments.probes.data import TASK_GENERATORS

    probe = ProbeConfig(task=info.name)
    generator = TASK_GENERATORS[info.name]
    stats: dict = {}
    while True:
        graph, label = generator(probe, stats)
        content, text, targets = split_prompt(graph)
        word = " Yes" if label else " No"
        if not text.endswith(word):
            raise ValueError(f"probes/{info.name}: prompt {text!r} does not end in {word!r}")
        question = text[: -len(word)]
        yield Draw(graph=content, question=question, answer=word, targets=targets,
                   key=graph_key(content, extra=question))


def _source_digests(config) -> dict:
    return code_digest("probes/data.py", "probes/config.py",
                       "expressiveness/data/data_gen.py")


DOMAIN_SPEC = GraphDomain(
    name=DOMAIN, prefix=PREFIX, adapter_version=ADAPTER_VERSION, tasks=_tasks(),
    config_class=ProbesAdapterConfig, draws=_draws, source_digests=_source_digests)

build = DOMAIN_SPEC.build
load = DOMAIN_SPEC.load
partition = DOMAIN_SPEC.partition
task_specs = DOMAIN_SPEC.task_specs
register = DOMAIN_SPEC.register
