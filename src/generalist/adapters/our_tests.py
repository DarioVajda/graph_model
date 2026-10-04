"""
D3 for ``our_tests`` — ``our_tests/kg_qa`` and the held-out ``our_tests/family_tree``.

`GRAPH_GENERALIST.md` §2: the synthetic organisational knowledge graph is in the
mixture; Family Tree is held out from all training (`PLAN.md` §3.3).

**kg_qa.** `experiments/our_tests/kgqa_gen.py` draws a graph of people,
projects and resources (30–50 entities, the specialist's range) and asks six
multi-hop yes/no questions about it, balanced by resampling. The graph enters in
the specialist's incidence form (`kgqa_prep.create_incidence_graph`: entity node
→ relation node → entity node) and the prompt node points at the entities the
question names. One graph gives six rows that share its key, so a graph is never
split across roles. Sizes are the specialist's 500 / 30 / 150 graphs, six rows
each.

**family_tree.** `family_gen.generate_family_tree` (3 generations, marriage 0.7,
child 0.75) and `generate_qa_pair`, which asks for a relative's favourite
colour, food or city; a tree with no unambiguous question is redrawn. Node text
and the spouse/child incidence form are `family_prep.prepare_graph`'s. The answer
is one word and is scored as a ``span``. 1,000 held-out rows, the specialist's
test size.

Both modules seed the global generators at import; `_graph._Stream` imports
before seeding, so that cannot reach a stream.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._graph import (Draw, GraphAdapterConfig, GraphDomain, TaskInfo, code_digest,
                     graph_key, split_prompt)

DOMAIN = "our_tests"
PREFIX = "our_tests/"
ADAPTER_VERSION = "1"

KGQA_QUESTIONS = 6
KGQA_GRAPHS = {"train": 500, "val": 30, "test": 150}
FAMILY_ROWS = {"held_out": 1000}


@dataclass
class OurTestsAdapterConfig(GraphAdapterConfig):
    max_length: int = 256
    kgqa_min_nodes: int = 30
    kgqa_max_nodes: int = 50
    family_generations: int = 3
    family_marriage_prob: float = 0.7
    family_child_prob: float = 0.75


def _tasks() -> tuple:
    return (
        TaskInfo(name="kg_qa", answer_kind="yesno", kind="generator",
                 metric="roc_auc", chunk=3000,
                 sizes={s: n * KGQA_QUESTIONS for s, n in KGQA_GRAPHS.items()}),
        TaskInfo(name="family_tree", answer_kind="span", kind="generator",
                 metric="em_accuracy", held_out=True, sizes=dict(FAMILY_ROWS),
                 chunk=1000),
    )


def _draws(config, info: TaskInfo, split: str, pass_id: int):
    if info.name == "kg_qa":
        from ...experiments.our_tests.kgqa_gen import KnowledgeGraphGenerator
        from ...experiments.our_tests.kgqa_prep import create_incidence_graph

        return _kg_qa(config, KnowledgeGraphGenerator, create_incidence_graph)
    from ...experiments.our_tests import family_gen, family_prep

    return _family(config, family_gen, family_prep)


def _kg_qa(config, generator_class, create_incidence_graph):
    generator = None
    while True:
        if generator is None:
            # Constructed inside the stream, so its name pools are the seeded ones.
            generator = generator_class()
        (graph,) = generator.generate(1, min_nodes=config.kgqa_min_nodes,
                                      max_nodes=config.kgqa_max_nodes, train=True)
        incidence = create_incidence_graph(graph)
        questions = incidence.graph.pop("questions")
        incidence.graph = {}
        key = graph_key(incidence)
        for func, (question, pointers, answer) in questions.items():
            if answer not in ("Yes", "No"):
                raise ValueError(f"our_tests/kg_qa: {func} answered {answer!r}")
            yield Draw(graph=incidence, question=question, answer=f" {answer}",
                       targets=tuple(pointers), key=key, meta={"question_type": func})


def _family(config, family_gen, family_prep):
    while True:
        tree = family_gen.generate_family_tree(
            generations=config.family_generations,
            marriage_prob=config.family_marriage_prob,
            child_prob=config.family_child_prob)
        person, question, answer = family_gen.generate_qa_pair(tree)
        if person is None:
            continue
        prepared = family_prep.prepare_graph(tree, person, question, answer,
                                             question_node="off")
        content, _text, targets = split_prompt(prepared)
        yield Draw(graph=content, question=question, answer=str(answer),
                   targets=targets, key=graph_key(content, extra=question))


def _source_digests(config) -> dict:
    return code_digest("our_tests/kgqa_gen.py", "our_tests/kgqa_prep.py",
                       "our_tests/family_gen.py", "our_tests/family_prep.py")


DOMAIN_SPEC = GraphDomain(
    name=DOMAIN, prefix=PREFIX, adapter_version=ADAPTER_VERSION, tasks=_tasks(),
    config_class=OurTestsAdapterConfig, draws=_draws,
    source_digests=_source_digests)

build = DOMAIN_SPEC.build
load = DOMAIN_SPEC.load
partition = DOMAIN_SPEC.partition
task_specs = DOMAIN_SPEC.task_specs
register = DOMAIN_SPEC.register
