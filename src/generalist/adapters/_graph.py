"""
D3 for the trunk's graph domains — one build path under six adapters.

`GRAPH_GENERALIST.md` §2 brings six sources into the mixture beside molecules:
GraphQA, the probe suite, expressiveness, ``our_tests``, KGQA and the TAG
benchmarks. Each already has a specialist pipeline that builds its own graphs,
and each of those pipelines does the same five things after the graph exists:
attach the question, tokenize, find the supervised span, compute SPD and the
magnetic Laplacian, save. Six copies of that tail would drift, and the drift
would land exactly where an arm comparison is decided. So this module owns the
tail, and an adapter owns only what is genuinely its own: **which graphs, with
which question, answer and prompt-node targets, under which key**.

**One format for every domain.** The specialists each spelled the prompt their
own way (``"Answer:"``, ``"A: "``, the answer glued onto the target node's
text). A trunk has one: the content nodes as the source builds them, an
edge-free question node holding ``fmt.question(question)``
(``question_node: isolated``, the settled default), and a prompt node holding
``fmt.answer_prefix`` plus the answer, wired by directed edges to the nodes the
question is about. That is `molecules/data.py::attach_question` exactly, so a
molecule row and a GraphQA row differ in their graph and nothing else. The
answer keeps one leading space, like the molecule answers, so ``" Yes"`` is the
same token id in every domain and the margin readout needs no per-domain case.

**Graph arm only.** None of these sources has a matched flat twin in this build;
the specialists' flat arms serialise differently per domain and none was
designed against this format. A build asked for any other arm is refused rather
than quietly producing a graph build under a flat name.

**What a domain supplies** is a :class:`GraphDomain`: its tasks as
:class:`TaskInfo` rows, a ``draws`` callable yielding :class:`Draw` objects, the
digests of its raw inputs, and — for a corpus — the partition claims. The draws
are a lazy iterator consumed in chunks, so a source whose graphs are large
(expressiveness at 2,400 nodes, CWQ subgraphs) never holds more than one chunk
of featurised graphs in memory. A chunk is saved as its own ``.gtds`` and the
:class:`GraphTaskSource` stitches them back together.

**Two partition regimes.**

* A **corpus** (GraphQA, KGQA, TAG) has fixed splits, so the partition is
  computed from the raw sources before anything is drawn and a row whose key
  lost its role to a higher-priority claim is dropped and counted. GraphQA is
  the reason this matters: its tasks share graphs, and 38 of a task's 1,000
  train graphs also sit in some test split.
* A **generator** (probes, expressiveness, ``our_tests``) has no raw split to
  read, so its partition is the record of what was drawn. Every draw is
  rejected if its key — a content hash of the graph — already holds a different
  role anywhere in the domain's build, and the rejections are counted. The
  partition is then assembled from the built sidecars.

**Generator randomness.** The specialist generators draw from the global
``random`` and ``numpy.random`` streams. Each ``(task, split, pass)`` stream is
seeded once from ``data_seed`` and its state is saved and restored around every
chunk, so a build is byte-identical whatever the chunk size and whatever the
featurisation code does to the global generators in between.

Nothing at module scope imports torch, networkx or transformers.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..registry import Registry, TaskSpec
from ..schema import SCHEMA_VERSION, Example, SchemaError, render, validate
from ._partition import Claim, Partition, PartitionError, build_partition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_CACHE_ROOT = os.path.join(_REPO_ROOT, "src", "generalist", "results", "data")

#: Answer kinds whose target closes its turn: the chat suffix, or the EOS token
#: under a format with no suffix. ``span`` is scored teacher-forced, but the stop
#: token is inside its exact-match span, so a model that would run on past the
#: answer scores as wrong — which is what an answer someone reads would be.
TERMINATED_KINDS = ("text", "smiles", "span", "entities")

#: A generator that rejects this many draws per accepted one is not drawing from
#: a pool that can fill the split; stop and say so rather than spin.
MAX_REJECTIONS_PER_ROW = 20

_QUESTION = "\x00question"
_PROMPT = "\x00prompt"


class GraphBuildError(ValueError):
    """A graph-domain build that cannot proceed. The message names the cause."""


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphAdapterConfig:
    """What every graph domain's build depends on. Domains subclass to add theirs.

    ``limit`` caps every split at that many rows (0 = the full size). It exists
    for smoke builds, and it is inside the build version, so a limited build can
    never be mistaken for the real one.

    ``max_length`` is the per-node token cap. A content node over it is truncated
    and counted in the manifest; a question or prompt node over it is an error,
    because a truncated question changes the task and a truncated prompt node
    loses the answer.
    """

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    prompt_style: str = None
    max_length: int = 512
    magnetic_q: float = 0.25
    magnetic_m: int = 0
    ordering: str = "rcm"
    data_seed: int = 0
    limit: int = 0
    answer_eos: bool = True
    cache_root: str = DEFAULT_CACHE_ROOT

    def resolved_style(self) -> str:
        from ...experiments.molecules.data import resolve_prompt_style

        return resolve_prompt_style(self.prompt_style, self.model_name)

    def validate(self) -> "GraphAdapterConfig":
        if self.ordering not in ("rcm", "original"):
            raise GraphBuildError(
                f"ordering must be 'rcm' or 'original', got {self.ordering!r}")
        if self.max_length < 16:
            raise GraphBuildError(f"max_length must be >= 16, got {self.max_length}")
        if self.limit < 0:
            raise GraphBuildError(f"limit must be >= 0, got {self.limit}")
        if self.magnetic_m < 0:
            raise GraphBuildError(f"magnetic_m must be >= 0, got {self.magnetic_m}")
        return self

    def hashed_payload(self) -> dict:
        payload = asdict(self)
        for drop in ("cache_root", "prompt_style"):
            payload.pop(drop)
        payload["prompt_style"] = self.resolved_style()
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# What a domain declares
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskInfo:
    """One task of a domain, as the domain declares it.

    ``splits`` are the splits built; ``eval_splits`` the ones a validator may
    score, which is narrower for a train-only task. A held-out task builds
    ``held_out`` and nothing else. ``sizes`` are rows per split for a generator
    (a corpus has the size its source has). ``magnetic_m_cap`` bounds the
    eigenvectors kept on a domain whose graphs are large enough for ``m = N`` to
    cost more than the bias is worth (the specialists ran expressiveness and
    KGQA at 128); 0 means no cap.
    """

    name: str
    answer_kind: str
    kind: str
    metric: str
    held_out: bool = False
    splits: tuple = ("train", "val", "test")
    eval_splits: tuple = ("val", "test")
    sizes: dict = None
    max_new_tokens: int = None
    magnetic_m_cap: int = 0
    chunk: int = 10_000
    question_template: str = None

    def built_splits(self) -> tuple:
        return ("held_out",) if self.held_out else tuple(self.splits)


@dataclass
class Draw:
    """One example before featurisation.

    ``graph`` holds the content nodes only, each with a ``text`` attribute; the
    question and prompt nodes are this module's to add. ``targets`` are content
    node ids the prompt node points at. ``answer`` is the clean answer — no
    leading space, no terminator — except for ``yesno``, where it is the label
    word itself (``" Yes"`` / ``" No"``). ``key`` is the partition key.
    """

    graph: object
    question: str
    answer: str
    targets: tuple
    key: str
    meta: dict = field(default_factory=dict)


@dataclass
class GraphDomain:
    """One graph domain: its tasks and the callables only it can supply.

    ``draws(config, info, split, pass_id)`` returns an iterator of
    :class:`Draw`. ``source_digests(config)`` names the raw inputs, which enter
    the build version. ``claims(config, infos)`` returns the corpus partition's
    :class:`Claim` list, and is ``None`` for a generator. ``role_of_split`` maps
    a split to the partition role its keys must hold, when the two differ — a
    GraphQA held-out task reads the official test file, whose graphs are
    ``test`` role because other tasks' test splits hold them too.
    """

    name: str
    prefix: str
    adapter_version: str
    tasks: tuple
    config_class: type
    draws: Callable
    source_digests: Callable
    claims: Callable = None
    role_of_split: Callable = None

    def __post_init__(self):
        self._by_name = {info.name: info for info in self.tasks}
        kinds = {info.kind for info in self.tasks}
        if len(kinds) != 1:
            raise GraphBuildError(f"{self.name}: a domain is all corpus or all "
                                  f"generator, got {sorted(kinds)}")
        if (self.claims is None) != (self.kind == "generator"):
            raise GraphBuildError(
                f"{self.name}: a corpus supplies partition claims and a generator "
                "does not")

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def kind(self) -> str:
        return self.tasks[0].kind

    def full_name(self, task: str) -> str:
        return task if task.startswith(self.prefix) else f"{self.prefix}{task}"

    def info(self, task: str) -> TaskInfo:
        bare = task[len(self.prefix):] if task.startswith(self.prefix) else task
        try:
            return self._by_name[bare]
        except KeyError:
            raise GraphBuildError(
                f"{task!r}: not a {self.name} task (have "
                f"{[self.full_name(t.name) for t in self.tasks]})") from None

    def role(self, info: TaskInfo, split: str) -> str:
        if self.role_of_split is not None:
            return self.role_of_split(info, split)
        return split

    # ── versioning and paths ─────────────────────────────────────────────────

    def build_version(self, config: GraphAdapterConfig) -> str:
        return _hash({"domain": self.name, "adapter_version": self.adapter_version,
                      "schema_version": SCHEMA_VERSION,
                      "config": config.hashed_payload(),
                      "sources": self.source_digests(config)})

    def build_dir(self, config) -> str:
        return os.path.join(config.cache_root, self.name, self.build_version(config))

    def manifest_path(self, config) -> str:
        return os.path.join(self.build_dir(config), "manifest.json")

    def partition_path(self, config) -> str:
        return os.path.join(self.build_dir(config), "partition.json")

    def source_path(self, config, task: str, split: str, arm: str, pass_id: int) -> str:
        return os.path.join(self.build_dir(config),
                            self.full_name(task).replace("/", "_"),
                            f"{split}.{arm}.p{int(pass_id)}")

    def magnetic_m(self, config, info: TaskInfo) -> int:
        """The run's ``m``, bounded by the task's cap. 0 means every eigenvector,
        so under a cap it means the cap."""
        cap, m = int(info.magnetic_m_cap), int(config.magnetic_m)
        if cap and (m == 0 or m > cap):
            return cap
        return m

    # ── registry ─────────────────────────────────────────────────────────────

    def task_specs(self, config, arm: str = "graph") -> dict:
        """``{name: TaskSpec}`` for every task of this domain.

        ``train_size`` and ``mean_tokens`` come from the manifest and are
        ``None`` until a build exists, which is what `wiring.unbuilt_tasks`
        reports. For a held-out task they describe its ``held_out`` split — the
        only split an ``adapt`` fork can train on (`registry.TaskSpec`).
        """
        manifest = self.read_manifest(config)
        version = self.build_version(config)
        out = {}
        for info in self.tasks:
            name = self.full_name(info.name)
            entry = (manifest.get("tasks", {}) or {}).get(name, {})
            by_arm = entry.get("arms", {}).get(arm, {})
            out[name] = TaskSpec(
                name=name, domain=self.name, adapter=self.name, kind=info.kind,
                answer_kind=info.answer_kind, held_out=info.held_out,
                metric=info.metric, passes=1,
                cap_per_pass=(by_arm.get("train_size") if info.kind == "generator"
                              else None),
                max_new_tokens=info.max_new_tokens, build_version=version,
                eval_splits=(() if info.held_out else tuple(info.eval_splits)),
                mean_tokens=by_arm.get("mean_tokens"),
                train_size=by_arm.get("train_size"),
                question_template=info.question_template)
        return out

    def register(self, registry: Registry, config, arm: str = "graph") -> Registry:
        for spec in self.task_specs(config, arm=arm).values():
            registry.register(spec)
        return registry

    # ── partition ────────────────────────────────────────────────────────────

    def partition(self, config) -> Partition:
        """The domain's partition: from the raw sources for a corpus, from the
        built sidecars for a generator (see the module docstring)."""
        if self.kind == "corpus":
            return build_partition(self.claims(config, self.tasks),
                                   {"domain": self.name})
        claims = {}
        for (task, split), keys in self._built_keys(config).items():
            info = self.info(task)
            claims[(f"{task}/{split}", self.role(info, split))] = keys
        return build_partition(
            [Claim(source, role, tuple(sorted(keys)))
             for (source, role), keys in sorted(claims.items())],
            {"domain": self.name})

    def _built_keys(self, config) -> dict:
        """``{(task, split): set(keys)}`` over every sidecar in the build dir."""
        root = self.build_dir(config)
        out: dict = {}
        if not os.path.isdir(root):
            return out
        for sub in sorted(os.listdir(root)):
            folder = os.path.join(root, sub)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".schema.json"):
                    continue
                with open(os.path.join(folder, name)) as fh:
                    side = json.load(fh)
                out.setdefault((side["task"], side["split"]), set()).update(
                    r["key"] for r in side["records"])
        return out

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, config, tasks=None, arms=("graph",), passes: int = 1,
              rebuild: bool = False) -> dict:
        """Materialise every split of ``tasks`` (default: all), ``passes`` train
        passes for a generator. Idempotent: a built artifact is reused unless
        ``rebuild``."""
        from transformers import AutoTokenizer

        config.validate()
        arms = tuple(arms)
        if arms != ("graph",):
            raise GraphBuildError(
                f"{self.name}: graph arm only, asked for {arms}. These sources have "
                "no flat twin in the trunk's format (module docstring).")
        infos = [self.info(t) for t in tasks] if tasks else list(self.tasks)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        manifest = self.read_manifest(config)
        manifest["domain"] = self.name
        manifest["build_version"] = self.build_version(config)
        manifest["config"] = config.hashed_payload()
        manifest["sources"] = self.source_digests(config)

        if self.kind == "corpus":
            part = self.partition(config)
            part.save(self.partition_path(config))
            manifest["partition"] = {"counts": part.counts,
                                     "role_totals": part.role_totals,
                                     "ledger": part.ledger}
            for info in infos:
                for split in info.built_splits():
                    self._build_split(config, info, split, 0, tokenizer, manifest,
                                      rebuild, part=part)
        else:
            claimed = self._claimed_roles(config)
            # Evaluation splits first, for every task, so train draws are the
            # ones that give way: an eval set is fixed, a train pass is one of many.
            for info in infos:
                for split in info.built_splits():
                    if split != "train":
                        self._build_split(config, info, split, 0, tokenizer,
                                          manifest, rebuild, claimed=claimed)
            for info in infos:
                if "train" in info.built_splits():
                    for pass_id in range(max(1, int(passes))):
                        self._build_split(config, info, "train", pass_id, tokenizer,
                                          manifest, rebuild, claimed=claimed)
            part = self.partition(config)
            part.save(self.partition_path(config))
            manifest["partition"] = {"counts": part.counts,
                                     "role_totals": part.role_totals}
        self.write_manifest(config, manifest)
        return manifest

    def _claimed_roles(self, config) -> dict:
        """``key -> role`` over everything already built, for draw rejection."""
        out = {}
        for (task, split), keys in self._built_keys(config).items():
            role = self.role(self.info(task), split)
            for key in keys:
                out.setdefault(key, role)
        return out

    def _target_rows(self, config, info: TaskInfo, split: str):
        size = (info.sizes or {}).get(split)
        if config.limit:
            return min(size, config.limit) if size else config.limit
        return size

    def _build_split(self, config, info, split, pass_id, tokenizer, manifest,
                     rebuild, part=None, claimed=None):
        name = self.full_name(info.name)
        entry = manifest.setdefault("tasks", {}).setdefault(
            name, {"arms": {}, "splits": {}})
        side_path = _sidecar_path(self.source_path(config, name, split, "graph", pass_id))
        if rebuild or not os.path.exists(side_path):
            sidecar = self._materialise_split(config, info, split, pass_id, tokenizer,
                                              part=part, claimed=claimed)
        else:
            with open(side_path) as fh:
                sidecar = json.load(fh)
            if claimed is not None:
                role = self.role(info, split)
                for record in sidecar["records"]:
                    claimed.setdefault(record["key"], role)
        entry["splits"][f"{split}.p{pass_id}"] = sidecar["stats"]
        if pass_id == 0 and split == ("held_out" if info.held_out else "train"):
            tokens = sidecar["num_tokens"]
            entry["arms"]["graph"] = {
                "train_size": len(tokens),
                "mean_tokens": sum(tokens) / len(tokens) if tokens else None}

    def _materialise_split(self, config, info, split, pass_id, tokenizer,
                           part=None, claimed=None) -> dict:
        from ...experiments.molecules.data import prompt_format

        name = self.full_name(info.name)
        spec = self.task_specs(config)[name]
        fmt = prompt_format(config.prompt_style, config.model_name)
        role = self.role(info, split)
        target = self._target_rows(config, info, split)
        if info.kind == "generator" and not target:
            raise GraphBuildError(f"{name}: generator task has no size for {split!r}")

        stream = _Stream(lambda: iter(self.draws(config, info, split, pass_id)),
                         f"{config.data_seed}|{name}|{split}|{pass_id}")
        stats = {"drawn": 0, "emitted": 0, "dropped_role": 0, "rejected_claimed": 0,
                 "truncated_content_nodes": 0, "rows_with_truncation": 0}
        path = self.source_path(config, name, split, "graph", pass_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        records, num_nodes, num_tokens, chunks = [], [], [], []
        m = self.magnetic_m(config, info)
        exhausted = False
        while not exhausted and (target is None or len(records) < target):
            want = info.chunk if target is None else min(info.chunk, target - len(records))
            batch = []
            while len(batch) < want:
                pulled = stream.take(want - len(batch))
                if not pulled:
                    exhausted = True
                    break
                for draw in pulled:
                    stats["drawn"] += 1
                    if part is not None and part.role(draw.key) != role:
                        stats["dropped_role"] += 1
                        continue
                    if claimed is not None:
                        held = claimed.get(draw.key)
                        if held is not None and held != role:
                            stats["rejected_claimed"] += 1
                            if stats["rejected_claimed"] > MAX_REJECTIONS_PER_ROW * max(
                                    1, target or 0):
                                raise GraphBuildError(
                                    f"{name}/{split}: {stats['rejected_claimed']} draws "
                                    "rejected as already claimed by another role; the "
                                    "generator's pool is too small for these sizes")
                            continue
                    batch.append(draw)
            if not batch:
                break
            if claimed is not None:
                for draw in batch:
                    claimed.setdefault(draw.key, role)
            chunk = _materialise_chunk(self, config, info, spec, split, pass_id,
                                       batch, tokenizer, fmt, m, stats,
                                       f"{path}.c{len(chunks)}")
            records.extend(chunk["records"])
            num_nodes.extend(chunk["num_nodes"])
            num_tokens.extend(chunk["num_tokens"])
            chunks.append(len(chunk["records"]))
        if target is not None and len(records) < target and info.kind == "generator":
            raise GraphBuildError(f"{name}/{split}: generator stopped at "
                                  f"{len(records)} of {target} rows")
        if not records:
            raise GraphBuildError(f"{name}/{split}: no rows survived "
                                  f"(stats {stats}); nothing to build")
        stats["emitted"] = len(records)
        stats["distinct_keys"] = len({r["key"] for r in records})
        sidecar = {"task": name, "split": split, "arm": "graph",
                   "pass_id": int(pass_id), "domain": self.name,
                   "answer_kind": info.answer_kind,
                   "build_version": self.build_version(config),
                   "magnetic_m": m, "chunks": chunks, "records": records,
                   "num_nodes": num_nodes, "num_tokens": num_tokens, "stats": stats}
        with open(_sidecar_path(path), "w") as fh:
            json.dump(sidecar, fh)
        return sidecar

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, task: str, split: str, arm: str, pass_id: int = 0, config=None,
             check_keys: int = 200) -> "GraphTaskSource":
        """The built source for one ``(task, split, arm, pass)``. Never builds."""
        if config is None:
            raise GraphBuildError(f"{self.name}.load needs config=")
        info = self.info(task)
        name = self.full_name(info.name)
        if split not in info.built_splits():
            raise GraphBuildError(
                f"{name}: split {split!r} is not one of {info.built_splits()}")
        path = self.source_path(config, name, split, arm, pass_id)
        if not os.path.exists(_sidecar_path(path)):
            raise GraphBuildError(
                f"{path} has not been built. Run data_prep for {name} "
                f"({split}/{arm}, pass {pass_id}).")
        with open(_sidecar_path(path)) as fh:
            sidecar = json.load(fh)

        from ...utils import TextGraphDataset

        datasets = [TextGraphDataset.load(f"{path}.c{k}")
                    for k in range(len(sidecar["chunks"]))]
        source = GraphTaskSource(datasets, sidecar, path)
        if check_keys:
            part_path = self.partition_path(config)
            if not os.path.exists(part_path):
                raise PartitionError(f"{part_path} is missing; rerun data_prep")
            part = Partition.load(part_path)
            role = self.role(info, split)
            keys = source.keys()
            rng = random.Random(f"recheck|{name}|{split}|{pass_id}")
            for key in keys if len(keys) <= check_keys else rng.sample(keys, check_keys):
                if not part.is_role(key, role):
                    raise PartitionError(
                        f"{name}/{split}: built key {key} has role {part.role(key)!r}, "
                        f"not {role!r}; the artifact and the partition disagree, "
                        "rerun data_prep")
        return source

    # ── manifest ─────────────────────────────────────────────────────────────

    def read_manifest(self, config) -> dict:
        path = self.manifest_path(config)
        if not os.path.exists(path):
            return {}
        with open(path) as fh:
            return json.load(fh)

    def write_manifest(self, config, manifest: dict) -> str:
        path = self.manifest_path(config)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        os.replace(tmp, path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Assembly and featurisation
# ─────────────────────────────────────────────────────────────────────────────

def assemble(draw: Draw, answer_kind: str, fmt):
    """Content graph + isolated question node + prompt node, prompt node last.

    The layout `attach_question` builds for molecules, on any content graph. An
    undirected source graph becomes a symmetric digraph first, which is how
    every specialist stored one.
    """
    import networkx as nx

    from ...experiments.molecules.data import relabel_for_dataset

    graph = draw.graph if draw.graph.is_directed() else draw.graph.to_directed()
    graph = nx.DiGraph(graph)
    for node in (_QUESTION, _PROMPT):
        if node in graph:
            raise GraphBuildError(f"content graph already has a node named {node!r}")
    for node, data in graph.nodes(data=True):
        if not isinstance(data.get("text"), str):
            raise GraphBuildError(f"content node {node!r} has no text")
    answer = draw.answer if answer_kind == "yesno" else f" {draw.answer}"
    suffix = fmt.answer_suffix if answer_kind in TERMINATED_KINDS else ""
    graph.add_node(_QUESTION, text=fmt.question(draw.question), kind="question")
    graph.add_node(_PROMPT, text=f"{fmt.answer_prefix}{answer}{suffix}", kind="prompt")
    for target in draw.targets:
        if target not in graph or target in (_QUESTION, _PROMPT):
            raise GraphBuildError(f"prompt edge target {target!r} is not a content node")
        graph.add_edge(_PROMPT, target)
    graph.graph = {"question_node": _QUESTION, "prompt_node": _PROMPT}
    return relabel_for_dataset(graph)


def _labels_fn(tokenizer, answer_kind: str, max_length: int, add_eos: bool,
               answer_prefix: str):
    """`schema.render`'s span on the prompt node, with the appended EOS supervised.

    Same construction as `adapters.molecules._labels_fn`: the span depends on the
    prompt node's text alone, so a one-node stub renders it at the cost of one
    tokenizer call. The prompt node here holds nothing before ``answer_prefix``,
    so the prefix's first occurrence is the boundary and an answer that happens
    to contain the prefix cannot move it.
    """

    def get_labels(example):
        text = example["text"][example["prompt_node"]]
        if not text.startswith(answer_prefix):
            raise SchemaError(f"prompt node {text[:80]!r} does not start with "
                              f"{answer_prefix!r}")
        stub = Example(task="_", domain="_", split="train", arm="graph",
                       graph={"text": [text], "prompt_node": 0, "num_nodes": 1},
                       question="_", answer=text[len(answer_prefix):],
                       answer_kind=answer_kind, key="_")
        labels = render(stub, tokenizer, max_length=max_length).labels
        if add_eos:
            if len(labels) >= max_length:
                del labels[max_length - 1:]
            labels.append(tokenizer.eos_token_id)
        return labels

    return get_labels


def _materialise_chunk(domain, config, info, spec, split, pass_id, draws, tokenizer,
                       fmt, magnetic_m, stats, path) -> dict:
    """Assemble, featurise, validate and save one chunk of draws."""
    from ...utils import TextGraphDataset

    name = domain.full_name(info.name)
    kind = info.answer_kind
    add_eos = (bool(config.answer_eos) and kind in TERMINATED_KINDS
               and not fmt.answer_suffix)
    graphs = [assemble(draw, kind, fmt) for draw in draws]

    ds = TextGraphDataset(graphs, dataset_label=name,
                          rcm_ordering=(config.ordering == "rcm"))
    # Tokens and labels while the table is light; the (N, N) features after
    # (`expressiveness/data/data_gen.py` measured why at large N).
    ds.tokenize(tokenizer, max_length=config.max_length, add_eos=add_eos)
    ds.compute_labels(_labels_fn(tokenizer, kind, config.max_length, add_eos,
                                 fmt.answer_prefix), num_proc=1)

    for i in range(len(ds)):
        g = ds.graphs[i]
        ids = ds._hf_dataset[i]["input_ids"]
        q, p = int(g.graph["question_node"]), int(g.graph["prompt_node"])
        for node in (q, p):
            if len(ids[node]) >= config.max_length:
                raise GraphBuildError(
                    f"{name}/{split}: row {stats['drawn']}'s "
                    f"{'question' if node == q else 'prompt'} node reaches the "
                    f"{config.max_length}-token cap; raise max_length for this "
                    "domain rather than train on a cut question or answer")
        cut = sum(1 for n, node_ids in enumerate(ids)
                  if n not in (q, p) and len(node_ids) >= config.max_length)
        stats["truncated_content_nodes"] += cut
        stats["rows_with_truncation"] += int(cut > 0)

    ds.compute_shortest_path_distances()
    ds.compute_magnetic_lap(q=config.magnetic_q, m=magnetic_m)
    ds.cast_float_features_to_fp32()

    records, num_nodes, num_tokens = [], [], []
    for i, draw in enumerate(draws):
        item = _item(ds, i, name)
        example = Example(task=name, domain=domain.name, split=split, arm="graph",
                          graph=item, question=draw.question, answer=draw.answer,
                          answer_kind=kind, key=draw.key, meta=dict(draw.meta))
        validate(example, spec)
        labels = item["labels"]
        if all(int(v) == -100 for v in labels):
            raise GraphBuildError(f"{name}/{split}: row {i} has no supervised token")
        records.append({"question": draw.question, "answer": draw.answer,
                        "key": draw.key, "meta": dict(draw.meta)})
        num_nodes.append(int(item["num_nodes"]))
        num_tokens.append(sum(len(node_ids) for node_ids in item["input_ids"]))
    ds.save(path)
    return {"records": records, "num_nodes": num_nodes, "num_tokens": num_tokens}


def _item(ds, i: int, task: str) -> dict:
    """One item with ``question_node`` restored (see `adapters.molecules._item`)."""
    item = dict(ds[i])
    item["ds_label"] = task
    item["question_node"] = int(ds.graphs[i].graph.get("question_node", -1))
    return item


def _sidecar_path(path: str) -> str:
    return path + ".schema.json"


# ─────────────────────────────────────────────────────────────────────────────
# The TaskSource
# ─────────────────────────────────────────────────────────────────────────────

class GraphTaskSource:
    """One built ``(task, split, arm, pass)`` over its chunks. See `adapters.TaskSource`."""

    def __init__(self, datasets: list, sidecar: dict, path: str):
        self._datasets = list(datasets)
        self._records = sidecar["records"]
        self._num_nodes = sidecar["num_nodes"]
        self._num_tokens = sidecar["num_tokens"]
        self._starts = []
        total = 0
        for ds, expected in zip(self._datasets, sidecar["chunks"]):
            if len(ds) != expected:
                raise GraphBuildError(
                    f"{path}: a chunk holds {len(ds)} graphs, the sidecar says {expected}")
            self._starts.append(total)
            total += len(ds)
        if total != len(self._records):
            raise GraphBuildError(
                f"{path}: {len(self._records)} records beside {total} graphs")
        self._total = total
        self.path = path
        self.task = sidecar["task"]
        self.split = sidecar["split"]
        self.arm = sidecar["arm"]
        self.pass_id = int(sidecar["pass_id"])
        self.domain = sidecar["domain"]
        self.answer_kind = sidecar["answer_kind"]
        self.build_version = sidecar["build_version"]

    def __len__(self) -> int:
        return self._total

    def _locate(self, i: int) -> tuple:
        if i < 0:
            i += self._total
        if not 0 <= i < self._total:
            raise IndexError(f"{i} out of range for {self!r}")
        chunk = bisect.bisect_right(self._starts, i) - 1
        return chunk, i - self._starts[chunk]

    def __getitem__(self, i: int) -> dict:
        chunk, local = self._locate(int(i))
        record = self._records[int(i)]
        return Example(task=self.task, domain=self.domain, split=self.split,
                       arm=self.arm, graph=_item(self._datasets[chunk], local, self.task),
                       question=record["question"], answer=record["answer"],
                       answer_kind=self.answer_kind, key=record["key"],
                       meta=record["meta"]).to_item()

    def example(self, i: int) -> Example:
        return Example.from_item(self[i], None, split=self.split)

    def keys(self) -> list:
        return [record["key"] for record in self._records]

    def lengths(self) -> tuple:
        return list(self._num_nodes), list(self._num_tokens)

    def __repr__(self) -> str:
        return (f"<GraphTaskSource {self.task} {self.split}/{self.arm} "
                f"p{self.pass_id} n={len(self)}>")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers the domains share
# ─────────────────────────────────────────────────────────────────────────────

class _Stream:
    """A draw iterator with its own ``random`` / ``numpy.random`` state.

    Seeded once, and swapped in only while it is being pulled from, so what it
    yields does not depend on how it is chunked or on anything the build does to
    the global generators between chunks.
    """

    def __init__(self, factory, seed: str):
        import numpy as np

        outer = (random.getstate(), np.random.get_state())
        try:
            # The factory runs *before* seeding: a domain imports its generator
            # module there, and `our_tests`' modules reseed the global generators
            # to 42 at import, which would otherwise overwrite this stream's seed
            # the first time and only the first time.
            self._it = factory()
            random.seed(seed)
            np.random.seed(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
            self._state = (random.getstate(), np.random.get_state())
        finally:
            random.setstate(outer[0])
            np.random.set_state(outer[1])

    def take(self, n: int) -> list:
        import itertools

        import numpy as np

        outer = (random.getstate(), np.random.get_state())
        random.setstate(self._state[0])
        np.random.set_state(self._state[1])
        try:
            return list(itertools.islice(self._it, n))
        finally:
            self._state = (random.getstate(), np.random.get_state())
            random.setstate(outer[0])
            np.random.set_state(outer[1])


def graph_key(graph, extra: str = "") -> str:
    """A content hash of a graph: sorted node texts and sorted edges by text.

    Node ids are construction order and mean nothing, so the hash is over texts.
    Where texts repeat (GraphQA's "0", "1", … are unique per graph; spreadsheet
    labels are too) an edge is identified by its endpoint texts, which is exact
    for every source here because node texts are unique within a graph. ``extra``
    folds in anything else that makes two examples the same (a question).
    """
    texts = {n: d.get("text", "") for n, d in graph.nodes(data=True)}
    payload = {"nodes": sorted(texts.values()),
               "edges": sorted([texts[u], texts[v]] for u, v in graph.edges()),
               "extra": extra}
    return _hash(payload)


def split_prompt(graph, prompt=None):
    """``(content graph, prompt text, targets)`` for a specialist-built graph.

    The probe and ``our_tests`` builders attach their own prompt node; this takes
    it back off, keeping where it pointed. A source-side question node is dropped
    too — the question is rebuilt from the draw.
    """
    graph = graph.copy()
    prompt = graph.graph.get("prompt_node") if prompt is None else prompt
    text = graph.nodes[prompt]["text"]
    targets = tuple(graph.successors(prompt)) if graph.is_directed() else tuple(
        graph.neighbors(prompt))
    graph.remove_node(prompt)
    question = graph.graph.get("question_node")
    if question is not None and question in graph:
        graph.remove_node(question)
    graph.graph = {}
    return graph, text, targets


def file_digest(path: str) -> str:
    """sha256 of a raw input, cached on ``(path, size, mtime)``."""
    if not os.path.exists(path):
        raise GraphBuildError(f"{path} is missing")
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _DIGESTS[key] = h.hexdigest()[:16]
    return _DIGESTS[key]


def experiment_path(*parts: str) -> str:
    """A path under `src/experiments/`, resolved without importing anything there.

    The build version is asked for by every registry build, `validate` mode
    included, and the experiment modules import torch at module scope.
    """
    return os.path.join(_REPO_ROOT, "src", "experiments", *parts)


def code_digest(*relative: str) -> dict:
    """Digests of generator source files under `src/experiments/`, which are a
    generator's raw input: a changed generator is a different dataset."""
    return {path: file_digest(experiment_path(path)) for path in relative}


_DIGESTS: dict = {}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()[:16]
