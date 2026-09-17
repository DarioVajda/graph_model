"""
D3 for text — ``text/replay``, the backbone's own answers as single-node examples.

`MOLECULE_GENERALIST.md` §9.1. A trunk that has never seen a single-node graph
answers one text prompt in seven with a ChEBI-20 caption (§7.6), and the test of
that hypothesis is a trunk that has. Every example here is a one-node graph
holding a user turn and the assistant turn the **backbone itself** wrote for it,
which is `dataset.build_flat_example`'s shape with no molecule in it. On one node
every structural bias is identically zero (Property 2), so the forward pass is
the base LLM's plus the adapter and nothing else — no model change, on either arm.

**What this file owns, and what it does not.** The prompts and the answers are
built upstream and versioned on disk, because both are expensive and neither may
change under a run: `tools/replay_prompts.py` selects and filters the prompts,
`tools/replay_generate.py` samples the answers on a GPU. This adapter reads that
directory and does the part every adapter does — split, featurize, validate,
save — and `load` never regenerates.

**The key is the prompt.** ``prompt_id`` hashes the normalised prompt text, and a
prompt holds one role — ``train`` or ``val`` — across every answer to it, so
three samples of one prompt can never straddle the split. The 48
`text_probes.json` prompts are the test set and were removed at selection
(exact and 8-gram), so there is no ``test`` role here.

**A generator, whose passes are answer indices.** Pass *p* serves, for every
train prompt, its answer ``(p · w + j) mod m`` for ``j < w`` — ``m`` the answers
that survived and ``w`` the prompt's weight (1, or 2 for the chemistry slice when
it is too small to carry its share at one). Nothing repeats until ``m / w``
passes have been drawn, and the repeats a longer run does draw are counted into
the manifest rather than left to be discovered.

**Its own node length.** Molecule nodes are cut at 512 tokens and that value is
inside their build hashes. A prompt and a natural answer do not fit in 512, and
a truncated target cuts off exactly the terminator the example exists to teach,
so the text task carries its own ``max_length`` (1,024) and an example that would
exceed it is **dropped and counted, never truncated**.

Nothing at module scope imports torch, RDKit or transformers, so ``validate``
mode still resolves a config that names this task on the login node.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass

from ..registry import TEXT_PREFIX, Registry, TaskSpec
from ..schema import SCHEMA_VERSION, Example, SchemaError, render, validate
from ._partition import Claim, Partition, PartitionError, build_partition

DOMAIN = "text"
ADAPTER_NAME = "text"

#: Bumped when this file changes what it writes to disk for a fixed config.
ADAPTER_VERSION = "1"

REPLAY_TASK = "replay"
REPLAY_NAME = f"{TEXT_PREFIX}{REPLAY_TASK}"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_REPLAY_ROOT = os.path.join(_REPO_ROOT, "src", "generalist", "results", "replay")
DEFAULT_CACHE_ROOT = os.path.join(_REPO_ROOT, "src", "generalist", "results", "data")

#: Generation cap for a scorer that ever runs this task generatively. It is the
#: cap the targets were sampled under, so a longer one would only ever run on.
MAX_NEW_TOKENS = 768

#: A special token spelled out in plain text. An answer that writes one would
#: re-tokenize into the special id and close its own turn early.
SPECIAL_TEXT = re.compile(r"<\|[a-z_0-9]+\|>")


class TextBuildError(ValueError):
    """A text build that cannot proceed. The message names the cause."""


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextAdapterConfig:
    """Everything that changes what this adapter writes to disk.

    ``replay_version`` names a directory under ``replay_root`` holding
    ``prompts.jsonl`` and ``answers/``. The directory's contents, not its name,
    enter the build version: a regenerated set under the same name is a
    different dataset.
    """

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    prompt_style: str = None
    max_length: int = 1024
    #: The run's magnetic Laplacian settings. On one node the feature is a
    #: constant, but its *shape* follows ``m`` and has to collate beside the
    #: molecule items in the same micro-batch.
    magnetic_q: float = 0.25
    magnetic_m: int = 0
    replay_version: str = "v1"
    #: The answers subdirectory, so a pilot's answers can be built and inspected
    #: through the same path as the full set's.
    answers_name: str = "answers"
    replay_root: str = DEFAULT_REPLAY_ROOT
    cache_root: str = DEFAULT_CACHE_ROOT

    def replay_dir(self) -> str:
        return os.path.join(self.replay_root, self.replay_version)

    def answers_dir(self) -> str:
        return os.path.join(self.replay_dir(), self.answers_name)

    def resolved_style(self) -> str:
        from ...experiments.molecules.data import resolve_prompt_style

        return resolve_prompt_style(self.prompt_style, self.model_name)

    def generated(self) -> bool:
        """Whether the answers exist yet. Before they do, the task registers as
        unbuilt — ``validate`` still resolves the config and prints the shares —
        and only ``build`` refuses."""
        return os.path.exists(os.path.join(self.answers_dir(), "generation.json"))

    def generation(self) -> dict:
        path = os.path.join(self.answers_dir(), "generation.json")
        if not os.path.exists(path):
            raise TextBuildError(
                f"{path} is missing: the replay answers for "
                f"{self.replay_version!r} have not been generated "
                "(tools/replay_generate.py).")
        with open(path) as fh:
            return json.load(fh)

    def validate(self) -> "TextAdapterConfig":
        """The run's backbone and format must be the ones that wrote the answers.

        Replay distils the backbone into the trunk. A set sampled from one model
        and trained into another is not replay but a teacher swap, and one
        sampled in chat markup and trained in ``Q:/A:`` is a question the trunk
        is never asked.
        """
        if self.max_length < 16:
            raise TextBuildError(f"max_length must be >= 16, got {self.max_length}")
        generation = self.generation()
        if generation.get("model_name") != self.model_name:
            raise TextBuildError(
                f"replay {self.replay_version!r} was generated by "
                f"{generation.get('model_name')!r}, and this run's backbone is "
                f"{self.model_name!r}. Generate a set for this backbone.")
        if generation.get("prompt_style") != self.resolved_style():
            raise TextBuildError(
                f"replay {self.replay_version!r} was generated in "
                f"{generation.get('prompt_style')!r} format and this run resolves "
                f"to {self.resolved_style()!r}.")
        return self

    # ── versioning (D3.2) ────────────────────────────────────────────────────

    def source_digests(self) -> dict:
        out = {"prompts": _file_digest(os.path.join(self.replay_dir(), "prompts.jsonl"))}
        answers = self.answers_dir()
        for name in sorted(os.listdir(answers)) if os.path.isdir(answers) else ():
            if name.endswith(".jsonl") or name == "generation.json":
                out[f"{self.answers_name}/{name}"] = _file_digest(
                    os.path.join(answers, name))
        return out

    def build_version(self) -> str:
        if not self.generated():
            return "ungenerated"
        payload = asdict(self)
        for drop in ("replay_root", "cache_root", "prompt_style"):
            payload.pop(drop)
        payload["prompt_style"] = self.resolved_style()
        return _hash({"adapter_version": ADAPTER_VERSION,
                      "schema_version": SCHEMA_VERSION,
                      "config": payload, "sources": self.source_digests()})

    # ── cache paths ──────────────────────────────────────────────────────────

    def build_dir(self) -> str:
        return os.path.join(self.cache_root, "text", self.build_version())

    def manifest_path(self) -> str:
        return os.path.join(self.build_dir(), "manifest.json")

    def source_path(self, task: str, split: str, arm: str, pass_id: int) -> str:
        return os.path.join(self.build_dir(), task.replace("/", "_"),
                            f"{split}.{arm}.p{int(pass_id)}")


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()[:16]


#: ``(path, size, mtime) -> digest``. The answers are ~100 MB and the build
#: version is asked for by every registry build, so hashing them once per
#: process rather than once per call is the difference between a validate that
#: returns and one that visibly thinks.
_DIGESTS: dict = {}


def _file_digest(path: str) -> str:
    if not os.path.exists(path):
        raise TextBuildError(f"{path} is missing; run tools/replay_prompts.py and "
                             "tools/replay_generate.py for this version first.")
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _DIGESTS[key] = h.hexdigest()[:16]
    return _DIGESTS[key]


# ─────────────────────────────────────────────────────────────────────────────
# Reading the versioned set
# ─────────────────────────────────────────────────────────────────────────────

def read_prompts(config: TextAdapterConfig) -> list:
    with open(os.path.join(config.replay_dir(), "prompts.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def usable_answer(row: dict) -> str | None:
    """Why an answer is not a target, or ``None`` when it is one."""
    if not row.get("terminated"):
        return "unterminated"
    if not row.get("text", "").strip():
        return "empty"
    if row.get("has_special") or SPECIAL_TEXT.search(row["text"]):
        return "special_token"
    return None


def read_answers(config: TextAdapterConfig) -> tuple:
    """``({prompt id: [answer text, ...]}, drop counts)``, answers in sample order."""
    answers_dir = config.answers_dir()
    if not os.path.isdir(answers_dir):
        raise TextBuildError(f"{answers_dir} does not exist")
    rows: dict = {}
    drops: dict = {}
    for name in sorted(os.listdir(answers_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(answers_dir, name)) as fh:
            for line in fh:
                row = json.loads(line)
                reason = usable_answer(row)
                if reason:
                    drops[reason] = drops.get(reason, 0) + 1
                    continue
                rows.setdefault(row["id"], []).append((int(row["sample"]), row["text"]))
    return ({pid: [text for _s, text in sorted(found)] for pid, found in rows.items()},
            drops)


# ─────────────────────────────────────────────────────────────────────────────
# Partition
# ─────────────────────────────────────────────────────────────────────────────

def partition(config: TextAdapterConfig) -> Partition:
    """One prompt, one role, as `tools/replay_prompts.py` assigned it."""
    claims = {}
    for prompt in read_prompts(config):
        claims.setdefault(prompt["role"], []).append(prompt["id"])
    return build_partition(
        [Claim("replay", role, tuple(keys)) for role, keys in sorted(claims.items())],
        {"sources": {"replay": {role: len(keys) for role, keys in claims.items()}}})


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def task_specs(config: TextAdapterConfig, arm: str = "graph") -> dict:
    """``{name: TaskSpec}`` for everything under ``text/``.

    ``cap_per_pass`` is the train examples one pass holds, which is a property of
    the build like ``mean_tokens`` — so it comes from the manifest and is
    ``None`` until one exists.

    **No evaluation splits.** The only generative read on this task that means
    anything is `text_behaviour`'s, over the 48 fixed probes: BLEU against one
    sampled continuation measures agreement with a coin flip, and generating 768
    tokens for 500 val prompts at every firing would buy exactly that. The val
    split is built and never trained on, so a loss-curve validator can read it
    when one exists.
    """
    manifest = read_manifest(config)
    entry = (manifest.get("tasks", {}) or {}).get(REPLAY_NAME, {})
    by_arm = entry.get("arms", {}).get(arm, {})
    return {REPLAY_NAME: TaskSpec(
        name=REPLAY_NAME, domain=DOMAIN, adapter=ADAPTER_NAME, kind="generator",
        answer_kind="text", metric="caption_rate", passes=1,
        cap_per_pass=by_arm.get("train_size"), max_new_tokens=MAX_NEW_TOKENS,
        build_version=config.build_version(), eval_splits=(),
        mean_tokens=by_arm.get("mean_tokens"), train_size=by_arm.get("train_size"),
        question_template=None)}


def register_text_tasks(registry: Registry, config: TextAdapterConfig,
                        arm: str = "graph") -> Registry:
    for spec in task_specs(config, arm=arm).values():
        registry.register(spec)
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Building
# ─────────────────────────────────────────────────────────────────────────────

def _draws(prompts: list, answers: dict, split: str, pass_id: int) -> tuple:
    """``[(prompt row, answer text, answer index)]`` for one split and pass."""
    draws, stats = [], {"prompts": 0, "no_answer": 0, "repeats": 0}
    for prompt in prompts:
        if prompt["role"] != split:
            continue
        found = answers.get(prompt["id"], [])
        if not found:
            stats["no_answer"] += 1
            continue
        stats["prompts"] += 1
        weight = int(prompt.get("weight", 1)) if split == "train" else 1
        for j in range(weight):
            index = (pass_id * weight + j) % len(found)
            if pass_id * weight + j >= len(found):
                stats["repeats"] += 1
            draws.append((prompt, found[index], index))
    return draws, stats


def _labels_fn(tokenizer, max_length: int, answer_prefix: str):
    """`schema.render`'s span over the text after the last assistant header.

    The answer is read back out of the node text rather than passed in because
    `compute_labels` hands its callable a row and no index; ``SPECIAL_TEXT`` has
    already refused any answer that could spell a second assistant header.
    """

    def get_labels(example):
        text = example["text"][example["prompt_node"]]
        index = text.rfind(answer_prefix)
        if index < 0:
            raise SchemaError(f"node {text[:80]!r} has no {answer_prefix!r}")
        stub = Example(task="_", domain=DOMAIN, split="train", arm="flat",
                       graph={"text": [text], "prompt_node": 0, "num_nodes": 1},
                       question="_", answer=text[index + len(answer_prefix):],
                       answer_kind="text", key="_")
        return render(stub, tokenizer, max_length=max_length).labels

    return get_labels


def _materialise(config, split, arm, pass_id, draws, spec, tokenizer):
    import networkx as nx

    from ...experiments.molecules.data import prompt_format
    from ...utils import TextGraphDataset

    fmt = prompt_format(config.prompt_style, config.model_name)
    texts = [f"{fmt.question(p['text'])}{fmt.answer_prefix}{answer}{fmt.answer_suffix}"
             for p, answer, _i in draws]

    # Over-length rows are dropped *before* the dataset exists, so truncation is
    # unreachable rather than detected afterwards.
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    kept = [i for i, n in enumerate(lengths) if n <= config.max_length]
    over_length = len(draws) - len(kept)
    draws = [draws[i] for i in kept]
    texts = [texts[i] for i in kept]

    graphs = []
    for text in texts:
        graph = nx.DiGraph()
        graph.add_node(0, text=text, kind="prompt")
        graph.graph["prompt_node"] = 0
        graphs.append(graph)

    ds = TextGraphDataset(graphs)
    ds.tokenize(tokenizer, max_length=config.max_length)
    ds.compute_labels(_labels_fn(tokenizer, config.max_length, fmt.answer_prefix),
                      num_proc=1)
    ds.compute_shortest_path_distances()
    ds.compute_magnetic_lap(q=config.magnetic_q, m=config.magnetic_m)
    ds.cast_float_features_to_fp32()

    records, num_nodes, num_tokens = [], [], []
    for i, (prompt, answer, index) in enumerate(draws):
        meta = {"source": prompt["source"], "sub": prompt["sub"],
                "chem": bool(prompt["chem"]), "answer_index": index}
        item = _item(ds, i)
        example = Example(task=REPLAY_NAME, domain=DOMAIN, split=split, arm=arm,
                          graph=item, question=prompt["text"], answer=answer,
                          answer_kind="text", key=prompt["id"], meta=meta)
        validate(example, spec)
        labels = item["labels"]
        if labels[-1] == -100 or len(item["input_ids"][0]) != lengths[kept[i]]:
            raise TextBuildError(
                f"{REPLAY_NAME}/{split}: row {i} lost its terminator or was "
                "truncated after the length filter; the filter and the tokenize "
                "call disagree about the text")
        records.append({"question": prompt["text"], "answer": answer,
                        "key": prompt["id"], "meta": meta})
        num_nodes.append(1)
        num_tokens.append(len(item["input_ids"][0]))

    path = config.source_path(REPLAY_NAME, split, arm, pass_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save(path)
    sidecar = {"task": REPLAY_NAME, "split": split, "arm": arm,
               "pass_id": int(pass_id), "domain": DOMAIN, "answer_kind": "text",
               "build_version": config.build_version(), "records": records,
               "num_nodes": num_nodes, "num_tokens": num_tokens,
               "dropped_over_length": over_length}
    with open(_sidecar_path(path), "w") as fh:
        json.dump(sidecar, fh)
    return sidecar


def _sidecar_path(path: str) -> str:
    return path + ".schema.json"


def _item(ds, i: int) -> dict:
    """One item, keyed like a molecule item so the two collate identically."""
    item = dict(ds[i])
    item["ds_label"] = REPLAY_NAME
    item["question_node"] = -1
    return item


def splits_for(task: str) -> tuple:
    return ("train", "val")


def build(config: TextAdapterConfig, tasks=None, arms=("graph", "flat"),
          passes: int = 1, rebuild: bool = False) -> dict:
    """Materialise ``passes`` train passes and one val pass per arm."""
    from transformers import AutoTokenizer

    config.validate()
    tasks = tuple(tasks) if tasks is not None else (REPLAY_TASK,)
    unknown = [t for t in tasks if t not in (REPLAY_TASK, REPLAY_NAME)]
    if unknown:
        raise TextBuildError(f"no such text task(s) {unknown}; have {REPLAY_NAME!r}")

    part = partition(config)
    prompts = read_prompts(config)
    answers, answer_drops = read_answers(config)
    specs = task_specs(config)
    spec = specs[REPLAY_NAME]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    manifest = read_manifest(config)
    manifest["build_version"] = config.build_version()
    manifest["generation"] = config.generation()
    manifest["answers_dropped"] = answer_drops
    manifest["partition"] = {"counts": part.counts, "role_totals": part.role_totals}
    entry = manifest.setdefault("tasks", {}).setdefault(
        REPLAY_NAME, {"arms": {}, "splits": {}})

    for split in ("train", "val"):
        for pass_id in range(passes if split == "train" else 1):
            draws, stats = _draws(prompts, answers, split, pass_id)
            for prompt, _answer, _index in draws:
                if not part.is_role(prompt["id"], split):
                    raise PartitionError(
                        f"{REPLAY_NAME}/{split}: prompt {prompt['id']} has role "
                        f"{part.role(prompt['id'])!r}")
            for arm in arms:
                path = _sidecar_path(config.source_path(REPLAY_NAME, split, arm, pass_id))
                if rebuild or not os.path.exists(path):
                    sidecar = _materialise(config, split, arm, pass_id, draws, spec,
                                           tokenizer)
                else:
                    with open(path) as fh:
                        sidecar = json.load(fh)
                stats["over_length"] = sidecar["dropped_over_length"]
                stats["emitted"] = len(sidecar["records"])
                if split == "train" and pass_id == 0:
                    tokens = sidecar["num_tokens"]
                    entry["arms"][arm] = {
                        "train_size": len(tokens),
                        "mean_tokens": sum(tokens) / len(tokens) if tokens else None}
            entry["splits"][f"{split}.p{pass_id}"] = stats

    write_manifest(config, manifest)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# The TaskSource
# ─────────────────────────────────────────────────────────────────────────────

class TextTaskSource:
    """One built ``(task, split, arm, pass)``. See `adapters.TaskSource`."""

    def __init__(self, dataset, sidecar: dict, path: str):
        self._ds = dataset
        self._records = sidecar["records"]
        self._num_nodes = sidecar["num_nodes"]
        self._num_tokens = sidecar["num_tokens"]
        self.path = path
        self.task = sidecar["task"]
        self.split = sidecar["split"]
        self.arm = sidecar["arm"]
        self.pass_id = int(sidecar["pass_id"])
        self.domain = sidecar["domain"]
        self.answer_kind = sidecar["answer_kind"]
        self.build_version = sidecar["build_version"]
        if len(self._records) != len(self._ds):
            raise TextBuildError(
                f"{path}: {len(self._records)} records beside {len(self._ds)} graphs")

    @property
    def dataset(self):
        return self._ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> dict:
        record = self._records[i]
        return Example(task=self.task, domain=self.domain, split=self.split,
                       arm=self.arm, graph=_item(self._ds, i),
                       question=record["question"],
                       answer=record["answer"], answer_kind=self.answer_kind,
                       key=record["key"], meta=record["meta"]).to_item()

    def example(self, i: int) -> Example:
        return Example.from_item(self[i], None, split=self.split)

    def keys(self) -> list:
        return [record["key"] for record in self._records]

    def lengths(self) -> tuple:
        return list(self._num_nodes), list(self._num_tokens)

    def __repr__(self) -> str:
        return (f"<TextTaskSource {self.task} {self.split}/{self.arm} "
                f"p{self.pass_id} n={len(self)}>")


def load(task: str, split: str, arm: str, pass_id: int = 0, config=None,
         check_keys: int = 200) -> TextTaskSource:
    """The built source for one ``(task, split, arm, pass)``. Never regenerates."""
    if config is None:
        raise TextBuildError("text.load needs config=; it has no process-wide default")
    if task not in (REPLAY_NAME, REPLAY_TASK):
        raise TextBuildError(f"{task}: not a text task (have {REPLAY_NAME!r})")
    if split not in splits_for(task):
        raise TextBuildError(f"{task}: split {split!r} is not one of {splits_for(task)}")
    path = config.source_path(REPLAY_NAME, split, arm, pass_id)
    if not os.path.exists(_sidecar_path(path)):
        raise TextBuildError(
            f"{path} has not been built. Run data_prep for {REPLAY_NAME} "
            f"({split}/{arm}, pass {pass_id}).")
    with open(_sidecar_path(path)) as fh:
        sidecar = json.load(fh)

    from ...utils import TextGraphDataset

    source = TextTaskSource(TextGraphDataset.load(path), sidecar, path)
    if check_keys:
        part = partition(config)
        keys = source.keys()
        rng = random.Random(f"recheck|{task}|{split}|{pass_id}")
        for key in keys if len(keys) <= check_keys else rng.sample(keys, check_keys):
            if not part.is_role(key, split):
                raise PartitionError(
                    f"{task}/{split}: built key {key} has role {part.role(key)!r}; "
                    "the artifact and prompts.jsonl disagree, rerun data_prep")
    return source


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def read_manifest(config: TextAdapterConfig) -> dict:
    if not config.generated():
        return {}
    path = config.manifest_path()
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def write_manifest(config: TextAdapterConfig, manifest: dict) -> str:
    path = config.manifest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path
