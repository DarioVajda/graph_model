"""
Build the `TextGraphDataset` for one (task, arm, encoding) — Tier A for now.

**The flat arm is a single-node graph.** By Property 2 (`CLAUDE_CONTEXT.md` §2.3)
GTLM's forward pass on a single-node graph is *exactly* the base LLM's, so the
flat control needs no separate trainer, no second code path, and no argument about
whether the two arms were trained comparably: they run through the same model, the
same collator, the same optimizer and the same metric. Only the input
representation differs, which is the entire point of a control. (This is the same
observation `src/generalist/PLAN.md` §1 makes about `adapters/text.py`.)

Both arms are supervised on the **last token of the prompt node**, which
`tasks.py` guarantees is the whole answer.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
from collections import Counter

from rdkit import Chem
from tqdm import tqdm
from transformers import AutoTokenizer

from ...utils import TextGraphDataset
from .data import (
    HELD_OUT_DATASETS,
    HELD_OUT_TIER_A_TASKS,
    TIER_B,
    attach_question,
    flat_serialize,
    load_tier_b,
    mol_to_graph,
    relabel_for_dataset,
    scaffold_split,
)
from .tasks import ANSWER_VOCAB, ATOM_LEVEL_TASKS, TASK_GENERATORS, TIER_A_TASKS
from .tier_b import TIER_B_TASKS, build_tier_b_examples

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EXPERIMENT_DIR, "datasets")

ARMS = ("graph", "flat")

#: One task axis over both tiers. A Tier-A name selects a generator; a Tier-B
#: name selects a MoleculeNet corpus. Keeping them on one axis is what lets a
#: later multi-task mixture (PLAN.md §4 arm 2) just list task names.
ALL_TASKS = TIER_A_TASKS + TIER_B_TASKS


def tier_of(task):
    return "A" if task in TIER_A_TASKS else "B"

#: The molecule pool Tier A draws from. Deliberately the Tier-B corpus: the same
#: chemistry the property tasks use, so a chemistry-generalist run (PLAN.md §4
#: arm 2) is measuring transfer between tasks rather than between distributions.
DEFAULT_POOL = ("hiv", "bace", "bbbp", "tox21", "lipo")


def get_prompt_node_labels(example):
    """Supervise the final token of the prompt node only; mask everything else.

    Same contract as `expressiveness`/`probes`. `tasks.py` emits single-token
    answers (` Yes`/` No`, or a numeral, whose numeral is the last token), so this
    supervises exactly the answer and nothing else.
    """
    labels = example["input_ids"][example["prompt_node"]].copy()
    labels[:-1] = [-100] * (len(labels) - 1)
    return labels


def build_graph_example(mol, question, answer, named_atoms, cfg):
    """Graph arm: atoms (+ Levi bond nodes) + an edge-free QUESTION node + PROMPT."""
    atom_level = cfg.task in ATOM_LEVEL_TASKS
    graph = mol_to_graph(mol, encoding=cfg.encoding, stereo_tags=cfg.stereo_tags,
                         atom_labels=atom_level)
    graph = attach_question(
        graph, question, answer,
        named_atoms=named_atoms,
        # Atom-level questions wire the prompt to the atoms they name; molecule-
        # level ones wire to every atom, because a prompt node with no edges has a
        # constant SPD row and the graph arm would be structurally blank exactly
        # where the answer is generated (`project-isolated-prompt-node`).
        prompt_edges="named" if (atom_level and named_atoms) else "all",
        question_node=cfg.question_node)
    return relabel_for_dataset(graph)


def build_flat_example(mol, question, answer, cfg):
    """Flat arm: ONE node holding question + SMILES + answer. Exactly base Llama."""
    import networkx as nx

    smiles = flat_serialize(mol, atom_labels=(cfg.task in ATOM_LEVEL_TASKS))
    graph = nx.DiGraph()
    graph.add_node(0, text=f"{question}\nSMILES: {smiles}\nA:{answer}", kind="prompt")
    graph.graph["prompt_node"] = 0
    return graph


#: Tier-A families that yield exactly ONE example per molecule: the question is
#: fixed for the family and the answer is a function of the molecule alone. A
#: molecule can therefore contribute at most one example, so a split can never be
#: larger than its share of the pool. Every other family varies something per draw
#: (the named atom, or which functional group is asked about) and can emit several
#: distinct examples from one molecule.
SINGLE_EXAMPLE_TASKS = frozenset({
    "longest_chain", "ring_count", "stereo_potential", "stereo_assigned",
})


def _molecule_pool(cfg):
    """The molecules Tier A draws from, deterministically ordered."""
    pool = []
    for name in cfg.pool:
        if name not in TIER_B:
            raise ValueError(f"unknown molecule source {name!r} (have {sorted(TIER_B)})")
        records, _, _ = load_tier_b(name)
        pool.extend(r["mol"] for r in records)
    return pool


def split_molecule_pool(cfg):
    """Partition the Tier-A pool into MOLECULE-DISJOINT train/val/test sets.

    This is what keeps a test example from being a memorised training example. The
    generator answers a question about a molecule, so a molecule appearing in two
    splits puts the answer on both sides of the boundary — and for a
    `SINGLE_EXAMPLE_TASKS` family that is an exact duplicate, same graph, same
    question, same answer.

    Bemis-Murcko scaffold, not a random partition, for the same reason Tier B uses
    one: it makes the test set *structurally* novel rather than merely unseen, which
    is the property a structural-reasoning claim actually needs. It also reuses the
    split Tier B already has under test.

    Pool fractions follow the requested example counts, so "a single-example family
    needs a pool at least as large as the total number of examples requested" is the
    whole sizing rule.
    """
    pool = _molecule_pool(cfg)
    total = cfg.train_size + cfg.val_size + cfg.test_size
    train_idx, val_idx, test_idx = scaffold_split(
        [Chem.MolToSmiles(m, canonical=True) for m in pool],
        frac_train=cfg.train_size / total,
        frac_valid=cfg.val_size / total,
        frac_test=cfg.test_size / total,
    )
    return {"train": [pool[i] for i in train_idx],
            "val": [pool[i] for i in val_idx],
            "test": [pool[i] for i in test_idx]}


def generate_examples(cfg, n, rng, molecules, split=""):
    """Draw ``n`` examples for ``cfg.task`` from ``molecules``. Generators may refuse.

    Molecules are consumed **without replacement** within a pass: the list is
    shuffled and walked, so no molecule is used twice until every usable one has
    been used once. A family that can emit several distinct examples per molecule
    (a different named atom, a different functional group) takes further passes; a
    `SINGLE_EXAMPLE_TASKS` family cannot, and asking for more examples than the
    split has molecules is an error rather than a silent duplicate.

    A second pass can still land on a (molecule, question) pair the first pass
    already emitted, because the generator picks the named atom at random. That is
    a within-split repeat, not a train/test leak -- it costs effective sample size
    and nothing else -- but it is exactly the class of thing that went unnoticed
    last time, so it is COUNTED into ``stats["repeats"]`` and travels into the run
    record rather than staying invisible.
    """
    generator = TASK_GENERATORS[cfg.task]
    single = cfg.task in SINGLE_EXAMPLE_TASKS
    graphs, stats = [], {"answers": {}, "attempts": 0, "molecules": 0, "repeats": 0}
    used = set()
    emitted = set()

    with tqdm(total=n, desc=f"Generating {cfg.task}/{cfg.arm}{'/' + split if split else ''}") as bar:
        while len(graphs) < n:
            order = list(range(len(molecules)))
            rng.shuffle(order)
            produced_this_pass = 0

            for i in order:
                if len(graphs) >= n:
                    break
                stats["attempts"] += 1
                made = generator(molecules[i], rng)
                if made is None:
                    continue
                question, answer, named = made
                if cfg.arm == "graph":
                    graph = build_graph_example(molecules[i], question, answer, named, cfg)
                else:
                    graph = build_flat_example(molecules[i], question, answer, cfg)
                graphs.append(graph)
                used.add(i)
                if (i, question, answer) in emitted:
                    stats["repeats"] += 1
                emitted.add((i, question, answer))
                produced_this_pass += 1
                stats["answers"][answer] = stats["answers"].get(answer, 0) + 1
                bar.update(1)

            if len(graphs) >= n:
                break
            if single:
                raise ValueError(
                    f"{cfg.task!r} yields one example per molecule, but the {split or 'this'} "
                    f"split has only {produced_this_pass} usable molecules of "
                    f"{len(molecules)} and {n} examples were requested. Repeating a "
                    "molecule would put an identical example in the split twice. Either "
                    "widen `pool` (RunConfig.pool defaults to five corpora; the §3.2 "
                    f"sweeps set only bace,bbbp) or lower the split size.")
            if produced_this_pass == 0:
                raise RuntimeError(
                    f"{cfg.task}: the generator refused every molecule in the "
                    f"{split or 'this'} split ({len(molecules)} molecules).")

    stats["molecules"] = len(used)
    return graphs, stats


def _build_split_graphs(items, cfg):
    """Turn ``[(mol, question, answer), ...]`` into arm-appropriate graphs."""
    graphs = []
    for mol, question, answer in tqdm(items, desc=f"Building {cfg.task}/{cfg.arm}"):
        if cfg.arm == "graph":
            # Tier B names no atom, so the prompt wires to every atom (see
            # `build_graph_example`) and atom labels stay off.
            graphs.append(build_graph_example(mol, question, answer, [], cfg))
        else:
            graphs.append(build_flat_example(mol, question, answer, cfg))
    return graphs


def prepare_tier_b_graphs(cfg):
    """Scaffold-split Tier-B graphs, ordered [train..., val..., test...].

    Caps subsample *randomly under `data_seed`*, never by slicing: the scaffold
    split emits groups largest-first, so a slice would take the most common
    scaffolds and quietly change the task.
    """
    splits, stats = build_tier_b_examples(cfg.task)
    rng = random.Random(cfg.data_seed)

    ordered, sizes, by_split = [], {}, {}
    for name in ("train", "val", "test"):
        items = splits[name]
        cap = cfg.max_train_examples if name == "train" else cfg.max_eval_examples
        if cap and len(items) > cap:
            items = rng.sample(items, cap)
        ordered.extend(items)
        sizes[name] = len(items)
        by_split[name] = dict(Counter(answer for _mol, _question, answer in items))

    # The answer distribution of what actually ends up in the artifact — computed
    # here rather than in `build_tier_b_examples` so a cap is reflected rather than
    # described. `answers` is the aggregate `_answer_stats` reads by default;
    # `answers_by_split` exists because Tier B's scaffold split moves the base rate
    # a long way between train and test (BBBP: 0.822 -> 0.524), so the floor a TEST
    # headline has to beat is not the corpus-wide one. PLAN.md §1 Tier B.
    stats["answers_by_split"] = by_split
    stats["answers"] = dict(sum((Counter(v) for v in by_split.values()), Counter()))
    stats["used_split_sizes"] = sizes
    return _build_split_graphs(ordered, cfg), stats, sizes


def dataset_path(cfg):
    """Artifact path encoding everything that changes the generated content."""
    if tier_of(cfg.task) == "B":
        tags = [cfg.task, cfg.arm, "scaffold"]
        if cfg.max_train_examples or cfg.max_eval_examples:
            tags.append(f"cap{cfg.max_train_examples}-{cfg.max_eval_examples}")
    else:
        total = cfg.train_size + cfg.val_size + cfg.test_size
        # `molsplit` marks a dataset built from MOLECULE-DISJOINT scaffold splits.
        # It is in the path so the artifacts built before that fix — where ~70% of a
        # test split was also in train — can never be silently loaded by the fixed
        # code. Their paths lack the tag, so they simply do not match.
        tags = [cfg.task, cfg.arm, f"{total}ex", "molsplit", "-".join(cfg.pool)]
    if cfg.arm == "graph":
        tags.append(cfg.encoding)
        tags.append("st1" if cfg.stereo_tags else "st0")
        # `question_node` changes the graph itself (it adds a node), so it belongs
        # in the cache key. The default is left UNTAGGED, which is what keeps every
        # already-built cache valid across the 2026-08-29 "isolated" -> "on"
        # rename: the value changed spelling, the path did not.
        if cfg.question_node != "on":
            tags.append(f"q{cfg.question_node}")
    model = str(cfg.model_name).replace("/", "-")
    tags += [model, cfg.ordering, f"ds{cfg.data_seed}"]
    return os.path.join(DATASETS_DIR, "_".join(tags) + ".gtds")


def prepare_dataset(cfg):
    """Generate + featurize the full (train+val+test) dataset. Deterministic."""
    if tier_of(cfg.task) == "B":
        graphs, stats, sizes = prepare_tier_b_graphs(cfg)
    else:
        # Generate PER SPLIT, from molecule-disjoint pools. Generating one stream and
        # slicing it is what let a molecule land in train and test at once.
        rng = random.Random(cfg.data_seed)
        pools = split_molecule_pool(cfg)
        wanted = {"train": cfg.train_size, "val": cfg.val_size, "test": cfg.test_size}

        graphs, sizes, by_split = [], {}, {}
        stats = {"answers": {}, "attempts": 0, "molecules": 0, "repeats_by_split": {},
                 "pool_sizes": {k: len(v) for k, v in pools.items()}}
        for name in ("train", "val", "test"):
            part, part_stats = generate_examples(
                cfg, wanted[name], rng, pools[name], split=name)
            graphs.extend(part)
            sizes[name] = len(part)
            by_split[name] = part_stats["answers"]
            stats["repeats_by_split"][name] = part_stats["repeats"]
            stats["attempts"] += part_stats["attempts"]
            stats["molecules"] += part_stats["molecules"]
            for answer, count in part_stats["answers"].items():
                stats["answers"][answer] = stats["answers"].get(answer, 0) + count
        # Same shape Tier B records, so `_answer_stats` takes the floor from the
        # TEST split for both tiers rather than from a mixed-distribution total.
        stats["answers_by_split"] = by_split
    stats["split_sizes"] = sizes

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    for answer in ANSWER_VOCAB:
        n = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
        if n > 2:
            raise AssertionError(
                f"answer {answer!r} tokenizes to {n} tokens; last-token "
                "supervision would not cover it")

    ds = TextGraphDataset(graphs, rcm_ordering=(cfg.ordering == "rcm"))
    ds.tokenize(tokenizer)
    ds.compute_labels(get_prompt_node_labels)
    # Both features always, so ONE artifact serves every bias arm. On the flat
    # arm these are 1x1 tensors — free, and it keeps the two arms' pipelines
    # byte-identical downstream.
    ds.compute_shortest_path_distances()
    ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m)
    ds.cast_float_features_to_fp32()
    return ds, stats


def load_or_create_dataset(cfg):
    """Load this config's `.gtds`, generating it if absent (flock'd for sbatch)."""
    path = dataset_path(cfg)
    if not os.path.exists(path):
        os.makedirs(DATASETS_DIR, exist_ok=True)
        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(path):
                print(f"Dataset not found at {path}. Generating...")
                ds, stats = prepare_dataset(cfg)
                ds.save(path)
                with open(path + ".meta.json", "w") as f:
                    json.dump(stats, f, indent=2)
                # `.get`, not `[...]`: this is a progress print, and it crashed
                # every Tier-B run at 010 because the Tier-B stats dict had no
                # `answers` key. A log line must never be what fails a job.
                print(f"[data] answer distribution: {stats.get('answers')}")
    ds = TextGraphDataset.load(path)
    print(f"Loaded dataset from {path} with {len(ds)} examples.")
    return ds


def load_dataset_stats(cfg):
    """The generation stats sidecar (answer distribution, split sizes), or ``{}``.

    Written once when the `.gtds` is generated. Read it rather than the generation
    print, because a run that hits a warm cache never prints it — which is most
    runs in a sweep, and exactly the ones whose base rate you later want.
    """
    path = dataset_path(cfg) + ".meta.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_data(cfg):
    """Return ``(train, val, test)``.

    Refuses to build a *training* split for a held-out task. The held-out
    declaration (PLAN.md §4.1) is only worth something if it is enforced in code
    rather than remembered — this is that enforcement.
    """
    held_out = set(HELD_OUT_TIER_A_TASKS) | set(HELD_OUT_DATASETS)
    if cfg.task in held_out and not cfg.held_out_eval:
        raise ValueError(
            f"{cfg.task!r} is permanently held out (PLAN.md §4.1) and must never "
            "enter a training mixture. Pass --held-out-eval to build it for "
            "held-out EVALUATION only.")

    ds = load_or_create_dataset(cfg)
    sizes = _split_sizes(cfg, ds)
    total = sum(sizes.values())
    if len(ds) < total:
        raise ValueError(
            f"Dataset at {dataset_path(cfg)} has {len(ds)} examples but {total} "
            "are configured — stale artifact. Delete it to regenerate.")
    train_end = sizes["train"]
    val_end = train_end + sizes["val"]
    return ds[:train_end], ds[train_end:val_end], ds[val_end:total]


def _split_sizes(cfg, ds):
    """Split sizes for this artifact.

    Tier A's are configured; Tier B's are a property of the scaffold split and are
    read back from the artifact's meta file, so a cap or a corpus change cannot
    silently shift the boundaries of an already-built dataset.
    """
    if tier_of(cfg.task) != "B":
        return {"train": cfg.train_size, "val": cfg.val_size, "test": cfg.test_size}
    meta_path = dataset_path(cfg) + ".meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} is missing; Tier-B split boundaries live there. "
            "Delete the .gtds and rebuild.")
    with open(meta_path) as f:
        return json.load(f)["split_sizes"]


def run_data_prep_mode(cfg):
    load_or_create_dataset(cfg)
    print("[data_prep] done.")
