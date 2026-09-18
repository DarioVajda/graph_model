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
    NOTATION_HEADERS,
    TIER_B,
    UNENCODABLE,
    EncodeUnsupported,
    attach_question,
    flat_serialize,
    load_tier_b,
    mol_to_graph,
    prompt_format,
    relabel_for_dataset,
    scaffold_split,
)
from .chebi import (
    CHEBI_TASK,
    DEFAULT_HEAVY_ATOM_CAP,
    build_chebi_examples,
)
from .tasks import ANSWER_VOCAB, ATOM_LEVEL_TASKS, TASK_GENERATORS, TIER_A_TASKS
from .tier_b import TIER_B_TASKS, build_tier_b_examples

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EXPERIMENT_DIR, "datasets")

#: The notation arms are flat arms in a different molecular string
#: (`data.NOTATIONS`); the generalist builds them for §9's ladder and the
#: specialist campaign never used them. They are listed so a config carrying one
#: validates rather than failing on an arm this package can serialise perfectly
#: well.
ARMS = ("graph", "flat", "flat_selfies", "flat_inchi")

#: One task axis over both tiers. A Tier-A name selects a generator; a Tier-B
#: name selects a MoleculeNet corpus. Keeping them on one axis is what lets a
#: later multi-task mixture (PLAN.md §4 arm 2) just list task names.
ALL_TASKS = TIER_A_TASKS + TIER_B_TASKS + (CHEBI_TASK,)


#: Tier C — ChEBI-20 captioning (`chebi.py`). One task, and it is the only one in
#: this package whose answer is free text rather than a token or a yes/no.
TIER_C_TASKS = (CHEBI_TASK,)


def tier_of(task):
    if task in TIER_C_TASKS:
        return "C"
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


def make_caption_labels(tokenizer, cfg):
    """Labels for Tier C: supervise the **whole answer span**, mask the prefix.

    Tiers A and B are one supervised token, which is why
    :func:`get_prompt_node_labels` masks all but the last. A caption is 50-100
    tokens, so the mask has to fall at the answer boundary instead.

    The boundary is the tokenized length of the prompt node's *prefix* — the
    ``answer_prefix`` (plain) or the assistant turn header (chat) that
    `attach_question` puts before the answer. That is a prefix of a longer string
    being tokenized as a whole, so BPE could in principle merge across the seam
    and make the count wrong by one. `verify_caption_labels` decodes the
    supervised span back and refuses the build if it is not exactly the answer,
    which is the assertion that makes this safe rather than probable.
    """
    fmt = prompt_format(getattr(cfg, "prompt_style", None), cfg.model_name)
    n_prefix = len(tokenizer(fmt.answer_prefix, add_special_tokens=False)["input_ids"])

    def labels(example):
        ids = example["input_ids"][example["prompt_node"]]
        out = list(ids)
        out[:n_prefix] = [-100] * min(n_prefix, len(out))
        return out

    labels.n_prefix = n_prefix
    return labels


def verify_generative_stop_token(ds, tokenizer, sample=64):
    """A generated answer must END with the stop token. Refuses the build if not.

    `PLAN.md` §9's rule, applied to the one quantity that has already cost this
    project two months: **a model never shown an end-of-text token does not learn
    where its answer stops.** It writes the right thing and runs on to the
    generation cap, and every generative metric then scores *stopping* rather
    than correctness — `generalist/MOLECULE_GENERALIST.md` §8.5, where a graph arm
    emitting the exactly-correct SMILES 46.5 % of the time as a prefix of its
    output was reported as 0.0000 exact match for two months.

    That defect was reintroduced here on 2026-09-18 by porting Tier C without
    carrying `add_eos` across: Tiers A and B answer in one teacher-forced token
    and correctly leave it off, so the default is wrong for exactly one tier. The
    build-time smoke did not catch it, because the smoke only established that
    training *ran* — it never looked at what was being supervised. This assert is
    what makes that impossible to repeat, and it is deliberately not a warning.
    """
    import random as _random

    if tokenizer.eos_token_id is None:
        raise AssertionError("a generative tier needs a tokenizer with an eos_token_id")
    rng = _random.Random(0)
    for i in rng.sample(range(len(ds)), min(sample, len(ds))):
        row = ds[i]
        ids = row["input_ids"][row["prompt_node"]]
        if int(ids[-1]) != int(tokenizer.eos_token_id):
            raise AssertionError(
                f"example {i}'s prompt node does not end with the stop token "
                f"({tokenizer.eos_token_id}); it ends with {int(ids[-1])}. A "
                "generative answer without one trains a model that never stops, "
                "and every caption metric computed from it scores stopping "
                "rather than correctness (see this function's docstring).")
    return True


def verify_caption_labels(ds, tokenizer, answers, n_prefix, sample=64):
    """The supervised span must decode to exactly the answer. Refuses if not.

    `PLAN.md` §9's rule: an instrument that is only ever read has no
    error-detecting surface. A silently-off-by-one mask trains the model to
    predict the ``A:`` of its own prompt and drops the caption's first word, and
    nothing downstream would say so — the loss would simply be a little worse.
    """
    import random as _random

    rng = _random.Random(0)
    n = min(sample, len(ds))
    for i in rng.sample(range(len(ds)), n):
        row = ds[i]
        ids = row["input_ids"][row["prompt_node"]]
        got = tokenizer.decode(ids[n_prefix:], skip_special_tokens=True)
        want = answers[i]
        if got.strip() != want.strip():
            raise AssertionError(
                f"caption label boundary is wrong at example {i}: the supervised "
                f"span decodes to {got[:80]!r} but the answer is {want[:80]!r}. "
                f"n_prefix={n_prefix} does not split this tokenizer's output.")
    return n


def prepare_chebi_graphs(cfg):
    """Tier C graphs, ordered [train..., val..., test...], on ChEBI-20's own split.

    **No scaffold split and no cross-source partition.** ChEBI-20 ships three
    files and the published baselines train and test on exactly those; applying
    our own split would make the number incomparable, and applying the
    generalist's D3.3 partition would withhold ChEBI *training* molecules that a
    MoleculeNet corpus happens to claim as test — a handicap no baseline pays
    (`chebi.py`).

    Caps subsample randomly under ``data_seed``, as Tier B does.
    """
    cap = getattr(cfg, "chebi_heavy_atom_cap", DEFAULT_HEAVY_ATOM_CAP)
    allow = getattr(cfg, "chebi_allow_disconnected", True)
    splits, stats = build_chebi_examples(heavy_atom_cap=cap,
                                         allow_disconnected=allow)
    rng = random.Random(cfg.data_seed)

    ordered, sizes = [], {}
    for name in ("train", "val", "test"):
        items = splits[name]
        limit = cfg.max_train_examples if name == "train" else cfg.max_eval_examples
        if limit and len(items) > limit:
            items = rng.sample(items, limit)
        ordered.extend(items)
        sizes[name] = len(items)

    # A caption is free text, so the Tier-A/B answer histogram is meaningless
    # here; what stands in for it is the caption length distribution, which is
    # what a reader needs to judge a BLEU against.
    lengths = [len(a.split()) for _m, _q, a in ordered]
    stats["answers"] = {}
    stats["answers_by_split"] = {}
    stats["caption_words"] = {
        "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }
    stats["used_split_sizes"] = sizes
    stats["heavy_atom_cap"] = cap
    stats["allow_disconnected"] = allow
    return _build_split_graphs(ordered, cfg), stats, sizes, ordered


def build_graph_example(mol, question, answer, named_atoms, cfg):
    """Graph arm: atoms (+ Levi bond nodes) + an edge-free QUESTION node + PROMPT."""
    atom_level = cfg.task in ATOM_LEVEL_TASKS
    graph = mol_to_graph(mol, encoding=cfg.encoding, stereo_tags=cfg.stereo_tags,
                         atom_labels=atom_level)
    graph = attach_question(
        graph, question, answer,
        fmt=prompt_format(getattr(cfg, "prompt_style", None), cfg.model_name),
        named_atoms=named_atoms,
        # Atom-level questions wire the prompt to the atoms they name; molecule-
        # level ones wire to every atom, because a prompt node with no edges has a
        # constant SPD row and the graph arm would be structurally blank exactly
        # where the answer is generated (`project-isolated-prompt-node`).
        prompt_edges="named" if (atom_level and named_atoms) else "all",
        question_node=cfg.question_node)
    return relabel_for_dataset(graph)


def build_flat_example(mol, question, answer, cfg):
    """Flat arm: ONE node holding question + the molecule string + answer.

    Exactly base Llama: one node, so every structural bias is identically zero.
    ``cfg.notation`` picks the string (`data.NOTATIONS`); the header names it, so
    a model reading InChI is not told it is reading SMILES and the three
    notations of §9's ladder differ in the string *and* in what they claim to be.
    """
    import networkx as nx

    notation = getattr(cfg, "notation", "smiles")
    try:
        text = flat_serialize(mol, atom_labels=(cfg.task in ATOM_LEVEL_TASKS),
                              notation=notation)
    except EncodeUnsupported:
        # A molecule this notation cannot express at all. Measured on the five
        # Tier-B corpora: 22 of 53,921 for SELFIES under `SELFIES_CONSTRAINTS`
        # (all in HIV — 9 train, 12 val, 1 test), 0 for InChI.
        #
        # **The row is kept, and it is kept so it can be dropped from every arm
        # rather than from one.** The notation ladder compares three strings for
        # the same molecules; dropping a row here and nowhere else would silently
        # shorten one arm's dataset and shift every index after it, so the arms
        # would no longer be scored on the same molecules. Keeping an unscoreable
        # placeholder preserves the alignment and makes the exclusion something a
        # reader can see and count (`notation_probe` drops `UNENCODABLE` rows from
        # all arms and reports how many).
        text = UNENCODABLE
    # The molecule string is part of the *user* turn: the flat arm's whole input
    # is that one node, so the turn has to close after the molecule and the
    # assistant turn opens where the answer does.
    fmt = prompt_format(getattr(cfg, "prompt_style", None), cfg.model_name)
    body = f"{question}\n{NOTATION_HEADERS[notation]}: {text}"
    graph = nx.DiGraph()
    graph.add_node(0, text=f"{fmt.question(body)}{fmt.answer_prefix}{answer}",
                   kind="prompt")
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
    if tier_of(cfg.task) == "C":
        # `own` marks the benchmark's own three files as the split, which is what
        # distinguishes this artifact from anything built under a scaffold split
        # or the generalist's cross-source partition. The heavy-atom cap is in the
        # path because it changes which molecules exist at all, and therefore the
        # denominator of every metric computed downstream.
        # `eos` marks a build whose captions carry a stop token. It is in the path
        # because it changes the supervised content, and an artifact built without
        # it scores whether the model stopped rather than whether it was right
        # (§8.5). A build made before the fix therefore cannot be silently reused:
        # its path lacks the tag, so it simply does not match.
        tags = [cfg.task, cfg.arm, "own", "eos",
                f"cap{getattr(cfg, 'chebi_heavy_atom_cap', DEFAULT_HEAVY_ATOM_CAP)}"]
        if not getattr(cfg, "chebi_allow_disconnected", True):
            tags.append("conn")
        if cfg.max_train_examples or cfg.max_eval_examples:
            tags.append(f"cap{cfg.max_train_examples}-{cfg.max_eval_examples}")
    elif tier_of(cfg.task) == "B":
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
    caption_answers = None
    if tier_of(cfg.task) == "C":
        graphs, stats, sizes, items = prepare_chebi_graphs(cfg)
        caption_answers = [answer for _m, _q, answer in items]
    elif tier_of(cfg.task) == "B":
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
    if tier_of(cfg.task) != "C":
        # Tier C's answer is a caption, so the single-token vocabulary check does
        # not apply to it; its boundary is checked by `verify_caption_labels`
        # after tokenization instead, which is the stronger statement of the same
        # requirement.
        for answer in ANSWER_VOCAB:
            n = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
            if n > 2:
                raise AssertionError(
                    f"answer {answer!r} tokenizes to {n} tokens; last-token "
                    "supervision would not cover it")

    ds = TextGraphDataset(graphs, rcm_ordering=(cfg.ordering == "rcm"))
    # ── TIER C MUST CARRY A STOP TOKEN, AND THIS PROJECT HAS PAID FOR IT ONCE ──
    # Tiers A and B answer in one token read teacher-forced, so where the answer
    # ends is never in question and `add_eos` stays off. A caption is generated,
    # and a model never shown an end-of-text token does not learn where a caption
    # stops: it writes the right thing and runs on to the generation cap, and the
    # metric then scores *stopping* rather than correctness.
    #
    # That is `generalist/MOLECULE_GENERALIST.md` §8.5 verbatim. It voided every
    # generation row of three campaigns, and for two months it was read as "the
    # graph arm cannot serialize a molecule" when the arm was in fact writing the
    # exactly-correct string 46.5 % of the time as a *prefix* of its output.
    #
    # On base weights the right token is `<|end_of_text|>` and not `<|eot_id|>`,
    # whose embedding row sits at the reserved block's norm — initialisation
    # rather than training, with a frozen output head. `tokenizer.eos_token_id`
    # resolves to the correct one on both backbones.
    ds.tokenize(tokenizer, add_eos=(tier_of(cfg.task) == "C"))
    if tier_of(cfg.task) == "C":
        labeller = make_caption_labels(tokenizer, cfg)
        ds.compute_labels(labeller)
        stats["caption_label_checked"] = verify_caption_labels(
            ds, tokenizer, caption_answers, labeller.n_prefix)
        stats["caption_prefix_tokens"] = labeller.n_prefix
        stats["stop_token_checked"] = verify_generative_stop_token(ds, tokenizer)
    else:
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

    Tier A's are configured; Tier B's and Tier C's are properties of the split
    (a scaffold split, or ChEBI-20's own three files) and are read back from the
    artifact's meta file, so a cap or a corpus change cannot silently shift the
    boundaries of an already-built dataset.
    """
    if tier_of(cfg.task) not in ("B", "C"):
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
