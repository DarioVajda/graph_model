"""
KGQA (SR-WebQSP) data preparation.

Turns GNN-RAG's SR-retrieved subgraphs into `.gtds` TextGraphDatasets that a
single GTLM consumes directly (replacing GNN-RAG's GNN-reasoner + LLM-reader).

Pipeline per question:
  raw SR record  ->  select triples (paths-guided size cap)
                 ->  directed per-triple Levi graph  (h -> rel -> t)
                 ->  single-parent CVT collapse       (p->rel0->cvt->rel_i->leaf  =>  p->rel0->rel_i->leaf)
                 ->  node text  (entities: entity_names.json only, else "unnamed entity";
                                 relations: last-segment verbalization)
                 ->  prompt node (question + graph-present answers), directed prompt -> topic
                 ->  full gold set stashed in graph.graph['gold_answers'] for the evaluator
  then TextGraphDataset: tokenize -> labels (mask up to "Answer:") -> SPD -> magnetic -> save

Node naming is v1 = entity_names.json ONLY. Harvesting answer `text` into node
text would leak the answer at eval (gold nodes would be the only newly-named
ones), so answer text feeds ONLY the target / eval matching.

This is the ``data_prep`` mode of the experiment. It consumes a ``RunConfig``
(not its own argparse) and is driven by ``run_data_prep_mode``; the entry point is

    python -m src.experiments.kgqa --mode data_prep [--max-nodes 512 --rel-mode last_1 ...]
"""

import os
import re
import json
import fcntl
import random
from collections import defaultdict, deque

import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer

from ...utils import TextGraphDataset

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
SR_DIR = os.path.join(EXPERIMENT_DIR, "data", "sr-webqsp")
ENTITY_NAMES_PATH = os.path.join(EXPERIMENT_DIR, "entities_names.json")
OUTPUT_ROOT = os.path.join(EXPERIMENT_DIR, "processed_datasets")

UNNAMED = "unnamed entity"
END_OF_HOP = "END OF HOP"
ANSWER_DELIM = "\nAnswer:"          # prompt = "{question}\nAnswer: a1, a2, ..."
ANSWER_SEP = ", "

SPLITS = {"train": "train.json", "dev": "dev.json", "test": "test.json"}


# --------------------------------------------------------------------------- #
# Node text: naming + relation verbalization
# --------------------------------------------------------------------------- #
def _decode_literal(s: str) -> str:
    """Freebase literal values are URL-ish encoded, e.g. 'Justin$002BBieber' -> 'Justin+Bieber'."""
    return re.sub(r"\$([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)


def resolve_entity_text(node, entity_names: dict) -> str:
    """v1 entity node text: entity_names.json only; unnamed MIDs -> 'unnamed entity'."""
    if not isinstance(node, str):
        return str(node)
    if node in entity_names:
        return entity_names[node]
    if node.startswith("m."):
        return UNNAMED                     # CVT or entity missing from the dict
    return _decode_literal(node)           # literal value node (date / number / string)


def verbalize_relation(rel: str, mode: str) -> str:
    """mode: 'last_1' -> property; 'last_2' -> type+property; 'full' -> whole dotted path."""
    if mode == "full":
        return rel.replace(".", " ").replace("_", " ")
    parts = rel.split(".")
    seg = " ".join(parts[-2:]) if mode == "last_2" else parts[-1]
    return seg.replace("_", " ")


# --------------------------------------------------------------------------- #
# Triple selection (answer-agnostic size cap via SR paths, then BFS proximity)
# --------------------------------------------------------------------------- #
def _instantiate_paths(record):
    """Triples traversed by the SR `paths`, round-robin-interleaved across paths.

    Each path is instantiated independently (following its relation sequence from
    the root); the per-path triple lists are then merged round-robin so no single
    high-fan-out path can starve the budget of later, answer-bearing paths.
    """
    by_hr = defaultdict(list)
    for tri in record["subgraph"]["tuples"]:
        by_hr[(tri[0], tri[1])].append(tuple(tri))

    per_path = []
    for path in record.get("paths", []):
        root, rels = path[0], path[1]
        frontier, tris = {root}, []
        for rel in rels:
            if rel == END_OF_HOP:
                break
            nxt = set()
            for u in frontier:
                for tri in by_hr.get((u, rel), []):
                    tris.append(tri)
                    nxt.add(tri[2])
            frontier = nxt
        per_path.append(tris)

    # round-robin merge, de-duplicating
    ordered, seen = [], set()
    for col in range(max((len(p) for p in per_path), default=0)):
        for p in per_path:
            if col < len(p) and p[col] not in seen:
                seen.add(p[col])
                ordered.append(p[col])
    return ordered


def _bfs_ordered(triples, all_tuples, topics):
    """Order `triples` by (undirected) hop-distance of their nearest endpoint to a topic."""
    adj = defaultdict(set)
    for h, _, t in all_tuples:
        adj[h].add(t)
        adj[t].add(h)
    dist, dq = {}, deque()
    for tp in topics:
        if tp in adj:
            dist[tp] = 0
            dq.append(tp)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    INF = 10 ** 9
    return sorted(triples, key=lambda tr: min(dist.get(tr[0], INF), dist.get(tr[2], INF)))


def _levi_node_estimate(triples):
    ents = set(x for t in triples for x in (t[0], t[2]))
    return len(ents) + len(set(triples))       # +1 for prompt added by caller's budget


def select_triples(record, max_nodes):
    """Return the triples to keep so the Levi graph (+ prompt) fits `max_nodes`."""
    tuples = [tuple(t) for t in record["subgraph"]["tuples"]]
    if _levi_node_estimate(tuples) + 1 <= max_nodes:
        return tuples

    topics = list(record["entities"])
    on_path = _instantiate_paths(record)
    on_path_set = set(on_path)
    remaining = [t for t in tuples if t not in on_path_set]
    priority = on_path + _bfs_ordered(remaining, tuples, topics)

    selected, ents = [], set()
    for tri in priority:
        new_ents = {tri[0], tri[2]} - ents
        # Levi nodes if we add this triple = |ents ∪ new| + (#selected + 1 rel node) + 1 prompt
        if len(ents) + len(new_ents) + (len(selected) + 1) + 1 > max_nodes:
            continue
        ents |= new_ents
        selected.append(tri)
    return selected


# --------------------------------------------------------------------------- #
# Levi construction + CVT collapse
# --------------------------------------------------------------------------- #
def build_base_levi(record, entity_names, rel_mode, max_nodes):
    """Directed per-triple Levi graph with node text and collapsed CVTs. No prompt node yet."""
    selected = select_triples(record, max_nodes)
    G = nx.DiGraph()
    for i, (h, rel, t) in enumerate(selected):
        rid = ("R", i)
        G.add_node(rid, text=verbalize_relation(rel, rel_mode), is_rel=True)
        G.add_edge(h, rid)
        G.add_edge(rid, t)

    # entity / value node text (relation nodes already have text)
    for n in G.nodes():
        if isinstance(n, tuple):
            continue
        G.nodes[n]["text"] = resolve_entity_text(n, entity_names)

    _collapse_cvts(G, set(record["entities"]))

    # ensure every topic entity is present so the prompt can attach
    for tp in record["entities"]:
        if tp not in G:
            G.add_node(tp, text=resolve_entity_text(tp, entity_names))
    return G


def _collapse_cvts(G, topics):
    """Contract single-parent unnamed mediator entity nodes into rel->rel chains."""
    for n in list(G.nodes()):
        if isinstance(n, tuple) or n in topics:
            continue
        if not (isinstance(n, str) and n.startswith("m.") and G.nodes[n].get("text") == UNNAMED):
            continue
        out_rels = list(G.successors(n))     # relation nodes where n is a head
        in_rels = list(G.predecessors(n))    # relation nodes where n is a tail
        if not out_rels or not in_rels:
            continue                         # pure leaf / root mediator -> leave in place
        parents = set()
        for r in in_rels:
            parents |= set(G.predecessors(r))
        if len(parents) > 1:
            continue                         # multi-parent -> skip (co-membership ambiguous)
        for ri in in_rels:
            for ro in out_rels:
                G.add_edge(ri, ro)
        G.remove_node(n)


# --------------------------------------------------------------------------- #
# Prompt node + answer targets
# --------------------------------------------------------------------------- #
def present_answer_texts(G, record):
    """Gold answer texts whose entity is a node in G (grounded), de-duplicated, order-stable."""
    out = []
    seen = set()
    for a in record["answers"]:
        if a["kb_id"] in G and a.get("text") and a["text"] not in seen:
            seen.add(a["text"])
            out.append(a["text"])
    return out


def full_gold_texts(record):
    out, seen = [], set()
    for a in record["answers"]:
        if a.get("text") and a["text"] not in seen:
            seen.add(a["text"])
            out.append(a["text"])
    return out


def add_prompt_node(G, record, answer_str, gold_answers):
    g = G.copy()
    g.add_node("PROMPT", text=f"{record['question']}{ANSWER_DELIM} {answer_str}")
    g.graph["prompt_node"] = "PROMPT"
    g.graph["gold_answers"] = gold_answers
    g.graph["question"] = record["question"]
    for tp in record["entities"]:
        if tp in g:
            g.add_edge("PROMPT", tp)
    return g


def build_question_graphs(record, entity_names, cfg, versions, rng):
    """Return a list of `versions` nx graphs for one question (empty if not trainable)."""
    base = build_base_levi(record, entity_names, cfg.rel_mode, cfg.max_nodes)
    present = present_answer_texts(base, record)
    gold = full_gold_texts(record)
    if not present:
        return []                             # no groundable answer -> unusable for supervision
    graphs = []
    for _ in range(versions):
        order = present[:]
        rng.shuffle(order)
        answer_str = ANSWER_SEP.join(order[: cfg.n_max])
        graphs.append(add_prompt_node(base, record, answer_str, gold))
    return graphs


# --------------------------------------------------------------------------- #
# Label masking (supervise the answer span after "Answer:")
# --------------------------------------------------------------------------- #
class AnswerLabelMasker:
    """Mask everything up to and including the `question_end` token subsequence to -100."""

    def __init__(self, question_end):
        if not question_end:
            raise ValueError("question_end must be a non-empty token-id list.")
        self.question_end = list(question_end)

    def __call__(self, example):
        ids = example["input_ids"][example["prompt_node"]]
        labels = list(ids)
        qe, end_idx = self.question_end, None
        for i in range(len(ids) - len(qe) + 1):
            if ids[i : i + len(qe)] == qe:
                if end_idx is not None:
                    raise ValueError(f"'Answer:' delimiter is ambiguous in prompt: {ids}")
                end_idx = i + len(qe) - 1
        if end_idx is None:
            raise ValueError(f"Could not find 'Answer:' delimiter in prompt tokens: {ids}")
        for i in range(end_idx + 1):
            labels[i] = -100
        return labels


# --------------------------------------------------------------------------- #
# Driver — data_prep mode of the experiment (consumes a RunConfig)
# --------------------------------------------------------------------------- #
def process_split(split, records, entity_names, tokenizer, question_end, cfg, out_dir):
    """Build + save one split's `.gtds` from raw SR records. Returns (kept, graphs)."""
    versions = cfg.versions if split == "train" else 1        # augmentation only for training
    # data_seed (not the training seed) drives the answer-order augmentation RNG;
    # per-split offset so train/dev/test don't share a shuffle stream.
    rng = random.Random(cfg.data_seed + hash(split) % 10_000)

    graphs, kept, skipped = [], 0, 0
    for rec in tqdm(records, desc=f"Building {split} graphs"):
        if not rec.get("answers"):
            skipped += 1
            continue
        gs = build_question_graphs(rec, entity_names, cfg, versions, rng)
        if not gs:
            skipped += 1
            continue
        graphs.extend(gs)
        kept += 1
    print(f"[{split}] kept {kept} questions ({len(graphs)} graphs, {versions}x), skipped {skipped}")

    ds = TextGraphDataset(graphs, dataset_label=f"kgqa/{split}",
                          per_graph_versions=versions, rcm_ordering=cfg.rcm)
    ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=True)
    ds.compute_labels(AnswerLabelMasker(question_end))
    ds.compute_shortest_path_distances(cutoff=cfg.max_spd, use_gpu=cfg.use_gpu)
    ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m, use_gpu=cfg.use_gpu)
    ds.cast_float_features_to_fp32()
    ds.save(os.path.join(out_dir, split))
    return kept, len(graphs)


def _build_split_if_missing(split, cfg, out_dir, entity_names, tokenizer, question_end):
    """Build one split unless its `.gtds` already exists (idempotent).

    A flock around the build lets concurrent jobs (e.g. many ``per_config`` sbatch
    tasks that share one data config) build each split exactly once: the first
    builds it, the rest wait then find it present. Mirrors the flock in the
    expressiveness ``load_or_create_dataset``.
    """
    split_dir = os.path.join(out_dir, split)
    if os.path.isdir(split_dir):
        print(f"[data_prep] {split}: already present at {split_dir} — skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(split_dir + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isdir(split_dir):
            print(f"[data_prep] {split}: built by a concurrent job — skipping.")
            return
        records = [json.loads(l) for l in open(os.path.join(SR_DIR, SPLITS[split]))]
        process_split(split, records, entity_names, tokenizer, question_end, cfg, out_dir)


def run_data_prep_mode(cfg, splits=("train", "dev", "test")):
    """Ensure this config's `.gtds` splits exist under ``OUTPUT_ROOT/data_config_key``.

    Routed to from ``__main__`` when ``--mode data_prep``. For a multi-config
    sweep, run it once in ``data_prep`` mode (each config builds its own splits;
    the per-split flock makes parallel jobs build each artifact exactly once)
    before running again in ``train`` mode.
    """
    out_dir = os.path.join(OUTPUT_ROOT, cfg.data_config_key())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"data_config_key": cfg.data_config_key(),
                   "rel_mode": cfg.rel_mode, "max_nodes": cfg.max_nodes, "n_max": cfg.n_max,
                   "versions": cfg.versions, "max_length": cfg.max_length, "rcm": cfg.rcm,
                   "data_seed": cfg.data_seed, "model_name": cfg.model_name,
                   "max_spd": cfg.max_spd, "magnetic_q": cfg.magnetic_q,
                   "magnetic_m": cfg.magnetic_m}, f, indent=2)

    print(f"[data_prep] out_dir={out_dir}")
    print(f"Loading entity names from {ENTITY_NAMES_PATH} ...")
    entity_names = json.load(open(ENTITY_NAMES_PATH))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    question_end = tokenizer("Answer:", add_special_tokens=False)["input_ids"]

    for split in splits:
        _build_split_if_missing(split, cfg, out_dir, entity_names, tokenizer, question_end)

    if cfg.analyse_dataset:
        from .analyse_dataset import run_analysis
        run_analysis(cfg, SR_DIR, {s: SPLITS[s] for s in splits}, out_dir)

    print(f"\n[data_prep] done. Cached dataset at {out_dir}")
