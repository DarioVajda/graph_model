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

Answer texts prefer the record's `text` and fall back to literal kb_ids (dates,
numbers — see ``answer_text``). Train keeps only supervisable questions; dev/test
also keep unanswerable-but-scoreable ones as empty-target rows so eval
denominators match the benchmark (GNN-RAG scores retrieval failures as 0 too).

Node naming reads the mid->name dict ONLY (``cfg.naming_version`` picks it:
1 = entities_names.v1.json, 2 = entities_names.json). Harvesting a question's own answer
`text` into node text would leak the answer at eval (gold nodes would be the
only newly-named ones), so answer text feeds ONLY the target / eval matching.
Naming v2 (data-format v3) extends that FILE with Freebase-native aliases
(in-subgraph ``type.object.name`` triples + the FB5M name dump — see
``build_entities_names_v2.py``); the no-per-question-harvesting rule stands.

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
from .config import ASSISTANT_HEADER, PINNED_SYSTEM_PROMPT
from .sr_records import SPLITS, load_sr_records

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(EXPERIMENT_DIR, "processed_datasets")

UNNAMED = "unnamed entity"
END_OF_HOP = "END OF HOP"
# Prompt = "{question}\nAnswer: a1<sep>a2<sep>...". The truncation anchor below is
# a fixed string; the answer separator is version-dependent (cfg.answer_sep:
# dfv2 ", ", dfv3 "\n" — collision-free, single-token, GNN-RAG's own format).
ANSWER_DELIM = "\nAnswer:"


def entity_names_path(cfg):
    """Path to the mid->name dict ``cfg.naming_version`` selects."""
    return os.path.join(EXPERIMENT_DIR, cfg.entity_names_file)


# --------------------------------------------------------------------------- #
# Node text: naming + relation verbalization
# --------------------------------------------------------------------------- #
def _decode_literal(s: str) -> str:
    """Freebase literal values are URL-ish encoded, e.g. 'Justin$002BBieber' -> 'Justin+Bieber'."""
    return re.sub(r"\$([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)


def resolve_entity_text(node, entity_names: dict) -> str:
    """Entity node text: entities_names.json only; unnamed MIDs -> 'unnamed entity'."""
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
            # sorted(): set iteration order is hash-randomized per process; without
            # it the triple order (and thus what survives the cap) is nondeterministic.
            for u in sorted(frontier, key=str):
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
ENTITY_TYPE_PREFIX = "entity "
RELATION_TYPE_PREFIX = "relation "
# The QUESTION node's uniform type-word sink (levi_typed). The PROMPT/answer node
# already starts with "Answer:" (its label-mask + generation anchor), so its first
# token is a uniform type word by construction and is deliberately left untyped.
QUESTION_TYPE_PREFIX = "question "


def _apply_type_prefix(G):
    """Prefix every node's ``text`` with a type word so its FIRST token is a
    uniform, type-specific semantic sink for the ``magnetic_content`` bias (which
    reads each node's first-token hidden state). Relation nodes (``is_rel``) get
    ``"relation "``, entity / value nodes get ``"entity "``. Applied AFTER CVT
    collapse, so contracted rel->rel chains are unaffected. Only the base Levi
    entity/relation nodes are typed — the PROMPT / QUESTION nodes are added later
    and carry the question text, not a type."""
    for n in G.nodes():
        prefix = RELATION_TYPE_PREFIX if G.nodes[n].get("is_rel", False) else ENTITY_TYPE_PREFIX
        G.nodes[n]["text"] = prefix + G.nodes[n]["text"]


def build_base_levi(record, entity_names, rel_mode, max_nodes, cvt_collapse=True, typed=False):
    """Directed per-triple Levi graph with node text (CVTs collapsed by default).
    No prompt node yet. ``cvt_collapse=False`` keeps mediators as nodes (the
    2x2 collapse ablation's uncollapsed graph arm); the ``select_triples``
    budget already counts PRE-collapse Levi nodes, so uncollapsed graphs still
    respect ``max_nodes``.

    ``typed=True`` (graph_construction="levi_typed") prefixes each node's text
    with "entity "/"relation " — identical topology, only the text (hence the
    first token) changes."""
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

    if cvt_collapse:
        _collapse_cvts(G, set(record["entities"]))

    # ensure every topic entity is present so the prompt can attach
    for tp in record["entities"]:
        if tp not in G:
            G.add_node(tp, text=resolve_entity_text(tp, entity_names))

    if typed:
        _apply_type_prefix(G)
    return G


def build_base_triplet(record, entity_names, rel_mode, max_nodes):
    """Triplet graph (graph_construction="triplet"): one node per selected triple.

    Content-fair twin of the flat serialization: the SAME ``select_triples``
    call (same budget, same triple set, raw/uncollapsed) and each node's text
    is exactly the flat arm's ``head | relation | tail`` line (``triple_lines``
    in flat_data.py). Symmetric edge pairs connect triples sharing an entity
    (head or tail; relations don't count). No prompt node yet; the triples
    containing a topic entity are stashed in ``G.graph["topic_nodes"]`` for
    ``add_prompt_node`` (topic entities aren't nodes here).

    Node count = #triples + 2 (PROMPT + QUESTION) never exceeds ``max_nodes``:
    the Levi estimate the budget caps counts every triple PLUS its entities.
    """
    selected = select_triples(record, max_nodes)
    G = nx.DiGraph()
    ent2nodes = defaultdict(list)
    for i, (h, rel, t) in enumerate(selected):
        nid = ("T", i)
        G.add_node(nid, text=(f"{resolve_entity_text(h, entity_names)} | "
                              f"{verbalize_relation(rel, rel_mode)} | "
                              f"{resolve_entity_text(t, entity_names)}"))
        for ent in {h, t}:
            ent2nodes[ent].append(nid)
    for nodes in ent2nodes.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                G.add_edge(nodes[a], nodes[b])
                G.add_edge(nodes[b], nodes[a])
    topic_nodes = sorted({n for tp in record["entities"] for n in ent2nodes.get(tp, ())})
    G.graph["topic_nodes"] = topic_nodes
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
def answer_text(a):
    """Scoreable text for one answer record: its `text`, else a literal kb_id.

    Some gold answers (dates, years, numbers, currency codes — 53/16602 on test)
    have an empty `text` but a *literal* kb_id that IS the answer string
    ("1945-09-02", "AUD") — the same string the graph shows for that node
    (`resolve_entity_text` falls through to `_decode_literal`). GNN-RAG scores
    these as plain strings too (RoG `answer` lists), so dropping them would be
    unfair to us. `m.`/`g.` mids without text stay unscoreable -> None.
    """
    if a.get("text"):
        return a["text"]
    kid = a.get("kb_id")
    if isinstance(kid, str) and kid and not kid.startswith(("m.", "g.")):
        return _decode_literal(kid)
    return None


def present_answer_texts(G, record):
    """Gold answer texts whose entity is a node in G (grounded), de-duplicated, order-stable."""
    out = []
    seen = set()
    for a in record["answers"]:
        text = answer_text(a)
        if a["kb_id"] in G and text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def full_gold_texts(record):
    """The evaluator's gold list — mirrors RoG/GNN-RAG's `answer` lists exactly.

    Scoreable texts via ``answer_text``; golds with no name at all (unnamed
    `m.`/`g.` mids) are kept as their raw kb_id, exactly like RoG's lists
    (verified identical by id on 1628/1628 test questions). These placeholders
    never match a generation, deflating recall the same way it deflates
    GNN-RAG's — dropping them instead would inflate our F1 on those questions.
    """
    out, seen = [], set()
    for a in record["answers"]:
        text = answer_text(a) or a.get("kb_id")
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def chat_prompt_text(question):
    """The chat-templated prompt prefix (D3, design locked 2026-07-08).

    One node holds the full templated string; the graph stays a plain-text
    modality in the other nodes. The leading BOS is kept verbatim mid-sequence
    (after the graph prefix) so from ``<|begin_of_text|>`` onward the stream
    matches the instruct SFT distribution exactly; the pinned system turn keeps
    cache content independent of build date. Targets follow the assistant
    header directly; ``tokenize(add_eos=True)`` appends the closing
    ``<|eot_id|>`` (the instruct tokenizer's EOS).
    """
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{PINNED_SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>" + ASSISTANT_HEADER)


def add_prompt_node(G, record, answer_str, gold_answers, prompt_style="plain",
                    question_node="off", typed=False):
    """Attach the PROMPT node (and, unless ``question_node="off"``, a QUESTION node).

    ``typed=True`` (levi_typed) prefixes the QUESTION node's text with "question "
    so its first token is a uniform type-word sink for magnetic_content, matching
    the "entity "/"relation " prefixes on the base nodes. The PROMPT/answer node is
    left untouched — it already starts with "Answer:" (a uniform first token) and
    that string is the load-bearing label-mask / generation anchor.

    ``question_node`` ("off" | "all" | "topics" | "isolated") moves the question
    text out of the PROMPT node into its own QUESTION *prefix* node: the model's
    bidirectional-prefix mask then lets every graph token attend to the question
    (question-conditioned graph encoding; the flat arm is question-first already).
    The mode picks QUESTION's directed OUT-edges — they feed the SPD/magnetic
    bias features, while token visibility comes from the mask regardless:
      * "all"      — an edge to every base-graph node (pre-PROMPT);
      * "topics"   — edges to the topic entities (mirrors PROMPT's own edges);
      * "isolated" — no edges (disconnected components already occur in
                     production — e.g. detached topic entities).
    PROMPT keeps its historical topic edges in every mode, so the only delta
    between modes is QUESTION's out-edge set. "off" = the historical
    single-prompt-node format, byte-identical to pre-feature builds.
    """
    g = G.copy()
    # Topic-attachment targets: the topic entity nodes themselves (Levi) or the
    # triplet nodes CONTAINING a topic entity (triplet construction — stashed by
    # build_base_triplet; popped so the stale original-id list never outlives
    # this function into the saved/relabeled graph).
    topic_targets = g.graph.pop("topic_nodes", None)
    if topic_targets is None:
        topic_targets = [tp for tp in record["entities"] if tp in g]
    if prompt_style == "chat":
        if question_node != "off":
            raise ValueError("question_node + chat prompt style is unsupported (see validate()).")
        # Empty target (unanswerable eval row): the prompt ends exactly at the
        # assistant header ("...\n\n") and labels are the terminator only.
        text = chat_prompt_text(record["question"]) + answer_str
    else:
        # No trailing space when the target is empty (unanswerable eval row): the
        # prompt then ends exactly at the "Answer:" delimiter and labels are EOS only.
        suffix = f" {answer_str}" if answer_str else ""
        if question_node != "off":
            # Question lives in the QUESTION node; the prompt node is target-only.
            text = f"Answer:{suffix}"
        else:
            text = f"{record['question']}{ANSWER_DELIM}{suffix}"
    if question_node != "off":
        base_nodes = list(g.nodes())            # pre-PROMPT, pre-QUESTION
        q_text = QUESTION_TYPE_PREFIX + record["question"] if typed else record["question"]
        g.add_node("QUESTION", text=q_text)
        g.graph["question_node"] = "QUESTION"
        if question_node == "all":
            for n in base_nodes:
                g.add_edge("QUESTION", n)
        elif question_node == "topics":
            for n in topic_targets:
                g.add_edge("QUESTION", n)
        # "isolated": no edges.
    g.add_node("PROMPT", text=text)
    g.graph["prompt_node"] = "PROMPT"
    g.graph["gold_answers"] = gold_answers
    g.graph["question"] = record["question"]
    g.graph["unanswerable"] = not answer_str
    for n in topic_targets:
        g.add_edge("PROMPT", n)
    return g


def build_question_graphs(record, entity_names, cfg, versions, rng, keep_unanswerable=False):
    """Return a list of `versions` nx graphs for one question.

    Empty when the question is unusable: no graph-present gold (nothing to
    supervise) — unless ``keep_unanswerable`` (dev/test), where every answered
    question is kept, with an EMPTY answer target when unanswerable, so eval
    denominators equal RoG/GNN-RAG's (all answered questions; such rows score
    ~0, like their retrieval failures).
    """
    typed = cfg.graph_construction == "levi_typed"   # prefix node text with type words
    if cfg.graph_construction == "triplet":
        # Content fairness with the flat arm trumps the question-node slot: use
        # the UNSHRUNK budget (the same select_triples call flat makes), so the
        # triple set is identical to the text-only LLM's. Node count stays
        # under max_nodes regardless (see build_base_triplet). Targets come
        # from the collapsed Levi base — the same graph build_flat_rows uses —
        # so they are byte-identical to both other arms'.
        base = build_base_triplet(record, entity_names, cfg.rel_mode, cfg.max_nodes)
        target_base = build_base_levi(record, entity_names, cfg.rel_mode, cfg.max_nodes)
    else:
        # A QUESTION node consumes one slot of the Levi budget (select_triples
        # reserves the PROMPT slot itself); shrink the cap so built graphs never
        # exceed max_nodes (which would spill into the next flex node bucket).
        budget = cfg.max_nodes - (0 if cfg.question_node == "off" else 1)
        base = build_base_levi(record, entity_names, cfg.rel_mode, budget,
                               cvt_collapse=cfg.resolved_cvt_collapse("graph"),
                               typed=typed)
        target_base = base
    present = present_answer_texts(target_base, record)
    gold = full_gold_texts(record)
    style = cfg.resolved_prompt_style
    if not present:
        if keep_unanswerable and gold:
            return [add_prompt_node(base, record, "", gold, prompt_style=style,
                                    question_node=cfg.question_node, typed=typed)]
        return []
    graphs = []
    for _ in range(versions):
        order = present[:]
        rng.shuffle(order)
        answer_str = cfg.answer_sep.join(order[: cfg.n_max])
        graphs.append(add_prompt_node(base, record, answer_str, gold, prompt_style=style,
                                      question_node=cfg.question_node, typed=typed))
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
    """Build + save one split's `.gtds` from raw SR records. Returns (kept, graphs).

    Train keeps only supervisable questions (>=1 graph-present gold). Dev/test
    additionally keep unanswerable-but-scoreable questions as empty-target rows,
    so eval metrics use benchmark-comparable denominators (those rows score ~0).
    """
    versions = cfg.versions if split == "train" else 1        # augmentation only for training
    keep_unanswerable = split != "train"
    # data_seed (not the training seed) drives the answer-order augmentation RNG;
    # per-split offset so train/dev/test don't share a shuffle stream. Seed with a
    # string (SHA-512-based, process-independent) — hash(str) varies per process.
    rng = random.Random(f"{cfg.data_seed}:{split}")

    graphs, kept, skipped = [], 0, 0
    for rec in tqdm(records, desc=f"Building {split} graphs"):
        if not rec.get("answers"):
            skipped += 1
            continue
        gs = build_question_graphs(rec, entity_names, cfg, versions, rng,
                                   keep_unanswerable=keep_unanswerable)
        if not gs:
            skipped += 1
            continue
        graphs.extend(gs)
        kept += 1
    n_unans = sum(1 for g in graphs if g.graph.get("unanswerable"))
    print(f"[{split}] kept {kept} questions ({len(graphs)} graphs, {versions}x; "
          f"{n_unans} unanswerable empty-target rows), skipped {skipped} unscorable")

    ds = TextGraphDataset(graphs, dataset_label=f"kgqa/{split}",
                          per_graph_versions=versions, rcm_ordering=cfg.rcm)
    ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=True)
    ds.compute_labels(AnswerLabelMasker(question_end))
    ds.compute_shortest_path_distances(cutoff=cfg.max_spd, use_gpu=cfg.use_gpu)
    ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m, use_gpu=cfg.use_gpu)
    ds.cast_float_features_to_fp32()
    # RRWP goes LAST, after the fp32 cast. Its (n, n, max_rw_steps) float32 column
    # is an order of magnitude larger than every other feature combined (~41 GB vs
    # 3.4 GB for WebQSP at 16 steps), and BOTH `.map()` and `.cast()` rewrite the
    # whole Arrow table — computing it earlier would make every later pass haul it
    # disk -> RAM -> disk for nothing. compute_rrwp already emits float32, so it
    # needs no cast and loses nothing by running after one.
    if cfg.rrwp:
        ds.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps, use_gpu=cfg.use_gpu)
    ds.save(os.path.join(out_dir, split))
    return kept, len(graphs)


def _build_split_if_missing(split, dataset, cfg, out_dir, entity_names, tokenizer, question_end):
    """Build one split unless its `.gtds` already exists (idempotent).

    A flock around the build lets concurrent jobs (e.g. many ``per_config`` sbatch
    tasks that share one data config) build each split exactly once: the first
    builds it, the rest wait then find it present. Mirrors the flock in the
    expressiveness ``load_or_create_dataset``.
    """
    split_dir = os.path.join(out_dir, split)
    built_dir = split_dir + ".gtds"          # TextGraphDataset.save appends the suffix
    if os.path.isdir(built_dir):
        print(f"[data_prep] {split}: already present at {built_dir} — skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(split_dir + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isdir(built_dir):
            print(f"[data_prep] {split}: built by a concurrent job — skipping.")
            return
        records = load_sr_records(dataset, split)
        process_split(split, records, entity_names, tokenizer, question_end, cfg, out_dir)


def role_splits(cfg, dataset):
    """The splits a run actually needs for ``dataset``: train iff it trains on
    it, dev+test iff it evaluates on it (a CWQ train build is expensive — don't
    build it for an eval-only role)."""
    splits = []
    if dataset in cfg.train_datasets:
        splits.append("train")
    if dataset in cfg.eval_datasets:
        splits += ["dev", "test"]
    return tuple(splits)


def run_data_prep_mode(cfg, splits=None):
    """Ensure every (dataset × resolved data config) cache this run references
    exists under ``OUTPUT_ROOT/<data_config_key(dataset)>``.

    Routed to from ``__main__`` when ``--mode data_prep``. For a multi-config
    sweep, run it once in ``data_prep`` mode (each config builds its own splits;
    the per-split flock makes parallel jobs build each artifact exactly once)
    before running again in ``train`` mode. ``splits=None`` builds what the
    dataset's role requires (see ``role_splits``).
    """
    names_path = entity_names_path(cfg)
    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f"naming_version={cfg.naming_version} needs {names_path}, which is missing.\n"
            f"See the README: download entities_names.json from GNN-RAG's release, then run "
            f"`python3 -m src.experiments.kgqa.build_entities_names_v2` (it writes naming v2 "
            f"and backs the v1 seed up to entities_names.v1.json)."
        )
    print(f"Loading entity names (v{cfg.naming_version}) from {names_path} ...")
    entity_names = json.load(open(names_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Style-dependent anchor: "Answer:" (plain) | assistant header (chat).
    question_end = tokenizer(cfg.question_end_str, add_special_tokens=False)["input_ids"]

    for dataset in cfg.datasets:
        view = cfg.for_dataset(dataset)      # per-dataset knobs resolved to scalars
        out_dir = os.path.join(OUTPUT_ROOT, view.data_config_key(dataset))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"data_config_key": view.data_config_key(dataset),
                       "dataset": dataset,
                       "data_format_version": view.data_format_version,
                       "naming_version": view.naming_version,
                       "rel_mode": view.rel_mode, "max_nodes": view.max_nodes,
                       "n_max": view.n_max, "question_node": view.question_node,
                       "graph_construction": view.graph_construction,
                       "versions": view.versions, "max_length": view.max_length,
                       "rcm": view.rcm,
                       "data_seed": view.data_seed, "model_name": view.model_name,
                       "max_spd": view.max_spd, "magnetic_q": view.magnetic_q,
                       "magnetic_m": view.magnetic_m,
                       "rrwp": view.rrwp, "max_rw_steps": view.max_rw_steps}, f, indent=2)

        ds_splits = splits if splits is not None else role_splits(cfg, dataset)
        print(f"[data_prep] {dataset}: out_dir={out_dir} splits={ds_splits}")
        for split in ds_splits:
            _build_split_if_missing(split, dataset, view, out_dir,
                                    entity_names, tokenizer, question_end)

        if cfg.analyse_dataset:
            from .analyse_dataset import run_analysis
            run_analysis(view, out_dir, ds_splits, dataset=dataset)

        print(f"[data_prep] {dataset}: done. Cached dataset at {out_dir}\n")
