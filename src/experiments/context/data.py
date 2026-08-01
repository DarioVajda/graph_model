"""
Star-graph synthesis for the context-exhaustion stress test (Needle in a Graph).

Pure CPU, no torch, no model: this module turns a ``RunConfig`` into NetworkX
graphs whose node texts are *exactly* T tokens long. ``process_dataset.py`` wraps
them into a ``TextGraphDataset``.

Topology (README §A.1), for a cell with N total nodes:

    node 0        QUESTION  "What is the access code for {gold_id}?"
                            prefix, undirected edge to every content node
    nodes 1..N-2  CONTENT   T tokens of wikitext with one KV sentence spliced in
                            ("The access code for {id} is {code}."), exactly one
                            of which is the GOLD node (its id is the one asked for)
    node N-1      PROMPT    "Answer: {gold_code}" — isolated, causal, supervised

Two invariants drive the whole file and are asserted at build time:

  * **Exact token counts.** T is an independent variable, so a content node must
    tokenize to exactly T tokens — not "about T". ``fit_node_text`` gets there
    through the public ``tokenizer`` path (never by writing ``input_ids``
    directly, which would mean reaching into ``TextGraphDataset``'s internals).
  * **Paired cells.** A graph's *blueprint* — node ids, codes, which slot is
    gold, the nesting order of slots, and each slot's filler offset — is drawn
    once and shared by all 25 cells. Along N the slot subsets are nested; along
    T only the within-node needle offset is redrawn (README §A.3).
"""

import hashlib
import os
import random
import string
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .config import CORPUS_CONFIG, CORPUS_REPO

# Every content node carries exactly one KV sentence in this form.
KV_TEMPLATE = "The access code for {node_id} is {code}."
QUESTION_TEMPLATE = "What is the access code for {node_id}?"
# The PROMPT node is "{ANSWER_PREFIX}{code}"; tokens up to the prefix are masked
# to -100, so the supervised span is exactly the code (+ EOS).
ANSWER_PREFIX = "Answer:"

# ── k-hop pointer chasing (cfg.hops > 0; README §A.1) ────────────────────
# Every content node also carries a pointer, so the answer node cannot be found by
# looking for "the node that mentions a code" — they all do, and they all point
# somewhere. The answer is the code at the node reached after ``hops`` steps from
# the node the QUESTION names, which is recoverable ONLY by traversing.
#
# This is what makes topology load-bearing: the pointers ARE the edges, the graph
# becomes DIRECTED (what magnetic_q encodes), and SPD stops being trivial the way
# it is on a star of diameter 2.
POINTER_TEMPLATE = " Continue at {node_id}."
CHAIN_QUESTION_TEMPLATE = ("Starting at {node_id}, follow {hops} references. "
                           "What is the access code there?")

# ── decoy references (cfg.fan_out > 1) ────────────────────────────────────────
# At fan_out=1 the content subgraph is functional, so SPD(start, answer) == hops
# identifies the answer uniquely and the graph arm never has to traverse. Decoy
# references break that: every content node emits fan_out out-edges, all of which
# enter the DiGraph identically, so ~fan_out**hops nodes sit at distance hops.
#
# The decoy is named as a decoy IN THE TEXT. That is deliberate: the point is not
# to make the task ambiguous but to move the disambiguating signal out of the
# topology and into the text, where BOTH arms can read it. The graph prunes the
# candidate set; only the text says which reference continues the chain.
DECOY_TEMPLATE = " Decoy reference: {node_id}."
CHAIN_QUESTION_TEMPLATE_FANOUT = (
    "Starting at {node_id}, follow {hops} 'Continue at' references, ignoring any "
    "decoy references. What is the access code there?")

# Corpus lines containing this are dropped: a corpus-native "access code" would
# be an uncontrolled second needle.
BANNED_SUBSTRING = "access code"
# Keep this many filler tokens after the needle so ``fit_node_text`` always has
# something to trim/extend without touching the KV sentence.
SUFFIX_SLACK = 8


def _rng(*parts):
    """A ``random.Random`` seeded by a stable hash of ``parts``.

    ``hash()`` is salted per process, so it cannot be used for anything that must
    reproduce across runs — every seed in this module goes through here instead.
    """
    key = "|".join(str(p) for p in parts).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


# ── Filler corpus ──────────────────────────────────────────────────────────────

def load_corpus(tokenizer, cache_dir, n_tokens, corpus_repo=CORPUS_REPO,
                corpus_config=CORPUS_CONFIG, verbose=True):
    """Return a 1-D int32 array of ``n_tokens`` filler token ids, cached on disk.

    The cache key includes the tokenizer, because a token stream is only filler
    *for the tokenizer that produced it* — reusing another model's stream would
    silently break the exact-T invariant.
    """
    tag = f"{corpus_config}_{tokenizer.name_or_path.split('/')[-1]}_{n_tokens}"
    path = os.path.join(cache_dir, f"filler_{tag}.npy")
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")

    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    if verbose:
        print(f"[corpus] building {n_tokens:,} filler tokens from {corpus_repo}/{corpus_config}")
    raw = load_dataset(corpus_repo, corpus_config, split="train")

    out, total = [], 0
    buf = []
    for line in raw["text"]:
        line = line.strip()
        # Skip headings ("= = Title = ="), blanks and any corpus-native needle.
        if not line or line.startswith("=") or BANNED_SUBSTRING in line.lower():
            continue
        buf.append(line)
        if len(buf) >= 1000:
            ids = tokenizer(" ".join(buf), add_special_tokens=False)["input_ids"]
            out.append(np.asarray(ids, dtype=np.int32))
            total += len(ids)
            buf = []
            if total >= n_tokens:
                break
    if buf and total < n_tokens:
        ids = tokenizer(" ".join(buf), add_special_tokens=False)["input_ids"]
        out.append(np.asarray(ids, dtype=np.int32))
        total += len(ids)

    stream = np.concatenate(out)[:n_tokens]
    if len(stream) < n_tokens:
        raise RuntimeError(
            f"corpus yielded only {len(stream):,} of the requested {n_tokens:,} tokens; "
            "lower corpus_tokens or use a larger split.")
    np.save(path, stream)
    if verbose:
        print(f"[corpus] cached {len(stream):,} tokens -> {path}")
    return np.load(path, mmap_mode="r")


# ── Codes and node ids ─────────────────────────────────────────────────────────

CODE_LETTERS = string.ascii_uppercase
CODE_DIGITS = string.digits
# Codes are drawn from a FIXED-length template with an interior digit ("AB1CD").
# Both properties are load-bearing, and both were learned the hard way:
#
#   * **Fixed character length.** Filtering only on TOKEN length let 4-, 5- and
#     6-character codes into one pool, and a short code can sit inside a long one
#     — gold "IREO" inside distractor "OIREO" made graph 113 of the first full
#     build ambiguous. Equal-length distinct strings cannot contain one another.
#   * **A digit flanked by letters.** An unconstrained alphabet produces
#     all-digit codes (1.0% of the old pool: "8192", "0595"), which occur freely
#     in wikitext as years and quantities — a needle that also appears in another
#     node's filler. "AB1CD" does not occur in English prose.
CODE_TEMPLATE = "LLDLL"


def build_code_pool(tokenizer, code_len, size, seed=0, max_tries=200_000):
    """Codes that tokenize to *exactly* ``code_len`` tokens when preceded by a space.

    Fixed token length is what makes exact match comparable across graphs: with a
    variable-length answer, "EM" would silently mix a 2-token and a 5-token
    prediction problem. The leading space matters — BPE merges it into the first
    token, and that is how the code appears in both the KV sentence and the
    PROMPT node. See :data:`CODE_TEMPLATE` for the two uniqueness properties the
    character template guarantees.
    """
    rng = random.Random(seed)
    pool, seen = [], set()
    for _ in range(max_tries):
        if len(pool) >= size:
            break
        cand = "".join(rng.choice(CODE_LETTERS if c == "L" else CODE_DIGITS)
                       for c in CODE_TEMPLATE)
        if cand in seen:
            continue
        seen.add(cand)
        if len(tokenizer(" " + cand, add_special_tokens=False)["input_ids"]) == code_len:
            pool.append(cand)
    if len(pool) < size:
        raise RuntimeError(
            f"only {len(pool)} of {max_tries} candidates tokenize to exactly {code_len} "
            f"tokens (needed {size}). Try a different code_len or CODE_TEMPLATE.")
    return pool


def build_id_pool(size):
    """Node identifiers, drawn from a pool much larger than any graph.

    Sampling ids from a large pool (rather than using ``range(N)``) stops the gold
    id from being predictable from N and stops the model learning an id->position
    prior.
    """
    return [f"NODE-{i:05d}" for i in range(size)]


# ── Exact-length node text ─────────────────────────────────────────────────────

# Common words that BPE encodes as exactly one token when appended to running
# text. ``_pad_words`` verifies this against the actual tokenizer rather than
# trusting it — they are the "+1 token" primitive ``fit_node_text`` closes the
# last few tokens of a node with.
_PAD_WORD_CANDIDATES = (" the", " of", " and", " in", " to", " that", " with", " for")
_PAD_WORD_CACHE = {}


def _pad_words(tokenizer):
    key = tokenizer.name_or_path
    if key not in _PAD_WORD_CACHE:
        base = tokenizer("word", add_special_tokens=False)["input_ids"]
        words = [w for w in _PAD_WORD_CANDIDATES
                 if len(tokenizer("word" + w, add_special_tokens=False)["input_ids"]) == len(base) + 1]
        if not words:
            raise RuntimeError(
                f"no single-token pad word found for {key}; exact node lengths are unreachable.")
        _PAD_WORD_CACHE[key] = words
    return _PAD_WORD_CACHE[key]


def fit_node_text(tokenizer, corpus, start, kv_text, offset, n_tokens, max_iters=16):
    """Build one content node's text: filler with ``kv_text`` spliced in at ``offset``.

    Returns ``(text, ids)`` with ``len(ids) == n_tokens`` **exactly** and the KV
    sentence present verbatim.

    Why this is not a slice: the text is assembled from *decoded* corpus token
    slices, and BPE does not guarantee that re-encoding a decoded slice reproduces
    the token count — the boundaries merge differently. Adjusting the filler
    length by the shortfall can also oscillate (removing one corpus token can
    change two tokens' worth of merges), so the algorithm is deliberately
    one-directional:

      1. shrink the filler until the node is at most ``n_tokens`` (monotone), then
      2. close the remaining gap with verified single-token pad words.

    Shrinking only ever touches filler AFTER the needle (and only then the filler
    before it), so the KV sentence can never be trimmed away.
    """
    kv_ids = tokenizer(kv_text, add_special_tokens=False)["input_ids"]
    budget = n_tokens - len(kv_ids)
    if budget < 0:
        raise ValueError(
            f"KV sentence is {len(kv_ids)} tokens but the node budget is {n_tokens}.")
    offset = max(0, min(offset, budget))
    prefix_len, suffix_len = offset, budget - offset

    def assemble():
        pre = tokenizer.decode(corpus[start:start + prefix_len].tolist()) if prefix_len else ""
        suf_start = start + prefix_len
        suf = tokenizer.decode(corpus[suf_start:suf_start + suffix_len].tolist()) if suffix_len else ""
        text = " ".join(p for p in (pre, kv_text, suf) if p).strip()
        return text, tokenizer(text, add_special_tokens=False)["input_ids"]

    # ── 1. shrink to <= n_tokens ───────────────────────────────────────────────
    text, ids = assemble()
    for _ in range(max_iters):
        excess = len(ids) - n_tokens
        if excess <= 0:
            break
        if suffix_len >= excess:
            suffix_len -= excess
        else:
            prefix_len = max(0, prefix_len - (excess - suffix_len))
            suffix_len = 0
        text, ids = assemble()
    if len(ids) > n_tokens:
        raise RuntimeError(
            f"could not shrink node text to {n_tokens} tokens (stuck at {len(ids)}).")

    # ── 2. close the gap with single-token words ───────────────────────────────
    words = _pad_words(tokenizer)
    for i in range(max_iters):
        gap = n_tokens - len(ids)
        if gap == 0:
            break
        text = text + "".join(words[(i + j) % len(words)] for j in range(gap))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > n_tokens:      # a pad word merged unexpectedly — shrink and retry
            text = tokenizer.decode(ids[:n_tokens])
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != n_tokens:
        raise RuntimeError(
            f"could not fit node text to exactly {n_tokens} tokens (got {len(ids)}).")
    # String containment, not token containment: the KV sentence's tokens in
    # context differ from its tokens in isolation (BPE merges the preceding space
    # into the first token), so a token-subsequence check would fail on a perfectly
    # good node. What has to hold is that the sentence is *there*.
    if kv_text not in text:
        raise RuntimeError("KV sentence did not survive assembly of the node text.")
    return text, ids


# ── Blueprints ─────────────────────────────────────────────────────────────────

@dataclass
class Blueprint:
    """Everything about one graph that is INVARIANT across the 25 grid cells.

    ``slot_order[0]`` is the gold slot; cell N uses ``slot_order[:N-2]``, so the
    subsets are nested along N and always contain the gold node.
    """
    graph_id: int
    node_ids: list          # per slot
    codes: list             # per slot
    slot_order: list        # nesting order; [0] is gold
    filler_at: list         # corpus offset per slot

    @property
    def gold_slot(self):
        return self.slot_order[0]


def make_blueprint(graph_id, cfg, code_pool, id_pool, corpus_len, split="train"):
    """Draw one graph's blueprint (see :class:`Blueprint`).

    Filler windows are drawn independently, so the same wikitext passage recurs
    across graphs (4000 graphs x 126 slots x up to 512 tokens is an order of
    magnitude more filler than the 20M-token stream holds). That is deliberate and
    harmless: the needle is what has to be unique, and it is — the filler is there
    to consume context, not to be learned.
    """
    rng = _rng(cfg.data_seed, cfg.data_format_version, split, graph_id)
    n_slots = cfg.n_content_max()
    max_node_tokens = max(cfg.token_counts)
    return Blueprint(
        graph_id=graph_id,
        node_ids=rng.sample(id_pool, n_slots),
        codes=rng.sample(code_pool, n_slots),
        slot_order=rng.sample(range(n_slots), n_slots),
        filler_at=[rng.randrange(0, corpus_len - max_node_tokens - 64) for _ in range(n_slots)],
    )


def needle_offsets(bp, cfg, t, split="train"):
    """Per-slot within-node needle offsets for tokens-per-node ``t`` (README §A.3).

    Drawn for EVERY slot (not just the ones a given N uses), so a slot's offset
    depends on T but not on N — cells stay paired along the N axis too.
    """
    rng = _rng(cfg.data_seed, cfg.data_format_version, "offset", split, bp.graph_id, t)
    return [rng.randrange(0, max(1, t)) for _ in range(len(bp.node_ids))]


def realize(bp, cfg, n, t, tokenizer, corpus, split="train"):
    """Build the NetworkX graph for blueprint ``bp`` at cell ``(n, t)``.

    Node texts are exact-length; graph attrs carry everything the evaluator and
    the tests need (gold code/id, the distractor codes, the cell).
    """
    slots = bp.slot_order[:n - 2]
    gold = bp.gold_slot
    offsets = needle_offsets(bp, cfg, t, split=split)

    # Re-randomize which content node lands where in the packed order. The model is
    # provably order-invariant over the prefix (Property 1), so this guards against
    # a build bug rather than a model artifact — but it is free.
    order_rng = _rng(cfg.data_seed, cfg.data_format_version, "order", split, bp.graph_id, n, t)
    placed = list(slots)
    order_rng.shuffle(placed)

    g = nx.Graph()
    g.add_node(0, text=QUESTION_TEMPLATE.format(node_id=bp.node_ids[gold]))

    kv_max_offset = None
    for position, slot in enumerate(placed, start=1):
        kv = KV_TEMPLATE.format(node_id=bp.node_ids[slot], code=bp.codes[slot])
        kv_len = len(tokenizer(kv, add_special_tokens=False)["input_ids"])
        # Cap the offset so a slice of filler always follows the needle (see
        # fit_node_text): the needle is uniform over the part of the node where it
        # can actually be placed, which is [0, T - |KV| - SUFFIX_SLACK].
        kv_max_offset = max(0, t - kv_len - SUFFIX_SLACK)
        text, _ = fit_node_text(tokenizer, corpus, bp.filler_at[slot], kv,
                                min(offsets[slot], kv_max_offset), t)
        g.add_node(position, text=text)
        g.add_edge(0, position)

    prompt_node = n - 1
    g.add_node(prompt_node, text=f"{ANSWER_PREFIX} {bp.codes[gold]}")   # isolated: no edges

    g.graph.update(
        prompt_node=prompt_node,
        question_node=0,
        graph_id=bp.graph_id,
        cell_n=n,
        cell_t=t,
        hops=0,
        gold_code=bp.codes[gold],
        gold_id=bp.node_ids[gold],
        gold_position=placed.index(gold) + 1,
        codes=[bp.codes[s] for s in placed],
        max_needle_offset=kv_max_offset,
    )
    return g


def chain_slots(bp, cfg, hops=None):
    """The ``hops + 1`` slots of the gold chain: [start, ..., answer].

    Taken from the FRONT of ``slot_order``, which is the nesting order, so the
    whole chain is present in every cell of the grid and the answer is identical
    across cells — the same pairing guarantee the lookup task gets from
    ``slot_order[0]`` being the gold.

    Because the chain is a PREFIX of ``slot_order``, one blueprint yields NESTED
    chains across k: same start node, progressively deeper answer. That is what
    makes the k-curve paired when k is a mixture axis, and it is free — do not
    replace the prefix with an independent draw per k.
    """
    return bp.slot_order[:(cfg.hops if hops is None else hops) + 1]


def sample_hops(bp_id, cfg, split="train"):
    """Pick this graph's hop count from the training mixture (uniform over ``hop_counts``).

    Mirrors :func:`sample_cell`: the axis is drawn per BLUEPRINT from a stable seeded
    stream, so the mixture is reproducible and independent of build order. Returns the
    scalar ``cfg.hops`` when no mixture is configured, which keeps every single-k build
    byte-identical.
    """
    ks = cfg.hops_list()
    if len(ks) == 1:
        return ks[0]
    rng = _rng(cfg.data_seed, cfg.data_format_version, "hops", split, bp_id)
    return rng.choice(list(ks))


def realize_chain(bp, cfg, n, t, tokenizer, corpus, split="train", hops=None):
    """Build the k-hop pointer-chasing graph for blueprint ``bp`` at cell ``(n, t)``.

    Differs from :func:`realize` in three ways, and in nothing else:

      * every content node's needle is ``KV + POINTER``, so each node both holds a
        code and points at another node;
      * the QUESTION names the START node and the hop count, never the answer;
      * the graph is a **DiGraph** whose edges are the pointers, so the topology
        the model is given is exactly the topology the task requires.

    Decoy pointers are drawn per CELL (not per blueprint) so they can only ever
    target a node that is actually present; the gold chain comes from the
    blueprint and is therefore identical in every cell.
    """
    hops = cfg.hops if hops is None else hops
    slots = bp.slot_order[:n - 2]
    chain = chain_slots(bp, cfg, hops)
    if len(slots) < len(chain):
        raise ValueError(
            f"cell N={n} has {len(slots)} content nodes, too few for a {hops}-hop "
            f"chain (needs {len(chain)}). Raise N or lower hops.")
    answer_slot = chain[-1]
    offsets = needle_offsets(bp, cfg, t, split=split)

    # Pointer target per slot: along the chain for chain[:-1], random otherwise
    # (including for the ANSWER node, so it is not identifiable as the one node
    # without an outgoing pointer).
    ptr_rng = _rng(cfg.data_seed, cfg.data_format_version, "ptr", split, bp.graph_id, n, t)
    successor = {chain[i]: chain[i + 1] for i in range(len(chain) - 1)}
    for slot in slots:
        if slot not in successor:
            choices = [s for s in slots if s != slot]
            successor[slot] = ptr_rng.choice(choices)

    # Decoy targets per slot, drawn from the same cell-local pool as the real
    # pointers and excluding the slot itself and its real successor, so a decoy is
    # never a duplicate edge. Drawn from a SEPARATE rng stream so that adding decoys
    # does not perturb the real pointers a fan_out=1 build would have produced.
    decoy_rng = _rng(cfg.data_seed, cfg.data_format_version, "decoy", split, bp.graph_id, n, t)
    decoys = {}
    for slot in slots:
        if cfg.fan_out > 1:
            pool = [s for s in slots if s != slot and s != successor[slot]]
            decoys[slot] = decoy_rng.sample(pool, cfg.fan_out - 1)
        else:
            decoys[slot] = []

    order_rng = _rng(cfg.data_seed, cfg.data_format_version, "order", split, bp.graph_id, n, t)
    placed = list(slots)
    order_rng.shuffle(placed)
    position_of = {slot: i for i, slot in enumerate(placed, start=1)}

    g = nx.DiGraph()
    question_template = (CHAIN_QUESTION_TEMPLATE_FANOUT if cfg.fan_out > 1
                         else CHAIN_QUESTION_TEMPLATE)
    g.add_node(0, text=question_template.format(
        node_id=bp.node_ids[chain[0]], hops=hops))

    kv_max_offset = None
    for slot in placed:
        needle = (KV_TEMPLATE.format(node_id=bp.node_ids[slot], code=bp.codes[slot])
                  + POINTER_TEMPLATE.format(node_id=bp.node_ids[successor[slot]])
                  + "".join(DECOY_TEMPLATE.format(node_id=bp.node_ids[d])
                            for d in decoys[slot]))
        needle_len = len(tokenizer(needle, add_special_tokens=False)["input_ids"])
        kv_max_offset = max(0, t - needle_len - SUFFIX_SLACK)
        text, _ = fit_node_text(tokenizer, corpus, bp.filler_at[slot], needle,
                                min(offsets[slot], kv_max_offset), t)
        g.add_node(position_of[slot], text=text)

    # Edges ARE the pointers, among the content nodes. Decoy edges are added the
    # SAME way and carry no attribute distinguishing them: the topology the model
    # sees must not encode which reference is real, or the shortcut returns.
    for slot in placed:
        g.add_edge(position_of[slot], position_of[successor[slot]])
        for d in decoys[slot]:
            g.add_edge(position_of[slot], position_of[d])

    # The QUESTION attaches to EVERY content node, not just the start.
    #
    # This matters more than it looks. With a single QUESTION -> start edge the
    # question has out-degree 1, so SPD(QUESTION, chain[i]) == i + 1 exactly and
    # the answer node is the unique node at distance hops+1 — the graph arm could
    # read the answer straight off the SPD bias without traversing anything, which
    # would measure "we labelled the answer" rather than "structure helps".
    # Fanning out makes SPD(QUESTION, ·) == 1 for every content node, so the only
    # place chain information lives is the content-to-content distances: the model
    # must first bind the start id to a node by name, then walk k steps. That is
    # the two-stage use of structure the experiment is actually asking about.
    for slot in placed:
        g.add_edge(0, position_of[slot])

    prompt_node = n - 1
    g.add_node(prompt_node, text=f"{ANSWER_PREFIX} {bp.codes[answer_slot]}")

    g.graph.update(
        prompt_node=prompt_node,
        question_node=0,
        graph_id=bp.graph_id,
        cell_n=n,
        cell_t=t,
        hops=hops,
        gold_code=bp.codes[answer_slot],
        gold_id=bp.node_ids[answer_slot],
        start_id=bp.node_ids[chain[0]],
        chain_ids=[bp.node_ids[s] for s in chain],
        gold_position=placed.index(answer_slot) + 1,
        codes=[bp.codes[s] for s in placed],
        max_needle_offset=kv_max_offset,
    )
    return g


def sample_cell(bp_id, cfg, split="train"):
    """Pick a training cell: N uniform, then T uniform among the admissible T at N.

    Balancing N (the axis the experiment is about) and letting T fill the
    remaining length budget — see README §A.6.
    """
    rng = _rng(cfg.data_seed, cfg.data_format_version, "cell", split, bp_id)
    admissible = {}
    for (n, t) in cfg.train_cells():
        admissible.setdefault(n, []).append(t)
    n = rng.choice(sorted(admissible))
    return n, rng.choice(admissible[n])


def build_split_graphs(cfg, tokenizer, corpus, code_pool, id_pool, split, n_graphs,
                       cell=None, hops=None, id_offset=0, verbose=True):
    """Build ``n_graphs`` graphs for one split.

    ``cell`` pins (N, T) — that is a grid cell (Phase 3). ``cell=None`` draws a
    cell per graph from the training mixture (Phase 2). ``hops`` pins k the same
    way; ``hops=None`` draws it from ``cfg.hop_counts`` per graph.
    """
    graphs = []
    for i in range(n_graphs):
        bp = make_blueprint(id_offset + i, cfg, code_pool, id_pool, len(corpus), split=split)
        n, t = cell if cell is not None else sample_cell(bp.graph_id, cfg, split=split)
        k = hops if hops is not None else sample_hops(bp.graph_id, cfg, split=split)
        if k > 0:
            g = realize_chain(bp, cfg, n, t, tokenizer, corpus, split=split, hops=k)
        else:
            g = realize(bp, cfg, n, t, tokenizer, corpus, split=split)
        graphs.append(g)
        if verbose and (i + 1) % 200 == 0:
            print(f"[data] {split}: {i + 1}/{n_graphs} graphs")
    return graphs


def answer_prefix_len(tokenizer):
    """Token count of ``ANSWER_PREFIX`` — the label mask boundary in the PROMPT node."""
    return len(tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"])
