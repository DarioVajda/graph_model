"""
Synthetic task generators for the probe suite (see README.md, the agreed spec).

Four probes, one generator each, all sharing the same skeleton: build a graph,
pick a balanced Yes/No question about it, verbalize the nodes, and attach a
*prompt node* whose text ends in the answer token (only that token is
supervised — ``get_prompt_node_labels``).

  direction     — directed reachability on an oriented connected graph.
  substructure  — ring membership on a molecule-like rings+trees graph.
  local_hop     — "is there a <color> node within r directed hops of X?",
                  with a guaranteed out-of-ball decoy in every example.
  text_path     — direction at bios scale: persons with templated multi-
                  sentence bios; edges appear redundantly in the text.

One `.gtds` artifact per (task, sizes, knobs, data_seed) is cached under
``datasets/`` and shared by every bias arm: it always carries BOTH the SPD and
magnetic features (at these node counts the collate cost of an unused feature
is negligible, so arm speed readouts stay honest). A ``.meta.json`` sidecar
records the generator statistics (label balance, negative mix, ...).
"""

import fcntl
import json
import os
import random

import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer

from ...utils import TextGraphDataset
from ..expressiveness.data.data_gen import make_node_labels, get_prompt_node_labels

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(EXPERIMENT_DIR, "datasets")

COLORS = ("red", "blue", "green", "yellow", "orange",
          "purple", "pink", "brown", "black", "white")

# text_path vocabulary. Names combine into globally-unique (first, last) pairs
# per graph; relation words are pure flavor (reachability never resolves a
# labeled chain, so no matching constraints are needed — README, probe 4).
FIRST_NAMES_M = ("James", "John", "Robert", "Michael", "William", "David",
                 "Richard", "Joseph", "Thomas", "Charles", "Daniel", "Matthew",
                 "Anthony", "Mark", "Steven", "Andrew", "Paul", "Joshua",
                 "Kevin", "Brian")
FIRST_NAMES_F = ("Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
                 "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Lisa",
                 "Nancy", "Sandra", "Ashley", "Emily", "Donna", "Michelle",
                 "Carol", "Amanda", "Laura")
LAST_NAMES = ("Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
              "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
              "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Clark",
              "Lewis", "Walker", "Young", "Allen", "King")
LIKES = ("chess", "sailing", "painting", "hiking", "baking", "photography",
         "gardening", "tennis", "reading", "cycling", "fishing", "dancing",
         "cooking", "running", "climbing", "skiing", "pottery", "archery",
         "surfing", "knitting")
HAIR_COLORS = ("brown", "black", "blond", "red", "gray")
EYE_COLORS = ("blue", "green", "brown", "gray", "hazel")
RELATIONS = ("sibling", "colleague", "friend", "neighbor", "cousin")
JOBS = ("teacher", "nurse", "carpenter", "software engineer", "chef",
        "librarian", "electrician", "journalist", "pharmacist", "architect",
        "plumber", "accountant", "photographer", "veterinarian", "translator",
        "mechanic", "florist", "dentist", "pilot", "tailor")
CITIES = ("Lisbon", "Oslo", "Prague", "Vienna", "Dublin", "Zagreb", "Ghent",
          "Porto", "Krakow", "Seville", "Turin", "Lyon", "Bergen", "Graz",
          "Utrecht", "Ljubljana", "Valencia", "Aarhus", "Bologna", "Leipzig")


# ── shared graph machinery ───────────────────────────────────────────────────

def _connected_skeleton(n, max_degree=None, max_extra_edges=None):
    """Connected undirected graph on ``n`` nodes: a random recursive tree
    (guaranteed connectivity, ~log diameter) plus random extra edges for cycles.
    ``max_degree`` caps every node's degree (a tree always has a leaf below any
    cap >= 2, so attachment never gets stuck)."""
    nodes = list(range(n))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    perm = nodes[:]
    random.shuffle(perm)
    for i in range(1, n):
        if max_degree is None:
            parent = perm[random.randint(0, i - 1)]
        else:
            parent = random.choice(
                [perm[j] for j in range(i) if G.degree(perm[j]) < max_degree])
        G.add_edge(perm[i], parent)

    cap = n if max_extra_edges is None else max_extra_edges
    for _ in range(random.randint(0, cap)):
        u, v = random.sample(nodes, 2)
        if G.has_edge(u, v):
            continue
        if max_degree is not None and (G.degree(u) >= max_degree or G.degree(v) >= max_degree):
            continue
        G.add_edge(u, v)
    return G


def _orient(G, p_bidirectional):
    """Orient an undirected graph: each edge stays bidirectional with
    ``p_bidirectional``, else it gets one direction (50/50)."""
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes)
    for u, v in G.edges:
        r = random.random()
        if r < p_bidirectional:
            D.add_edge(u, v)
            D.add_edge(v, u)
        elif r < p_bidirectional + (1.0 - p_bidirectional) / 2:
            D.add_edge(u, v)
        else:
            D.add_edge(v, u)
    return D


def _sample_reachability_pair(D, label, tries=20):
    """Sample (x, y) with y (label=True) / not (label=False) reachable from x.

    The undirected skeleton is connected, so every negative pair is
    undirected-connected by construction — direction is the only signal.
    Returns ``(x, y, neg_type)`` (neg_type None for positives) or None when no
    such pair was found (caller regenerates the graph)."""
    nodes = list(D.nodes)
    for _ in range(tries):
        x = random.choice(nodes)
        reachable = nx.descendants(D, x)
        if label:
            if reachable:
                return x, random.choice(sorted(reachable)), None
        else:
            unreachable = sorted(set(nodes) - reachable - {x})
            if unreachable:
                y = random.choice(unreachable)
                neg_type = ("reverse_reachable" if x in nx.descendants(D, y)
                            else "mutually_unreachable")
                return x, y, neg_type
    return None


def _attach_prompt_node(D, question):
    """Append the prompt node (packed last by the collator) with the question
    text; the caller adds its edges to the queried node(s)."""
    prompt_id = max(D.nodes) + 1
    D.add_node(prompt_id, text=question)
    D.graph["prompt_node"] = prompt_id
    return prompt_id


def _assign_spreadsheet_labels(D):
    texts = make_node_labels(D.number_of_nodes())
    for node, text in zip(D.nodes, texts):
        D.nodes[node]["text"] = text


# ── probe 1: direction ───────────────────────────────────────────────────────

def make_direction_graph(cfg, stats):
    lo, hi = cfg.node_range()
    while True:
        D = _orient(_connected_skeleton(random.randint(lo, hi)), cfg.p_bidirectional)
        label = random.random() < 0.5
        pair = _sample_reachability_pair(D, label)
        if pair is not None:
            break
    x, y, neg_type = pair
    if neg_type:
        stats[neg_type] = stats.get(neg_type, 0) + 1
    _assign_spreadsheet_labels(D)
    question = (f"Is there a directed path from{D.nodes[x]['text']} to"
                f"{D.nodes[y]['text']}? {'Yes' if label else 'No'}")
    prompt_id = _attach_prompt_node(D, question)
    D.add_edge(prompt_id, x)
    D.add_edge(prompt_id, y)
    return D, label


# ── probe 2: substructure ────────────────────────────────────────────────────

def make_substructure_graph(cfg, stats):
    lo, hi = cfg.node_range()
    n_target = random.randint(lo, hi)

    G = nx.Graph()
    ring_nodes = set()
    size = random.randint(3, 8)
    nx.add_cycle(G, range(size))
    ring_nodes.update(range(size))
    next_id = size

    while G.number_of_nodes() < n_target:
        budget = n_target - G.number_of_nodes()
        # Glue a ring at a low-degree node (degree <= 2 keeps the max-4 cap),
        # else attach a tree node. Rings glued at single vertices keep every
        # cycle exactly a constructed ring, so ring membership is exact.
        glue_candidates = [v for v in G.nodes if G.degree(v) <= 2]
        if budget >= 2 and glue_candidates and random.random() < 0.5:
            size = random.randint(3, min(8, budget + 1))
            glue = random.choice(glue_candidates)
            new_nodes = list(range(next_id, next_id + size - 1))
            next_id += size - 1
            cycle = [glue] + new_nodes
            nx.add_cycle(G, cycle)
            ring_nodes.update(cycle)
        else:
            parent = random.choice([v for v in G.nodes if G.degree(v) < 4])
            G.add_edge(parent, next_id)
            next_id += 1

    tree_nodes = sorted(set(G.nodes) - ring_nodes)
    if not tree_nodes:  # guarantee both classes exist (may exceed n_target by 1)
        parent = random.choice([v for v in G.nodes if G.degree(v) < 4])
        G.add_edge(parent, next_id)
        tree_nodes = [next_id]

    label = random.random() < 0.5
    x = random.choice(sorted(ring_nodes) if label else tree_nodes)
    stats["ring_share"] = stats.get("ring_share", 0) + len(ring_nodes) / G.number_of_nodes()

    D = G.to_directed()  # symmetric digraph: undirected probe, uniform storage
    _assign_spreadsheet_labels(D)
    question = (f"Is the node{D.nodes[x]['text']} part of a ring? "
                f"{'Yes' if label else 'No'}")
    prompt_id = _attach_prompt_node(D, question)
    D.add_edge(prompt_id, x)
    return D, label


# ── probe 3: local_hop ───────────────────────────────────────────────────────

def make_local_hop_graph(cfg, stats):
    lo, hi = cfg.node_range()
    colors = COLORS[:cfg.num_colors]
    radius = cfg.hop_radius
    while True:
        D = _orient(_connected_skeleton(random.randint(lo, hi)), cfg.p_bidirectional)
        nodes = list(D.nodes)
        for _ in range(20):
            x = random.choice(nodes)
            dists = nx.single_source_shortest_path_length(D, x, cutoff=radius)
            ball = sorted(v for v in dists if v != x)              # 1..r directed hops
            out_of_ball = sorted(set(nodes) - set(dists))          # distance > r (incl. inf)
            if ball and out_of_ball:
                break
        else:
            continue
        break

    for v in nodes:
        D.nodes[v]["color"] = random.choice(colors)
    label = random.random() < 0.5
    target = random.choice([c for c in colors if c != D.nodes[x]["color"]])
    if label:
        D.nodes[random.choice(ball)]["color"] = target
    else:
        for v in ball:
            if D.nodes[v]["color"] == target:
                D.nodes[v]["color"] = random.choice([c for c in colors if c != target])
    # Every example carries >= 1 out-of-ball decoy of the target color, so
    # "the color exists somewhere" carries no signal (README, probe 3).
    if not any(D.nodes[v]["color"] == target for v in out_of_ball):
        D.nodes[random.choice(out_of_ball)]["color"] = target
    stats["ball_size_sum"] = stats.get("ball_size_sum", 0) + len(ball)

    labels = make_node_labels(D.number_of_nodes())
    for node, lbl in zip(D.nodes, labels):
        D.nodes[node]["text"] = f"{lbl}: {D.nodes[node]['color']}"
    x_label = labels[list(D.nodes).index(x)]
    question = (f"Is there a {target} node within {radius} hops of{x_label} "
                f"(following edge directions)? {'Yes' if label else 'No'}")
    prompt_id = _attach_prompt_node(D, question)
    D.add_edge(prompt_id, x)
    return D, label


# ── probe 4: text_path ───────────────────────────────────────────────────────

def _sample_unique_people(n):
    """n distinct (full_name, is_male) pairs."""
    people, used = [], set()
    while len(people) < n:
        is_male = random.random() < 0.5
        first = random.choice(FIRST_NAMES_M if is_male else FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        if (first, last) in used:
            continue
        used.add((first, last))
        people.append((f"{first} {last}", is_male))
    return people


def _person_bio(name, is_male, neighbor_names):
    """Templated bio: name-first likes sentence, then attribute + relation
    sentences in shuffled order. The attribute sentences are natural filler
    carrying no task signal, sized so bios land in the README's ~40-60
    tokens/node budget. Every textual mention of a person is exactly one
    out-edge (text <-> edges fully redundant — README, probe 4)."""
    likes = random.sample(LIKES, random.randint(2, 3))
    likes_text = " and ".join([", ".join(likes[:-1]), likes[-1]] if len(likes) > 2 else likes)
    he, his = ("He", "His") if is_male else ("She", "Her")
    sentences = [
        f"{he} has {random.choice(HAIR_COLORS)} hair and {random.choice(EYE_COLORS)} eyes.",
        f"{he} is {random.randint(18, 80)} years old and works as a {random.choice(JOBS)}.",
        f"{he} lives in {random.choice(CITIES)}.",
    ]
    sentences += [f"{his} {random.choice(RELATIONS)} is {neighbor}."
                  for neighbor in neighbor_names]
    random.shuffle(sentences)
    return " ".join([f"{name} likes {likes_text}."] + sentences)


def make_text_path_graph(cfg, stats):
    lo, hi = cfg.node_range()
    while True:
        # Degree cap 3 on the skeleton bounds out-degree at 3 mentions per bio,
        # keeping every bio inside the ~40-60 token budget.
        skeleton = _connected_skeleton(random.randint(lo, hi), max_degree=3,
                                       max_extra_edges=max(1, (hi - lo) // 4))
        D = _orient(skeleton, cfg.p_bidirectional)
        label = random.random() < 0.5
        pair = _sample_reachability_pair(D, label)
        if pair is not None:
            break
    x, y, neg_type = pair
    if neg_type:
        stats[neg_type] = stats.get(neg_type, 0) + 1

    people = _sample_unique_people(D.number_of_nodes())
    names = {node: person[0] for node, person in zip(D.nodes, people)}
    genders = {node: person[1] for node, person in zip(D.nodes, people)}
    for node in D.nodes:
        D.nodes[node]["text"] = " " + _person_bio(
            names[node], genders[node],
            [names[nb] for nb in D.successors(node)])

    question = (f"Is there a chain of acquaintances leading from {names[x]} "
                f"to {names[y]}? {'Yes' if label else 'No'}")
    prompt_id = _attach_prompt_node(D, question)
    D.add_edge(prompt_id, x)
    D.add_edge(prompt_id, y)
    return D, label


TASK_GENERATORS = {
    "direction": make_direction_graph,
    "substructure": make_substructure_graph,
    "local_hop": make_local_hop_graph,
    "text_path": make_text_path_graph,
}


# ── dataset build / cache / splits ───────────────────────────────────────────

def dataset_path(cfg):
    """Artifact path encoding everything that changes the generated content."""
    lo, hi = cfg.node_range()
    total = cfg.train_size + cfg.val_size + cfg.test_size
    tags = [cfg.task, f"{total}ex", f"n{lo}-{hi}"]
    if cfg.task in ("direction", "local_hop", "text_path"):
        tags.append(f"bi{cfg.p_bidirectional:g}")
    if cfg.task == "local_hop":
        tags += [f"c{cfg.num_colors}", f"r{cfg.hop_radius}"]
    if cfg.magnetic_m:
        tags.append(f"m{cfg.magnetic_m}")
    tags += [cfg.ordering, f"ds{cfg.data_seed}"]
    return os.path.join(DATASETS_DIR, "_".join(tags) + ".gtds")


def prepare_dataset(cfg):
    """Generate + featurize the full (train+val+test) dataset for ``cfg``.

    Deterministic given ``cfg.data_seed``. Both SPD and magnetic features are
    always computed so ONE artifact serves every bias arm.
    """
    random.seed(cfg.data_seed)
    generator = TASK_GENERATORS[cfg.task]
    total = cfg.train_size + cfg.val_size + cfg.test_size

    stats = {"yes": 0, "no": 0}
    graphs = []
    for _ in tqdm(range(total), desc=f"Generating {cfg.task}"):
        graph, label = generator(cfg, stats)
        stats["yes" if label else "no"] += 1
        graphs.append(graph)

    ds = TextGraphDataset(graphs, rcm_ordering=(cfg.ordering == "rcm"))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Tokenize + labels first, while the table is light (see expressiveness
    # data_gen for why); then the heavy (N,N) features.
    ds.tokenize(tokenizer)
    ds.compute_labels(get_prompt_node_labels)
    ds.compute_shortest_path_distances()
    ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m)
    ds.cast_float_features_to_fp32()
    return ds, stats


def load_or_create_dataset(cfg):
    """Load the cached `.gtds` for ``cfg``, generating it if absent (flock'd so
    parallel sbatch jobs build each artifact exactly once)."""
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
                print(f"[data] stats: {stats}")
    ds = TextGraphDataset.load(path)
    print(f"Loaded dataset from {path} with {len(ds)} examples.")
    return ds


def load_data(cfg):
    """Return the ``(train, val, test)`` split for ``cfg`` (sizes from config)."""
    ds = load_or_create_dataset(cfg)
    total = cfg.train_size + cfg.val_size + cfg.test_size
    if len(ds) < total:
        raise ValueError(
            f"Dataset at {dataset_path(cfg)} has {len(ds)} examples but "
            f"{total} are configured — stale artifact. Delete it to regenerate.")
    train_end = cfg.train_size
    val_end = cfg.train_size + cfg.val_size
    return ds[:train_end], ds[train_end:val_end], ds[val_end:total]


def run_data_prep_mode(cfg):
    """Ensure this config's `.gtds` exists (build if missing)."""
    load_or_create_dataset(cfg)
    print("[data_prep] done.")
