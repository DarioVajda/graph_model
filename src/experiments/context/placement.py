"""
Controlled node placement — the lost-in-the-middle probe (README §3.4).

§3.3 varies N, T and k, never *where* the evidence sits: content nodes are shuffled per
graph, so the gold chain lands at a uniformly random place in the flat serialization.
That supports "flat degrades with node count" but not "flat gets lost in the middle",
which is a claim about position. This mode puts position under experimental control.

For each test graph the k+1 chain nodes (start, ..., answer) are placed as ONE
contiguous block, and the block is moved through the sequence:

  * ``random``   the §3.3 order, unchanged. It reproduces the §3.3 cell exactly and is
                 the sanity check that the pipeline scores what the grid scored.
  * ``fwd@p``    the chain in hop order (start -> answer), block starting at relative
                 position p among the content nodes. p=0 is the start of the context;
                 p=1 puts the block last, directly before the QUESTION in the flat arm.
  * ``rev@p``    the same with the chain reversed (answer -> start), so every hop points
                 BACK to text the causal model has already read.

Only the order of the node texts changes. The texts, the question, the non-chain nodes'
relative order and the supervised span are identical across conditions, so every graph
is its own control and the conditions are compared paired.

Both arms are scored. The flat arm re-serializes; the graph arm permutes the node
indices of the stored item (texts, token ids, SPD, magnetic eigenvectors, edges), which
changes the order nodes are packed in and nothing else. Under per-node RoPE reset and a
bidirectional prefix the graph arm should be invariant to that permutation, so its rows
are the control that the effect in the flat rows is serialization, not data.

Every item is written out — correct/incorrect, the failure class, and the summed
log-probability of the gold code tokens — because accuracy near floor (k=4 at N=128 is
0.08) or ceiling cannot show a position effect, while the log-probability can.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ...train import get_device

from .data import KV_TEMPLATE
from .evaluate import _classify, windowed_forward
from .flat import FlatCellView, FlatCollator, content_order, serialize_graph
from .process_dataset import cell_split_name, load_split
from ._io import append_jsonl

PLACEMENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITIONS = (("random",)
              + tuple(f"fwd@{p:.2f}" for p in PLACEMENTS)
              + tuple(f"rev@{p:.2f}" for p in PLACEMENTS))

# Node-indexed columns the permutation knows how to reorder. Anything else that is
# node-indexed and present would be silently left in the old order, so it is refused.
_UNSUPPORTED_NODE_COLUMNS = ("laplacian_coordinates", "rwse", "rrwp", "landmark")


def parse_condition(cond):
    """``"fwd@0.50"`` -> ``("fwd", 0.5)``; ``"random"`` -> ``("random", None)``."""
    if cond == "random":
        return "random", None
    direction, _sep, p = cond.partition("@")
    if direction not in ("fwd", "rev") or not _sep:
        raise ValueError(f"unknown placement condition {cond!r} (expected one of {CONDITIONS})")
    return direction, float(p)


def chain_nodes(g):
    """Node indices of the gold chain, in hop order: ``[start, ..., answer]``.

    A chain node is found by its KV sentence, which names its own id exactly once in
    the whole graph. Pointer and decoy sentences name ids too, so matching the bare id
    would also hit the nodes that point AT a chain node.
    """
    qn, pn = g.graph["question_node"], g.graph["prompt_node"]
    out = []
    for cid in g.graph["chain_ids"]:
        needle = KV_TEMPLATE.format(node_id=cid, code="")[:-1]   # "...for NODE-x is "
        hits = [j for j in g.nodes if j not in (qn, pn) and needle in g.nodes[j]["text"]]
        if len(hits) != 1:
            raise ValueError(f"graph {g.graph.get('graph_id')}: chain id {cid} matched "
                             f"{len(hits)} nodes by its KV sentence, expected exactly 1")
        out.append(hits[0])
    return out


def placed_order(base, chain, direction, p):
    """``base`` with the chain lifted out and re-inserted as one block at relative position p.

    The non-chain nodes keep their ``base`` relative order, so two conditions differ
    only in where the block sits and which way it runs.
    """
    chain_set = set(chain)
    others = [j for j in base if j not in chain_set]
    block = list(chain) if direction == "fwd" else list(reversed(chain))
    at = int(round(p * len(others)))
    return others[:at] + block + others[at:]


def condition_order(g, base, cond):
    """Content-node order for condition ``cond``, given the graph's §3.3 order ``base``."""
    direction, p = parse_condition(cond)
    if direction == "random":
        return list(base)
    return placed_order(base, chain_nodes(g), direction, p)


def answer_position(order, g):
    """Relative position of the ANSWER node in ``order``: 0 = first content node, 1 = last."""
    ans = chain_nodes(g)[-1]
    return order.index(ans) / max(1, len(order) - 1)


# ── The flat arm ───────────────────────────────────────────────────────────────

class PlacedFlatCollator(FlatCollator):
    """``FlatCollator`` whose content order is set by a placement condition.

    The base order is the collator's own §3.3 shuffle (``data_seed + index``), so the
    ``random`` condition is byte-identical to what ``flat_trained_grid.jsonl`` scored.
    """

    def __init__(self, tokenizer, code_len, data_seed=42, condition="random"):
        super().__init__(tokenizer, code_len, data_seed=data_seed)
        self.condition = condition

    def order_for(self, graph, index):
        return condition_order(graph, content_order(graph, self.data_seed + index),
                               self.condition)

    def _row(self, graph, index):
        prefix = serialize_graph(graph, self.order_for(graph, index))
        full = f"{prefix} {graph.graph['gold_code']}"
        ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        ids = ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(ids)
        labels[-self.n_supervised:] = ids[-self.n_supervised:]
        return ids, labels


# ── The graph arm ──────────────────────────────────────────────────────────────

def graph_base_order(g):
    """The graph arm's stored content order: node-index order, as ``_pack_one`` packs it."""
    qn, pn = g.graph["question_node"], g.graph["prompt_node"]
    return [j for j in sorted(g.nodes) if j not in (qn, pn)]


def permute_item(item, g, content):
    """Relabel a stored item so its content nodes are packed in the order ``content``.

    New index 0 is the QUESTION, 1..M the content nodes in ``content`` order, and the
    PROMPT last — the collator packs by index with the prompt moved to the end, so this
    sets the pack order and nothing else. Every node-indexed field moves together:
    token ids and text by row, SPD by row and column, magnetic eigenvectors by row
    (the eigenvalues are permutation-invariant), edges by relabelling.
    """
    qn, pn = g.graph["question_node"], g.graph["prompt_node"]
    new_to_old = [qn] + list(content) + [pn]
    if sorted(new_to_old) != list(range(item["num_nodes"])):
        raise ValueError("placement order is not a permutation of the graph's nodes")
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    for col in _UNSUPPORTED_NODE_COLUMNS:
        if item.get(col) is not None:
            raise ValueError(f"permute_item does not reorder {col!r}; add it before using it")

    out = dict(item)
    out["text"] = [item["text"][j] for j in new_to_old]
    out["input_ids"] = [item["input_ids"][j] for j in new_to_old]
    out["prompt_node"] = old_to_new[pn]
    out["edges"] = [(old_to_new[u], old_to_new[v]) for u, v in item["edges"]]
    out["original_ids"] = {k: old_to_new[v] for k, v in item["original_ids"].items()}
    idx = torch.as_tensor(new_to_old, dtype=torch.long)
    if item.get("shortest_path_dists") is not None:
        out["shortest_path_dists"] = item["shortest_path_dists"][idx][:, idx]
    if item.get("magnetic_V") is not None:
        out["magnetic_V"] = item["magnetic_V"][idx]
    return out


class PlacedGraphView:
    """A built split whose items are permuted into a placement condition's pack order."""

    def __init__(self, split, condition="random"):
        self._split = split
        self.graphs = split.graphs
        self.condition = condition

    def __len__(self):
        return len(self._split)

    def order_for(self, graph, index):
        return condition_order(graph, graph_base_order(graph), self.condition)

    def __getitem__(self, i):
        g = self.graphs[i]
        return permute_item(self._split[i], g, self.order_for(g, i))


# ── Scoring ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_items(model, dataset, collator, tokenizer, device):
    """Per-item scoring: ``grid_eval``'s decisions plus the gold-code log-probability.

    ``code_correct`` and the failure class are computed exactly as ``grid_eval`` computes
    them, so the per-condition means are comparable with every §3.3 number.
    """
    model.eval()
    rows = []
    for i in range(len(dataset)):
        batch = collator([dataset[i]])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits, window_labels = windowed_forward(model, batch, labels)
        shift = logits[0, :-1, :].float()
        gold = window_labels[0, 1:]
        mask = gold != -100
        pred_ids = shift.argmax(dim=-1)[mask].tolist()
        gold_ids = gold[mask].tolist()
        # The supervised span is [code tokens..., EOS]; the log-probability is of the
        # code alone, the same span code_acc judges.
        logp = F.log_softmax(shift[mask], dim=-1)
        code_lp = logp[torch.arange(len(gold_ids) - 1), gold[mask][:-1]].sum().item()

        g = dataset.graphs[i].graph
        text_ids = [t for t in pred_ids if t != tokenizer.eos_token_id]
        pred_text = tokenizer.decode(text_ids).strip()
        distractors = {c for c in g.get("codes", []) if c != g["gold_code"]}
        rows.append({
            "item": i,
            "code_correct": int(pred_ids[:-1] == gold_ids[:-1]),
            "em": int(pred_ids == gold_ids),
            "cls": _classify(pred_ids, gold_ids, pred_text, g["gold_code"], distractors),
            "code_logprob": code_lp,
            "packed_len": int(batch["input_ids"].shape[1]),
        })
    return rows


def _load_flat(cfg, device):
    """Base backbone + the trained LoRA adapter, exactly as the flat arm trained it."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.torch_dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, cfg.checkpoint_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def run_placement_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    """Score one checkpoint under every placement condition on the selected cells and k.

    Writes one summary record per (cell, k, condition) to ``runs_jsonl`` and every item
    to ``<runs_jsonl dir>/items/<run_name>.jsonl`` — one file per run, so parallel jobs
    never append to the same per-item file.
    """
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    runs_jsonl = runs_jsonl or os.path.join(exp_dir, "results", "placement", "runs.jsonl")
    items_path = os.path.join(os.path.dirname(runs_jsonl), "items",
                              f"{run_name or 'local'}.jsonl")
    os.makedirs(os.path.dirname(items_path), exist_ok=True)
    sweep_meta = {k: v for k, v in (("sweep_id", sweep_id), ("sweep_run", run_name)) if v}

    torch.set_num_threads(1)
    device = get_device()
    arm = cfg.placement_arm
    if arm == "flat":
        model, tokenizer = _load_flat(cfg, device)
    else:
        from .model import build_collator, load_checkpoint_model
        model, tokenizer, _ = load_checkpoint_model(cfg.checkpoint_path, cfg, device)
        graph_collator = build_collator(cfg, tokenizer, for_grid=True)

    conditions = cfg.selected_conditions()
    mixed = bool(cfg.hop_counts)
    train_cells = set(cfg.train_cells())
    cells = sorted(cfg.selected_cells(), key=lambda c: cfg.cell_length(*c), reverse=True)
    for (n, t) in cells:
      for k in cfg.selected_hops():
        split = load_split(cfg, cell_split_name(n, t, k if mixed else None))
        n_items = len(split) if not cfg.placement_max_items else min(len(split), cfg.placement_max_items)
        for cond in conditions:
            if arm == "flat":
                collator = PlacedFlatCollator(tokenizer, cfg.code_len,
                                              data_seed=cfg.data_seed, condition=cond)
                view = FlatCellView(split)
                order_of = collator.order_for
            else:
                collator = graph_collator
                view = PlacedGraphView(split, condition=cond)
                order_of = view.order_for
            view = _Head(view, n_items)
            rows = score_items(model, view, collator, tokenizer, device)

            direction, p = parse_condition(cond)
            with open(items_path, "a") as f:
                for r in rows:
                    g = split.graphs[r["item"]]
                    order = order_of(g, r["item"])
                    f.write(json.dumps({
                        "arm": arm, **sweep_meta, "checkpoint_path": cfg.checkpoint_path,
                        "n_nodes": n, "tokens_per_node": t, "hops": k,
                        "condition": cond, "direction": direction, "placement": p,
                        "answer_pos": answer_position(order, g),
                        **r,
                    }) + "\n")

            acc = float(np.mean([r["code_correct"] for r in rows]))
            lp = float(np.mean([r["code_logprob"] for r in rows]))
            append_jsonl(runs_jsonl, {
                "mode": "placement", "arm": arm, **sweep_meta,
                "checkpoint_path": cfg.checkpoint_path,
                "n_nodes": n, "tokens_per_node": t, "hops": k,
                "in_train_distribution": (n, t) in train_cells,
                "condition": cond, "direction": direction, "placement": p,
                "n": len(rows), "code_acc": acc, "mean_code_logprob": lp,
                "em": float(np.mean([r["em"] for r in rows])),
                "distractor_rate": float(np.mean([r["cls"] == "distractor" for r in rows])),
                "malformed_rate": float(np.mean([r["cls"] == "malformed" for r in rows])),
                "packed_len": rows[0]["packed_len"] if rows else None,
            })
            print(f"[placement:{arm}] N={n:4d} T={t:4d} k={k} {cond:>9s}  "
                  f"code_acc={acc:.3f}  logp={lp:.3f}", flush=True)
        del split


class _Head:
    """The first ``n`` items of a view, keeping ``.graphs`` aligned with the items."""

    def __init__(self, view, n):
        self._view, self._n = view, n
        self.graphs = view.graphs

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if i >= self._n:
            raise IndexError(i)
        return self._view[i]
