"""
Answer-coverage ceiling analysis for SR-WebQSP (the README tables).

Every coverage metric is measured twice per split:

  * **uncapped** — a gold answer is *present* iff its ``kb_id`` occurs as a node
    in the raw ``subgraph.tuples``. This is the pure SR-retrieval ceiling: no
    data-prep choice of ours can push a model above it.
  * **pipeline** — a gold answer is *present* iff it survives the actual
    data-prep graph (``select_triples(max_nodes)`` -> Levi -> CVT collapse) AND
    it has a scoreable text (its ``text``, else a literal kb_id — see
    ``process_dataset.answer_text``). This mirrors
    ``process_dataset.present_answer_texts``, i.e. exactly what the built
    ``.gtds`` splits can supervise and score.

All ceilings assume perfect precision (the model emits only correct, present
golds), so real achievable numbers are strictly below.

Per question (``golds`` = unique gold kb_ids, ``present`` = golds present):
  * hits1_ceiling     — fraction of questions with >= 1 present gold (bounds Hits@1).
  * recall_macro      — mean over questions of present/total.
  * recall_micro      — sum(present)/sum(total); dominated by the >n_max
                        enumeration tail, diagnostic only (NOT a benchmark ceiling).
  * f1_macro          — mean of 2R/(1+R) (perfect precision => P=1, F1=2R/(1+R));
                        this is the WebQSP metric, the operative ceiling.
  * recall_macro_cap / f1_macro_cap — same, with emitted answers capped at
                        ``n_max`` (R_c = min(present, n_max)/total).

The gap between the two variants is decomposed per question into its first
failing gate (``drop_decomposition``):
  * not_retrieved   — no gold kb_id anywhere in the raw tuples (SR failure).
  * no_text         — retrieved, but no present gold has a scoreable text
                      (``answer_text``: ``text``, else a literal kb_id).
  * lost_to_cap     — a scoreable gold was retrieved, but ``select_triples``
                      truncated all of them out at ``max_nodes``.
  * lost_to_collapse— survived the cap, but CVT collapse contracted every
                      remaining gold node away (unnamed single-parent mediators).
  * answerable      — at least one gold survives the full pipeline. Train keeps
                      only these; dev/test keep ALL answered questions (the
                      others as empty-target rows that score 0), so the eval
                      denominator equals RoG/GNN-RAG's. ``n_scoreable`` counts
                      questions with >=1 matchable gold text (where >0 score is
                      even possible).

Additionally, per *built* split (read from the ``.gtds`` next to the output),
token/node-count statistics of what the model actually sees are reported.

Questions with no gold kb_id at all are excluded (nothing to score against).

Runs as part of data prep when the config sets ``analyse_dataset``:

    python -m src.experiments.kgqa --mode data_prep --analyse-dataset <flags>

and saves the numbers next to the built splits as ``coverage_analysis.json``.
"""

import json
import os

import numpy as np

from .process_dataset import (entity_names_path, answer_text, build_base_levi,
                              select_triples)

# Row order + labels mirror the README table.
_METRICS = (
    ("hits1_ceiling", ">=1 gold present (Hits@1 ceiling)"),
    ("recall_macro", "Recall - macro"),
    ("recall_micro", "Recall - micro (diagnostic)"),
    ("f1_macro", "F1 - macro, perfect precision"),
    ("recall_macro_cap", "Recall - macro, cap n_max"),
    ("f1_macro_cap", "F1 - macro, cap n_max"),
)

_DROP_KEYS = ("answerable", "not_retrieved", "no_text", "lost_to_cap", "lost_to_collapse")

_PERCENTILES = (50, 75, 90, 95, 99)


def _ceilings(per_q, n_max):
    """Ceiling metrics from per-question (present, total) pairs (total > 0)."""
    n = len(per_q)
    recalls = [p / t for p, t in per_q]
    capped = [min(p, n_max) / t for p, t in per_q]
    return {
        "hits1_ceiling": sum(p >= 1 for p, _ in per_q) / n,
        "recall_macro": sum(recalls) / n,
        "recall_micro": sum(p for p, _ in per_q) / sum(t for _, t in per_q),
        "f1_macro": sum(2 * r / (1 + r) for r in recalls) / n,
        "recall_macro_cap": sum(capped) / n,
        "f1_macro_cap": sum(2 * r / (1 + r) for r in capped) / n,
    }


def analyse_split(records, entity_names, cfg):
    """Uncapped + pipeline coverage ceilings and drop decomposition for one split."""
    raw_pq, pipe_pq = [], []
    drops = {k: 0 for k in _DROP_KEYS}
    n_scoreable = 0

    for rec in records:
        golds = {a["kb_id"] for a in rec.get("answers", []) if a.get("kb_id")}
        if not golds:
            continue
        total = len(golds)

        raw_nodes = set()
        for h, _, t in rec["subgraph"]["tuples"]:
            raw_nodes.add(h)
            raw_nodes.add(t)
        retrieved = golds & raw_nodes
        raw_pq.append((len(retrieved), total))

        # scoreable = has a text or a literal kb_id to match against (answer_text)
        if any(answer_text(a) for a in rec["answers"]):
            n_scoreable += 1
        # golds that could ever be supervised: retrieved AND scoreable
        textful = {a["kb_id"] for a in rec["answers"]
                   if a["kb_id"] in retrieved and answer_text(a)}

        pipeline_present = set()
        if textful:
            G = build_base_levi(rec, entity_names, cfg.rel_mode, cfg.max_nodes)
            pipeline_present = {kb for kb in textful if kb in G}
        pipe_pq.append((len(pipeline_present), total))

        # first failing gate
        if pipeline_present:
            drops["answerable"] += 1
        elif not retrieved:
            drops["not_retrieved"] += 1
        elif not textful:
            drops["no_text"] += 1
        else:
            sel = select_triples(rec, cfg.max_nodes)
            sel_nodes = set(x for tri in sel for x in (tri[0], tri[2]))
            if textful & sel_nodes:
                drops["lost_to_collapse"] += 1   # survived the cap, died in CVT collapse
            else:
                drops["lost_to_cap"] += 1

    return {
        "n_questions": len(raw_pq),
        "n_skipped_no_gold": len(records) - len(raw_pq),
        "n_scoreable": n_scoreable,
        "uncapped": _ceilings(raw_pq, cfg.n_max),
        "pipeline": _ceilings(pipe_pq, cfg.n_max),
        "drop_decomposition": drops,
    }


def analyse_built_split(split_dir):
    """Token/node statistics of one built `.gtds` split (what the model sees).

    Counted per stored example; the train split holds ``versions`` augmented
    copies per question whose lengths differ only by answer order.
    """
    features_path = os.path.join(split_dir, "features")
    if not os.path.isdir(features_path):
        return None
    from datasets import load_from_disk
    ds = load_from_disk(features_path)

    tokens = np.array([sum(len(ids) for ids in ex) for ex in ds["input_ids"]])
    nodes = np.array(ds["num_nodes"])
    stat = lambda a: {
        "mean": float(a.mean()), "max": int(a.max()),
        **{f"p{q}": float(np.percentile(a, q)) for q in _PERCENTILES},
    }
    return {
        "n_examples": len(ds),
        "tokens": stat(tokens),
        "nodes": stat(nodes),
        "tokens_per_node": float(tokens.sum() / nodes.sum()),
    }


# --------------------------------------------------------------------------- #
# README-style markdown
# --------------------------------------------------------------------------- #
def _format_coverage_table(results, n_max):
    """Ceiling table with 'uncapped / pipeline' cells."""
    splits = list(results)
    head = "| Ceiling (uncapped / capped) | " + " | ".join(
        f"{s} (n={results[s]['n_questions']})" for s in splits) + " |"
    lines = [head, "|---|" + "---|" * len(splits)]
    for key, label in _METRICS:
        label = label.replace("n_max", f"N_max={n_max}")
        row = " | ".join(
            f"{results[s]['uncapped'][key] * 100:.1f}% / {results[s]['pipeline'][key] * 100:.1f}%"
            for s in splits)
        lines.append(f"| {label} | {row} |")
    return "\n".join(lines)


def _format_drop_table(results):
    splits = list(results)
    head = "| Questions | " + " | ".join(splits) + " |"
    lines = [head, "|---|" + "---|" * len(splits)]
    labels = {
        "answerable": "**answerable** (supervisable; all of train's kept rows)",
        "not_retrieved": "answer not in SR subgraph",
        "no_text": "retrieved, no scoreable answer text",
        "lost_to_cap": "lost to the `max_nodes` cap",
        "lost_to_collapse": "lost to CVT collapse",
    }
    for key in _DROP_KEYS:
        row = " | ".join(str(results[s]["drop_decomposition"][key]) for s in splits)
        lines.append(f"| {labels[key]} | {row} |")
    lines.append("| scoreable (>=1 matchable gold text) | " + " | ".join(
        str(results[s]["n_scoreable"]) for s in splits) + " |")
    lines.append("| **total answered** (dev/test keep ALL of these; eval denominator) | " + " | ".join(
        str(results[s]["n_questions"]) for s in splits) + " |")
    return "\n".join(lines)


def _format_token_table(built):
    splits = [s for s, st in built.items() if st]
    cols = ["mean"] + [f"p{q}" for q in _PERCENTILES] + ["max"]
    lines = ["| Split (examples) | " + " | ".join(cols) + " | tokens/node |",
             "|---|" + "---|" * (len(cols) + 1)]
    for s in splits:
        st = built[s]
        cells = " | ".join(f"{st['tokens'][c]:.0f}" for c in cols)
        lines.append(f"| {s} (n={st['n_examples']}) | {cells} | {st['tokens_per_node']:.2f} |")
    return "\n".join(lines)


def run_analysis(cfg, sr_dir, split_files, out_dir):
    """Analyse each split (raw + built), print README-style tables, save the JSON.

    ``split_files`` maps split name -> filename under ``sr_dir`` (the driver
    passes ``process_dataset.SPLITS``). Results land in
    ``<out_dir>/coverage_analysis.json`` next to the built ``.gtds`` splits.
    """
    entity_names = json.load(open(entity_names_path(cfg)))

    results, built = {}, {}
    for split, fname in split_files.items():
        records = [json.loads(l) for l in open(os.path.join(sr_dir, fname))]
        print(f"[analyse_dataset] {split}: analysing {len(records)} records ...")
        results[split] = analyse_split(records, entity_names, cfg)
        built[split] = analyse_built_split(os.path.join(out_dir, f"{split}.gtds"))

    print(f"\n[analyse_dataset] answer-coverage ceilings "
          f"(perfect precision, n_max={cfg.n_max}, max_nodes={cfg.max_nodes}):")
    print(_format_coverage_table(results, cfg.n_max))
    print("\n[analyse_dataset] drop decomposition (first failing gate per question):")
    print(_format_drop_table(results))
    if any(built.values()):
        print("\n[analyse_dataset] built-split token lengths (per stored example):")
        print(_format_token_table(built))

    payload = {"n_max": cfg.n_max, "max_nodes": cfg.max_nodes,
               "splits": results, "built_splits": built}
    out_path = os.path.join(out_dir, "coverage_analysis.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[analyse_dataset] saved to {out_path}")
    return payload
