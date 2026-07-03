"""
Answer-coverage ceiling analysis for SR-WebQSP (the README table).

These ceilings bound how well *any* model can do given SR retrieval: a gold
answer is **present** iff its ``kb_id`` occurs as a node in ``subgraph.tuples``.
Every metric assumes perfect precision (the model emits only correct, present
golds), so real achievable numbers are strictly below.

Per question (``golds`` = unique gold kb_ids, ``present`` = golds in the graph):
  * hits1_ceiling     — fraction of questions with >= 1 present gold (bounds Hits@1).
  * recall_macro      — mean over questions of present/total.
  * recall_micro      — sum(present)/sum(total); dominated by the >n_max
                        enumeration tail, diagnostic only (NOT a benchmark ceiling).
  * f1_macro          — mean of 2R/(1+R) (perfect precision => P=1, F1=2R/(1+R));
                        this is the WebQSP metric, the operative ceiling.
  * recall_macro_cap / f1_macro_cap — same, with emitted answers capped at
                        ``n_max`` (R_c = min(present, n_max)/total).

Questions with no gold kb_id at all are excluded (nothing to score against).

Runs as part of data prep when the config sets ``analyse_dataset``:

    python -m src.experiments.kgqa --mode data_prep --analyse-dataset <flags>

and saves the numbers next to the built splits as ``coverage_analysis.json``.
"""

import json
import os

# Row order + labels mirror the README table.
_METRICS = (
    ("hits1_ceiling", ">=1 gold present (Hits@1 ceiling)"),
    ("recall_macro", "Recall - macro"),
    ("recall_micro", "Recall - micro (diagnostic)"),
    ("f1_macro", "F1 - macro, perfect precision"),
    ("recall_macro_cap", "Recall - macro, cap n_max"),
    ("f1_macro_cap", "F1 - macro, cap n_max"),
)


def _question_coverage(record):
    """(present, total) unique gold kb_ids for one question; total may be 0."""
    golds = {a["kb_id"] for a in record.get("answers", []) if a.get("kb_id")}
    nodes = set()
    for h, _, t in record["subgraph"]["tuples"]:
        nodes.add(h)
        nodes.add(t)
    return len(golds & nodes), len(golds)


def analyse_split(records, n_max):
    """Coverage-ceiling metrics for one split's raw SR records."""
    per_q = [pt for pt in map(_question_coverage, records) if pt[1] > 0]
    n = len(per_q)
    recalls = [p / t for p, t in per_q]
    capped = [min(p, n_max) / t for p, t in per_q]
    return {
        "n_questions": n,
        "n_skipped_no_gold": len(records) - n,
        "hits1_ceiling": sum(p >= 1 for p, _ in per_q) / n,
        "recall_macro": sum(recalls) / n,
        "recall_micro": sum(p for p, _ in per_q) / sum(t for _, t in per_q),
        "f1_macro": sum(2 * r / (1 + r) for r in recalls) / n,
        "recall_macro_cap": sum(capped) / n,
        "f1_macro_cap": sum(2 * r / (1 + r) for r in capped) / n,
    }


def _format_table(results, n_max):
    splits = list(results)
    head = "| Ceiling | " + " | ".join(
        f"{s} (n={results[s]['n_questions']})" for s in splits) + " |"
    lines = [head, "|---|" + "---|" * len(splits)]
    for key, label in _METRICS:
        label = label.replace("n_max", f"N_max={n_max}")
        row = " | ".join(f"{results[s][key] * 100:.1f}%" for s in splits)
        lines.append(f"| {label} | {row} |")
    return "\n".join(lines)


def run_analysis(cfg, sr_dir, split_files, out_dir):
    """Analyse each raw split, print the README-style table, save the JSON.

    ``split_files`` maps split name -> filename under ``sr_dir`` (the driver
    passes ``process_dataset.SPLITS``). Results land in
    ``<out_dir>/coverage_analysis.json`` next to the built ``.gtds`` splits.
    """
    results = {}
    for split, fname in split_files.items():
        records = [json.loads(l) for l in open(os.path.join(sr_dir, fname))]
        results[split] = analyse_split(records, cfg.n_max)

    print(f"\n[analyse_dataset] answer-coverage ceilings (perfect precision, n_max={cfg.n_max}):")
    print(_format_table(results, cfg.n_max))

    payload = {"n_max": cfg.n_max, "splits": results}
    out_path = os.path.join(out_dir, "coverage_analysis.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[analyse_dataset] saved to {out_path}")
    return payload
