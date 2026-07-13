"""
D2.1 — flat (text-serialization) data prep: the same capped SR subgraph as
plain triple text, no graph structure (README goal #2 control arm).

Serializes the *raw selected triples* — ``select_triples(max_nodes)`` output,
PRE-Levi / PRE-CVT-collapse: the collapse is part of OUR graph processing, so
the honest "subgraph as text" control skips it — with the same entity texts
(entities_names v2 via ``resolve_entity_text``) and the same relation
verbalization, one ``head | relation | tail`` line per triple:

    {question}\n{h | r | t per line}\nAnswer: {a1}\n{a2}...

Targets are IDENTICAL to the graph arm's by construction: the same
``present_answer_texts`` on the same built Levi graph, the same per-split
augmentation RNG stream (same seed string, same consumption order), the same
``n_max`` truncation and newline join, the same unanswerable-eval-row rule.
Only the *input representation* differs — which is the point.

Output: ``processed_datasets/<flat key>/{split}.jsonl`` rows
``{question, lines, target, gold_answers, unanswerable}`` — text is composed
at train time (flat_train), so the seq-len-4096 outlier truncation (drop
trailing triple lines) never mutates the stored data.

Train rows follow the graph pipeline's versions layout: [q1_v1..q1_v8,
q2_v1..], consumed by a dataset whose __getitem__ samples one version.

CPU-only. Routed from __main__ via ``--mode flat_data_prep``.
"""

import json
import os
import random

from tqdm import tqdm

from .config import NAMING_VERSION
from .process_dataset import (
    entity_names_path, OUTPUT_ROOT,
    build_base_levi, present_answer_texts, full_gold_texts,
    resolve_entity_text, verbalize_relation, select_triples)
from .sr_records import load_sr_records


def flat_data_config_key(cfg, dataset):
    """Cache dir for ONE dataset's flat serialization (only flat-relevant knobs).

    Graph-feature keys (spd/magnetic/rcm/max_length) are irrelevant here;
    ``max_nodes`` stays because it selects WHICH triples survive the cap.
    Dataset-resolvable knobs embed their RESOLVED values (cf. data_config_key).
    """
    model = str(cfg.model_name).replace("/", "-")
    ps = cfg.resolved_prompt_style
    # f"sr-{dataset}" keeps pre-existing keys identical (prefix was "sr-webqsp-flat_").
    return (f"sr-{dataset}-flat_{model}_v{cfg.rel_mode}"
            f"_cap{cfg.resolved_max_nodes(dataset)}"
            f"_nmax{cfg.resolved_n_max(dataset)}"
            f"_ver{cfg.resolved_versions(dataset)}_seed{cfg.data_seed}"
            f"_dfv{cfg.data_format_version}"
            + ("" if ps == "plain" else f"_ps{ps}")
            + ("" if cfg.naming_version == NAMING_VERSION else f"_nm{cfg.naming_version}")
            + ("_cvt" if cfg.resolved_cvt_collapse("flat") else ""))


def triple_lines(record, entity_names, cfg):
    """The serialized subgraph: one 'h | r | t' line per raw selected triple."""
    lines = []
    for h, r, t in select_triples(record, cfg.max_nodes):
        lines.append(f"{resolve_entity_text(h, entity_names)} | "
                     f"{verbalize_relation(r, cfg.rel_mode)} | "
                     f"{resolve_entity_text(t, entity_names)}")
    return lines


MAX_REL_CHAIN = 8   # collapse chains are short; the cap only guards degenerate cycles


def collapsed_triple_lines(record, entity_names, cfg):
    """Text serialization of the COLLAPSED Levi graph (2x2 collapse ablation).

    The graph arm's collapse contracts p -> rel0 -> cvt -> rel_i -> leaf into
    p -> rel0 -> rel_i -> leaf; its exact text equivalent composes the relation
    texts through the removed mediator: 'p | rel0 rel_i | leaf'. Uncollapsed
    mediators (multi-parent / leaf-only) remain endpoints, exactly as the graph
    arm keeps them as nodes. One line per entity->...->entity relation path."""
    G = build_base_levi(record, entity_names, cfg.rel_mode, cfg.max_nodes,
                        cvt_collapse=True)
    lines = []
    for e in G.nodes():
        if G.nodes[e].get("is_rel"):
            continue
        for r0 in G.successors(e):
            stack = [(r0, (G.nodes[r0]["text"],), frozenset([r0]))]
            while stack:
                rn, rels, on_path = stack.pop()
                for s in G.successors(rn):
                    if G.nodes[s].get("is_rel"):
                        if s in on_path or len(rels) >= MAX_REL_CHAIN:
                            continue
                        stack.append((s, rels + (G.nodes[s]["text"],),
                                      on_path | {s}))
                    else:
                        lines.append(f"{G.nodes[e]['text']} | "
                                     f"{' '.join(rels)} | {G.nodes[s]['text']}")
    return lines


def build_flat_rows(record, entity_names, cfg, versions, rng, keep_unanswerable):
    """Mirror ``build_question_graphs`` exactly (skip rules + RNG consumption),
    emitting flat rows instead of graphs. ``present`` always comes from the
    COLLAPSED base graph (like the graph arm's targets — collapse never removes
    a nameable answer node, so the target sets are identical either way)."""
    base = build_base_levi(record, entity_names, cfg.rel_mode, cfg.max_nodes)
    present = present_answer_texts(base, record)
    gold = full_gold_texts(record)
    lines_fn = (collapsed_triple_lines if cfg.resolved_cvt_collapse("flat")
                else triple_lines)
    if not present:
        if keep_unanswerable and gold:
            lines = lines_fn(record, entity_names, cfg)
            return [{"question": record["question"], "lines": lines,
                     "target": "", "gold_answers": gold, "unanswerable": True}]
        return []
    lines = lines_fn(record, entity_names, cfg)
    rows = []
    for _ in range(versions):
        order = present[:]
        rng.shuffle(order)
        rows.append({"question": record["question"], "lines": lines,
                     "target": cfg.answer_sep.join(order[: cfg.n_max]),
                     "gold_answers": gold, "unanswerable": False})
    return rows


def run_flat_data_prep_mode(cfg, splits=None):
    """Serialize every (dataset × flat config) cache this run references.

    ``splits=None`` builds what each dataset's role requires (train iff
    trained on, dev+test iff evaluated on — see ``role_splits``).
    """
    from .process_dataset import role_splits
    if cfg.resolved_prompt_style != "plain":
        raise NotImplementedError(
            "flat + chat formatting is deliberately deferred until the scale "
            "run needs it (decision 2026-07-08).")
    entity_names = json.load(open(entity_names_path(cfg)))

    for dataset in cfg.datasets:
        view = cfg.for_dataset(dataset)      # per-dataset knobs resolved to scalars
        out_dir = os.path.join(OUTPUT_ROOT, flat_data_config_key(view, dataset))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"flat_data_config_key": flat_data_config_key(view, dataset),
                       "dataset": dataset,
                       "data_format_version": view.data_format_version,
                       "naming_version": view.naming_version,
                       "rel_mode": view.rel_mode, "max_nodes": view.max_nodes,
                       "n_max": view.n_max, "versions": view.versions,
                       "data_seed": view.data_seed, "model_name": view.model_name},
                      f, indent=2)

        ds_splits = splits if splits is not None else role_splits(cfg, dataset)
        print(f"[flat_data_prep] {dataset}: out_dir={out_dir} splits={ds_splits}")

        for split in ds_splits:
            out_path = os.path.join(out_dir, f"{split}.jsonl")
            if os.path.exists(out_path):
                print(f"[flat_data_prep] {split}: already present — skipping.")
                continue
            records = load_sr_records(dataset, split)
            versions = view.versions if split == "train" else 1
            keep_unanswerable = split != "train"
            rng = random.Random(f"{view.data_seed}:{split}")

            rows, kept, skipped = [], 0, 0
            for rec in tqdm(records, desc=f"Serializing {split}"):
                if not rec.get("answers"):
                    skipped += 1
                    continue
                rs = build_flat_rows(rec, entity_names, view, versions, rng,
                                     keep_unanswerable)
                if not rs:
                    skipped += 1
                    continue
                rows.extend(rs)
                kept += 1
            n_unans = sum(1 for r in rows if r["unanswerable"])
            print(f"[{split}] kept {kept} questions ({len(rows)} rows, {versions}x; "
                  f"{n_unans} unanswerable empty-target rows), skipped {skipped}")
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, out_path)

        print(f"[flat_data_prep] {dataset}: done. Cached dataset at {out_dir}\n")
