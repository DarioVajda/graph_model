"""Print assembled neighborhoods as the model would read them. The PLAN.md M4 gate.

The point is to be *read by a human*. It answers questions no metric will: does the
neighborhood plausibly contain evidence for the label, is the auto-derived column spec
emitting junk, is one relation drowning out the others, how many tokens does it cost.

It goes through the same `data.py` assembly the cache build uses, so what is printed is
exactly what gets tokenized -- a dump with its own rendering path would drift and stop being
evidence about anything.

    python3 -m src.experiments.relbench --mode dump --dataset rel-f1 --task driver-dnf
    python3 -m src.experiments.relbench --mode dump --dataset rel-trial --task study-outcome \\
        --max-nodes 24 --max-value-chars 1200 --max-node-chars 4000 --dump-n 2
"""

import numpy as np

from .data import (
    answer_text, build_flat_graph, build_graph, make_builders, question_text,
)

# Spacing between sampled row indices, so consecutive examples are not near-duplicates of
# each other (task tables are ordered by timestamp).
STRIDE = 137


def _assemble(cfg, task, sampler, renderer, description, df, i, split):
    row = df.iloc[i]
    seed_ts = int(row[task.time_col].timestamp())
    entity = int(row[task.entity_col])
    question = question_text(task, description, entity, seed_ts)
    answer = answer_text(task, row[task.target_col])

    sampled = sampler.sample(entity, seed_ts, identity=(cfg.dataset, cfg.task, split))
    graph = (build_flat_graph(sampled, renderer, seed_ts, question, answer,
                              text_mode=cfg.text_mode) if cfg.is_flat()
             else build_graph(sampled, renderer, seed_ts, question, answer,
                              text_mode=cfg.text_mode,
                              question_node=cfg.question_node,
                              prompt_node=cfg.prompt_node))
    return sampled, graph


def _document(graph):
    return "\n".join(str(graph.nodes[n]["text"]) for n in graph.nodes)


def dump(cfg, n=3, split="train", stats_n=100, tokenizer_name=None):
    task, sampler, renderer, description = make_builders(cfg)
    df = task.get_table(split, mask_input_cols=False).df

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name or cfg.model_name)
    except Exception as exc:                                    # noqa: BLE001
        print(f"(no tokenizer: {exc}; reporting characters only)\n")

    print("=" * 78)
    print(f"{cfg.dataset}/{cfg.task} [{split}] arm={cfg.arm()} max_nodes={cfg.max_nodes} "
          f"sampling={cfg.neighbor_sampling} text={cfg.text_mode} time={cfg.time_encoding} "
          f"anonymize={cfg.anonymize} max_value_chars={cfg.max_value_chars}")
    print("=" * 78)

    report = renderer.column_report()
    print("\nCOLUMN DERIVATION (auto-derived; no hand-written spec anywhere)")
    for name in sorted(report):
        r = report[name]
        line = f"  {name:24s} keep={r['kept']}"
        if r["dropped_null"]:
            line += f"  drop_missing={r['dropped_null']}"
        if r["anonymized"]:
            line += f"  ANON={r['anonymized']}"
        print(line)

    for k in range(n):
        i = (k * STRIDE) % len(df)
        sampled, graph = _assemble(cfg, task, sampler, renderer, description, df, i, split)
        doc = _document(graph)
        print("\n" + "-" * 78)
        print(f"EXAMPLE {k}  (table row {i})  nodes={len(graph)}  edges={graph.number_of_edges()}"
              f"  chars={len(doc)}" + (f"  tokens={len(tok.encode(doc))}" if tok else ""))
        print(f"sampled tables: {sampled.tables()}")
        print("-" * 78)
        for pos, node in enumerate(graph.nodes):
            hop = sampled.nodes[node][2] if isinstance(node, int) else "-"
            print(f"  [{str(node):>12}] h{hop} {graph.nodes[node]['text']}")
        print(f"EDGES (child->parent): {list(graph.edges)}")

    # -- size summary -------------------------------------------------------
    stats_n = min(stats_n, len(df))
    nodes, chars, tokens, node_texts = [], [], [], []
    for k in range(stats_n):
        i = (k * STRIDE) % len(df)
        sampled, graph = _assemble(cfg, task, sampler, renderer, description, df, i, split)
        doc = _document(graph)
        nodes.append(len(sampled))
        chars.append(len(doc))
        node_texts.extend(str(graph.nodes[n]["text"]) for n in graph.nodes)
        if tok:
            tokens.append(len(tok.encode(doc)))

    def p(a, q):
        return int(np.percentile(a, q))

    print("\n" + "=" * 78)
    print(f"SIZE over {stats_n} examples")
    print(f"  sampled rows  mean={np.mean(nodes):6.1f}  p50={p(nodes,50):5d}  "
          f"p90={p(nodes,90):5d}  max={max(nodes):5d}")
    print(f"  chars         mean={np.mean(chars):6.0f}  p50={p(chars,50):5d}  "
          f"p90={p(chars,90):5d}  max={max(chars):5d}")
    if tokens:
        print(f"  tokens        mean={np.mean(tokens):6.0f}  p50={p(tokens,50):5d}  "
              f"p90={p(tokens,90):5d}  max={max(tokens):5d}")
        print(f"  tokens/row    {np.sum(tokens)/np.sum(nodes):.1f}")
    empty = sum(1 for c in nodes if c <= 1)
    print(f"  seed-only graphs (no eligible history): {empty}/{stats_n} "
          f"({100*empty/stats_n:.1f}%)")

    # Truncation is invisible in a token count -- the document just gets quietly smaller.
    # A `max_node_chars` of 600 once cut 95.5% of rel-trial's `studies` rows, which made
    # `max_value_chars` unreachable and left the free text this experiment is about on the
    # floor. Report it so it cannot happen again without someone seeing it.
    if cfg.max_node_chars:
        cut = sum(1 for t in node_texts if t.endswith("…"))
        print(f"  nodes cut by max_node_chars={cfg.max_node_chars}: {cut}/{len(node_texts)} "
              f"({100*cut/max(1,len(node_texts)):.1f}%)"
              + ("   <-- raise it or set it to None" if cut else ""))
    else:
        print("  max_node_chars: none (per-field max_value_chars is the only text cap)")
