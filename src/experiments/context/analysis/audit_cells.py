"""Pre-registered structural audit of the built data — the gate before any GPU time.

Two different questions get conflated when people ask "is this comparison fair?", and
this script answers them separately because they have different remedies.

**Leakage** — can the graph channel identify the answer without reading the text? An
oracle that sees every shortest-path distance and no text plays the best distance-only
strategy available; its accuracy above uniform guessing is the leak. This is what
`fan_out=2` exists to remove, and it is what the ORIGINAL Phase 2 gate got wrong: it
printed "answer ALONE at distance k" while computing `len(shell_k) == 1`, i.e. how often
the distance-k shell is a singleton, never whether that singleton is the answer. At
N=16/k=4 those are 34.5% and 9%. A whole design decision (dropping N=16) rested on the
mislabeled number. Hence this file: the statistic is named after what it computes.

**Signal** — is there anything in the topology to use? `P(answer in shell_k)` collapses
as k grows and as N shrinks, because a decoy can shortcut the answer to below distance k.
Where it is low, a graph≈flat result is uninterpretable: you cannot separate "the model
failed" from "the structure was uninformative". That is not a fairness problem and not a
reason to drop a cell — it is a covariate to publish beside the accuracy heatmap, because
it predicts the shape of the surface.

Also re-run here per cell, because they are cheap and a build bug would otherwise reach
the figure: out-degree uniformity, and the edge <-> verbalized-reference bijection that
established the arms see the same information (README §A.10).

    ./.venv/bin/python -m src.experiments.context.analysis.audit_cells
"""

import argparse
import collections
import json
import os
import re

import networkx as nx

from ..config import RunConfig
from ..process_dataset import OUTPUT_ROOT, cell_split_name, load_split

# Node ids contain a hyphen (NODE-02314). An earlier version of this regex used
# [A-Za-z0-9]+ and matched nothing, which reported "3600 edges not verbalized" — a
# fabricated cheat finding. Keep the hyphen in the class.
REF = re.compile(r"(?:Continue at|Decoy reference:) ([A-Za-z0-9-]+)\.")
DEFINES = re.compile(r"access code for ([A-Za-z0-9-]+) is")


def audit_split(ds, hops, fan_out, limit=200):
    """Every statistic for one (N, T, k) split."""
    n_graphs = min(limit, len(ds))
    outdeg = collections.Counter()
    shell_sizes, in_shell = [], 0
    acc_shell = acc_deep = 0.0
    unverbalized = orphan_refs = 0
    q_outdeg = collections.Counter()
    checked = 0

    for gi in range(n_graphs):
        g = ds.graphs[gi]
        qn, pn = g.graph["question_node"], g.graph["prompt_node"]
        content = [v for v in g.nodes if v not in (qn, pn)]
        q_outdeg[g.out_degree(qn)] += 1
        for v in content:
            outdeg[g.out_degree(v)] += 1

        idof = {}
        for v in content:
            m = DEFINES.search(g.nodes[v].get("text", ""))
            if m:
                idof[m.group(1)] = v

        # The fairness check: the edge set and the verbalized-reference set must be
        # the same set. An edge with no sentence behind it is information the graph
        # arm has and the flat arm does not — a cheat. The converse is a build bug.
        for v in content:
            named = {idof[r] for r in REF.findall(g.nodes[v].get("text", "")) if r in idof}
            linked = {w for w in g.successors(v) if w in content}
            unverbalized += len(linked - named)
            orphan_refs += len(named - linked)

        start, answer = idof.get(g.graph["start_id"]), idof.get(g.graph["gold_id"])
        if start is None or answer is None:
            continue
        checked += 1

        dist = nx.single_source_shortest_path_length(g.subgraph(content), start)
        shell = [v for v, d in dist.items() if d == hops]
        shell_sizes.append(len(shell))
        if answer in shell:
            in_shell += 1
            acc_shell += 1.0 / len(shell)
        dmax = max(dist.values())
        deepest = [v for v, d in dist.items() if d == dmax]
        if answer in deepest:
            acc_deep += 1.0 / len(deepest)

    n_content = len(content)
    chance = 1.0 / n_content
    spd_only = max(acc_shell, acc_deep) / max(1, checked)
    return {
        "graphs": checked,
        "p_answer_in_shell": in_shell / max(1, checked),
        "mean_shell": sum(shell_sizes) / max(1, len(shell_sizes)),
        "spd_only_acc": spd_only,
        "chance": chance,
        "leak": spd_only - chance,
        "outdeg_uniform": set(outdeg) == {fan_out},
        "outdeg": dict(outdeg),
        "question_outdeg_uniform": len(q_outdeg) == 1,
        "edges_not_verbalized": unverbalized,
        "refs_without_edge": orphan_refs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--node-counts", default="16,32,64,128")
    ap.add_argument("--token-counts", default="64,128,256,512")
    ap.add_argument("--hop-counts", default="1,2,3,4")
    ap.add_argument("--fan-out", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--data-seed", type=int, default=42)
    # Part of data_config_key(), so it has to be settable or the audit silently
    # reads a different build than the one about to be trained on (bias_sharing's
    # 4k-capped tree is the first consumer that is not the 16k default).
    ap.add_argument("--max-train-len", type=int, default=16384)
    ap.add_argument("--magnetic-m", type=int, default=128)
    ap.add_argument("--limit", type=int, default=200, help="graphs sampled per cell")
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    args = ap.parse_args()

    ints = lambda s: tuple(int(v) for v in s.split(","))
    cfg = RunConfig(node_counts=ints(args.node_counts), token_counts=ints(args.token_counts),
                    hop_counts=ints(args.hop_counts), fan_out=args.fan_out,
                    n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test,
                    data_seed=args.data_seed, max_train_len=args.max_train_len,
                    magnetic_m=args.magnetic_m,
                    mode="data_prep").validate()

    print(f"\ndata: {os.path.join(OUTPUT_ROOT, cfg.data_config_key())}\n")
    header = (f"{'cell':>14}  {'P(ans in shell)':>15}  {'|shell|':>7}  "
              f"{'SPD-only':>8}  {'chance':>7}  {'leak':>7}  {'FAIRNESS':>8}")
    print(header)
    print("-" * len(header))

    rows, failures, missing = [], [], []
    for (n, t) in cfg.cells():
        for k in cfg.hops_list():
            name = cell_split_name(n, t, k)
            try:
                ds = load_split(cfg, name)
            except FileNotFoundError:
                print(f"{f'N{n} T{t} k{k}':>14}  NOT BUILT")
                missing.append((n, t, k))
                continue
            r = audit_split(ds, k, cfg.fan_out, limit=args.limit)
            r.update(n_nodes=n, tokens_per_node=t, hops=k)
            rows.append(r)

            clean = (r["edges_not_verbalized"] == 0 and r["refs_without_edge"] == 0
                     and r["outdeg_uniform"] and r["question_outdeg_uniform"])
            if not clean:
                failures.append(r)
            print(f"{f'N{n} T{t} k{k}':>14}  {r['p_answer_in_shell']:>14.1%}  "
                  f"{r['mean_shell']:>7.2f}  {r['spd_only_acc']:>7.1%}  {r['chance']:>6.1%}  "
                  f"{r['leak']:>+6.1%}  {'ok' if clean else 'FAIL':>8}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {len(rows)} cell records -> {args.out}")

    print()
    # A gate that passes on an empty audit is worse than no gate: it reports a clean
    # bill of health for a build it never opened. Every expected cell must be present.
    if missing or not rows:
        print(f"GATE FAILED: {len(missing)} of {len(cfg.cells()) * len(cfg.hops_list())} "
              "cells are not built — nothing was audited for them.")
        for cell in missing[:10]:
            print(f"  missing: N={cell[0]} T={cell[1]} k={cell[2]}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print("Check that the data build finished and that the axes/seed/magnetic_m "
              "given here match the ones it was built with (they form the cache key).")
        return 1
    if failures:
        print(f"GATE FAILED on {len(failures)} cell(s) — do NOT train on this build:")
        for r in failures:
            print(f"  N={r['n_nodes']} T={r['tokens_per_node']} k={r['hops']}: "
                  f"edges_not_verbalized={r['edges_not_verbalized']} "
                  f"refs_without_edge={r['refs_without_edge']} outdeg={r['outdeg']}")
        return 1
    print("GATE PASSED: edge set == verbalized-reference set, out-degree uniform, in "
          "every cell. The graph channel is a re-encoding of what the text already says.")
    print("Publish the leak column beside the accuracy heatmap — it is a covariate, not "
          "a pass/fail: low P(answer in shell) means the topology is uninformative there, "
          "so a graph~=flat result in that cell is not evidence about the architecture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
