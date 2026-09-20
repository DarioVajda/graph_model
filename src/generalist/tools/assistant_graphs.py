"""Materialise a composed assistant set as graphs, and report what they cost.

The end of §9.4's pipeline: build -> write -> accept -> compose -> **this**. It
turns each composed row into the graph `mol/assistant` would train on and reports
the shape of what comes out, because the few-shot axis buys its demonstrations
with context and nothing else in this plan measures that.

    RUNMOD=src.generalist.tools.assistant_graphs src/generalist/tools/run_cli.sh \
        --composed src/generalist/results/assistant/v2/composed \
        --config src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc \
        --cell molecule_generalist_instruct_graph_s0

**Why token counts and not node counts.** A demonstration adds a molecule's worth
of nodes, and the node count is the cheap thing to report, but the packed
sequence is what has to fit: `TextGraphDataset` concatenates every node's text,
so four demonstrations is five molecules of atom text plus four question/answer
pairs at the front of the prompt. A row that does not fit is not a slow row, it
is a truncated one — and truncation would take the answer off the end.

The report is per shot count, so the cost of the axis is readable as a
difference rather than as an average over a set that is 78 % single-molecule.
"""

import argparse
import json
import os
import statistics
import sys


def _load(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composed", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many rows (0 = all)")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from rdkit import Chem
    from transformers import AutoTokenizer

    from ..adapters.molecules import resolved_prompt_style
    from ..assistant import named_atoms_for, question_text, shot_molecules
    from ..config import RunConfig, load_config_file
    from ...experiments.molecules.config import RunConfig as MolConfig
    from ...experiments.molecules.dataset import build_assistant_example

    config = RunConfig(**load_config_file(args.config, args.cell)).validate()
    adapter = config.adapter_config()
    # `mol/assistant` is a corpus task with a free-text answer, so `task` here is
    # only a stand-in that carries the encoding: the scope of an example comes
    # from its own facts, through `named_atoms_for`, not from the task name.
    cfg = MolConfig(task="ring_count", arm="graph",
                    encoding=adapter.encoding,
                    stereo_tags=adapter.stereo_tags,
                    model_name=adapter.model_name,
                    question_node=adapter.question_node,
                    prompt_style=resolved_prompt_style(adapter)).validate()

    tokenizer = AutoTokenizer.from_pretrained(adapter.model_name)
    rows = _load(os.path.join(args.composed, f"{args.split}.jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} {args.split} rows")

    by_shots, failures = {}, []
    for row in rows:
        mol = Chem.MolFromSmiles(row["key"])
        if mol is None:
            failures.append({"id": row["id"], "why": "unparseable key"})
            continue
        shots = shot_molecules(row)
        try:
            graph = build_assistant_example(mol, question_text(row),
                                            row["answer"],
                                            named_atoms_for(row["facts"]),
                                            shots, cfg)
        except Exception as exc:                     # noqa: BLE001 — reported
            failures.append({"id": row["id"], "why": f"{type(exc).__name__}: {exc}"})
            continue
        text = "".join(graph.nodes[i].get("text", "")
                       for i in range(graph.number_of_nodes()))
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        entry = by_shots.setdefault(len(shots), {"rows": 0, "nodes": [],
                                                 "tokens": []})
        entry["rows"] += 1
        entry["nodes"].append(graph.number_of_nodes())
        entry["tokens"].append(tokens)

    report = {"split": args.split, "rows": len(rows),
              "failures": len(failures), "failure_detail": failures[:20],
              "by_shot_count": {}}
    for count in sorted(by_shots):
        entry = by_shots[count]
        report["by_shot_count"][str(count)] = {
            "rows": entry["rows"],
            "nodes_mean": round(statistics.mean(entry["nodes"]), 1),
            "nodes_max": max(entry["nodes"]),
            "tokens_mean": round(statistics.mean(entry["tokens"]), 1),
            "tokens_p95": sorted(entry["tokens"])[int(0.95 * (len(entry["tokens"]) - 1))],
            "tokens_max": max(entry["tokens"]),
        }
    all_tokens = [t for entry in by_shots.values() for t in entry["tokens"]]
    if all_tokens:
        report["tokens_max_overall"] = max(all_tokens)
        report["tokens_mean_overall"] = round(statistics.mean(all_tokens), 1)
    print(json.dumps(report, indent=1, sort_keys=True))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
