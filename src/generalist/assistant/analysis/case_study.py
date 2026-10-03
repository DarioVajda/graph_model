"""Thirty hand-written questions, put to one or more checkpoints, printed side by side.

§9.4's step 12, and the only read in this stack that is not downstream of the
composed set's own distribution. `score.py` measures the test split,
which was drawn, rendered, voiced and filtered by the same pipeline that made the
training rows; every automated check it runs was built against that surface and
is blind in the same direction. These thirty are written by hand, over test-role
molecules, and are **read, not scored** — `probe` says what each row is for and
`expect` is a reading aid for the checklist. Nothing here computes a rate.

    GPU=1 src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.analysis.case_study \\
        --config src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc \\
        --cell molecule_generalist_instruct_graph_s0 \\
        --checkpoint control=.../replay_anneal15_graph_s0/anneal/checkpoint-12255 \\
        --checkpoint assistant=.../assistant_anneal_graph_s0/anneal/checkpoint-12255 \\
        --out src/generalist/results/assistant/case_study

**Zero shots, on purpose.** 7,682 of the composed set's 9,869 train rows carry no
demonstration, and with no shots `assistant.question_text` adds no pointer line,
so a bare hand-written question is exactly the shape those rows have. Asking with
shots would also make the reply readable as a copy of the nearest component,
which is the thing a case study is least able to tell apart by eye.

**The examples are built through `_materialise`, not beside it.** That is what
makes these graphs the same object the leg trained on — same prompt format, same
prompt-node wiring, same SPD and magnetic features — rather than a second
construction that could differ in a way the reading would then be about. The one
thing it needs that a case-study row does not have is an answer: the schema
refuses an empty one, and the packed text has to reach the answer boundary for a
prompt to end where generation starts. So `expect` is stored in the answer slot.
It is **not a reference string**: no metric is computed against it here, and any
overlap score taken against it later would be measuring a note to a reader.

The artifact lands under a cache root of its own, so it cannot collide with the
`mol/assistant` test split in the campaign's build directory. `cache_root` is
popped from `build_version`, so the two agree on everything but where they sit.
"""

import argparse
import json
import os
import sys
from dataclasses import replace

#: The hand-written set. One JSON object per line, the first of which is a
#: `_comment` describing the file and is skipped.
CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "case_study.jsonl")


def load_cases(path: str) -> list:
    rows = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "_comment" in row:
                continue
            rows.append(row)
    return rows


def draws_for(cases: list) -> list:
    """The adapter's 6-tuples for a list of case rows.

    ``atoms`` is 1-based in the file, the fact sheet's convention, and
    `build_assistant_example` wires the prompt node with RDKit's 0-based indices
    — the same conversion `assistant.named_atoms_for` makes, done here because a
    case row carries indices rather than facts. An empty list is a
    molecule-level question and wires the prompt to the whole molecule.
    """
    from rdkit import Chem

    draws = []
    for row in cases:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"{row['id']}: unparseable SMILES {row['smiles']!r}")
        named = sorted({int(a) - 1 for a in row.get("atoms") or []})
        for index in named:
            if not 0 <= index < mol.GetNumAtoms():
                raise ValueError(
                    f"{row['id']}: atom {index + 1} is outside a molecule with "
                    f"{mol.GetNumAtoms()} atoms")
        meta = {"id": row["id"], "probe": row.get("probe", ""),
                "expect": row.get("expect", ""), "smiles": row["smiles"],
                "atoms": list(row.get("atoms") or []), "shots": []}
        # See the module docstring: the answer slot holds the reading aid, and
        # the leading space is `_materialise`'s convention for a text answer.
        draws.append((mol, row["question"], " " + (row.get("expect") or "—"),
                      named, row["id"], meta))
    return draws


def build_source(config, adapter_config, out: str, cases: list):
    """A `MoleculeTaskSource` over the case rows, built by the adapter's own path."""
    from ...adapters import molecules as adapter
    from ....utils import TextGraphDataset

    cache = replace(adapter_config, cache_root=os.path.join(out, "cache"))
    cache.validate()
    adapter.configure(cache)
    spec = adapter.task_specs(cache)[
        f"{adapter.MOLECULE_PREFIX}{adapter.ASSISTANT_TASK}"]
    sidecar = adapter._materialise(cache, adapter.ASSISTANT_TASK, "test",
                                   config.arm, 0, draws_for(cases), spec)
    path = cache.source_path(adapter.ASSISTANT_TASK, "test", config.arm, 0)
    return adapter.MoleculeTaskSource(TextGraphDataset.load(path), sidecar, path)


def run(args) -> int:
    from ...config import RunConfig, load_config_file
    from ...evaluate.scorers import generate_predictions
    from ...fork import load_start_weights
    from ... import wiring

    checkpoints = []
    for item in args.checkpoint:
        label, _, path = item.partition("=")
        if not path:
            label, path = "model", item
        checkpoints.append((label, path))

    cases = load_cases(args.cases)
    config = RunConfig(**load_config_file(args.config, args.cell)).validate()
    _registry, adapter_config = wiring.build_registry(config)

    os.makedirs(args.out, exist_ok=True)
    source = build_source(config, adapter_config, args.out, cases)
    indices = list(range(len(source)))
    print(f"{len(indices)} case rows, arm {config.arm}", flush=True)

    run_ = wiring.build_run(config, output_dir=os.path.join(args.out, "scratch"),
                            fire_validators=False)
    replies = {}
    for label, path in checkpoints:
        load_start_weights(run_.trainer, path)
        predictions, _targets = generate_predictions(
            run_.model, run_.tokenizer, run_.collator, source, indices,
            max_new_tokens=args.max_new_tokens, device=run_.device)
        replies[label] = predictions
        print(f"{label}: {len(predictions)} replies from {path}", flush=True)

    rows = []
    for i, case in enumerate(cases):
        rows.append({"id": case["id"], "smiles": case["smiles"],
                     "atoms": case.get("atoms") or [],
                     "question": case["question"], "probe": case.get("probe"),
                     "expect": case.get("expect"),
                     "replies": {label: replies[label][i] for label, _ in checkpoints}})

    with open(os.path.join(args.out, "case_study.jsonl"), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(args.out, "case_study.md"), "w") as handle:
        handle.write(_markdown(rows, [label for label, _ in checkpoints],
                               checkpoints))
    print(f"wrote {len(rows)} cases to {args.out}/case_study.md")
    return 0


def _markdown(rows: list, labels: list, checkpoints: list) -> str:
    """The readable artifact. One section per case, one block per checkpoint."""
    out = ["# Assistant case study", "",
           "Hand-written questions over test-role molecules, read rather than "
           "scored. `expect` is a reading aid and no metric is computed against "
           "it.", ""]
    for label, path in checkpoints:
        out.append(f"* **{label}** — `{path}`")
    out.append("")
    for row in rows:
        out.append(f"## {row['id']} — {row['probe']}")
        out.append("")
        out.append(f"`{row['smiles']}`"
                   + (f", atoms {row['atoms']}" if row["atoms"] else ""))
        out.append("")
        out.append(f"> {row['question']}")
        out.append("")
        out.append(f"*expect:* {row['expect']}")
        out.append("")
        for label in labels:
            reply = row["replies"][label] or "(empty)"
            out.append(f"**{label}:** {reply}")
            out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH, repeatable; the label heads its column")
    parser.add_argument("--cases", default=CASES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
