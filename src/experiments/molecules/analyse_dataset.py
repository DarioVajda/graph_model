"""M0 — measure Tier B before committing to any modelling choice.

Answers the three questions PLAN.md §3.1 currently answers with arithmetic:

1. How big are these graphs really — atoms, bonds, Levi nodes?
2. How many tokens, under `rich` vs `terse` vs the flat SMILES control? The ratio
   **nodes per token** is the quantity the measured overhead curve keys on
   (7.9x at 2048 nodes x 2 tokens vs 1.45x at 512 x 32, `CLAUDE_CONTEXT.md` §4.5),
   so it, not node count, is what decides whether molecules are cheap.
3. Does every molecule in every dataset survive the round trip?

Login-node safe: no torch, no GPU, and sampled by default. Run:

    .venv/bin/python -m src.experiments.molecules.analyse_dataset
    .venv/bin/python -m src.experiments.molecules.analyse_dataset --full --sample 0
"""

import argparse
import statistics as stats

from .data import (
    ENCODINGS,
    TIER_B,
    attach_question,
    flat_serialize,
    load_tier_b,
    mol_to_graph,
    roundtrip_check,
    scaffold_split,
)


def _tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def _summary(values):
    if not values:
        return dict(mean=0.0, median=0, p95=0, max=0)
    ordered = sorted(values)
    return dict(
        mean=stats.mean(values),
        median=ordered[len(ordered) // 2],
        p95=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        max=ordered[-1],
    )


def analyse(name, count_tokens, sample=1000, roundtrip=True):
    records, spec, dropped = load_tier_b(name)
    total = len(records)
    subset = records[:: max(1, total // sample)] if sample else records

    atoms, bonds = [], []
    tokens = {enc: [] for enc in ENCODINGS}
    flat_tokens = []
    rt_fail = {enc: 0 for enc in ENCODINGS}

    for record in subset:
        mol = record["mol"]
        atoms.append(mol.GetNumAtoms())
        bonds.append(mol.GetNumBonds())
        flat_tokens.append(count_tokens(flat_serialize(mol)))

        for enc in ENCODINGS:
            graph = attach_question(
                mol_to_graph(mol, encoding=enc),
                question="Question: does this molecule cross the blood-brain barrier?",
                answer=" yes", prompt_edges="all")
            tokens[enc].append(
                sum(count_tokens(d["text"]) for _, d in graph.nodes(data=True)))
            if roundtrip and not roundtrip_check(mol, encoding=enc)[0]:
                rt_fail[enc] += 1

    train, valid, test = scaffold_split([r["smiles"] for r in records])

    return dict(
        name=name, kind=spec.kind, total=total, dropped=dropped,
        sampled=len(subset),
        n_targets=len(subset[0]["targets"]) if subset else 0,
        atoms=_summary(atoms), bonds=_summary(bonds),
        tokens={e: _summary(v) for e, v in tokens.items()},
        flat=_summary(flat_tokens),
        rt_fail=rt_fail,
        split=(len(train), len(valid), len(test)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(TIER_B))
    ap.add_argument("--sample", type=int, default=1000,
                    help="molecules per dataset; 0 = all (slow on HIV)")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--no-roundtrip", action="store_true")
    args = ap.parse_args()

    count_tokens = _tokenizer(args.tokenizer)
    rows = [analyse(name, count_tokens, sample=args.sample,
                    roundtrip=not args.no_roundtrip) for name in args.datasets]

    print("\n### Corpus and graph size\n")
    print("| dataset | kind | mols | dropped (parse/bond) | endpoints | "
          "atoms mean/p95/max | bonds mean | Levi N mean/p95 | scaffold split |")
    print("|---|---|---:|---:|---:|---|---:|---|---|")
    for r in rows:
        levi_mean = r["atoms"]["mean"] + r["bonds"]["mean"] + 2
        levi_p95 = r["atoms"]["p95"] + r["bonds"]["p95"] + 2
        drop = f"{r['dropped']['parse']}/{r['dropped']['unsupported_bond']}"
        print(f"| `{r['name']}` | {r['kind'][:5]} | {r['total']} | {drop} | "
              f"{r['n_targets']} | "
              f"{r['atoms']['mean']:.1f} / {r['atoms']['p95']} / {r['atoms']['max']} | "
              f"{r['bonds']['mean']:.1f} | {levi_mean:.1f} / {levi_p95} | "
              f"{'/'.join(str(x) for x in r['split'])} |")

    print("\n### Tokens per example, and the cost-driving ratio\n")
    print("| dataset | flat SMILES | rich_levi | terse_levi | rich_atom_only | "
          "nodes/token rich | nodes/token terse |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        levi_n = r["atoms"]["mean"] + r["bonds"]["mean"] + 2
        rich = r["tokens"]["rich_levi"]["mean"]
        terse = r["tokens"]["terse_levi"]["mean"]
        print(f"| `{r['name']}` | {r['flat']['mean']:.0f} | {rich:.0f} | {terse:.0f} | "
              f"{r['tokens']['rich_atom_only']['mean']:.0f} | "
              f"{levi_n / max(rich, 1):.2f} | {levi_n / max(terse, 1):.2f} |")

    print("\n### Round-trip failures (must be 0 everywhere)\n")
    print("| dataset | sampled | " + " | ".join(f"`{e}`" for e in ENCODINGS) + " |")
    print("|---|---:|" + "---:|" * len(ENCODINGS))
    for r in rows:
        cells = " | ".join(str(r["rt_fail"][e]) for e in ENCODINGS)
        print(f"| `{r['name']}` | {r['sampled']} | {cells} |")

    worst = max((r["rt_fail"][e] for r in rows for e in ENCODINGS), default=0)
    print(f"\nMax round-trip failures in any cell: **{worst}**")


if __name__ == "__main__":
    main()
