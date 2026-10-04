"""Are the graph-domain builds what `adapters/_graph.py` says they are?

Run after `data_prep` on a config that names graph-domain tasks. For every built
``(task, split)`` of every graph domain the mixture touches — held-out tasks
included — it loads the source through `wiring.load_source`, exactly as a run
does, and checks each row:

  * the item validates as a schema `Example` and carries the record's key;
  * the question and prompt nodes are two distinct nodes; the question node has
    no edges, and the prompt node has out-edges only and at least one of them
    (where they sit is the ordering's business, not the domain's);
  * the supervised span decodes to the stored answer, followed by the stop
    suffix (or EOS) on the terminated kinds and by nothing on ``yesno``;
  * nothing before the cut already contains the answer (``yesno`` excepted:
    "Yes" can sit in a question);
  * an ``entities`` row carries its gold list in ``meta["gold"]``, and an oracle
    pass of the stored answers through `entity_scores` is reported — it is the
    ceiling the generated F1 can reach on the rows built;
  * no key sits in two roles across the domain's built splits.

It needs the tokenizer and torch, so it runs on a compute node:

    src/generalist/tools/launch/run_py.sh src/generalist/tools/checks/graph_domains_smoke.py \\
        --config src/generalist/configs/probes/014_graph_smoke.jsonc
"""

import argparse
import sys
from collections import defaultdict


def _edges(item):
    import numpy as np

    edges = np.asarray(item["edges"])
    if edges.size == 0:
        return []
    if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    return [(int(u), int(v)) for u, v in edges.reshape(-1, 2)]


def main() -> int:
    from src.generalist import wiring
    from src.generalist.adapters import adapter_for, get_adapter
    from src.generalist.adapters._graph import TERMINATED_KINDS
    from src.generalist.config import RunConfig, load_config_file
    from src.generalist.evaluate.scorers import answer_start, entity_scores
    from src.generalist.schema import SIDECAR_KEY, Example, validate

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="src/generalist/configs/probes/014_graph_smoke.jsonc")
    parser.add_argument("--limit", type=int, default=0,
                        help="rows checked per source (0 = all)")
    parser.add_argument("--show", type=int, default=1,
                        help="decoded rows printed per source")
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config)).validate()
    registry, _ = wiring.build_registry(config)

    from transformers import AutoTokenizer

    from src.experiments.molecules.data import prompt_format

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    fmt = prompt_format(config.prompt_style, config.model_name)

    failures = []
    for domain in config.graph_domains():
        spec_domain = get_adapter(domain).DOMAIN_SPEC
        domain_config = config.domain_adapter_config(domain)
        roles = defaultdict(set)
        print(f"== {domain}  build {spec_domain.build_version(domain_config)}")
        for spec in registry:
            if adapter_for(spec.name) != domain:
                continue
            info = spec_domain.info(spec.name)
            for split in info.built_splits():
                try:
                    source = wiring.load_source(config, spec.name, split)
                except Exception as exc:                          # noqa: BLE001
                    print(f"  {spec.name}/{split}: not built ({type(exc).__name__})")
                    continue
                role = spec_domain.role(info, split)
                n = len(source) if not args.limit else min(len(source), args.limit)
                nodes, tokens = source.lengths()
                bad = defaultdict(int)
                golds, answers = [], []
                for i in range(n):
                    item = source[i]
                    side = item[SIDECAR_KEY]
                    roles[side["key"]].add(role)
                    validate(Example.from_item(item, None, split=split), spec)
                    num = int(item["num_nodes"])
                    q, p = int(item["question_node"]), int(item["prompt_node"])
                    edges = _edges(item)
                    # No positional check: under `ordering: rcm` the dataset
                    # relabels every node, these two included, and the collator
                    # packs the prompt node last from `prompt_node` wherever it is.
                    if q == p or not (0 <= q < num and 0 <= p < num):
                        bad["layout"] += 1
                    if any(q in e for e in edges):
                        bad["question_has_edges"] += 1
                    if any(v == p for _u, v in edges) or not any(
                            u == p for u, _v in edges):
                        bad["prompt_edges"] += 1
                    ids = list(item["input_ids"][p])
                    start = answer_start(item)
                    after = tokenizer.decode(ids[start:])
                    before = tokenizer.decode(ids[:start])
                    answer = side["answer"]
                    expected = (answer if spec.answer_kind == "yesno"
                                else f" {answer}")
                    if spec.answer_kind in TERMINATED_KINDS:
                        expected += fmt.answer_suffix or tokenizer.eos_token
                    if after != expected:
                        bad["span_mismatch"] += 1
                        if bad["span_mismatch"] <= 2:
                            print(f"    row {i}: span {after!r} != {expected!r}")
                    if (spec.answer_kind != "yesno" and answer.strip()
                            and answer.strip() in before):
                        bad["answer_before_cut"] += 1
                    if spec.answer_kind == "entities":
                        gold = (side.get("meta") or {}).get("gold")
                        if not gold:
                            bad["no_gold"] += 1
                        golds.append(list(gold or []))
                        answers.append(answer)
                    if i < args.show:
                        texts = item["text"]
                        print(f"    [{spec.name}/{split} row {i}] nodes {num}, "
                              f"targets {sorted(v for u, v in edges if u == p)}")
                        print(f"      content[0] {texts[0][:100]!r}")
                        print(f"      question   {texts[q][:200]!r}")
                        print(f"      prompt     {texts[p][:200]!r}")
                line = (f"  {spec.name}/{split} p0: {len(source)} rows, checked {n}, "
                        f"nodes mean {sum(nodes) / len(nodes):.1f} max {max(nodes)}, "
                        f"tokens mean {sum(tokens) / len(tokens):.1f} max {max(tokens)}")
                if golds:
                    oracle = entity_scores(answers, golds)
                    line += (f", oracle f1 {oracle['f1']:.3f} hit1 "
                             f"{oracle['hit1']:.3f}")
                print(line + (f"  FAIL {dict(bad)}" if bad else "  ok"))
                if bad:
                    failures.append((spec.name, split, dict(bad)))
        crossed = {k: r for k, r in roles.items() if len(r) > 1}
        print(f"  keys in more than one role: {len(crossed)}")
        if crossed:
            failures.append((domain, "partition", {"crossed_keys": len(crossed)}))

    print()
    print("FAILURES" if failures else "ALL OK")
    for failure in failures:
        print(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
