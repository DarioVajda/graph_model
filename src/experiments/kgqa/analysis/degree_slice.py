"""
E4.5 degree-slice analysis: does the graph arm's F1 deficit vs flat concentrate
on questions whose subgraphs contain high-degree (hub) nodes?

Hypothesis (TODO_cwq E4.5 / bias-channel capacity): the soft bias marks all
spd-1 neighbors of a node uniformly, so relation binding at hubs must be
resolved by content alone — errors should grow with hub degree for the graph
arm faster than for flat. The headline runs only logged split means, so this
reruns generation per question from the surviving best checkpoints.

Three subcommands (stats/report are CPU; eval needs a GPU):

  python -m src.experiments.kgqa.analysis.degree_slice stats \
      --resolved <resolved_cfg.json> --out <stats.jsonl>
  python -m src.experiments.kgqa.analysis.degree_slice eval \
      --resolved <resolved_cfg.json> --ckpt <checkpoint_dir> --out <preds.jsonl>
  python -m src.experiments.kgqa.analysis.degree_slice report \
      --stats <stats.jsonl> --preds <preds.jsonl>... [--buckets q4]

Per-question protocol, skip rules, decoding, and scoring are byte-identical to
evaluate.generative_eval / flat_train.flat_generative_eval (greedy, eager path,
GNN-RAG metrics); only the aggregation differs (JSONL rows instead of means).
Graph stats come from the graph-arm .gtds test set; the flat rows mirror
build_question_graphs' skip rules + order (flat_data.py), so rows join on idx.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from ..config import RunConfig
from ..evaluate import (_find_prefix_len, eval_f1, eval_hit1, eval_hit,
                       parse_answer_list)


# --------------------------------------------------------------------------- #
# stats — per-question structural statistics (CPU)
# --------------------------------------------------------------------------- #
def run_stats(cfg, out_path):
    from ..load_data import load_data
    _, _, test_sets = load_data(cfg)
    (ds_name, dataset), = test_sets.items()
    with open(out_path, "w") as f:
        for i, g in enumerate(dataset.graphs):
            pn = g.graph.get("prompt_node")
            gold = [a for a in g.graph.get("gold_answers", []) if a]
            ent = [n for n in g.nodes
                   if n != pn and not g.nodes[n].get("is_rel", False)]
            und = g.to_undirected(as_view=True)
            # entity degree in the Levi graph = # incident relation instances
            ent_deg = [sum(1 for nb in und.neighbors(n) if nb != pn) for n in ent]
            topics = [n for n in und.neighbors(pn)] if pn is not None else []
            topic_deg = [sum(1 for nb in und.neighbors(n) if nb != pn)
                         for n in topics]
            f.write(json.dumps({
                "idx": i,
                "gold": gold,
                "n_nodes": g.number_of_nodes(),
                "n_entities": len(ent),
                "n_rels": g.number_of_nodes() - len(ent) - (pn is not None),
                "max_ent_deg": max(ent_deg, default=0),
                "mean_ent_deg": float(np.mean(ent_deg)) if ent_deg else 0.0,
                "n_topics": len(topics),
                "max_topic_deg": max(topic_deg, default=0),
                "n_gold": len(gold),
            }) + "\n")
    print(f"stats: {ds_name} test, {len(dataset.graphs)} questions -> {out_path}")


# --------------------------------------------------------------------------- #
# eval — per-question generative predictions from a checkpoint (GPU)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_eval_graph(cfg, ckpt, out_path, max_samples=None):
    from transformers import AutoTokenizer
    from src.models.modeling_gtlm_llama import GTLMLlamaForCausalLM
    from src.utils import GraphCollatorV2
    from ..load_data import load_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # generative_eval always runs generation on the dense eager path
    model = GTLMLlamaForCausalLM.from_pretrained(
        ckpt, graph_attn_impl="eager", torch_dtype=cfg.torch_dtype)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    question_end = tokenizer(cfg.question_end_str,
                             add_special_tokens=False)["input_ids"]

    _, _, test_sets = load_data(cfg)
    (ds_name, dataset), = test_sets.items()
    magnetic_m = cfg.magnetic_m if (cfg.magnetic or cfg.magnetic_shared) else 0
    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=magnetic_m, pad_to_block=False)

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    with open(out_path, "w") as f:
        for i in range(n):
            item = dataset[i]
            pn = int(item["prompt_node"])
            ids = list(item["input_ids"][pn])
            cut = _find_prefix_len(ids, question_end)
            gold = [a for a in dataset.graphs[i].graph.get("gold_answers", [])
                    if a]
            if cut is None or not gold:
                continue                       # same skip rule as generative_eval
            gen_item = dict(item)
            gen_item["input_ids"] = [list(x) for x in item["input_ids"]]
            gen_item["input_ids"][pn] = ids[:cut]
            gen_item.pop("labels", None)
            batch = collator([gen_item])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model.generate(
                **batch, max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][batch["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            _write_row(f, i, text, gold, cfg.answer_parse_sep)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    print(f"eval(graph): {ds_name} test -> {out_path}")


@torch.no_grad()
def run_eval_flat(cfg, ckpt, out_path, max_samples=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..flat_data import flat_data_config_key
    from ..flat_train import tokenize_row
    from ..process_dataset import OUTPUT_ROOT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.torch_dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ckpt)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    question_end = tokenizer(cfg.question_end_str,
                             add_special_tokens=False)["input_ids"]

    (ds_name,) = cfg.eval_datasets
    view = cfg.for_dataset(ds_name)
    path = os.path.join(OUTPUT_ROOT, flat_data_config_key(view, ds_name),
                        "test.jsonl")
    rows = [tokenize_row(json.loads(l), tokenizer, question_end, cfg.seq_len)
            for l in open(path)]

    n = len(rows) if max_samples is None else min(max_samples, len(rows))
    with open(out_path, "w") as f:
        for i in range(n):
            row = rows[i]
            gold = [a for a in row["gold_answers"] if a]
            if not gold:
                continue                  # same skip rule as flat_generative_eval
            ids = torch.tensor([row["input_ids"][: row["prefix_len"]]],
                               device=device)
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=cfg.gen_max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][ids.shape[1]:],
                                    skip_special_tokens=True)
            _write_row(f, i, text, gold, cfg.answer_parse_sep)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    print(f"eval(flat): {ds_name} test -> {out_path}")


def _write_row(f, idx, text, gold, sep):
    pred = parse_answer_list(text, sep=sep)
    row = {"idx": idx, "gold": gold, "pred": pred,
           "f1": float(eval_f1(pred, gold)[0]) if pred else 0.0,
           "hits1": float(eval_hit1(pred, gold)) if pred else 0.0,
           "hit_star": float(eval_hit(pred, gold)) if pred else 0.0}
    f.write(json.dumps(row) + "\n")
    f.flush()


# --------------------------------------------------------------------------- #
# report — join preds with stats, slice by structural buckets (CPU)
# --------------------------------------------------------------------------- #
def _load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def run_report(stats_path, pred_paths, slice_keys, n_buckets=4):
    stats = {r["idx"]: r for r in _load_jsonl(stats_path)}
    # arm -> idx -> [f1 per seed]; arm inferred from filename (train|flat_train)
    arms = defaultdict(lambda: defaultdict(list))
    for p in pred_paths:
        arm = "flat" if "flat_train" in os.path.basename(p) else "graph"
        for r in _load_jsonl(p):
            arms[arm][r["idx"]].append(r["f1"])
    common = sorted(set.intersection(*(set(d) for d in arms.values())))
    print(f"questions scored by every run: {len(common)}  "
          f"(runs: { {a: len(set.union(*[set()]) | set(d)) for a, d in arms.items()} })")
    per_q = {i: {a: float(np.mean(arms[a][i])) for a in arms} for i in common}

    for key in slice_keys:
        vals = np.array([stats[i][key] for i in common])
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_buckets + 1)))
        which = np.clip(np.searchsorted(edges, vals, side="right") - 1,
                        0, len(edges) - 2)
        print(f"\n== slice by {key} (quantile buckets) ==")
        print(f"{'bucket':>18} {'n':>5}  " + "  ".join(f"{a:>7}" for a in arms)
              + f"  {'flat-graph':>10}  {'95% CI':>16}")
        for b in range(len(edges) - 1):
            sel = [i for i, w in zip(common, which) if w == b]
            if not sel:
                continue
            lo, hi = edges[b], edges[b + 1]
            means = {a: np.mean([per_q[i][a] for i in sel]) for a in arms}
            diff = [per_q[i]["flat"] - per_q[i]["graph"] for i in sel]
            boot = [np.mean(np.random.choice(diff, len(diff)))
                    for _ in range(2000)]
            ci = np.percentile(boot, [2.5, 97.5])
            print(f"{f'[{lo:g}, {hi:g}]':>18} {len(sel):>5}  "
                  + "  ".join(f"{means[a]:>7.4f}" for a in arms)
                  + f"  {np.mean(diff):>+10.4f}  "
                  + f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stats")
    ps.add_argument("--resolved", required=True)
    ps.add_argument("--out", required=True)

    pe = sub.add_parser("eval")
    pe.add_argument("--resolved", required=True)
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--max-samples", type=int, default=None)

    pr = sub.add_parser("report")
    pr.add_argument("--stats", required=True)
    pr.add_argument("--preds", nargs="+", required=True)
    pr.add_argument("--slice", nargs="+",
                    default=["max_ent_deg", "max_topic_deg", "n_nodes",
                             "n_gold"])
    pr.add_argument("--buckets", type=int, default=4)

    args = p.parse_args()
    if args.cmd == "report":
        np.random.seed(0)
        run_report(args.stats, args.preds, args.slice, args.buckets)
        return

    raw = json.load(open(args.resolved))
    ds = raw.pop("dataset", None)         # CLI alias: sets both dataset roles
    if ds is not None:
        raw["train_datasets"] = raw["eval_datasets"] = (ds,)
    raw = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    cfg = RunConfig(**raw)
    if args.cmd == "stats":
        run_stats(cfg, args.out)
    elif cfg.mode == "train":
        run_eval_graph(cfg, args.ckpt, args.out, args.max_samples)
    elif cfg.mode == "flat_train":
        run_eval_flat(cfg, args.ckpt, args.out, args.max_samples)
    else:
        raise SystemExit(f"unsupported mode {cfg.mode!r}")


if __name__ == "__main__":
    main()
