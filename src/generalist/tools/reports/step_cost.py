"""What a training micro-batch costs on the card, per source and padded shape.

`GRAPH_GENERALIST.md` §3 owes D5's caps (``max_nodes``, ``max_tokens``, a
sampler per oversized task) "against measured s/it on the largest components":
expressiveness, CWQ and TAG. `shapes.py` says where their rows land on the
ladder; this says what each of those shapes costs to train on.

    GPU=1 GPU_CONSTRAINT='GPU_BRD:H100' src/generalist/tools/launch/run_py.sh \\
        src/generalist/tools/reports/step_cost.py \\
        --config src/generalist/configs/probes/014_graph_smoke.jsonc \\
        --data-config src/generalist/configs/probes/015_graph_full_build.jsonc

The model, LoRA and collator are built as `wiring.build_run` builds them from
``--config``; the rows come from ``--data-config``'s build, so the smoke's
training setup is measured on the full data. For each source, every padded
``(N, L)`` shape holding at least ``--min-share`` of its train rows gets one
micro-batch of real rows of that shape, as many as the sampler's row cap allows
under ``micro_batch_tokens`` (and ``micro_batch_node_pairs``). Each runs the
trainer's forward, its supervised-position cross-entropy and a backward:

* **first** — the first call's wall time: compile, autotune and one step. A
  run pays it once per shape, per rank, unless the inductor cache is primed.
* **step** — the median of ``--repeats`` further calls, the steady state.
* **peak** — the card's peak allocation for the shape.

The per-source line weights each shape's ms per row by its share of the rows,
which is the steady-state cost of one of that source's examples. The optimizer
step is not timed: it is per step, not per row, and the same for every source.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict

DEFAULT_TASKS = ("kgqa/cwq", "kgqa/webqsp", "expressiveness/hard", "tag/ogbn-arxiv",
                 "tag/cora", "probes/text_path", "graphqa/node_count")


def main() -> int:
    import torch
    import torch.nn.functional as F

    from src.experiments.expressiveness.training.dispatch import (
        build_collator, build_model, select_active_params,
    )
    from src.generalist import wiring
    from src.generalist.config import RunConfig, load_config_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="the model / training setup")
    parser.add_argument("--data-config", required=True, help="whose build to read")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-share", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--micro-batch-tokens", type=int, default=0,
                        help="override the config's padded budget")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config)).validate()
    data_config = RunConfig(**load_config_file(args.data_config)).validate()
    budget = args.micro_batch_tokens or config.micro_batch_tokens
    pair_budget = config.micro_batch_node_pairs
    if not budget:
        print("no micro_batch_tokens in the config and none given")
        return 1

    device = torch.device("cuda")
    model, tokenizer = build_model(
        config.impl, config.model_name, config.model_bias_config(),
        config.k_hop, config.k_hop_directed, device, config.flex_compile_mode)
    model.config.flex_cache_size_limit = max(
        getattr(model.config, "flex_cache_size_limit", 0), wiring.FLEX_CACHE_SIZE_LIMIT)
    model = select_active_params(model, active_params=list(wiring.ACTIVE_PARAMS),
                                 lora=config.lora_config())
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id)
    collator = build_collator(config.impl, tokenizer, pad_token_id, config.k_hop,
                              config.k_hop_directed, magnetic_m=config.magnetic_m)
    shape_fn = wiring.collator_shape_fn(collator)
    if shape_fn is None:
        print(f"impl {config.impl}: the collator has no ladder, so no shapes to measure")
        return 1

    def row_cap(shape):
        nodes, tokens = shape
        cap = budget // tokens
        if pair_budget:
            cap = min(cap, pair_budget // (nodes * nodes))
        return max(1, int(cap))

    def step(batch):
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        labels = inputs.pop("labels")
        logits = model(**inputs).logits
        shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
        mask = shift_labels.ne(-100).reshape(-1)
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1])[mask].float(),
            shift_labels.reshape(-1)[mask])
        loss.backward()
        model.zero_grad(set_to_none=True)

    def timed(batch):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(batch)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f"model {config.model_name} impl {config.impl}; budget {budget} padded "
          f"tokens" + (f", {pair_budget} node pairs" if pair_budget else "")
          + f"; data {args.data_config}; {torch.cuda.get_device_name()}")
    print(f"\n{'task':<22}{'N':>6}{'L':>7}{'B':>4}{'share':>7}{'first s':>9}"
          f"{'step ms':>9}{'ms/row':>8}{'raw tok/s':>11}{'peak GB':>9}")
    results, summary = [], {}
    for task in args.tasks.split(","):
        try:
            source = wiring.load_source(data_config, task, args.split)
        except Exception as exc:                          # noqa: BLE001 - reported
            print(f"{task:<22}not loadable: {type(exc).__name__}: {exc}")
            continue
        nodes, tokens = source.lengths()
        by_shape = defaultdict(list)
        for i, (n, t) in enumerate(zip(nodes, tokens)):
            by_shape[tuple(int(v) for v in shape_fn(int(n), int(t)))].append(i)
        total = len(nodes)
        weighted, covered = 0.0, 0.0
        for shape in sorted(by_shape, key=lambda s: (s[1], s[0])):
            rows = by_shape[shape]
            share = len(rows) / total
            if share < args.min_share:
                continue
            b = min(row_cap(shape), len(rows))
            items = [dict(source[i]) for i in rows[:b]]
            batch = collator(items)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                first = timed(batch)
                times = [timed(batch) for _ in range(args.repeats)]
            except torch.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"{task:<22}{shape[0]:>6}{shape[1]:>7}{b:>4}{share:>7.3f}  OOM")
                results.append({"task": task, "nodes": shape[0], "tokens": shape[1],
                                "rows": b, "share": share, "oom": True})
                continue
            ms = 1000 * statistics.median(times)
            peak = torch.cuda.max_memory_allocated() / 2**30
            raw = sum(int(tokens[i]) for i in rows[:b])
            print(f"{task:<22}{shape[0]:>6}{shape[1]:>7}{b:>4}{share:>7.3f}{first:>9.1f}"
                  f"{ms:>9.1f}{ms / b:>8.1f}{raw / (ms / 1000):>11.0f}{peak:>9.1f}")
            results.append({"task": task, "nodes": shape[0], "tokens": shape[1],
                            "rows": b, "share": share, "first_s": first, "step_ms": ms,
                            "ms_per_row": ms / b, "raw_tokens": raw, "peak_gb": peak})
            weighted += share * ms / b
            covered += share
        if covered:
            summary[task] = {"ms_per_row": weighted / covered, "covered": covered}
            print(f"{task:<22}-> {weighted / covered:.1f} ms/row over "
                  f"{100 * covered:.1f}% of its rows\n")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"config": args.config, "data_config": args.data_config,
                       "budget": budget, "pair_budget": pair_budget,
                       "device": torch.cuda.get_device_name(),
                       "shapes": results, "per_task": summary}, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
