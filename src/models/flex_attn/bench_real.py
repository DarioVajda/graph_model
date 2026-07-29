"""
Real-input benchmark: an experiment's own cached batches through its training stack.

The rest of this package measures *synthetic* graph batches, which is the right
tool for a scaling study (it can dial L to 70k) but the wrong tool for the
question "how much faster did the paper's own experiments get?". This module
answers that one: it loads the actual cached splits, collates them with the
production ``GraphCollatorV2``, builds the real GTLM model with LoRA exactly as
the experiment's own ``train.py`` does, and times training steps.

Two experiments are wired (``--experiment``), and they sit in opposite regimes:

  * ``tag`` — text-attributed graphs (Cora/PubMed/ogbn-arxiv/Reddit). L ≈ 600–1300,
    bf16, B=1, gradient checkpointing on. This is where flex was expected to pay.
  * ``graphqa`` — algorithmic questions over small graphs. L ≈ 29 (standard
    encoding) to ≈ 150 (incidence), fp32, B=4, no checkpointing. Far below the
    lengths flex targets; ``graphqa``'s own default backend is ``v2-eager`` for
    exactly that reason. Measured here to *establish the regime boundary*, not to
    claim a speedup — see ``--len-bucket-multiple``, without which the production
    512-length ladder pads a 29-token sequence ~17x and the comparison is a
    strawman.

Methods
-------
  * ``eager`` — ``graph_attn_impl="eager"``: the dense ``(B,1,L,L)`` structural
    mask plus a per-layer token-expanded ``(B,H,L,L)`` soft bias. **This is the
    path the paper's results were produced on** (flex did not exist yet).
  * ``flex``  — ``graph_attn_impl="flex"``: the same model and weights, node-level
    bias gathered inside the kernel against a sparse ``BlockMask``.
  * ``flex-nobias`` — the flex kernel and the graph structural mask, but no bias
    modules at all. Sits between ``flex`` and ``sdpa`` and splits the gap: the
    distance to ``sdpa`` is what the graph mask + kernel cost, the distance to
    ``flex`` is what computing the per-layer bias costs.
  * ``sdpa``  — stock ``LlamaForCausalLM`` with ``attn_implementation="sdpa"``
    (PyTorch selects its fused flash kernel for plain causal attention), LoRA
    applied at the same rank, fed the **identical** ``input_ids`` /
    ``attention_mask`` / ``labels``. No graph structure, no bias — the
    "plain LLM at equal sequence length" floor. Not functionally equivalent;
    it answers "what does the graph machinery still cost over a vanilla LLM?".
    Deliberately the most favourable possible baseline: plain causal attention
    is flash-eligible, and block skipping is an optimization GTLM gives up by
    construction. Keep it as the theoretical floor — do not "fix" it.
  * ``sdpa-graphmask`` — the same stock model, but handed GTLM's own dense
    structural mask (causal **relaxed to bidirectional between prefix tokens**,
    plus padding and the K-hop gate) as a 4-D additive mask. Still no bias. An
    arbitrary mask makes flash ineligible, so this is the dense SDPA path. It
    exists because plain causal attends to strictly *fewer* positions than GTLM's
    mask allows, so ``sdpa`` alone understates the work GTLM must do; this arm
    prices the mask *shape* separately from the bias. Compare it against
    ``flex-nobias`` (same mask, same absence of bias, different kernel) to
    isolate the kernel, and against ``eager`` to isolate the bias.

All three methods see byte-identical token tensors: one collator instance is
shared, so bucket padding (needed by flex) is applied to every arm. That makes L
exactly equal across methods at the cost of charging eager/sdpa for padding they
would not otherwise pay; ``--pad-mode batch`` drops the bucketing for a
flex-free comparison.

Protocol
--------
A fixed, strided sample of real train graphs is collated once into a list of
batches (strided, not head-of-split, so the length distribution is
representative). Each method then runs that list ``--passes`` times:

  * **pass 0 is discarded** and reported separately as ``cold_pass_s`` — it
    carries torch.compile / autotune for every distinct (L, N) bucket, which is
    a one-time cost amortized over a real run of thousands of steps.
  * the remaining passes are timed per step with CUDA events, giving the
    steady-state number a real epoch actually pays.

Peak memory is measured over the steady passes only. No optimizer step is
included (the paper's config accumulates 32 steps, so it is ~1/32 of the cost
and identical across methods).

Usage
-----
    python -m src.models.flex_attn.bench_real --experiment tag --arm cora \
        --methods eager flex flex-nobias sdpa --n-batches 24 --passes 3 \
        --out-dir src/models/flex_attn/results_h100_tag

    python -m src.models.flex_attn.bench_real --experiment graphqa \
        --arm standard/node_count --len-bucket-multiple 128 \
        --out-dir src/models/flex_attn/results_h100_tag
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from typing import Optional

import torch

from src.models.flex_attn.bench_isolation import _mb, emit_result

METHODS = ("eager", "flex", "flex-nobias", "sdpa-graphmask", "sdpa")

# Methods that are a stock LLM (no GTLM wrapper, no bias modules).
PLAIN_LLM_METHODS = ("sdpa", "sdpa-graphmask")

# The per-dataset configuration the TAG paper selected (Table 8), mirrored from
# src/experiments/tag_benchmarks/configs/002_paper_tag_repro.jsonc. Benchmarking
# anything else would not answer "how much faster are the paper's runs".
TAG_ARMS = {
    "cora":       dict(max_neighbors=60, text_mapping="target_abstract",
                       text_mapping_param=None, lora_r=64),
    "pubmed":     dict(max_neighbors=30, text_mapping="target_abstract",
                       text_mapping_param=None, lora_r=32),
    "ogbn-arxiv": dict(max_neighbors=60, text_mapping="target_abstract",
                       text_mapping_param=None, lora_r=64),
    "reddit":     dict(max_neighbors=30, text_mapping="truncated_text",
                       text_mapping_param=128, lora_r=32),
}

# GraphQA arms are "<graph_type>/<task>"; every other knob is the recipe from
# configs/003_ablation.jsonc, which is the RunConfig default set.
GRAPHQA_GRAPH_TYPES = ("standard", "incidence")


# ── config + data ─────────────────────────────────────────────────────────────

def _tag_config(arm: str, *, batch_size, gradient_checkpointing, k_hop, dtype):
    from src.experiments.tag_benchmarks.config import RunConfig
    cfg = RunConfig(
        mode="train", dataset=arm, impl="v2-flex", dtype=dtype or "bf16",
        k_hop=k_hop,
        # All three structural biases on — the paper's full model (Table 6 values
        # are the RunConfig defaults).
        spd=True, rrwp=True, magnetic=True,
        lora=True, lora_dropout=0.05,
        batch_size=batch_size if batch_size is not None else 1,
        accumulation_steps=32,
        gradient_checkpointing=(True if gradient_checkpointing is None
                                else gradient_checkpointing),
        **TAG_ARMS[arm],
    )
    return cfg.validate()


def _graphqa_config(arm: str, *, batch_size, gradient_checkpointing, k_hop, dtype):
    from src.experiments.graphqa.config import RunConfig
    graph_type, _, task = arm.partition("/")
    if graph_type not in GRAPHQA_GRAPH_TYPES or not task:
        raise SystemExit(
            f"--arm for graphqa must be '<graph_type>/<task>' with graph_type in "
            f"{GRAPHQA_GRAPH_TYPES}; got {arm!r}.")
    # Defaults below are already the 003_ablation recipe (fp32, B=4, lora_r 16,
    # no gradient checkpointing); only override what the CLI explicitly set.
    cfg = RunConfig(
        mode="train", task=task, graph_type=graph_type,
        impl="v2-flex", dtype=dtype or "fp32", k_hop=k_hop,
        spd=True, rrwp=True, magnetic=True,
        batch_size=batch_size if batch_size is not None else 4,
        gradient_checkpointing=(False if gradient_checkpointing is None
                                else gradient_checkpointing),
    )
    return cfg.validate()


# name -> (config builder, load_data, default arm)
EXPERIMENTS = {
    "tag":     (_tag_config,     "src.experiments.tag_benchmarks.data", "cora"),
    "graphqa": (_graphqa_config, "src.experiments.graphqa.data", "standard/node_count"),
}


def build_config(experiment: str, arm: str, **kw):
    """A ``RunConfig`` for this experiment's arm, matching its paper recipe."""
    return EXPERIMENTS[experiment][0](arm, **kw)


def _load_data(experiment: str, cfg):
    import importlib
    return importlib.import_module(EXPERIMENTS[experiment][1]).load_data(cfg)


def sample_batches(experiment: str, cfg, n_batches: int, batch_size: int, split: str,
                   pad_mode: str, len_bucket_multiple: int,
                   device) -> tuple[list[dict], dict]:
    """Collate ``n_batches`` real batches, strided across ``split``.

    Striding (rather than taking the head of the split) keeps the sampled
    sequence-length distribution representative: the cached splits are in node
    order, which correlates with neighbourhood size.
    """
    from transformers import AutoTokenizer
    from src.utils import GraphCollatorV2
    from src.models.flex_kernel import bucket_len

    train, val, test = _load_data(experiment, cfg)
    dataset = {"train": train, "val": val, "test": test}[split]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # The production ladder steps in 512s, which is right when L is ~1000 and
    # absurd when L is ~30 (GraphQA standard): it would pad 17x and measure the
    # padding, not the kernel. `--len-bucket-multiple` lets the ladder match the
    # regime; it must stay a multiple of the 128 block size.
    if len_bucket_multiple % 128 != 0:
        raise SystemExit(
            f"--len-bucket-multiple must be a multiple of 128 (the flex block "
            f"size); got {len_bucket_multiple}.")
    len_buckets = (None if len_bucket_multiple == 512
                   else (lambda L: bucket_len(L, len_bucket_multiple)))
    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(pad_mode == "bucket"), max_spd=cfg.max_spd,
        len_buckets=len_buckets)

    need = n_batches * batch_size
    stride = max(1, len(dataset) // need)
    idx = [(i * stride) % len(dataset) for i in range(need)]

    batches = []
    for b in range(n_batches):
        items = [dataset[i] for i in idx[b * batch_size:(b + 1) * batch_size]]
        batch = collator(items)
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()})

    lens = [int(b["input_ids"].shape[1]) for b in batches]
    nodes = [int(b["num_nodes"].max()) for b in batches]
    real = [int(b["attention_mask"].sum()) for b in batches]
    # The compiled flex kernel guards on the *bucketed* node dim — the width of
    # the (B, N, N) feature tensors — not on the true node count in `num_nodes`.
    # Counting distinct (L, true N) would overstate the number of compiled
    # shapes several-fold.
    node_dim = [int(b["shortest_path_dists"].shape[1]) for b in batches]
    meta = {
        "split": split, "n_batches": n_batches, "batch_size": batch_size,
        "dataset_size": len(dataset), "stride": stride, "pad_mode": pad_mode,
        "seq_len_mean": sum(lens) / len(lens), "seq_len_min": min(lens),
        "seq_len_max": max(lens), "seq_len_total": sum(lens),
        "real_tokens_total": sum(real),
        "padding_frac": 1.0 - sum(real) / (sum(lens) * batch_size),
        "num_nodes_mean": sum(nodes) / len(nodes), "num_nodes_max": max(nodes),
        "node_dim_buckets": sorted(set(node_dim)),
        "len_buckets": sorted(set(lens)),
        "len_bucket_multiple": len_bucket_multiple,
        # Distinct compiled shapes = distinct (L, bucketed N) pairs.
        "distinct_shapes": len({(l, n) for l, n in zip(lens, node_dim)}),
    }
    return batches, meta


# ── models ────────────────────────────────────────────────────────────────────

def build_model(method: str, cfg, device):
    """The real training-time model for one method (LoRA applied as in train.py)."""
    from transformers import set_seed
    from src.train import select_active_params

    set_seed(cfg.seed)

    if method in PLAIN_LLM_METHODS:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=cfg.torch_dtype(),
            attn_implementation="sdpa")
        active = None                      # no graph_bias modules to unfreeze
    else:
        # graphqa selects its GTLM classes from the backbone named in model_name
        # (llama / bloom / gemma-3); tag_benchmarks is Llama-only.
        if hasattr(cfg, "gtlm_classes"):
            gtlm_config_cls, gtlm_model_cls = cfg.gtlm_classes()
        else:
            from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
            gtlm_config_cls, gtlm_model_cls = GTLMLlamaConfig, GTLMLlamaForCausalLM
        backend = "flex" if method.startswith("flex") else method
        # `flex-nobias` keeps the flex kernel, the graph structural mask and the
        # whole GTLM wrapper, but builds no bias modules. It splits the residual
        # gap to the plain-LLM floor into "attention/mask machinery" (this arm
        # minus sdpa) and "per-layer bias computation" (flex minus this arm) —
        # the decomposition a reviewer asks for on seeing flex still above sdpa.
        bias = {} if method == "flex-nobias" else cfg.bias_params()
        config = gtlm_config_cls.from_pretrained(
            cfg.model_name, **bias,
            k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
            graph_attn_impl=backend,
            **({"flex_compile_mode": cfg.flex_compile_mode} if backend == "flex" else {}),
        )
        model = gtlm_model_cls.from_pretrained(
            cfg.model_name, config=config, graph_attn_impl=backend,
            torch_dtype=cfg.torch_dtype())
        active = ["graph_bias"]

    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model = select_active_params(model, active_params=active, lora=cfg.lora_config())

    if cfg.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


def _graph_mask_4d(batch: dict, cfg, dtype: torch.dtype) -> torch.Tensor:
    """GTLM's own ``(B, 1, L, L)`` additive structural mask for a collated batch.

    Reuses the model's builder rather than reimplementing it, so this baseline
    cannot silently drift from the mask GTLM actually applies.
    """
    from src.models.structural_mask import build_dense_structural_mask

    node_ids = batch["node_ids"]
    L = node_ids.shape[1]
    return build_dense_structural_mask(
        node_ids=node_ids, prompt_node=batch["prompt_node"],
        pad_mask=batch["attention_mask"], k_hop_mask=batch.get("k_hop_mask"),
        k_hop=cfg.k_hop, q_len=L, kv_len=L,
        dtype=dtype, device=node_ids.device,
    )


def _inputs_for(method: str, batch: dict, graph_mask=None) -> dict:
    """The kwargs one method's forward takes from a collated batch."""
    if method == "sdpa":
        # Identical tokens, mask and supervision — the graph tensors are simply
        # not consumed. Equal sequence length by construction.
        return {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
    if method == "sdpa-graphmask":
        # HF passes a 4-D attention_mask through untouched (see
        # LlamaModel._prepare_4d_causal_attention_mask_with_cache_position:
        # "if the input attention_mask is already 4D, do nothing").
        return {"input_ids": batch["input_ids"], "labels": batch["labels"],
                "attention_mask": graph_mask}
    return dict(batch)


def _assert_graph_mask_applied(model, batch, graph_mask) -> float:
    """Fail loudly if HF ignored our 4-D mask.

    A silently-dropped mask would make this arm a duplicate of ``sdpa`` while
    still looking plausible, so verify the mask changes the loss before timing
    anything. Returns the relative loss difference for the record.
    """
    with torch.no_grad():
        plain = model(**_inputs_for("sdpa", batch)).loss.float()
        masked = model(**_inputs_for("sdpa-graphmask", batch, graph_mask)).loss.float()
    rel = float((masked - plain).abs() / plain.abs().clamp_min(1e-6))
    if rel < 1e-4:
        raise RuntimeError(
            f"sdpa-graphmask produced the same loss as plain causal "
            f"(rel diff {rel:.2e}) — the 4-D structural mask was ignored, so this "
            f"arm would silently duplicate `sdpa`. Check HF's mask handling.")
    return rel


# ── one method's run ──────────────────────────────────────────────────────────

def run_method(method: str, cfg, batches: list[dict], passes: int) -> dict:
    """Cold pass (compile) + ``passes`` timed steady-state passes over ``batches``."""
    out = {"method": method, "ok": False, "error": None}
    model = None
    try:
        model = build_model(method, cfg, torch.device("cuda"))

        # The graph-masked plain-LLM arm needs GTLM's dense mask per batch; build
        # it once (it is a pure function of the batch) so the timing measures
        # attention, not mask construction — matching flex, whose BlockMask is
        # also built once per batch, and eager, which builds its mask inside the
        # model but only once per forward.
        masks = [None] * len(batches)
        if method == "sdpa-graphmask":
            dtype = cfg.torch_dtype()
            masks = [_graph_mask_4d(b, cfg, dtype) for b in batches]
            out["mask_check_rel_loss_diff"] = _assert_graph_mask_applied(
                model, batches[0], masks[0])

        def _step(batch, mask=None):
            model.zero_grad(set_to_none=True)
            loss = model(**_inputs_for(method, batch, mask)).loss
            loss.backward()
            return float(loss.detach())

        # ── cold pass: compile / autotune every distinct (L, N) bucket ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch, mask in zip(batches, masks):
            _step(batch, mask)
        torch.cuda.synchronize()
        out["cold_pass_s"] = time.perf_counter() - t0

        # ── steady state ──
        # Deliberately NO empty_cache() here. It returns the allocator's blocks to
        # the driver, so the first timed step to meet each shape pays a cudaMalloc
        # storm — measured at up to 6.6 s against a 103 ms median, which nearly
        # doubled the mean of a ~100 ms arm. It also buys nothing:
        # max_memory_allocated() counts *allocated* bytes, while cached-but-free
        # blocks are *reserved*, so the peak below is unaffected by the pool the
        # cold pass already grew. (Warming just batches[0] was not enough — a
        # later, longer batch still triggered the growth.)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_ms, losses = [], []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall0 = time.perf_counter()
        for _ in range(passes):
            for batch, mask in zip(batches, masks):
                torch.cuda.synchronize()
                start.record()
                losses.append(_step(batch, mask))
                end.record()
                torch.cuda.synchronize()
                step_ms.append(start.elapsed_time(end))
        wall = time.perf_counter() - wall0
        torch.cuda.synchronize()

        t = torch.tensor(step_ms)
        # Trimmed mean (drop the top 2%) alongside the plain mean: the plain mean
        # is the right estimator for epoch wall clock -- epoch time IS the sum of
        # step times, and the spread across the length distribution is real, not
        # noise -- but it is also the statistic a stray stall corrupts. If the two
        # disagree by more than a few percent, suspect an artifact, not physics.
        keep = max(1, int(len(t) * 0.98))
        trimmed = torch.sort(t).values[:keep]
        out.update(
            ok=True,
            step_ms_mean=float(t.mean()), step_ms_std=float(t.std()),
            step_ms_median=float(t.median()),
            step_ms_trimmed_mean=float(trimmed.mean()),
            step_ms_min=float(t.min()), step_ms_max=float(t.max()),
            # What one pass over the sampled batches costs — the quantity that
            # extrapolates to epoch wall clock.
            pass_s=wall / passes,
            n_steps=len(step_ms),
            peak_mem_mb=_mb(torch.cuda.max_memory_allocated()),
            loss_mean=float(sum(losses) / len(losses)),
        )
    except torch.cuda.OutOfMemoryError as e:
        out["error"] = "OOM"
        out["error_detail"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        import traceback
        out["error"] = type(e).__name__
        out["error_detail"] = traceback.format_exc()[-800:]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return out


# ── reporting ─────────────────────────────────────────────────────────────────

def summarize(record: dict) -> str:
    """A markdown table of the arms, with speedups relative to eager."""
    arms = {r["method"]: r for r in record["methods"]}
    base = arms.get("eager")
    base_ms = base.get("step_ms_mean") if base and base.get("ok") else None

    d = record["data"]
    lines = [
        f"### {record.get('experiment', 'tag')} / {record['dataset']} — "
        f"{d['n_batches']} real batches (B={d['batch_size']}, {d['split']} split)",
        "",
        f"L: mean {d['seq_len_mean']:.0f}, "
        f"range {d['seq_len_min']}–{d['seq_len_max']}, "
        f"{d['distinct_shapes']} distinct (L,N) shapes "
        f"(ladder step {d.get('len_bucket_multiple', 512)}); "
        f"N max {d['num_nodes_max']}; "
        f"padding {d['padding_frac'] * 100:.0f}%; "
        f"dtype={record['config']['dtype']}; "
        f"k_hop={record['config']['k_hop']}; "
        f"grad-ckpt={record['config']['gradient_checkpointing']}",
        "",
        "| method | step ms | vs eager | peak GB | cold pass s | epoch est. |",
        "|---|---|---|---|---|---|",
    ]
    steps_per_epoch = record.get("steps_per_epoch")
    for m in METHODS:
        r = arms.get(m)
        if r is None:
            continue
        if not r.get("ok"):
            lines.append(f"| `{m}` | **{r.get('error')}** | — | — | — | — |")
            continue
        speed = f"{base_ms / r['step_ms_mean']:.2f}×" if base_ms else "—"
        epoch = ""
        if steps_per_epoch:
            epoch = f"{r['step_ms_mean'] * steps_per_epoch / 3.6e6:.2f} h"
        lines.append(
            f"| `{m}` | {r['step_ms_mean']:.1f} ± {r['step_ms_std']:.1f} | {speed} | "
            f"{r['peak_mem_mb'] / 1024:.2f} | {r['cold_pass_s']:.0f} | {epoch} |")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--experiment", default="tag", choices=tuple(EXPERIMENTS))
    p.add_argument("--arm", default=None,
                   help="tag: a dataset (cora|pubmed|ogbn-arxiv|reddit). "
                        "graphqa: '<graph_type>/<task>', e.g. standard/node_count. "
                        "Defaults to the experiment's first arm.")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--n-batches", type=int, default=24,
                   help="real batches sampled (strided) from the split")
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: the experiment's paper setting (tag 1, graphqa 4)")
    p.add_argument("--dtype", default=None, choices=("bf16", "fp32"),
                   help="default: the experiment's paper setting (tag bf16, graphqa fp32)")
    p.add_argument("--len-bucket-multiple", type=int, default=512,
                   help="length-ladder step, a multiple of 128. 512 is the production "
                        "ladder (right for TAG); use 128 for GraphQA, whose sequences "
                        "are ~30-150 tokens and would otherwise be padding-dominated.")
    p.add_argument("--passes", type=int, default=3,
                   help="timed passes over the sample, after one discarded cold pass")
    p.add_argument("--k-hop", type=int, default=0,
                   help="0 is the paper setting; >0 is where flex's block sparsity pays")
    p.add_argument("--pad-mode", default="bucket", choices=("bucket", "batch"),
                   help="'bucket' pads L/N to flex's buckets for EVERY arm (equal L); "
                        "'batch' pads to the batch max (flex cannot run)")
    p.add_argument("--gradient-checkpointing", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="default: the experiment's paper setting (tag on, graphqa off)")
    p.add_argument("--out-dir", default=None,
                   help="append the record to {out-dir}/{experiment}.jsonl and "
                        "rewrite {experiment}.md")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    assert torch.cuda.is_available(), "this benchmark needs a GPU (sbatch it)"
    if a.pad_mode == "batch" and any(m.startswith("flex") for m in a.methods):
        raise SystemExit("--pad-mode batch cannot run flex (unaligned L); drop flex.")

    dev = torch.device("cuda")
    arm = a.arm or EXPERIMENTS[a.experiment][2]
    # The config is backend-independent for data loading; the backend is chosen
    # per method in build_model.
    cfg = build_config(a.experiment, arm, batch_size=a.batch_size,
                       gradient_checkpointing=a.gradient_checkpointing,
                       k_hop=a.k_hop, dtype=a.dtype)
    batch_size = cfg.batch_size
    batches, data_meta = sample_batches(a.experiment, cfg, a.n_batches, batch_size,
                                        a.split, a.pad_mode,
                                        a.len_bucket_multiple, dev)
    print(f"[bench_real] {a.experiment}/{arm}: {json.dumps(data_meta, indent=2)}",
          flush=True)

    record = {
        "kind": "real_inputs", "experiment": a.experiment, "dataset": arm,
        "gpu": torch.cuda.get_device_name(0),
        "config": {
            "model_name": cfg.model_name, "dtype": cfg.dtype, "k_hop": cfg.k_hop,
            "lora_r": cfg.lora_r,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "flex_compile_mode": cfg.flex_compile_mode,
            "spd": cfg.spd, "rrwp": cfg.rrwp, "magnetic": cfg.magnetic,
            "passes": a.passes,
            # Experiment-specific provenance, so a record identifies its own arm.
            **({"max_neighbors": cfg.max_neighbors,
                "text_mapping": cfg.mapping_name()} if a.experiment == "tag" else
               {"task": cfg.task, "graph_type": cfg.graph_type,
                "question_node": cfg.question_node}),
        },
        "data": data_meta,
        "methods": [],
    }
    # Steps in one real epoch of the paper's run, for the wall-clock extrapolation.
    record["steps_per_epoch"] = data_meta["dataset_size"] // batch_size

    for method in a.methods:
        print(f"\n[bench_real] ── {method} ──", flush=True)
        res = run_method(method, cfg, batches, a.passes)
        print(json.dumps(res, indent=2)[:1200], flush=True)
        record["methods"].append(res)

    print("\n" + summarize(record), flush=True)
    emit_result(record)

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        stem = a.experiment
        with open(os.path.join(a.out_dir, f"{stem}.jsonl"), "a") as fh:
            fh.write(json.dumps(record) + "\n")
        with open(os.path.join(a.out_dir, f"{stem}.jsonl")) as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
        with open(os.path.join(a.out_dir, f"{stem}.md"), "w") as fh:
            fh.write(f"# {stem} — real inputs, eager vs flex vs plain-LLM sdpa\n\n")
            fh.write(f"GPU: {record['gpu']}\n\n")
            fh.write("\n\n".join(summarize(r) for r in recs) + "\n")

    return 0 if all(r["ok"] for r in record["methods"]) else 1


if __name__ == "__main__":
    sys.exit(main())
