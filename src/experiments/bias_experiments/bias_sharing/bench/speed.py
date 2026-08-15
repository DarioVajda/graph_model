"""
Step-time benchmark for the magnetic bias: sharing granularity, and the plain-LLM floor.

Answers two questions §4's sweeps could not:

1. **How does the `G` saving scale with graph size?** All three sweeps sit at or
   below 512 nodes, and the bias is ``(B, H, N, N)`` in nodes — so they priced the
   knob in the regime where the shared object is smallest. ``--source synth``
   runs WebQSP's token profile at N ∈ {512, 1024, 2048, 4096} (see `synth.py`).
2. **What does the graph machinery cost against a vanilla LLM at the same
   sequence length?** The ``llm`` arm is a stock ``LlamaForCausalLM`` handed the
   *identical* ``input_ids`` / ``attention_mask`` / ``labels`` tensors, with the
   same LoRA rank, dtype and gradient checkpointing. Backend: `sdpa` where GTLM
   uses flex, `eager` where GTLM uses eager (GraphQA).

Recipes are **not transcribed**. Each source reads the resolved RunConfig that
the corresponding sweep actually ran (`results/<sweep>/resolved/*.json`) and
overrides only `magnetic_groups`, so a benchmark arm cannot silently drift from
the training arm it claims to price.

    python3 -m src.experiments.bias_experiments.bias_sharing.bench.speed --source synth
    python3 -m src.experiments.bias_experiments.bias_sharing.bench.speed --source webqsp graphqa context

Measurement protocol
--------------------
* **Compilation is excluded by construction.** Flex compiles and autotunes one
  kernel per distinct ``(L, N)`` shape; that cost lands in warm-up passes whose
  wall time is recorded separately (`warmup_s`) and never enters the statistics.
  ``--warmup-passes`` defaults to 2 full passes over the batch list, so every
  shape is compiled *and* re-executed before timing starts.
* **Contamination is checked, not assumed.** Each arm reports
  `first_over_median` (first timed step ÷ median). A leaked compile shows up
  there as a large ratio; the runner warns above 1.5.
* Steps are timed with CUDA events around a synchronized region; the reported
  step is one ``forward + backward`` micro-batch. A HuggingFace Trainer "it" is
  ``accumulation_steps`` of these plus one optimizer step, so §4's `s/it` numbers
  are ~4× (WebQSP) or ~8× (context) a step here.
* No ``empty_cache()`` between arms' timed regions — returning blocks to the
  driver makes the next allocation a cudaMalloc storm, which is measured latency
  that training would never pay.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import shlex
import time
from typing import Optional

import torch

from .synth import SynthSpec, build_batch, verify_against_webqsp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "results", "bench")

ARMS = ("g0", "g1", "g2", "g4", "g8", "g16", "nobias", "llm", "llm_causal")
SYNTH_NODES = (512, 1024, 2048, 4096)


# ── sources ───────────────────────────────────────────────────────────────────
#
# Each entry names the sweep whose resolved config defines the recipe, the module
# holding that experiment's RunConfig, and the attention backend the plain-LLM
# floor should use (matching what GTLM itself runs on that experiment).

SOURCES = {
    "synth":   dict(sweep="002_webqsp_g_sweep",    llm_attn="sdpa"),
    "webqsp":  dict(sweep="002_webqsp_g_sweep",    llm_attn="sdpa"),
    "graphqa": dict(sweep="001_graphqa_g_sweep",   llm_attn="eager"),
    "context": dict(sweep="003_context4k_g_sweep", llm_attn="sdpa"),
}

# Flags the sweep runner adds for bookkeeping; they steer output, not the model.
_RUNNER_FLAGS = ("--runs-jsonl", "--run-name", "--sweep-id")


def sweep_argv(sweep: str) -> tuple[str, list[str]]:
    """``(experiment_module, argv)`` from the command the sweep actually ran.

    Not the resolved JSON: those keys are *CLI* names (``--dataset`` is an alias
    that expands to train/eval dataset tuples), so feeding them to ``RunConfig``
    directly fails or, worse, silently maps wrong. Replaying the emitted argv
    through the experiment's own ``build_parser`` / ``config_from_args`` is the
    same chain `tests/experiments/test_magnetic_groups_cli.py` pins, so a config
    built here cannot differ from the one that produced §4's numbers.
    """
    jobs = os.path.join(RESULTS, sweep, "jobs")
    script = os.path.join(jobs, sorted(os.listdir(jobs))[0])
    with open(script) as f:
        line = next(l for l in f if l.lstrip().startswith(("python ", "python3 ")))

    parts = shlex.split(line)
    module = parts[parts.index("-m") + 1]
    argv = parts[parts.index("-m") + 2:]

    cleaned, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in _RUNNER_FLAGS:
            skip = True
            continue
        cleaned.append(token)
    return module, cleaned


def load_run_config(source: str, magnetic_groups: int = 0):
    """The RunConfig the sweep ran, with `--magnetic-groups` overridden."""
    module, argv = sweep_argv(SOURCES[source]["sweep"])
    if "--magnetic-groups" in argv:
        argv[argv.index("--magnetic-groups") + 1] = str(magnetic_groups)
    else:
        argv += ["--magnetic-groups", str(magnetic_groups)]

    main_mod = importlib.import_module(f"{module}.__main__")
    args = main_mod.build_parser().parse_args(argv)
    cfg = main_mod.config_from_args(args)
    if cfg.magnetic_groups != magnetic_groups:      # the exact bug §5 records
        raise RuntimeError(
            f"--magnetic-groups {magnetic_groups} did not reach {module}'s RunConfig "
            f"(got {cfg.magnetic_groups}).")
    return cfg


def _backend(cfg) -> str:
    """'eager' | 'flex' for this config, across the three experiments' spellings."""
    if hasattr(cfg, "backend"):
        return cfg.backend()                       # graphqa: derived from `impl`
    return cfg.graph_attn_impl                     # kgqa / context


def _dtype(cfg) -> torch.dtype:
    d = getattr(cfg, "torch_dtype", None)
    return d() if callable(d) else d


# ── models ────────────────────────────────────────────────────────────────────

def build_gtlm(cfg, device, flex_compile_mode: Optional[str] = None,
               bias: bool = True):
    """The experiment's own GTLM model, built the way its train.py builds it.

    ``flex_compile_mode`` overrides the recipe's choice. The autotuning default
    ("max-autotune-no-cudagraphs") costs over an hour of compile at
    ``(L=15360, N=4096)``; passing "default" prices whether that buys anything at
    run time. It is safe to vary in isolation at ``k_hop=0``: `flex_block_size`
    resolves to 128 for both modes there, so the block mask, the bucket ladder and
    every tensor shape are unchanged and only kernel selection differs.
    """
    from transformers import set_seed
    from src.train import select_active_params

    set_seed(getattr(cfg, "seed", 0))
    backend = _backend(cfg)

    if hasattr(cfg, "gtlm_classes"):               # graphqa dispatches on backbone
        config_cls, model_cls = cfg.gtlm_classes()
    else:
        from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
        config_cls, model_cls = GTLMLlamaConfig, GTLMLlamaForCausalLM

    # Compile mode is per-experiment and it is NOT cosmetic: context pins
    # "default" while the model's own default is "max-autotune-no-cudagraphs",
    # which compiles far longer and can run faster. Mirror each experiment's
    # choice, and pass nothing where the experiment passes nothing (kgqa), so the
    # model default applies exactly as it did in the sweep.
    flex_kwargs = {}
    if backend == "flex":
        flex_kwargs["flex_cache_size_limit"] = 128
        for attr in ("flex_compile_mode", "compile_mode"):
            if hasattr(cfg, attr):
                flex_kwargs["flex_compile_mode"] = getattr(cfg, attr)
                break
        if flex_compile_mode is not None:
            flex_kwargs["flex_compile_mode"] = flex_compile_mode

    # `bias=False` (the `nobias` arm): the full GTLM wrapper, the graph structural
    # mask and the flex kernel, but no bias modules at all — so no per-layer SPD
    # compute and no score_mod gather. It splits the distance from the plain-LLM
    # floor into "mask shape + kernel" (llm -> nobias) and "everything the bias
    # costs" (nobias -> gN), which no other pair of arms separates.
    bias_params = cfg.bias_params() if bias else {}

    config = config_cls.from_pretrained(
        cfg.model_name, **bias_params,
        k_hop=cfg.k_hop, k_hop_directed=getattr(cfg, "k_hop_directed", False),
        graph_attn_impl=backend, **flex_kwargs,
    )
    model = model_cls.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=backend, torch_dtype=_dtype(cfg))
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return select_active_params(model, active_params=["graph_bias"], lora=cfg.lora_config())


def build_llm(cfg, device, attn_implementation: str):
    """Stock ``LlamaForCausalLM`` — same backbone, LoRA, dtype; no graph anything.

    Deliberately the most favourable baseline available: plain causal attention is
    flash-eligible under sdpa, while GTLM's bidirectional-prefix mask is not, and
    block skipping is an optimization GTLM gives up by construction. Read it as a
    floor, not as an equivalent model.
    """
    from transformers import AutoModelForCausalLM, set_seed
    from src.train import select_active_params

    set_seed(getattr(cfg, "seed", 0))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=_dtype(cfg), attn_implementation=attn_implementation)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return select_active_params(model, active_params=None, lora=cfg.lora_config())


def _enable_checkpointing(model, cfg):
    if getattr(cfg, "gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


# ── batches ───────────────────────────────────────────────────────────────────

LLM_KEYS = ("input_ids", "attention_mask", "labels")


def synth_batches(cfg, n_nodes: int, batch_size: int, n_batches: int, device):
    """``n_batches`` synthetic batches at one node count, one per seed."""
    batches, metas = [], []
    for i in range(n_batches):
        spec = SynthSpec(
            n_nodes=n_nodes, batch_size=batch_size,
            magnetic_m=cfg.magnetic_m, max_spd=cfg.max_spd, k_hop=cfg.k_hop, seed=i,
        )
        batch, meta = build_batch(spec, device, dtype=_dtype(cfg),
                                  pad_to_block=(_backend(cfg) == "flex"))
        batches.append(batch)
        metas.append(meta)
    return batches, metas


def real_batches(source: str, cfg, batch_size: int, n_batches: int, device):
    """Real cached training batches, collated exactly as the experiment does."""
    from transformers import AutoTokenizer
    from src.utils import GraphCollatorV2

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if source == "webqsp":
        from src.experiments.kgqa.load_data import load_data
        dataset = load_data(cfg)[0]
        collator = GraphCollatorV2(
            tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
            magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
            pad_to_block=(_backend(cfg) == "flex"),
            node_position_mode=cfg.node_position_mode, max_spd=cfg.max_spd)
    elif source == "graphqa":
        from src.experiments.graphqa.data import load_data
        dataset = load_data(cfg)[0]
        collator = GraphCollatorV2(
            tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
            magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
            pad_to_block=(_backend(cfg) == "flex"), max_spd=cfg.max_spd)
    elif source == "context":
        from src.experiments.context.model import build_collator
        from src.experiments.context.process_dataset import load_split
        dataset = load_split(cfg, "train")
        collator = build_collator(cfg, tokenizer)
    else:
        raise ValueError(f"{source!r} has no real split")

    # Stride rather than take the head: cached splits are in construction order,
    # which correlates with graph size, so the head is not length-representative.
    need = n_batches * batch_size
    stride = max(1, len(dataset) // need)
    idx = [(i * stride) % len(dataset) for i in range(need)]

    batches, metas = [], []
    for b in range(n_batches):
        items = [dataset[i] for i in idx[b * batch_size:(b + 1) * batch_size]]
        batch = collator(items)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batches.append(batch)
        metas.append({
            "seq_len": int(batch["input_ids"].shape[1]),
            "real_tokens": int(batch["attention_mask"].sum().item()),
            "n_nodes": int(batch["num_nodes"].max().item()),
            "node_slots": (int(batch["shortest_path_dists"].shape[1])
                           if "shortest_path_dists" in batch else None),
            "batch_size": batch_size,
        })
    return batches, metas


# ── timing ────────────────────────────────────────────────────────────────────

def time_arm(model, batches, plain_llm: bool, warmup_passes: int, passes: int,
             drop_attention_mask: bool = False) -> dict:
    """Warm up (compiling every shape), then time ``passes`` passes with CUDA events.

    ``drop_attention_mask`` is the `llm_causal` arm. Passing an ``attention_mask``
    that contains zeros makes transformers build an explicit 4D float mask, and
    sdpa given an explicit mask cannot take the ``is_causal`` fast path — it
    computes the full square instead of the triangle. Every batch here is padded
    (61% of WebQSP's tokens are padding), so the padded `llm` arm is a floor that
    has been handicapped by exactly the thing flex's BlockMask skips. Dropping the
    mask restores the fast path and gives the *best case* a plain LLM can reach at
    this sequence length. The loss it computes is wrong — it attends to padding —
    which is fine, because only the kernel's speed is being read off.
    """
    def inputs(batch):
        if not plain_llm:
            return batch
        keys = [k for k in LLM_KEYS if not (drop_attention_mask and k == "attention_mask")]
        return {k: batch[k] for k in keys}

    def step(batch):
        model.zero_grad(set_to_none=True)
        loss = model(**inputs(batch)).loss
        loss.backward()
        return float(loss.detach())

    # ── warm-up: every distinct (L, N) shape compiles and autotunes here ──
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(warmup_passes):
        for batch in batches:
            step(batch)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    step_ms, step_len, losses = [], [], []
    for _ in range(passes):
        for batch in batches:
            torch.cuda.synchronize()
            start.record()
            losses.append(step(batch))
            end.record()
            torch.cuda.synchronize()
            step_ms.append(start.elapsed_time(end))
            step_len.append(int(batch["input_ids"].shape[1]))

    t = torch.tensor(step_ms)
    median = float(t.median())
    # Real batches do NOT share a sequence length (context spans 2048..4096), so a
    # median pooled over mixed shapes lands on whichever shape happens to sit in
    # the middle and can jump between arms — it once made a 48-bias-compute arm
    # look faster than a 24-compute one. Keep the per-step series and a per-shape
    # median; report.py compares arms shape by shape.
    by_len: dict[int, list[float]] = {}
    for ms, L in zip(step_ms, step_len):
        by_len.setdefault(L, []).append(ms)
    per_shape = {str(L): {"median": float(torch.tensor(v).median()),
                          "mean": float(torch.tensor(v).mean()),
                          "n": len(v)}
                 for L, v in sorted(by_len.items())}

    return {
        "ok": True,
        "warmup_s": warmup_s,
        "step_ms": [round(x, 3) for x in step_ms],
        "step_seq_len": step_len,
        "per_shape": per_shape,
        "step_ms_mean": float(t.mean()),
        "step_ms_median": median,
        "step_ms_std": float(t.std()) if t.numel() > 1 else 0.0,
        "step_ms_min": float(t.min()),
        "step_ms_max": float(t.max()),
        # >1.5 means a compile or an allocator storm leaked past warm-up.
        "first_over_median": float(t[0]) / median if median else None,
        "n_steps": int(t.numel()),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "loss_mean": sum(losses) / len(losses),
    }


def compile_label(flex_compile_mode: Optional[str]) -> str:
    """Run-level grouping key for the compile-mode comparison.

    It must be set for EVERY arm, including `llm`, which is a stock model with no
    `flex_compile_mode` on its config at all. Deriving the label from the model
    would give `llm` a null key in both runs, so the two runs' `llm` rows would
    collide and one would silently overwrite the other — which is exactly what
    happened before this existed.
    """
    return flex_compile_mode or "recipe"


def run_point(source: str, arm: str, *, n_nodes: Optional[int], batch_size: Optional[int],
              n_batches: int, warmup_passes: int, passes: int,
              flex_compile_mode: Optional[str] = None) -> dict:
    """One (source, arm, node-count) cell, model built and torn down in isolation."""
    device = torch.device("cuda")
    LLM_ARMS = ("llm", "llm_causal")
    magnetic_groups = 0 if arm in ("g0", "nobias") + LLM_ARMS else int(arm[1:])
    cfg = load_run_config(source, magnetic_groups)
    bs = batch_size if batch_size is not None else cfg.batch_size

    record = {
        "source": source, "arm": arm, "magnetic_groups": magnetic_groups,
        "n_nodes_target": n_nodes, "batch_size": bs,
        "backend": SOURCES[source]["llm_attn"] if arm in LLM_ARMS else _backend(cfg),
        "dtype": str(_dtype(cfg)), "lora_r": cfg.lora_r,
        "gradient_checkpointing": bool(getattr(cfg, "gradient_checkpointing", False)),
        "sweep_recipe": SOURCES[source]["sweep"], "ok": False, "error": None,
        "compile_label": compile_label(flex_compile_mode),
    }
    model = None
    try:
        if source == "synth":
            batches, metas = synth_batches(cfg, n_nodes, bs, n_batches, device)
        else:
            batches, metas = real_batches(source, cfg, bs, n_batches, device)
        record["shape"] = {
            "seq_len_mean": sum(m["seq_len"] for m in metas) / len(metas),
            "seq_len_max": max(m["seq_len"] for m in metas),
            "real_tokens_mean": sum(m["real_tokens"] for m in metas) / len(metas),
            "node_slots_max": max(m["node_slots"] or 0 for m in metas),
            "distinct_seq_lens": sorted({m["seq_len"] for m in metas}),
        }
        record["shape"]["padding_frac"] = 1.0 - (
            record["shape"]["real_tokens_mean"] / record["shape"]["seq_len_mean"] / bs)

        if arm in LLM_ARMS:
            model = build_llm(cfg, device, SOURCES[source]["llm_attn"])
        else:
            model = build_gtlm(cfg, device, flex_compile_mode, bias=(arm != "nobias"))
        model = _enable_checkpointing(model, cfg)
        record["flex_compile_mode"] = getattr(model.config, "flex_compile_mode", None)

        record.update(time_arm(model, batches, plain_llm=(arm in LLM_ARMS),
                               warmup_passes=warmup_passes, passes=passes,
                               drop_attention_mask=(arm == "llm_causal")))
    except torch.cuda.OutOfMemoryError as e:
        record["error"] = "OOM"
        record["error_detail"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        import traceback
        record["error"] = type(e).__name__
        record["error_detail"] = traceback.format_exc()[-800:]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", default=["synth"], choices=sorted(SOURCES))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--nodes", nargs="+", type=int, default=list(SYNTH_NODES),
                   help="synth only: node counts to sweep")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override the recipe's batch size (synth defaults to 1)")
    p.add_argument("--n-batches", type=int, default=4,
                   help="distinct batches per cell; each is timed --passes times")
    p.add_argument("--warmup-passes", type=int, default=2)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--flex-compile-mode", default=None,
                   help="override the recipe's torch.compile mode for the flex "
                        "kernel, e.g. 'default' to skip autotuning. Recorded on "
                        "every row so the two modes can be compared directly.")
    p.add_argument("--out", default=os.path.join(DEFAULT_OUT, "speed.jsonl"))
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # kgqa/train.py pins this before anything compiles: dynamo guards compiled
    # flex kernels on the process-global thread count, so a later flip doubles the
    # recompile count. There is no DataLoader here to flip it, but matching the
    # training environment costs nothing and removes a way for these timings to
    # differ from the ones they are compared against.
    torch.set_num_threads(1)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    # Record the synthetic-vs-WebQSP token fidelity once, alongside the timings,
    # so the "matches WebQSP" claim travels with the numbers it justifies.
    if "synth" in args.source:
        fidelity = verify_against_webqsp(SynthSpec(n_nodes=max(args.nodes)), n_graphs=4)
        with open(args.out, "a") as f:
            f.write(json.dumps({"kind": "fidelity", **fidelity}) + "\n")
        print("token fidelity (prefix-node tokens): "
              f"webqsp mean {fidelity['webqsp']['mean']:.3f} sd {fidelity['webqsp']['std']:.3f} "
              f"| synth mean {fidelity['synthetic']['mean']:.3f} sd {fidelity['synthetic']['std']:.3f}")

    for source in args.source:
        node_points = args.nodes if source == "synth" else [None]
        for n_nodes in node_points:
            for arm in args.arms:
                bs = args.batch_size if args.batch_size is not None else (
                    1 if source == "synth" else None)
                rec = run_point(source, arm, n_nodes=n_nodes, batch_size=bs,
                                n_batches=args.n_batches,
                                warmup_passes=args.warmup_passes, passes=args.passes,
                                flex_compile_mode=args.flex_compile_mode)
                rec["kind"] = "timing"
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")

                tag = f"{source}" + (f"/N={n_nodes}" if n_nodes else "") + f"/{arm}"
                if rec["ok"]:
                    warn = ""
                    if rec["first_over_median"] and rec["first_over_median"] > 1.5:
                        warn = f"  ⚠ first/median={rec['first_over_median']:.2f} (compile leak?)"
                    print(f"  {tag:28} L={rec['shape']['seq_len_mean']:.0f} "
                          f"step {rec['step_ms_median']:8.1f} ms  "
                          f"peak {rec['peak_mem_mb']:7.0f} MB  "
                          f"warmup {rec['warmup_s']:5.1f} s{warn}")
                else:
                    print(f"  {tag:28} FAILED: {rec['error']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
