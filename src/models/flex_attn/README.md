# FlexAttention for GTLM — benchmarks & optimization log

Does `torch` FlexAttention beat the dense graph-attention path on compute and
memory — especially with sparse K-hop masks — while preserving the SPD and
magnetic-Laplacian biases? This suite answers that with GPU measurements, and
this README doubles as the log of the optimization rounds built on top of it.

**Results directories are per-GPU** — never compare numbers across them:
`results_a100/` is the archived A100_80GB sweep behind the original study;
`results_h100*/` hold the H100 PCIe runs behind everything below (~1.3× faster
on the flash floor than the A100, so mixed-dir ratios are meaningless).

---

## State of play

### Headline results (H100, best settings)

**Isolated attention layer** (k=2 K-hop mask, RCM ordering, autotune + 64-blocks;
fwd+bwd ms / peak GB — `results_h100_blocksize/`):

| config | L | flex — *full graph bias* | `flash` (causal, bias-free floor) | `flash_nc` (work-matched) | `eager` |
|---|---|---|---|---|---|
| 512×32 | 17.6k | 31.0 / 0.82 | 22.8 / 1.11 | 43.3 / 1.11 | **OOM** |
| 2048×8 | 17.4k | **20.5** / 6.08 | 22.0 / 5.57 | 42.6 / 5.57 | **OOM** |
| 2048×32 | 69k | **115.5** / 7.38 | 330.6 / 7.86 | 654.2 / 7.86 | **OOM** |

At the operating point flex now runs the **full graph bias at or below the cost
of bias-free causal flash** — at 2048×32 it is ~3× faster than the floor with
*lower* peak memory, and the forward alone sits within ~1.2× of flash's.

**Full model** (Llama-3.2-1B, spd+magnetic, k=2, flex routed; step ms / peak GB —
bias-checkpointing #7 on):

| config | L | flex | + decoder ckpt (#9) | `flash` floor (no ckpt) | history |
|---|---|---|---|---|---|
| 2048×2 | 4.5k | 1343 / 20.6 | — | 171 / 16.8 | was 1157 / **56.9** pre-#7 |
| 512×32 | 17.6k | 1294 / 56.9 | 1464 / **30.7** | 895 / 59.1 | eager OOM |
| 2048×8 | 17.4k | 2045 / 60.3 | 2520 / **30.3** | 876 / 58.5 | was **OOM** pre-#7 |

### Recommended settings & tradeoffs

| knob | recommendation | the trade |
|---|---|---|
| compile mode | `max-autotune-no-cudagraphs` (**default**, `flex_core.DEFAULT_COMPILE_MODE`) | 1.2–1.5× faster step for a ~320s one-time compile per bucketed shape (vs ~16s); amortized by bucketing + inductor's on-disk cache. `--compile-mode default` for quick iteration. |
| `BLOCK_SIZE` | **64 when k>0, 128 when k=0** | 64-blocks: 1.3–1.6× on fwd *and* bwd at k>0, but ~20% slower fwd at k=0 (nothing to skip, smaller tiles on dense work). Sub-128 requires the autotune mode. Harness default stays 128 for table continuity — sweep `--block-sizes 128 64`. |
| `node_id` dtype (#10) | **int32** (cast before capture) | free & lossless (bitwise-identical): net fwd+bwd −1.4 to −20% vs int64, biggest where the backward gather is DRAM-bound (large-N). int16 impossible (torch index-dtype rule). `--node-id-dtype int32`. |
| `checkpoint_graph_bias` (#7) | **True** (model-config default) | ~free at small N (+0.7%); at N=2048 costs ~+20% step for **−36 GB** and turns `2048×8` from OOM into runnable. Training-only; eval/decode untouched. |
| decoder gradient checkpointing (#9) | on when L ≳ 16k or memory-bound | +13–23% step for −26/−30 GB at L≈17.5k. `--gradient-checkpointing` on the bench; `model.gradient_checkpointing_enable(use_reentrant=False)` in training. Nests exactly with #7. |
| padding & bucketing | pad L to a block multiple (**mandatory** — ~14× cliff otherwise); bucket L *and* N with `flex_core.bucket_len` in the dataloader | ≤33% padding waste for ~20 distinct compile shapes below 100k tokens; per-batch BlockMask rebuild is then ~3 ms. |
| node ordering | RCM | one-time O(V+E) in preprocessing; concentrates K-hop neighbourhoods into skippable blocks. |
| K-hop | k>0 is where flex wins outright; k=0 is a worst-case reference where flex's value is memory-only (it runs where eager OOMs) | — |

### Known walls

- **L≈70k OOMs for *every* method, even flash with full checkpointing** — the
  unchunked LM-head cross-entropy (~90 GB of logits+loss at L=70k) → item #12.
- **The flex backward is ~66% atomic scatter-add** into the bias grad — pure
  same-address contention, dtype-independent. Remaining levers: #11 (model
  change) or the parked custom-kernel ideas at the bottom.

---

## Design: how flex serves the graph model

1. **The node-bias gather is the core trick.** The per-layer soft bias stays at
   node level `(B,H,N,N)`; a flex `score_mod` gathers
   `node_bias[b,h,node[q],node[k]]` inside the kernel, so the dense token-level
   `(B,H,L,L)` blow-up never materializes. The kernel is agnostic to how
   `node_bias` was produced — **all bias types (SPD, magnetic, …) come for
   free**, no per-type flex work.
2. **The structural mask becomes a sparse `BlockMask`.** Causal +
   bidirectional-prefix + K-hop + padding + diagonal guard, built once per
   batch as a `mask_mod`; flex then *skips* fully-masked blocks — the compute
   win.
3. **Realized speedup tracks block-level density, not element-level.** A 95%
   element-sparse mask whose allowed pairs are scattered can be only ~50%
   block-sparse. `density.py` reports both; measured speedups track
   `1/block_density` almost exactly (verified down to 64-blocks).
4. **Sequence length must be padded to a multiple of the block size.** The
   Triton kernel hits a ~14× cliff at non-aligned L (L=8847 → 83 ms vs
   L=8960 → 5.7 ms, same work). `flex_core.pad_to_block` is not optional.
5. **RCM node ordering matters**: it makes each node's K-hop neighbourhood
   contiguous, so far more blocks are fully skippable.

The original A100 study that established all this ended at: `eager` 1607 ms /
20 GB → `flex` 197 ms / 10 GB (L≈2.3k, k=2; 8× faster, ½ memory, within 1.5×
of the bias-free flash floor), with `eager` OOMing at any larger L. Everything
in the optimization log below starts from there.

---

## The benchmark suite

### Files

| File | Purpose |
|---|---|
| `inputs.py` | Synthetic graph-batch generator. Real topology (random-geometric → recoverable locality) + real K-hop/SPD; **synthetic** magnetic eigenvectors (shape-faithful — values don't affect timing). RCM/random/identity node ordering. Packs via the real `GraphCollatorV2`. |
| `density.py` | Element-level vs **block-level** mask coverage, chunked so it never materializes `L×L`. Block density predicts flex speedup (`~1/block_density`). |
| `flex_core.py` | The real flex implementation: `pad_to_block`, `build_block_mask` (mask_mod → `BlockMask`; block size int or `(Q, KV)`), `make_score_mod` (node-bias gather), `flex_attention_forward` (compiled with `DEFAULT_COMPILE_MODE = max-autotune-no-cudagraphs`), `bucket_len` (the L/N padding ladder for training), and a `dense_reference` for parity. Drops into the `graph_attention_v2.py` flex scaffold. |
| `bench_isolation.py` | One attention layer, 4 methods (`eager` dense SDPA / `flex` / `flash` causal floor / `flash_nc` non-causal work-matched floor). Times fwd, fwd+bwd, and the backward directly (`bwd`) + peak memory; warm BlockMask timing; `--bias-mode` decomposition, `--compile-mode`, `--block-size`, and `--node-id-dtype` (#10) knobs. Correct input-grad lifecycle (fresh leaves per step). |
| `bench_full_model.py` | Llama-3.2-1B fwd+bwd, same 3 methods. flex is routed at runtime (monkeypatch — the model source has no flex backend yet); `--checkpoint-bias` / `--gradient-checkpointing` arms for #7 / #9. |
| `run_sweep.py` | Drives the 2D (`nodes × tokens/node`) sweep × ordering × K-hop × method (× bias modes × block sizes), each run in a **fresh subprocess** (OOM isolation), with K-independent methods deduped to one run per config. Also `--recompile-probe`. |
| `profile_ncu.py` | Nsight Compute roofline: per-kernel DRAM% vs compute% (memory- vs compute-bound), fwd vs bwd. |
| `exp_node_id_dtype.py` | The #10 driver: flex-output **parity** across `node_ids` dtypes (int32 bitwise-identical to int64; int16 rejected) + a subprocess-isolated int64-vs-int32 **timing** sweep. Configurable `--nodes/--tpn/--k-hops` grid; `--repeats N --fresh-autotune` re-autotunes each repeat from a unique `TORCHINDUCTOR_CACHE_DIR` to expose autotune variance (reports mean ± std). Writes `results_h100_nodeid_v2/`. |
| `plot_results.py` | Publication figures (latency / peak memory vs sequence length) from the sweep JSONLs. |

### Running

```bash
# from repo root; modules use `src.models.flex_attn.*`

# single isolation run (one method, one config)
python -m src.models.flex_attn.bench_isolation \
    --method flex --n-nodes 512 --tokens-per-node 16 --k-hop 2 --ordering rcm

# single full-model run (add --no-checkpoint-bias / --gradient-checkpointing
# to flip the #7 / #9 arms)
python -m src.models.flex_attn.bench_full_model \
    --method flex --n-nodes 128 --tokens-per-node 16 --k-hop 2

# the full sweep (subprocess-isolated). --out-dir is a folder; the sweep writes
# {kind}.jsonl (full detail) and {kind}.md (latency + memory pivot tables).
# K-independent methods (flash, flash_nc, eager) run once per (nodes, tpn,
# ordering), not once per K — extra --k-hops values only add flex runs.
python -m src.models.flex_attn.run_sweep --kind isolation \
    --methods flash flash_nc eager flex --k-hops 0 2 4 --orderings rcm random \
    --block-sizes 128 64 \
    --out-dir src/models/flex_attn/results_h100

python -m src.models.flex_attn.run_sweep --kind full_model \
    --methods flash eager flex --k-hops 0 2 4 \
    --out-dir src/models/flex_attn/results_h100

# backward decomposition (#3): split the flex bwd into kernel / gather / scatter.
# (separate out-dir — a sweep overwrites {kind}.jsonl in its out-dir)
python -m src.models.flex_attn.run_sweep --kind isolation \
    --methods flex --bias-modes full frozen none --k-hops 0 2 \
    --nodes 512 2048 --tpn 8 32 128 --orderings rcm --max-tokens 400000 \
    --out-dir src/models/flex_attn/results_h100_bwd_decomp

# regenerate the markdown tables from a saved JSONL (writes <name>.md beside it)
python -m src.models.flex_attn.run_sweep --summarize src/models/flex_attn/results_h100/isolation.jsonl

# does bucketing keep torch.compile stable across heterogeneous graphs?
python -m src.models.flex_attn.run_sweep --recompile-probe

# memory-vs-compute roofline on a small representative set (slow; ncu)
python -m src.models.flex_attn.profile_ncu --max-configs 6
# ncu-free fallback (use when DCGM blocks the profiler, as on this box):
python -m src.models.flex_attn.profile_ncu --analytic --max-configs 6
```

Sweep axes default to `nodes ∈ {8,32,128,512,2048}`, `tokens/node ∈
{2,8,32,128,512}`; the prompt node is ~128 tokens; sizes are jittered ±20%.
Cells exceeding `--max-tokens` (default 100k) are pre-skipped.

### Reading the results

A sweep produces two artifacts in `--out-dir`:

- **`{kind}.md`** — two at-a-glance pivot tables over the same rows:
  **latency** (fwd+bwd ms) and **peak memory** (GB). One row per
  `(nodes, tpn, ordering)` config, with K folded into the columns. The
  K-independent methods (`flash`, `flash_nc`, `eager` — their cost doesn't
  depend on the K-hop mask, so the sweep runs them once per config) get a
  single column; flex gets one `flex-{K}` column per swept K, preceded in the
  latency table by that mask's `tokSp-{K}`/`blkSp-{K}` (token-/block-level
  sparsity). Bias-mode and block-size variants are suffixed: `flex[frozen]`,
  `flex@64`. `OOM` and error tags appear in the latency table. Regenerate
  anytime with `--summarize` (works on legacy one-row-per-K JSONLs too —
  duplicate K-independent runs collapse to the first seen).
- **`{kind}.jsonl`** — the full detail, one record per line, for analysis
  (`pandas.json_normalize(map(json.loads, open(path)))`).

A single direct run (`bench_isolation` / `bench_full_model`) prints the same
record as pretty JSON between `===RESULT_JSON_BEGIN===` / `===RESULT_JSON_END===`
sentinels. Each record is grouped into:

- `spec` — the swept graph parameters (`n_nodes`, `tokens_per_node`, `k_hop`, `ordering`, …).
- `run` — kernel/harness knobs (`num_heads`, `head_dim`, `block_size`, `dynamic`,
  `bias_mode`, `compile_mode`, `node_id_dtype`, `checkpoint_bias`, `gradient_checkpointing`, …).
- `shape` — realized sizes after jitter (`seq_len`, `max_num_nodes`, `total_tokens`).
- `density` — `element_density`/`element_sparsity` (token level) and
  `block_density`/`block_sparsity` (block level — what flex skips), `expected_attn_speedup`.
- `timing_ms` — `fwd`, `fwd_bwd`, `bwd` (the backward timed directly — **use
  this one**; `bwd_est = fwd_bwd − fwd` is kept for legacy files but is biased,
  since the no-grad and grad-mode forwards can select different compiled
  kernels — measured ~10 ms apart at L≈17k, enough to push `bwd_est` near 0),
  plus `compile` (flex one-time; near-zero when inductor's on-disk cache hits
  from an earlier run at the same shape, so only first-encounter values are
  meaningful), `blockmask_build` (cold: includes the one-time builder compile
  per distinct L) vs `blockmask_build_warm` (steady-state per batch — the
  number training pays once lengths are bucketed), `reorder_mean`/`reorder_max`
  (the RCM cost).
- `memory_mb` — `peak_fwd`, `peak_fwd_bwd`.
- `ok` / `error` — `OOM`, `TIMEOUT`, `CRASH`, or an exception name.

**Interpreting the memory-vs-compute split (`profile_ncu.py`).** Memory and
compute overlap on-GPU, so there is no clean millisecond split. ncu reports, per
kernel, achieved DRAM throughput and SM (compute) throughput as % of peak: mem%
≫ compute% ⇒ memory-bound, and vice-versa. We attribute kernels to fwd vs bwd by
running both phases. On this box **ncu is blocked by DCGM** (`nv-hostengine`
holds the counters), so use `--analytic`: it compares each config's arithmetic
intensity (FLOP/byte) to the bf16 ridge. Measured result: the graph-attention
forward is **memory-bound** across the sweep (AI ≈ 74–124) — flex's win is
bandwidth/footprint, and the backward `node_bias` scatter (a memory op) is the
cost to watch, consistent with the isolation timings.

### Notes / caveats

- The `flash` method is the bias-free **floor**, not a functional equivalent —
  FlashAttention cannot express the graph bias, bidirectional prefix, or K-hop.
  (`flash_attn` isn't installed here, so it's PyTorch SDPA's flash kernel.)
- **k=0 is a worst-case reference, not a target.** At k=0 the graph mask is
  bidirectional-prefix and ~dense (blkSp ≈ 0): flex has no blocks to skip, so
  it structurally cannot win there — while the `flash` floor runs
  `is_causal=True` and does ~half the work. Compare k=0 flex against
  `flash_nc` (non-causal, work-matched), and read k>0 as the operating point.
- Magnetic eigenvectors are synthesized (shapes are real). To benchmark with true
  spectral values, swap `inputs._build_items` to call the real
  `get_magnetic_laplacian_coords`.
- **bf16 bug fixed in passing:** `MagneticBias` divided by `valid.float()` (fp32),
  which broke the magnetic bias under bf16 (`mat1/mat2 dtype` error). Fixed to
  `.to(h_i.dtype)` in `src/models/graph_bias.py` — fp32-neutral, required for
  bf16 training with magnetic enabled.
- Sub-128 BlockMask block sizes require a `max-autotune` compile mode: torch
  2.6's default mode generates only 128-tile kernel configs, and tiles must
  divide the mask block size (`flex_attention_forward` raises a clear error).
  Rectangular `(Q, KV)` block sizes are plumbed through the APIs but
  unvalidated — square 64/128 are the tested options.
- If `profile_ncu.py` reports "ncu produced no CSV", the profiler lacks GPU
  performance-counter permission (needs `CAP_SYS_ADMIN` / admin-enabled counters).

---

## Optimization log

Ten experiments in three rounds, all complete; raw numbers in `results_h100*/`.
Phases 1–2 fixed the measurements and tuned the kernel; Phase 3 fixed memory;
#10 is a free, lossless index-width micro-opt on the gather.

| # | Experiment | Outcome |
|---|---|---|
| 1 | Honest k=0 floor (`flash_nc`) | causal flash undercounts the k=0 work ~2× — confirmed (1.93–1.99×); k=0 declared a worst-case reference, not a target |
| 2 | BlockMask build cost | a one-time compile per distinct L (4–39 s), **not** a per-batch cost (~3 ms warm); `bucket_len` ladder bounds the compiles |
| 3 | Backward decomposition | **66–77% atomic scatter** / ~8% gather / ~25% kernel floor; also exposed and fixed a biased `bwd_est` metric |
| 4 | `max-autotune-no-cudagraphs` | fwd **4.7×** at k=2 (lands near the flash floor), net step 1.47× — adopted as the default |
| 5 | fp32 `node_bias` | refuted by codegen inspection: the scatter already runs native fp32 atomics; its cost is pure contention |
| 6 | Block size 64 vs 128 | 64-blocks win **1.3–1.6× on fwd *and* bwd** at k=2 (speedup ≈ active-block ratio); ~20% *slower* fwd at k=0 — 64 when k>0, 128 when k=0 |
| 7 | Checkpoint the bias modules | **−36.3 GB at 2048×2, un-OOMs `2048×8`** for +20% step at N=2048 (~0 at small N); grads bitwise identical; on by default |
| 8 | MagneticBias restructure | −8.9 GB (no-ckpt path), −3% step with ckpt; **exact parity** (fp64 at machine epsilon); folded path is the default |
| 9 | Decoder gradient checkpointing | **−26/−30 GB at L≈17.5k** for +13/+23%; exposed #12 (unchunked CE) as the true large-L wall |
| 10 | int32 `node_ids` | lossless (bitwise-identical), free one-line cast: **net fwd+bwd faster in all 8 configs** (−1.4 to −20.1%, 3 fresh-autotune repeats); biggest where the backward gather is DRAM-bound (large-N k=0 −24.6% bwd). Earlier single-shot +9.7% bwd was autotune noise (→ −6.6% repeated). int16 impossible (torch index-dtype rule) |

### 1. Honest baselines: `flash_nc` and the k=0 scope note

The original `flash` floor runs `is_causal=True` and computes ~half the score
blocks, while the k=0 graph mask is bidirectional-prefix and ~dense
(blkSp ≈ 0) — so the A100 headline "flex is 5× (fwd) / 22× (bwd) flash at
k=0" overstated the kernel overhead by ~2×. The `flash_nc` method (non-causal
SDPA flash, full L×L) is the work-matched floor: measured at **1.93–1.99×
`flash`** for L ≥ 17k, exactly the predicted factor. Against it, the
pre-autotune k=0 gap was ~8–9× fwd+bwd, not 17–19×. And since flex's whole
mechanism is skipping fully-masked blocks, k=0 (nothing to skip) is a
**worst-case reference, not an optimization target** — k>0 is the operating
point. (`flash_nc` is isolation-only: the full-model `flash` baseline is a
stock causal Llama, and a non-causal variant of it wouldn't mean anything.)

### 2. The BlockMask build is a compile cost, not a per-batch cost

The alarming 10–43 s `blockmask_build` readings are ~entirely the one-time
`_compile=True` builder compile that dynamo pays per *distinct L*; rebuilding
a mask at an already-compiled L costs **~3 ms** at L≈17.6k (reported as
`blockmask_build_warm`). The mask itself genuinely must be rebuilt every
batch — each batch is a different graph — but the expensive part is compiling
the builder *code*, and that is keyed on L. `flex_core.bucket_len` is the
padding ladder training should use: powers-of-2 multiples of the block size
with 1.5× midpoints (… 384, 512, 768, 1024 …), bounding padding waste at ~33%
and the distinct-shape count at ~20 below 100k tokens. It applies to N as
well as L (the compiled flex kernel also guards on the captured
`node_bias`/`node_ids` shapes); `--recompile-probe` validates that this
bucketing keeps the compile count flat across heterogeneous graphs. Wiring
the ladder into the training dataloader lands with the `graph_attention_v2`
integration.

### 3. Anatomy of the flex backward

Two findings. First, a metric bug: `bwd_est = fwd_bwd − fwd` is biased — the
no-grad and grad-mode forwards can select different compiled kernels
(measured ~10 ms apart at L≈17k), enough to push the estimate to near-zero
nonsense on some variants. Both benches now time `.backward()` directly and
report it as `timing_ms.bwd`; use that one (`bwd_est` remains only for legacy
files).

Second, the decomposition itself, via `--bias-mode {full,frozen,none}`
(full = gather + scatter-add grad; frozen = bias captured with
`requires_grad=False`, so gather only; none = `score_mod=None`, the bare
masked kernel) — also a `--bias-modes` sweep axis, rendered as
`flex[frozen]` / `flex[none]` table columns. At 512×32, k=2 (pre-autotune,
bwd = 36.1 ms):

| backward component | ms | share |
|---|---|---|
| atomic scatter-add into the bias grad (`full − frozen`) | 23.8 | 66% |
| bias gather / score recompute (`frozen − none`) | 2.9 | 8% |
| masked-attention kernel floor (`none`) | 9.4 | 26% |

The scatter dominates — and per finding #5, not for dtype reasons: it is pure
same-address contention (every token pair of a node pair adds into one
address, so conflicts scale with tokens-per-node²).

The same decomposition at k=0 (dense mask ⇒ ~10× the active blocks; 512×32,
default mode) makes the structure unmistakable:

| | fwd | bwd |
|---|---|---|
| none (bare masked kernel) | 19.6 ms | 38.1 ms |
| frozen (+ gather) | 85.4 ms (**+66 = 77%**) | 68.0 ms (+30) |
| full (+ scatter) | 83.2 ms | 300.0 ms (**+232 = 77%**) |

The bare kernel is only ~1.35× the work-matched `flash_nc` (58 vs 43 ms
fwd+bwd) — flex's k=0 overhead is ~86% *bias machinery*: the score_mod does
per-**token**-pair work (one gather in fwd, one atomic in bwd, per (q,k)
element) for logically per-**node**-pair data, a tokens-per-node² redundancy.
The scatter scales exactly with active token pairs (232 ≈ 10 × the 23.8 ms at
k=2), so the same per-active-block costs explain every k — and the parked
custom-kernel ideas at the bottom all follow from this one fact.

### 4. `max-autotune-no-cudagraphs` — adopted as the default

Inductor's default flex configs come from a small heuristic table; under
`max-autotune` it benchmarks candidate Triton configs per shape and keeps the
winner (`-no-cudagraphs` because the per-batch BlockMask tensors and fresh
grad leaves are not CUDA-graph-safe). Measured at 512×32 (H100):

| | fwd | bwd (direct) | net fwd+bwd |
|---|---|---|---|
| k=2, default → autotune | 35.9 → **7.6 ms (4.7×)** | 36.1 → 41.2 ms | **1.47×** |
| k=0, default → autotune | 83.0 → 46.3 ms (1.8×) | 299.8 → 277.6 ms | 1.18× |

The autotuned k=2 forward lands within ~1.2× of the causal-flash forward at
the same shape — **the forward is essentially solved at the operating point**.
The small bwd regression at k=2 is the autotuner microbenchmarking configs
without the in-situ atomic contention; the net step wins everywhere measured
(and #6's 64-blocks later more than recovered the regression). Adopted as
`flex_core.DEFAULT_COMPILE_MODE`; the benches and `run_sweep` default to it
too (`--compile-mode default` opts back into the fast-compiling heuristics
while iterating). Cost: ~320 s autotune per distinct shape (vs ~16 s),
amortized by the #2 bucketing ladder and persisted by inductor's on-disk
cache (re-runs at known shapes compile in ~1 s); the sweep's default
`--timeout` is 1800 s to absorb it.

### 5. fp32 `node_bias` — hypothesis refuted by the generated code

The hypothesis: bf16 atomics are CAS-emulated, so keeping the bias leaf in
fp32 would swap retry loops for native `atomicAdd` and reclaim much of the
scatter cost. Inspecting the inductor-generated backward
(`TORCH_LOGS=output_code`) killed it: the dbias accumulator is **already
allocated fp32** regardless of the leaf dtype
(`empty_strided_cuda((B,H,N,N), …, torch.float32)`; kernel arg `*fp32`),
`dsT` is cast `.to(tl.float32)` *before* `tl.atomic_add`, and the grad is
downcast to bf16 afterwards in a separate cheap kernel. The scatter already
runs native fp32 atomics; its ~24 ms is pure same-address **contention**.
Dtype is not a lever. The levers that remain: per-KV-head bias (#11 — 4×
fewer conflicting atomics), or the parked custom-kernel ideas.

### 6. Block size: 64 vs 128 — 64 wins wherever there is sparsity

Smaller BlockMask blocks fit scattered K-hop neighbourhoods more tightly, so
more blocks become fully skippable. Measured at k=2 (autotune mode,
`results_h100_blocksize/`):

| config | L | blkSp 128 → 64 | fwd 128 → 64 | bwd 128 → 64 |
|---|---|---|---|---|
| 512×8 | 4.5k | 0.81 → 0.85 | 1.3 → 1.0 ms | 5.9 → 3.2 ms |
| 512×32 | 17.6k | 0.90 → 0.93 | 7.3 → 5.2 ms | 40.9 → 25.7 ms |
| 2048×8 | 17.4k | 0.92 → 0.95 | 6.8 → 4.2 ms | 25.3 → 16.2 ms |
| 2048×32 | 69k | 0.97 → 0.98 | 32.3 → 22.1 ms | 137.6 → 93.1 ms |

**1.3–1.6× on both forward and backward, everywhere measured** — including
the large-tpn cells where a per-block kernel-efficiency penalty was expected
and did not materialize (under autotune, which carries proper 64-tile
configs). The speedup tracks the active-block ratio almost exactly
(`(1−blkSp₆₄)/(1−blkSp₁₂₈)` ≈ the measured time ratio; e.g. 0.62 predicted
vs 0.62 measured at 2048×8), confirming the block-density model down to finer
granularity.

**At k=0 there is no benefit, as the model predicts** (measured on the same
six graphs): blkSp stays ≈ 0 at both block sizes — nothing to skip — and the
64-block *forward* is ~20% **slower** at the meaningful sizes (e.g. 512×32:
46.0 → 55.2 ms; 2048×32: 709 → 865 ms), the tile-efficiency cost of capping
the kernel at 64-wide tiles on dense work; the backward is ~flat. So the
block size follows the K-hop gate: **64 when k>0, 128 when k=0**.
Rectangular `(Q, KV)` blocks were dropped as unjustified complexity.

### 7. Checkpoint the per-layer bias modules — on by default

At N=2048 autograd kept every layer's MagneticBias einsum outputs, the
`(B,N,N,2m)` cat, MLP hiddens, and the `(B,H,N,N)` output alive ×16 layers
(~40 GB) — the reason `2048×8` OOMed on flex while flash ran. Now
`GTLMLlamaConfig.checkpoint_graph_bias=True` (default) recomputes the bias in
backward via `torch.utils.checkpoint(..., use_reentrant=False)`, implemented
in the real `GTLMLlamaAttention.forward` (training-only; eval/generation
untouched; the bench takes `--no-checkpoint-bias` for baseline arms). Safe
because the bias cache is never *read* in training, so the recompute is
deterministic — verified by a gradient-parity test: identical loss, all 144
bias-param grads **bitwise identical** on/off. In passing, training-mode cache
*writes* were disabled in `graph_bias.py` (the cache is an eval/decode
feature; writing in training pinned every layer's `(B,H,N,N)` output ≈ 4.3 GB
at N=2048 for nothing).

**Measured** (flex, k=2): `2048×2` peak **56.9 → 20.6 GB (−36.3 GB)** at
+19.6% step time — the bias forward is a large share of the step at N=2048
and backward re-runs it (not "nearly free" as originally hoped, but cheap for
what it buys); **`2048×8` goes OOM → runnable**; `512×32` −2.3 GB at +0.7%,
the small-N control where the peak is generic activations (#9's territory).

### 8. MagneticBias restructure: fold `W1` into `phi`, drop the `cat`

The first `proj` layer is linear, so it commutes with the Σ over eigenvectors
in the einsums: project `phi` (B,M,m — tiny) into `phiR`/`phiI` *before* the
N² einsums and emit the (B,N,N,m) hidden directly — `real`/`imag` and the
(B,N,N,2m) cat never exist. Folded path is the default in `graph_bias.py`;
the original is kept behind a `legacy_unfolded` module attribute; parameters
are unchanged, so existing checkpoints load as-is.

**Parity:** fp64 outputs agree to 5e-16 and all param grads to 5e-15 —
machine epsilon, i.e. mathematically identical (fp32/bf16 differ only at
their own epsilon scale from op reordering), in both full (M=N) and truncated
(M<N) eigenvector regimes. **Measured** (flex, k=2, vs the #7 arms):
`2048×2` without bias-checkpointing **56.9 → 48.0 GB (−8.9 GB)**, matching
the saved-tensor accounting (~0.54 GB × 16 layers); with checkpointing, time
1384 → 1343 ms and `2048×8` 2089 → 2045 ms (memory flat — the ckpt-on peak is
not recompute-dominated). The recompute-tax recovery is modest because the
four N² *einsums* dominate the bias forward and are mathematically required —
only the cat write and the 2m-wide Linear disappeared.

### 9. Decoder gradient checkpointing — and the discovery of the real wall

`--gradient-checkpointing` on `bench_full_model` (default off — this one
costs real latency), via HF's `gradient_checkpointing_enable(...,
use_reentrant=False)`, for both the GTLM and stock-flash arms. The
`_graph_ctx` module-attribute design survives the recompute as intended, and
nesting with the #7 bias checkpoint is exact — verified by a gradient-parity
test (identical loss; bias *and* decoder-layer grads bitwise identical).

**Measured** (flex, k=2, bias-ckpt on): `512×32` **56.9 → 30.7 GB** at +13%
time; `2048×8` **60.3 → 30.3 GB** at +23%. **But the L≈70k cells still OOM —
including flash with full checkpointing** — which exposes the real wall: the
unchunked LM-head loss. `peak_fwd` (a no-grad forward!) is already ~23.5 GB
at L=17.6k: logits are (L, 128k-vocab) → bf16 + fp32-upcast + log_softmax
≈ 23 GB there and ≈ **90 GB at L=70k**, before any transformer activation.
The original study's "512×128 OOMs for every method" was the cross-entropy
all along → item #12.

### 10. int32 `node_ids` — a free, lossless win on the gather

The score_mod's core op gathers `node_bias[b,h,node_ids[q],node_ids[k]]` per
(q,kv) element, and the structural BlockMask builder indexes the same
`node_ids`. Both bake the tensor's dtype into the compiled Triton, so a 64-bit
id means an 8-byte index load per element and 64-bit address arithmetic in the
inner loop. Node counts are ≤ a few thousand, so narrowing the captured ids is
lossless — the only question is whether it buys time, and where. Run via
`bench_isolation --node-id-dtype {int64,int32}` and the `exp_node_id_dtype.py`
driver; the authoritative run is `results_h100_nodeid_v2/` (8-config grid,
3 fresh-autotune repeats, `--repeats 3 --fresh-autotune`).

**int16 is impossible, int32 is the floor.** `node_ids[q]` is *used as an index*
into `node_bias`, and torch requires index tensors to be long/int/byte/bool
("tensors used as indices must be long, int, byte or bool") — int16 is rejected
at trace time. uint8 (`byte`) would pass but caps at 255 nodes, so **int32 is
the narrowest generally-usable width.**

**Parity:** flex with int32 ids is **bitwise identical** to int64 (max|Δ| = 0,
k=0 and k=2) — only the index width changes, never a value — and matches the
dense reference. So this is a pure-upside cast with no ablation needed.

**Measured** (H100, autotune + #6 block-size gate; **8-config grid × 3 repeats,
each repeat re-autotuned from a fresh inductor cache** so the numbers carry an
honest variance band; %Δ is on the means, full mean ± std (min) in
`results_h100_nodeid_v2/`):

| config | L | blkSp | fwd Δ | bwd Δ | **fwd+bwd Δ** |
|---|---|---|---|---|---|
| k=0 512×8 | 4.5k | 0.03 | −6.5% | −6.8% | **−5.8%** |
| k=0 512×32 | 17.6k | 0.01 | −5.5% | −9.9% | **−8.7%** |
| k=0 2048×8 | 17.4k | 0.01 | +6.3% | −24.6% | **−20.1%** |
| k=0 2048×32 | 69k | 0.00 | −0.7% | −8.8% | **−7.8%** |
| k=2 512×8 | 4.5k | 0.85 | −2.3% | −7.5% | **−6.9%** |
| k=2 512×32 | 17.6k | 0.93 | −2.6% | −6.6% | **−5.8%** |
| k=2 2048×8 | 17.4k | 0.95 | −2.4% | −1.6% | **−1.4%** |
| k=2 2048×32 | 69k | 0.98 | −2.2% | −4.9% | **−4.5%** |

**Net (fwd+bwd) is faster with int32 in all 8 configs — −1.4% to −20.1%, never
a net regression.** The repeated run also settled the one doubt from the first
pass: the originally-reported **+9.7% k=2 512×32 backward was an autotune
artifact** — over 3 fresh-autotune repeats it is **−6.6%**. That single-shot
number was the autotuner picking a worse config on one draw (the same per-shape
selection variance #4 flagged); averaging independent autotunes removes it.

The win tracks the gather's share. The **backward gains most where the gather is
*bandwidth*-bound** — k=0's dense gather and especially **large-N k=0 2048×8
(−24.6%)**, where the `(B,H,N,N)` bias table spills L2 and the gather goes
DRAM-bound (finding #5), exactly where halving the index-load width pays.
Forward improves in 6/8 (−2 to −6.5%) and is flat at k=0 2048×32; the one
reproducible forward *regression* is k=0 2048×8 (**+6.3%**, std≈0 — the
autotuner lands on a worse fwd config there), but that same config posts the
largest backward win, so net is −20.1%. Bonus: int32 often **stabilizes** kernel
selection too (k=0 2048×8 backward std 67 ms → 0.26 ms).

**Verdict: adopt.** Free (a one-line `node_ids.to(torch.int32)` *before
capture*, so the compiled kernel emits int32 loads), lossless (bitwise-identical
output), and **net faster in every config measured** (1.4–20%), with the biggest
wins exactly where the backward gather is the bottleneck. Fold the cast into the
`graph_attention_v2` integration where `node_ids` is captured for the score_mod
and BlockMask builder.

---

## Remaining work

**11. Per-KV-head bias (32 → 8 heads) — modeling change, needs a training ablation.**
- **Why:** GQA already shares K/V across query-head groups; sharing the graph
  bias the same way shrinks the `(B,H,N,N)` tensor, the gather traffic, and the
  scatter-add contention all by 4× — it attacks the speed *and* memory
  bottleneck at once. Per finding #5 it is the only remaining lever on the
  backward scatter short of custom kernels; per finding #3 it also improves
  gather locality exactly where it degrades (at N=2048 the bias table exceeds
  L2, making gathers DRAM-bound).
- **How:** parameterize `GraphAttentionBias` with `num_kv_heads`; index the
  gather with `h // (H // Hkv)` in `make_score_mod`; expand with
  `repeat_interleave` on the dense path for parity.
- **Evaluate:** speed/memory via the isolation sweep, but this changes the
  model — it needs a small training ablation (loss curves vs per-query-head
  bias) before adoption. Quality impact unknown until ablated.
- **Sequencing:** the `flex_core` API (#6), kernel config (#4), and the
  MagneticBias rewrite (#8) are all settled — this is ready whenever a
  training run is affordable.

**12. Chunked / fused cross-entropy — the L≈70k gate (discovered by #9).**
- **Why:** at L≈70k the (L, 128k-vocab) logits cost ~18 GB bf16 + ~36 GB
  fp32-upcast + ~36 GB log_softmax ≈ **90 GB before any transformer
  activation** — every method incl. flash OOMs on it, with or without
  checkpointing. Even at L=17.6k it sets the post-#9 floor (~23 GB no-grad
  forward peak). Entirely orthogonal to the graph machinery.
- **How:** compute the LM head + cross-entropy in sequence chunks so only one
  chunk's logits ever materialize — e.g. run the base model to hidden states,
  then a checkpointed `lm_head`+CE per chunk with weighted accumulation (or
  adopt a fused linear-CE kernel à la Liger). Needs care in the GTLM forward,
  which installs the graph context before delegating.
- **Evaluate:** does `512×128` / `2048×32` (L≈70k) become runnable for flash
  and flex with #9 on; new floor at L=17.6k.
- **Gain:** unlocks the largest configs for *all* methods; several-× lower
  loss-spike memory everywhere. Independent of #11.

## Parked: deeper kernel ideas (revisit if the last 1.5–2× matters)

Deferred deliberately — flex already beats the causal-flash floor at the k>0
operating point, and small-input / k=0 runs are fast in absolute terms. All of
these attack the one measured core bottleneck (finding #3's k=0 decomposition):
the `score_mod` does per-**token**-pair work for per-**node**-pair data, a
tokens-per-node² redundancy that flex's elementwise API cannot see.

- **Custom dbias backward kernel** (the big one, separable): run flex with the
  bias frozen (dQ/dK/dV stay correct) and compute the bias grad in a dedicated
  Triton kernel — recompute P from the saved LSE per tile, pool `dS` by node
  pair in SMEM, one atomic per node pair per tile (~16 instead of ~16k at
  tpn=32). Kills the scatter (77% of bwd at k=0, 66% at k=2) at every k.
- **Custom forward ("graph-flash-attention")**: same idea on the gather side —
  load the few-element per-tile bias sub-block to SMEM once and broadcast.
  Removes the gather (77% of the k=0 forward). Requires replacing flex's
  forward entirely; biggest effort.
- ~~**int32 `node_ids`**~~: **done — promoted to #10** (free, lossless; net
  fwd+bwd faster in all 8 configs, −1.4 to −20%, biggest where the backward
  gather is DRAM-bound). int16 is not an option (torch index-dtype rule).
- **Replica-split scatter** (~1 afternoon): capture the bias as a materialized
  `(B,H,R,N,N)` and index with `q_idx % R` — R× less atomic contention for R×
  bias-grad memory; autograd reduces replicas with one sum.
- **Re-measure on torch upgrades**: captured-buffer grads in flex are a known
  upstream sore point; new releases may move these numbers for free.
