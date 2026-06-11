# FlexAttention benchmarks for GTLM

Does `torch` FlexAttention beat the current dense graph-attention path on compute
and memory — especially with sparse K-hop masks — while preserving the SPD and
magnetic-Laplacian biases? This suite answers that with measurements on an A100_80GB GPU.

## TODO

Follow-ups from analyzing the sweep results. The two headline bottlenecks the
items attack:

- **Speed — the flex *backward*.** At k=0, L≈70k the forward is 5× the flash
  floor but the backward is **22×**. The gradient of the `score_mod` gather is
  a scatter-add into `(B,H,N,N)` with atomic adds; every token pair of the same
  node pair hits the *same address*, so contention scales with tokens-per-node².
- **Memory — the per-layer bias intermediates, not attention.** Attention O(L²)
  is solved. At N=2048, L=4457 the peak goes 12.3 GB (fwd) → 56.9 GB (fwd+bwd)
  vs flash's 16.8 GB: that delta is 16 layers of autograd-saved `(B,N,N,·)`
  MagneticBias intermediates, and it is why `2048×8` OOMs on flex while flash runs.

**Ordering is by dependency, not pure priority**: Phase 1 changes the harness
and baselines that every later item is *measured* against; Phase 2 finalizes
`flex_core` knobs and APIs that later work builds on; Phases 3–4 are the big
wins, landed last so they don't get refactored under shifting foundations.

### At a glance

| # | Item | Type | Effort | Expected gain |
|---|---|---|---|---|
| 1 | Honest k=0 floor + scope note | reporting | S | correct baselines for everything below |
| 2 | Warm BlockMask timing + L-bucketing | harness/infra | S | kills a hidden 10–20 s per-novel-shape stall |
| 3 | Backward cost decomposition | measurement | S | bounds the payoff of #4, #5, #10 |
| 4 | `max-autotune` flex compile | kernel knob | S | ~1.3–2× on flex bwd kernel |
| 5 | fp32 `node_bias` on the flex path | kernel knob | S | up to a large slice of the 22× bwd gap |
| 6 | `BLOCK_SIZE` sweep (64 / rectangular) | kernel knob + API | M | up to ~2–3× fwd at small tokens-per-node |
| 7 | Checkpoint the bias modules | memory | S | **~40 GB at N=2048; un-OOMs `2048×8`** |
| 8 | MagneticBias restructure (fold `W1`, drop `cat`) | memory | M | halves the largest per-layer intermediate |
| 9 | Gradient-checkpoint decoder layers | memory | S | several-× activation memory at large L |
| 10 | Per-KV-head bias (32 → 8) | modeling | L | 4× bias memory *and* 4× scatter traffic |

### Phase 1 — measurement & reporting groundwork

Small harness changes. They come first because they define the baselines and
result schema every later item is evaluated through — landing them late would
mean re-running and re-interpreting earlier sweeps.

**1. Honest k=0 floor + scope note.**
- **Why:** the flash floor runs `is_causal=True` (skips half the blocks) while
  the k=0 graph mask is bidirectional-prefix and ~dense (blkSp ≈ 0) — "flex fwd
  5× flash at k=0" overstates the kernel overhead by ~2×. And with no blocks to
  skip, flex structurally cannot win at k=0; k>0 is the operating point.
- **How:** add a non-causal flash variant to `bench_isolation` (or an
  `expected_work` correction column in the summary tables); add a README note
  declaring k=0 a worst-case reference, not a target.
- **Evaluate:** n/a — reporting change.
- **Gain:** none at runtime; prevents misreading every comparison below and
  chasing a config flex can't win.
- **Sequencing:** first, so all subsequent sweeps are read against a fair floor.

**2. Steady-state BlockMask build timing + explicit L-bucketing.**
- **Why:** the reported `blockmask_build` of 10–43 s is dominated by the
  one-time `_compile=True` dynamo compile per *distinct L*, not the per-batch
  cost — but it also means every new L>8192 in training pays ~10–20 s.
- **How:** in `bench_isolation`, time a second `build_block_mask` at the same L
  and report it as `blockmask_build_warm`; in the training dataloader, bucket
  padded lengths to a small fixed set (e.g. powers-of-2 multiples of 128) so
  the builder compile caches hit; optionally prefetch the build.
- **Evaluate:** `blockmask_build_warm` ≪ cold; `--recompile-probe` confirms a
  bounded compile count over a heterogeneous epoch.
- **Gain:** removes a hidden multi-second per-novel-shape training stall;
  corrects a misleading benchmark number.
- **Sequencing:** before the kernel-knob sweeps (#4–#6), which all rebuild
  BlockMasks and should report the warm number.

**3. Decompose the flex backward cost.**
- **Why:** the bwd is 13–22× the flash floor at k=0. Before optimizing, split
  it into (kernel / gather / scatter) so every speed fix has a measured ceiling.
- **How:** add an isolation-bench axis running flex three ways: `score_mod=None`
  (mask only), bias present but `requires_grad=False` (gather, no scatter), and
  full (gather + scatter-add).
- **Evaluate:** compare `bwd_est` across the three at a few (tpn, k) points.
- **Gain:** no direct speedup — it quantifies how much of the 22× is atomic
  scatter vs the Triton bwd kernel itself, and directs #4, #5, and #10.
- **Sequencing:** before #4/#5/#10; their expected payoff is unknown without it.

### Phase 2 — `flex_core` knobs & API

Settle the kernel configuration and function signatures *before* the big
memory/modeling work: #4–#6 shift the speed baselines (any sweep run before
them is stale) and #6 changes `flex_core` APIs that #10 and the eventual
`graph_attention_v2` integration will call.

**4. Compile flex with `mode="max-autotune-no-cudagraphs"`.**
- **Why:** inductor's default flex kernel configs are conservative, especially
  for the backward; autotuning frequently buys 1.3–2× on flex bwd.
- **How:** one-line change in `flex_core.get_flex_attention`; key the compiled-
  callable cache on the mode so both variants stay comparable.
- **Evaluate:** isolation sweep `fwd`/`bwd_est`, plus the one-time `compile`
  cost (autotune compiles are much slower — amortization matters).
- **Gain:** ~1.3–2× on the flex backward kernel for a config flag.
- **Sequencing:** decide the mode now — it changes every speed number, so
  later items shouldn't be benchmarked twice.

**5. Keep `node_bias` in fp32 on the flex path.**
- **Why:** the backward scatter does atomic adds into `node_bias`; A100 has
  fast native fp32 `atomicAdd` while bf16 atomics are slower, and the tpn²
  contention compounds that. The 2×-bigger forward gather reads are cheap —
  the backward is what we're buying.
- **How:** in `_build_flex` / the full-model monkeypatch, keep the bias leaf in
  fp32 (don't cast the bias module output down to bf16).
- **Evaluate:** isolation `bwd_est` at k=0 with high tpn (worst contention);
  parity vs `dense_reference`.
- **Gain:** unknown until #3 bounds it; potentially a large fraction of the
  bwd gap if bf16 atomics dominate.
- **Sequencing:** the chosen bias dtype is an input to #7, #8, and #10 (they
  all touch how the bias tensor is produced/consumed) — fix it before them.

**6. `BLOCK_SIZE` sweep (64, and rectangular `(128, 64)`).**
- **Why:** realized speedup tracks *block* sparsity, and small tokens-per-node
  leaves a lot on the table (512×2 k=2: tokSp 0.97 but blkSp 0.67 → flex
  captures ~3× of a theoretical ~30×). Smaller KV blocks fit scattered K-hop
  neighbourhoods more tightly, at some per-block kernel-efficiency cost.
- **How:** extend `pad_to_block` / `build_block_mask` to accept a
  `(Q_BLOCK, KV_BLOCK)` tuple (`create_block_mask` already does; padding must
  use the larger of the two); add `--block-size` as a sweep axis.
- **Evaluate:** isolation `fwd`/`bwd_est` and `blkSp` at small tpn (2, 8),
  k ∈ {2, 4}; watch for large-tpn regressions.
- **Gain:** plausibly the biggest remaining *forward* win for small-tpn graphs
  (up to ~2–3× where block sparsity lags element sparsity).
- **Sequencing:** the signature change ripples into every `flex_core` caller —
  land it before #10 and before wiring flex into `graph_attention_v2`, so that
  code is written against the final API.

### Phase 3 — memory

Independent of the speed knobs; landed after Phase 2 only so their evaluation
sweeps run on the final kernel config. #7 is the highest-leverage item on the
whole list.

**7. Checkpoint the per-layer bias modules.**
- **Why:** the ~44 GB fwd→fwd+bwd delta at N=2048 is autograd keeping every
  layer's MagneticBias einsum outputs, the `(B,N,N,2·magnetic_dim)` cat, MLP
  hiddens, and the `(B,H,N,N)` output alive ×16 layers. The bias compute is
  milliseconds (~17 GFLOP at N=2048), so recomputing in backward is nearly free.
- **How:** wrap the `self.graph_bias(...)` call in `GTLMLlamaAttention.forward`
  (and the flex monkeypatch in `bench_full_model.py`) in
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
- **Evaluate:** full-model sweep; `peak_fwd_bwd` at `n_nodes ∈ {512, 2048}`;
  does `2048×8` (L=17392) still OOM?
- **Gain:** large-N peak collapses to ~one layer's bias intermediates
  (**~40 GB saved at N=2048**); should un-OOM `2048×8`.
- **Sequencing:** independent of everything else — could land any time; kept
  here only so its before/after sweep uses the final kernel config.

**8. Restructure `MagneticBias`: fold `W1` into `phi`, drop the `cat`.**
- **Why:** the `(B,N,N,2·magnetic_dim)` cat (~537 MB/layer at N=2048) and the
  four einsum outputs are the largest saved intermediates. The first projection
  is linear, so it can be folded into `phi` *before* the N² einsum.
- **How:** in `src/models/graph_bias.py`, split `W1` into real/imag halves,
  precompute `phiR = phi @ W1[:, :K].T`, `phiI = phi @ W1[:, K:].T` (small,
  `(B,N,dim)`), and emit the first hidden layer `(B,N,N,magnetic_dim)` directly
  from the einsums (then bias, SiLU, second linear as before). Keep the old
  path behind a flag for a parity test.
- **Evaluate:** `torch.allclose` vs the current implementation (fp32) first;
  then full-model `peak_fwd_bwd` at N=2048.
- **Gain:** roughly halves the largest per-layer intermediate; also cuts the
  *recompute* peak once #7 is in. Smaller than #7 but complementary.
- **Sequencing:** after #5 (bias dtype fixed) and before #10 — validate the
  rewrite at H=32 against the existing implementation while it still exists
  unchanged.

**9. Gradient-checkpoint the decoder layers (large-L regime).**
- **Why:** at L≈17.5k flex and flash converge on ~56–59 GB — generic
  transformer activations (MLP etc.), nothing graph-specific left to win; the
  512×128 full-model cell OOMs for *every* method.
- **How:** `model.gradient_checkpointing_enable()` in `bench_full_model` (the
  `_graph_ctx` module-attribute design already survives recompute by
  construction).
- **Evaluate:** `peak_fwd_bwd` and the latency penalty (expect ~1.3×) at
  L ≥ 17k; whether 512×128 becomes runnable.
- **Gain:** standard several-× activation-memory reduction; orthogonal to flex
  but required to reach the largest configs.
- **Sequencing:** fully independent; listed late only because it's not
  flex-specific.

### Phase 4 — modeling change (needs a training ablation)

**10. Per-KV-head bias (32 → 8 heads).**
- **Why:** GQA already shares K/V across query-head groups; sharing the graph
  bias the same way shrinks the `(B,H,N,N)` tensor, the gather traffic, and the
  scatter-add contention all by 4× — it attacks the speed *and* memory
  bottleneck at once.
- **How:** parameterize `GraphAttentionBias` with `num_kv_heads`; index the
  gather with `h // (H // Hkv)` in `make_score_mod`; expand with
  `repeat_interleave` on the dense path for parity.
- **Evaluate:** speed/memory via the isolation sweep, but this changes the
  model — it needs a small training ablation (loss curves vs per-query-head
  bias) before adoption.
- **Gain:** 4× smaller bias memory and 4× less scatter traffic; quality impact
  unknown until ablated.
- **Sequencing:** last deliberately — it builds on the final `flex_core` API
  (#6), the chosen bias dtype (#5), and the validated MagneticBias rewrite
  (#8), and it's the only item whose cost is a training run rather than a
  benchmark sweep.

## TL;DR of what we found

1. **The node-bias gather is cheap; materializing `(B,H,L,L)` is the problem.**
   Keeping the per-layer soft bias at node level `(B,H,N,N)` and gathering
   `node_bias[b,h,node[q],node[k]]` inside a flex `score_mod` avoids the dense
   token-level blow-up. Because the kernel is agnostic to *how* `node_bias` was
   produced, **all bias types (SPD, magnetic, …) come for free** — no per-type
   flex work.
2. **Sequence length MUST be padded to a multiple of the block size (128).**
   flex's Triton kernel hits a ~**14× cliff** at non-128-aligned `L`
   (e.g. L=8847 → 83 ms vs L=8960 → 5.7 ms, same work). `flex_core.pad_to_block`
   handles this; it is not optional.
3. **Realized speedup tracks *block-level* density, not element-level.** flex only
   skips a 128-block when it is *entirely* masked. A mask that is 95%
   element-sparse but whose allowed pairs are scattered (small tokens-per-node)
   can be only ~50% block-sparse → far less than the naive "95% sparse → 20×".
   `density.py` reports both.
4. **Node ordering matters for K-hop sparsity.** A locality-preserving relabel
   (RCM) concentrates each node's K-hop neighbourhood into contiguous blocks, so
   more blocks become fully-skippable. The reorder is O(V+E), computed once in
   preprocessing, and is negligible vs the O(N²)–O(N³) SPD/magnetic precompute that
   must happen anyway.
5. **End result (Llama-3.2-1B, L≈2.3k, spd+magnetic, k=2):**
   `eager` 1607 ms / 20 GB (fwd+bwd) → `flex` **197 ms / 10 GB** (8× faster,
   ½ memory), within 1.5× of the bias-free `flash` floor (133 ms). At larger L,
   `eager` simply **OOMs** where flex runs.

## Files

| File | Purpose |
|---|---|
| `inputs.py` | Synthetic graph-batch generator. Real topology (random-geometric → recoverable locality) + real K-hop/SPD; **synthetic** magnetic eigenvectors (shape-faithful — values don't affect timing). RCM/random/identity node ordering. Packs via the real `GraphCollatorV2`. |
| `density.py` | Element-level vs **block-level** mask coverage, chunked so it never materializes `L×L`. Block density predicts flex speedup (`~1/block_density`). |
| `flex_core.py` | The real flex implementation: `pad_to_block`, `build_block_mask` (mask_mod → `BlockMask`), `make_score_mod` (node-bias gather), `flex_attention_forward`, and a `dense_reference` for parity. Drops into the `graph_attention_v2.py` flex scaffold. |
| `bench_isolation.py` | One attention layer, 3 methods (`eager` dense SDPA / `flex` / `flash` floor). Times fwd and fwd+bwd separately + peak memory. Correct input-grad lifecycle (fresh leaves per step). |
| `bench_full_model.py` | Llama-3.2-1B fwd+bwd, same 3 methods. flex is routed at runtime (monkeypatch) — no edits to the model source. |
| `run_sweep.py` | Drives the 2D (`nodes × tokens/node`) sweep × ordering × K-hop × method, each `(method,config)` in a **fresh subprocess** (OOM isolation). Also `--recompile-probe`. |
| `profile_ncu.py` | Nsight Compute roofline: per-kernel DRAM% vs compute% (memory- vs compute-bound), fwd vs bwd. |

## Running

```bash
# from repo root; modules use `src.models.flex_attn.*`

# single isolation run (one method, one config)
python -m src.models.flex_attn.bench_isolation \
    --method flex --n-nodes 512 --tokens-per-node 16 --k-hop 2 --ordering rcm

# single full-model run
python -m src.models.flex_attn.bench_full_model \
    --method flex --n-nodes 128 --tokens-per-node 16 --k-hop 2

# the full sweep (subprocess-isolated). --out-dir is a folder; the sweep writes
# {kind}.jsonl (full detail) and {kind}.md (a readable pivot table) into it.
python -m src.models.flex_attn.run_sweep --kind isolation \
    --methods flash eager flex --k-hops 0 2 4 --orderings rcm random \
    --out-dir src/models/flex_attn/results

python -m src.models.flex_attn.run_sweep --kind full_model \
    --methods flash eager flex --k-hops 0 2 4 \
    --out-dir src/models/flex_attn/results

# regenerate the markdown table from a saved JSONL (writes <name>.md beside it)
python -m src.models.flex_attn.run_sweep --summarize src/models/flex_attn/results/isolation.jsonl

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

## Reading the results

A sweep produces two artifacts in `--out-dir`:

- **`{kind}.md`** — the at-a-glance pivot table: one row per config, with each
  method's `ms` (forward+backward latency) and `GB` (peak memory) as separate
  columns, plus `tokSp`/`blkSp` (token-/block-level sparsity) and `L`. `OOM` and
  error tags appear in the method's `ms` column. Regenerate anytime with
  `--summarize`.
- **`{kind}.jsonl`** — the full detail, one record per line, for analysis
  (`pandas.json_normalize(map(json.loads, open(path)))`).

A single direct run (`bench_isolation` / `bench_full_model`) prints the same
record as pretty JSON between `===RESULT_JSON_BEGIN===` / `===RESULT_JSON_END===`
sentinels. Each record is grouped into:

- `spec` — the swept graph parameters (`n_nodes`, `tokens_per_node`, `k_hop`, `ordering`, …).
- `run` — kernel/hardware knobs (`num_heads`, `head_dim`, `block_size`, `dynamic`, …).
- `shape` — realized sizes after jitter (`seq_len`, `max_num_nodes`, `total_tokens`).
- `density` — `element_density`/`element_sparsity` (token level) and
  `block_density`/`block_sparsity` (block level — what flex skips), `expected_attn_speedup`.
- `timing_ms` — `fwd`, `fwd_bwd`, `bwd_est`, plus `compile` (flex one-time),
  `blockmask_build` (per batch), `reorder_mean`/`reorder_max` (the RCM cost).
- `memory_mb` — `peak_fwd`, `peak_fwd_bwd`.
- `ok` / `error` — `OOM`, `TIMEOUT`, `CRASH`, or an exception name.

**Interpreting the memory-vs-compute split (`profile_ncu.py`).** Memory and
compute overlap on-GPU, so there is no clean millisecond split. ncu reports, per
kernel, achieved DRAM throughput and SM (compute) throughput as % of peak: mem%
≫ compute% ⇒ memory-bound, and vice-versa. We attribute kernels to fwd vs bwd by
running both phases; the backward of the flex path includes the scatter-add into
`node_bias` (the gather's transpose) — watch its mem%.

On this box **ncu is blocked by DCGM** (`nv-hostengine` holds the counters), so
use `--analytic`: it compares each config's arithmetic intensity (FLOP/byte) to
the A100 bf16 ridge (~156). Measured result: the graph-attention forward is
**memory-bound** across the sweep (AI ≈ 74–124) — denser masks (k=0) sit closest
to the ridge, sparser K-hop pushes it further memory-bound. So flex's win is
bandwidth/footprint, and the backward `node_bias` scatter (a memory op) is the
cost to watch, consistent with the isolation timings.

## Notes / caveats

- The `flash` method is the bias-free **floor**, not a functional equivalent —
  FlashAttention cannot express the graph bias, bidirectional prefix, or K-hop.
  (`flash_attn` isn't installed here, so it's PyTorch SDPA's flash kernel.)
- Magnetic eigenvectors are synthesized (shapes are real). To benchmark with true
  spectral values, swap `inputs._build_items` to call the real
  `get_magnetic_laplacian_coords`.
- **bf16 bug fixed in passing:** `MagneticBias` divided by `valid.float()` (fp32),
  which broke the magnetic bias under bf16 (`mat1/mat2 dtype` error). Fixed to
  `.to(h_i.dtype)` in `src/models/graph_bias.py` — fp32-neutral, required for
  bf16 training with magnetic enabled.
- If `profile_ncu.py` reports "ncu produced no CSV", the profiler lacks GPU
  performance-counter permission (needs `CAP_SYS_ADMIN` / admin-enabled counters).
