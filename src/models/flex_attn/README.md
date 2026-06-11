# FlexAttention benchmarks for GTLM

Does `torch` FlexAttention beat the current dense graph-attention path on compute
and memory — especially with sparse K-hop masks — while preserving the SPD and
magnetic-Laplacian biases? This suite answers that with measurements on an A100_80GB GPU.

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
