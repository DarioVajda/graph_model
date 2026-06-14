# GTLM models

For now, only the `Llama-3` models were modified for GTLM.

---

# FlexAttention integration — working plan & progress log

This section is a **living scratchpad** for integrating the FlexAttention path
(validated in `src/models/flex_attn/`, see its README) into the production v2
model. It is updated as work proceeds.

## Decisions (locked)

- **Kernel home:** promote the core of `flex_attn/flex_core.py` into a
  model-level module `src/models/flex_kernel.py`. The bench keeps working via a
  re-export shim.
- **Decode path:** flex fires only on full-sequence (`q_len == kv_len`:
  training + prefill). Incremental decode (`q_len < kv_len`) falls back to the
  existing dense `sdpa`/`eager` path. Generation stays untouched.
- **Padding + bucketing:** in `GraphCollatorV2` (opt-in `pad_to_block`). Bucket
  **both L and N** — both drive flex recompiles (the kernel guards on the
  `(B,H,N,N)` bias shape, confirmed empirically). Defaults: L → 512-multiple +
  midpoint ladder; N → power-of-two floored at 32. Custom `len_buckets` /
  `node_buckets` (None | list | callable) override. L buckets must be multiples
  of the kernel `block_size` (64/128).
- **Recompile cap:** dynamo `cache_size_limit` defaults to **8** — past 8
  distinct (L,N) shapes the flex frame silently falls back to eager. The model
  raises it to `config.flex_cache_size_limit` (default 32) on the flex path.
  Keep the realistic co-occurring (L,N) pair count below it (~≤16 target).
- **RCM node ordering:** computed in the **dataset class**
  (`TextGraphDataset`) — a method that computes the per-graph RCM permutation,
  plus an `__init__` flag that precomputes/applies it across the whole dataset.
- **Compile mode:** default `max-autotune-no-cudagraphs` (#4), exposed as a
  configurable `GTLMLlamaConfig.flex_compile_mode`.
- **Block size:** `flex_block_size(k_hop)` = 64 when `k_hop>0` else 128 (#6);
  overridable via `GTLMLlamaConfig.flex_block_size`.
- **node_ids dtype:** int32 copy captured for the flex gather + BlockMask (#10).
- **Default `graph_attn_impl`:** stays `"eager"` — flex is strictly opt-in.
- **Compiled-kernel cache shipping:** bundle the compiled flex kernels with the
  model so they're reused on load (local + HF Hub). Route is **version-adaptive**:
  torch≥2.7 uses the portable `torch.compiler.save/load_cache_artifacts` blob;
  torch 2.6 falls back to tar-bundling the `TORCHINDUCTOR_CACHE_DIR` directory.
  An env-fingerprint (`torch`/`triton`/`cuda`/GPU-arch) gates load — **mismatch
  → warn + skip, never fatal** (compiled kernels are arch+version specific, so
  the bundle is a fast path only when the env matches; otherwise it recompiles).
  Cache population is via an **opt-in warmup helper** (compile each (L,N) bucket)
  the user calls before saving — `save_pretrained` does not auto-compile.

## Out of scope (parked)
- #11 per-KV-head bias (needs a training ablation).
- #12 chunked cross-entropy (the L≈70k wall; orthogonal).

## Commit order (stop for review after each step)

1. **Promote kernel + shim** — new `src/models/flex_kernel.py`; `flex_core.py`
   becomes a re-export shim. No behavior change. ✅
2. **Scaffold impl + config** — implement the `graph_attention_v2.py` flex
   stubs over `flex_kernel`; add `flex_compile_mode` / `flex_block_size` config
   fields + `flex_block_size()` gate helper. ✅
3. **Model forward + attention branch** — `GTLMLlamaForCausalLM.forward` builds
   a per-batch `BlockMask` (+ int32 node_ids) for full-seq flex and falls back
   to dense for decode; `GTLMLlamaAttention.forward` gets the flex branch
   (node-level bias + score_mod), preserving #7/#9 checkpointing. ✅
4. **Collator padding + dataset RCM** — `GraphCollatorV2` opt-in bucket padding;
   `TextGraphDataset` RCM method + precompute flag. ✅
5. **Tests** — model parity (flex vs eager fwd + grads, k=0/k=2), ckpt nesting
   (#7/#9), generation smoke (dense fallback), plus collator L/N bucketing and
   dataset RCM/ordering unit tests. ✅
6. **Compiled-kernel cache shipping** — version-adaptive `save_compile_cache` /
   `load_compile_cache` (blob on torch≥2.7, dir-tar on 2.6) with env-fingerprint
   gating; hooked into `save_pretrained` / `from_pretrained` (non-fatal, Hub-
   compatible); opt-in `warm_flex_cache(buckets)` helper; round-trip tests. ⬜

## Progress log

- **Step 1 done.** Added `src/models/flex_kernel.py` (production kernel core:
  alignment/bucketing, `make_mask_mod`/`build_block_mask`, `make_score_mod`,
  `flex_attention_forward`, `DEFAULT_COMPILE_MODE`, and the new
  `flex_block_size(k_hop)` gate helper). Rewrote `flex_attn/flex_core.py` as a
  re-export shim that keeps the bench-only `dense_reference` + self-test.
  Verified: imports resolve, bench symbols intact, no dangling refs.
- **Step 2 done.** Implemented the `graph_attention_v2.py` flex seams as thin
  wrappers over `flex_kernel`: `build_flex_block_mask`, `make_soft_score_mod`,
  `flex_attention_forward` (returns `(B,q,H,d)` + `None` to match the dense
  backends), plus a `flex_block_size()` re-export. `graph_attention_dispatch`
  now rejects `impl='flex'` with a clear message (flex is routed from the
  attention forward in step 3). Added `GTLMLlamaConfig.flex_compile_mode`
  (default autotune) and `flex_block_size` (None = K-hop gate). Verified:
  wrappers callable, dispatch guard fires, config defaults + serialization
  round-trip, gate returns 64/128.
- **Step 3 done.** `GTLMLlamaForCausalLM.forward`: when `impl=="flex"` and
  `q_len==kv_len`, casts `node_ids` to int32 (#10), resolves block size from the
  K-hop gate (or `config.flex_block_size`), asserts L-alignment, and builds the
  shared per-batch `BlockMask`; otherwise builds the dense mask, with flex decode
  (`q_len<kv_len`) downgraded to `sdpa`. `ctx` now carries `node_ids_flex` +
  `block_mask`. `GTLMLlamaAttention.forward` branches on `ctx["impl"]`: flex
  builds the score_mod from node-level bias and calls the flex kernel (#7 bias
  checkpoint still wraps the bias compute); dense path unchanged. Smoke-tested
  on H100 (real Llama-3.2-1B, spd+magnetic): flex vs eager loss 13.50 vs 13.50
  (bf16, rel|Δlogits|≈0.04), 9/9 finite layer-0 bias grads, and `generate()`
  runs the dense decode fallback. Rigorous fp32 parity deferred to step 5.
- **Step 4 done.** `GraphCollatorV2` gained `pad_to_block` / `block_size`: when
  on, the packed length is raised via `bucket_len` (the existing pad defaults are
  already correct for the extra positions, so only the alloc length changes —
  zero change to the dense path). `TextGraphDataset` gained `_rcm_relabel_graph`
  (RCM reorder, remaps `prompt_node`, keeps `original_id`), `apply_rcm_ordering()`
  (post-construction, rebuilds `text`/`prompt_node`, raises if node-indexed
  features already exist), and an `rcm_ordering=True` `__init__` flag (reorders
  before features so they inherit the order). Verified: bucketed L is
  128-aligned and real positions / labels are byte-identical to the unpadded
  collation; RCM preserves node/edge counts, remaps prompt in range, original_id
  stays a permutation, the guard fires when features are present.
- **Step 4 follow-up (persistent ordering label).** `TextGraphDataset` now
  records `node_ordering` (`"original"`/`"rcm"`/`"mixed"`, via `ORDERING_*`
  constants) with an `is_rcm_ordered` property. Set by the `rcm_ordering` flag
  and `apply_rcm_ordering()`; **persisted in `metadata.json`** (legacy datasets
  default to `"original"`); propagated through `select()` (carry) and `__add__`
  (`"mixed"` if the two sides differ); shown in `__repr__`. So a reloaded
  dataset always knows its node ordering. Verified end-to-end: construct/flag/
  method → save → reload round-trips the label; select carries it; merge yields
  rcm/mixed correctly.
- **Step 4 follow-up (L+N bucketing & recompile cap).** Confirmed empirically
  that N recompiles the flex kernel (guard failure on the captured `node_bias`
  N dim). Fixes: `flex_kernel` gained `bucketize(value, spec)` (None | list |
  callable) + `default_len_buckets` (×512 midpoint) / `default_node_buckets`
  (pow2 floored at 32). `GraphCollatorV2` now buckets **both** L and N under
  `pad_to_block`, with overridable `len_buckets` / `node_buckets` and an L-bucket
  alignment guard; `_collate_features` + k_hop_mask pad to the N bucket. Added
  `GTLMLlamaConfig.flex_cache_size_limit` (32); the forward raises dynamo's
  `cache_size_limit` to it on the flex path (only ever upward). Verified: ladders,
  bucketize forms + overflow error, config round-trip, features/k_hop padded to
  the N bucket, custom specs, alignment guard, and the dynamo raise.
- **Step 5 done.** Three test files (95 tests pass total, no regressions):
  `test_dataset_ordering.py` (RCM relabel/structure, init flag + method, feature
  guard, select/add/save-load label propagation), `test_collator_bucketing.py`
  (ladders, bucketize, L+N padding, real-position preservation, custom specs,
  alignment guard, and an **fp64 eager loss-neutrality** check at atol=1e-9),
  and `test_flex_attention.py` (GPU-gated: flex-vs-eager forward + bias-grad
  parity at k=0/2, #7 and #9 checkpoint parity, decode-fallback generation;
  forced block_size=128 + default compile mode + a fixed (L,N) bucket to stay
  fast — 22s cold, faster warm via inductor's on-disk cache).
- **Step 5 follow-up (permutation invariance + a real RCM bug it caught).**
  Added `test_flex_permutation_invariance_via_rcm`: reorder the prefix via the
  dataset's RCM and assert the **prompt-span logits are unchanged**. Writing it
  surfaced a genuine bug: `nx.relabel_nodes` keeps the original *insertion*
  order, so after RCM `g.nodes()` was not `0..N-1` in order, and feature
  computations that use `nx.to_numpy_array` (SPD/RRWP/magnetic) came out
  misaligned with the node labels the model indexes by — silently corrupting
  every RCM run. Fixed `_rcm_relabel_graph` to rebuild the graph in sorted-label
  order. Notes on the property tested: the prompt node is always packed last, so
  RCM relabeling it is output-invariant (verified bit-identical); the model's
  prompt representation is permutation-invariant over the prefix; only the
  first-prompt-token *loss* shifts (causal teacher-forcing boundary), which is
  never a prediction target — so the test compares prompt logits, not loss.
