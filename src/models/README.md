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
- **Padding + bucketing:** in `GraphCollatorV2` (opt-in flag). Pad to a 128
  multiple via the `bucket_len` ladder (128-aligned is also 64-divisible, safe
  for both block sizes).
- **RCM node ordering:** computed in the **dataset class**
  (`TextGraphDataset`) — a method that computes the per-graph RCM permutation,
  plus an `__init__` flag that precomputes/applies it across the whole dataset.
- **Compile mode:** default `max-autotune-no-cudagraphs` (#4), exposed as a
  configurable `GTLMLlamaConfig.flex_compile_mode`.
- **Block size:** `flex_block_size(k_hop)` = 64 when `k_hop>0` else 128 (#6);
  overridable via `GTLMLlamaConfig.flex_block_size`.
- **node_ids dtype:** int32 copy captured for the flex gather + BlockMask (#10).
- **Default `graph_attn_impl`:** stays `"eager"` — flex is strictly opt-in.

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
   (node-level bias + score_mod), preserving #7/#9 checkpointing. ⬜
4. **Collator padding + dataset RCM** — `GraphCollatorV2` opt-in bucket padding;
   `TextGraphDataset` RCM method + precompute flag. ⬜
5. **Tests** — model parity (flex vs eager fwd + grads, k=0/k=2), ckpt nesting
   (#7/#9), generation smoke (dense fallback), plus collator padding and dataset
   RCM unit tests. ⬜

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
