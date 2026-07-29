# Rebuttal note — the "eager attention only" limitation is resolved

Planning + evidence for the NeurIPS rebuttal on the FlexAttention backend, which
did not exist at submission time. The submitted paper states that GTLM is forced
onto eager attention and that training is correspondingly slow. That limitation
is now gone, and this note is where we decide exactly what to claim, on what
evidence, and — as important — what *not* to claim.

Everything here is measured on **one H100 80GB** (`ixh`). Numbers from different
GPUs are never mixed: `results_a100/` is the archived A100 study, `results_h100*/`
is everything quoted below.

---

## 1. What the paper says today

The limitations paragraph concedes that the graph bias makes the fused attention
kernels ineligible, so every experiment ran on a dense `eager` path that
materializes a `(B, H, L, L)` token-level bias per layer, and that this bounds
the sequence lengths we can train on.

Two distinct reviewer concerns hide behind that sentence:

1. **"Your method is impractically slow."** — an efficiency objection.
2. **"Your method cannot scale to realistic graph inputs."** — a capability
   objection, and the more damaging one, because it suggests the architecture is
   a toy.

The rebuttal must answer both, and they want *different* evidence. Conflating
them is the main failure mode here.

---

## 2. The three tiers to present

### Tier 1 — the exact speedup, on the paper's own benchmarks

**This is the headline and it must come first.** Reviewers discount synthetic
microbenchmarks; they do not discount "the experiment in your Table 2 now runs
N× faster." Measured by `bench_real.py` on the real cached TAG splits, with the
paper's *selected* per-dataset configuration (Table 8), the production collator,
LoRA at the paper's rank, and `k_hop=0` as the paper used.

Three arms, all fed byte-identical token tensors:

| arm | what it is |
|---|---|
| `eager` | the backend every number in the submitted paper was produced on |
| `flex` | the same model and weights on the new kernel |
| `flex-nobias` | flex kernel + graph mask, no bias modules — splits the residual gap |
| `sdpa` | stock `LlamaForCausalLM`, PyTorch SDPA (fused flash), same `input_ids` — a plain LLM at equal sequence length, no graph structure |

The `sdpa` arm is the one that turns a defensive answer into an offensive one.
"We got faster" invites "faster than your own slow baseline, so what?". Dividing
every arm through by that floor is what produced the framing the rebuttal should
actually lead with — see **§5, "The unifying result"**: GTLM has a ~2× floor over
a plain LLM which is bias computation, eager diverged from it without bound as L
grew, and flex restores it.

→ Results in §5.

### Tier 2 — capability, not just speed: the scaling study

The synthetic sweep is the right instrument for the *capability* objection,
because it can reach sequence lengths the TAG benchmarks never produce. One
figure, two panels (already generated — `results_h100/fig_isolation_all_methods.png`):

- **latency vs L** — `eager` terminates at L≈4K (OOM past it); `flex` runs to
  L≈256K, and with the K-hop mask at k=2 it crosses *below* the bias-free causal
  flash floor at L≈64K.
- **peak memory vs L** — `eager` is at 30 GB by L=4.5K; `flex` is at 0.2 GB for
  the same work, tracking flash almost exactly.

The claim this supports: *the architecture was never intrinsically expensive —
the dense bias materialization was. Removing it puts graph-biased attention on
the same asymptote as ordinary flash attention.*

Supporting table (H100, isolated attention layer, fwd+bwd ms — from
`results_h100/isolation.md`):

| L | `eager` | `flex` (k=0) | `flex` (k=2) | `flash` floor |
|---|---|---|---|---|
| 1,216 | 21.5 | 3.0 | 2.1 | 0.5 |
| 4,478 | 263.8 | 26.9 | 15.3 | 2.0 |
| 17,506 | **OOM** | 411.6 | 110.0 | 22.4 |
| 69,121 | **OOM** | 5,511.5 | 282.7 | 330.6 |
| 276,067 | **OOM** | timeout | 1,834.0 | 5,404.4 |

### Tier 3 — the results did not change

A reviewer's immediate follow-up to any "we made it faster" claim is "so are the
reported numbers still valid?". Answer it before it is asked, with the parity
tests rather than with prose:

Re-run on the H100 on 2026-07-26 (job 117591): **11 passed** in
`tests/models/test_flex_attention.py`, **39 passed** in
`tests/models/test_modeling_gtlm_llama_v2.py`.

- `::test_flex_matches_eager_forward` — flex output vs the dense eager path,
  over k_hop ∈ {0, 2} × {no bias, **all biases on**}. Note the bias axis is
  none/all, not one case per bias type; "all on" is the paper's configuration,
  so state it that way rather than as "every bias type".
- `::test_flex_bias_grad_parity` — gradient parity on the bias parameters,
  k_hop ∈ {0, 2}.
- `::test_flex_checkpoint_bias_parity`, `::test_flex_decoder_ckpt_parity` — both
  checkpointing modes leave loss and gradients unchanged.
- `::test_flex_permutation_invariance_via_rcm` — the RCM reordering flex relies
  on does not change the function computed.
- `::test_flex_generation_dense_fallback` — decode (q_len < kv_len) still routes
  to the dense path, so evaluation is untouched.

So: flex is an implementation of the same function, agreeing to kernel
accumulation-order tolerance. **No number in the paper changes.** Flex is a
speed result, not a results result — say so explicitly.

---

## 3. Guardrails — claims to avoid

These are the ways this rebuttal could get us caught, in decreasing order of risk.

1. **Do not quote the k=2 speedups as if they described the paper's runs.**
   The headline "3× faster than the bias-free flash floor" is a **k_hop=2**
   number. Every TAG experiment in the paper runs **k_hop=0**, where there is no
   K-hop block sparsity to exploit and flex's win comes from *not materializing
   the `(B,H,L,L)` bias* plus kernel fusion — a different mechanism. Keep the two
   labelled separately in every table. A reviewer who notices us blurring them
   will discount the whole rebuttal.

2. **Do not present the scaling figure as a benchmark result.** It is synthetic
   (real topology and real SPD/K-hop, but synthetic magnetic eigenvector *values*
   — shape-faithful, which is all that timing depends on). Say "synthetic
   graph batches" in the caption. It is evidence about the kernel, not about
   Cora.

3. **The TAG benchmarks are a short-sequence regime.** Measured true (unpadded)
   tokens per graph on the sampled batches: Cora 663, PubMed 950, Reddit 976,
   ogbn-arxiv 1,065. That is 1–2 orders of magnitude below where the scaling
   study gets dramatic. Quote *true* length, not the padded tensor width the
   harness reports as `seq_len_mean` (917–1,387) — they differ by 7–27%.
   This cuts *for* us if we frame it honestly — the Tier-1 speedup is achieved in
   the regime least favourable to flex — and against us if a reviewer discovers
   we implied the 64K numbers describe Table 2.

4. **Report the compile cost.** `max-autotune-no-cudagraphs` costs a one-time
   compile per distinct (L, N) bucket. It is amortized over thousands of steps
   and the collator's bucketing bounds the bucket count to ~4 (Cora) or ~10, but a reviewer
   running the code will hit it, and volunteering it is cheap credibility.

5. **Do not claim we can now train on 256K-token graphs.** The isolation bench
   reaches those lengths; the *full model* hits a separate wall (unchunked LM-head
   cross-entropy, ~90 GB of logits at L=70K). Claim what the full model does.

6. **Do not rewrite the paper's limitations section into a victory lap.** The
   honest revision is: the limitation was implementation-level, we have since
   removed it, here is the measurement — with the accuracy numbers unchanged.

---

## 4. What to actually put in the rebuttal

Space is tight, so in priority order:

1. **One table** — Tier 1, the four TAG datasets × {eager, flex, sdpa}, step
   latency + peak memory + speedup. Two sentences of setup.
2. **One sentence** on parity, citing the test names.
3. **One figure** — Tier 2, the two-panel latency/memory scaling plot, captioned
   as synthetic.
4. **One sentence** conceding the compile cost and the short-L regime.
5. **A proposed one-paragraph edit to the limitations section** for the camera
   ready, quoted verbatim so reviewers can see exactly what changes.

---

### Deliberately out of scope

**A `k_hop>0` arm.** Flex makes the K-hop attention gate cheap — that is where
the block sparsity actually pays, and it is genuinely new capability. But every
result in the paper uses `k_hop=0`, so a k>0 arm would be a *different model*
whose accuracy we have not measured. Offering it in a rebuttal invites "so what
does it score?", which we cannot answer in the rebuttal window. Mention K-hop
only as kernel-level scaling evidence (Tier 2), never as a model result.

**Retraining anything.** The rebuttal claim is about cost, not accuracy. Parity
tests carry the "results unchanged" argument; retraining to re-confirm it would
burn the window and invite fresh variance questions.

---

## 5. Measurements

### Protocol

`python -m src.models.flex_attn.bench_tag --dataset <ds> --methods eager flex sdpa
--n-batches 24 --passes 3` (submit with `sbatch_bench_tag.sh H100 <ds>`).

24 real batches, strided across the split so the length distribution is
representative; one discarded cold pass (compile/autotune) then 3 timed passes,
per-step CUDA-event timing, forward+backward, no optimizer step (the paper
accumulates 32 steps, so it is ~1/32 of cost and identical across arms). All arms
share one collator, so bucket padding — which flex requires — is charged to every
arm and L is exactly equal across methods. Run twice per dataset: with gradient
checkpointing (the paper's setting) and without (isolates the attention backend
from recompute).

### Results — all four TAG datasets (H100, paper config, gradient checkpointing on)

Step latency, ms, 24 real train batches, B=1, `k_hop=0`, bf16, all biases on.
All numbers post-fix (see "Measurement corrections" below).

| dataset | L real | `eager` | `flex` | **speedup** | `flex-nobias` | `sdpa+mask` | `sdpa` | flex/sdpa |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| cora | 663 | 836.9 | 226.8 | **3.69×** | 143.0 | 107.0 | 118.6 | 1.91× |
| pubmed | 950 | 1,255.3 | 215.9 | **5.81×** | 137.2 | 113.3 | 112.1 | 1.93× |
| reddit | 976 | 406.3 | 201.3 | **2.02×** | 125.5 | 103.6 | 103.6 | 1.94× |
| ogbn-arxiv | 1,065 | 2,006.2 | 208.3 | **9.63×** | 126.8 | 107.5 | 103.7 | 2.01× |

**The regularity is the story, not the average.** Flex lands at 201–227 ms and
`sdpa` at 104–119 ms on *every* dataset; only `eager` moves, from 406 to 2,006 ms.

- **Flex's cost is bounded and nearly dataset-independent; eager's is not.** The
  spread in "speedup" (2.0×–9.6×) is a statement about how badly eager degrades
  with sequence length, not about how variable flex is.
- **Flex sits at a stable 1.91–2.01× the plain-LLM floor on all four.** That is
  the robust claim. Quote the range; *never* average the four speedups into one
  number — the average is an artifact of which datasets we happen to report.
- Do not lead with 9.63×. Lead with the table and the ~2× line; a single
  cherry-picked multiplier is what a sceptical reviewer discounts first.

**Wall clock, one epoch** (training steps only, no eval): ogbn-arxiv **50.7 h →
5.3 h**, pubmed 4.12 h → 0.71 h, reddit 0.38 h → 0.19 h, cora 0.38 h → 0.10 h.

**Where the residual ~2× over a plain LLM goes** (reddit, the clearest case):

| | step ms | Δ over previous |
|---|--:|--:|
| `sdpa` — plain Llama-3.2-1B, plain causal | 103.6 | — |
| `sdpa+mask` — + GTLM's structural mask | 103.6 | **+0.0 ms** |
| `flex-nobias` — + flex kernel | 125.5 | +21.9 ms |
| `flex` — + per-layer bias modules | 201.3 | **+75.8 ms** |
| `eager` — dense `(B,H,L,L)` instead of flex | 406.3 | +205.0 ms |

Two things to carry into the rebuttal:

1. **The mask is free.** `sdpa+mask` costs 0.90–1.04× plain causal across the four
   datasets even though GTLM's mask admits 0.70 of the L×L matrix against causal's
   0.50, and even though an arbitrary mask makes flash ineligible. At L≈1,000
   attention is a small share of a 1B model's step. So the ~2× is *not* the price
   of bidirectional prefix attention.
2. **It is bias-module compute.** ~78% of the gap. `flex-nobias/sdpa` is 1.21–1.22×
   on every single dataset — mask + kernel overhead is essentially a constant, and
   everything above it is the bias MLPs. Flex was never meant to address that.
   "Why is flex still 2× sdpa?" is the obvious follow-up; this answers it in a row.

### Cora in detail, and the gradient-checkpointing arm

| method | step ms | vs `eager` | vs `sdpa` | peak GB |
|---|--:|--:|--:|--:|
| `eager` — the submitted paper's path | 836.9 | 1.00× | 7.06× | 6.33 |
| `flex` | **226.8** | **3.69×** | 1.91× | 4.85 |
| `sdpa` — plain Llama-3.2-1B, equal L | 118.6 | 7.06× | 1.00× | 4.85 |

**Eager's step time is far more variable than flex's** (cora: eager ±465 ms,
flex ±37 ms in the pre-fix run). Eager's cost is ~quadratic in L, so it swings
~9× across the batch-length distribution; flex is nearly flat. Beyond the mean,
flex makes step time *predictable*, which is what makes bucketing and batch-size
selection tractable at all.

**Without gradient checkpointing** the backend-only speedup is larger (cora
5.98×, pubmed 8.82× in the pre-fix runs) because checkpointing adds a fixed
recompute cost that dilutes the attention win. Quote the checkpointed numbers as
the headline — that is what the paper's configuration runs — and the others only
if a reviewer asks specifically about the attention implementation.

**The compile cost is cacheable, not per-run.** Flex's cold pass was 176 s on
first encounter but 12 s on a second run in the same job, off inductor's on-disk
cache. Cora compiles 4 distinct (L, N) shapes, not the 19 an early version of the
harness reported.

### The unifying result — normalize everything to the plain-LLM floor

Every arm was also run as stock `LlamaForCausalLM` + SDPA on identical tokens.
Dividing through by that floor is what makes TAG and GraphQA one story rather
than two:

| regime | L | `eager`/sdpa | `flex`/sdpa |
|---|--:|--:|--:|
| graphqa standard/node_count | 28 | 2.14× | 2.46× |
| graphqa standard/shortest_path | 37 | 2.23× | 2.45× |
| graphqa incidence/node_count | 138 | 2.11× | 2.49× |
| graphqa incidence/shortest_path | 147 | 2.29× | 2.47× |
| TAG cora | 663 | 7.06× | 1.91× |
| TAG pubmed | 950 | 11.20× | 1.93× |
| TAG reddit | 976 | 3.92× | 1.94× |
| TAG ogbn-arxiv | 1,065 | 19.35× | 2.01× |

(All rows bf16, B=1 — GraphQA re-measured off its fp32/B=4 paper recipe so the
rows are actually comparable. See "Measurement corrections".)

**GTLM has an irreducible floor of ~2× a plain LLM: the per-layer bias-module
compute** — the same term `flex-nobias` isolates (+78.7 ms of reddit's +106.7 ms
gap). The whole result then reads as one sentence:

> On short sequences the dense path already sits at that floor, so there is
> nothing to recover. On long sequences it diverges from the floor without bound
> — up to 19× — and flex is what pulls the cost back down to it.

This is the framing to use. It converts "we made it faster" into a structural
claim: *the graph bias costs ~2× a plain LLM, inherently and independent of
length; the dense implementation added an unbounded length-dependent penalty on
top, and flex removes that penalty.* The 1.7×–9.3× speedup spread becomes a
derived quantity — how far eager had drifted from a floor that was always ~2× —
rather than a headline to defend.

**Caveat to state, not hide.** GraphQA is fp32, TAG is bf16, and the floor reads
2.31–2.36× vs 1.96–2.07×. The gap is plausibly dtype (fp32 makes the bias MLPs
relatively dearer), but that has *not* been isolated. Quote the floor as "~2×";
do not present the two as the same measurement.

### Results — GraphQA: where flex stops paying

Same harness, GraphQA's own recipe (fp32, B=4, LoRA r=16, no checkpointing,
`k_hop=0`). Two encodings × two tasks. **Do not quote these as speedups** — the
point is the regime boundary.

Flex requires block-aligned L and the block size is 128, so a 33-token GraphQA
sequence is padded ~4× *however tight the ladder*. Charging `eager` that padding
would be measuring flex's constraint, not eager's cost, so the dense arms were
also run at their natural per-batch L (`--pad-mode batch`). The production
question is answered by **flex@bucket-128 vs eager@batch**:

| arm | natural L | `eager` @natural | `flex` @L=128 | verdict |
|---|--:|--:|--:|---|
| standard/node_count | 29 | **101.5** | 114.0 | flex 12% slower |
| standard/shortest_path | 38 | **101.8** | 113.4 | flex 11% slower |
| incidence/node_count | 137 | 362.0 | **319.4** | flex 13% faster |
| incidence/shortest_path | 147 | 372.8 | **319.1** | flex 17% faster |

**That table is GraphQA's own fp32 / B=4 recipe. Normalized to bf16 / B=1 — the
settings TAG uses, and the only basis on which the two experiments can be put in
one table — flex loses on every arm, incidence included:**

| arm | L | `eager` | `flex` | `flex-nobias` | `sdpa` | flex vs eager |
|---|--:|--:|--:|--:|--:|--:|
| standard/node_count | 28 | 112.3 | 129.1 | 69.5 | 52.5 | 0.87× |
| standard/shortest_path | 37 | 113.0 | 124.0 | 66.2 | 50.7 | 0.91× |
| incidence/node_count | 138 | 115.1 | 135.7 | 73.6 | 54.5 | 0.85× |
| incidence/shortest_path | 147 | 112.9 | 121.8 | 67.1 | 49.3 | 0.93× |

Two effects flip incidence: B=1 cuts it to 208 tokens/step (from 1,301), and bf16
makes eager's dense bias far cheaper, removing what flex was recovering. Note
`eager` sits at 112–115 ms on all four arms regardless of L — this regime is
launch-bound, and attention is not what is being measured.

**The crossover is somewhere between L ≈ 150 and L ≈ 660.** We cannot place it
more tightly: there is no dataset in that gap. Do **not** repeat the earlier
"L ≈ 150–250" estimate — it came from the fp32/B=4 measurement and does not
survive normalization.

**Why the equal-L view is a trap here.** At padded L=128 flex reads as 1.13×
*faster* than eager on `standard` — the opposite sign from the truth, because the
padding is charged to both arms. For TAG this distinction was immaterial (5–27%
padding); at L=33 it inverts the conclusion. Both tasks agree within each
encoding, so this is not noise. If we ever show a GraphQA number, it must be the
natural-L one.

**Use this affirmatively.** It shows backend selection is by regime, not a blanket
swap, and it independently confirms the default already shipped in the code
(`graphqa` → `v2-eager`, `tag_benchmarks` → `v2-flex`). A rebuttal that says
"flex everywhere, always faster" is easy to attack; one that says "here is the
crossover, and our defaults sit on the right side of it" is not.

**Operational note (fp32).** The incidence arms needed ~45 min of autotuning
(7 distinct (L,N) shapes × 2 flex arms) against 10–176 s for TAG in bf16: in fp32
many Triton candidates exceed the H100's 232 KB shared-memory budget and get
pruned, so the autotuner searches far longer. A GraphQA flex path would want bf16
or `--compile-mode default`.

### Reproducing

```bash
./src/models/flex_attn/sbatch_bench_tag.sh H100 cora   # one TAG dataset
./src/models/flex_attn/sbatch_bench_tag_rest.sh H100   # the rest + parity tests
./src/models/flex_attn/sbatch_bench_graphqa.sh H100    # both GraphQA encodings
python -m src.models.flex_attn.report_real \
    --results-dir src/models/flex_attn/results_h100_tag   # consolidated table
```

Raw records: `results_h100_tag/{tag,graphqa}.jsonl`, per-experiment tables in
`{tag,graphqa}.md`, one cross-experiment table in `summary.md`.

---

## 6. Draft replacement for the limitations paragraph (camera ready)

Quote this in the rebuttal so reviewers see the exact edit. `<S>` / `<P>` are
filled from §5.

> **Attention efficiency.** The learned graph bias is a per-head, per-layer
> additive term on the attention logits, which makes it ineligible for the fused
> attention kernels that assume no arbitrary bias. At submission we therefore ran
> every experiment on a dense implementation that materializes the bias at token
> level, `(B, H, L, L)` per layer. We have since implemented the bias as a
> FlexAttention `score_mod` that keeps it at node level, `(B, H, N, N)`, and
> gathers it inside the kernel, with the structural mask expressed as a sparse
> `BlockMask`. On the text-attributed-graph benchmarks this reduces the training
> step by 2.0×–9.6×, growing with sequence length; one epoch of ogbn-arxiv falls
> from 50.7 h to 5.3 h. The residual cost of the graph bias over an unmodified
> Llama-3.2-1B on identical sequences is then 1.9×–2.0× and nearly independent of
> length, where the dense implementation had grown to as much as 19×. That
> residual is the per-layer bias computation, not the attention: a plain model
> given our structural mask but no bias costs the same as plain causal attention.
> The two implementations compute the same function (verified to kernel tolerance
> on forward outputs and bias gradients, with all bias types enabled and with the
> K-hop mask both off and on), so the reported results are unchanged. On tasks
> with very short sequences — our GraphQA setting, ~30–150 tokens — the dense path
> remains the better choice, since FlexAttention requires block-aligned sequence
> lengths with a 128-token block; we select the backend per experiment.

---

## 7. Measurement corrections (2026-07-26)

Recorded so the numbers above can be trusted and the mistakes are not repeated.

1. **Allocator stall.** `run_method` called `torch.cuda.empty_cache()` immediately
   before the timed loop, so the first timed step to meet each shape re-acquired
   the pool from the driver — up to **6.6 s** against a 103 ms median. Impact:
   0.3–6.1% on the original TAG arms (they ran after eager had grown the pool),
   but up to **−46%** on a job where a cheap arm ran first. Fixed by removing the
   call (`max_memory_allocated` counts *allocated* bytes, so cached-free blocks
   never affected the peak reading anyway). A `step_ms_trimmed_mean` is now
   recorded and the reporter flags any arm whose mean exceeds it by >10%.

2. **The `sdpa` floor was doing less work than GTLM.** Plain causal attention
   admits 0.50 of the L×L matrix; GTLM's mask (causal relaxed to bidirectional
   between prefix tokens) admits 0.70. Added `sdpa-graphmask`, the same stock
   model handed GTLM's own dense mask, with a self-check that hard-fails if HF
   ignores the 4-D mask. Result: the mask is essentially free (0.90–1.04× plain
   causal), which is *why* the residual 2× is attributable to the bias modules.

3. **Cross-experiment absolute times were not comparable.** GraphQA's recipe is
   fp32 / B=4, TAG's is bf16 / B=1. A GraphQA row therefore showed a *larger*
   absolute cost than a longer TAG row: at B=4 its incidence arm pushes 1,301
   tokens/step, not the 137 its per-graph `L` suggested, and fp32 costs ~3× bf16
   on `sdpa` (183.0 → 60.2 ms). GraphQA was re-measured at bf16/B=1 for the
   cross-experiment table; `dtype` and `batch_size` are now part of the reporter's
   dedup key, and `tok/step` is a column.

4. **Sample size.** Every number rests on 24 strided batches — 0.03% of
   ogbn-arxiv's 90,941 graphs. Adequate for the ratios, thin for the epoch
   extrapolation, since step cost is ~quadratic in L and the length tail is
   undersampled. `--n-batches 200` is cheap now that the kernels are in
   inductor's on-disk cache, and is the first thing to do if a reviewer presses
   on the wall-clock claim.

---

## 8. Status

- [x] Real-input benchmark harness (`bench_real.py`), launchers, consolidated
      reporter (`report_real.py`)
- [x] H100 scaling figures regenerated (`results_h100/fig_*.png`)
- [x] All four TAG datasets, both gradient-checkpointing settings
- [x] `sdpa-graphmask` (mask-matched floor) and `flex-nobias` (bias-free) arms —
      together they localise the residual 2× to the bias modules
- [x] GraphQA at its paper recipe *and* normalized to bf16/B=1
- [x] Parity re-verified on the H100 (11 + 39 tests, job 117591)
- [x] Measurement bugs found, quantified and fixed (§7)
- [ ] Decide the final rebuttal wording; optionally re-run at `--n-batches 200`
      to firm up the epoch extrapolation

Not done, deliberately: no k_hop>0 arm, no retraining (§4 "Deliberately out of
scope"); GraphQA `incidence` at its fp32 paper recipe was not re-measured
post-fix (the artifact there is ≤2.3% and does not change the verdict).
