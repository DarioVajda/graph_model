# `linear_bias` — what does linearizing the magnetic bias cost?

**Status:** Phases 0–3 complete (Phases 0–2 2026-08-05, Phase 3 2026-08-06).
Verdict: **linearization is not free, and its price is dataset-dependent by two
orders of magnitude** — but a large part of what was being charged to it was the
intra-node diagonal mask, which the factorization cannot express anyway (§6).
Read at the configuration the deferred backbone would actually run, the WebQSP
price is 5.6% of the magnetic headroom rather than 18.8%, the 4k context task
saturates, and GraphQA turns from free to a systematic ~5.7% cost. The
`LINEAR_BIAS.md` §7 factorized backbone is **still not** cleared to start; §7
below (Conclusions) names the one sweep that would decide it.

This package prices the plan in `src/models/LINEAR_BIAS.md`: replacing
`MagneticBias`'s 2-layer MLP head with a single linear map, which turns the bias
into a bilinear form — an inner product attention already computes — so a future
backbone could fold it into wider Q/K and delete the `(B,N,N,·)` tensor and the
flex `score_mod` outright.

---

## 0. What this package is

A **measurement package**: two offline scripts, eight sweep configs, and the
results. The model changes themselves are `LinearMagneticBias` in
`src/models/bias.py` (flag `--magnetic-linear`) and the diagonal switch
`--bias-self-node`; every sweep invokes an existing experiment normally and only
points at a config that lives here.

```bash
# Phase 0 — offline, reads trained bias_parameters.pt from the g-sweep checkpoints
sbatch src/experiments/bias_experiments/linear_bias/sbatch_phase0.sh
python3 -m src.experiments.bias_experiments.linear_bias.analyse

# Phase 1 — correctness gate (tests 9-14 of LINEAR_BIAS.md §5.3)
sbatch src/experiments/bias_experiments/linear_bias/sbatch_tests.sh

# Phase 2 — dry-run every job through the real parser/validator, then submit
sbatch src/experiments/bias_experiments/linear_bias/sbatch_preflight.sh
python3 -m sweep src.experiments.kgqa    src/experiments/bias_experiments/linear_bias/configs/010_webqsp_linear.jsonc
python3 -m sweep src.experiments.context src/experiments/bias_experiments/linear_bias/configs/011_context4k_linear.jsonc
python3 -m sweep src.experiments.context src/experiments/bias_experiments/linear_bias/configs/012_context4k_linear_long.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/bias_experiments/linear_bias/configs/013_graphqa_linear.jsonc
python3 -m sweep src.experiments.kgqa    src/experiments/bias_experiments/linear_bias/configs/014_webqsp_magdim256.jsonc

# Phase 3 — the diagonal mask (§6)
python3 -m sweep src.experiments.kgqa    src/experiments/bias_experiments/linear_bias/configs/015_webqsp_selfnode.jsonc
python3 -m sweep src.experiments.context src/experiments/bias_experiments/linear_bias/configs/016_context4k_selfnode.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/bias_experiments/linear_bias/configs/017_graphqa_selfnode.jsonc
```

Each config is its source experiment's headline recipe **verbatim**, with only
the bias axes changed. `results/` is gitignored except the run records.

**SPD and RRWP are off in every arm.** Neither has an `f(i)·g(j)` form, so an arm
that kept them would price a recipe the factorization can never reach — and two
strong alternative structural channels would absorb whatever the magnetic head
loses. Absolute numbers here therefore sit **below** each experiment's
recipe-of-record; only the within-config gaps are meant to be read.

## 1. The arms

| arm | config | role |
|---|---|---|
| B | `magnetic` | incumbent MLP head |
| C | `magnetic_linear` | the candidate |
| D | no soft bias | headroom denominator |

The headline is quoted as **B→C as a fraction of the D→B headroom**, not as raw
pp: raw pp is not comparable across three datasets whose metrics and difficulty
differ.

Phase 3 (§6) crosses B and C with a second axis, the intra-node diagonal mask
(`--bias-self-node`), because the factorization can only run one side of it.

---

## 2. Phase 0 — offline (`p0_*.json`, `analyse.py`)

Least-squares fit of a linear head to the **trained MLP head's own output**, on
real batches, per layer and head, over the M-grid.

| | WebQSP | context |
|---|---|---|
| R² (median seed, M=128 / M=64) | 0.960 | 0.947 |
| residual / bias std | 0.10 | 0.12 |
| worst layer R² | 0.65 (L5) | 0.63 (L6) |
| energy inside the rank-2M cap | **100%** | **100%** |
| rank at 90% / 99% of energy | 2 / 22 | 24 / 34 |

Three things it established, and one it got wrong.

**(a) The rank ceiling never binds.** 100% of the trained bias's spectral energy
lies inside the rank-2M cap, and the matrices are nearly rank-1 (90% of WebQSP's
energy in **two** singular values). Whatever the M-sweep costs, it is truncation
of *information*, not of rank. This held up in Phase 2 and is the one Phase 0
result that transfers to the 1024–4096-node trunk.

**(b) The nonlinearity is nearly idle — in most layers.** At M=128, thirteen of
sixteen WebQSP layers fit at R² = 1.000; the exceptions are L5 ≈ 0.55, L6 ≈ 0.92,
L14 ≈ 0.97. Report the worst layer, never the mean: the mean (0.96) reads as
"linearization is nearly free" and hides exactly the layer where it is not.

**(c) R² is flat in M** (0.973 → 0.960 across a 16× range), so Phase 0 pruned
nothing and `LINEAR_BIAS.md` §6.1 ran the full grid.

**What it got wrong: Phase 0 is not predictive of Phase 2.** R² ≈ 0.96 on WebQSP
preceded a 4.91 pp loss, and R² ≈ 0.93 on context preceded near-total collapse —
the *worse* offline number belongs to the dataset that survived best, and the
flat-in-M curve preceded a Phase 2 M-curve that is not flat. This is a
methodological result, not a defect of the measurement: P0 asks "can a linear
head **imitate this already-trained** bias?", Phase 2 asks "can a linear head be
**trained** to solve the task?" — and the second is not implied by the first,
because imitation is scored on the MLP's solution rather than on the task.
**Do not gate future GPU-days on an offline imitation R² alone.**

> `analyse.py` prints a measured `resid/std` (0.08–0.10 on WebQSP). An earlier
> revision of `LINEAR_BIAS.md` §4.1 tabulated 0.165–0.199 for the same runs; that
> column was the `sqrt(1-R²)` fallback taken when `resid_mean_over_layers` was
> absent from the JSON, not a measurement. The measured value is the correct one.
> Condition numbers on the normal equations reach 1e25, so read R² and the
> measured residual, not the fitted coefficients.

---

## 3. Phase 2 — WebQSP (`010`, `014`)

3 seeds, median-seed rule, test F1 (Llama-3.2-1B, LoRA r=64, 15 epochs).

| arm | M | test F1 | Hits@1 | B→C | as % of headroom |
|---|---:|---:|---:|---:|---:|
| B magnetic | 128 | **0.7190** | 0.7789 | — | — |
| B magnetic | 16 | 0.7060 | 0.7611 | — | — |
| C linear | 128 | 0.6699 | 0.7359 | −4.91 pp | **18.8%** |
| C linear | 64 | 0.6717 | 0.7383 | −4.73 pp | 18.1% |
| C linear | 32 | 0.6565 | 0.7267 | −6.25 pp | 24.0% |
| C linear | 16 | 0.6424 | 0.7144 | −7.65 pp | 29.3% |
| D no bias | — | 0.4580 | 0.5338 | — | — |

**The headline: 4.91 pp of test F1, 18.8% of what the magnetic bias is worth.**
Consistent across seeds and mirrored in Hits@1, so it is the bias and not the
decoder.

**The linear head degrades faster in M than the MLP head.** 128→16 costs B 1.30 pp
but C 2.74 pp — the two are not interchangeable as M shrinks, which is why B was
swept too. But M=64 is **free** for C (0.6717 vs 0.6699, inside seed noise), and
M=64 means 2M = 128 extra head dims against Llama-1B's `head_dim` = 64. So the
operating point for a factorized backbone is **M=64, not the incumbent 128** —
half the head-width blowup for no measured quality.

**Capacity is not the explanation (`014`).** Arm C changes two things at once: it
drops the nonlinearity *and* shrinks the head (24 864 vs 53 664 params/layer,
2.16×). Giving the linear head a 92%-matched budget at `magnetic_dim` 256 yields
test F1 **0.6647** — it recovers *none* of the 4.91 pp and lands marginally below
d=128. The gap is the bilinear family's real price, and widening Ψ will not buy
it back. (This does not test whether the *MLP* head would also gain from d=256;
the comparison asked was parameter-matched, and that is what it answers.)

---

## 4. Phase 2 — GraphQA (`013`)

3 tasks × 3 seeds × {5e-3, 2e-2} bias_lr, test accuracy, each arm read at its own
best LR.

| task | B magnetic | C linear | D no bias | B→C | as % of headroom |
|---|---:|---:|---:|---:|---:|
| node_degree | 0.984 | 0.974 | 0.086 | −1.0 pp | +1.1% |
| shortest_path | 0.970 | 0.944 | 0.470 | −2.6 pp | +5.2% |
| edge_count | 0.470 | **0.496** | 0.026 | +2.6 pp | **−5.9%** |
| | | | | | **mean +0.2%** |

**Linearization costs nothing here, and wins one task.** At matched bias_lr = 5e-3
the linear head is actually ahead on average (mean −3.0% of headroom, i.e. a small
gain). GraphQA's graphs are ~20 nodes with a complete eigenbasis, `magnetic_dim`
is 32 so the two heads are within 1.17× on parameters, and the tasks are
algorithmic. Every condition that could make linearization hurt is absent — and
it does not hurt. This bounds the WebQSP result: 18.8% is not a property of the
method in general.

---

## 5. Phase 2 — 4k context (`011`, `012`)

Needle-retrieval code accuracy, 3 seeds, 16–128 nodes / 4096 tokens.

`011`, at the recipe's own budget (2 epochs, bias_lr 5e-3):

| arm | M | code_acc (median) |
|---|---:|---:|
| B magnetic | 128 | **0.995** |
| B magnetic | 16 | 0.940 |
| C linear | 128 / 64 / 32 / 16 | 0.080 / 0.095 / 0.080 / 0.080 |
| D no bias | — | 0.030 |

Read as a table that says linearization *destroys* the task — 94.8% of the
headroom. It does not say that. The metric is all-or-nothing on an exact 3-token
code, so it stays at noise until the model starts using the graph; B falls off a
cliff between evals 2 and 3, while C's eval loss was still descending
monotonically at ~0.014/eval when the budget ran out, far below D's plateau. That
is an **unfired metric**, not a floor.

`012` separates "cannot" from "under-stepped" by sweeping bias_lr at 2× budget
(4 epochs), M=128 only:

| arm | bias_lr | code_acc (median) | seeds |
|---|---:|---:|---|
| C linear | 5e-3 | 0.090 | 0.080 / 0.090 / 0.115 |
| C linear | **2e-2** | **0.590** | 0.435 / 0.590 / 0.775 |
| C linear | 5e-2 | 0.070 | 0.065 / 0.070 / 0.360 |
| D no bias | 5e-3 | 0.035 | 0.025 / 0.035 / 0.045 |

**`011`'s collapse was largely an optimization artifact.** bias_lr 5e-3 was tuned
for the MLP head's 36 864 bias parameters; the linear head has 8 192 and no hidden
nonlinearity, and at 4× the LR it goes from 0.09 to **0.59** — from noise to
firing. It is still short of B's 0.995 and the seed spread is enormous
(0.435–0.775), which is what a transition caught mid-flight looks like. 5e-2
diverges, so the useful range is narrow.

The load-bearing consequence: **`011`'s M-grid was run at an LR now known to be
wrong for the linear head**, so the context M-curve is currently uninterpretable.

---

## 6. Phase 3 — the intra-node diagonal (`015`, `016`, `017`)

`LINEAR_BIAS.md` §7.3 records that `_finalize` zeroes $b_{ii}$ and that the
factorized form **cannot** express that zeroing — an inner product gives
$q_i \cdot k_i$ and there is nothing to subtract it with. For arm C the mask is
therefore not a setting: unmasked is the only configuration the deferred backbone
can actually run, and every number in §3–§5 was measured with it on. The flag
`--bias-self-node` (default **off**, so all earlier results stand bit-for-bit)
keeps the diagonal; these three sweeps price it.

Each masked cell is a deliberate **rerun** of the corresponding `010`/`013`
headline cell, which makes it both a within-sweep control and a regression check
on the refactor that turned `_finalize` into an instance method:

| cell | `010` / `013` | rerun |
|---|---:|---:|
| WebQSP B magnetic, M=128 | 0.7190 | 0.7104 ± 0.0168 |
| WebQSP C linear, M=128 | 0.6699 | 0.6715 ± 0.0048 |
| GraphQA C `node_degree` | 0.9767 | 0.9820 |
| GraphQA C `shortest_path` | 0.9393 | 0.9387 |
| GraphQA C `edge_count` | 0.4967 | 0.4887 |

All within seed noise, so the default path is unchanged and the cross-sweep
comparisons below are sound.

### 6.1 WebQSP (`015`) — the diagonal recovers two thirds of the penalty

3 seeds, M=128, bias_lr 5e-3, 15 epochs — `010`'s headline recipe exactly.

| head | mask | test F1 | Hits@1 | Δ F1 vs masked |
|---|---|---:|---:|---:|
| B magnetic | on | 0.7104 ± 0.0168 | 0.7713 | — |
| B magnetic | off | 0.6930 ± 0.0106 | 0.7510 | −1.74 pp |
| C linear | on | 0.6715 ± 0.0048 | 0.7336 | — |
| C linear | **off** | **0.6962 ± 0.0024** | 0.7529 | **+2.47 pp** |

Paired by seed, C gains on 3/3 (+2.07, +2.19, +3.15 pp); B loses on 2/3. **The
effect is head-dependent, and the B arm is what establishes that** — without it
"the diagonal helps C" could not be separated from "this bias likes a diagonal".

Reading each head at its own best mask setting — B masked, C unmasked, since B has
no reason to give up a mask it *can* implement — the residual B→C is −1.42 pp,
**5.6% of the D→B headroom**, against 15.4% within this sweep with both masked and
the 18.8% headline of `010`. Quoted against `010`'s B instead, the residual is 8.7%.
**The WebQSP penalty does not vanish; it drops by roughly two thirds.**

### 6.2 4k context (`016`) — the mask, not the family, caused the fragility

Arm C only, 3 seeds, M=128, 4 epochs, `012`'s bias_lr pair.

| bias_lr | mask | code_acc (mean) | seeds |
|---|---|---:|---|
| 5e-3 | on | 0.133 | 0.085 / 0.140 / 0.175 |
| 5e-3 | **off** | **1.000** | 1.000 / 1.000 / 1.000 |
| 2e-2 | on | 0.983 | 0.975 / 0.980 / 0.995 |
| 2e-2 | off | 0.985 | 0.970 / 0.985 / 1.000 |

The largest effect in the package. Masked, C fires only at 2e-2 — the LR
sensitivity §5 diagnosed. Unmasked it saturates at **both** LRs, and at 4 epochs
reaches 1.000, above `011`'s B (0.995). On this task the linear head is not short
of capacity at all; it was short of a diagonal.

Two limits on what this licenses: `016` ran **arm C only**, so there is no B or D
control inside it, and two LR points are not a curve. It supports "the mask, not
the bilinear family, was responsible for C's context fragility" — it is not a new
B-vs-C number, and §5's void M-curve stays void.

### 6.3 GraphQA (`017`) — the diagonal costs here, systematically

3 tasks × 2 heads × 2 masks × 3 seeds, bias_lr 5e-3, 20 epochs.

| task | head | mask on | mask off | Δ | as % of headroom |
|---|---|---:|---:|---:|---:|
| node_degree | C | 0.9820 | 0.9633 | −1.87 pp | 2.1% |
| shortest_path | C | 0.9387 | 0.9253 | −1.33 pp | 2.9% |
| edge_count | C | 0.4887 | 0.4333 | −5.53 pp | **12.0%** |
| | | | | | **mean 5.7%** |
| node_degree | B | 0.9653 | 0.9707 | +0.53 pp | — |
| shortest_path | B | 0.9280 | 0.9460 | +1.80 pp | — |
| edge_count | B | 0.4880 | 0.4247 | −6.33 pp | — |

(Headroom = this sweep's masked B minus `013`'s D at the same bias_lr.)

**Arm C is negative in 9 of 9 seed-paired comparisons** (sign test p ≈ 0.002), so
this is systematic, not seed noise. `edge_count` — the one task the linear head
*won* in §4 — takes the largest hit, and its seed spread inflates 14× (σ 0.0031 →
0.0424). §4's "linearization is free on GraphQA" holds only under the mask; the
configuration the factorization can actually run costs ~5.7% of headroom here.

### 6.4 Why the sign flips across datasets

The diagonal is a **node-level** quantity and `expand_node_to_token_bias` lifts
node pairs to token pairs, so zeroing $b_{ii}$ removes the structural bias from
*every token pair inside one node*, not merely a token attending to itself. WebQSP
entity names and the context task's nodes are multi-token, so the mask suppresses a
large block of genuine pairs; GraphQA's nodes are ~1–2 tokens, where the same mask
costs almost nothing and the self-energy $K(i,i) = \sum_l |V_{il}|^2 \phi_l$ acts
closer to an unwanted per-node constant. That is a hypothesis consistent with all
three datasets — **these sweeps do not test it**, and node-token-length is
confounded with dataset, metric and graph size.

---

## 7. Conclusions

1. **Linearization has a real, non-zero price, and it is dataset-dependent.**
   Under the diagonal mask: free on GraphQA (+0.2% of headroom, wins `edge_count`),
   18.8% of headroom on WebQSP, unresolved-but-large on 4k context. No single
   number describes it — and §6 moves all three, in both directions.
2. **The price is the bilinear *family*, not the parameter count** — `014` settles
   this on WebQSP: a 92%-matched linear head recovers none of the 4.91 pp.
3. **The rank ceiling never binds** at any scale measured (P0b: 100% of energy
   inside the cap, WebQSP nearly rank-1). This is the result that extrapolates to
   the trunk, and it is the good news: the factorization's rank budget is
   enormous relative to what the bias uses.
4. **M = 64 is the operating point**, not 128. Free on WebQSP, halves the deferred
   backbone's head-width blowup from 5× to 3×. M = 32 costs a further 1.3 pp.
5. **bias_lr is arm-dependent and must be swept per arm.** The linear head wants
   ~4× the MLP head's LR on context; B itself preferred 2e-2 on `shortest_path`.
   Any B-vs-C comparison at one shared LR prices optimization, not math.
6. **Offline imitation R² did not predict trained quality** (§2). Phase 0's real
   contribution was the rank spectrum, not the fit.
7. **The diagonal mask was a confound worth pricing, and it moved the headline**
   (§6). Because the factorization cannot express the mask, arm C's honest
   configuration is unmasked — and unmasked it is a *different arm*: WebQSP's
   penalty falls 15.4% → 5.6% of headroom, context's LR fragility disappears
   entirely, and GraphQA's "free" becomes a systematic ~5.7% cost.
8. **The diagonal's sign is dataset-dependent and head-dependent.** It helps C and
   hurts B on WebQSP; it helps C enormously on context; it hurts both heads on
   GraphQA. Any future claim about it needs its own control arm — §6.1's B arm is
   what made the WebQSP reading unambiguous.

### Verdict on §7 (the factorized backbone)

**Still not cleared, but the case is materially stronger than it was.** The
trunk's regime is long-context and 1024–4096 nodes, and that is where §6.2 lands
its biggest result: unmasked, the linear head saturates the 4k task at both LRs and
at half `012`'s budget, which removes the specific failure that blocked clearance.
The WebQSP price for deleting the `score_mod` is now 5.6% of headroom rather than
18.8% — a clearly defensible bargain given that ~86% of flex's k=0 overhead is bias
machinery. What blocks clearance is no longer plausibility but coverage: `016` had
no B or D arm, so there is still **no clean B-vs-C number in the long-context
regime at the configuration the backbone would run**.

**The one sweep that would decide it:** re-run `011`'s M-grid unmasked at
bias_lr 2e-2 with a budget past the transition (≥6 epochs), **plus arm B at the
same budget and both mask settings**. If C lands near B, §7 proceeds at M=64. If C
plateaus below B with both the LR and the mask confounds removed, the long-context
regime is where the bilinear family genuinely fails and §7 should not be built.

### Outstanding

- **No B or D control in the long-context regime at the unmasked configuration**
  (§6.2). This is now the single blocker on the §7 verdict; the sweep above closes it.
- **`magnetic` at d=256 was not run**, so "is 128 under-provisioned for *both*
  heads?" is open (§3).
- The context M-curve is void — `011`'s grid was run at both the wrong LR and the
  wrong mask setting.
- **§6.4's explanation for the sign flip is untested.** Node token-length is
  confounded with dataset, metric and graph size; a within-dataset test would need
  node text length varied directly.
