# `linear_bias` — what does linearizing the magnetic bias cost?

**Status:** Phases 0–2 complete (2026-08-05). Verdict: **linearization is not
free, and its price is dataset-dependent by two orders of magnitude** — free on
GraphQA, 18.8% of the magnetic headroom on WebQSP, and near-total on the 4k
context task at the tuned LR (partly, not wholly, an optimization artifact).
The §7 factorized backbone is **not** cleared to start; §6 below names the one
sweep that would decide it.

This package prices the plan in `src/models/LINEAR_BIAS.md`: replacing
`MagneticBias`'s 2-layer MLP head with a single linear map, which turns the bias
into a bilinear form — an inner product attention already computes — so a future
backbone could fold it into wider Q/K and delete the `(B,N,N,·)` tensor and the
flex `score_mod` outright.

---

## 0. What this package is

A **measurement package**: two offline scripts, five sweep configs, and the
results. The model change itself is `LinearMagneticBias` in `src/models/bias.py`
(flag `--magnetic-linear`); every sweep invokes an existing experiment normally
and only points at a config that lives here.

```bash
# Phase 0 — offline, reads trained bias_parameters.pt from the g-sweep checkpoints
sbatch src/experiments/linear_bias/sbatch_phase0.sh
python3 -m src.experiments.linear_bias.analyse

# Phase 1 — correctness gate (tests 9-14 of LINEAR_BIAS.md §5.3)
sbatch src/experiments/linear_bias/sbatch_tests.sh

# Phase 2 — dry-run every job through the real parser/validator, then submit
sbatch src/experiments/linear_bias/sbatch_preflight.sh
python3 -m sweep src.experiments.kgqa    src/experiments/linear_bias/configs/010_webqsp_linear.jsonc
python3 -m sweep src.experiments.context src/experiments/linear_bias/configs/011_context4k_linear.jsonc
python3 -m sweep src.experiments.context src/experiments/linear_bias/configs/012_context4k_linear_long.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/linear_bias/configs/013_graphqa_linear.jsonc
python3 -m sweep src.experiments.kgqa    src/experiments/linear_bias/configs/014_webqsp_magdim256.jsonc
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
nothing and §6.1 ran the full grid.

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

## 6. Conclusions

1. **Linearization has a real, non-zero price, and it is dataset-dependent.**
   Free on GraphQA (+0.2% of headroom, wins `edge_count`), 18.8% of headroom on
   WebQSP, unresolved-but-large on 4k context. No single number describes it.
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

### Verdict on §7 (the factorized backbone)

**Not cleared.** The trunk's regime is long-context and 1024–4096 nodes — the
regime the optimization exists for and the one where the linear head performed
worst. Trading 18.8% of the magnetic headroom (WebQSP, the only clean number) for
deleting the `score_mod` is a defensible bargain given that ~86% of flex's k=0
overhead is bias machinery, but it should not be taken on a context result that is
currently confounded with its learning rate.

**The one sweep that would decide it:** re-run `011`'s M-grid at bias_lr 2e-2 with
a budget long enough for the transition (≥6 epochs), plus arm B at the same
budget. If C fires and lands near B, the WebQSP 18.8% is the worst case and §7
proceeds at M=64. If C plateaus below B with the LR confound removed, the
long-context regime is where the bilinear family genuinely fails, and §7 should
not be built.

### Outstanding

- **The diagonal-mask ablation was not run.** `LINEAR_BIAS.md` §7.3 asks Phase 2
  to also ablate `_finalize`'s zeroing of b_ii, because the factorized form cannot
  express it and the future delta would otherwise be confounded with it. Every
  number above is with the mask on.
- **`magnetic` at d=256 was not run**, so "is 128 under-provisioned for *both*
  heads?" is open (§3).
- The context M-curve is void pending the sweep above.
