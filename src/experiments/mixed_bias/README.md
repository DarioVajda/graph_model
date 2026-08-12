# mixed_bias — decoupled magnetic bias (phase vs. magnitude)

Phase 2 of `src/models/MIXED_BIAS.md`. `linear_bias` established that dropping the
MLP head costs **5.6% of the magnetic headroom on WebQSP** and **~5.7% on
GraphQA** when read at the mask setting a factorized backbone could actually run.
This experiment asks whether a non-linear **magnitude** channel — the one kind of
non-linearity a bilinear form admits — recovers it.

The plan document is the authority on the maths and the reasoning; this file is
the operational record.

## Arms

| Arm | Config key | Factorization | Diagonal | Isolates |
|---|---|---|---|---|
| 0 | *(no soft bias)* | — | — | the floor (headroom denominator) |
| 1 | `magnetic` | $O(N^2)$ dense | **masked** | the ceiling (incumbent target) |
| 2 | `magnetic_linear` | phase only | unmasked | cost of losing non-linear routing |
| 3 | `magnetic_magnitude` | magnitude only | unmasked | cost of losing directed flow |
| 4 | `magnetic_hybrid` | phase + magnitude | unmasked | **the proposed $O(N)$ replacement** |

Arms 2–4 run unmasked because an inner product yields $\langle q_i, k_i\rangle$
and cannot be forced to zero — unmasked is the only configuration a factorized
backbone can run. Arm 1 keeps the mask because it is the legacy method and can
implement one. `linear_bias` Phase 3 established this is not a free choice and
that its sign is head-dependent.

## Sweeps

| Config | Experiment | Runs | GPU | Budget |
|---|---|---:|---|---|
| `018_webqsp_mixed` | `kgqa` | 24 → **12** | H100 | 15 epochs, `12:00:00` |
| `019_graphqa_mixed` | `graphqa` | 36 → **0** | A100 (40/80 GB) **or** H100 | 20 epochs, `04:00:00` |
| `020_context4k_mixed` | `context` | 27 → **15** | B200/B300 | 6 epochs, `16:00:00` |

> **Arms 3 and 4 were withdrawn mid-flight** (2026-08-11) after four divergences
> showed the magnitude parameterization is not trainable as specified — see
> "Divergences" below and `MIXED_BIAS.md` §5.7. The 27 remaining runs are the
> **baselines only**: arm 0 (floor), arm 1 `magnetic` (ceiling) and arm 2
> `magnetic_linear`. Those have never diverged and are worth having regardless,
> since any future arm-3/4 sweep is read against exactly them. `019` is cancelled
> in full because all 36 of its runs were arms 3/4.

87 runs, `max_concurrent: 16` each. Every arm at **M = 64** — the operating point
the optimized backbone would ship (`MIXED_BIAS.md` §2.5), which is why WebQSP
re-runs arms 1 and 2 rather than quoting `015` (M=128 only) and `010` (M=64 but
masked). GraphQA reuses `013` D and `017` B/C: its cache is built at
`magnetic_m: 0` on ~20-node graphs, so truncation at 64 and at 128 are the same
no-op. Context runs all five arms in-sweep, which closes the hole
`linear_bias/README.md` §7 names as the blocker on the factorized backbone.

`magnetic_m_collate` is collator-only and outside `data_config_key`, so all three
sweeps reuse existing builds — preflight verifies this before submission.

```bash
python3 -m src.experiments.mixed_bias.preflight          # 1. every run parses, validates, gets its head
./src/experiments/mixed_bias/sbatch_tests.sh             # 2. the §4.2 correctness gate (must be green)
python3 -m sweep src.experiments.kgqa    src/experiments/mixed_bias/configs/018_webqsp_mixed.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/mixed_bias/configs/019_graphqa_mixed.jsonc
python3 -m sweep src.experiments.context src/experiments/mixed_bias/configs/020_context4k_mixed.jsonc
python3 -m sweep.report src/experiments/mixed_bias/results/<sweep>
```

## Divergences (runs with no metric)

Recorded here because a cancelled run leaves no `runs.jsonl` line, and a silent
gap in a results table is indistinguishable from a run that was never submitted.

**`results/diverged_runs.tsv` is the live, authoritative list** — it is appended
automatically as runs fail, so it does not go stale the way this table does. As of
writing:

| # | sweep | cell | first NaN |
|---|---|---|---|
| 1 | `020` context | arm 4 `magnetic_hybrid`, `bias_lr` **2e-2**, seed 1 | epoch 0.27 |
| 2 | `018` WebQSP | arm 3 `magnetic_magnitude`, `bias_lr` **2e-2**, seed 2 | epoch 2.48 |
| 3 | `018` WebQSP | arm 3 `magnetic_magnitude`, `bias_lr` **2e-2**, seed 0 | epoch 3.17 |
| 4 | `018` WebQSP | arm 4 `magnetic_hybrid`, `bias_lr` **5e-3**, seed 0 | epoch 1.77 |

**Divergence 4 is at the LOW learning rate**, which retires the reading that this
is a 2e-2 problem. Worse, the ordering is not monotone in LR:

| arm | `bias_lr` | outcome |
|---|---|---|
| magnitude | 5e-3 | clean at ep 4.91 — longest-surviving magnitude run |
| magnitude | 2e-2 | died at 2.48 **and** 3.17 (2 of 2 seeds) |
| **hybrid** | **5e-3** | **died at 1.77** |
| hybrid | 2e-2 | died at 0.27 |
| linear (control) | 2e-2 | clean at ep 6.13 |

The hybrid at **one quarter** the learning rate failed *sooner* than magnitude-only
at full rate. Lowering the LR delays the failure; it does not remove it. That is
what §5.7 predicts for a quartic term and it is why arms 3 and 4 were withdrawn
rather than re-run at a third LR.

No `magnetic` or `magnetic_linear` run has diverged, at either LR, on either
dataset — which is why the baselines were kept running.

Both are unrecoverable rather than transient. No `max_grad_norm` is set, so HF's
default clipping of 1.0 applies, and `clip_grad_norm_` scales *every* gradient by
a total norm of NaN — one poisoned step destroys all parameters permanently, which
is why loss pins to exactly 0.0 from then on.

### What the four events have in common

Across two datasets (context 4k / WebQSP), two batch sizes (1 / 2), two sequence
regimes (T=4096 flex / short), two arms (3 and 4), two learning rates and three
seeds, exactly **one** factor is common to every divergence and absent from every
survivor: **the magnitude channel is present**. It is not the dataset, the batch
size, the sequence length, the LR, or the seed.

### Retraction

An earlier revision of this section proposed a **shared-trunk coupling** mechanism
specific to arm 4 (two heads on one DeepSets trunk). That is now **disconfirmed**:
arm 3 has only one head and diverged too, on a different dataset. The hypothesis
that a single pathological sample was responsible is also weakened — it would have
to recur across two datasets and two batch sizes.

### The explanation the evidence now supports

**Both sides of the magnitude bilinear form are learned and unbounded, and the
bias is therefore quadratic in `Z`.**

    b_magnitude(i,j) = <Z_i . s^(h), Z_j W_K^(g)>        Z = MLP_magnitude(S)

Compare the phase channel, where `K_phase = [V_R || V_I]` is **parameter-free**,
and bounded by orthonormality — measured across 7 856 node-rows of the context dev
split, `sum_l |V_il|^2` is exactly 1.0000 (min = median = max). So `b_phase` is
*linear* in the learned parameters with a bounded partner, and nothing about it
can run away.

`b_magnitude` has no such anchor. `S` is bounded (it is a convex combination of
the phi rows), but `Z = MLP(S)` is not, and both factors of the inner product are
functions of it. Growth in `Z` feeds back quadratically into the bias, into the
loss, and into the gradients w.r.t. the MLP, `s` and `W_K`. The failure is *late*
(epoch 1.77-3.17 on WebQSP) precisely because `s` starts at zero and has to grow
first — and a lower LR only makes the growth slower, which is why 5e-3 postpones
the divergence without preventing it.

The WebQSP trajectory is the clean signature of exactly this — note the **loss**
diverging in lockstep, unlike the context event:

    ep2.44  loss=0.67   grad=8.7        <- healthy
    ep2.45  loss=9.38   grad=895
    ep2.46  loss=6.72   grad=2.2e5
    ep2.47  loss=6.95   grad=3.2e7
    ep2.47  loss=11.45  grad=1.075e12
    ep2.48  loss=46.90  grad=NaN

### Design consequence — BUILT, 2026-08-12

The magnitude channel wanted its bilinear form anchored the way the phase channel
is. As built (`MIXED_BIAS.md` §2.3 and §5.8) the anchor is on the **factors**, not
on `Z`: `Q_magnitude` and `K_magnitude` are L2-normalised per node row *after*
`s` and `W_K` are applied, and a per-head scalar gain carries the range.

Normalising `Z` alone would have been weaker in a way that matters here. The §5.7
fixture varies only the trunk scale, so it cannot distinguish "the trunk grows"
from "`s` or `W_K` grows"; normalising `Z` is invariant to the first by
construction and does nothing about the second. Normalising the factors is
invariant to all three at once — which is the property worth having, because the
diagnostic that would have named the culprit never ran (job 125240, cancelled at
0:00 elapsed by the same 21:40 mass-cancel that ended the sweep).

The zero-init moved from `s` onto the gain: `s` sits inside the normalised vector,
and normalising a deliberately-zero vector has Jacobian `I/eps` ≈ 1e12 at step 0.

**These runs are therefore not re-runnable as-is.** Arms 3 and 4 now describe a
different parameterisation, so the four divergences above are a record of what was
withdrawn, not a baseline for what replaces it. Arms 0-2 are unchanged and their
`018`/`021` numbers stand.

### Still open

The context event does not look like the WebQSP one: its **loss stayed at
1.23-1.45** through grad norms of 7e4, and it *recovered* to grad_norm 11 before
dying. A pure runaway does not recover. So either the hybrid fails by a second
route, or the recovery is an artifact of clipping bounding each update while the
underlying state drifts. Diagnostic job 125240 re-runs the arm-4 config with
per-parameter-group gradient logging to separate these.

**Cost exposure.** 30 of the 87 runs are magnitude-bearing arms (3 and 4) at 2e-2:
6 on WebQSP, 18 on GraphQA, 6 on context. If the instability is systematic rather
than seed-specific, most of those produce no metric.


## Reading the result (`MIXED_BIAS.md` §5.6)

* **Arm 4 within seed noise of arm 1** clears the path to build the $O(N)$ flex
  backend and delete the `score_mod`.
* **Arm 3 alone is the diagnostic.** If magnitude alone recovers most of the
  headroom, directed flow is not what the magnetic bias is selling. If it is near
  the floor, the phase channel is load-bearing and the tandem's gain is additive.
* **Arm 4 ≈ arm 2** would say the magnitude channel is inert, and the residual
  against arm 1 is pairwise non-linearity no factorization reaches. That closes
  `LINEAR_BIAS.md` §7 negatively, which is also a result.

Two standing caveats. On **context**, arms 1 and 2 have already hit 0.995/1.000,
so if everything lands at ceiling the finding is "every factorizable arm reaches
the ceiling in the long-context regime" — *not* "the arms are equivalent". And
**arm 3 is predicted weak on context by construction**: `data.py` gives every
content node the same distance to the QUESTION node and adds indistinguishable
decoy edges, so the discriminating information lives in relative position along
the chain, which is the phase channel. That prediction is tested, not acted on.
