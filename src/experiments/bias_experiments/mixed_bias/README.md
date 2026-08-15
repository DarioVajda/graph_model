# mixed_bias — decoupled magnetic bias (phase vs. magnitude)

Phase 2 of `src/models/MIXED_BIAS.md`. `linear_bias` established that dropping the
MLP head costs **5.6% of the magnetic headroom on WebQSP** and **~5.7% on
GraphQA** when read at the mask setting a factorized backbone could actually run.
This experiment asks whether a non-linear **magnitude** channel — the one kind of
non-linearity a bilinear form admits — recovers it.

The plan document is the authority on the maths and the reasoning; this file is
the operational record.

## Status — COMPLETE on WebQSP and GraphQA, arms 0–4 and v2 (2026-08-14)

**The answer is no.** The magnitude channel is trainable once normalised (§5.8),
but it buys nothing that survives seed noise, and it is the expensive half of the
head to ship. `MIXED_BIAS.md` §5.6's third reading is the one that landed.

* Arm 4 − arm 2 = **+0.15 pp** F1 on WebQSP (seed sd 0.19–0.89 pp) and **+1.5 pp**
  on GraphQA paired by seed (sd 2.9 pp, positive in 6/9 pairs). Not resolved.
* Arm 3 is at the WebQSP floor (**1.8%** of headroom) but recovers ~100% on
  GraphQA `node_degree` and `edge_count`. The channel is a *local* feature
  detector; it fails exactly where the answer is a path.
* The residual to dense arm 1 on WebQSP is **12% of headroom** and neither
  factorized arm closes it — `LINEAR_BIAS.md` §7 closes **negatively**.
* Cost: the magnitude channel is half the width of the phase channel and costs
  **4× the K-side storage** (per-KV-group vs. head-independent). See
  "Head-width verdict" below.

* **Arm v2** (`magnetic_linear_v2`, the multiplicative placement) is the third and
  last way to spend that per-node feature, and it is also null: **−0.82 pp** F1
  vs. arm 2 on WebQSP at 5e-3, negative at both LRs and on every metric. Its gate
  provably moved (`||W2||_F` 0.0 → 8–107), so this is a real null and not an
  untrained module. **Closed — do not re-test.** See "Arm v2 — result".

All three placements of the per-node magnitude feature are now measured — alone,
additive, multiplicative — and none of them moves WebQSP's 12.2% residual. That
residual is *pairwise* non-linearity, and the constraint `MIXED_BIAS.md` §1 states
is exactly why no per-node feature reaches it, wherever it is spent. This line of
attack is finished.

> **Superseded in part, 2026-08-15 — `nonlinear_bias`.** The *recommendation*
> above stands and was reinforced. The *reason* given for it is false as stated:
> "no per-node feature reaches it" was refuted by an arm this document did not
> anticipate. All three placements here spend the same weak feature — the
> **diagonal** `Re K(i,i)`. `magnetic_nonlinear` spends a learned pool of the whole
> kernel **row and column**, and it reaches 101.0% of headroom on GraphQA's
> `shortest_path`, beating arm 2 (95.6%) and arm 4 (96.9%) — the first factorizable
> arm to reach the dense ceiling there. So a per-node feature *can* carry pairwise
> structure; the diagonal simply cannot.
>
> It is still null on WebQSP (22.7% of headroom vs arm 2's 87.8%), and `036`
> measured why: the deficit is **flat in graph size** (Pearson +0.09), 17.7 pp even
> on graphs under 32 nodes. A pooled marginal is a good *structural* summary and a
> poor *identity* resolver — which is a statement about the form of the feature,
> not about per-node features as a class. See
> `src/experiments/bias_experiments/nonlinear_bias/README.md`.

**Recommendation: build the $O(N)$ backend with phase only.** Spend head width on
$M$, not on the magnitude channel. Nothing in arms 3, 4 or v2 changes this, and v2
in particular gets the recommendation for free — it costs no appended width. The
one regime never tested is context 4k — see "Still open".

## Arms

| Arm | Config key | Factorization | Diagonal | Isolates |
|---|---|---|---|---|
| 0 | *(no soft bias)* | — | — | the floor (headroom denominator) |
| 1 | `magnetic` | $O(N^2)$ dense | **masked** | the ceiling (incumbent target) |
| 2 | `magnetic_linear` | phase only | unmasked | cost of losing non-linear routing |
| 3 | `magnetic_magnitude` | magnitude only | unmasked | cost of losing directed flow |
| 4 | `magnetic_hybrid` | phase + magnitude | unmasked | **the proposed $O(N)$ replacement** |
| v2 | `magnetic_linear_v2` | phase, gated | unmasked | the same feature spent *multiplicatively* |

Arms 2–v2 run unmasked because an inner product yields $\langle q_i, k_i\rangle$
and cannot be forced to zero — unmasked is the only configuration a factorized
backbone can run. Arm 1 keeps the mask because it is the legacy method and can
implement one. `linear_bias` Phase 3 established this is not a free choice and
that its sign is head-dependent.

## Sweeps

| Config | Experiment | Arms | Recorded | GPU | Budget |
|---|---|---|---:|---|---|
| `018_webqsp_mixed` | `kgqa` | 0–4 | **0 / 24** | H100 | 15 epochs, `12:00:00` |
| `019_graphqa_mixed` | `graphqa` | 3, 4 | **36 / 36** | H100 | 20 epochs, `04:00:00` |
| `020_context4k_mixed` | `context` | 0–2 | **15 / 27** | B200/B300 | 6 epochs, `16:00:00` |
| `021_webqsp_baselines` | `kgqa` | 1, 2 | **12 / 12** | B200/B300 | 15 epochs, `12:00:00` |
| `022_webqsp_magnitude_repro` | `kgqa` | 3, 4 | **4 / 4** | H100 | 15 epochs, `12:00:00` |
| `023_webqsp_mixed_arms34` | `kgqa` | 3, 4 | **12 / 12** | H100/B200/B300 | 15 epochs, `12:00:00` |
| `024_graphqa_linear_v2` | `graphqa` | v2 | **18 / 18** | A100/H100/B200/B300 | 20 epochs, `00:40:00` |
| `025_webqsp_linear_v2` | `kgqa` | v2 | **6 / 6** | H100/B200/B300 | 15 epochs, `12:00:00` |
| `026_webqsp_v2_ddp_probe` | `kgqa` | v2 | *timeout by design* 0/1 | B200/B300 ×8 | throughput probe, `00:38:00` |

Read the table in two halves, split by the §5.8 rebuild on 2026-08-12 —
plus `024`/`025`, which are a **follow-up arm** rather than part of the grid
above and are read against it (see "Arm v2").

**Before.** `018` recorded **nothing** — arms 3/4 diverged (below), and every one
of its 12 baselines was either OOM-killed at `mem: 64G` or cancelled behind that
failure. There is no `018/runs.jsonl`. `021` is its baselines re-run at 128G and
is the WebQSP arm-1/arm-2 reference everything else is read against. `020` ran its
baselines only; its arms 3/4 were withdrawn and never resubmitted.

**After.** `022` is the stability reproducer — the three cells that reliably died,
re-run verbatim under the normalised form. `023` is the full WebQSP arm-3/4 grid
and `019` is the GraphQA one, both post-fix (jobs 125683 / 125684, gain logged and
zero-init). `019` is **not** cancelled: the line in an earlier revision of this
file saying so was true for ~28 hours and is now wrong.

103 runs recorded, `max_concurrent: 16` each. Every arm at **M = 64** — the operating point
the optimized backbone would ship (`MIXED_BIAS.md` §2.5), which is why WebQSP
re-runs arms 1 and 2 rather than quoting `015` (M=128 only) and `010` (M=64 but
masked). GraphQA reuses `013` D and `017` B/C: its cache is built at
`magnetic_m: 0` on ~20-node graphs, so truncation at 64 and at 128 are the same
no-op. Context was *designed* to run all five arms in-sweep, which would have
closed the hole `linear_bias/README.md` §7 names as the blocker on the factorized
backbone; it ran arms 0-2 only, so that hole is still open (see "Still open").

`magnetic_m_collate` is collator-only and outside `data_config_key`, so every
sweep here reuses existing builds — preflight verifies this before submission.

```bash
python3 -m src.experiments.bias_experiments.mixed_bias.preflight          # 1. every run parses, validates, gets its head
./src/experiments/bias_experiments/mixed_bias/sbatch_tests.sh             # 2. the §4.2 correctness gate (must be green)
python3 -m sweep src.experiments.kgqa    src/experiments/bias_experiments/mixed_bias/configs/021_webqsp_baselines.jsonc
python3 -m sweep src.experiments.kgqa    src/experiments/bias_experiments/mixed_bias/configs/023_webqsp_mixed_arms34.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/bias_experiments/mixed_bias/configs/019_graphqa_mixed.jsonc
python3 -m sweep src.experiments.context src/experiments/bias_experiments/mixed_bias/configs/020_context4k_mixed.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/bias_experiments/mixed_bias/configs/024_graphqa_linear_v2.jsonc
python3 -m sweep src.experiments.kgqa    src/experiments/bias_experiments/mixed_bias/configs/025_webqsp_linear_v2.jsonc
python3 -m sweep.report src/experiments/bias_experiments/mixed_bias/results/<sweep>
```

`gate_audit.py` checks that arm v2's gate actually left its identity
initialisation — mandatory before reading any v2 null, since the arm is
bit-identical to arm 2 at `g == 1`. It needs torch, which is not installed on the
login node, so it runs as a job and needs an explicit container workdir:

```bash
sbatch -p frida -c 4 --mem 32G -t 00:10:00 -A povejmo -J gate_audit -o <log> --wrap \
  'exec srun --container-image=<sqsh> --container-mounts=/shared:/shared \
   --container-workdir=/shared/workspace/povejmo/graph_model \
   env HOME=$HOME bash -c ".venv/bin/python -m src.experiments.bias_experiments.mixed_bias.gate_audit \
   checkpoints/kgqa/025_webqsp_linear_v2_\*"'
```

### Two traps when reading these results

Both silently corrupt an arm-keyed analysis rather than erroring, and both were
hit while writing the results below.

1. **The `arm` label in `013`/`017` `runs.jsonl` is misaligned with the flags.**
   Config arm order is `[magnetic, magnetic, linear, linear, none]`, and run
   `0004` — the **no-bias** cell — carries the label `no-spd+rrwp+magnetic`. The
   booleans (`magnetic`, `magnetic_linear`) are correct; the string is not. Key
   off the flags. Keying off the label swaps the GraphQA floor and ceiling, which
   makes arm 0 look like it scores 0.97 on `node_degree` without seeing the graph.
   Note also that `no-spd+rrwp` parses as "no (spd+rrwp)" — `rrwp` is `False`
   everywhere in these sweeps.
2. **The `magnetic` key is absent from the `kgqa` run records.** `021`/`022`/`023`
   log `magnetic_linear`, `magnetic_magnitude` and `magnetic_hybrid` but not
   `magnetic`, so a naive `r["magnetic"]` reader silently files every arm-1 run as
   arm 0. Parse `sweep_run` instead, which is faithful.

`results/021_webqsp_baselines/report.md` is also stale — it shows 4 runs against
12 in the `runs.jsonl`. Regenerate with `sweep.report` before quoting it.

## Results

### Stability — fixed, by the strict test

All **52** arm-3/4 runs under §5.8 completed their full budget with no NaN,
including all three cells that reliably died in `018`. `022`'s stricter criterion
was the gain bound, not the absence of NaN: `bias_gain_absmax` **plateaus in every
run**, reaching ~1.0–1.5 at `bias_lr` 5e-3 and ~2.8–3.8 at 2e-2, flat from ~25% of
training onward. Fixed, not postponed.

Note the gain does **not** collapse to zero in the hybrid (1.00 / 1.02 / 1.49 at
5e-3, the same range as magnitude-only). The channel had a free path to switch
itself off and did not take it, so its null result reads as *redundancy with the
phase channel* rather than inertness.

### WebQSP — test F1, M = 64, 3 seeds unless noted

Floor from `010` arm D; ceiling is arm 1 @5e-3 in `021`. Headroom = 0.2632.

| arm | `bias_lr` | n | test F1 | sd | % headroom |
|---|---:|---:|---:|---:|---:|
| 0 no soft bias | — | 3 | 0.4566 | .0038 | 0% |
| 1 `magnetic` (dense, masked) | 5e-3 | 3 | **0.7198** | .0174 | 100% |
| 1 `magnetic` | 2e-2 | 3 | 0.6297 | .0031 | 65.8% |
| 2 `magnetic_linear` | 5e-3 | 3 | 0.6876 | .0019 | 87.8% |
| 2 `magnetic_linear` | 2e-2 | 3 | 0.6734 | .0034 | 82.4% |
| 3 `magnetic_magnitude` | 5e-3 | 3 | 0.4614 | .0080 | **1.8%** |
| 3 `magnetic_magnitude` | 2e-2 | 5 | 0.4664 | .0090 | 3.7% |
| 4 `magnetic_hybrid` | 5e-3 | 5 | **0.6891** | .0089 | **88.3%** |
| 4 `magnetic_hybrid` | 2e-2 | 3 | 0.6746 | .0122 | 82.8% |

Arm 4 − arm 2 holds at +0.15 pp across every metric (Hits@1 +0.22, Hit\* −0.03,
EM +0.11 pp) — consistent in sign, an order of magnitude below the arm-1 gap.

The 12.2% shortfall against arm 1 is **not** comparable to the header's 5.6%: that
figure was read against an *unmasked* arm 1, this one against the masked arm 1 that
`021` actually ran. `linear_bias` Phase 3 established the mask is worth a large and
head-dependent amount, so the two denominators differ by roughly that.

### GraphQA — test accuracy, 3 seeds, arms 0–2 from `013`/`017`

| arm | `bias_lr` | `edge_count` | `node_degree` | `shortest_path` |
|---|---:|---:|---:|---:|
| 0 no soft bias | 5e-3 | 0.026±.004 | 0.082±.007 | 0.469±.001 |
| 1 `magnetic` +selfnode | 5e-3 | 0.425±.022 | 0.971±.015 | 0.946±.012 |
| 2 `magnetic_linear` +selfnode | 5e-3 | 0.433±.042 | 0.963±.014 | 0.925±.007 |
| 3 `magnetic_magnitude` +selfnode | 5e-3 | 0.418±.010 | 0.972±.013 | **0.655±.006** |
| 3 `magnetic_magnitude` +selfnode | 2e-2 | 0.427±.052 | 0.987±.008 | 0.749±.125 |
| 4 `magnetic_hybrid` +selfnode | 5e-3 | 0.461±.028 | 0.975±.011 | 0.931±.005 |
| 4 `magnetic_hybrid` +selfnode | 2e-2 | 0.431±.013 | 0.965±.005 | 0.937±.013 |

As % of the selfnode-matched headroom: magnitude-only recovers **98% / 100% / 39%**
against phase-only's **102% / 99% / 96%**. This is the §5.6 diagnostic landing, and
it is more informative than "inert" — the magnitude channel fully covers the local
counting tasks and fails only where the answer is a path. WebQSP is ~entirely
path-structured, which is why arm 3 sits at its floor there.

Comparability checked: `017` and `019` match on every hyperparameter except
`magnetic_m_collate` (128 vs 64), a genuine no-op here since the cache is built at
`magnetic_m: 0` on ~20-node graphs.

### Context 4k — dev EM, 3 seeds (baselines only)

| arm | `bias_lr` | dev EM |
|---|---:|---:|
| 0 no soft bias | 5e-3 | 0.035 |
| 1 `magnetic` | 5e-3 | **1.000** |
| 1 `magnetic` | 2e-2 | 0.183 |
| 2 `magnetic_linear` | 5e-3 | **1.000** |
| 2 `magnetic_linear` | 2e-2 | **1.000** |

The §5.6 caveat applies as written — both factorizable arms are at ceiling, so this
says "everything reaches the ceiling in this regime", not "the arms are equivalent".
The one thing it *does* separate is robustness: arm 1 loses 82 pp going 5e-3 → 2e-2
while arm 2 does not move. The factorized parameterization is the easier one to
tune, independent of its accuracy — the same pattern as WebQSP, where arm 1 drops
9 pp across the same LR step and arms 2/4 drop 1.4.

### Head-width verdict (`MIXED_BIAS.md` §2.5)

§2.5 prices the arms in appended width; the K side is where they actually differ.
`K_phase` is parameter-free **and head-independent** — one 2M-wide vector per node
serves all H_Q heads. `K_magnitude^(g)` is per-KV-group by construction
(`bias.py:264`). At Llama-1B's `head_dim` 64, H_Q 32, H_KV 8:

| config | QKᵀ dot width | K-cache / token / layer |
|---|---:|---:|
| content only | 64 (1.0×) | 512 (1.00×) |
| phase only | 192 (3.0×) | 640 (1.25×) |
| hybrid | 256 (4.0×) | 1152 (**2.25×**) |

**The magnitude channel is half the width of the phase channel and costs 4× the
K-side storage.** Phase → hybrid is +33% on the matmul but **+80% on the K cache**,
plus 1.3 M parameters (§2.6).

Priced per appended dimension on WebQSP:

| spend | buys | per 64 dims |
|---|---:|---:|
| +64 dims of magnitude (hybrid vs. phase @ M=64) | +0.15 pp | +0.15 pp |
| +128 dims of phase (M 64→128, `015` vs `021`) | +0.86 pp | **+0.43 pp** |

Same budget, ~3× the return, on the channel already known to be load-bearing, and
without the K-side asymmetry. That is the whole argument for shipping phase only.

## Arm v2 — the gated linear head (`024`, `025`, COMPLETE 2026-08-14)

Built and submitted **after** the verdict above, because that verdict leaves one
specific thing untried. Arm 4's null has a mechanism: an *additive* per-node
channel is redundant with the phase channel (the "Stability" note is sharp on
this — the gain had a free path to switch itself off and did not, so the channel
is redundant rather than inert). Arm v2 spends the **same** per-node feature
**multiplicatively**, inside the spectral sum, where it modulates the phase
channel instead of sitting beside it:

    g_i = 1 + tanh(MLP(S_i)) in (0,2),   S_i = sum_l |V_il|^2 phi_l = Re K(i,i)
    b^(h)(i,j) = sum_c g_[i,c] ( W_R[c,h] Re K_c(i,j) + W_I[c,h] Im K_c(i,j) ) + beta_h

i.e. **arm 2 with per-query-node channel-mixing weights**. Class:
`GatedLinearMagneticBias` in `src/models/bias.py`, whose docstring carries the
maths and the three safety properties.

Read against **existing** controls — neither sweep re-runs a baseline. `024` vs
`017` C (unmasked `magnetic_linear`) and `013` D's floors; `025` vs `021` arm 2
and `023` arm 4. `bias_lr` 5e-3 is listed first in both bundles so it occupies the
array's low-index tasks.

**What would count as a result.** The target is the WebQSP residual: arm 2 is at
87.8% of headroom and arm 1 at 100%, so 12.2% is what no factorization has
reached. v2 is still a factorization — same `min(N, 2M)` rank ceiling as arm 2,
proved below — so it cannot reach *pairwise* non-linearity. It can only reach the
part of that residual that is per-node routing. If it lands at arm 2 ± noise, the
negative closure of `LINEAR_BIAS.md` §7 gets stronger rather than weaker, and it
is worth having for that. GraphQA has little room to show anything (arm 2 already
sits at 96–102% of selfnode-matched headroom on all three tasks); `shortest_path`
at 0.925 vs the dense 0.946 is the only cell with visible daylight.

Three properties make this safe to run where arms 3/4 were not — all asserted in
`tests/models/test_gated_linear_magnetic_bias.py` (26 tests, green before
submission), and all three respond to a specific finding above:

* **Scale stability.** `tanh` bounds the gate, so the shared DeepSets trunk enters
  the bias at degree 1. This is precisely the anchor "The explanation the evidence
  now supports" says `b_magnitude` lacked: the key side here is still
  `[V_R ‖ V_I]`, parameter-free with unit row energy, so nothing can run away.
  The bias is bounded by 2× arm 2's own bound whatever the gate does.
* **Exact arm-2 equivalence at initialisation.** The gate's output layer is
  zero-initialised, so `g == 1` identically and the module is bit-identical to
  `LinearMagneticBias` at step 0 — asserted bit-for-bit. Everything measured is
  what the gate *learns*. It also means the §5.8 zero-init hazard does not recur:
  there is no normalisation of a deliberately-zero vector here.
* **Invariance.** The gate reads only `_self_energy`, the sole per-node magnitude
  feature invariant to both eigenbasis ambiguities, and multiplies a query row,
  so the U(1)/U(k) argument for the kernel is untouched.

**It buys no extra rank.** The right factor `X = [V_R ‖ V_I]` is shared across
channels, so `b = (sum_c diag(g[:,c]) X S_c) Xᵀ` keeps arm 2's ceiling of
`min(N, 2M)`. An earlier draft of the plan claimed the gate broke that bottleneck;
it does not, and the bound is now a test rather than a claim.

**Head-width note.** Unlike the magnitude channel, v2 appends **nothing** to Q/K:
the gate's width is internal (once per node per forward, never seen by attention)
and `K_struct` stays the same head-independent 2M-wide vector. On the
"Head-width verdict" table it sits on the *phase only* row — 192 dot width, 1.25×
K cache — not the hybrid row. So the §2.5 argument that killed arm 4 on cost does
not apply to it.

### Arm v2 — result: NULL on WebQSP, and the pre-registered reading applies

Both sweeps complete: `024` 18/18, `025` 6/6, no divergence anywhere.

**WebQSP (`025` vs `021` arm 2, `023`/`022` arm 4).** Same floor and headroom as
the table above (floor 0.4566, headroom 0.2632).

| arm | `bias_lr` | n | test F1 | sd | % headroom | H@1 |
|---|---:|---:|---:|---:|---:|---:|
| 2 `magnetic_linear` | 5e-3 | 3 | **0.6876** | .0019 | **87.8%** | 0.7445 |
| 4 `magnetic_hybrid` | 5e-3 | 5 | **0.6891** | .0089 | **88.3%** | 0.7467 |
| v2 `magnetic_linear_v2` | 5e-3 | 3 | 0.6794 | .0080 | 84.7% | 0.7422 |
| 2 `magnetic_linear` | 2e-2 | 3 | 0.6734 | .0034 | 82.4% | 0.7373 |
| 4 `magnetic_hybrid` | 2e-2 | 3 | 0.6746 | .0122 | 82.8% | 0.7394 |
| v2 `magnetic_linear_v2` | 2e-2 | 3 | 0.6552 | .0265 | 75.5% | 0.7189 |

v2 − arm 2 is **−0.82 pp** at 5e-3 (Welch t = −1.72, df 2.2) and **−1.82 pp** at
2e-2 (t = −1.18, df 2.1). Neither is resolved at n = 3, and the sign is negative
at both LRs and on every metric. **It does not beat arm 2, and nothing suggests a
larger n would reverse the sign.** This is exactly the case "What would count as a
result" pre-registered: *if it lands at arm 2 ± noise, the negative closure of*
`LINEAR_BIAS.md` *§7 gets stronger rather than weaker.* Three distinct placements
of the per-node magnitude feature — alone (arm 3), additive (arm 4), multiplicative
inside the spectral sum (v2) — now all fail to move the 12.2% residual. That
residual is *pairwise* non-linearity, and no per-node feature reaches it.
(**Revised 2026-08-15**: true of the *diagonal* feature all three arms spend, false
of per-node features in general — see the superseded-in-part note above.)

**Comparability, checked run-record against run-record:** every field logged by
`train.py` is identical across `021` arm 2, `023` arm 4 and `025` — model, LoRA,
epochs, batch, `magnetic_m`, `max_nodes`, `question_node`, `data_seed`, all of it.
The only differing fields are the arm flags themselves. This is a clean cross-sweep
read, not an approximate one.

**The gate is not inert — verified, because a null would otherwise be
uninterpretable.** v2 is bit-identical to arm 2 at `g == 1`, which is its
initialisation, so "no effect" and "no gate" produce the same table.
`gate_audit.py` reads the saved `bias_parameters.pt`; `||W2||_F` is **exactly 0.0
at init** and ends at:

| | `bias_lr` 5e-3 | `bias_lr` 2e-2 |
|---|---|---|
| `\|\|W2\|\|_F`, median over 16 layers | 14.6 – 16.0 | 36.6 – 50.9 |
| `\|\|W2\|\|_F`, max | 23.8 – 34.6 | 95.5 – 107.2 |
| `\|b2\|_inf`, median | 0.14 – 0.23 | 0.56 – 0.59 |

Nonzero in **all 16 layers of all 6 runs**. `025` measured a gate that moved.

**Saturation — a HYPOTHESIS, not a finding, and flagged as such deliberately.**
Weights of that magnitude are *consistent with* `tanh` being pinned at ±1, which
would degenerate the gate from smooth per-node routing into a fixed binary channel
mask with a vanishing gradient. Circumstantial support: the 2e-2 cell has ~3×
larger weights and is also where v2's seed sd is **.0265 against arm 2's .0034** —
7.8× the spread, driven by seed 0 at 0.6274. **But the realised gate distribution
was never measured** — `gate_audit.py` reads weights, not activations, and the
inference from one to the other needs a forward pass on real graphs that was not
run. Do not cite this as the mechanism. Confirming it costs ~20 min on one GPU
(hook `gate_mlp[2]`, log the fraction of `|tanh| > 0.99`) and is worth doing only
if the claim needs to be load-bearing somewhere.

The `tanh` bound itself is not in question: it was there to prevent divergence and
there was none, in any run, at either LR — the arm-3/4 failure mode is cleanly
absent. Bounded, however, is not the same as well-conditioned.

**Verdict: stop here. Do not re-test v2.** The obvious rescue — weight decay or a
pre-activation scale on `gate_mlp` to un-saturate it — targets the 2e-2 regime, but
**the 5e-3 cell has ~3× smaller weights and is also negative**, so the failure is
not confined to the regime the rescue would fix. Combined with the sign being
negative at both LRs and on every metric, and with arms 3 and 4 already null, the
expected value of another sweep is low.

**Keep the code.** It appends nothing to Q/K, so it costs nothing to leave in; it
has 26 green tests; and it is the **query-only special case of a node-pair gate**
(`H_K ≡ 1`), which makes it the ready-made control if that design is ever built.

**The one prior this updates.** A node-pair gate `H_Q,il · H_K,jl` is the strictly
larger family — it reaches `c_l(i,j)` factorizing across *both* `i` and `j`, where
v2 reaches only `c_l(i) = g_i Ψ_l`. v2 therefore tested the **query half** of that
design, which is precisely the half that keeps `K_phase = [V_R ‖ V_I]`
parameter-free and head-independent. That half is null. Any node-pair gate's case
now rests entirely on **key-side** gating — the half that gives up the
parameter-free key block and makes it per-KV-group. That is a materially worse
trade than it looked before `025` ran, and it is the live question here, not
whether v2 can be rescued.

**GraphQA (`024` vs `017` C / `013` D, test accuracy, 3 seeds).**

| arm | `bias_lr` | `edge_count` | `node_degree` | `shortest_path` |
|---|---:|---:|---:|---:|
| 1 `magnetic` +selfnode | 5e-3 | 0.425±.022 | 0.971±.015 | 0.946±.012 |
| 2 `magnetic_linear` +selfnode | 5e-3 | 0.433±.042 | 0.963±.014 | 0.925±.007 |
| 4 `magnetic_hybrid` +selfnode | 5e-3 | 0.461±.028 | 0.975±.011 | 0.931±.005 |
| v2 `magnetic_linear_v2` | 5e-3 | 0.450±.033 | 0.958±.010 | 0.934±.003 |
| v2 `magnetic_linear_v2` | 2e-2 | 0.432±.038 | 0.968±.016 | **0.942±.009** |

`edge_count` and `node_degree` are null as predicted — arm 2 is already at 99–102%
of selfnode-matched headroom, so there is nothing to win. `shortest_path` is the
one cell with daylight, and v2 recovers **97.5%** (5e-3) / **99.2%** (2e-2) of the
headroom against arm 2's 95.6% and arm 4's 96.9%. Read this as suggestive only:
n = 3, +0.9/+1.7 pp against a sd of .003–.009, and it is the single cell out of
three that moved. Note the LR ordering is **opposite** to WebQSP's — 2e-2 is v2's
best GraphQA cell and its worst WebQSP cell.

### Why WebQSP did not fit the 2026-08-13 maintenance window

Recorded because "we could have parallelised it" is the obvious question and the
arithmetic is not. A cluster-wide reservation opened at 13:00 with ~60 minutes'
notice. `024` fits trivially — GraphQA runs measure 11–27 min — and its `time` is
set to `00:40:00` **from the window, not from the workload**: at `019`'s
`04:00:00` Slurm would not have started the array at all, since it cannot fit a
4 h job before a reservation an hour away. WebQSP does not fit:

| | |
|---|---|
| measured single-GPU (`023`, 12 runs) | 3:55:31 – 5:21:19, median ~4:52 |
| usable window | ~55 min |
| required speedup | **5.3×** at the median, 5.8× at the slowest |

8-way DDP was **tried and measured** — sweep `026_webqsp_v2_ddp_probe`, job 126003,
TIMEOUT at 38 min having reached step 800/4890 (epoch 2.45). The verdict on the
window is unchanged, but every mechanism proposed before the measurement was
wrong, so the numbers are recorded here in place of the reasoning.

| | single-GPU (`023`) | 8×DDP `bs1/accum1` (`026`) |
|---|---:|---:|
| optimizer steps | 4890 | **4890** (effective batch matched) |
| first step | 179.9 s | 36.1 s |
| **training step, steady** | 1.1 s/it | **~0.22 s/it (≈5× faster)** |
| **end-to-end** | 2.89 s/it | **3.0 s/it (no gain)** |
| projected total | 3 h 55 m | ~4 h 05 m |

**The training steps really are ~5× faster; the entire win is eaten by redundant
compilation.** The diagnosis, from the signature counts rather than from
inference:

> **159 `AUTOTUNE flex_attention` events but only 15 DISTINCT kernel signatures** —
> each appearing 9–12 times, i.e. **once per rank**. All 8 ranks compile the same
> 15 kernels independently, and because Triton codegen is CPU-bound they contend
> for host CPU and largely serialize rather than overlap. At ~6.6 s per autotune
> (`0.18 s benchmarking + 6.43 s precompiling for 6 choices`), 159 × 6.6 ≈ **1050 s**
> — which accounts for the wall-clock that the per-step rate does not.

Everything else is behaving exactly as designed, and three tempting explanations
are ruled out by the same data:

* **No shape explosion and no cache thrash.** The 15 signatures are precisely the
  documented `len_buckets × node_buckets`: 5 sequence buckets
  (512/1024/1536/2048/3072) × 3 node-bias buckets (the `1x32x256x256` /
  `1x32x512x512` tail of the signature). `flex_cache_size_limit: 32` sits
  comfortably above 15. **`batch_size 1` does NOT break the collator's bucketing** —
  an earlier revision of this section claimed it did, on 10 steps of evidence.
* **Eval is not the bottleneck.** `eval_webqsp_runtime` is **1.8–1.9 s** after the
  first (that one costs 159 s, compiling the eval shapes).
* **Not Amdahl on a serial head, and not allreduce.** Warmup is real but bounded;
  the cost is the *count* of redundant compiles, not a fixed serial prologue.

**The fix, and it already exists in the infrastructure.** `sweep/execute.py`
supports `execution.sbatch.inductor_cache`, a compile cache shared across a
sweep's runs, "used only if it already exists". The 15 signatures depend on the
collator's buckets, **not** on rank count or batch size, so a cache primed by any
single-GPU WebQSP run is valid for the DDP ranks. Priming it would delete ~17 min
of the ~30 min run and let the 5× step speedup reach the wall clock. **Do that
before the next windowed DDP attempt** — at either 8×`bs1` or the cleaner
**4 ranks × `bs 2` × `accum 1`**, which preserves per-GPU batch as well as
effective batch.

Two things that turned out **not** to be blockers, both confirmed live:

* The sampler trap. `sweep/execute.py` warns `gpus_per_config > 1` is meaningless
  without a distribution-aware sampler; neither trainer overrides
  `_get_train_sampler`, so HF's default `DistributedSampler` applies. All 8 ranks
  trained and the step count matched single-GPU exactly.
* Compile cost per shape. The ~320 s figure in the configs is the **context 4k**
  shapes; WebQSP's 1024-length shapes autotune in 6.4 s.

Incidental, and free throughput whenever DDP is used for real:
`find_unused_parameters=True` is set in the DDP constructor while PyTorch reports
no unused parameters actually exist — an extra autograd-graph traversal every
iteration on every rank.

Comparability is a separate and *weaker* objection than an earlier revision
claimed. Effective batch is 8 either way and the step count is identical, so the
gradient is a mean over 8 samples per optimizer step in both; what differs is
which 8 samples are grouped (strided across ranks vs. consecutive micro-batches)
and bf16 accumulation order. Still, no arm-2 control has been run under a DDP
recipe, and arm 4 − arm 2 is 0.15 pp against a within-cell sd of 0.19–0.89 pp —
so `026`'s header still forbids tabling its metrics beside `021`/`023`.

### What `026` did establish: arm v2 does not diverge

The probe's actual purpose. At `bias_lr` **5e-3**, reaching **epoch 2.45** with
**zero NaN**, and `grad_norm` *falling* over the run — 19–35 at epoch 1.84 down to
6.9–8.2 at 2.45, the opposite of the runaway signature in "Divergences".

That clears divergence **#4** (`magnetic_hybrid`, the same `bias_lr` 5e-3, dead at
epoch **1.77**) and effectively #2 (2.48); #3 at 3.17 was not reached. It is the
first evidence on real data that the `tanh` bound does what §8 claims, and it is
why `025` was left to run its overnight slot rather than being held for review.

`025` ran behind the reservation (job 125997) and completed 6/6 on 2026-08-14,
01:00–04:34, 2:55–3:33 each. `024` is job 125979, all 18 tasks concurrent,
11–21 min each, 18/18 COMPLETED. Results in "Arm v2 — result" above.

## Divergences (pre-§5.8, runs with no metric)

**Historical.** Everything in this section describes the *withdrawn*
parameterization and is retained as the record of why §5.8 exists. All four cells
below have since been re-run under the normalised form and survive — see
"Stability" above. Do not read these as live failures.

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
withdrawn, not a baseline for what replaces it. Arms 0-2 are unchanged, so `021`
is the WebQSP reference (`018` recorded nothing).

### The context event — never diagnosed, and now moot

The context event did not look like the WebQSP one: its **loss stayed at
1.23-1.45** through grad norms of 7e4, and it *recovered* to grad_norm 11 before
dying. A pure runaway does not recover. Diagnostic job 125240 was to separate "the
hybrid fails by a second route" from "clipping bounds each update while the state
drifts" — it was **cancelled at 0:00 elapsed** and never ran.

It was not resubmitted, and the question is now unanswerable as posed: §5.8 changed
the parameterization the event occurred in. What replaced the diagnostic is weaker
but sufficient for the decision — the WebQSP cells that reliably died survive under
the new form with the gain bound flat. **The context arm-4 cell itself has not been
re-run**, so the second-route hypothesis is untested rather than refuted.

**Cost exposure — resolved.** The 30 magnitude-bearing runs feared lost were not
lost; under §5.8 all 52 arm-3/4 runs recorded a metric. The cost that did
materialise was elsewhere: `018`'s 12 baselines, killed by `mem: 64G`, which `021`
re-ran at 128G.


## Reading the result (`MIXED_BIAS.md` §5.6) — RESOLVED

§5.6 pre-registered three readings. The third is the one that landed.

* ~~**Arm 4 within seed noise of arm 1** clears the path to build the $O(N)$ flex
  backend and delete the `score_mod`.~~ **No.** Arm 4 recovers 88.3% of WebQSP
  headroom against arm 1's 100%; the residual is far outside seed noise (3.1 pp gap
  against sd 0.9 pp).
* ~~**Arm 3 alone is the diagnostic.**~~ **Split by task, which §5.6 did not
  anticipate.** Magnitude alone is at the floor on WebQSP (1.8%) and recovers ~100%
  on GraphQA `node_degree` and `edge_count` but only 39% on `shortest_path`. So
  directed flow *is* what the magnetic bias is selling **wherever the answer is a
  path**, and is simply not needed for local counting. The phase channel is
  load-bearing; the tandem's gain over it is not additive but ~zero.
* **Arm 4 ≈ arm 2 — CONFIRMED.** +0.15 pp on WebQSP, +1.5 pp on GraphQA, neither
  resolved. The magnitude channel is redundant with the phase channel, and the
  residual against arm 1 is pairwise non-linearity no factorization reaches. This
  closes `LINEAR_BIAS.md` §7 **negatively**, which is the result.

Both standing caveats held. Context landed at ceiling for arms 1 and 2, so it reads
as "every factorizable arm reaches the ceiling in this regime". And **arm 3's
predicted weakness was confirmed on its GraphQA analogue**: `shortest_path`, the one
task whose discriminating information is relative position along a chain, is exactly
where arm 3 collapses (0.655 vs. 0.931). The prediction was made for context by
construction of `data.py`; context arms 3/4 never ran, so this is the analogue
rather than the test itself.

## Still open

1. **Context arms 3/4 under §5.8.** `020` is baselines-only and there is no
   follow-up config. This is the T=4096 regime and the only one where arms visibly
   separate on robustness, so it is where a hybrid advantage could still hide — and
   it is the untested half of the second-route hypothesis above. Given WebQSP and
   GraphQA agree, it is confirmatory rather than decisive.
2. **Statistical power.** Resolving a 1.5 pp effect at GraphQA's 2.9 pp seed sd
   needs ~30 seeds per cell. Nothing here can call an effect that size; the 5/5
   same-sign aggregate is undercut by the paired-by-seed data (6/9).
3. **The M-scaling curve.** The +0.43 pp/64-dims figure is a single 64→128 step on
   one dataset. If head width binds in the real backbone, sweep phase-only over
   M ∈ {32, 64, 128, 256} to find where it flattens. Worth more than anything
   remaining on arms 3/4.
4. **`edge_count`.** The only cell where arm 4 beat even the dense ceiling (0.461
   vs 0.425). Probably noise at sd 2.8 pp, but it is the one counting-shaped task
   and the magnitude channel is a counting-shaped feature.
5. **Key-side gating.** The only untested member of this family, and the one v2's
   null pushes onto rather than closes — see "Arm v2 — result". It is not free:
   it costs `K_phase`'s parameter-free, head-independent structure, so it should
   be priced against the M-scaling curve (item 3) before it is built, not after.
6. **`shortest_path` at `bias_lr` 2e-2.** v2's 0.942±.009 is the closest anything
   factorized has come to the dense 0.946, and it is the one cell in `024` that
   moved. n = 3 and one cell of three, so it is a lead, not a result — and note
   the LR ordering is *opposite* to WebQSP's, which is itself unexplained.
