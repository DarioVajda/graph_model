# factorized_rwpe — Phase 0 feature diagnostic (COMPLETE, 2026-08-15)

## Status

| | |
|---|---|
| **Question** | Does the per-node encoding `FACTORIZED_RWPE_BIAS.md` §3 proposes carry information about WebQSP graphs at all, before any head is built? |
| **Answer** | **No, as written.** 94.5% of WebQSP nodes — and 97.0% of gold-answer nodes — receive an identically-zero feature vector. Every variant that is *not* zero loses to a free degree baseline. |
| **Cost** | two CPU jobs, 11 min and 13 min (126659, 126663). No GPU, no training. |
| **Verdict** | Do not build the head as specified. The one component with incremental value over degree is the *undirected* block, which is not in the doc. |

Run: `./src/experiments/bias_experiments/factorized_rwpe/sbatch_diagnostic.sh`
(`DIAG_ARGS="--only webqsp"` to skip GraphQA). Results in `results/`.

Data: the WebQSP cache `021`/`023` trained on
(`sr-webqsp_..._nmax50_ver8_spd64_magq0.25m128_len1024_rcm1_seed42_dfv3`),
20 856 train graphs / 2.27 M nodes, median 55 nodes per graph; plus the 1 628-graph
test split and three GraphQA tasks.

---

## What is measured, and why it is measured here

The proposed pipeline is `p_i -> MLP_PE -> z_i -> appended to Q and K`. This stops
at the first arrow. Everything after `p_i` is a design choice that can be iterated;
`p_i` is fixed by the data, and two nodes with the same `p_i` are assigned the same
`z_i` by any deterministic head at any width. A degeneracy found here is permanent.

This is deliberately **not** the Phase 0 `linear_bias` ran. That one fit the trained
bias offline, and its own Conclusion 6 is that offline imitation R² did not predict
trained quality — it measured how well a head could FIT a target. This measures
whether the INPUT separates nodes at all.

## Feature sets

| set | `p_i` | note |
|---|---|---|
| `dir` | `[(D_out⁻¹A)^t]_ii` ‖ `[(D_in⁻¹Aᵀ)^t]_ii`, t = 1..24 | the doc as written; a sink's row is left at zero |
| `dir_sink` | same, self-loop on every sink first | `utils/rrwp.py:38`'s convention — what a reimplementation reusing that helper gets by default |
| `undir` | `[(D⁻¹A_sym)^t]_ii`, t = 1..24 | **not in the doc.** 24 dims, not 48: for symmetric `A` the backward series is bit-identical |
| `dir+undir` | the concatenation | 72 dims. Free in attention width — only `MLP_PE`'s input grows |

## Results — WebQSP

`ansZero` = fraction of gold-answer nodes whose whole vector is zero. `degR2` =
between-group / total sum of squares when nodes are grouped by exact
(in-deg, out-deg), i.e. the R² of the **best possible** function of degree.
`pc90` = PCA components for 90% of variance. `ansAUC` = 5-fold CV logistic probe
on `p_i` predicting is-gold-answer, as ranking AUC.

| split | set | dims | zero | ansZero | degR2 | pc90 | ansAUC | +degree |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | `dir` | 48 | **0.945** | **0.970** | 0.456 | 4 | **0.529** | 0.813 |
| train | `dir_sink` | 48 | 0.555 | 0.108 | **0.999** | **1** | 0.774 | 0.811 |
| train | `undir` | 24 | 0.000 | 0.000 | 0.272 | 2 | 0.641 | **0.839** |
| train | `dir+undir` | 72 | 0.000 | 0.000 | 0.299 | 3 | 0.645 | 0.839 |
| test | `dir` | 48 | 0.940 | 0.969 | 0.525 | 4 | 0.525 | 0.785 |
| test | `dir_sink` | 48 | 0.547 | 0.157 | 0.999 | 1 | 0.744 | 0.784 |
| test | `undir` | 24 | 0.000 | 0.000 | 0.259 | 2 | 0.637 | **0.815** |
| test | `dir+undir` | 72 | 0.000 | 0.000 | 0.298 | 3 | 0.637 | 0.814 |

**Baselines on the same nodes** (no random walks, no feature cache):

| baseline | train | test |
|---|---:|---:|
| degree basis — in/out, log, reciprocal, `==0` indicators | **0.806** | **0.777** |
| the single bit `out_degree == 0` | 0.749 | 0.720 |

## Conclusions

1. **`dir` is dead on WebQSP, not weak.** A return probability `(M^t)_ii` is
   non-zero only for a node on a directed cycle, and the Levi graph
   (`entity -> relation -> entity`) rarely has one. 94.5% of nodes get `MLP(0)` —
   the same vector — so the structural term contributes an identical value for all
   of them. Its 0.529 AUC is chance: with 94.5% of rows tied, that blob contributes
   exactly 0.5 and only the live 5.5% can move the number.

2. **The self-loop convention manufactures a feature out of nothing.** `dir_sink`'s
   `degR2 = 0.999` and `pc90 = 1` (of 48) say it is one bit re-parameterized. That
   bit is `out_degree == 0` — "this is a tail entity" — which alone scores 0.720 on
   test against `dir_sink`'s 0.744. A first pass scored it against a *linear probe
   on raw (in, out)* and it appeared to win by 30 pp; that baseline could not
   express a threshold and was simply mis-specified. Under a fair basis the win
   disappears. The raw-linear probe also swung 0.73 train / 0.44 test, which the
   fair basis does not (0.806 / 0.777) — the swing was mis-specification, not
   distribution shift.

3. **Every feature set loses to a degree baseline it should have beaten.** 0.777 on
   test, from two integers per node, against 0.637 for the best RWPE set alone.

4. **The undirected block is the only component with incremental value, and it is
   not in the doc.** Adding it to the degree basis moves test AUC 0.777 -> 0.815
   (+3.8 pp); adding the doc's 48 directed dims on top moves it 0.815 -> 0.814,
   i.e. nothing. Alone, `undir` never dies (0.000 zero rows) but compresses to 2
   effective dimensions of 24, and 55.4% of WebQSP graphs are bipartite, which
   kills every odd-`t` entry (measured liveness at t = 1,3,5,7: 0.000 / 0.037 /
   0.151 / 0.298, against 1.000 at every even `t`).

5. **This agrees with `mixed_bias` rather than adding to it.** `MIXED_BIAS.md` §2.3
   records that `S_i = Σ_l |V_il|² f(λ_l) = [diag f(L)]_i` spans the
   return-probability family, so this proposal is arm 3 (`magnetic_magnitude`) with
   `f = λ^k` on the plain transition matrix instead of the magnetic Laplacian. Arm 3
   measured **1.8% of WebQSP headroom** and ~100% on GraphQA `node_degree`. A
   degree-shaped feature that is dead where the answer is a path is exactly that
   result, seen in the input rather than in the F1.

## GraphQA — what it does and does not validate

Features are alive there (`dir` zero rows 0.143 against WebQSP's 0.945; bipartite
fraction 0.018–0.175), which is the contrast that shows the WebQSP result is a
property of *those graphs* and not of the metric.

It does **not** reproduce arm 3's per-task split (98% / 100% / 39% on
`edge_count` / `node_degree` / `shortest_path`). All three GraphQA tasks share one
graph generator, so the diagnostic sees a near-identical distribution for each
(`degR2` 0.38 / 0.38 / 0.41) and cannot separate them. The split lives in what each
task *asks*, not in the graphs — which is itself the point: a per-node feature
cannot see the difference, and neither can this diagnostic.

## Open, if anyone revisits this

* `k = 24` is unnecessary either way: on WebQSP no `t` beyond ~12 adds liveness, and
  the repo's `max_rw_steps` default is 8.
* The diagnostic scores a **linear** probe. A non-linear readout of `undir` could
  exceed +3.8 pp — but `MLP_PE` feeds a *bilinear* attention term, so the linear
  probe is the closer analogue of what the head can spend it on.
* Nothing here measures the head's spec gaps (no zero-init, no normalization/gain
  against the `MIXED_BIAS.md` §5.7 quartic instability, one shared `W_pos` giving
  every head an identical bias, the `1/sqrt(d_Q + d_pos)` rescaling of pretrained
  content attention). They are listed in the review of the doc, and they only
  matter if the feature question is ever answered differently.
