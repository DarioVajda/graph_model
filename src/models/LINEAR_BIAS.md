# Linear Magnetic Laplacian Bias for GTLM

## Status

| | |
|---|---|
| **Now** | Measure what linearizing the magnetic bias costs, on the **existing** backbone. |
| **Deferred** | The $O(N)$ factorized implementation. It needs a purpose-built backbone; §7 records the constraints so they are not re-derived. |

Order of work: **§4 Phase 0** (offline, gates everything) → **§5 Phase 1**
(implement `magnetic_linear` and *prove it correct* with tests) → **§6 Phase 2**
(the training comparison). Tests precede the sweep deliberately: a wiring or
indexing bug does not announce itself in a training curve, it just produces a
plausible-looking negative result. GPU-days are only spent on a bias that has
already been shown to be the bias we think it is.

**Both measurement phases sweep $M$** (the eigenvector count, `magnetic_m`), for
the reason in §2.6: quality-vs-$M$ is a property of the *math*, so the deferred
factorized backend inherits this curve instead of re-running the sweep, and only
has to measure speed and memory against it. Test 12 is what licenses that reuse.

Nothing here commits to the optimized implementation. If linearization is too
expensive in quality, §7 never happens and the cost was two offline scripts, one
module, and one sweep.

---

## 1. Motivation

`MagneticBias` (`bias.py:140`) builds an explicit `(B, N, N, magnetic_dim)`
per-edge spectral tensor and pushes it through a 2-layer MLP. At N=4096,
`magnetic_dim=32`, bf16 that hidden is ~1.07 GB *per layer per batch element*,
and the `(B, H, N, N)` output another ~1.07 GB. `checkpoint_graph_bias`,
`magnetic_groups`, and the flex `score_mod` all exist to manage a tensor that
would not have to exist if the bias were linear.

If the bias is **linear** in the spectral features it becomes a bilinear form in
the eigenvectors, and every bilinear form is a dot product — which attention
already computes. The N² object then disappears entirely into wider Q and K.
That is the prize; this document measures its price first.

The measured cost profile that motivates it (`flex_attn/README.md` §3, 512×32,
k=0): the bare masked kernel (`--bias-mode none`) is **19.6 ms fwd / 38.1 ms
bwd**; adding the gather takes fwd to 85.4 (+77%), adding the scatter takes bwd
to 300.0 (+77%). Flex's k=0 overhead is **~86% bias machinery**, and the bare
kernel is only ~1.35× the work-matched `flash_nc` floor. The single largest line
item — the atomic scatter-add into the bias grad, same-address contention
scaling as tokens-per-node² — **ceases to exist** under a factorization, because
the gradient reaches the weights through a dense `(H, N, 2k)` tensor instead.

---

## 2. Mathematical formulation

### 2.1 Inputs

* $N$ — nodes. $M$ — eigenvectors kept (`magnetic_m`; 0 = all N).
* $d_{mag}$ — `magnetic_dim`, width of the DeepSets eigenvalue features.
* $H_Q$ / $H_{KV}$ — query / KV heads ($H_{KV} \le H_Q$ under GQA).
* $V_R, V_I \in \mathbb{R}^{N \times M}$ — real/imag parts of the eigenvectors.
* $\Phi \in \mathbb{R}^{M \times d_{mag}}$ — eigenvalues after DeepSets (`_phi`).

### 2.2 What the current bias computes

The four einsums at `bias.py:205-208` are the real and imaginary parts of the
Hermitian outer product

$$K(i,j) = \sum_{l} V_{i,l}\,\overline{V_{j,l}}\,\Phi_l$$

(with $V_{i,l}=a+bi$, $V_{j,l}=c+di$, the product is $(ac+bd) + i(bc-ad)$ —
exactly those four terms). The MLP is then applied to $[\mathrm{Re}\,K \,\|\,
\mathrm{Im}\,K]$.

### 2.3 The linear replacement

Replace `proj` with a single $W = [W_R; W_I] \in \mathbb{R}^{2d_{mag} \times H_Q}$:

$$b^{(h)}(i,j) = \sum_c W_R[c,h]\,\mathrm{Re}\,K_c(i,j) + W_I[c,h]\,\mathrm{Im}\,K_c(i,j)$$

Define the $O(M)$ eigenvalue projection $\Psi_R = \Phi W_R$, $\Psi_I = \Phi W_I
\in \mathbb{R}^{M \times H_Q}$. Substituting:

$$b^{(h)}(i,j) = \sum_l \big[(V_R^{i,l}V_R^{j,l} + V_I^{i,l}V_I^{j,l})\Psi_R^{l,h}
+ (V_I^{i,l}V_R^{j,l} - V_R^{i,l}V_I^{j,l})\Psi_I^{l,h}\big]$$

### 2.4 The factorization (verified)

Collecting the terms above by what they multiply on the $j$ side gives, with
$\tilde\Psi^{(h)}_R = \mathrm{diag}(\Psi_{R,:,h})$ and likewise for $I$:

$$Q_{struct}^{(h)} = \Big[ (V_R \tilde\Psi_R^{(h)} + V_I \tilde\Psi_I^{(h)}) \;\|\; (V_I \tilde\Psi_R^{(h)} - V_R \tilde\Psi_I^{(h)}) \Big] \in \mathbb{R}^{N \times 2M}$$

$$K_{struct} = \big[ V_R \;\|\; V_I \big] \in \mathbb{R}^{N \times 2M}$$

so that $b^{(h)}(i,j) = \langle Q_{struct}^{(h)}[i], K_{struct}[j]\rangle$.

**Every learned parameter lands on the query side**, so $K_{struct}$ carries no
head dimension. It is a universal structural dictionary that broadcasts across
GQA groups with no parameter conflict, while per-query-head expressiveness is
fully retained. This is the property that makes the whole approach GQA-native.

### 2.5 What linearization actually costs — two separate ceilings

Do not conflate these. They bind in different regimes and Phase 0 reports them
separately.

**(a) Rank ceiling.** $\mathrm{rank}(B^{(h)}) \le \min(N, 2M)$ by construction.
The MLP starts from the same rank-$M$ kernel but its pointwise SiLU lifts the
effective rank toward full.

Values below are the **actual recipes** (read from the g-sweep job scripts):
both use `--magnetic-dim 128 --magnetic-m 128`.

| target | N | `magnetic_m` | $M$ | cap $2M$ | binding? |
|---|---:|---:|---:|---:|---|
| context | 16–128 | 128 | $=N$ | $2N$ | **never** — the eigenbasis is complete |
| WebQSP | median **52**, mean ~115, max 512 | 128 | $\min(128,N)$ | 256 | **only on ~12% of graphs** |
| trunk | 4096 | 32 | 32 | 64 | **severely** (64×) |

The WebQSP row is measured, not assumed (`num_nodes` over dev+test): `--max-nodes
512` is a *cap*, and the realised distribution sits far below it. Median $N$ is
52–62; only **26%** of graphs have $N>128$ (where $M$ truncates at all) and only
**11–14%** have $N>256$ (where the rank cap $2M<N$ can bind). So for the typical
WebQSP graph the eigenbasis is complete and ceiling (a) is inactive — the same
regime as context, not the "mildly 2×" one an earlier reading of the cap
suggested.

**(b) Bilinear-family ceiling.** Even where rank does not bind, the bias must be
*this* form — $V\,\mathrm{diag}(\Psi)\,V^*$ — not an arbitrary rank-$2M$ matrix.

On WebQSP and context at their shipped $M$, **(b) is the binding constraint** —
(a) is barely or not at all active. Phase 0 must therefore not report a single
"rank loss" number; it sweeps $M$ (§2.6), which is what walks the configuration
into the regime where (a) does bind.

**Parameters.** At $d_{mag}{=}128$, $H_Q{=}32$: the linear head $W$ is
$2 d_{mag} H_Q = 8192$, against the MLP head's `proj[0]` $(256{\times}128)=32768$
plus `proj[2]` $(128{\times}32)=4096$ — **36 864 total, so $W$ is ~22% of it**.
`lambda_lin` and `deep_set` are untouched in both. State this honestly: the arm
removes a nonlinearity *and* 4.5× of the head's parameters, so a loss is not
attributable to nonlinearity alone. (Both are negligible against the model —
the whole bias is ~0.015% of parameters — so widening $d_{mag}$ is available as
a follow-up if capacity, not form, turns out to be the binding cost.)

### 2.6 Why both phases sweep $M$

$M$ is not a detail to be fixed at its incumbent value — it is **the knob that
sets the cost of the deferred backend**, because the factorization concatenates
$2M$ dimensions onto each head.

| $M$ | extra dims $2M$ | vs Llama-1B `head_dim` = 64 |
|---:|---:|---|
| 8 | 16 | +25% |
| 16 | 32 | +50% |
| 32 | 64 | **2×** |
| 64 | 128 | 3× |
| 128 (incumbent) | 256 | **5×** |

At the shipped $M{=}128$ the "optimization" would quintuple the head width — so
the incumbent is almost certainly *not* the operating point of the optimized
version. The sweep is therefore not curiosity about a degradation curve; it
finds **the smallest $M$ that holds quality**, which is the number the backend
gets built around.

The decisive property: **quality-vs-$M$ is implementation-independent.** The
factorized backend computes the same bias by different arithmetic (test 12 pins
this to fp64), so it inherits this curve and only needs speed/memory-vs-$M$
measured. Running the sweep now on the existing backbone means it is never run
again.

**Regimes differ between the two datasets** — do not expect one curve to predict
the other. On context $M{=}N$, so the basis is complete and reducing $M$ is a
*first* truncation. On WebQSP $M{=}128 < N{\le}512$, so it is a *further* one.

**Logistics — $M$ costs nothing on the data side, if a trap is avoided.**
`magnetic_m` is part of `data_config_key` (`experiments/kgqa/config.py:430`), so
sweeping `--magnetic-m` naively rebuilds the ~3.4 GB WebQSP dataset per value.
It does not have to: `eigh` returns **ascending** eigenvalues and truncation is a
prefix slice (`utils/magnetic_lap.py:84-87`), while the collator independently
truncates to `min(stored_m, magnetic_m)`
(`utils/text_graph_collator_v2.py:355-374`). Slicing the stored-128 dataset to 32
at collate time is therefore **bit-identical** to a dataset built at $m{=}32$.
Required change: a **collator-only $M$ override that stays out of the cache
key**, so the whole sweep runs off the existing cached data.

> Naming: this repo uses $M$ (`magnetic_m`) for the eigenvector count and $K$
> (`k_hop`) for hop distance — see `experiments/graphqa_mag_khop`. Written as $M$
> throughout so this is not read as a k-hop sweep.

---

## 3. Why SPD is excluded from the candidate arm

`SPDBias` is a learned lookup keyed by shortest-path distance — a genuinely
pairwise quantity with no $f(i)\cdot g(j)$ decomposition. It cannot ride along in
the concatenation, so an arm that keeps it would price a recipe the optimization
can never reach.

This makes the comparison two-variable, which §6 handles with an explicit pivot
arm. (Partial recovery later: `node_position_mode="spd_depth"` is a *node*
property and does survive factorization.)

---

## 4. Phase 0 — offline measurement, no training

Answers "how much does the bilinear family and the rank cap cost?" from already
trained checkpoints. Hours, no GPU-days, and it gates whether Phase 1 runs.

**Inputs.** Bias weights live in `bias_parameters.pt` (keys
`…self_attn.graph_bias.bias_modules.{i}.{lambda_lin,deep_set.0,proj.0,proj.2}`),
**not** in `adapter_model.safetensors`, which is LoRA-only.

* WebQSP — `checkpoints/kgqa/002_webqsp_g_sweep_*` (3 seeds)
* context — `checkpoints/context/003_context4k_g_sweep_*` (3 seeds)

**P0a — constrained fit (the binding number).** On real batches, per layer and
head, least-squares fit $W$ to reproduce the trained MLP's bias output. Report
$R^2$ and residual in units of the bias's own std. This measures ceiling (b)
directly, at the true parameter budget.

**P0b — rank spectrum (the diagnostic).** SVD the trained $B^{(h)}$; report
energy captured by the top $2M$ singular values. This is what extrapolates to the
1024–4096-node trunk.

**Every measurement is run over the $M$-grid $\{8,16,32,64,128\}$** (§2.6), by
truncating the stored eigenvectors — free, no reprocessing, no training. Report
$R^2(M)$, KL$(M)$ and spectral energy$(M)$ as curves, not points.

**P0c — attention-level effect.** $R^2$ on the bias is not the quantity that
matters; the softmax is. Report the KL between attention distributions under the
true and the fitted bias, on real batches. A large bias residual in a saturated
row is harmless; a small one in a flat row is not.

**Gate.** If P0a shows high $R^2$ and P0c small KL on both datasets, the
linearization is predicted to hold and the Phase 2 GPU-days are justified. If
they disagree between WebQSP and context, that difference is itself the result
and Phase 2 targets the worse one.

**Phase 0 also prunes the Phase 2 grid.** An $M$ that is already destroyed
offline does not get GPU time; §6.1's grid is whatever survives here.

### 4.1 Results — WebQSP (3 seeds, 64 graphs, all 16 layers)

| $M$ | $R^2$ (median seed) | resid / bias std | worst layer |
|---:|---:|---:|---:|
| 8 | 0.9729 | 0.165 | 0.703 (L6) |
| 16 | 0.9663 | 0.184 | 0.660 (L6) |
| 32 | 0.9629 | 0.193 | 0.670 (L5) |
| 64 | 0.9610 | 0.197 | 0.650 (L5) |
| 128 | 0.9603 | 0.199 | 0.650 (L5) |

Three findings, and the second is the load-bearing one.

**(a) Rank is inactive — decisively.** P0b: **100%** of the trained bias's
spectral energy lies inside the rank-$2M$ cap, and the matrices are nearly
rank-1 — 90% of energy in the top **2** singular values, 99% within **22**.
The factorization's rank budget is enormous relative to what the bias actually
uses. Ceiling (a) is not a constraint at these scales and the $M$-sweep's cost is
truncation of *information*, not of rank.

**(b) The nonlinearity is nearly idle — except in one layer.** Per-layer $R^2$ at
$M{=}128$ (seed 1): thirteen of sixteen layers sit at **1.000**; the exceptions
are **L5 ≈ 0.55**, L6 ≈ 0.92, L14 ≈ 0.97. So a linear head reproduces most of
this trained bias exactly, and the MLP's SiLU is doing real work in essentially
one layer. Report the worst layer, never the mean: the mean (0.96) reads as
"linearization is nearly free" and hides precisely the layer where it is not.

**(c) $R^2$ is almost flat in $M$** (0.973 → 0.960 across a 16× range). The cost
of linearizing barely depends on the eigenvector count, which is a positive
signal for choosing a small $M$ for the deferred backbone: shrinking $M$ does not
make linearization *additionally* worse. It says nothing about the absolute
quality of a model trained at small $M$ — that is Phase 2.

Phase 0 therefore does **not** prune the grid: no $M$ is disqualified, so §6.1
runs the full $\{16,32,64,128\}$.

---

## 5. Phase 1 — implement `magnetic_linear`, then verify it

Deliberately **not** optimized: same backbone, same dense `(B,H,N,N)` bias, same
eager/flex path. Only the head changes, so any quality delta measured in §6 is
attributable to the math and not to an implementation.

**No training job is submitted until §5.3 is green.**

### 5.1 Implementation

`LinearMagneticBias(MagneticBias)` with `config_key = 'magnetic_linear'`, appended
to `BIAS_TYPES` — the documented extension protocol (`bias.py:8-12`). It reuses
`_phi` unchanged and replaces `proj` with a single linear layer, zero-initialised
like `proj[2]` so the bias starts at exactly 0.

> The old plan's `bias_type = "linear_magnetic"` flag is stale: there is no
> `bias_type` mechanism outside `legacy/`. Naming follows the existing
> `magnetic_shared` / `magnetic_content` / `magnetic_groups` family.

**Wiring — the silent-failure surface.** `cfg.magnetic` currently gates the model
bias *and* the dataset feature (`context/process_dataset.py:68`), `magnetic_m`
(`context/model.py:103`), and `WIRED_FEATURES` (`context/config.py:49`). A flag
that misses those yields `magnetic=None` → `forward` returns `None` → a run that
trains cleanly with **no bias at all** and looks like a clean negative result.
Every gate must read `magnetic or magnetic_linear`, config validation must
**raise** on `magnetic_linear` without features rather than returning None, and
§5.3 test 10 pins it.

### 5.2 Correctness gate — what must pass before any sbatch

These four are the ones that can silently invalidate §6, so they are the gate:

* **Permutation invariance** (test 1) — a $\Psi$-indexing or $V_R/V_I$ split bug
  breaks this and almost nothing else.
* **No silent no-op** (test 10) — the failure mode that would make §6 report a
  clean-looking negative for a run that had no bias at all.
* **Zero-init inertness** (test 9) — pins the step-0 starting point.
* **Backward compatibility** (test 3) — the new type is fully inert when off.
* **$M$-truncation equivalence** (test 14) — if collate-time truncation is not
  equivalent to building at $m{=}M$, every §6 $M$-curve is mislabelled.

The rest of §5.3 runs too, and any failure is fixed before submission; these four
are simply the ones whose failure is invisible downstream.

### 5.3 Test plan — existing backbone, `magnetic_linear` enabled

**Existing suites that matter, in priority order:**

1. `tests/models/test_flex_cpu.py` — **eager permutation invariance**. The
   sharpest structural check: relabelling nodes must not change prompt logits. A
   bug in the $\Psi$ indexing or the $V_R/V_I$ split breaks this and little else.
2. `tests/models/test_v2_ragged_magnetic_padding.py` — padded eigenvector slots
   must not leak. The new $\Psi$ projection is a fresh place for padded
   eigenvalues to enter; directly on point.
3. `tests/models/test_modeling_gtlm_llama_v2.py` — **backward compatibility**:
   with `magnetic_linear=False` everything must be bit-identical to today, i.e.
   the new type is fully inert when off.
4. `tests/models/test_flex_attention.py` (GPU) — flex-vs-eager parity, **bias
   checkpointing** (the new module must be recompute-safe under
   `checkpoint_graph_bias`), permutation invariance.
5. `tests/models/test_graph_bias.py` — registration, accumulation with other bias
   types, K-hop gate, `expand_node_to_token_bias`.
6. `tests/test_collator_bucketing.py` — fp64 padding loss-neutrality.
7. `tests/models/test_bias_regularization.py` — the trainer's shape-based
   weight-decay rule. $W$ is 2-D but semantically a projection; confirm it is
   classified as intended.
8. `tests/models/test_bias_sharing.py` — only if `magnetic_groups` compatibility
   is wanted. Otherwise assert mutual exclusion in config, as `magnetic_groups`
   already does.

**New tests:**

9. **Zero-init inertness** — at step 0 the bias is exactly 0 and logits match the
   no-bias model.
10. **No silent no-op** — `magnetic_linear=True` with features absent must
    **raise**, not return `None` (§5.1).
11. **Folded/unfolded parity** — the folded `_folded_spectral` path equals the
    naive $[\mathrm{Re}\|\mathrm{Im}] @ W$ path to fp64, mirroring
    `legacy_unfolded`.
12. **Factorization parity (fp64, CPU, no kernel)** — $\langle Q_{struct},
    K_{struct}\rangle$ reproduces the dense linear bias off-diagonal. Pure math,
    cheap, and it de-risks all of §7 before any backbone work starts.
13. **Save/load round-trip** — `bias_parameters.pt` carries $W$.
14. **$M$-truncation equivalence** — collate-time truncation to $M$ produces
    bit-identical `magnetic_V` / `magnetic_lambdas` to a dataset *built* at
    $m{=}M$, and the collator-only override does **not** perturb
    `data_config_key`. §6's entire $M$-grid rests on this; if it is false the
    sweep is measuring a different truncation than it reports. Also assert
    `_phi`'s `n_valid` normalisation tracks the truncated count (`bias.py:184`).

---

## 6. Phase 2 — the training comparison

Runs **only** after §5.2 is green. This is the phase that costs GPU-days.

### 6.1 Arms and the $M$-grid

| arm | config | $M$ | isolates |
|---|---|---|---|
| A | `magnetic + spd` | 128 | incumbent (recipe of record) — reuse g-sweep |
| B | `magnetic` alone | 128, 16 | A→B = cost of dropping SPD; **B's own $M$-curve** |
| C | `magnetic_linear` alone | 16, 32, 64, 128 | **B→C = cost of linearization, per $M$** |
| D | no soft bias | — | headroom denominator |

Report B→C as the headline and as a fraction of the (D→B) magnetic headroom, not
as raw pp. A and D are reusable from the existing g-sweeps **only** if the recipe
lineage matches exactly; otherwise re-run.

**Why B is swept too, and not just pinned at 128.** Without a B point at low $M$,
a drop in C at $M{=}16$ is unattributable — linearization or simply fewer
eigenvectors? The two are expected to differ in shape precisely because the rank
ceiling $2M$ binds C and not B, which is the whole reason to suspect the linear
form needs a larger $M$. Two B points bracket the grid for one extra config.

**Cost.** 6 configs × 3 seeds × 2 datasets = **36 runs**, at ~2.8 h (WebQSP) and
~3.5 h (context) per run measured from the g-sweeps via `sacct`. One array-job
wall-clock given cluster width, and Phase 0 is expected to prune the grid first.

### 6.2 Protocol

* Datasets: WebQSP (`002_webqsp_g_sweep` recipe) and context
  (`003_context4k_g_sweep` recipe). 3 seeds, median-seed rule.
* **Replay, do not retype** recipes: read the sweep's own `results/<sweep>/jobs/*.sh`
  and push it back through `build_parser` → `config_from_args`, overriding only
  the bias flags — the chain `tests/experiments/test_magnetic_groups_cli.py` pins.
* Same budget as the arm each is compared against. Floored runs here stagnate then
  jump; extend the budget rather than calling a floor early.
* Capture sbatch job ids at submission; an unchecked id is how a sweep silently
  fails to exist.
* **Do not pass `--magnetic-m` to vary $M$** — it is in `data_config_key` and
  would trigger a full dataset rebuild per value. Use the collator-only override
  from §2.6; the cached $m{=}128$ dataset serves every arm.

---

## 7. Deferred — the factorized backbone

Not scheduled. Recorded so the constraints are not re-derived. Each was
identified against the current code and each is a real blocker, not a nicety.

1. **FlashAttention-2 is not the target.** GTLM's mask is bidirectional-prefix +
   causal prompt node + padding + optional k-hop; FA2's vocabulary is
   causal/sliding-window/varlen and cannot express it (`README.md:162`). Removing
   the bias does not make FA2 reachable. The real target is **flex**: `BlockMask`
   already handles the mask, and the factorization deletes the `score_mod` and the
   N² tensor. State the goal as "delete the score_mod", not "enable FA2".
2. **Softmax scale.** Kernels derive $1/\sqrt{d}$ from the *passed* head_dim.
   Growing $d_{head} \to d_{head}+2M$ silently rescales pretrained content
   attention by $\sqrt{d/(d+2M)}$. Pass an explicit `softmax_scale` and fold the
   correction into $W$ deliberately.
3. **The diagonal mask is not expressible.** `_finalize` (`bias.py:211`) zeroes
   $b_{ii}$; the factorized form gives $q_i \cdot k_i \neq 0$ with no way to
   subtract it. **Phase 2 should therefore also ablate the diagonal mask**, so the
   future delta is not confounded with it.
4. **"$O(N)$" is conditional on truncation.** The extra width is $2M$. Untruncated
   ($M=N$) the method degenerates to $O(N^2)$. At $M{=}32$, $2M{=}64$ equals
   Llama-1B's whole `head_dim`: expect ~1.6–1.9× on the attention step from the
   doubled $d$, against `--bias-mode none`'s numbers — not `none` for free.
5. **`magnetic_content` factorizes better than expected.** `proj[2]`
   (`bias.py:316`) is linear over `[spectral ‖ Zu ‖ Zv]`, and the $Z_u$/$Z_v$ terms
   are row-only and column-only — each rank-1, costing **2 extra head dims total**,
   not $2 d_{proj}$. Only the SiLU on the spectral part blocks it, and the linear
   variant drops that anyway.
