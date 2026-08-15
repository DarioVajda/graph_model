# Reminder: (1) Double the dimension of Z and split for K and Q; (2) add normalisation to the key and query vectors to avoid explosions

# Evaluation Plan: Decoupled Magnetic Bias (Phase vs. Magnitude)

## Status

| | |
|---|---|
| **Objective** | Isolate and measure the individual and tandem contributions of linear phase features (directed flow) and non-linear magnitude features (structural role) against the standard magnetic baseline (from the $O(N^2)$ implementation). |
| **Method** | Implement `magnetic_magnitude` and `magnetic_hybrid` bias arms. Gate with correctness tests before committing GPU-days. |
| **Next Step** | **Phase 2 is submitted** — 87 runs across `018`/`019`/`020` in `src/experiments/mixed_bias/`. Phase 1 is complete and the §4.2 gate is green. Aggregate with `python3 -m sweep.report` as each sweep lands. |

**There is no offline Phase 0.** `linear_bias` ran one and its own Conclusion 6 is
"offline imitation $R^2$ did not predict trained quality" — $R^2 \approx 0.96$
preceded a 4.91 pp loss, and the dataset with the *worse* fit survived best. A
tandem fit would also no longer be a least-squares problem (the magnitude
features are an MLP output), so it would cost a gradient fit to re-measure a
quantity already shown non-predictive. Phase 1 is the first phase.

---

## 1. Motivation

The pure linear factorization (`magnetic_linear`) maintained basis invariance and
an $O(N)$ memory footprint but lost the non-linear spatial routing of the
original MLP. Read at the configuration a factorized backbone could actually run
(intra-node diagonal **kept** — `linear_bias/README.md` §6), that costs **5.6% of
the magnetic headroom on WebQSP** and **~5.7% on GraphQA**, while the 4k context
task saturates.

To recover the non-linearity without breaking basis invariance or the $O(N)$
footprint, we decouple the bias into two separately factorizable mechanisms:

1. **Phase (linear, pairwise):** preserves directed distances — the $V_R, V_I$
   cross-terms, i.e. the whole of `magnetic_linear`.
2. **Magnitude (non-linear, node-level):** each node's spectral self-energy
   pushed through an MLP, capturing structural role.

The second is the only kind of non-linearity that survives factorization at all:
a bilinear form $\langle f(i), g(j)\rangle$ places no constraint on how $f$ and
$g$ are computed from a *single* node's features, so an MLP is free there. What
it can never reproduce is a non-linearity applied to a *pairwise* quantity, which
is what `MagneticBias`'s SiLU actually is.

**Set expectations accordingly.** The tandem arm is not a universal approximator
of the MLP head and no amount of width makes it one. It adds one non-linear
node-level channel to one linear pairwise channel. The reason to expect that to
be enough is empirical, not structural: P0b measured the trained bias to be
nearly rank-1 (90% of WebQSP's spectral energy in **two** singular values, 99%
within 22), so a modest-rank additive channel is large relative to what the bias
demonstrably uses.

---

## 2. Mathematical Formulation

### 2.1 Notation

| symbol | meaning |
|---|---|
| $N$ | nodes; $M$ — eigenvectors kept (`magnetic_m`) |
| $d_{mag}$ | `magnetic_dim`, the DeepSets eigenvalue width — as in `LINEAR_BIAS.md`. **Never means "magnitude".** |
| $V_R, V_I \in \mathbb{R}^{N \times M}$ | real/imag parts of the eigenvectors |
| $\Phi \in \mathbb{R}^{M \times d_{mag}}$ | eigenvalues after the DeepSets (`_phi`, unchanged) |
| $H_Q / H_{KV}$ | query / KV heads ($H_{KV} \le H_Q$ under GQA) |
| $d_{\text{magnitude-repr}}$ | **internal** width of $\mathrm{MLP}_{magnitude}$ — computed once per node, never touches attention, so it is genuinely free |
| $d_{\text{magnitude}}$ | width $Q_{magnitude}/K_{magnitude}$ append to each head — **not** free, see §2.5 |

### 2.2 Arm A — linear phase (unchanged, `magnetic_linear`)

Reproduced from `LINEAR_BIAS.md` §2.4 for reference; nothing here is new.

* Project the eigenvalue features to per-head scalars: $\Psi_R = \Phi W_R$,
  $\Psi_I = \Phi W_I \in \mathbb{R}^{M \times H_Q}$.
* For query head $h$, with $\tilde\Psi_R^{(h)} = \mathrm{diag}(\Psi_{R,:,h})$
  and likewise for $I$:

$$Q_{phase}^{(h)} = \Big[ \big(V_R \tilde\Psi_R^{(h)} + V_I \tilde\Psi_I^{(h)}\big) \;\big\|\; \big(V_I \tilde\Psi_R^{(h)} - V_R \tilde\Psi_I^{(h)}\big) \Big] \in \mathbb{R}^{N \times 2M}$$

$$K_{phase} = \big[ V_R \;\big\|\; V_I \big] \in \mathbb{R}^{N \times 2M}$$

Every learned parameter is on the query side, so $K_{phase}$ carries no head
dimension and broadcasts across GQA groups. **Arm B must preserve this
property** (§2.3).

### 2.3 Arm B — non-linear magnitude

**The invariance constraint comes first**, because it determines the form.

Taking $|V_{il}|^2$ removes the per-eigenvector *phase* ambiguity
($|e^{i\theta}v|^2 = |v|^2$), which is the entire ambiguity when the spectrum is
simple. It does **not** remove the *degenerate-block* ambiguity: when
$\lambda_l = \lambda_{l'}$, any unitary mixing of that block is an equally valid
eigenbasis, and the per-column magnitudes pick up the cross terms. §3 measures
this — it is large and it fires on ordinary graphs.

The per-node features invariant to *both* ambiguities are exactly the diagonals
of matrix functions of the Laplacian:

$$S_i \;=\; \sum_{l=1}^{M} |V_{il}|^2\, f(\lambda_l) \;=\; \big[\mathrm{diag}\, f(L)\big]_i$$

— **linear in the magnitudes, arbitrary in $\lambda$**. Here $l$ indexes the
**eigenpairs** (columns of $V$), $l = 1 \dots M$, the same index the four einsums
in `_folded_spectral` contract over; $i$ indexes nodes.

*Where the identity comes from.* $L$ is Hermitian, so $L = V \Lambda V^*$ and a
matrix function acts on the eigenvalues, $f(L) = V f(\Lambda) V^*$. Reading off
the $i$-th diagonal entry, $[f(L)]_{ii} = \sum_l V_{il} f(\lambda_l)
\overline{V_{il}} = \sum_l |V_{il}|^2 f(\lambda_l)$ — the cross terms vanish
because $i = j$, which is also why $S_i$ is real. $f = e^{-t\lambda}$ gives the
heat-kernel diagonal and $f = \lambda^k$ the return-probability family, so this
is the structural-role family (RWSE included) generalized to the magnetic
Laplacian.

Two caveats on the identity. $f$ is **vector-valued** ($\Phi_l \in
\mathbb{R}^{d_{mag}}$), so it is $d_{mag}$ independent copies of the sum, one per
channel. And it is exact only at $M = N$: with $M < N$ the sum runs over the
retained eigenpairs, making $S_i$ the diagonal of $f$ on the **low-frequency
subspace** rather than of $f(L)$ itself. Invariance is unaffected — a sum over
whole degenerate blocks is still block-invariant — but truncation must not
*split* a block, which an ascending-eigenvalue prefix slice can in principle do.
That last risk is **pre-existing and shared**: $\sum_l V_{il}\overline{V_{jl}}\Phi_l$
truncated mid-block is basis-dependent in exactly the same way, so `magnetic` and
`magnetic_linear` already carry it at every $M < N$. Arm B does not introduce it
and this plan does not fix it.

*Why the sum is what buys invariance.* Inside a degenerate block $f(\lambda_l)$
is the same vector for every $l$ in the block, so it factors out and what remains
is $\sum_{l \in \text{block}} |V_{il}|^2$ — that block's projector diagonal, which
no unitary mixing can move. Individually the terms move a great deal (§3); their
sum does not.

The non-linearity therefore has to go *after* the pool over $l$, not before it.
Nothing is lost that was well defined: a per-$l$ MLP's extra capacity is
interactions between individual eigenvector magnitudes at one node, and those are
precisely the quantities degeneracy leaves undefined.

This is also how the eigenvalues enter, and taking $f = \Phi$ — the DeepSets
output the phase arm already computes — reuses `_phi` verbatim:

$$S_i \;=\; \sum_l \big[(V_R^{i,l})^2 + (V_I^{i,l})^2\big]\, \Phi_l \;\in\; \mathbb{R}^{d_{mag}}, \qquad
Z \;=\; \mathrm{MLP}_{magnitude}(S) \;\in\; \mathbb{R}^{N \times d_{\text{magnitude}}}$$

$$Q_{magnitude}^{(h)} = Z \odot s^{(h)}, \qquad K_{magnitude}^{(g)} = Z\,W_K^{(g)}$$

with $s^{(h)} \in \mathbb{R}^{d_{\text{magnitude}}}$ a per-query-head **diagonal**
rescaling and $W_K^{(g)} \in \mathbb{R}^{d_{\text{magnitude}} \times
d_{\text{magnitude}}}$ a full mixing per **KV group**, $g(h) = h \,//\, (H_Q/H_{KV})$
— HF's `repeat_kv` convention. $\mathrm{MLP}_{magnitude}$ is
$d_{mag} \to d_{\text{magnitude-repr}} \to d_{\text{magnitude}}$, so
$d_{\text{magnitude-repr}}$ is purely internal.

The per-head bilinear form is $\mathrm{diag}(s^{(h)})\,W_K^{(g)}$ — still full
rank, with per-head diagonal modulation on top of per-group mixing. The diagonal
query side mirrors arm A, where each head's freedom is likewise a diagonal
rescaling of a shared dictionary ($\tilde\Psi^{(h)}$), and it is what keeps this
arm bias-sized: see §2.6.

Three things follow, and they are the reasons for this form rather than the
per-column one:

* $S_i = \mathrm{Re}\,K(i,i)$ — the **diagonal of the kernel the code already
  builds**. It is the "spectral self-energy" `bias.py:_finalize` names, and
  `_folded_spectral` already contains every ingredient. Implementation is
  `(V_R**2 + V_I**2) @ phi`, $O(N M d_{mag})$, no $N^2$ intermediate.
* Parameter shapes are **independent of $M$**, so the $M$-sweep compares like with
  like and a checkpoint trained at one $M$ loads at another.
* $s^{(h)}$ is **per query head** and $W_K$ is **per KV group** — the finest
  granularity GQA physically allows, and it is free. The key tensor is already
  materialized as $(B, H_{KV}, N, d_{head})$, so every group has its own row and
  writing a *different* structural block into each costs no memory and no FLOPs;
  broadcasting one block into all of them would leave capacity unused for nothing.
  Per-*query*-head keys are what §2.2's "no parameter conflict" rules out: the $G$
  query heads in a group share one physical key row, so that would force $G$
  copies and defeat GQA. The group is the ceiling.

  The capacity this buys is real, not cosmetic. With a single shared $W_K$ every
  head would read the *same* projection of $Z$ on the key side and only the query
  side could rotate — under a diagonal query side, that leaves each head a mere
  rescaling of one globally fixed key map. Per-group $W_K$ lets each group choose
  its own key subspace.

  $H_{KV}$ must be read as `getattr(bias_config, 'num_key_value_heads',
  num_heads)`: the repo carries a Bloom backbone with full MHA and no such
  attribute, where per-group degenerates to per-head. That case is fine — at
  $H_{KV} = H_Q = 32$, $W_K$ is 131k/layer (2.1M over 16 layers, ~3.6× the current
  bias head) and still far under the rejected full-$W_Q$ design's 12.1M. Parameter
  count tracks the number of key rows that physically exist, so no width is ever
  paid for a slot the tensor does not already have.

  Arm A is shared for a different reason and the two do not conflict: there
  $K_{phase} = [V_R \| V_I]$ is parameter-free and head-independence falls out of
  the algebra, so there is nothing to make per-group.
* $d_{\text{magnitude-repr}}$ is free while $d_{\text{magnitude}}$ is not: the MLP
  may be wide internally and still hand attention a narrow key.

The pairwise term is

$$b^{(h)}_{magnitude}(i,j) = g^{(h)}\,\big\langle \widehat{Z_i \odot s^{(h)}},\; \widehat{Z_j W_K^{(g)}} \big\rangle,
\qquad \hat x = x/\lVert x \rVert_2$$

— structural role against structural role, with **no relative-geometry content at
all**. That is exactly what the arm is meant to isolate, and §5.6 reads it that way.

**The normalization and the per-head gain $g^{(h)}$ are not cosmetic.** As first
specified this term was $\langle Z_i \odot s^{(h)}, Z_j W_K^{(g)}\rangle$, which is
quartic in the shared trunk and unbounded in all three of its learned factors;
four Phase-2 runs diverged to `NaN` and arms 3/4 were withdrawn. §5.7 is the
diagnosis and §5.8 the remedy as built. Three properties are load-bearing and are
pinned by tests:

* $|b_{magnitude}| \le |g^{(h)}|$, so the channel's range is one auditable scalar
  per head rather than an unbounded product of three tensors;
* $b$ is **exactly invariant** to a uniform rescaling of the trunk, of $s^{(h)}$
  and of $W_K^{(g)}$ — the fix does not depend on knowing which of the three grows;
* $g^{(h)} = 0$ at init, so the bias is still exactly $0$ at step 0. The zero sits
  on the gain and **not** on $s^{(h)}$, because normalizing a deliberately-zero
  vector has Jacobian $I/\varepsilon \approx 10^{12}$ — a step-0 instability in
  place of a step-800 one.

### 2.4 Arm C — tandem

$$Q_{tandem}^{(h)} = \big[\,Q_{phase}^{(h)} \;\big\|\; Q_{magnitude}^{(h)}\,\big], \qquad
K_{tandem}^{(g)} = \big[\,K_{phase} \;\big\|\; K_{magnitude}^{(g)}\,\big]$$

so $b_{tandem} = b_{phase} + b_{magnitude}$, and the dense Phase 1 implementation
is literally that sum.

The magnitude block is normalized **before** the concatenation and the phase block
is not normalized at all. Normalizing the stacked vector would make each channel's
scale a function of the other's — reintroducing at the output exactly the coupling
§5.7 suspects the shared $\Phi$ trunk of already providing. The phase channel needs
nothing regardless: $K_{phase}$ is parameter-free with unit row norm, so its learned
parameters enter at degree 1, and it has never diverged at either LR on either
dataset. Leaving it untouched also keeps arm C's phase half comparable to arm A.

### 2.5 The head-width budget

`LINEAR_BIAS.md` §2.6 makes appended head width the cost model of the deferred
backbone, so it is tracked here too. Against Llama-1B's `head_dim` = 64:

| arm | appended per head | at $M{=}128$ | at $M{=}64$ (**operating point**) |
|---|---|---:|---:|
| phase | $2M$ | 256 (4×) | 128 (2×) |
| magnitude | $d_{\text{magnitude}}$ | 64 (1×) | 64 (1×) |
| **tandem** | $2M + d_{\text{magnitude}}$ | **320 (5×)** | **192 (3×)** |

The right-hand column is what §5.2 runs. 5× head width is not a configuration the
optimized backbone would ship, so measuring there would price a recipe that is
never built — the error Phase 3 caught with the diagonal mask.

$d_{\text{magnitude-repr}}$ does not appear: it is an intermediate width of a
node-level MLP evaluated once per forward, $O(N \cdot d^2)$, and never enters
attention. 256 is fine.

### 2.6 The parameter budget, and why the query side is diagonal

The alternative to §2.3's diagonal $s^{(h)}$ is a full per-head matrix
$W_Q^{(h)} \in \mathbb{R}^{d_{\text{magnitude-repr}} \times d_{\text{magnitude}}}$
projecting a 256-wide $Z$ down to 64. Both append the same 64 dims per head — §2.5
is unchanged either way — so the only difference is learned parameters between $Z$
and the appended vectors. At $d_{mag}{=}128$, $d_{\text{magnitude-repr}}{=}256$,
$d_{\text{magnitude}}{=}64$, $H_Q{=}32$, $H_{KV}{=}8$, 16 layers:

| | full $W_Q$ | diagonal $s^{(h)}$ |
|---|---:|---:|
| $\mathrm{MLP}_{magnitude}$ | $128{\to}256$ + $256{\to}256$ = 98 304 | $128{\to}256$ + $256{\to}64$ = 49 152 |
| query side | $32 \times 256 \times 64$ = 524 288 | $32 \times 64$ = **2 048** |
| key side | $8 \times 256 \times 64$ = 131 072 | $8 \times 64 \times 64$ = 32 768 |
| per layer | 753 664 | **83 968** |
| whole model | **12.1 M** | **1.3 M** |

Against `MagneticBias`'s head — `proj[0]` (256×128) + `proj[2]` (128×32) = 36 864
per layer, 590 k over 16 layers (`LINEAR_BIAS.md` §2.5; `lambda_lin` and
`deep_set` are excluded since every arm keeps them unchanged) — that is **20× the
entire current bias head** versus **2.3×**.

Diagonal is chosen for attribution, not thrift. At 12.1 M the bias stops being
bias-sized and becomes comparable to the LoRA adapter, so arm 3 would vary *two*
things at once: a new structural channel **and** 20× the bias capacity. `014`
already established capacity is not the binding constraint here — a
92%-parameter-matched linear head recovered none of the 4.91 pp — so the extra
parameters are unlikely to be buying the result while being certain to muddy its
reading. Full $W_Q$ stays available as a follow-up axis if arm 3 lands well and
looks capacity-bound.

---

## 3. Why the per-column magnitude is excluded

Recorded so it is not re-derived, mirroring `LINEAR_BIAS.md` §3.

The natural first formulation is $Z = \mathrm{MLP}\big([\,|V_{i1}|^2 \cdots
|V_{iM}|^2\,]\big)$ — an MLP over the raw per-eigenvector magnitude vector. It is
excluded because it is not permutation-invariant in the presence of spectral
degeneracy, which is not a corner case.

Measured on a 5-node star (hub + four interchangeable leaves; $\lambda =
[0,1,1,1,2]$, a triply degenerate block), computing the eigenvectors of the same
graph under two node orderings through `get_magnetic_laplacian_coords`:

| quantity | max $|\Delta|$ under relabelling |
|---|---:|
| per-column $|V_{il}|^2$ | **0.674** |
| $\sum_l |V_{il}|^2 f(\lambda_l)$ (§2.3) | 8.3e-7 (fp noise) |
| $V V^*$ restricted to the block (what `MagneticBias` uses) | 5.9e-8 |

LAPACK returns a genuinely different basis inside the block, so the per-column
form makes prompt logits depend on node labelling — `tests/models/test_flex_cpu.py`
test 1 would fail, correctly. Worse than failing a test, it assigns *different
structural roles to automorphic nodes*: two interchangeable leaves have identical
magnitude rows before a block rotation and different ones after.

Star-shaped neighbourhoods with interchangeable leaves are the dominant WebQSP
subgraph shape, and `magnetic_lap.py` already carries solver workarounds for
near-degenerate spectra and isolated nodes — degeneracy is the common case, not
the pathological one.

---

## 4. Phase 1 — Implementation & Verification Gate

Two new config keys, `magnetic_magnitude` and `magnetic_hybrid`, following the
`BIAS_TYPES` extension protocol (`bias.py:8-12`). **Deliberately not optimized**:
same dense `(B,H,N,N)` bias, same eager/flex path, so any quality delta measured
in §5 is attributable to the math and not to an implementation. **No sbatch until
§4.2 is green.**

### 4.1 Implementation notes

* `MagneticMagnitudeBias(MagneticBias)` and `MagneticHybridBias(LinearMagneticBias)`.
  A shared `_self_energy(V_real, V_imag, phi) -> (B, N, d_mag)` helper on
  `MagneticBias` keeps the two in sync; the hybrid's `forward` is the linear arm's
  output plus the magnitude term.
* **Zero-init: zero $s^{(h)}$ only, and leave $W_K^{(g)}$ at its default init.**
  Zeroing both sides makes $\langle Z \odot s, Z W_K\rangle$ a dead saddle —
  $\partial b/\partial s \propto Z W_K = 0$ and $\partial b/\partial W_K \propto
  Z \odot s = 0$, so both gradients are exactly zero forever and the arm never
  leaves the origin. With only $s$ zeroed the bias is still exactly 0 at step 0,
  $s$ has a non-zero gradient immediately, and $W_K$ starts moving once $s$ does.
* **$H_{KV}$ and the group map.** `getattr(bias_config, 'num_key_value_heads',
  num_heads)`, and $g(h) = h \,//\, (H_Q/H_{KV})$ to match HF `repeat_kv`
  (`flex_attn/flex_core.py:74-76`). A mismatch here is invisible in the dense path
  — it just permutes which head gets which key map — and fatal in the factorized
  one. Test 12 is what pins it.
* Extend `structural_factors` to both new classes, returning the concatenated
  $(Q, K)$ of §2.4. It is not used by `forward`; it exists so §4.2 test 12 can pin
  the factorization in fp64 before any backbone work.
* **Wiring is the silent-failure surface, and it is now 3×.** `magnetic_linear`
  had to be threaded through `kgqa`, `context` and `graphqa` — each has its own
  `config.py` gate, `__main__.py` flag, `train.py` bias-param dict, plus
  `data_config_key`, `WIRED_FEATURES` and the collator. A gate that misses the new
  keys yields `magnetic=None` → `forward` returns `None` → a run that trains
  cleanly **with no bias at all** and reads as a clean negative result. Every gate
  that reads `magnetic or magnetic_linear` must also read the two new keys, and
  config validation must **raise**, not return `None`.
* No new save/load code: `active_params = ("graph_bias",)` is a substring match
  (`models/io.py`), so anything under the bias module is captured automatically —
  but test 13 pins it rather than assuming it.

### 4.2 Correctness gate — what must pass before any sbatch

Numbered to match `LINEAR_BIAS.md` §5.3, since most of that suite applies
verbatim. The first four are the gate; the rest run too and any failure is fixed
before submission.

1. **Permutation invariance** (`test_flex_cpu.py`) — relabelling nodes must leave
   prompt logits bit-identical. This is the test §3 exists to keep green, and it
   only bites if the fixture graph has a degenerate spectrum: **add a star and a
   cycle**, which the current fixtures' random graphs do not guarantee.
2. **Padded eigenvector slots** (`test_v2_ragged_magnetic_padding.py`) — the
   pool over $l$ is a fresh place for padded columns to enter. It is safe by
   construction ($|V|^2 = 0$ there kills any $\Phi$ value), but that is an
   assertion, not an argument.
9. **Zero-init inertness** — at step 0 both new modules are exactly 0 and logits
   match the no-bias model. Plus: **assert the gradient of $W_Q$ is non-zero at
   step 0**, which is what distinguishes a correct zero-init from the dead saddle.
10. **No silent no-op** — either new key with features absent must **raise**.
3. **Backward compatibility** (`test_modeling_gtlm_llama_v2.py`) — with both keys
   off, bit-identical to today.
12. **Factorization parity (fp64, CPU)** — $\langle Q_{tandem}, K_{tandem}\rangle$
    reproduces the dense hybrid bias on the **full** matrix under
    `bias_self_node=True` (and off-diagonal only when masked). Cheap, pure math,
    and it de-risks the whole backbone.
13. **Save/load round-trip** — `bias_parameters.pt` carries $\mathrm{MLP}_{magnitude}$,
    $W_Q$ and $W_K$.
7. **Bias regularization** (`test_bias_regularization.py`) — the trainer's rule is
    `p.ndim >= 2` (`text_graph_trainer_v2.py:82`); confirm the new 2-D weights are
    classified as intended.
15. **Degenerate-spectrum invariance (new)** — the §3 measurement as a unit test:
    a star graph, two node orderings, assert $S_i$ matches to fp tolerance. It
    pins the property §2.3's whole form exists to satisfy, and it fails loudly if
    someone later "simplifies" the pool away.

---

## 5. Phase 2 — The Training Comparison

Runs only after §4.2 is green. This is the phase that costs GPU-days.

### 5.1 Arms and the diagonal-mask policy

| Arm | Config | Factorization | Diagonal | Isolates |
|---|---|---|---|---|
| **0** | no soft bias | — | — | headroom denominator (the floor) |
| **1** | `magnetic` | $O(N^2)$ dense | **masked** | incumbent target (the ceiling) |
| **2** | `magnetic_linear` | phase only | **unmasked** | cost of losing non-linear structural routing |
| **3** | `magnetic_magnitude` | magnitude only | **unmasked** | cost of losing directed distance/flow |
| **4** | `magnetic_hybrid` | phase + magnitude | **unmasked** | **the proposed $O(N)$ replacement** |

**Each arm at the mask setting it could actually ship with.** Arms 2–4 run
`--bias-self-node`, because an inner product yields $q_i \cdot k_i$ and cannot be
forced to zero — unmasked is the only configuration a factorized backbone can
run. Arm 1 keeps the mask, because it is the legacy method and can implement one.
`linear_bias` Phase 3 established this matters and that its sign is
head-dependent, so it is not a free choice: on WebQSP the diagonal is worth
+2.47 pp to the linear head and −1.74 pp to the MLP head.

**Consequence for reuse.** §5.3's table is what is actually reusable; the earlier
claim that "only two arms have to be re-run" holds for WebQSP and GraphQA but not
for 4k context, where `016` ran arm C only — no B, no D.

### 5.2 Hyperparameters

* **$M = 64$ for every arm.** 128 is not a shippable operating point: it puts the
  tandem at $2M + d_{\text{magnitude}} = 320$ appended dims, 5× Llama-1B's
  `head_dim`, against 192 (3×) at 64 — and `linear_bias` Conclusion 4 already
  found 64 free for the linear head on WebQSP. Measuring at a width the optimized
  version would never run is the same error Phase 3 caught with the diagonal mask.

  What that costs in reusable controls, per dataset:
  * **GraphQA — nothing.** `013`/`017` build the cache at `magnetic_m: 0` on
    ~20-node graphs, so $\min(N, 64) = \min(N, 128) = N$. The controls are
    unaffected and stay reusable.
  * **WebQSP — arms 1 and 2 must be re-run.** `015` ran $M{=}128$ only; `010` has
    the linear head at 64 but **masked**, and `magnetic` only at 128 and 16.
    Neither control exists at the configuration this sweep needs.
  * **context — free**, since all five arms already run in-sweep (§5.4). Node
    counts reach 128 there, so truncation now binds on the larger graphs; nothing
    is lost, as that dataset's $M$-curve was already void.

  Arm 4 at $M{=}128$ on WebQSP is an optional 3-run follow-up if the hybrid looks
  starved of eigenvectors — deliberately not in the headline grid, since 128 is
  the width this exercise exists to avoid.
* $d_{\text{magnitude}} = 64$, $d_{\text{magnitude-repr}} = 256$, diagonal query
  side (§2.5, §2.6). Widening $d_{\text{magnitude}}$ or promoting $s^{(h)}$ to a
  full $W_Q^{(h)}$ are cheap follow-up axes if arm 3 lands well; neither is in the
  headline grid.
* **`bias_lr` is swept per arm**, over $\{5\text{e-}3, 2\text{e-}2\}$, and each arm
  is read at its own best. `linear_bias` Conclusion 5: the linear head wanted ~4×
  the MLP head's LR, and a shared LR prices optimization rather than math. Arms 3
  and 4 have parameter counts unlike either incumbent, so neither existing LR can
  be assumed.
* **Budgets: WebQSP 15 epochs, GraphQA 20, context 6.** The first two are forced —
  they must match the `015` / `017` controls being reused. Context is chosen for
  convergence (§5.4), not inherited.
* Everything else exactly as `linear_bias`: SPD and RRWP off in every arm, 3 seeds,
  median-seed rule, recipes replayed from the sweep's own `jobs/*.sh` rather than
  retyped.

### 5.3 Reuse ledger

All at $M{=}64$ (§5.2), which is what decides each cell.

| dataset | arm 0 | arm 1 (masked) | arm 2 (unmasked) | arms 3, 4 |
|---|---|---|---|---|
| WebQSP | `010` D — bias-free, so $M$-independent | **new** — `015` is $M{=}128$ | **new** — `010`'s C@64 is masked | **new** |
| GraphQA (3 tasks) | `013` D | `017` B masked | `017` C unmasked | **new** |
| context 4k | **new** | **new** | **new** | **new** — all five arms in one sweep, §5.4 |

Only GraphQA reuses its controls, because $M{=}64$ is a no-op on ~20-node graphs.
WebQSP's exist but at the wrong $M$ (and, for the linear head at 64, the wrong
mask). Context has none at a matched budget regardless (§5.4).

### 5.4 4k context — all five arms, in one sweep

**Decided: every arm runs on every benchmark, at the existing recipes.** No task
is altered and no arm is dropped on prediction. An arm that fails where it was
predicted to fail is a stronger result than an arm that was never run, and the
predictions below are worth *testing* rather than acting on.

The one change against §5.3's ledger: context's controls are **run inside this
sweep** rather than borrowed. `011`'s B ran 2 epochs at bias_lr 5e-3 masked and
`016`'s C ran 4 epochs unmasked, so reusing either would be a cross-sweep
comparison at a different budget. This is not a change to the task — it is
including arms 0–2 instead of quoting them — and it closes the exact hole
`linear_bias/README.md` §7 names as the blocker on the factorized backbone.

**Budget: 6 epochs, the same for every arm.** The principle is that each arm is
read at *its own* ceiling, so the budget is set by the slowest arm and then shared.
Evidence: `011` ran 2 epochs at 3.6 h/run (~1.8 h/epoch); `012`'s header computes
the masked linear arm's transition at **~epoch 4.75** at bias_lr 5e-3; and at
2e-2 with 4 epochs `012`'s C landed at 0.590 with seeds spanning 0.435–0.775 — a
transition caught mid-flight, i.e. 4 epochs is demonstrably *not* convergence
there. `016` shows the unmasked arm saturating by 4, and `README` §7 asks for ≥6.
Six clears the latest projected transition by ~1.25 epochs.

An earlier revision of this section argued for **8**, on the grounds that it gives
every arm ≥3 epochs of margin past that transition. Six was chosen instead, and
the honest statement of what that buys is the one above: it is the *tightest*
number the evidence supports rather than a comfortable one. An arm that has not
moved by 6 has had a full epoch past `012`'s measured transition in which to do
so, which is enough to call it a ceiling — but the margin is one epoch, not three,
and a null result on arm 3 here should be read with that in mind.

This **reverses `012`'s recorded decision** that 8 epochs was "padding". That was
right for `012`'s objective — separating "cannot" from "under-stepped" only needs
the transition to *start*. It does not carry to an experiment whose question is
where each arm plateaus, and arms 3 and 4 are new heads with unknown transition
points. Arm 3 is the exposure: it is the arm predicted weak here, and a weak arm 3
at 4 epochs would reproduce exactly the ambiguity that cost `011`→`012` a whole
extra sweep. The context metric is all-or-nothing on a 3-token code, so an
under-budgeted arm reads as a clean floor rather than as an unfinished run —
which is why this task, and not WebQSP or GraphQA, is the one that needs the
budget argued rather than inherited.

Two things to keep in view when reading the result:

* **Saturation makes context a ceiling test, not a ranking.** Unmasked arm 2 has
  already reached **1.000** and masked arm 1 0.995. If arms 1–4 all land at
  ceiling, the finding is "every factorizable arm reaches the ceiling in the
  long-context regime" — which is precisely what the §7 verdict needs — and it
  must **not** be read as "the arms are equivalent" in general. If a ranking is
  wanted later, `hops` and `fan_out` are existing difficulty knobs that raise the
  bar without touching what the task tests; that is a follow-up, not a change to
  make now.
* **Arm 3 is predicted to be weak here, by construction.** `data.py` attaches the
  QUESTION node to *every* content node so `SPD(QUESTION, ·) == 1` uniformly, and
  adds `fan_out - 1` decoy edges that "enter the DiGraph identically" so
  ~$\text{fan\_out}^{hops}$ nodes sit at the same distance; both comments state
  the intent outright — the topology must not encode which reference is real. Arm
  3 consumes a **per-node** feature, and the discriminating information lives in
  relative position along the chain, which is the phase channel. Contrast GraphQA
  `node_degree`, where $\mathrm{diag} f(L)$ is essentially a degree / centrality
  measure and arm 3 should be strong. This is a falsifiable prediction read off
  the generator, and the sweep tests it — if arm 3 works here anyway, that is
  informative about what the magnitude channel actually encodes.

### 5.5 Cost

Per-run wall clock: WebQSP ~2.8 h (`linear_bias` §6.1); context **~1.8 h/epoch**,
derived from `011`'s measured 3.6 h at 2 epochs, so ~10.8 h at the §5.4 budget of
6 — `time: 16:00:00`, which leaves headroom for a slow node without asking for
the 24 h that makes a job harder to schedule. GraphQA ~17 min (`013`).

| block | runs | config | GPU | notes |
|---|---:|---|---|---|
| WebQSP arms 1–4 | 24 | `018` | H100 | 4 arms × 3 seeds × 2 LR, ~67 GPU-h. Arms 1,2 re-run at $M{=}64$ (§5.2); arm 0 reused |
| GraphQA arms 3,4 | 36 | `019` | A100 | 2 arms × 3 tasks × 3 seeds × 2 LR; short runs; controls reused |
| context arms 0–4 | 27 | `020` | B200/B300 | 4 arms × 3 seeds × 2 LR, **plus arm 0 at 3** — no soft bias means no bias parameters, so `bias_lr` is inert (`012` ran D at one LR for this reason). ~292 GPU-h |
| | **87** | | | `max_concurrent: 16` on each |

**On the GPU assignments.** They are a scheduling choice, not a measurement one:
accuracy is the only thing read here and it is platform-independent. Two
consequences to keep in mind when reading `sacct`. WebQSP's step times are **not**
comparable to `015`'s, which ran on B200/B300; GraphQA's **are** comparable to
`013`'s, which also ran on A100 (but not to `017`'s, which moved to B200/B300).
And `"gpus": ["A100"]` must stay in **list** form: it renders as
`--constraint GPU_BRD:A100`, which covers the 40 GB nodes (`aga`, `axa`) *and* the
80 GB one (`ana`); the string form would pin the gres type and silently exclude
`ana`, whose gres is registered as `A100_80GB`. There is no `GPU_BRD:A100_80GB`
feature to name.

**Why context dominates.** A context epoch is ~10× a WebQSP epoch (1.8 h against
0.19 h): every sample is a 4096-token sequence at batch 1 × accum 8 — 2000
optimizer steps per epoch over 16 000 samples — and `expand_node_to_token_bias`
lifts the node bias to a `(B,H,T,T)` object at T=4096, while WebQSP's median graph
is 52 nodes on far shorter sequences. Three factors then multiply: 10× per epoch,
3× the epochs of `011`, and no reusable controls so five arms run instead of two.
The 6-epoch budget is itself one of the two levers that were available for
bringing this down, and it is the one taken (8 epochs would have been ~390
GPU-h). The other — pinning arms 1–2 at their known-best LR, saving 6 runs — was
**not** taken, because a budget change is exactly when an inherited best-LR stops
being safe.

An earlier revision of this table put context at ~105 GPU-h; that assumed
3.5 h/run, which is the **2-epoch** cost, not a converged one.

Capture sbatch job ids at submission — an unchecked id is how a sweep silently
fails to exist.

### 5.6 Success criteria

* **Arm 4 within seed noise of arm 1** clears the path to build the $O(N)$ flex
  backend and delete the `score_mod`. On WebQSP that means closing whatever
  residual arm 2 leaves against arm 1 — 1.42 pp at $M{=}128$ (`015`), but this
  sweep runs at 64 and re-measures both, so the target is established in-sweep
  rather than quoted.
* **Arm 3 alone is the diagnostic**, and it is worth reporting even if arm 4 wins:
  if magnitude alone recovers most of the headroom, directed flow is not what the
  magnetic bias is selling and a much cheaper bias exists. If arm 3 is near the
  floor, the phase channel is load-bearing and the tandem's gain is genuinely
  additive.
* **Arm 4 ≈ arm 2** would say the magnitude channel is inert, and the remaining
  gap to arm 1 is pairwise non-linearity that no factorization reaches. That
  closes `LINEAR_BIAS.md` §7 negatively, which is also a result.

---

## 5.7 Scale instability of the magnitude channel (found during Phase 2)

**The channel as specified in §2.3 is not scale-stable, and this is a property of
the form, not of the implementation.** Three runs diverged to `NaN` at
`bias_lr` 2e-2 — arm 3 on WebQSP at epochs 2.48 and 3.17 (2 of 2 seeds), arm 4 on
context at 0.27 — while `magnetic` and `magnetic_linear` never diverged at either
LR on either dataset. See `experiments/mixed_bias/README.md` for the full record.

*Why it is the form.* In $b = \langle Z_i \odot s^{(h)}, Z_j W_K^{(g)}\rangle$ the
same $Z$ feeds **both** sides, so $\mathrm{MLP}_{magnitude}$, `deep_set` and
`lambda_lin` all enter **squared**. Every other head here is multilinear — each
tensor appears once. Standard attention avoids this with $W_Q \neq W_K$; arm A
avoids it because $K_{phase} = [V_R \| V_I]$ is parameter-free with unit row norm
(measured: $\sum_l |V_{il}|^2 = 1.0000$ across 7 856 node-rows). The magnitude
channel is anchored on neither side.

Measured on real eigenvectors at the operating point, scaling the shared trunk by
$k$:

| | $k{=}1{\to}2$ | $k{=}2{\to}4$ | $k{=}3{\to}6$ | degree |
|---|---|---|---|---|
| phase | 5.8× | 4.5× | 4.1× | $k^2$ |
| magnitude | 16.3× | 27.6× | 23.0× | $k^4$ |

— exactly double the exponent. And the absolute scale explains the *suddenness*:
at $k{=}1$ the magnitude bias is 0.021, **15× smaller** than phase's 0.309; it
crosses phase near $k{\approx}4$ and reaches 52.85 at $k{=}6$, which saturates
softmax outright. A quartic term under an exponential is invisible for thousands
of steps and then dominates within tens.

*Why clipping does not prevent it.* `max_grad_norm` is 1.0, but the optimizer is
AdamW, whose update $lr \cdot \hat m/(\sqrt{\hat v}+\epsilon)$ is approximately
scale-invariant — clipping bounds the step, not the trajectory. Each bias
parameter drifts up to `bias_lr` per step with `bias_weight_decay = 0`, so
cumulative growth is unopposed.

*Two things this is NOT.* Not the unmasked diagonal: $|b_{ii}|$ and the
off-diagonal maximum track each other at ratio 1.00 at every scale, so
`bias_self_node` is exonerated. Not the input features: the row-energy identity
above bounds $S$ by $\max_l |\Phi_l|$.

**Fix, for after this sweep: normalize $Z$ before the inner product.** Measured on
the same fixture, scaling the trunk by $k$:

| $k$ | shared $Z$ (current) | split $Z_Q/Z_K$ | split + L2-norm |
|---:|---|---|---|
| 1 | 0.0227 | 0.0162 | 0.0756 |
| 2 | 0.363 (×16.0) | 0.443 (×27.4) | 0.0968 (×1.3) |
| 4 | 7.40 (×20.3) | 12.98 (×29.3) | 0.1038 (×1.1) |
| 8 | 136.1 (×18.4) | 217.0 (×16.7) | 0.1038 (×1.0) |

**Splitting the MLP output into $Z_Q \| Z_K$ does NOT fix it** — tested, still
quartic. The split decouples only the MLP's *last* layer; `deep_set`,
`lambda_lin` and the MLP's first layer still feed both halves and remain squared.
Two separate MLPs fail for the same reason ($S$ feeds both). The split is still
worth doing for **capacity** — the two sides currently see the same vector and
differ only by a per-head diagonal and a per-group matrix — and it is nearly free
(+262 k params over 16 layers, **no change to appended head width**, so §2.5's
budget table is unaffected). It is simply not a stability fix.

With $\hat Z$ unit-norm, $|b| \le \|s\|\,\|W_K\|$: the bias magnitude is set by
the head parameters at degree 1 each, exactly as in arm A. It plateaus *low*, so a
learnable gain is needed to restore range — which is what standard QK-norm does.

*This is not a novel remedy.* Uncontrolled attention-logit growth ending in
softmax saturation is a documented instability, and **QK-LayerNorm** is its
standard fix (Wortsman et al. 2023, *Small-scale proxies for large-scale
Transformer training instabilities*; shipped in ViT-22B, Dehghani et al. 2023;
earlier as Henry et al. 2020, *Query-Key Normalization for Transformers*; cf. Zhai
et al. 2023, σReparam). Our bias is an additive term rather than the QK product
itself, but it enters the same softmax and grows the same way. The degree argument
above was derived independently and lands on the same remedy.

Non-zero `bias_weight_decay` only delays the failure, and a lower `bias_lr` is a
workaround — so arm 3/4 results at 5e-3 must be read as "the largest LR this
parameterization survives", not as a free choice.

## 5.8 The remedy as built

§2.3 carries the resulting form. Three design points are worth recording because
each had a plausible alternative that does not work here.

**Normalize the factors, not $Z$.** The §5.7 fixture varies only the trunk scale
$k$, so its "split + L2-norm" column cannot discriminate between *the trunk grows*
and *$s$ or $W_K$ grows* — normalizing $Z$ is invariant to the former by
construction and does nothing about the latter, which remain learned, unbounded and
weight-decay-free. Normalizing $Q_{magnitude}$ and $K_{magnitude}$ after $s$ and
$W_K$ are applied is invariant to all three at once. Since the diagnostic that
would have identified the culprit never ran (job 125240 was cancelled at 0:00
elapsed by the same mass-cancel that ended the sweep), the fix that does not
require knowing the answer is the one to build.

**Everything must happen before the dot product.** The deferred backbone
concatenates these factors onto the content $Q/K$ and takes one fused
flex/flash dot product, so nothing may be applied to $b$ afterwards. Row-wise
normalization survives that ($\hat q_i$ depends only on $i$, $\hat k_j$ only on
$j$, so the bilinear form is preserved) and so does a per-head scalar gain, which
folds into the query block: $\lVert q'_i \rVert = |g^{(h)}|$ and
$\langle q'_i, \hat k_j \rangle = g^{(h)} \langle \hat q_i, \hat k_j \rangle$. A
$\tanh$ or a clamp on $b$ would bound it just as well and is **not available** —
non-bilinear in $(i,j)$, same reason arms 3/4 run unmasked (§2.4). Normalization
plus a gain is the only bounding mechanism that survives factorization at all.

**The gain, not $s$, holds the zero-init.** Covered in §2.3; the test is
`test_gate_gradient_is_not_the_eps_bomb`. One consequence for reading the next
sweep: $s$ and $W_K$ now have exactly zero gradient at step 0 and start moving one
step later, when $g$ leaves the origin. That is a one-step delay, not the dead
saddle §4.2 gates against, and it is asserted directly.

**What this does not bound.** $g^{(h)}$ is itself learned and unopposed
(`bias_weight_decay = 0`), so a sufficiently determined run can still grow it —
linearly, at $\approx$ `bias_lr` per step under AdamW, rather than quartically.
Normalization converts a runaway into a drift; it does not prove boundedness. The
difference that matters operationally is that the drift is now visible in one
scalar per head, so "survived because it is fixed" and "survived because it is
postponed" are distinguishable without re-deriving anything. Log $g$ and read it.

---

## 6. What this plan does not test

* Whether the non-linearity `MagneticBias` uses at L5 is *the same* non-linearity
  arm 3 adds. §1 argues no per-node channel can be, and §5.6 reads the arms
  accordingly, but nothing here measures the mechanism.
  **Measured since, 2026-08-15 (`NON_LINEAR_BIAS.md`):** §1's argument holds for
  the *diagonal* feature every arm in this plan spends, and fails as a general
  claim about per-node channels. A pooled row/column of the kernel reaches the
  dense ceiling on GraphQA `shortest_path` (101.0% of headroom). It is still null
  on WebQSP, but for a reason orthogonal to §1 — pooling marginalizes the partner
  index, so it cannot resolve *which* node, at any width.
* Whether `magnetic` itself gains from a wider $d_{mag}$ — still open from
  `linear_bias` §3 (`014` widened only the linear head).
* The speed/memory of the factorized backbone. Every arm here is dense; §2.5's
  head-width table is arithmetic, not a measurement.
