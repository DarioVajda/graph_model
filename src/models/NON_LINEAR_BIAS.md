# Non-Linear Magnetic Bias via Attention-Pooled Node Features

## Status

| | |
|---|---|
| **Objective** | Recover the pairwise non-linearity `magnetic_linear` gives up, while keeping the bias expressible as a dot product between appended Q/K channels — i.e. **flash-attention-compatible**, with no N² tensor inside the attention kernel. |
| **Scope** | Evaluation only. The dense simulation measures whether the bias type works; the fused kernel is out of scope (§9). |
| **Status** | **COMPLETE, 2026-08-15 — negative on WebQSP.** 72 training runs + `036`, a 9-run eval-only pass that measured the mechanism; results and verdict in `src/experiments/nonlinear_bias/README.md`. |

**Headline.** The arm reaches the **dense ceiling on GraphQA's path task**
(0.951±0.008, 101.0% of headroom, beating `magnetic_linear`'s 95.6% and
`magnetic_hybrid`'s 96.9% — the first factorizable arm to do so) and reaches only
**22.7% of headroom on WebQSP** against `magnetic_linear`'s 87.8%, a −17.1 pp
deficit and just +6.0 pp above no bias at all.

§9's condition for building the kernel — closing at least half of WebQSP's
residual — is not met by any margin, so **the kernel is not scheduled**.

The negative is strong rather than weak: the pool audit shows the mechanism
engaged on every run (`‖W_attn‖_F` 6.8–20 against 3.27 at init), the LR grid
brackets an interior optimum (16.1% → 22.7% → 17.6% over a 20× span), and the
learned pool loses to mean-pooling on WebQSP (−2.10 pp, 0/3 seeds at 2e-2).

**What this revises.** `MIXED_BIAS.md` §1 holds that the residual is *pairwise*
non-linearity and that no per-node feature reaches it. This arm is a per-node
feature that reaches the dense ceiling on a path task, so that claim is false as
stated. The defensible version is narrower: a per-node feature built by pooling the
pairwise kernel is a good **structural summary** and a poor **identity resolver**.
Where the answer is a structural quantity it reaches the dense ceiling; where the
answer is a specific node it loses a flat ~25% relative to the exact pair
representation.

**A scaling explanation was proposed and then measured false.** This document and
the experiment README first attributed the WebQSP loss to *capacity per node* —
64 pooled dimensions being enough for ~20 partners and not for 512. `036`
stratified the per-example deficit by graph size and refuted it: the deficit is
flat in `N` (Pearson +0.09, the wrong sign), and is already 17.7 pp on graphs
under 32 nodes, where the pool has twice the width it needs. The premise was also
wrong on its own terms — "512" was the `max_nodes` cap, while the built split's
median is 61.5 nodes. **The obstacle is the form of the representation, not its
width**, so no budget fixes it. §3 below pre-registered exactly this and was
right; the capacity gloss was added afterwards and was not.

`mixed_bias` closed negatively: all three placements of the per-node magnitude
feature — alone, additive, multiplicative — left WebQSP's 12% residual untouched,
and its conclusion was that the residual is *pairwise* non-linearity, which no
per-node feature reaches.

This arm is also a per-node feature, and by the letter of that conclusion it
should also be null. It is worth running anyway because arms 3, 4 and v2 all spent
the **same** feature: $S_i = \mathrm{Re}\,K(i,i)$, the spectral self-energy — a
strictly local scalar, which `mixed_bias/README.md` diagnoses as "a *local*
feature detector; it fails exactly where the answer is a path." Here the node
feature is a learned, non-linear, **asymmetric** pool of the node's entire row and
column of the pairwise kernel. A path-reachability signature can live there; it
provably cannot live in $K(i,i)$.

So the prior arms tested *is the diagonal enough?* This tests *is any node-level
feature enough?* — and a null here closes the line properly rather than by
extrapolation.

Secondary prize: appended head width becomes $d_{struct}$, chosen freely, instead
of `magnetic_linear`'s $2M$. At the operating point that is **64 vs 128** — 1×
`head_dim` instead of 2×.

---

## 1. Notation

| symbol | meaning |
|---|---|
| $B, N$ | batch; nodes, **padded to the batch maximum** (`GraphCollatorV2`) |
| $n_b$ | true node count of graph $b$ |
| $M$ | eigenvectors kept (`magnetic_m`); operating point **64** |
| $d_{mag}$ | `magnetic_dim`, width of the pair features — **64**, see §5.1 |
| $d_{struct}$ | width appended to each attention head — **64** |
| $H_Q, H_{KV}$ | query / KV heads; $g(h) = h \,//\,(H_Q/H_{KV})$ is HF's `repeat_kv` map |
| $V \in \mathbb{R}^{B\times N\times M\times 2}$ | eigenvectors (real/imag), as `magnetic` carries them |
| $\Phi \in \mathbb{R}^{B\times M\times d_{mag}}$ | eigenvalues after the DeepSets (`_phi`, unchanged) |
| $V_{attn}$ | the backbone's attention **value** tensor — never the eigenvectors |

$h$ is the folded pre-activation and $i,j$ index nodes; node indices are $p,q$
when a third is needed.

---

## 2. Formulation

### 2.1 Shared pair features

Computed **once per forward**, before the decoder layers:

$$h(i,j) \;=\; \big[\text{folded spectral pre-activation}\big] \;\in\; \mathbb{R}^{B\times N\times N\times d_{mag}},
\qquad E(i,j) \;=\; \mathrm{SiLU}\big(h(i,j)\big)$$

$h$ is exactly `MagneticBias._folded_spectral` — the Hermitian outer product
$K(i,j) = \sum_l V_{il}\overline{V_{jl}}\Phi_l$ pushed through `proj[0]` — and the
`SiLU` is `proj[1]`. $E$ is therefore the incumbent MLP head's hidden activation,
unmodified. What changes is only what happens to it next: `MagneticBias` collapses
it to a per-head scalar with `proj[2]`; here it is pooled into node features.

### 2.2 The validity mask

$$\Omega(i,j) \;=\;
\begin{cases}
0 & \text{if } (i < n_b \;\wedge\; j < n_b) \;\vee\; i = j\\[2pt]
-\infty & \text{otherwise}
\end{cases}$$

The $i = j$ clause is what keeps padded slots finite. A padded row would otherwise
be all $-\infty$ and its softmax $0/0$; leaving its diagonal open makes it attend
entirely to itself, with weight exactly $1$. Since $V$ is zero at a padded slot
$p$, all four einsums vanish and $E(p,p) = \mathrm{SiLU}(b_1)$, a finite non-zero
constant. The clause is inert for real nodes, whose diagonal is unmasked already.
Padded features are never gathered (`expand_node_to_token_bias` indexes by
`node_ids`, which names only real nodes), so they carry no gradient.

### 2.3 Layer-specific asymmetric pooling

Per layer: $W^{out}_{attn} \in \mathbb{R}^{d_{mag}\times H_Q}$,
$W^{out}_{val} \in \mathbb{R}^{d_{mag}\times H_Q\times d_{struct}}$, and
$W^{in}_{attn} \in \mathbb{R}^{d_{mag}\times H_{KV}}$,
$W^{in}_{val} \in \mathbb{R}^{d_{mag}\times H_{KV}\times d_{struct}}$.

**Outgoing (row $i$, softmax over $j$), per query head $h$:**

$$a^{out}_{ijh} = \frac{E(i,j)\,W^{out}_{attn,h}}{\sqrt{d_{mag}}} + \Omega(i,j),
\qquad
w^{out}_{ijh} = \frac{\exp a^{out}_{ijh}}{\sum_{j'} \exp a^{out}_{ij'h}}$$

$$\bar E^{out}_{ih} \;=\; \sum_{j} w^{out}_{ijh}\,E(i,j) \;\in\; \mathbb{R}^{d_{mag}},
\qquad
\tilde z^{out}_{ih} \;=\; \bar E^{out}_{ih}\,W^{out}_{val,h} \;\in\; \mathbb{R}^{d_{struct}}$$

**Incoming (column $j$, softmax over $i$), per KV head $g$:**

$$a^{in}_{ijg} = \frac{E(i,j)\,W^{in}_{attn,g}}{\sqrt{d_{mag}}} + \Omega(i,j),
\qquad
w^{in}_{ijg} = \frac{\exp a^{in}_{ijg}}{\sum_{i'} \exp a^{in}_{i'jg}}$$

$$\bar E^{in}_{jg} \;=\; \sum_{i} w^{in}_{ijg}\,E(i,j) \;\in\; \mathbb{R}^{d_{mag}},
\qquad
\tilde z^{in}_{jg} \;=\; \bar E^{in}_{jg}\,W^{in}_{val,g} \;\in\; \mathbb{R}^{d_{struct}}$$

**Pool before projecting.** $W_{val}$ does not depend on the pooled index, so it
commutes with the sum:

$$\sum_j w^{out}_{ijh}\big(E(i,j)\,W^{out}_{val,h}\big) \;=\; \Big(\sum_j w^{out}_{ijh}\,E(i,j)\Big) W^{out}_{val,h}$$

This is an identity, not an approximation, and it is why the per-pair value vectors
never exist. Ordinary attention materializes $V_{attn}$ because it is a *per-token*
tensor $(N, d)$ reused by every query; here the value is a *per-pair* tensor
$(N^2, d)$ and a linear image of something already held, so it is pooled in place.
At $N{=}512,\,H_Q{=}32$ that is $2{\cdot}10^6$ elements instead of $5{\cdot}10^8$,
and $N^2 H d_{mag}$ FLOPs instead of $N^2 H d_{mag} d_{struct}$.

### 2.4 Normalization

$$z^{out}_{ih} = \mathrm{RMSNorm}\big(\tilde z^{out}_{ih}\big) \odot \gamma^{out}_h,
\qquad
z^{in}_{jg} = \mathrm{RMSNorm}\big(\tilde z^{in}_{jg}\big) \odot \gamma^{in}_g$$

with per-channel gains $\gamma^{out} \in \mathbb{R}^{H_Q \times d_{struct}}$ and
$\gamma^{in} \in \mathbb{R}^{H_{KV}\times d_{struct}}$.

**Initialize $\gamma^{out} = 1$ and $\gamma^{in} = 0$** — one side only. The bias is
then exactly $0$ at step 0, while $\gamma^{in}$ has a non-zero gradient
immediately and everything upstream unfreezes as soon as it moves. Zeroing *both*
gives $\partial b/\partial\gamma^{out} \propto \gamma^{in} \odot z^{in} = 0$ and
symmetrically: the dead saddle of `MIXED_BIAS.md` §4.1, which never leaves the
origin. Zeroing $W_{val}$ instead would put the zero *inside* the norm, whose
Jacobian at zero is $I/\varepsilon$ — `MIXED_BIAS.md` §2.3 pays for that mistake
already.

RMSNorm is also what bounds the degree. It removes uniform rescaling of the trunk
entirely on each side, so $b$ is degree-0 in the trunk and degree-1 in each gain —
the property that stopped arm 3's four `NaN` divergences. It does **not** give
that arm's single-scalar range: $\lVert \hat z\rVert_2 = \sqrt{d_{struct}}$, so
$|b| \le \max|\gamma^{out}|\cdot\max|\gamma^{in}|\cdot d_{struct}$. That product is
logged (§5.3) so a divergence stays diagnosable.

### 2.5 Token expansion and the attention form

Gather node features to token length $T$ via `node_ids`:

$$q^{mag}_{th} = z^{out}_{\mathrm{id}(t),\,h}, \qquad k^{mag}_{tg} = z^{in}_{\mathrm{id}(t),\,g}$$

$$Q'_{th} = \big[\,Q_{th} \,\big\|\, q^{mag}_{th}\,\big], \qquad
K'_{tg} = \big[\,K_{tg} \,\big\|\, k^{mag}_{tg}\,\big] \;\in\; \mathbb{R}^{d_{head}+d_{struct}}$$

so the structural term the attention sees is

$$b^{(h)}(i,j) \;=\; \big\langle z^{out}_{ih},\; z^{in}_{j,\,g(h)}\big\rangle$$

**The dense simulation is the definition.** It adds $b$ to the post-scale logits,
matching where every existing bias enters (`dispatch.py:98`). A future kernel
receiving $Q'$/$K'$ multiplies the appended block by the softmax scale instead, so
it must pass an explicit `softmax_scale` and pre-multiply the structural features
by $1/\sqrt{\text{scale}}$ on **each** side — splitting the factor keeps both
halves in bf16 range. `LINEAR_BIAS.md` §7.2 is the standing warning; nothing in
this phase depends on it.

---

## 3. Properties

**Basis invariance holds.** $h$ is a linear map of the invariant Hermitian
contraction, `SiLU` is pointwise on it, and softmax pooling is
permutation-equivariant — so the whole construction is invariant to the per-vector
$U(1)$ phase and to $U(k)$ mixing inside a degenerate block. The non-linearity sits
*after* the pool over $l$, which is the condition `MIXED_BIAS.md` §2.3 derives. The
pre-existing risk of an $M<N$ prefix slice splitting a degenerate block is
inherited, not added. §7 tests 1 and 15 verify this empirically; the theory is not
taken on trust, because a transposed pooling axis would break it silently.

**Directionality survives.** $K$ is Hermitian, so its real part is symmetric and
its imaginary part antisymmetric: $h(i,j) \ne h(j,i)$ genuinely. Pooling that
asymmetric feature along the two different axes is what makes $z^{out}$ encode a
node's outgoing structure and $z^{in}$ its incoming structure. This is why the arm
needs no phase channel bolted on.

**What it still cannot express**, stated so a null is readable. The pool
marginalizes over the partner index, so $z^{out}_i$ cannot retain *which
particular* node a relation was with. `magnetic_linear` passes eigenvector
coordinates through undamaged and reconstructs the kernel exactly; this arm
compresses each row to $d_{struct}$ dimensions through a softmax. Directionality is
a property of the aggregate and survives; pair-specific resolution is what is at
risk. Arm 2 as comparator and the uniform-weight ablation (§8) are what measure
that.

> **Confirmed, 2026-08-15.** This is the paragraph the whole experiment turned on,
> and `036` vindicated it in its pre-registered form: the loss is exactly the loss
> of pair-specific resolution, and it is *uniform* — flat in graph size, flat in
> answer-set size, present at every scale. Note what it does **not** say: nothing
> here predicts the loss grows with $N$. The later capacity reading added that and
> was wrong. The distinction matters because it is the difference between a bias
> type that is mis-sized (fixable) and one that is mis-shaped (not).

**The intra-node diagonal is unmasked.** An inner product yields $\langle z^{out}_i,
z^{in}_i\rangle$ and cannot be forced to zero, so every run here sets
`bias_self_node=True`. `linear_bias` Phase 3 established this is not a free choice
and that its sign is dataset-dependent — which is why arm 2, also unmasked, is the
primary comparator and dense arm 1 is context only.

---

## 4. Budget

At $d_{mag}{=}64$, $d_{struct}{=}64$, $H_Q{=}32$, $H_{KV}{=}8$, 16 layers:

| | params |
|---|---:|
| shared trunk (`lambda_lin`, `deep_set`, `pair_proj`) — **once** | ~16.6 k |
| $W^{out}_{attn}$ / $W^{out}_{val}$ per layer | 2 k / 131 k |
| $W^{in}_{attn}$ / $W^{in}_{val}$ per layer | 0.5 k / 33 k |
| $\gamma^{out}, \gamma^{in}$ per layer | 2.6 k |
| **per layer** | **169 k** |
| **whole model** | **~2.7 M** |

Against `MagneticBias`'s head at 590 k over 16 layers, and the magnitude arm's
1.3 M. It is ~4.6× the incumbent bias head and well under the 12.1 M design
`MIXED_BIAS.md` §2.6 rejected as no-longer-bias-sized. This is the reason §8 moves
the LR grid down.

Appended head width is $d_{struct} = 64$ = 1× `head_dim`, against
`magnetic_linear`'s $2M = 128$ = 2× at the same operating point.

---

## 5. Implementation — the dense simulation

Deliberately unoptimized, following the `BIAS_TYPES` protocol (`bias.py:8-12`) so
any quality delta is attributable to the maths and not to an implementation.

### 5.1 The trunk is shared; the pooling heads are not

$d_{mag}{=}64$ — twice the code default — is affordable **because the trunk is
amortized**: $E$ is computed once per forward and every layer pools it. Giving
each layer its own trunk instead is a different model: 16 independent $E$ tensors
and 16× the N² einsums under gradient-checkpoint recompute.

Neither existing mechanism does this. `shared = True`
(`MagneticSharedBias`) shares the **final** $(B,H,N,N)$ bias across layers, and
`magnetic_groups` builds $G$ **independent full copies**. What is needed is a
shared *trunk* with per-layer heads, which mirrors the `shared_node_bias` path
exactly:

* a top-level module computing $E$ once, outside the checkpointed decoder layers;
* a new `GraphContext` field (`shared_pair_features`, `(B,N,N,d_mag)`) alongside
  `shared_node_bias`;
* per-layer `MagneticNonlinearBias` modules that read it and own only the four
  $W$ matrices and the two $\gamma$.

Memory: $E$ is held live across the forward — at $N{=}512$, bf16, that is 33 MB per
batch element, once rather than per layer. The per-layer pooling logits are
$(B,N,N,H_Q{+}H_{KV})$, ~21 MB per layer per element at the same $N$. Both are
within the envelope dense `magnetic` already trains in. Neither WebQSP nor GraphQA
is near the 4k regime where this would bind.

### 5.2 Wiring

`magnetic_nonlinear` threaded through `kgqa` and `graphqa`: each has its own
`config.py` gate, `__main__.py` flag, `train.py` bias-param dict, plus
`data_config_key`, `WIRED_FEATURES` and the collator. **Config validation must
raise, not return `None`** — `MIXED_BIAS.md` §4.1's silent-failure mode is a gate
that misses the new key, yielding `magnetic=None` → `forward` returns `None` → a
run that trains cleanly *with no bias at all* and reads as a clean negative result.

No new save/load code: `active_params = ("graph_bias",)` is a substring match
(`models/io.py`), so anything under the bias module is captured — the new top-level
trunk module must therefore be named to contain `graph_bias`, as
`shared_graph_bias` already is. Test 13 pins it.

### 5.3 Instrumentation

Two numbers logged per layer, both mandatory before any null is read:

* **`bias_gain_absmax`-analogue** — $\max|\gamma^{out}|\cdot\max|\gamma^{in}|\cdot d_{struct}$,
  the §2.4 range bound.
* **Pool entropy** — $H(w^{out})$ against $\log n_b$. This is the v2 lesson: arm v2's
  null was only readable because `gate_audit.py` proved its gate had left its
  identity initialization. If the pooling stays uniform, a null means "the
  mechanism never engaged", not "the feature does not help", and the
  uniform-weight ablation (§8) becomes the entire result.

---

## 6. What Phase 1 must not get wrong

Ranked by how silently each fails.

1. **Pooling axis.** $w^{out}$ normalizes over $j$, $w^{in}$ over $i$. Transposing
   one makes both features symmetric summaries, kills directionality, and changes
   nothing that any shape assertion would catch. Test 1 is the backstop.
2. **The GQA map.** $g(h) = h\,//\,(H_Q/H_{KV})$, matching `repeat_kv`
   (`flex_attn/flex_core.py:74-76`). Wrong here just permutes which head reads
   which key map — invisible in the dense path.
3. **The $i=j$ mask clause** (§2.2), in **both** pools.
4. **One-sided zero-init** (§2.4).

---

## 7. Correctness gate — before any sbatch

Numbered to match `MIXED_BIAS.md` §4.2; most of that suite applies verbatim.

1. **Permutation invariance** (`test_flex_cpu.py`) — relabelling nodes leaves
   prompt logits bit-identical. Fixtures must include a **star and a cycle**, so
   the spectrum is actually degenerate.
15. **Degenerate-spectrum invariance** — the 5-node star under two orderings;
    $z^{out}$, $z^{in}$ match to fp tolerance.
2. **Padded slots** — a batch with node counts `(4, 40, 7)`, as
   `test_v2_ragged_magnetic_padding.py` uses: assert no `NaN` in forward *or*
   backward, and that the small graphs' bias is unchanged by the presence of the
   large one.
9. **Zero-init inertness** — logits bit-identical to the no-bias model at step 0,
   **and** $\partial \mathcal{L}/\partial \gamma^{in} \ne 0$. Both halves: the
   second is what separates a correct zero-init from the dead saddle.
10. **No silent no-op** — the key with features absent must **raise**.
3. **Backward compatibility** — key off, bit-identical to today.
11. **Appended-Q/K parity (fp64, CPU)** — $\langle z^{out}_i, z^{in}_j\rangle$ built
    from the concatenated $Q'$/$K'$ reproduces the dense bias on the full matrix.
    Pure maths, cheap, and it is the whole de-risking of §9.
13. **Save/load round-trip** — `bias_parameters.pt` carries the trunk, both $W$
    pairs and both $\gamma$.
7. **Bias regularization** — the trainer's rule is `p.ndim >= 2`
    (`text_graph_trainer_v2.py:82`); confirm the new weights land as intended and
    that the 2-D $\gamma$ are classified deliberately.

---

## 8. Phase 2 — the training comparison

WebQSP and GraphQA together. Everything at $M{=}64$, `bias_self_node=True`.

| arm | key | isolates |
|---|---|---|
| 0 | *(no soft bias)* | the floor |
| 1 | `magnetic` (masked) | the dense ceiling — context only, not like-for-like |
| 2 | `magnetic_linear` | **the primary comparator**; this arm replaces it |
| N | `magnetic_nonlinear` | the proposal |
| N-u | `magnetic_nonlinear`, uniform weights | is the *learned* pool doing the work? |

Arms 0–2 are **reused**, not re-run: `021`/`023` already measured WebQSP arms 1
and 2 at $M{=}64$ unmasked, and `019`/`013`/`017` cover GraphQA.

**LR grid `{1e-3, 5e-3}`**, down from `{5e-3, 2e-2}`, because §4 is ~4.6× the
incumbent bias head. Retaining 5e-3 is what makes the reused baselines comparable.
3 seeds; read paired by seed.

WebQSP is the arm with statistical power: the residual there is ~3 pp against a
seed sd of 0.19–0.89 pp. GraphQA is the discriminating test — arm 3 recovered ~100%
on `node_degree` and `edge_count` and sat at the floor on path tasks, so a
row-pooled feature should differ there if the hypothesis is right. Context-4k stays
open, as it has since `020`.

Read the two traps in `mixed_bias/README.md` before aggregating: the `arm` label in
`013`/`017` `runs.jsonl` is misaligned with the flags — key off the flags.

---

## 9. Deferred — the fused kernel

Out of scope. Recorded so the constraints are not re-derived.

The target is **not** deleting an N² tensor — §2.1 keeps one. It is that the bias
never enters the *attention* kernel: attention becomes a stock flash call on
$Q'$/$K'$, and the N² work moves into a separate pooling kernel outside the
attention path. Because of §2.3, that kernel's entire job is to emit
$\bar E \in (B,N,H,d_{mag})$ — a masked softmax-weighted pooling of
$\mathrm{SiLU}(h)$ over the node axis, with $h$ recomputed on the fly from
$V_i, V_j, \Phi$. $W_{val}$, RMSNorm and $\gamma$ all apply outside it as ordinary
node-level ops.

Test 11 is what licenses building it. It is scheduled only if Phase 2 closes at
least half of WebQSP's residual to dense `magnetic`.

**Not scheduled, 2026-08-15.** Phase 2 closed **none** of that residual (22.7% of
headroom against arm 2's 87.8%). `036` additionally removed the one route by which
a rebuild could have been justified: had the loss been a capacity effect, a wider
$d_{struct}$ would have been worth fusing, but the deficit is flat in $N$, so
there is no size at which this kernel would pay. The pooling kernel described
above should not be built.

---

## 10. Work plan

Ordered by dependency. Nothing below step 5 runs until step 5 is green.

### Step 1 — `bias.py`

- [ ] `MagneticPairTrunk(MagneticBias)`, `config_key = 'magnetic_nonlinear'`,
      `shared = True`. Builds `lambda_lin` / `deep_set` / a 2-slot `proj`
      (`Linear(2m, m)` + `SiLU`) so `_phi` and `_folded_spectral` are inherited
      verbatim. `forward` returns $E$ `(B,N,N,d_mag)` — **not** a bias, so it is
      excluded from the shared-bias *sum* and threaded on its own context field.
- [ ] `MagneticNonlinearBias(BaseBias)`, `config_key = 'magnetic_nonlinear'`,
      per layer. Owns $W^{out}_{attn}, W^{out}_{val}, W^{in}_{attn}, W^{in}_{val},
      \gamma^{out}, \gamma^{in}$ only. Consumes `pair_features` + `num_nodes`,
      returns `(B,H,N,N)`.
- [ ] `_pool(E, W_attn, W_val, over_j: bool, num_nodes)` — one helper, both
      directions, so the two pools cannot drift apart. §2.2 mask, §2.3 pool-first.
- [ ] `structural_factors` returning $(z^{out}, z^{in})$ for test 11.
- [ ] Register both in `BIAS_TYPES`.

### Step 2 — plumbing

- [ ] `GraphContext.shared_pair_features: Optional[torch.Tensor]`.
- [ ] `GraphCausalLMMixin.__init__`: build the trunk into an attribute whose name
      contains `graph_bias` (`io.py` saves by substring). `forward`: compute once,
      outside the checkpointed layers, eval-cached like `shared_node_bias`.
- [ ] `dispatch.compute_node_bias`: pass `pair_features=ctx.shared_pair_features`
      **inside** the checkpoint closure, so the N²·H pooling is recomputed in
      backward while $E$ itself is saved once.
- [ ] `GraphAttentionBias.forward`: accept and forward the kwarg.

### Step 3 — config

- [ ] `GraphConfigMixin`: `magnetic_nonlinear`, `magnetic_struct_dim` (=64);
      add to the mutual-exclusion rule and to the `placements` enumeration.

### Step 4 — experiment wiring (`kgqa`, `graphqa`)

- [ ] `config.py`: fields, `uses_magnetic`, the bias-cfg branch, the arm label,
      `validate`. **`magnetic_dim` is not in `data_config_key`** — verified — so
      $d_{mag}{=}64$ reuses every existing build.
- [ ] `__main__.py`: `--magnetic-nonlinear`, `--magnetic-struct-dim`.
- [ ] `train.py`: the new flag in the run record.

### Step 5 — the §7 gate

- [ ] Tests 1, 15, 2, 9, 10, 3, 11, 13, 7 in
      `tests/models/test_magnetic_nonlinear_bias.py` (+ fixtures where §7 says so).
- [ ] Run on a **compute node**. All green, no exceptions.

### Step 6 — instrumentation

- [ ] Pool entropy $H(w^{out})/\log n_b$ and the §2.4 range bound, logged per layer.
- [ ] `pool_audit.py` over a finished checkpoint, mirroring `gate_audit.py`.

### Step 7 — sweeps

- [ ] `src/experiments/nonlinear_bias/` — `preflight.py`, `README.md`, configs
      `030_webqsp_nonlinear` and `031_graphqa_nonlinear`.
- [ ] Preflight green → submit both.

### Step 8 — read

- [x] Pool audit first: if the pool never left uniform, the result is unreadable.
- [x] `sweep.report`, paired by seed, against the reused arm-0/1/2 references.
- [x] Verdict into `src/experiments/nonlinear_bias/README.md`.

**All steps complete.** One defect reached the queue and voided a sweep: the head's
parameters were initialised by an `nn.init.*` applied *after* registration, which
`from_pretrained` does not preserve for a bare `nn.Parameter` on a custom module —
`W_attn` came out exactly 0, silently collapsing the learned pool into the uniform
ablation. Step 5's suite missed it because every test built the module with the
constructor, where the deferred init works. The pool audit of Step 6 caught it
before any number was read. Both the fix (initialise inside the `nn.Parameter(...)`
call) and the regression test that pins it are in place; the full account is in the
experiment README under "030 and 031 are VOID".
