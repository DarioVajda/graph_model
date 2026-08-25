### Landmark-Based Factorized Positional Encoding for GTLM

The bias is a **learned soft-min over the classical landmark distance oracle**
$d(u,v) \approx \min_j\big[d(u,a_j) + d(a_j,v)\big]$. The oracle's $\min$ over
*matched* anchor indices is what a bilinear form can approximate as a sum of
products — so the anchor axis is carried intact into the inner product and never
mixed by an MLP.

#### 1. Hyperparameters & Dimensions
*   $N$: nodes in $G=(V,E)$; $H_Q / H_{KV}$: query / KV heads ($32/8$ on Llama-1B)
*   $k$: landmark anchors (Configurable: $k \in \{8,16,32\}$); $k_{val} = \min(k,N)$ slots are real anchors, the remainder are $\textsf{PAD}$
*   $d_{max} = 8$: SPD clip threshold. Measured, not assumed — 99.999% of finite WebQSP anchor distances are $\le 8$ and the mode is 3, so the original 16 was dead range (`landmark/README.md` §5).
*   $S = d_{max}+3 = 11$: distance alphabet — $\{0,\dots,d_{max}\}$, $\textsf{UNREACH}$, $\textsf{PAD}$
*   $d_{pos} = 3k \in \{24,48,96\}$: structural dims appended to each head. The $2\times$ `head_dim` cap binds at $k = 42$; the directed-only ablation (§4) is $2k$.
*   $d_{head}$: the backbone's own head dim (64), **unchanged**

#### 2. Pre-computation (Static, Graph-Level)
Anchors $A = \{a_1,\dots,a_k\} \subset V$ chosen by a rule that is a function of
the graph **up to isomorphism** — otherwise relabelling changes the anchor set and
Property 1 fails. Ties broken on a structural signature, never on node index.
Uniform-random selection is a function of the *labelling*, so it is a Phase-0
diagnostic only, never a candidate.

**The rule is `degree`** (in + out), settled by Phase 0 across every $k$
(`landmark/README.md` §2). `betweenness` — the rule the mechanism implies, since
$\hat o = d$ iff an anchor lies on a shortest $u{\to}v$ path — is a near-tie that
loses on 3 of 4 metrics at $O(NM)$ instead of $O(1)$; keep it only as a cheap
training arm. `pagerank`, `mixed` and `fps` are **eliminated**, each for a
mechanism rather than a margin: directed PageRank mass accumulates at **sinks**,
whose $D_{in}$ column is dead by construction; FPS seeks the *periphery*, which in
a sparse directed graph neither reaches nor is reached (45% of nodes get a fully
dead row); `mixed` inherits FPS's half. Uniform-random loses by 45 pp of
$\hat o = d$, so the rule is load-bearing, not a free parameter.

**Selection is component-stratified**, and anchors are emitted **round-robin**
across weakly connected components (components by size desc, within a component by
degree desc). Stratification is what stops `degree` from pooling every anchor in
the largest component and leaving all other nodes with an all-$\textsf{UNREACH}$
row — GraphQA's relational tasks are 24–26% disconnected, so this is a quarter of
that data, not an edge case. Round-robin is what makes a *prefix* of the stored
anchors a valid smaller-$k$ selection, which keeps the $k$-sweep out of the dataset
cache key (`landmark_k_collate`, the trick `magnetic_m_collate` plays for $M$).

Weak components, not strong: inside a weakly-connected component
$\text{SPD}(i,a_j)$ can still be $\infty$ — a measured risk (94.4% of WebQSP pairs)
that the allocation cannot fix and channel 3 exists to answer.

Ties are broken by directed **Weisfeiler-Lehman** colour refinement, not by degree
alone: on a KG every degree-1 leaf looks alike under degree, and a tie broken by
node index is a tie broken by the labelling. WL cannot separate nodes inside one
automorphism orbit; there the index decides, and that residual is real — choosing
$k$ of $n$ interchangeable nodes requires breaking a symmetry no deterministic rule
can break equivariantly.

Distances come from $k$-source BFS, not from the stored APSP: only $3k$ rows are
needed, the undirected skeleton is not stored at all, and $O(k(N{+}E))$ beats
reading an $N^2$ column.
$$ (D_{out})_{i,j} = \min\big(\text{SPD}(i, a_j),\, d_{max}\big), \qquad
   (D_{in})_{i,j}  = \min\big(\text{SPD}(a_j, i),\, d_{max}\big) \;\in\; \{0,\dots,S{-}1\} $$
and the same distances on the **undirected skeleton**, which is a different
quantity and not recoverable from the two above:
$$ (D_{und})_{i,j} = \min\big(\widetilde{\text{SPD}}(i, a_j),\, d_{max}\big) $$
*   **Tensor shapes:** $D_{out}, D_{in}, D_{und} \in \mathbb{N}^{N \times k}$ — $O(Nk)$, not $O(N^2)$
*   Unreachable $\to \textsf{UNREACH}$; unused anchor slots when $k > N \to \textsf{PAD}$

**Why the undirected block is not optional on a KG.** Phase 0 measured **94.4% of
WebQSP node pairs with no directed path**, inside a graph that is a *single* weak
component — the Levi construction only permits forward traversal along triples. The
directed channels are therefore silent on 18 of every 19 pairs, while the
undirected skeleton reaches all of them. `factorized_rwpe`'s Phase 0 reached the
same conclusion independently on the same dataset, for a different bias family.

Anchor *ordering* is irrelevant by construction (§4), so no canonical sort is
needed — but which nodes are *selected* still must be equivariant, which is what
the tie-break policy above is for.

#### 3. Parameters (Per Layer $l$)
Six distance→scalar lookup tables. Every learned parameter that carries a head
index sits on the **query** side:
$$ F_1^{(l)}, F_2^{(l)}, F_3^{(l)} \in \mathbb{R}^{S \times H_Q}, \qquad
   G_1^{(l)}, G_2^{(l)}, G_3^{(l)} \in \mathbb{R}^{S}, \qquad \gamma^{(l)} \in \mathbb{R}^{H_Q} $$

plus a per-head gain $\gamma^{(l)} \in \mathbb{R}^{H_Q}$, which is what bounds the
bias (see §4) rather than a redundant scale.

**Initialization.** $F_\cdot[d,h] = G_\cdot[d] = e^{-d/\tau}$ (with
$\textsf{UNREACH}, \textsf{PAD} \mapsto 0$) — both sides *start* as the distance
oracle — and $\gamma = 0$. So $b \equiv 0$ at step 0, while
$\partial b/\partial\gamma = \langle\hat q, \hat k\rangle \neq 0$ moves $\gamma$
at step 1 and the tables at step 2. A two-step unroll, not a saddle.

Zeroing $G$ instead is the trap: under the norm, $\hat k = \mathrm{normalize}(0) =
0$ kills $\partial b/\partial\gamma$ as well, and nothing leaves the origin —
`NON_LINEAR_BIAS.md` §4.4's dead saddle.

**$\textsf{PAD}$ is inert structurally, not just at init.** $F[\textsf{PAD}]$ and
$G[\textsf{PAD}]$ are trainable rows like any other, so once training moves them a
padded anchor slot would inject a constant into every pair — and, worse, would
inflate the $L_2$ norm it is divided by. Both sides are masked by
$(\text{slot} \neq \textsf{PAD})$ *before* normalization. $\textsf{UNREACH}$ is
*not* masked: "no path" is a real symbol carrying real information, and it must
stay learnable.

#### 4. The Bias
Three channels: **(1)** $u \to a_j \to v$, **(2)** its reverse $v \to a_j \to u$,
and **(3)** the undirected route through $a_j$.
With $\widehat{\;\cdot\;}$ denoting $L_2$ normalization over the $k$ anchors of one channel:
$$ Q_{pos}[u,h,:] = \gamma_h\Big[\, \widehat{F_1[(D_{out})_{u,:},h]} \,\big\|\, \widehat{F_2[(D_{in})_{u,:},h]} \,\big\|\, \widehat{F_3[(D_{und})_{u,:},h]} \,\Big] \in \mathbb{R}^{3k} $$
$$ K_{pos}[v,:]   = \Big[\, \widehat{G_1[(D_{in})_{v,:}]} \,\big\|\, \widehat{G_2[(D_{out})_{v,:}]} \,\big\|\, \widehat{G_3[(D_{und})_{v,:}]} \,\Big] \in \mathbb{R}^{3k} $$
$$ b^{(h)}(u,v) = \big\langle Q_{pos}[u,h,:],\, K_{pos}[v,:] \big\rangle
   = \gamma_h \sum_{c=1}^{3}\sum_{j=1}^{k_{val}} \Big( \hat F_c \,\hat G_c \Big)_j \;\in\; \big[-3|\gamma_h|,\, 3|\gamma_h|\big] $$

The **directed-only $2k$ form** (channels 1–2, i.e. $F_3 = G_3 = 0$) is the
ablation that prices channel 3 against the head width it costs.

*   **Asymmetric**, hence directional: channel 1 reads $u$'s out-distances against $v$'s in-distances, channel 2 the transpose. $b(u,v) \neq b(v,u)$ genuinely. Channel 3 reads a symmetric *feature*, but it is still an asymmetric *bias* — $F_3 \neq G_3$ — so it adds coverage without diluting direction.
*   **Anchor-permutation invariant**: $F,G$ are indexed by *distance*, never by anchor slot $j$, so the sum is over an unordered set. This is also why anchors need not correspond across graphs.
*   **GQA-native**: $K_{pos}$ has no head dimension — one universal structural dictionary broadcast across all groups, exactly `LinearMagneticBias`'s property (`bias.py:568`). $H_Q$-way expressiveness is retained on the query side.
*   **Normalization** — load-bearing, and measured to be so. Each side's per-$(node, channel)$ anchor vector is $L_2$-normalized and a per-head gain $\gamma$ carries the magnitude, so Cauchy-Schwarz gives $|b| \le 3\max|\gamma|$: a hard bound, independent of $F$ and $G$. The bias is degree-0 in table scale and degree-1 in $\gamma$.
    An earlier draft argued no normalization was needed because the form is "degree-1 in each side". That is true and irrelevant: $b$ is the product of two *trainable* tables, hence **degree-2 in the learned parameters**, with no bound on $|F||G|$ at all. `magnetic_linear` is not a counterexample — one side there is the raw orthonormal eigenvector ($\lVert V\rVert \le 1$) and only $W$ is learned. Sweep 040 ran the unnormalized form and measured $\lVert b\rVert_{max} = 9$–$240$ against $O(1\text{–}10)$ attention logits; it scored **10 pp below the no-bias floor** and got worse as `bias_lr` rose (`landmark/README.md` §8). This is `MIXED_BIAS.md` §5.7 in a different costume and §5.8 is the remedy.
    Normalizing over the anchor axis also removes the $k$-dependence that a $1/k_{val}$ mean existed to remove, so no separate $k$ normalization is needed. $\textsf{PAD}$ slots are zeroed *before* the norm, so they cannot inflate it.
*   **Budget**: $3SH_Q + 3S \approx 1.1$k per layer, ~18k model-wide — still under the incumbent SPD table even with the third channel, because $d_{max}$ dropped from 16 to 8. Appended head width $d_{pos} = 3k$: 1.5× `head_dim` at $k{=}32$.

#### 5. Computation
**Phase 1 (this implementation) — dense simulation.** `LandmarkBias.forward` builds
the $(B,H,N,N)$ tensor by einsum over $Q_{pos}, K_{pos}$ ($O(N^2k)$) and adds it to
the post-scale logits where every existing bias enters (`dispatch.py:98`).
`structural_factors` returns $(Q_{pos}, K_{pos})$ so the equivalence is pinned in
fp64 before any backbone is built on it (`LINEAR_BIAS.md` §7).

**Deferred — the fused form.** Concatenating $[\,Q \parallel Q_{pos}\,]$ and
$[\,K \parallel K_{pos}\,]$ gives the bias for free inside the kernel, but the
kernel then derives $1/\sqrt{d_{head}+d_{pos}}$ and silently rescales pretrained
content attention by $\sqrt{d/(d+d_{pos})}$. Pass an explicit `softmax_scale` and
pre-multiply the structural block by $1/\sqrt{\text{scale}}$ on each side
(`LINEAR_BIAS.md` §7.2). Target is **flex** — deleting the `score_mod` — not FA2.

**The intra-node diagonal is unmasked.** An inner product yields $\langle
Q_{pos}[u], K_{pos}[u]\rangle$ and cannot be forced to $0$, so every arm runs
`bias_self_node=True`. Note this currently **excludes a joint `spd` arm** —
`config.py:174` raises, since `SPDBias` has no table row for self-distance.