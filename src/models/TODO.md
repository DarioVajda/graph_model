## GTLM v2 Architecture Specification: Content-Conditioned Structural Routing

### 1. Architectural Objective
The standard GTLM architecture forces the language model to route attention across
RoPE-reset node boundaries using purely structural, input-static biases (e.g. the
magnetic Laplacian bias, whose spectral features are precomputed once and are
identical at every layer). This specification introduces a **Content-Conditioned
Bias** (`magnetic_content`, internally `mag_cont` — "magnetic + content") that
injects a dynamically aggregated, low-dimensional semantic summary of each node
into the existing magnetic-Laplacian edge feature. The result acts as a smart
semantic gate while strictly preserving permutation equivariance and a tiny
(~0.015%) parameter budget.

`magnetic_content` is a **new, additional** bias type registered alongside the
existing biases. The current `magnetic` / `magnetic_shared` types are kept
unchanged as options.

### 2. The First-Token API (Semantic Sink)
The architecture makes no assumptions about specific token vocabularies (e.g.
hardcoded `<NODE_SUMMARY>` tokens). Instead it relies on a structural
**First-Token Assumption**.

*   **Data Layout:** The dataloader can place any user-defined tag (e.g.
    `<ENTITY>`, `<RELATION>`) as the first token of a node's sequence.
*   **Extraction:** At a compute layer $l$, the model slices the hidden states of
    the first token of every node to act as the dynamic semantic summary.

$$H_{\text{sum}}^{(l)} = H^{(l)}[\text{node\_start\_indices}] \in \mathbb{R}^{N \times d_{\text{model}}}$$

*   **`node_start_indices` — the one genuinely new data path.** Every existing
    bias consumes precomputed *node-level* structural features; `magnetic_content`
    is the first to consume the **live token-level hidden states** $H^{(l)}$. The
    model already carries `node_ids` `(B, kv_len)` (token → node). This bias needs
    the **inverse**: `node_start_indices` `(B, N)` = position of the first token
    whose `node_ids == n`, derived once from `node_ids` (first occurrence per id)
    and reused at every compute layer.
*   **Batching / padding.** `node_start_indices` is padded to `N_max`. Padded node
    slots MUST use a **safe sentinel index (clamp to 0), never an out-of-bounds
    gather**. Their gathered summaries are harmless: `num_nodes`/`valid` masking
    and the final `node_ids` token-expansion (`expand_node_to_token_bias`) already
    ensure padded nodes never map to real tokens. The only invariant to preserve
    is that a padded gather cannot produce a NaN/Inf that survives into a valid
    $(u,v)$ entry before masking — a plain MLP over finite clamped-index values
    satisfies this.

### 3. Bias-Stride Mechanism (unifies `shared` into a general knob)
`bias_stride` generalizes the existing per-layer / shared distinction into a
single integer:

| Regime            | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `stride = 1`      | per-layer compute (≡ old `shared = False`) — DEFAULT |
| `stride = L`      | computed once, reused everywhere (≡ old `shared = True`, `L = n_layers`) |
| `1 < stride < L`  | the new intermediate capability                      |

**Semantics of the backend** (applies to every bias type): the bias is computed
only at layers where $l \bmod \text{stride} = 0$; that layer uses its own weights,
caches the result in the graph context, and the subsequent $\text{stride}-1$
layers reuse the cached tensor.

Worked example — 8 layers, `stride = 4`: layer 1 computes (own weights) → uses →
caches; layers 2–4 reuse layer 1's cache; layer 5 computes (own weights) → uses →
caches; layers 6–8 reuse layer 5's cache.

**Two placement regimes the unified backend must honor** (the old `shared` flag
was never *only* caching — it was also placement):

*   **Static-input biases** (spd, laplacian, magnetic, rwse, rrwp): input is
    identical at every layer, so stride only controls **parameter sharing** across
    strided groups. When `stride ≥ n_layers`, the bias is additionally eligible for
    the **hoist-outside-checkpointing** optimization (compute once, outside the
    gradient-checkpointed decoder layers, so the $O(N^2)$ work runs once per
    forward instead of once-per-layer-per-recompute — the current
    `magnetic_shared` win, which must be preserved).
*   **Dynamic-input bias** (`magnetic_content`): the input *is* the evolving
    hidden state, so it **must** be computed inside the layer stack at each compute
    layer. It cannot be hoisted and is inherently subject to recompute.

The backend therefore carries a "hoistable?" capability flag = (static input) ∧
(`stride ≥ n_layers`). The default stride for `magnetic_content` is **1**.

*Note:* the mechanism supports intermediate strides for the static biases for
free, but the only effect there is parameter-sharing groups (dubious value); leave
that undocumented/untuned until there is a reason. The interesting stride knob is
on `magnetic_content`.

### 4. Config Surface — additive, backward-compatible
The internal model gets the stride-unified backend, but the **config surface only
grows** — nothing existing migrates:

*   Add an optional per-bias `stride` field, defaulting so existing configs behave
    **identically** (currently-per-layer types default `stride = 1`; the shared
    magnetic keeps its current behavior).
*   Keep `magnetic_shared: true` working as a **thin alias** routing to the unified
    backend with `stride = n_layers`. Old configs and call sites are untouched.
*   `magnetic_content` is a new registered type with its own `stride` (default 1).

This avoids a global config-schema rewrite (and the "million call sites" churn):
new behavior is strictly opt-in.

### 5. The Content-Conditioned Bias Formulation
The injection reuses the magnetic-Laplacian machinery (see `MagneticBias` in
`bias.py`). That module folds a per-node feature `phi` through the eigenvector
einsums into a per-edge, basis-invariant spectral feature
$\text{Spectral}(u,v) \in \mathbb{R}^{N \times N \times m}$ (the `hidden` tensor,
i.e. the input to the final projection `proj[2]`).

`magnetic_content` widens exactly that final-MLP input. At each compute layer $l$:

1.  **Down-project the summaries** through a small MLP:
    $$Z^{(l)} = \text{MLP}_{\text{down}}\!\left(H_{\text{sum}}^{(l)}\right) \in \mathbb{R}^{N \times d_{\text{proj}}}$$
    with $d_{\text{proj}}$ configurable (default $128$).
2.  **Concatenate** the down-projected summaries of the two endpoints onto the
    existing per-edge spectral feature, endpoint order $(u, v)$:
    $$\text{feat}(u,v) = \left[\, \text{Spectral}(u,v) \;\|\; Z_u^{(l)} \;\|\; Z_v^{(l)} \,\right]$$
3.  **Feed the (widened) final MLP** to produce the scalar per-head offset:
    $$b_{\text{mag\_cont}}^{(l)}(u,v) = \text{MLP}_{\text{final}}\big(\text{feat}(u,v)\big)$$

Concretely this replaces `edge_features` with
`cat(edge_features, sum_u_down, sum_v_down)` at the `proj[2]` boundary and widens
that layer's input dimension accordingly.

### 6. Modified Attention Matrix
For any attention layer $k$ within the active stride window
($l \le k < l + \text{stride}$), the logits are:

$$A_{ij}^{(k)} = \frac{Q_i^{(k)} {K_j^{(k)}}^T}{\sqrt{d}} + b_{\text{spd}}(u,v) + b_{\text{mag\_cont}}^{(l)}(u,v)$$

*   **Intra-node ($u = v$):** structural biases zero out (diagonal masked as in
    `MagneticBias`); standard RoPE guides 1D token composition.
*   **Inter-node ($u \neq v$):** the content-conditioned $b_{\text{mag\_cont}}^{(l)}$
    provides a semantically gated routing scalar, bypassing cross-node RoPE
    scrambling.

### 7. Optimization and Gradient Flow
*   **Full end-to-end backprop:** no gradient detachment (`stop_gradient`) on the
    summary tokens. Gradients from the $O(N^2)$ structural routing loss flow
    backward through the final MLP, through $\text{MLP}_{\text{down}}$, and directly
    into the sequence hidden states — incentivizing the LM's early attention layers
    to aggregate critical semantic payload into each node's first token, serving
    both text generation and graph routing.
*   **Zero-init the final MLP layer** (as `MagneticBias` already does at `proj[2]`):
    the content term starts at 0 and grows in, so it cannot destabilize training
    from step 0.
*   **Layer-0 behavior is intended and safe.** At `stride = 1` the bias at layer $l$
    reads layer-$l$ hidden states, so the earliest bias is conditioned on near-raw
    embeddings. If all nodes share the same summary token this degenerates to a
    structure-only bias (a fine prior); if the summary token is type-specific it is
    strictly better. Be aware of the intended feedback loop (summaries both shape
    and are shaped by routing) when debugging training dynamics — zero-init keeps it
    from diverging from the start.

### 8. Scaling
No fixed `max N` for now. Each `magnetic_content` compute is $O(N^2)$ in the final
MLP; `bias_stride` amortizes how *often* that runs but not the per-invocation cost.
Measure empirically and adjust $d_{\text{proj}}$ / inputs as needed.
