## GTLM v2 Feature Spec: Content-Conditioned Magnetic Bias

### 1. Objective
The standard GTLM magnetic bias routes attention across RoPE-reset node
boundaries using a purely **structural, input-static** signal: the magnetic
Laplacian spectral features are precomputed once and are identical at every
layer. This spec adds a **Content-Conditioned Bias** (`magnetic_content`,
internally `mag_cont` — "magnetic + content") that injects a dynamically
aggregated, low-dimensional **semantic summary of each node** into that existing
magnetic-Laplacian edge feature. The result acts as a smart semantic gate on
cross-node routing while strictly preserving permutation equivariance and a tiny
(~0.015%) parameter budget.

The single research goal is to **isolate the effect of conditioning the magnetic
bias on content priors** on downstream performance. `magnetic_content` is a
**new, additional** bias type registered alongside the existing biases; it is a
per-layer bias (one instance per transformer layer, exactly like `magnetic`).
The current `magnetic` / `magnetic_shared` types are kept **unchanged** as
options, so the ablation is a clean one-flag swap.

### 2. The First-Token API (Semantic Sink)
The architecture makes no assumptions about specific token vocabularies (e.g.
hardcoded `<NODE_SUMMARY>` tokens). It relies on a structural **First-Token
Assumption**: whatever token happens to be first in a node's sequence acts as
that node's semantic summary slot. (The dataloader *may* later place a
user-defined tag — `<ENTITY>`, `<RELATION>`, … — there; the mechanism works on
whatever the first token is and needs no data change to function.)

*   **Extraction:** at each layer $l$, the model slices the hidden states of the
    first token of every node to form the dynamic per-node summary:

    $$H_{\text{sum}}^{(l)} = H^{(l)}[\text{node\_start\_indices}] \in \mathbb{R}^{N \times d_{\text{model}}}$$

*   **`node_start_indices` — the one genuinely new data path.** Every existing
    bias consumes precomputed *node-level* structural features; `magnetic_content`
    is the first to consume the **live token-level hidden states** $H^{(l)}$. The
    model already carries `node_ids` `(B, kv_len)` (token → node). This bias needs
    the **inverse**: `node_start_indices` `(B, N)` = position of the first token
    whose `node_ids == n`, derived **once** from `node_ids` (first occurrence per
    id) and reused by every layer.

*   **Obtaining $H^{(l)}$ (the new plumbing).** Under Strategy B there is no
    attention `forward` override, and the registered `gtlm_*` function receives
    only q/k/v (post-projection), **not** the raw residual-stream hidden state.
    `magnetic_content` therefore captures $H^{(l)}$ with a **forward pre-hook on
    each attention module** that stashes the layer's input `hidden_states` onto
    the module (mirroring the existing set-and-leave `_graph_ctx` pattern in
    `context.py`). This preserves the "init-only swaps, no forward override"
    invariant, matches the module-attached-context idiom already in use, and is
    correct under gradient checkpointing for free — the pre-hook fires again on
    recompute, so the stashed $H^{(l)}$ is always the freshly-recomputed tensor.

*   **Batching / padding.** `node_start_indices` is padded to `N_max`. Padded node
    slots MUST use a **safe sentinel index (clamp to 0), never an out-of-bounds
    gather**. Their gathered summaries are harmless: `num_nodes`/`valid` masking
    and the final `node_ids` token-expansion (`expand_node_to_token_bias`) already
    ensure padded nodes never map to real tokens. The only invariant to preserve
    is that a padded gather cannot produce a NaN/Inf that survives into a valid
    $(u,v)$ entry before masking — a plain MLP over finite clamped-index values
    satisfies this.

### 3. Config Surface — additive, backward-compatible
Nothing existing migrates; the config surface only grows:

*   `magnetic_content: bool = False` — enables the new bias type.
*   `magnetic_content_dim: int = 128` — the down-projection width $d_{\text{proj}}$.
*   The existing `magnetic` / `magnetic_shared` fields and every current config
    are **untouched** and behave identically.

`magnetic_content` reuses the magnetic-Laplacian machinery, so it consumes the
**same input features** as `magnetic` (the eigenvectors/eigenvalues
`magnetic_V` / `magnetic_lambdas` supplied by the dataloader) plus the live
hidden states. Enabling it therefore requires the magnetic features to be present
in the batch, exactly as `magnetic` does.

### 4. The Content-Conditioned Bias Formulation
The injection reuses the magnetic-Laplacian machinery (see `MagneticBias` in
`bias.py`). That module folds a per-node feature `phi` through the eigenvector
einsums into a per-edge, basis-invariant spectral feature
$\text{Spectral}(u,v) \in \mathbb{R}^{N \times N \times m}$ (the `hidden` tensor,
i.e. the input to the final projection `proj[2]`).

`magnetic_content` widens exactly that final-MLP input. At each layer $l$:

1.  **Down-project the summaries** through a small MLP:
    $$Z^{(l)} = \text{MLP}_{\text{down}}\!\left(H_{\text{sum}}^{(l)}\right) \in \mathbb{R}^{N \times d_{\text{proj}}}$$
    with $d_{\text{proj}}$ = `magnetic_content_dim` (default $128$).
2.  **Concatenate** the down-projected summaries of the two endpoints onto the
    existing per-edge spectral feature, endpoint order $(u, v)$:
    $$\text{feat}(u,v) = \left[\, \text{Spectral}(u,v) \;\|\; Z_u^{(l)} \;\|\; Z_v^{(l)} \,\right]$$
3.  **Feed the (widened) final MLP** to produce the scalar per-head offset:
    $$b_{\text{mag\_cont}}^{(l)}(u,v) = \text{MLP}_{\text{final}}\big(\text{feat}(u,v)\big)$$

Concretely this replaces `edge_features` with
`cat(edge_features, sum_u_down, sum_v_down)` at the `proj[2]` boundary and widens
that layer's input dimension by $2 \cdot d_{\text{proj}}$ accordingly.

### 5. Modified Attention Matrix
For attention layer $l$ the logits are:

$$A_{ij}^{(l)} = \frac{Q_i^{(l)} {K_j^{(l)}}^T}{\sqrt{d}} + b_{\text{spd}}(u,v) + b_{\text{mag\_cont}}^{(l)}(u,v)$$

*   **Intra-node ($u = v$):** structural biases zero out (diagonal masked as in
    `MagneticBias`); standard RoPE guides 1D token composition.
*   **Inter-node ($u \neq v$):** the content-conditioned $b_{\text{mag\_cont}}^{(l)}$
    provides a semantically gated routing scalar, bypassing cross-node RoPE
    scrambling.

### 6. Optimization and Gradient Flow
*   **Full end-to-end backprop:** no gradient detachment (`stop_gradient`) on the
    summary tokens. Gradients from the $O(N^2)$ structural routing loss flow
    backward through the final MLP, through $\text{MLP}_{\text{down}}$, and directly
    into the sequence hidden states — incentivizing the LM's early attention layers
    to aggregate critical semantic payload into each node's first token, serving
    both text generation and graph routing.
*   **Zero-init the final MLP layer** (as `MagneticBias` already does at `proj[2]`):
    the content term starts at 0 and grows in, so it cannot destabilize training
    from step 0.
*   **Layer-0 behavior is intended and safe.** The bias at layer $l$ reads
    layer-$l$ hidden states, so the earliest bias is conditioned on near-raw
    embeddings. If all nodes share the same summary token this degenerates to a
    structure-only bias (a fine prior); if the summary token is type-specific it is
    strictly better. Be aware of the intended feedback loop (summaries both shape
    and are shaped by routing) when debugging training dynamics — zero-init keeps it
    from diverging from the start.

### 7. Scaling
No fixed `max N` for now. Each `magnetic_content` layer costs $O(N^2)$ in the
final MLP (it runs once per layer, like `magnetic`). Measure empirically and
adjust $d_{\text{proj}}$ / inputs as needed.
