"""
Model-agnostic graph attention bias module.

Each bias type is a BaseBias subclass registered in BIAS_TYPES.
GraphAttentionBias iterates that list, instantiates every enabled type, and
accumulates their outputs into a single (B, H, N, N) node-level tensor.

To add a new bias type:
  1. Create a class that inherits BaseBias.
  2. Set config_key to the attribute name on bias_config that enables it.
  3. Implement forward(**kwargs) → (B, H, N, N) tensor (or None if input absent).
  4. Append the class to BIAS_TYPES — no other changes needed.

Supported bias types
--------------------
SPDBias       - learnable lookup table keyed by shortest-path distance
LaplacianBias - learnable scalar weight × L2 distance between spectral embeddings
RWSEBias      - same pattern for random-walk structural encodings
RRWPBias      - small MLP applied to multi-hop random-walk probability vectors
MagneticBias  - complex-eigenvector-based directional encoding via deep-set MLP
MagneticContentBias - MagneticBias widened with a live per-node content summary
LinearMagneticBias  - MagneticBias with a linear head instead of the MLP, so the
                      bias is a bilinear form (see LINEAR_BIAS.md)
K-hop gate (hard) - -inf for node pairs more than K hops apart; K=0 = disabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Base class ────────────────────────────────────────────────────────────────

class BaseBias(nn.Module):
    """
    Base class for a single graph attention bias type.

    Subclass protocol:
      config_key : str   — attribute on bias_config that enables this type.
      shared     : bool  — False (default): one instance per transformer layer
                           (inside each layer's GraphAttentionBias). True: ONE
                           instance on the top-level model, computed once per
                           forward and added to every layer's bias (see
                           GraphCausalLMMixin / GraphContext.shared_node_bias).
      forward(**kwargs)  — consumes whatever it needs from the shared kwargs
                           dict and returns a (B, H, N, N) float tensor,
                           or None when its required input is absent.
    """

    config_key: str = ""
    shared: bool = False

    @classmethod
    def is_enabled(cls, bias_config) -> bool:
        return bool(getattr(bias_config, cls.config_key, False))


# ── Bias type implementations ─────────────────────────────────────────────────

class SPDBias(BaseBias):
    """Learnable lookup table indexed by shortest-path distance."""

    config_key = 'spd'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.max_spd = getattr(bias_config, 'max_spd', 32)
        self.weights = nn.Parameter(torch.zeros(self.max_spd, num_heads))
        # 2-D by shape but semantically an additive logit lookup (64 globally
        # shared values per head — it cannot memorize examples): exempt from the
        # trainer's shape-based weight-decay rule.
        self.weights._no_weight_decay = True

    def forward(self, *, dtype, device, spd=None, **kwargs) -> Optional[torch.Tensor]:
        if spd is None:
            return None
        non_zero = (spd > 0).unsqueeze(1)
        idx = torch.clamp(spd - 1, 0, self.max_spd - 1)
        b = F.embedding(idx, self.weights).permute(0, 3, 1, 2).to(dtype)
        return b * non_zero                                         # (B, H, N, N)


class LaplacianBias(BaseBias):
    """Learnable scalar weight x pairwise L2 distance between spectral embeddings."""

    config_key = 'laplacian'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_heads) * 0.02)

    def forward(self, *, dtype, device, laplacian=None, **kwargs) -> Optional[torch.Tensor]:
        if laplacian is None:
            return None
        dist = torch.cdist(laplacian, laplacian, p=2.0)            # (B, N, N)
        return dist.unsqueeze(1) * self.weights.view(-1, 1, 1).to(dtype)


class RWSEBias(BaseBias):
    """Same scalar-weight pattern as LaplacianBias, for random-walk SE features."""

    config_key = 'rwse'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_heads) * 0.02)

    def forward(self, *, dtype, device, rwse=None, **kwargs) -> Optional[torch.Tensor]:
        if rwse is None:
            return None
        dist = torch.cdist(rwse, rwse, p=2.0)                      # (B, N, N)
        return dist.unsqueeze(1) * self.weights.view(-1, 1, 1).to(dtype)


class RRWPBias(BaseBias):
    """Small MLP applied to multi-hop random-walk probability vectors."""

    config_key = 'rrwp'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        max_rw_steps = getattr(bias_config, 'max_rw_steps', 8)
        self.bias_self_node = getattr(bias_config, 'bias_self_node', False)
        hidden = 4 * max_rw_steps
        self.proj = nn.Sequential(
            nn.Linear(max_rw_steps, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, num_heads, bias=True),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        self.proj[2]._is_hf_initialized = True

    def forward(self, *, dtype, device, rrwp=None, **kwargs) -> Optional[torch.Tensor]:
        if rrwp is None:
            return None
        b = self.proj(rrwp).permute(0, 3, 1, 2).contiguous()      # (B, H, N, N)
        if self.bias_self_node:
            return b
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)


class MagneticBias(BaseBias):
    """Complex-eigenvector-based directional encoding via a deep-set MLP."""

    config_key = 'magnetic'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        magnetic_dim = getattr(bias_config, 'magnetic_dim', 32)
        # Read once here so every subclass (Shared/Content/Linear) inherits it via
        # super().__init__ and _finalize needs no per-class plumbing.
        self.bias_self_node = getattr(bias_config, 'bias_self_node', False)
        self.lambda_lin = nn.Linear(1, head_dim, bias=True)
        self.deep_set = nn.Sequential(
            nn.Linear(head_dim * 2, magnetic_dim, bias=True),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(magnetic_dim * 2, magnetic_dim, bias=True),
            nn.SiLU(),
            nn.Linear(magnetic_dim, num_heads, bias=True),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        self.proj[2]._is_hf_initialized = True

    # ── Shared spectral machinery (reused by MagneticContentBias) ──────────────

    def _phi(
        self, magnetic, num_nodes, device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Fold the eigenvalues into the deep-set per-node feature ``phi``.

        Returns ``(V_real, V_imag, phi)`` — the eigenvector real/imag parts
        ``(B, N, N)`` and ``phi`` ``(B, M, magnetic_dim)`` — or ``None`` when the
        magnetic input is absent.
        """
        if magnetic is None or num_nodes is None:
            return None
        V, lambdas = magnetic                                     # (B,N,N,2), (B,N)
        V_real, V_imag = V[..., 0], V[..., 1]                     # (B, N, N) each

        h_i   = self.lambda_lin(lambdas.unsqueeze(-1))            # (B, M, head_dim)
        valid = (torch.arange(lambdas.shape[1], device=device).unsqueeze(0)
                 < num_nodes.unsqueeze(1))                        # (B, M) bool

        # Divide by the number of valid eigenvalues, not num_nodes.
        # With full eigenvectors (M=N) these are equal; with truncated (M<N) they differ.
        n_valid = valid.sum(dim=1, keepdim=True).unsqueeze(-1).to(h_i.dtype).clamp(min=1)       # (B, 1, 1)
        h_avg   = (h_i * valid.unsqueeze(-1)).sum(1, keepdim=True) / n_valid                   # (B, 1, head_dim)

        phi = self.deep_set(torch.cat([h_i, h_avg.expand_as(h_i)], dim=-1))                    # (B, N, magnetic_dim)
        return V_real, V_imag, phi

    def _folded_spectral(self, V_real, V_imag, phi) -> torch.Tensor:
        """Folded per-edge spectral feature ``(B, N, N, magnetic_dim)`` — the
        input to ``proj[1]``.

        The first proj layer is linear, so project ``phi`` (B,M,m — tiny) BEFORE
        the N² einsums instead of their (B,N,N,2m) cat after. The (B,N,N,m)
        hidden is emitted directly; ``real``/``imag`` and the cat never exist,
        halving the largest per-layer intermediates. Uses the same parameters —
        ``proj[0]``'s weight is just split into its real/imag column halves.
        """
        W1, b1 = self.proj[0].weight, self.proj[0].bias           # (out, 2m), (out)
        # Split at HALF THE INPUT width — the real/imag halves of proj[0]'s input
        # — not at its output width. The two coincide for MagneticBias (out =
        # magnetic_dim = in/2) but not for LinearMagneticBias, whose head maps
        # straight to num_heads.
        m = W1.shape[1] // 2
        phiR = phi @ W1[:, :m].T                                  # (B, M, out)
        phiI = phi @ W1[:, m:].T                                  # (B, M, out)
        return (
            torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phiR)
            + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phiR)
            + torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phiI)
            - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phiI)
        ) + b1                                                    # (B, N, N, m)

    def _finalize(self, b, device) -> torch.Tensor:
        """``(B, N, N, H)`` → ``(B, H, N, N)``, zeroing the intra-node diagonal
        unless ``bias_self_node`` is set.

        The diagonal is a node-level quantity, and ``expand_node_to_token_bias``
        lifts node pairs to token pairs — so zeroing b_ii means EVERY token pair
        inside the same node gets no structural bias, not merely a token relative
        to itself.

        ``bias_self_node=True`` keeps it. At i=j the imaginary part of the
        Hermitian kernel vanishes identically and ``K(i,i) = sum_l |V_il|^2 phi_l``
        — a real per-node spectral self-energy, not noise. It is also the one part
        of the bias the factorization can reproduce for free (an inner product
        gives <q_i, k_i> and cannot be forced to 0), which is why this switch
        exists: see LINEAR_BIAS.md §7.3.

        Default False, so every existing config and checkpoint keeps the masked
        behaviour bit-for-bit.
        """
        b = b.permute(0, 3, 1, 2).contiguous()                    # (B, H, N, N)
        if getattr(self, 'bias_self_node', False):
            return b
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)

    def forward(
        self, *, dtype, device,
        magnetic: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_nodes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        parts = self._phi(magnetic, num_nodes, device)
        if parts is None:
            return None
        V_real, V_imag, phi = parts

        if getattr(self, "legacy_unfolded", False):
            # Original formulation, kept for parity testing: materializes the
            # (B,N,N,magnetic_dim) real/imag tensors AND their (…, 2m) cat
            # before the first projection.
            real = (torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phi) + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phi))
            imag = (torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phi) - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phi))
            b = self.proj(torch.cat([real, imag], dim=-1))
        else:
            hidden = self._folded_spectral(V_real, V_imag, phi)   # (B, N, N, m)
            b = self.proj[2](self.proj[1](hidden))                # SiLU, Linear

        return self._finalize(b, device)


class MagneticContentBias(MagneticBias):
    """Content-conditioned magnetic bias (per-layer).

    Reuses ``MagneticBias``'s spectral machinery up to the ``proj[1]`` output —
    the per-edge spectral feature ``Spectral(u,v) ∈ (B,N,N,m)`` — then widens the
    final projection with a low-dimensional semantic summary of each endpoint
    node, extracted live from the first token of every node's sequence.

    At each layer l:
      Z          = MLP_down(H^(l)[node_start_indices])   # (B, N, d_proj)
      feat(u,v)  = [ Spectral(u,v) ‖ Z_u ‖ Z_v ]         # (B, N, N, m + 2·d_proj)
      b(u,v)     = proj[2](feat)                          # (B, N, N, num_heads)

    proj[2] is zero-initialised (inherited pattern), so the content term starts
    at 0 and grows in — it cannot destabilise training from step 0. Gradients
    flow end-to-end into the hidden states (no detach), incentivising the LM to
    aggregate semantic payload into each node's first token.

    Consumes the same magnetic features as ``MagneticBias`` plus the live
    ``hidden_states`` (stashed on the attention module by a forward pre-hook) and
    ``node_start_indices`` (first-token position per node, derived from
    ``node_ids``). Returns ``None`` if any required input is absent.
    """

    config_key = 'magnetic_content'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__(num_heads, head_dim, bias_config)
        magnetic_dim = getattr(bias_config, 'magnetic_dim', 32)
        d_proj = getattr(bias_config, 'magnetic_content_dim', 128)
        d_model = getattr(bias_config, 'hidden_size')
        # Down-project each node's first-token hidden state to a tiny summary.
        self.down = nn.Sequential(
            nn.Linear(d_model, d_proj, bias=True),
            nn.SiLU(),
        )
        # Widen the final projection input by the two endpoint summaries, and
        # re-apply MagneticBias's zero-init to the (now wider) final layer.
        self.proj[2] = nn.Linear(magnetic_dim + 2 * d_proj, num_heads, bias=True)
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        self.proj[2]._is_hf_initialized = True

    def forward(
        self, *, dtype, device,
        magnetic: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_nodes: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        node_start_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if hidden_states is None or node_start_indices is None:
            return None
        parts = self._phi(magnetic, num_nodes, device)
        if parts is None:
            return None
        V_real, V_imag, phi = parts

        spectral = self.proj[1](self._folded_spectral(V_real, V_imag, phi))    # (B, N, N, m)

        # Per-node summary = first-token hidden state. node_start_indices is
        # clamp-to-0 padded; padded node slots gather a harmless real row (they
        # never map to real tokens after expand_node_to_token_bias / masking).
        d = hidden_states.shape[-1]
        idx = node_start_indices.unsqueeze(-1).expand(-1, -1, d)               # (B, N, d_model)
        h_sum = torch.gather(hidden_states, 1, idx)                            # (B, N, d_model)
        Z = self.down(h_sum.to(spectral.dtype))                               # (B, N, d_proj)

        N = spectral.shape[1]
        Zu = Z.unsqueeze(2).expand(-1, -1, N, -1)                             # (B, N, N, d_proj) endpoint u (row)
        Zv = Z.unsqueeze(1).expand(-1, N, -1, -1)                             # (B, N, N, d_proj) endpoint v (col)
        feat = torch.cat([spectral, Zu, Zv], dim=-1)                          # (B, N, N, m + 2·d_proj)

        b = self.proj[2](feat)                                                # (B, N, N, H)
        return self._finalize(b, device)


class LinearMagneticBias(MagneticBias):
    """MagneticBias with the 2-layer MLP head replaced by a single linear map.

    Why: the MLP's pointwise SiLU is the *only* thing standing between this bias
    and a bilinear form. Drop it and

        b^(h)(i,j) = sum_c W[c,h] * [Re K ‖ Im K]_c(i,j)

    becomes an inner product of two per-node vectors, i.e. exactly what attention
    already computes — so a future backbone can fold it into wider Q/K and delete
    the (B,N,N,·) tensor and the flex ``score_mod`` outright. See
    ``src/models/LINEAR_BIAS.md``; that optimized backbone is deliberately NOT
    what this class is. This is the same dense path as ``MagneticBias``, so any
    measured quality delta is attributable to the math and not an implementation.

    Reuses ``_phi`` unchanged (the eigenvalue DeepSets is untouched); only the
    head differs. ``proj`` is kept as an ``nn.Sequential`` of length 3 with
    Identity in the nonlinearity slot so that ``_folded_spectral`` — which reads
    ``self.proj[0]`` — works verbatim, and the folded/unfolded parity test applies
    to both classes without a special case.

    Zero-initialised like ``MagneticBias.proj[2]``, so the bias starts at exactly
    0 and cannot destabilise training from step 0.
    """

    config_key = 'magnetic_linear'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__(num_heads, head_dim, bias_config)
        magnetic_dim = getattr(bias_config, 'magnetic_dim', 32)
        # One linear map over [Re K ‖ Im K] (2*magnetic_dim) -> heads. Slot 1 is
        # Identity rather than SiLU: that single substitution IS the experiment.
        self.proj = nn.Sequential(
            nn.Linear(magnetic_dim * 2, num_heads, bias=True),
            nn.Identity(),
            nn.Identity(),
        )
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        self.proj[0]._is_hf_initialized = True

    def forward(
        self, *, dtype, device,
        magnetic: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_nodes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        parts = self._phi(magnetic, num_nodes, device)
        if parts is None:
            return None
        V_real, V_imag, phi = parts

        if getattr(self, "legacy_unfolded", False):
            # Parity path: materialize [Re ‖ Im] and apply the head to it, the
            # naive reading of the formula above.
            real = (torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phi)
                    + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phi))
            imag = (torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phi)
                    - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phi))
            b = self.proj[0](torch.cat([real, imag], dim=-1))
        else:
            # The head is linear, so it folds into phi exactly as proj[0] does in
            # MagneticBias: _folded_spectral already emits (B,N,N,out_features)
            # with the head applied, and no further layer is needed.
            b = self._folded_spectral(V_real, V_imag, phi)        # (B, N, N, H)

        return self._finalize(b, device)

    def structural_factors(
        self, magnetic: Tuple[torch.Tensor, torch.Tensor],
        num_nodes: torch.Tensor, device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The bilinear factorization: ``(Q_struct, K_struct)`` with

            b^(h)(i,j) = <Q_struct[b,h,i,:], K_struct[b,j,:]>

        shaped ``(B, H, N, 2M)`` and ``(B, N, 2M)``. Every learned parameter lands
        on the query side, so ``K_struct`` carries no head dimension — it is a
        universal structural dictionary that broadcasts across GQA groups.

        Not used by ``forward`` (which stays dense so Phase 2 measures math, not
        an implementation). It exists so the equivalence can be *tested* now, in
        fp64, before any backbone is built on top of it — see LINEAR_BIAS.md §7.

        NOTE: with the default ``bias_self_node=False`` this reproduces the bias
        off the diagonal only — ``_finalize`` zeroes b_ii and an inner product
        cannot express that zeroing (§7.3). With ``bias_self_node=True`` the
        equivalence is exact on the FULL matrix, which is the configuration the
        deferred factorized backbone would actually run.
        """
        V_real, V_imag, phi = self._phi(magnetic, num_nodes, device)
        W = self.proj[0].weight                                   # (H, 2m)
        m = W.shape[1] // 2
        # Psi = Phi @ W_R / W_I  -> (B, M, H): the eigenvalue projection, O(M).
        psi_R = phi @ W[:, :m].T
        psi_I = phi @ W[:, m:].T

        # Q = [ V_R*psi_R + V_I*psi_I ‖ V_I*psi_R - V_R*psi_I ], K = [ V_R ‖ V_I ]
        vr, vi = V_real.unsqueeze(1), V_imag.unsqueeze(1)         # (B,1,N,M)
        pr, pi = psi_R.transpose(1, 2).unsqueeze(2), psi_I.transpose(1, 2).unsqueeze(2)
        q_struct = torch.cat([vr * pr + vi * pi, vi * pr - vr * pi], dim=-1)
        k_struct = torch.cat([V_real, V_imag], dim=-1)            # (B, N, 2M)
        return q_struct, k_struct


class MagneticSharedBias(MagneticBias):
    """MagneticBias computed ONCE per forward and shared by every layer.

    Identical math and parameters to MagneticBias; the difference is placement:
    ``shared = True`` keeps it out of the per-layer GraphAttentionBias modules,
    and the causal-LM mixin instantiates a single copy on the top-level model,
    runs it once per forward (outside the gradient-checkpointed decoder layers,
    so the O(N²·M·m) einsums execute once instead of once per layer per
    recompute), and threads the resulting (B, H, N, N) tensor to every layer via
    ``GraphContext.shared_node_bias``.
    """

    config_key = 'magnetic_shared'
    shared = True


# ── Registration list ─────────────────────────────────────────────────────────

BIAS_TYPES: list[type[BaseBias]] = [
    SPDBias,
    LaplacianBias,
    RWSEBias,
    RRWPBias,
    MagneticBias,
    MagneticSharedBias,
    MagneticContentBias,
    LinearMagneticBias,
]
"""Add new bias types here — GraphAttentionBias picks them up automatically."""


# ── Top-level module ──────────────────────────────────────────────────────────

class GraphAttentionBias(nn.Module):
    """
    Per-layer, model-agnostic graph attention bias module.

    Instantiates every enabled bias type from BIAS_TYPES, accumulates their
    (B, H, N, N) outputs, then optionally applies a hard K-hop gate.
    Returns None when the result would be all-zero with no gate applied.
    """

    def __init__(
        self,
        num_heads:   int,
        head_dim:    int,
        layer_idx:   int,
        bias_config,
        k_hop:       int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.k_hop     = k_hop

        # Shared types are instantiated once on the top-level model (see
        # build_shared_bias_modules), never per layer.
        active_types = [cls for cls in BIAS_TYPES
                        if cls.is_enabled(bias_config) and not cls.shared]
        self.bias_modules = nn.ModuleList(
            [cls(num_heads, head_dim, bias_config) for cls in active_types]
        )
        self._active = {cls.config_key for cls in active_types}

    # ── Convenience flags (used by the attention layer to skip fetching data) ──

    @property
    def has_soft_bias(self) -> bool:
        return len(self.bias_modules) > 0

    @property
    def require_spd(self) -> bool:       return 'spd'       in self._active

    @property
    def require_laplacian(self) -> bool: return 'laplacian' in self._active

    @property
    def require_rwse(self) -> bool:      return 'rwse'      in self._active

    @property
    def require_rrwp(self) -> bool:      return 'rrwp'      in self._active

    @property
    def require_magnetic(self) -> bool:  return 'magnetic'  in self._active

    @property
    def require_magnetic_content(self) -> bool: return 'magnetic_content' in self._active

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        dtype:       torch.dtype,
        device:      torch.device,
        num_nodes:   Optional[torch.Tensor]                           = None,
        spd:         Optional[torch.Tensor]                           = None,
        laplacian:   Optional[torch.Tensor]                           = None,
        rwse:        Optional[torch.Tensor]                           = None,
        rrwp:        Optional[torch.Tensor]                           = None,
        magnetic:    Optional[Tuple[torch.Tensor, torch.Tensor]]      = None,
        hidden_states: Optional[torch.Tensor]                         = None,
        node_start_indices: Optional[torch.Tensor]                    = None,
        k_hop_mask:  Optional[torch.Tensor]                           = None,
        cache_dict:  Optional[dict]                                   = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute the (B, H, N, N) node-level attention bias.

        Returns None when the result would be all-zero with no gate applied
        (no enabled soft biases and K=0 or no mask provided).
        """
        if not self.has_soft_bias and (self.k_hop == 0 or k_hop_mask is None):
            return None

        cache_key = f'cached_graph_bias_{self.layer_idx}'
        if not self.training and cache_dict is not None and cache_key in cache_dict:
            return cache_dict[cache_key]

        node_bias = None
        for module in self.bias_modules:
            b = module(
                dtype=dtype, device=device,
                num_nodes=num_nodes, spd=spd, laplacian=laplacian,
                rwse=rwse, rrwp=rrwp, magnetic=magnetic,
                hidden_states=hidden_states, node_start_indices=node_start_indices,
            )
            if b is not None:
                node_bias = b if node_bias is None else node_bias + b

        node_bias = self._apply_k_hop_gate(node_bias, k_hop_mask, dtype, device)

        # Cache only in eval (the read above is eval-only too): it exists for
        # autoregressive decode. Writing during training would just pin every
        # layer's (B,H,N,N) output in memory for the rest of the step.
        if node_bias is not None and cache_dict is not None and not self.training:
            cache_dict[cache_key] = node_bias

        return node_bias

    # ── K-hop gate ────────────────────────────────────────────────────────────

    def _apply_k_hop_gate(
        self,
        node_bias:   Optional[torch.Tensor],
        k_hop_mask:  Optional[torch.Tensor],
        dtype:       torch.dtype,
        device:      torch.device,
    ) -> Optional[torch.Tensor]:
        """Apply -inf to positions outside the K-hop neighbourhood."""
        if self.k_hop == 0 or k_hop_mask is None:
            return node_bias

        gate = k_hop_mask.unsqueeze(1)                             # (B, 1, N, N)

        if node_bias is None:
            B, N, _ = k_hop_mask.shape
            node_bias = torch.zeros(B, self.num_heads, N, N, dtype=dtype, device=device)

        return node_bias.masked_fill(~gate, torch.finfo(dtype).min)


# ── Shared (once-per-forward) bias modules ────────────────────────────────────

def build_shared_bias_modules(num_heads: int, head_dim: int, bias_config) -> Optional[nn.ModuleList]:
    """Instantiate every enabled ``shared = True`` bias type (or None).

    The causal-LM mixin owns the returned ModuleList (named so its parameters
    match the ``graph_bias`` active-params substring), runs each module once per
    forward, and shares the summed (B, H, N, N) tensor across all layers.
    """
    shared_types = [cls for cls in BIAS_TYPES if cls.is_enabled(bias_config) and cls.shared]
    if not shared_types:
        return None
    return nn.ModuleList([cls(num_heads, head_dim, bias_config) for cls in shared_types])


def compute_shared_node_bias(
    modules: Optional[nn.ModuleList],
    *,
    dtype: torch.dtype,
    device: torch.device,
    num_nodes: Optional[torch.Tensor],
    features: dict,
) -> Optional[torch.Tensor]:
    """Sum the shared bias modules' (B, H, N, N) outputs (or None)."""
    if modules is None:
        return None
    node_bias = None
    for module in modules:
        b = module(
            dtype=dtype, device=device, num_nodes=num_nodes,
            spd=features.get("spd"), laplacian=features.get("laplacian"),
            rwse=features.get("rwse"), rrwp=features.get("rrwp"),
            magnetic=features.get("magnetic"),
        )
        if b is not None:
            node_bias = b if node_bias is None else node_bias + b
    return node_bias


# ── Layer-grouped bias (magnetic_groups) ──────────────────────────────────────

def layer_group_map(num_layers: int, num_groups: int) -> list[int]:
    """Layer index → group index, as evenly as possible.

    ``l * G // L`` keeps groups contiguous, uses every group when ``G <= L`` (so
    no module is left without a gradient, which DDP would reject), and sizes them
    within one layer of each other. ``G == L`` gives one group per layer; ``G == 1``
    gives a single group.
    """
    if not 1 <= num_groups <= num_layers:
        raise ValueError(f"num_groups={num_groups} must be in [1, {num_layers}].")
    return [l * num_groups // num_layers for l in range(num_layers)]


def build_group_bias_modules(
    num_heads: int, head_dim: int, bias_config, num_layers: int,
) -> Optional[nn.ModuleList]:
    """Instantiate ``magnetic_groups`` copies of :class:`MagneticBias` (or None).

    Owned by the causal-LM mixin under an attribute whose name contains
    ``graph_bias``, so the standard active-params substring unfreezes them.
    """
    num_groups = getattr(bias_config, "magnetic_groups", 0)
    if not num_groups:
        return None
    layer_group_map(num_layers, num_groups)          # validate G against L
    return nn.ModuleList(
        [MagneticBias(num_heads, head_dim, bias_config) for _ in range(num_groups)])


class GroupBiasCache:
    """One magnetic bias per layer *group*, computed once per group per pass.

    Placement is what makes this legal under HF's per-layer gradient
    checkpointing, whose contract is that a region's recompute must save the same
    tensors its forward did. Each group has one **owner** — its lowest layer:

    * the owner's region computes the bias *with grad*, in the forward and again
      automatically in its own recompute, so its saved-tensor frame matches;
    * every **follower** region only ever reads a value, so no bias intermediates
      enter its frame in either direction.

    Backward runs layers in reverse, so followers are reached before the owner.
    The first one rematerialises the value under ``no_grad`` — adding nothing to
    its own frame — and hands it on as a **leaf that requires grad**, so the ops
    after it save exactly what they saved in the forward. That leaf's ``.grad`` is
    discarded: gradient reaches the parameters through the graph node the owner
    built in the forward, which every consumer in the group is attached to.

    Each group's tensor is released once its last consumer has taken it, so peak
    residency is one ``(B, H, N, N)`` tensor rather than ``G`` of them. The
    owner's intermediates live inside its layer's checkpoint region and are
    therefore transient too.

    In eval / generation there is no backward, so the whole scheme collapses to
    "compute once per group and keep" — matching the shared-bias path, and
    reusing ``cache_dict`` across autoregressive decode steps.
    """

    def __init__(
        self,
        modules: nn.ModuleList,
        *,
        num_layers: int,
        dtype: torch.dtype,
        device: torch.device,
        num_nodes: Optional[torch.Tensor],
        features: dict,
        training: bool,
        cache_dict: Optional[dict] = None,
    ):
        self.modules = modules
        self.group_of = layer_group_map(num_layers, len(modules))
        self.size = [self.group_of.count(g) for g in range(len(modules))]
        self.owner_of = [self.group_of.index(g) for g in range(len(modules))]
        # Features are bound HERE, not read back from a mutable GraphContext: the
        # backward rematerialisation must see this batch's inputs even if another
        # forward has run in between (e.g. an adapters-off teacher pass).
        self.dtype, self.device = dtype, device
        self.num_nodes = num_nodes
        self.features = dict(features)
        self.training = training
        self.cache_dict = cache_dict
        self.live: list[Optional[torch.Tensor]] = [None] * len(modules)
        self.taken = [0] * len(modules)

    def _compute(self, g: int) -> torch.Tensor:
        return self.modules[g](
            dtype=self.dtype, device=self.device, num_nodes=self.num_nodes,
            spd=self.features.get("spd"), laplacian=self.features.get("laplacian"),
            rwse=self.features.get("rwse"), rrwp=self.features.get("rrwp"),
            magnetic=self.features.get("magnetic"),
        )

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        g = self.group_of[layer_idx]

        if not self.training:
            # No backward: compute once per group and hold, across decode steps.
            key = f"group_node_bias_{g}"
            if self.cache_dict is not None and key in self.cache_dict:
                return self.cache_dict[key]
            b = self._compute(g)
            if self.cache_dict is not None:
                self.cache_dict[key] = b
            return b

        if layer_idx == self.owner_of[g]:
            b = self._compute(g)                       # always, both directions
        else:
            b = self.live[g]
            if b is None:                              # backward: owner not yet reached
                with torch.no_grad():
                    value = self._compute(g)
                b = value.detach().requires_grad_(True)

        self.live[g] = b
        self.taken[g] += 1
        if self.taken[g] == self.size[g]:              # last consumer of this pass
            self.taken[g] = 0
            self.live[g] = None                        # drop our only strong ref
        return b