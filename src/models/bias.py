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
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)


class MagneticBias(BaseBias):
    """Complex-eigenvector-based directional encoding via a deep-set MLP."""

    config_key = 'magnetic'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        magnetic_dim = getattr(bias_config, 'magnetic_dim', 32)
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

    def forward(
        self, *, dtype, device,
        magnetic: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_nodes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if magnetic is None or num_nodes is None:
            return None
        V, lambdas = magnetic # (B,N,N,2), (B,N)
        V_real, V_imag = V[..., 0], V[..., 1] # (B, N, N) each

        h_i   = self.lambda_lin(lambdas.unsqueeze(-1))                                          # (B, M, head_dim)
        valid = (torch.arange(lambdas.shape[1], device=device).unsqueeze(0)
                 < num_nodes.unsqueeze(1))                                                       # (B, M) bool
        # Divide by the number of valid eigenvalues, not num_nodes.
        # With full eigenvectors (M=N) these are equal; with truncated (M<N) they differ.
        n_valid = valid.sum(dim=1, keepdim=True).unsqueeze(-1).to(h_i.dtype).clamp(min=1)       # (B, 1, 1)
        h_avg   = (h_i * valid.unsqueeze(-1)).sum(1, keepdim=True) / n_valid                   # (B, 1, head_dim)

        phi = self.deep_set(torch.cat([h_i, h_avg.expand_as(h_i)], dim=-1))                                 # (B, N, magnetic_dim)

        if getattr(self, "legacy_unfolded", False):
            # Original formulation, kept for parity testing: materializes the
            # (B,N,N,magnetic_dim) real/imag tensors AND their (…, 2m) cat
            # before the first projection.
            real = (torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phi) + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phi))
            imag = (torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phi) - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phi))
            b = self.proj(torch.cat([real, imag], dim=-1))
        else:
            # Folded formulation (algebraically identical): the first proj
            # layer is linear, so project phi (B,M,m — tiny) BEFORE the N²
            # einsums instead of their (B,N,N,2m) cat after. The first hidden
            # layer (B,N,N,m) is emitted directly; `real`/`imag` and the cat
            # never exist, halving the largest per-layer intermediates (and
            # the #7 recompute cost). Uses the same parameters — proj[0]'s
            # weight is just split into its real/imag column halves.
            W1, b1 = self.proj[0].weight, self.proj[0].bias       # (m, 2m), (m)
            m = W1.shape[0]
            phiR = phi @ W1[:, :m].T                              # (B, M, m)
            phiI = phi @ W1[:, m:].T                              # (B, M, m)
            hidden = (
                torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phiR)
                + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phiR)
                + torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phiI)
                - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phiI)
            ) + b1                                                # (B, N, N, m)
            b = self.proj[2](self.proj[1](hidden))                # SiLU, Linear

        b = b.permute(0, 3, 1, 2).contiguous()                    # (B, H, N, N)
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)


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