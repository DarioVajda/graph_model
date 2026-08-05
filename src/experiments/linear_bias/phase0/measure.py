"""Phase 0 measurements: how much of the trained magnetic bias survives linearity.

Three quantities, all offline (LINEAR_BIAS.md §4):

* **P0a** — the best *linear* head, fitted to the trained MLP head's own output.
  This is ceiling (b), the bilinear-family restriction, measured at the true
  parameter budget rather than argued about.
* **P0b** — the rank spectrum of the trained bias, i.e. how much of it lives in
  the top-``2M`` subspace the factorization can reach. This is ceiling (a).
* **P0c** — the same comparison after the softmax (see ``attention.py``), because
  a bias residual only matters insofar as it moves attention.

**On sampling.** P0a fits ``2*magnetic_dim + 1`` coefficients per head. Using all
``N^2`` pairs would cost ``N^2 * M * magnetic_dim`` flops per graph to estimate
257 parameters; instead pairs are sampled and *exact* normal equations are
accumulated over them (never a giant design matrix). The reported R^2 is
therefore an estimate, and ``n_pairs`` is recorded so its precision is auditable.
P0b, which needs whole matrices, runs on fewer graphs instead.

**What the M-sweep does and does not say.** Truncating to ``M`` and re-running the
trained MLP measures how well the linear family tracks the MLP family *in that
feature space*. It is not the accuracy of a model trained at that ``M`` — those
weights were trained at ``m=128``. Absolute quality per ``M`` is Phase 2's job;
this is the cheap predictor that prunes Phase 2's grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ── The trained bias, recomputed from loaded weights ──────────────────────────

def compute_phi(w: dict[str, torch.Tensor], lambdas: torch.Tensor,
                num_nodes: torch.Tensor) -> torch.Tensor:
    """Replicate ``MagneticBias._phi`` (bias.py:164) from a loaded state dict.

    Reimplemented rather than instantiated so Phase 0 can run without building a
    backbone. The ``n_valid`` normalisation is over *valid eigenvalues*, which is
    what makes truncation to ``M`` behave like a dataset built at that ``M``.
    """
    M = lambdas.shape[1]
    h_i = F.linear(lambdas.unsqueeze(-1), w["lambda_lin.weight"], w["lambda_lin.bias"])
    # num_nodes is kept host-side by MagneticBatch.to (it is read as ints in the
    # inner loop); bring it over for this one comparison.
    valid = (torch.arange(M, device=lambdas.device).unsqueeze(0)
             < num_nodes.to(lambdas.device).unsqueeze(1))          # (B, M)
    n_valid = valid.sum(1, keepdim=True).unsqueeze(-1).to(h_i.dtype).clamp(min=1)
    h_avg = (h_i * valid.unsqueeze(-1)).sum(1, keepdim=True) / n_valid
    z = torch.cat([h_i, h_avg.expand_as(h_i)], dim=-1)
    return F.silu(F.linear(z, w["deep_set.0.weight"], w["deep_set.0.bias"]))


def spectral_pairs(V_real, V_imag, phi, rows, cols) -> torch.Tensor:
    """``[Re K ‖ Im K]`` for the listed node pairs — the MLP head's actual input.

    ``K(i,j) = sum_l V[i,l] conj(V[j,l]) phi[l]`` (bias.py:205-208). Contracting
    over ``l`` *before* multiplying by ``phi`` keeps this ``O(P·M·d_mag)`` instead
    of the ``O(N^2·M·d_mag)`` the dense path pays.
    """
    vr_i, vi_i = V_real[rows], V_imag[rows]                        # (P, M)
    vr_j, vi_j = V_real[cols], V_imag[cols]
    g_real = vr_i * vr_j + vi_i * vi_j                             # (P, M)
    g_imag = vi_i * vr_j - vr_i * vi_j
    return torch.cat([g_real @ phi, g_imag @ phi], dim=-1)         # (P, 2*d_mag)


def mlp_head(w: dict[str, torch.Tensor], feats: torch.Tensor) -> torch.Tensor:
    """The trained head: ``proj[2](SiLU(proj[0](feats)))`` -> (P, H)."""
    h = F.linear(feats, w["proj.0.weight"], w["proj.0.bias"])
    return F.linear(F.silu(h), w["proj.2.weight"], w["proj.2.bias"])


# ── P0a: streaming least squares ──────────────────────────────────────────────

@dataclass
class LinearFit:
    """Exact normal-equation accumulator for ``feats @ W + b ~ target``.

    Streaming: only ``(D+1)^2`` and ``(D+1)*H`` state is ever held, so batch count
    and pair count are unbounded by memory. The intercept is carried as a constant
    column so it is fitted, not assumed zero.
    """
    n_features: int
    n_heads: int
    device: torch.device = torch.device("cpu")
    gram: torch.Tensor = field(init=False)
    cross: torch.Tensor = field(init=False)
    tgt_sq: torch.Tensor = field(init=False)
    tgt_sum: torch.Tensor = field(init=False)
    count: int = 0

    def __post_init__(self):
        d = self.n_features + 1
        kw = dict(dtype=torch.float64, device=self.device)
        self.gram = torch.zeros(d, d, **kw)
        self.cross = torch.zeros(d, self.n_heads, **kw)
        self.tgt_sq = torch.zeros(self.n_heads, **kw)
        self.tgt_sum = torch.zeros(self.n_heads, **kw)

    def update(self, feats: torch.Tensor, target: torch.Tensor) -> None:
        ones = torch.ones(feats.shape[0], 1, dtype=feats.dtype, device=feats.device)
        x = torch.cat([feats, ones], 1)
        # Accumulate in float64 regardless of the compute dtype: R^2 near 1 is a
        # small difference of large moments, which is exactly where fp32
        # accumulation loses the digits the answer lives in.
        x, y = x.double(), target.double()
        self.gram += x.T @ x
        self.cross += x.T @ y
        self.tgt_sq += (y * y).sum(0)
        self.tgt_sum += y.sum(0)
        self.count += x.shape[0]

    def solve(self, rcond: float = 1e-12) -> torch.Tensor:
        """Return the fitted ``W`` ``(D+1, H)``.

        Uses a PSEUDO-inverse, not a plain solve. The spectral features are
        genuinely rank-deficient on some graph families — context's stars and
        chains have highly degenerate magnetic spectra, so whole groups of
        ``[Re K ‖ Im K]`` columns become collinear and the Gram matrix is
        singular. A plain solve there returns an enormous, arbitrary ``W`` that
        happens to satisfy the normal equations; the pseudo-inverse returns the
        minimum-norm solution instead, which is the one that means something.
        """
        if self.count == 0:
            raise RuntimeError("LinearFit.solve() before any update()")
        return torch.linalg.pinv(self.gram, rtol=rcond) @ self.cross

    def condition_number(self) -> float:
        """Gram condition number — the diagnostic that tells you whether a bad
        fit is the model's fault or the geometry's."""
        sv = torch.linalg.svdvals(self.gram)
        return float(sv[0] / sv[-1].clamp(min=1e-300))


@dataclass
class FitScore:
    """Second-pass accumulator for R^2, measured against the data directly.

    Deliberately NOT computed from ``LinearFit``'s moments. R^2 near 1 is a small
    difference of large numbers there (``tgt_sq - 2·lin + quad``), and when ``W``
    is large — which is exactly what a rank-deficient Gram produces — that
    subtraction loses every significant digit and reports nonsense (observed:
    R^2 = -7e16 on context). Recomputing residuals from the data costs one extra
    pass over batches already in memory and cannot cancel.
    """
    n_heads: int
    device: torch.device = torch.device("cpu")
    ss_res: torch.Tensor = field(init=False)
    tgt_sq: torch.Tensor = field(init=False)
    tgt_sum: torch.Tensor = field(init=False)
    count: int = 0

    def __post_init__(self):
        kw = dict(dtype=torch.float64, device=self.device)
        self.ss_res = torch.zeros(self.n_heads, **kw)
        self.tgt_sq = torch.zeros(self.n_heads, **kw)
        self.tgt_sum = torch.zeros(self.n_heads, **kw)

    def update(self, feats: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        ones = torch.ones(feats.shape[0], 1, dtype=feats.dtype, device=feats.device)
        x = torch.cat([feats, ones], 1).double()
        y = target.double()
        resid = y - x @ weights
        self.ss_res += (resid * resid).sum(0)
        self.tgt_sq += (y * y).sum(0)
        self.tgt_sum += y.sum(0)
        self.count += x.shape[0]

    def r2(self) -> torch.Tensor:
        ss_tot = self.tgt_sq - self.tgt_sum ** 2 / self.count
        deg = self.is_degenerate()
        r2 = 1.0 - self.ss_res / ss_tot.clamp(min=1e-300)
        # A constant target has zero variance, so R^2 is 0/0 — undefined, not bad.
        # It is a real case: some context layers never move their zero-initialised
        # head off zero, and returning -inf for them poisons any mean over layers.
        # Score such a head 1.0 when the fit reproduces the constant (the intercept
        # always can) and 0.0 otherwise; use is_degenerate() to exclude them from
        # aggregates rather than silently averaging them in.
        scale = (self.tgt_sq / self.count).clamp(min=1e-300)
        trivial = (self.ss_res <= 1e-12 * scale).to(r2.dtype)
        return torch.where(deg, trivial, r2)

    def is_degenerate(self) -> torch.Tensor:
        """Per-head mask: the trained bias is constant, so there is nothing to fit."""
        ss_tot = self.tgt_sq - self.tgt_sum ** 2 / self.count
        scale = (self.tgt_sq / self.count).clamp(min=1e-300)
        return ss_tot <= 1e-12 * scale

    def resid_frac_of_std(self) -> torch.Tensor:
        """RMS residual as a fraction of the target's own std — the units that
        make fit quality interpretable next to the bias's operating scale.
        Computed directly, so it stays meaningful even when R^2 goes negative."""
        var = (self.tgt_sq - self.tgt_sum ** 2 / self.count).clamp(min=0) / self.count
        rms = (self.ss_res / self.count).sqrt()
        return torch.where(self.is_degenerate(),
                           torch.zeros_like(rms),
                           rms / var.sqrt().clamp(min=1e-300))

    def target_std(self) -> torch.Tensor:
        var = (self.tgt_sq - self.tgt_sum ** 2 / self.count).clamp(min=0) / self.count
        return var.sqrt()


# ── P0b: rank spectrum ────────────────────────────────────────────────────────

def rank_spectrum(bias: torch.Tensor, cap: int) -> dict[str, float]:
    """Singular-value energy of one ``(N, N)`` bias matrix, and how much of it the
    rank-``cap`` factorization can reach.

    ``cap`` is ``2M``: the factorized form is a product of two ``N x 2M`` matrices,
    so ``2M`` is a hard ceiling on its rank regardless of how well ``W`` is fitted.
    """
    sv = torch.linalg.svdvals(bias.double())
    energy = (sv ** 2)
    total = energy.sum().clamp(min=1e-30)
    keep = min(cap, sv.numel())
    cum = energy.cumsum(0) / total
    return {
        "energy_in_cap": float(cum[keep - 1]),
        "rank_90": int((cum < 0.90).sum()) + 1,
        "rank_99": int((cum < 0.99).sum()) + 1,
        "n_singular": int(sv.numel()),
        "cap": keep,
    }


# ── P0c: attention-level effect ───────────────────────────────────────────────

def attention_kl(scores: torch.Tensor, bias_true: torch.Tensor,
                 bias_fit: torch.Tensor, mask: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
    """Mean KL( softmax(scores+true) || softmax(scores+fit) ) per row.

    The quantity that actually matters: an identical bias residual is harmless in
    a saturated attention row and decisive in a flat one, and only this sees the
    difference. ``scores`` are pre-softmax content logits *including* the model's
    1/sqrt(d) scaling.
    """
    p_logits, q_logits = scores + bias_true, scores + bias_fit
    if mask is not None:
        neg = torch.finfo(p_logits.dtype).min
        p_logits = p_logits.masked_fill(~mask, neg)
        q_logits = q_logits.masked_fill(~mask, neg)
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    if mask is not None:
        row_ok = mask.any(-1)
        return kl[row_ok]
    return kl.flatten()
