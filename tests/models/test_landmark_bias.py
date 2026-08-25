"""Correctness gate for `LandmarkBias` (see src/models/biases/LANDMARK_BIAS.md).

The properties whose failure is INVISIBLE in a training curve come first: a bias
that is silently absent, mis-indexed, or not permutation-invariant trains cleanly
and reads as "landmark did not help", which is the conclusion the sweep exists to
draw. Everything here guards that blast radius.
"""

import numpy as np
import pytest
import torch

from src.models.bias import GraphAttentionBias, LandmarkBias
from src.utils.landmark import landmark_coords, select_anchors

H, K, DMAX = 4, 6, 8
PAD, UNREACH = DMAX + 2, DMAX + 1


class Cfg:
    def __init__(self, **kw):
        self.landmark = True
        self.landmark_k = K
        self.landmark_k_collate = 0
        self.landmark_d_max = DMAX
        self.landmark_tau = 2.0
        self.landmark_channels = 3
        self.landmark_norm = True
        self.bias_self_node = True
        self.num_attention_heads = H
        self.head_dim = 8
        for k, v in kw.items():
            setattr(self, k, v)


def _mod(**kw):
    """fp64 by default. The normalization runs in the PARAMETER dtype (before the
    output cast), which is correct for training — normalizing in fp32 and casting
    to bf16 beats normalizing in bf16 — but it means an fp32 module only agrees
    with itself to ~1e-7 once the arithmetic ORDER changes. Tests that permute or
    rescale must therefore hold a double module, or they measure fp32 noise."""
    torch.manual_seed(0)
    return LandmarkBias(H, 8, Cfg(**kw)).double()


def _coords(B=2, N=5, k=K, seed=0):
    g = torch.Generator().manual_seed(seed)
    lm = torch.randint(0, DMAX + 2, (B, N, 3, k), generator=g)
    return lm


def _graph(n=9, p=0.3, seed=0, directed=True):
    rng = np.random.default_rng(seed)
    e = [(i, j) for i in range(n) for j in range(n)
         if i != j and rng.random() < p]
    return e


# ── 1. the bias is exactly zero at init, and NOT structurally dead ────────────

def test_zero_at_init_but_gradient_flows():
    m = _mod()
    lm = _coords()
    b = m(dtype=torch.float32, device=torch.device("cpu"), landmark=lm)
    assert b is not None, "landmark returned None with a feature present"
    assert torch.count_nonzero(b) == 0, "bias must be exactly 0 at step 0"

    # Normalized form: gain carries the zero, so it moves at step 1
    # (db/dgain = <q_hat, k_hat> != 0) and F/G at step 2. A dead saddle would have
    # BOTH zero — that is the failure this pins. Zeroing G instead would give
    # k_hat = normalize(0) = 0 and kill db/dgain too.
    b.sum().backward()
    assert m.gain.grad is not None and m.gain.grad.abs().sum() > 0, \
        "dead saddle: dgain is zero at init, so nothing ever leaves the origin"

    with torch.no_grad():
        m.gain += 0.1
    m.F.grad = None
    m(dtype=torch.float32, device=torch.device("cpu"), landmark=lm).sum().backward()
    assert m.F.grad.abs().sum() > 0, "F still frozen after gain moved"


def test_bias_is_nonzero_once_trained():
    """Guards the silent no-op: a wired-but-inert bias is the expensive failure."""
    m = _mod()
    with torch.no_grad():
        m.gain.normal_(0, 1)
    b = m(dtype=torch.float32, device=torch.device("cpu"), landmark=_coords())
    assert b.abs().max() > 1e-6


# ── 2. the factorization is exact (fp64) — the deferred backbone rests on it ──

def test_dense_equals_factorized_fp64():
    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
    lm = _coords(B=3, N=7)
    dense = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    q, k = m.structural_factors(lm, dtype=torch.float64)
    ref = torch.einsum('bhnc,bmc->bhnm', q, k)
    assert torch.allclose(dense, ref, atol=1e-12), \
        (dense - ref).abs().max().item()


def test_key_side_has_no_head_dim():
    """GQA-native: K_pos must broadcast across groups, i.e. carry no head axis."""
    m = _mod()
    q, k = m.structural_factors(_coords())
    assert q.dim() == 4 and q.shape[1] == H
    assert k.dim() == 3, f"K_pos must be (B,N,Ck), got {tuple(k.shape)}"


# ── 3. padding contributes exactly nothing ───────────────────────────────────

def test_pad_slots_are_inert():
    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
    lm = _coords(B=1, N=4, k=K)
    lm[..., 3:] = PAD                       # 3 real anchors, 3 pad slots
    b_pad = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm.clone())

    m2 = _mod(landmark_k=3)
    with torch.no_grad():
        m2.F.copy_(m.F); m2.G.copy_(m.G); m2.gain.copy_(m.gain)
    b_small = m2(dtype=torch.float64, device=torch.device("cpu"),
                 landmark=lm[..., :3].clone())
    assert torch.allclose(b_pad, b_small, atol=1e-12), \
        "PAD anchor slots changed the bias — k_val or the table zero row is wrong"


def test_k_val_is_per_batch_item():
    """A graph with fewer real anchors must divide by ITS count, not the batch max."""
    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
    lm = _coords(B=2, N=4)
    lm[..., 2:] = PAD                       # both items: 2 real anchors
    b_both = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm.clone())
    single = m(dtype=torch.float64, device=torch.device("cpu"),
               landmark=lm[:1].clone())
    assert torch.allclose(b_both[:1], single, atol=1e-12)


# ── 4. anchor-permutation invariance (the design's central claim) ────────────

def test_invariant_to_anchor_order():
    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
    lm = _coords(B=2, N=6)
    perm = torch.randperm(K)
    a = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm.clone())
    b = m(dtype=torch.float64, device=torch.device("cpu"),
          landmark=lm[..., perm].clone())
    assert torch.allclose(a, b, atol=1e-12), \
        "F/G are indexed by distance, so anchor order must not matter"


def _keys_are_distinct(edges, n):
    from scipy.sparse import csr_matrix
    from src.utils.landmark import _struct_key
    r, c = zip(*edges) if edges else ((), ())
    A = csr_matrix((np.ones(len(r)), (list(r), list(c))), shape=(n, n))
    keys, _ = _struct_key(A, n)
    return len(set(keys)) == n


@pytest.mark.parametrize("seed", [3, 5, 11, 17])
def test_node_permutation_equivariance_end_to_end(seed):
    """Relabel the GRAPH: anchors must follow, and the bias must permute with it.

    Run on graphs whose WL keys are all distinct, i.e. where the anchor rule is
    *well defined* — that is what this asserts. Nodes inside one automorphism
    orbit have genuinely equal keys and are covered by the test below.
    """
    n = 10
    e = _graph(n, 0.35, seed=seed)
    if not _keys_are_distinct(e, n):
        pytest.skip("graph has structural ties; covered by the orbit test")
    perm = np.random.default_rng(seed).permutation(n)
    inv = np.argsort(perm)
    e_p = [(int(inv[u]), int(inv[v])) for u, v in e]

    c0 = landmark_coords(e, n, k=K, d_max=DMAX)
    c1 = landmark_coords(e_p, n, k=K, d_max=DMAX)

    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
    f = lambda c: m(dtype=torch.float64, device=torch.device("cpu"),
                    landmark=torch.tensor(c.astype(np.int64)).unsqueeze(0))
    # inv[u] is the NEW label of original node u, so new node j is original
    # node perm[j] — the reindex is by perm, not by inv.
    b0, b1 = f(c0), f(c1)
    b0_p = b0[:, :, perm, :][:, :, :, perm]
    assert torch.allclose(b0_p, b1, atol=1e-12), \
        (b0_p - b1).abs().max().item()


def test_wl_refinement_separates_what_degree_cannot():
    """The tie-break must be finer than degree, or a KG's leaves are all equal."""
    from scipy.sparse import csr_matrix
    from src.utils.landmark import _struct_key
    # Two degree-1 leaves hanging off hubs of DIFFERENT degree: identical by
    # degree, separable by one round of refinement.
    e = [(0, 2), (1, 3), (3, 4), (3, 5)]
    n = 6
    r, c = zip(*e)
    A = csr_matrix((np.ones(len(r)), (list(r), list(c))), shape=(n, n))
    keys, deg = _struct_key(A, n)
    assert deg[0] == deg[1], "precondition: equal degree"
    assert keys[0] != keys[1], "WL failed to separate leaves under different hubs"


def test_automorphism_orbit_is_the_documented_residual():
    """Nodes in one orbit are genuinely interchangeable: the key ties, and the
    index decides. Recorded so the limitation is a known property, not a surprise."""
    from scipy.sparse import csr_matrix
    from src.utils.landmark import _struct_key
    e = [(0, 1), (0, 2), (0, 3)]                 # 1,2,3 are one orbit
    n = 4
    r, c = zip(*e)
    A = csr_matrix((np.ones(len(r)), (list(r), list(c))), shape=(n, n))
    keys, _ = _struct_key(A, n)
    assert keys[1] == keys[2] == keys[3]


# ── 5. the feature itself ────────────────────────────────────────────────────

def test_coords_semantics_on_a_path():
    """0->1->2->3: check both directed channels and the undirected one by hand."""
    e = [(0, 1), (1, 2), (2, 3)]
    c = landmark_coords(e, 4, k=4, d_max=DMAX)      # k=n, so anchors = all nodes
    a = select_anchors_for(e, 4, 4)
    j = list(a).index(3)                             # the anchor that is node 3
    assert c[0, 0, j] == 3, "D_out(0->3) must be 3"
    assert c[0, 1, j] == UNREACH, "D_in(3->0) must be unreachable on a path"
    assert c[0, 2, j] == 3, "undirected 0-3 must be 3"


def select_anchors_for(edges, n, k):
    from scipy.sparse import csr_matrix
    r, c = zip(*edges) if edges else ((), ())
    A = csr_matrix((np.ones(len(r)), (list(r), list(c))), shape=(n, n))
    return select_anchors(A, n, k)


def test_every_component_gets_an_anchor():
    """Component stratification: no node may be left with an all-UNREACH row."""
    e = [(0, 1), (1, 2), (2, 0)] + [(3, 4), (4, 3)]          # 2 comps + isolate
    n = 6                                                     # node 5 isolated
    for k in (3, 4, 6):
        c = landmark_coords(e, n, k=k, d_max=DMAX)
        dead = ((c[:, 0, :] >= UNREACH) & (c[:, 1, :] >= UNREACH)
                & (c[:, 2, :] >= UNREACH)).all(1)
        assert not dead.any(), f"k={k}: dead rows at {np.flatnonzero(dead)}"


def test_prefix_slice_keeps_components_covered():
    """The round-robin order is what makes landmark_k_collate a valid k-sweep:
    any prefix must still touch every component, so no node is orphaned into an
    all-UNREACH row by shrinking k.

    Only meaningful for k >= #components — with fewer anchors than components,
    coverage is impossible by counting, and `apportion` gives the k largest one
    each by design.
    """
    e = ([(i, i + 1) for i in range(20)]          # a 21-node path
         + [(21, 22), (22, 21)]                    # a 2-cycle
         + [(23, 24)])                             # a 2-node chain
    n = 25                                         # 3 components exactly
    full = landmark_coords(e, n, k=16, d_max=DMAX)
    for kk in (3, 4, 8, 16):
        sl = full[:, :, :kk]
        dead = ((sl[:, 0, :] >= UNREACH) & (sl[:, 1, :] >= UNREACH)
                & (sl[:, 2, :] >= UNREACH)).all(1)
        assert not dead.any(), f"prefix k={kk} orphaned a component"


def test_fewer_anchors_than_components_is_handled():
    """k < #components cannot cover everything; it must not crash or mis-shape."""
    e = [(2 * i, 2 * i + 1) for i in range(8)]     # 8 disjoint edges
    n = 16
    c = landmark_coords(e, n, k=3, d_max=DMAX)
    assert c.shape == (n, 3, 3)
    covered = ~((c[:, 0, :] >= UNREACH) & (c[:, 1, :] >= UNREACH)
                & (c[:, 2, :] >= UNREACH)).all(1)
    assert covered.sum() >= 3, "the k chosen components must at least be covered"


# ── 6. the 2-channel ablation is a strict restriction ────────────────────────

def test_two_channel_ablation_ignores_undirected_block():
    m3 = _mod(landmark_channels=3)
    m2 = _mod(landmark_channels=2)
    with torch.no_grad():
        for m in (m3, m2):
            m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
        m2.F.copy_(m3.F[:2]); m2.G.copy_(m3.G[:2]); m2.gain.copy_(m3.gain)
        m3.F[2].zero_(); m3.G[2].zero_()
    lm = _coords(B=2, N=5)
    a = m3(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    b = m2(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    assert torch.allclose(a, b, atol=1e-12)


# ── 7. wiring: the module is reachable through GraphAttentionBias ────────────

def test_reachable_through_graph_attention_bias():
    gb = GraphAttentionBias(num_heads=H, head_dim=8, layer_idx=0,
                            bias_config=Cfg(), k_hop=0)
    assert gb.require_landmark and gb.has_soft_bias
    with torch.no_grad():
        for mod in gb.bias_modules:
            mod.gain.normal_(0, 1)
    out = gb(dtype=torch.float32, device=torch.device("cpu"),
             num_nodes=torch.tensor([5, 5]), landmark=_coords())
    assert out is not None and out.abs().max() > 0, \
        "landmark is not wired into GraphAttentionBias"


def test_diagonal_mask_toggle():
    m_on = _mod(bias_self_node=True)
    m_off = _mod(bias_self_node=False)
    with torch.no_grad():
        for m in (m_on, m_off):
            m.gain.normal_(0, 1)
        m_off.F.copy_(m_on.F); m_off.G.copy_(m_on.G); m_off.gain.copy_(m_on.gain)
    lm = _coords(B=1, N=5)
    a = m_on(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    b = m_off(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    d = torch.arange(5)
    assert b[:, :, d, d].abs().max() == 0, "mask off must zero b_ii"
    assert a[:, :, d, d].abs().max() > 0, "self-node on must keep b_ii"


def test_forward_does_not_mutate_the_feature_tensor():
    """The batch belongs to the caller, and with `checkpoint_graph_bias` the bias
    forward is RECOMPUTED in backward — i.e. it runs twice on one tensor. An
    in-place clamp bumps that tensor's version counter and autograd rejects the
    second pass, which surfaces only under gradient checkpointing.
    """
    m = _mod()
    with torch.no_grad():
        m.gain.normal_(0, 1)
    lm = _coords(B=2, N=6)
    before = lm.clone()
    v0 = lm._version
    m(dtype=torch.float32, device=torch.device("cpu"), landmark=lm)
    assert torch.equal(lm, before), "forward modified the landmark feature values"
    assert lm._version == v0, "forward bumped the feature's version counter"


def test_two_forwards_then_backward():
    """The checkpoint-recompute pattern, reduced to its essentials."""
    m = _mod()
    lm = _coords(B=2, N=6)
    a = m(dtype=torch.float32, device=torch.device("cpu"), landmark=lm)
    b = m(dtype=torch.float32, device=torch.device("cpu"), landmark=lm)
    (a.pow(2).mean() - b.mean()).backward()      # must not raise
    assert m.gain.grad is not None and m.gain.grad.abs().sum() > 0


# ── 8. the magnitude bound that sweep 040 lacked ─────────────────────────────

def test_bias_magnitude_is_bounded_by_the_gain():
    """|b| <= C * max|gain| by Cauchy-Schwarz, whatever F and G do.

    Sweep 040 ran the unnormalized form and measured |b|max = 9-240 against
    attention logits of O(1-10); it scored BELOW the no-bias floor and got worse
    as bias_lr rose. The bound is the fix, so it is asserted, with the tables
    pushed to absurd values to prove the bound does not depend on them.
    """
    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 50); m.G.normal_(0, 50)      # absurd table scale
        m.gain.fill_(0.5)
    b = m(dtype=torch.float64, device=torch.device("cpu"), landmark=_coords(B=2, N=6))
    assert b.abs().max() <= 3 * 0.5 + 1e-9, f"bound violated: {b.abs().max()}"


def test_unnormalized_form_still_available():
    """`landmark_norm=False` reproduces 040 exactly, so those numbers stay checkable."""
    m = _mod(landmark_norm=False)
    assert m.gain is None
    with torch.no_grad():
        m.G.normal_(0, 1)
    b = m(dtype=torch.float64, device=torch.device("cpu"), landmark=_coords())
    assert b.abs().max() > 0


def test_normalized_form_is_scale_invariant_in_the_tables():
    """Degree-0 in table scale: doubling F must not change the bias at all.
    This is the property whose absence produced 040's runaway."""
    m = _mod()
    with torch.no_grad():
        m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.fill_(1.0)
    lm = _coords(B=2, N=6)
    a = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    with torch.no_grad():
        m.F *= 3.0; m.G *= 7.0
    c = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    assert torch.allclose(a, c, atol=1e-10), (a - c).abs().max().item()


# ── 6. landmark_gain_scale — the tandem's only magnitude knob ────────────────

def test_gain_scale_multiplies_the_bias_exactly():
    """`gain_scale` must be a clean linear scale on the bias, because that is the
    whole basis for using it as a stand-in for a per-arm bias_lr: |b| grows like
    gain_scale * lr * steps only if it enters linearly."""
    lm = _coords(B=2, N=6)
    ref = _mod()
    with torch.no_grad():
        ref.F.normal_(0, 1); ref.G.normal_(0, 1); ref.gain.normal_(0, 1)
    a = ref(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)

    scaled = _mod(landmark_gain_scale=0.25)
    with torch.no_grad():
        scaled.F.copy_(ref.F); scaled.G.copy_(ref.G); scaled.gain.copy_(ref.gain)
    c = scaled(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    assert torch.allclose(c, 0.25 * a, atol=1e-12), (c - 0.25 * a).abs().max().item()


def test_gain_scale_defaults_to_identity():
    """The default must be bit-for-bit the pre-knob model, or every earlier
    landmark number silently stops being comparable."""
    lm = _coords(B=2, N=6)
    a, b = _mod(), _mod(landmark_gain_scale=1.0)
    for m in (a, b):
        with torch.no_grad():
            torch.manual_seed(3)
            m.F.normal_(0, 1); m.G.normal_(0, 1); m.gain.normal_(0, 1)
    x = a(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    y = b(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    assert torch.equal(x, y)


def test_gain_scale_is_not_a_parameter():
    """It is a fixed constant, not something the optimizer can undo. If it were a
    parameter it would just be reabsorbed into `gain` and decouple nothing."""
    m = _mod(landmark_gain_scale=0.25)
    assert "gain_scale" not in dict(m.named_parameters())
    assert isinstance(m.gain_scale, float)


def test_gain_scale_still_zero_at_init_with_live_gradient():
    """Scaling the gain must not reintroduce the dead saddle: b is still 0 at
    step 0, and d b/d gain must still be non-zero."""
    m = _mod(landmark_gain_scale=0.05)
    lm = _coords(B=2, N=6)
    b = m(dtype=torch.float64, device=torch.device("cpu"), landmark=lm)
    assert b.abs().max() == 0.0
    b.sum().backward()
    assert m.gain.grad is not None and m.gain.grad.abs().max() > 0
