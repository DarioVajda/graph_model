"""
Tests for GraphAttentionBias, the K-hop collator mask, and the
expand_node_to_token_bias utility.

Run with:  pytest tests/test_graph_bias.py -v
from the repo root (graph_model/).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn
import math

from src.models.bias import GraphAttentionBias, MagneticBias
from src.models.legacy.modeling_gtlm_llama_v1 import expand_node_to_token_bias
from src.utils.text_graph_collator import GraphCollator, _k_hop_reachability


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _BiasConfig:
    """Minimal stand-in for GraphAttnBiasConfig."""
    def __init__(self, **kwargs):
        self.spd       = kwargs.get('spd',       False)
        self.max_spd   = kwargs.get('max_spd',   16)
        self.laplacian = kwargs.get('laplacian', False)
        self.rwse      = kwargs.get('rwse',      False)
        self.rrwp      = kwargs.get('rrwp',      False)
        self.max_rw_steps = kwargs.get('max_rw_steps', 4)
        self.magnetic  = kwargs.get('magnetic',  False)
        self.magnetic_dim = kwargs.get('magnetic_dim', 8)
        self.magnetic_eigvec_dropout = kwargs.get('magnetic_eigvec_dropout', 0.0)
        self.magnetic_mlp_dropout    = kwargs.get('magnetic_mlp_dropout',    0.0)
        self.bias_droppath           = kwargs.get('bias_droppath',           0.0)
        self.bias_dropout            = kwargs.get('bias_dropout',            0.0)


def make_bias(k_hop=0, **cfg_kwargs) -> GraphAttentionBias:
    cfg = _BiasConfig(**cfg_kwargs)
    return GraphAttentionBias(num_heads=4, head_dim=8, layer_idx=0, bias_config=cfg, k_hop=k_hop)


NEG_INF = -float('inf')
DTYPE   = torch.float32
DEVICE  = torch.device('cpu')


def _is_neg_inf(t: torch.Tensor) -> torch.Tensor:
    """Element-wise check for -inf."""
    return t == torch.finfo(DTYPE).min


# ─── _k_hop_reachability ──────────────────────────────────────────────────────

class TestKHopReachability:

    def test_k1_chain(self):
        # 0-1-2-3 chain, K=1: only direct neighbours reachable
        A = torch.zeros(4, 4, dtype=torch.bool)
        for u, v in [(0,1),(1,2),(2,3)]:
            A[u,v] = A[v,u] = True
        R = _k_hop_reachability(A, K=1)
        assert R[0, 1] and R[1, 0]   # direct edge
        assert not R[0, 2]            # 2 hops away
        assert not R[0, 3]            # 3 hops away

    def test_k2_chain(self):
        A = torch.zeros(4, 4, dtype=torch.bool)
        for u, v in [(0,1),(1,2),(2,3)]:
            A[u,v] = A[v,u] = True
        R = _k_hop_reachability(A, K=2)
        assert R[0, 1] and R[0, 2]   # within 2 hops
        assert not R[0, 3]            # 3 hops away

    def test_k3_chain(self):
        A = torch.zeros(4, 4, dtype=torch.bool)
        for u, v in [(0,1),(1,2),(2,3)]:
            A[u,v] = A[v,u] = True
        R = _k_hop_reachability(A, K=3)
        assert R[0, 3]                # 3 hops — exactly K

    def test_k1_disconnected(self):
        # Two separate edges: 0-1 and 2-3.  K=1: cross-component unreachable.
        A = torch.zeros(4, 4, dtype=torch.bool)
        A[0,1] = A[1,0] = True
        A[2,3] = A[3,2] = True
        R = _k_hop_reachability(A, K=1)
        assert not R[0, 2] and not R[1, 3]

    def test_k_larger_than_diameter(self):
        # Triangle: diameter = 2.  K=10 should reach everything.
        A = torch.zeros(3, 3, dtype=torch.bool)
        for u, v in [(0,1),(1,2),(0,2)]:
            A[u,v] = A[v,u] = True
        R = _k_hop_reachability(A, K=10)
        assert R.all()                # all pairs reachable

    def test_symmetry_undirected(self):
        A = torch.zeros(5, 5, dtype=torch.bool)
        for u, v in [(0,1),(1,2),(2,3),(3,4)]:
            A[u,v] = A[v,u] = True
        R = _k_hop_reachability(A, K=2)
        assert (R == R.T).all(), "Undirected reachability must be symmetric"

    def test_isolated_node(self):
        # Node 2 has no edges.
        A = torch.zeros(3, 3, dtype=torch.bool)
        A[0,1] = A[1,0] = True
        R = _k_hop_reachability(A, K=5)
        assert not R[0, 2]
        assert not R[2, 0]
        assert not R[1, 2]


# ─── Collator K-hop mask ──────────────────────────────────────────────────────

class TestCollatorKHopMask:

    def _make_item(self, num_nodes, edges, prompt_node):
        return {
            'num_nodes':   num_nodes,
            'prompt_node': prompt_node,
            'edges':       edges,
            'input_ids':   [['tok'] for _ in range(num_nodes)],
        }

    def _collator(self, k_hop):
        return GraphCollator(k_hop=k_hop)

    # ── K=0 produces no mask key ──────────────────────────────────────────────

    def test_k0_no_mask_key(self):
        col  = self._collator(k_hop=0)
        item = self._make_item(3, [(0,1),(1,2)], prompt_node=2)
        batch_dict = col._build_k_hop_masks([item], max_num_nodes=3)
        # _build_k_hop_masks is only called when k_hop>0; here we test
        # that __call__ does NOT add the key when k_hop=0
        col0 = self._collator(k_hop=0)
        # The collator without k_hop adds no k_hop_mask key
        # (we verify by checking _build_k_hop_masks is not invoked):
        assert not hasattr(col0, '_k_hop_mask_called')   # just a meta check
        # Direct API check: call with a minimal mock batch
        mock_batch = [{
            'num_nodes': 3, 'prompt_node': 0,
            'edges': [(0,1)],
            'input_ids': [[1, 2], [3], [4, 5, 6]],   # integer token ids per node
            'laplacian_coordinates': torch.zeros(3, 2),
        }]
        result = GraphCollator(k_hop=0)(mock_batch)
        assert 'k_hop_mask' not in result, "k_hop=0 must not add k_hop_mask to the batch"

    # ── Basic mask shape and dtype ────────────────────────────────────────────

    def test_mask_shape_and_dtype(self):
        item  = self._make_item(4, [(0,1),(1,2),(2,3)], prompt_node=3)
        masks = self._collator(3)._build_k_hop_masks([item], max_num_nodes=4)
        assert masks.shape == (1, 4, 4)
        assert masks.dtype == torch.bool

    def test_padded_batch_shape(self):
        items = [
            self._make_item(3, [(0,1),(1,2)], prompt_node=2),
            self._make_item(5, [(0,1),(1,2),(2,3),(3,4)], prompt_node=4),
        ]
        masks = self._collator(2)._build_k_hop_masks(items, max_num_nodes=5)
        assert masks.shape == (2, 5, 5)

    # ── Diagonal always True ──────────────────────────────────────────────────

    def test_diagonal_always_true(self):
        item  = self._make_item(5, [(0,1),(1,2),(2,3),(3,4)], prompt_node=4)
        mask  = self._collator(1)._single_k_hop_mask(item['edges'], 5, 4)
        for i in range(5):
            assert mask[i, i], f"Self-attention must be allowed: mask[{i},{i}] is False"

    # ── Prompt node uses its own edges ────────────────────────────────────────

    def test_prompt_uses_own_edges_k1(self):
        # Star: prompt=4 connects to {0, 1, 2}; node 3 is NOT connected to prompt.
        edges = [(4,0),(4,1),(4,2)]
        item  = self._make_item(5, edges, prompt_node=4)
        mask  = self._collator(1)._single_k_hop_mask(edges, 5, 4)
        assert mask[4, 0] and mask[4, 1] and mask[4, 2]   # prompt→connected
        assert not mask[4, 3]                               # prompt→unconnected

    def test_prompt_does_not_bridge_prefix_nodes(self):
        # Chain: 0 -- prompt(1) -- 2.  Without the bridge, 0 and 2 are 2 hops apart
        # through node 1.  With the prompt removed as bridge, K=1 should NOT allow
        # 0 to attend to 2.
        edges = [(0,1),(1,2)]
        item  = self._make_item(3, edges, prompt_node=1)
        mask  = self._collator(1)._single_k_hop_mask(edges, 3, 1)
        assert not mask[0, 2], "Prefix nodes must not reach each other via the prompt bridge"
        assert not mask[2, 0]

    def test_prefix_reachability_through_non_prompt(self):
        # 0-1-2-3 (all prefix nodes, prompt=4 isolated).  K=2: 0 should reach 2.
        edges = [(0,1),(1,2),(2,3)]
        item  = self._make_item(5, edges, prompt_node=4)
        mask  = self._collator(2)._single_k_hop_mask(edges, 5, 4)
        assert mask[0, 2]    # 2-hop through non-prompt
        assert not mask[0, 3]  # 3-hop, outside K=2

    # ── Symmetry ──────────────────────────────────────────────────────────────

    def test_mask_symmetric_for_prefix_nodes(self):
        edges = [(0,1),(1,2),(2,3)]
        mask  = self._collator(2)._single_k_hop_mask(edges, 5, 4)
        # Check prefix-prefix block
        prefix = [0,1,2,3]
        for i in prefix:
            for j in prefix:
                assert mask[i,j] == mask[j,i], f"Symmetry violated at ({i},{j})"

    # ── K larger than diameter: full attention within component ───────────────

    def test_k_larger_than_diameter(self):
        # Triangle of prefix nodes 0,1,2; prompt=3
        edges = [(0,1),(1,2),(0,2),(3,0)]
        mask  = self._collator(100)._single_k_hop_mask(edges, 4, prompt_node=3)
        assert mask[0,1] and mask[1,2] and mask[0,2]   # all prefix pairs reachable

    # ── Disconnected components ───────────────────────────────────────────────

    def test_disconnected_components_masked(self):
        # Two components: {0,1} and {2,3}; prompt=4 isolated
        edges = [(0,1),(2,3)]
        mask  = self._collator(3)._single_k_hop_mask(edges, 5, 4)
        assert mask[0,1] and mask[2,3]
        assert not mask[0,2] and not mask[1,3]


# ─── GraphAttentionBias ───────────────────────────────────────────────────────

class TestGraphAttentionBiasNoBias:

    def test_no_soft_bias_k0_returns_none(self):
        gb     = make_bias(k_hop=0)
        result = gb(dtype=DTYPE, device=DEVICE)
        assert result is None

    def test_no_soft_bias_no_mask_returns_none(self):
        gb     = make_bias(k_hop=3)
        result = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=None)
        assert result is None

    def test_no_soft_bias_k1_with_mask(self):
        # No soft biases, but K-hop gate present:
        # within-K-hop → 0.0, outside → -inf
        mask = torch.tensor([[[True, True, False],
                               [True, True, True],
                               [False, True, True]]])  # (1, 3, 3)
        gb   = make_bias(k_hop=1)
        out  = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask)
        assert out is not None
        assert out.shape == (1, 4, 3, 3)
        # Within-hop positions are 0.0
        assert out[0, :, 0, 1].eq(0.0).all()
        # Outside-hop positions are -inf
        assert _is_neg_inf(out[0, :, 0, 2]).all()
        assert _is_neg_inf(out[0, :, 2, 0]).all()


class TestGraphAttentionBiasK0SoftBiases:
    """K=0: gate disabled, only soft biases applied."""

    def test_spd_k0_no_gate(self):
        # SPD only, K=0.  Should return non-None tensor, no -inf values.
        B, N = 1, 4
        spd = torch.tensor([[[0,1,2,3],
                              [1,0,1,2],
                              [2,1,0,1],
                              [3,2,1,0]]])  # (1,4,4)
        gb  = make_bias(k_hop=0, spd=True, max_spd=8)
        # Force weights to a known value for a deterministic check
        with torch.no_grad():
            gb.bias_modules[0].weights.fill_(0.0)
            gb.bias_modules[0].weights[0] = -1.0   # distance 1 → -1.0 per head
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd)
        assert out is not None
        assert not _is_neg_inf(out).any(), "K=0 must not produce -inf"
        # Distance-1 pairs should have bias -1.0
        assert torch.allclose(out[0, :, 0, 1], torch.full((4,), -1.0))
        # Self-pairs (spd=0) must have bias 0.0
        assert torch.allclose(out[0, :, 0, 0], torch.zeros(4))

    def test_rrwp_diagonal_zeroed(self):
        B, N, S = 1, 3, 4
        rrwp = torch.rand(B, N, N, S)
        gb   = make_bias(k_hop=0, rrwp=True, max_rw_steps=S)
        with torch.no_grad():
            gb.bias_modules[0].proj[2].weight.fill_(0.0)
            gb.bias_modules[0].proj[2].bias.fill_(1.0)   # constant non-zero output
        out = gb(dtype=DTYPE, device=DEVICE, rrwp=rrwp)
        for i in range(N):
            assert torch.allclose(out[0, :, i, i], torch.zeros(4)), \
                f"RRWP diagonal at [{i},{i}] must be zero"

    def test_laplacian_bias_symmetric(self):
        B, N, D = 1, 4, 3
        coords = torch.randn(B, N, D)
        gb     = make_bias(k_hop=0, laplacian=True)
        out    = gb(dtype=DTYPE, device=DEVICE, laplacian=coords)
        # cdist is symmetric, so bias should be symmetric (modulo head sign)
        assert torch.allclose(out, out.transpose(-1, -2)), \
            "Laplacian bias must be symmetric in (N, N)"


class TestGraphAttentionBiasKHopWithSoftBias:
    """K>0: gate AND soft biases active together."""

    def _spd_tensor(self):
        # 4-node path graph: spd[i][j] = |i - j|, 0 on diagonal
        spd = torch.zeros(1, 4, 4, dtype=torch.long)
        for i in range(4):
            for j in range(4):
                spd[0, i, j] = abs(i - j)
        return spd

    def _k2_mask(self):
        # K=2 on 4-node path: allow if |i-j| <= 2
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        for i in range(4):
            for j in range(4):
                if abs(i - j) <= 2:
                    mask[0, i, j] = True
        return mask

    def test_soft_bias_inside_khop_preserved(self):
        spd  = self._spd_tensor()
        mask = self._k2_mask()
        gb   = make_bias(k_hop=2, spd=True, max_spd=8)
        with torch.no_grad():
            gb.bias_modules[0].weights.fill_(0.0)
            gb.bias_modules[0].weights[0] = -1.0   # distance 1 → -1.0
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd, k_hop_mask=mask)
        # (0,1) is within K=2 and distance 1 → bias = -1.0
        assert torch.allclose(out[0, :, 0, 1], torch.full((4,), -1.0)), \
            "Soft bias value must be preserved inside K-hop region"

    def test_k_hop_gate_applied_outside(self):
        spd  = self._spd_tensor()
        mask = self._k2_mask()
        gb   = make_bias(k_hop=2, spd=True, max_spd=8)
        out  = gb(dtype=DTYPE, device=DEVICE, spd=spd, k_hop_mask=mask)
        # (0,3) is 3 hops away → masked out
        assert _is_neg_inf(out[0, :, 0, 3]).all(), \
            "Positions outside K-hop must be -inf even with soft bias"

    def test_diagonal_not_masked(self):
        mask = self._k2_mask()
        # Self-pairs appear in mask as True (set by collator); verify gate doesn't kill them
        gb  = make_bias(k_hop=2)
        out = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask)
        for i in range(4):
            assert not _is_neg_inf(out[0, :, i, i]).any(), \
                f"Self-attention must never be -inf (position [{i},{i}])"

    def test_k0_gate_disabled_with_soft_bias(self):
        spd  = self._spd_tensor()
        mask = self._k2_mask()
        gb   = make_bias(k_hop=0, spd=True, max_spd=8)
        out  = gb(dtype=DTYPE, device=DEVICE, spd=spd, k_hop_mask=mask)
        assert not _is_neg_inf(out).any(), \
            "K=0 must disable the gate even when a k_hop_mask is provided"

    def test_k_larger_than_diameter_no_neg_inf(self):
        # K=100 on a 4-node graph: no pair should be gated
        mask = torch.ones(1, 4, 4, dtype=torch.bool)   # all True = all within K
        gb   = make_bias(k_hop=100)
        out  = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask)
        assert not _is_neg_inf(out).any(), \
            "When all pairs are within K hops, no position should be -inf"


class TestGraphAttentionBiasCaching:

    def test_cache_hit_returns_same_tensor(self):
        mask  = torch.ones(1, 3, 3, dtype=torch.bool)
        gb    = make_bias(k_hop=1)
        cache = {}
        gb.eval()
        out1 = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask, cache_dict=cache)
        out2 = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask, cache_dict=cache)
        assert out1 is out2, "Second call must return the cached tensor object"

    def test_cache_not_read_during_training(self):
        # Pre-fill the cache with a sentinel value.  In training mode the
        # sentinel must NOT be returned — the bias must be recomputed fresh.
        sentinel = torch.full((1, 4, 3, 3), 999.0)
        cache    = {'cached_graph_bias_0': sentinel}
        mask     = torch.ones(1, 3, 3, dtype=torch.bool)
        gb       = make_bias(k_hop=1)
        gb.train()
        out = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask, cache_dict=cache)
        assert not (out == 999.0).any(), \
            "Training mode must recompute bias, not read the cached sentinel"

    def test_no_cache_dict_still_works(self):
        mask = torch.ones(1, 3, 3, dtype=torch.bool)
        gb   = make_bias(k_hop=1)
        gb.eval()
        out = gb(dtype=DTYPE, device=DEVICE, k_hop_mask=mask, cache_dict=None)
        assert out is not None


# ─── expand_node_to_token_bias ────────────────────────────────────────────────

class TestExpandNodeToTokenBias:

    def test_single_token_per_node(self):
        # 3 nodes, 1 token each, 2 heads, bias = node index product
        H, N = 2, 3
        node_bias = torch.zeros(1, H, N, N)
        for i in range(N):
            for j in range(N):
                node_bias[0, :, i, j] = float(i * N + j)

        node_ids = torch.tensor([[0, 1, 2]])   # (B=1, seq_len=3)
        token_bias = expand_node_to_token_bias(node_bias, node_ids, q_len=3, kv_len=3)

        assert token_bias.shape == (1, H, 3, 3)
        for i in range(N):
            for j in range(N):
                expected = float(i * N + j)
                assert token_bias[0, :, i, j].eq(expected).all(), \
                    f"Token bias at ({i},{j}) should be {expected}"

    def test_multiple_tokens_per_node(self):
        # 2 nodes: node 0 has 2 tokens, node 1 has 3 tokens.  1 head.
        H, N = 1, 2
        node_bias = torch.tensor([[[[10., 20.],
                                     [30., 40.]]]])   # (1, 1, 2, 2)
        node_ids = torch.tensor([[0, 0, 1, 1, 1]])   # seq_len=5
        token_bias = expand_node_to_token_bias(node_bias, node_ids, q_len=5, kv_len=5)
        assert token_bias.shape == (1, 1, 5, 5)
        # Token pair (0, 2): node 0 → node 1 → should be 20.0
        assert token_bias[0, 0, 0, 2].item() == 20.0
        # Token pair (2, 0): node 1 → node 0 → should be 30.0
        assert token_bias[0, 0, 2, 0].item() == 30.0
        # Token pair (0, 1): both in node 0 → 10.0
        assert token_bias[0, 0, 0, 1].item() == 10.0

    def test_kv_cache_slice(self):
        # q_len=1 (generating one token), kv_len=3 (full history)
        H, N = 2, 3
        node_bias = torch.arange(N * N).float().view(1, 1, N, N).expand(1, H, N, N).contiguous()
        node_ids  = torch.tensor([[0, 1, 2]])   # full history node ids
        token_bias = expand_node_to_token_bias(node_bias, node_ids, q_len=1, kv_len=3)
        assert token_bias.shape == (1, H, 1, 3)
        # Query is node 2 (last token), key is nodes [0, 1, 2]
        expected = torch.tensor([2*N+0, 2*N+1, 2*N+2], dtype=torch.float)
        assert torch.allclose(token_bias[0, 0, 0, :], expected)

    def test_batch_dimension(self):
        B, H, N = 2, 2, 3
        node_bias = torch.randn(B, H, N, N)
        node_ids  = torch.tensor([[0, 1, 2], [2, 1, 0]])
        token_bias = expand_node_to_token_bias(node_bias, node_ids, q_len=3, kv_len=3)
        assert token_bias.shape == (B, H, 3, 3)


# ─── Bias-path regularization knobs (TODO_reg, 2026-07-11) ───────────────────

class _FixedRand:
    """torch.rand stand-in returning canned tensors keyed by requested shape."""

    def __init__(self, by_shape):
        self.by_shape = {tuple(k): v for k, v in by_shape.items()}

    def __call__(self, *args, **kwargs):
        shape = (tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, torch.Size))
                 else tuple(args))
        return self.by_shape[shape].clone()


def _magnetic_module(p_eig=0.0, p_mlp=0.0, seed=0):
    torch.manual_seed(seed)
    cfg = _BiasConfig(magnetic=True, magnetic_dim=8,
                      magnetic_eigvec_dropout=p_eig, magnetic_mlp_dropout=p_mlp)
    m = MagneticBias(num_heads=4, head_dim=8, bias_config=cfg)
    # proj[2] is zero-init (the bias starts as a no-op); randomize it so the
    # output is non-trivial and dropout effects are observable.
    with torch.no_grad():
        nn.init.normal_(m.proj[2].weight, std=0.5)
        nn.init.normal_(m.proj[2].bias, std=0.5)
    return m


def _magnetic_inputs(B=2, N=5, seed=1):
    torch.manual_seed(seed)
    V = torch.randn(B, N, N, 2)
    lambdas = torch.randn(B, N)
    return (V, lambdas), torch.full((B,), N, dtype=torch.long)


class TestMagneticEigvecDropout:

    def test_eval_mode_is_noop(self):
        m = _magnetic_module(p_eig=0.5, p_mlp=0.5).eval()
        mag, nn_ = _magnetic_inputs()
        out = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        m.eigvec_dropout = 0.0
        m.mlp_dropout = 0.0
        ref = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        assert torch.equal(out, ref), "eval-mode dropout must be an exact no-op"

    def test_dropped_eigvec_equals_truncation(self, monkeypatch):
        # Dropping eigenvector j must equal physically removing it: same
        # 1/sqrt(1-p) scale on kept columns, same h_avg validity set.
        p, j, keep_idx = 0.25, 2, [0, 1, 3, 4]
        m = _magnetic_module(p_eig=p).train()
        (V, lam), nn_ = _magnetic_inputs(B=2, N=5)
        mask_full = torch.ones(2, 5)
        mask_full[:, j] = 0.0                       # rand < p  =>  dropped
        monkeypatch.setattr(torch, "rand", _FixedRand({
            (2, 5): mask_full, (2, 4): torch.ones(2, 4)}))

        out_masked = m(dtype=DTYPE, device=DEVICE, magnetic=(V, lam), num_nodes=nn_)
        out_trunc = m(dtype=DTYPE, device=DEVICE,
                      magnetic=(V[:, :, keep_idx, :], lam[:, keep_idx]), num_nodes=nn_)
        assert torch.allclose(out_masked, out_trunc, atol=1e-5), \
            "a dropped eigenvector must contribute exactly nothing"

    def test_rescaling_preserves_expectation(self):
        p = 0.1
        m = _magnetic_module(p_eig=p)
        mag, nn_ = _magnetic_inputs(B=1, N=6)
        m.eval()
        ref = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        m.train()
        torch.manual_seed(123)
        acc = torch.zeros_like(ref)
        n_draws = 1500
        for _ in range(n_draws):
            acc += m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        rel = (acc / n_draws - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 0.1, f"mean over draws deviates {rel:.3f} from the p=0 output"


class TestMagneticMLPDropout:

    def test_eval_noop_train_active(self):
        m = _magnetic_module(p_mlp=0.5)
        mag, nn_ = _magnetic_inputs()
        m.eval()
        out_eval = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        m.mlp_dropout = 0.0
        ref = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        assert torch.equal(out_eval, ref)
        m.mlp_dropout = 0.5
        m.train()
        torch.manual_seed(0)
        out_train = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        assert not torch.allclose(out_train, ref), "train-mode MLP dropout must act"

    def test_folded_legacy_parity_with_dropout(self):
        # Both branches consume identical RNG shapes, so folded and
        # legacy_unfolded stay in exact parity even with dropout active.
        m = _magnetic_module(p_eig=0.2, p_mlp=0.3).train()
        mag, nn_ = _magnetic_inputs()
        torch.manual_seed(7)
        out_folded = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        m.legacy_unfolded = True
        torch.manual_seed(7)
        out_legacy = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        assert torch.allclose(out_folded, out_legacy, atol=1e-5)


class TestBiasDropPath:

    def _gb(self, p, k_hop=0, seed=0):
        torch.manual_seed(seed)
        gb = make_bias(k_hop=k_hop, spd=True, max_spd=8, bias_droppath=p)
        with torch.no_grad():
            gb.bias_modules[0].weights.normal_(std=1.0)
        return gb

    def test_eval_mode_is_noop(self):
        gb = self._gb(0.5).eval()
        spd = torch.randint(0, 5, (2, 4, 4))
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd)
        gb.droppath = 0.0
        assert torch.equal(out, gb(dtype=DTYPE, device=DEVICE, spd=spd))

    def test_per_sample_all_or_nothing(self, monkeypatch):
        p = 0.5
        gb = self._gb(p).train()
        spd = torch.randint(0, 5, (4, 6, 6))
        gb_ref = self._gb(0.0).train()
        gb_ref.load_state_dict(gb.state_dict())
        ref = gb_ref(dtype=DTYPE, device=DEVICE, spd=spd)
        monkeypatch.setattr(torch, "rand", _FixedRand({
            (4, 1, 1, 1): torch.tensor([0.1, 0.9, 0.2, 0.8]).view(4, 1, 1, 1)}))
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd)
        for b in (0, 2):     # rand < p  =>  dropped: exactly zero
            assert torch.equal(out[b], torch.zeros_like(out[b])), f"sample {b}"
        for b in (1, 3):     # kept: exactly bias/(1-p)
            assert torch.allclose(out[b], ref[b] / (1 - p)), f"sample {b}"

    def test_k_hop_gate_unaffected(self, monkeypatch):
        p, B = 0.5, 2
        mask = torch.tensor([[True, True, False],
                             [True, True, True],
                             [False, True, True]]).unsqueeze(0).repeat(B, 1, 1)
        gb = self._gb(p, k_hop=1).train()
        spd = torch.randint(0, 5, (B, 3, 3))
        monkeypatch.setattr(torch, "rand", _FixedRand({
            (B, 1, 1, 1): torch.tensor([0.1, 0.9]).view(B, 1, 1, 1)}))
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd, k_hop_mask=mask)
        # Dropped sample: gate survives (-inf outside), soft bias gone (0 inside).
        assert _is_neg_inf(out[0, :, 0, 2]).all() and _is_neg_inf(out[0, :, 2, 0]).all()
        assert out[0][:, mask[0]].eq(0.0).all()
        # Kept sample: gate applied, soft bias present inside.
        assert _is_neg_inf(out[1, :, 0, 2]).all()
        assert not out[1][:, mask[1]].eq(0.0).all()


class TestMagneticSharedKeepMask:

    def test_provided_mask_equals_own_sampling(self, monkeypatch):
        # A pre-drawn magnetic_keep must act exactly like the module sampling
        # the same mask itself (the shared-mask path is just mask injection).
        p = 0.25
        m = _magnetic_module(p_eig=p).train()
        mag, nn_ = _magnetic_inputs(B=2, N=5)
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 2] = False
        out_injected = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_,
                         magnetic_keep=mask)
        monkeypatch.setattr(torch, "rand", _FixedRand({(2, 5): mask.float()}))
        out_sampled = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        assert torch.equal(out_injected, out_sampled)

    def test_mask_ignored_when_dropout_off(self):
        m = _magnetic_module(p_eig=0.0).train()
        mag, nn_ = _magnetic_inputs()
        mask = torch.zeros(2, 5, dtype=torch.bool)     # would kill everything
        out = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_,
                magnetic_keep=mask)
        ref = m(dtype=DTYPE, device=DEVICE, magnetic=mag, num_nodes=nn_)
        assert torch.equal(out, ref), "p=0 must ignore any provided mask"


class TestElementwiseBiasDropout:

    def _gb(self, p, seed=0):
        torch.manual_seed(seed)
        gb = make_bias(k_hop=0, spd=True, max_spd=8, bias_dropout=p)
        with torch.no_grad():
            gb.bias_modules[0].weights.normal_(std=1.0)
        return gb

    def test_eval_mode_is_noop(self):
        gb = self._gb(0.5).eval()
        spd = torch.randint(0, 5, (2, 6, 6))
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd)
        gb.bias_dropout = 0.0
        assert torch.equal(out, gb(dtype=DTYPE, device=DEVICE, spd=spd))

    def test_train_mode_element_semantics(self):
        p = 0.3
        gb = self._gb(p).train()
        spd = torch.randint(0, 5, (2, 8, 8))
        gb_ref = self._gb(0.0).train()
        gb_ref.load_state_dict(gb.state_dict())
        ref = gb_ref(dtype=DTYPE, device=DEVICE, spd=spd)
        torch.manual_seed(11)
        out = gb(dtype=DTYPE, device=DEVICE, spd=spd)
        # Survivors are exactly ref/(1-p); the rest are exactly zero.
        nz = out != 0.0
        assert torch.allclose(out[nz], (ref / (1 - p))[nz])
        # Dropped fraction over ref's nonzero entries is roughly p.
        base = ref != 0.0
        frac = 1.0 - (nz & base).sum().item() / base.sum().item()
        assert abs(frac - p) < 0.05, f"dropped fraction {frac:.3f} vs p={p}"


# ─── Collator backward-compatibility ─────────────────────────────────────────

class TestCollatorBackwardCompatibility:
    """GraphCollator(k_hop=0) must behave exactly as the original collator."""

    def _minimal_item(self, n=3):
        return {
            'num_nodes':             n,
            'prompt_node':           0,
            'edges':                 [(0,1),(1,2)],
            'input_ids':             [[1, 2] for _ in range(n)],   # integer token ids
            'laplacian_coordinates': torch.zeros(n, 2),
        }

    def test_output_keys_unchanged(self):
        original_keys = {
            'num_nodes', 'prompt_node', 'input_ids', 'labels',
            'laplacian_coordinates', 'shortest_path_dists',
            'rwse', 'rrwp', 'magnetic',
        }
        col    = GraphCollator(k_hop=0)
        result = col([self._minimal_item()])
        assert set(result.keys()) == original_keys

    def test_k_hop_adds_single_key(self):
        col    = GraphCollator(k_hop=1)
        result = col([self._minimal_item()])
        assert 'k_hop_mask' in result
        # All other keys must still be present
        for key in ('num_nodes', 'prompt_node', 'shortest_path_dists'):
            assert key in result

    def test_tokenizer_kwarg_still_accepted(self):
        # Existing code may pass tokenizer= as a positional or keyword arg
        col = GraphCollator(tokenizer=None, k_hop=0)
        assert col.tokenizer is None
        assert col.k_hop == 0
