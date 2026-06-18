"""
Backward-compatibility tests: KHopGraphLlamaForCausalLM(k=0) must produce
numerically identical logits to GraphLlamaForCausalLM when given the same
inputs and weights.

The plugin refactor moved graph-bias parameters from direct attributes on
LlamaAttentionWithBias (e.g. self_attn.spd_weights) into a nested
GraphAttentionBias module (self_attn.graph_bias.bias_modules[i].weights).
These tests verify that the two models compute equivalent outputs after an
explicit weight transfer.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import networkx as nx
from src.utils.magnetic_lap import get_magnetic_laplacian_coords
from src.models.legacy.modeling_gtlm_llama_v0 import (
    GraphLlamaForCausalLM,
    GraphLlamaConfig,
    GraphAttnBiasConfig,
)
from src.models.legacy.modeling_gtlm_llama_v1 import KHopGraphLlamaForCausalLM, KHopLlamaConfig

DEVICE = torch.device('cpu')

# ── Shared tiny config ────────────────────────────────────────────────────────

_BASE_KWARGS = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=256,
    pad_token_id=0,
    _attn_implementation='eager',
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_old_model(bias_cfg: GraphAttnBiasConfig) -> GraphLlamaForCausalLM:
    cfg = GraphLlamaConfig(graph_attn_bias=bias_cfg.to_dict(), **_BASE_KWARGS)
    return GraphLlamaForCausalLM(cfg).eval()


def _make_new_model(bias_cfg: GraphAttnBiasConfig, k_hop: int = 0) -> KHopGraphLlamaForCausalLM:
    cfg = KHopLlamaConfig(graph_attn_bias=bias_cfg.to_dict(), k_hop=k_hop, **_BASE_KWARGS)
    return KHopGraphLlamaForCausalLM(cfg).eval()


def _make_batch(num_nodes: int = 3, tokens_per_node: int = 5, seed: int = 42) -> dict:
    """Minimal input_graph_batch for a single graph."""
    torch.manual_seed(seed)
    return {
        'input_ids': [[
            torch.randint(1, 256, (tokens_per_node,))
            for _ in range(num_nodes)
        ]],
        'prompt_node': torch.tensor([num_nodes - 1]),
        'num_nodes': torch.tensor([num_nodes]),
    }


def _transfer_bias_weights(old_model, new_model, bias_cfg: GraphAttnBiasConfig):
    """
    Copy graph-bias parameters from old_model into new_model.

    Old model stores them directly on self_attn (e.g. spd_weights).
    New model stores them in self_attn.graph_bias.bias_modules[i], ordered
    according to BIAS_TYPES = [SPDBias, LaplacianBias, RWSEBias, RRWPBias, MagneticBias].
    """
    for i in range(len(old_model.model.layers)):
        old_attn = old_model.model.layers[i].self_attn
        new_attn = new_model.model.layers[i].self_attn
        mod_idx = 0

        if bias_cfg.spd:
            new_attn.graph_bias.bias_modules[mod_idx].weights.data.copy_(
                old_attn.spd_weights.data)
            mod_idx += 1

        if bias_cfg.laplacian:
            new_attn.graph_bias.bias_modules[mod_idx].weights.data.copy_(
                old_attn.laplacian_weights.data)
            mod_idx += 1

        if bias_cfg.rwse:
            new_attn.graph_bias.bias_modules[mod_idx].weights.data.copy_(
                old_attn.rwse_weights.data)
            mod_idx += 1

        if bias_cfg.rrwp:
            new_mod = new_attn.graph_bias.bias_modules[mod_idx]
            for j in range(len(old_attn.rrwp_proj)):
                if hasattr(old_attn.rrwp_proj[j], 'weight'):
                    new_mod.proj[j].weight.data.copy_(old_attn.rrwp_proj[j].weight.data)
                    new_mod.proj[j].bias.data.copy_(old_attn.rrwp_proj[j].bias.data)
            mod_idx += 1

        if bias_cfg.magnetic:
            new_mod = new_attn.graph_bias.bias_modules[mod_idx]
            # lambda_lin: R -> R^head_dim
            new_mod.lambda_lin.weight.data.copy_(old_attn.magnetic_lambda_lin.weight.data)
            new_mod.lambda_lin.bias.data.copy_(old_attn.magnetic_lambda_lin.bias.data)
            # deep_set: R^(2*head_dim) -> R^magnetic_dim
            for j in range(len(old_attn.magnetic_lambda_deep_set)):
                if hasattr(old_attn.magnetic_lambda_deep_set[j], 'weight'):
                    new_mod.deep_set[j].weight.data.copy_(old_attn.magnetic_lambda_deep_set[j].weight.data)
                    new_mod.deep_set[j].bias.data.copy_(old_attn.magnetic_lambda_deep_set[j].bias.data)
            # proj: R^(2*magnetic_dim) -> R^num_heads
            for j in range(len(old_attn.magnetic_bias_proj)):
                if hasattr(old_attn.magnetic_bias_proj[j], 'weight'):
                    new_mod.proj[j].weight.data.copy_(old_attn.magnetic_bias_proj[j].weight.data)
                    new_mod.proj[j].bias.data.copy_(old_attn.magnetic_bias_proj[j].bias.data)
            mod_idx += 1


def _assert_logits_equal(out_old, out_new, atol=1e-5):
    diff = (out_old.logits - out_new.logits).abs().max().item()
    assert torch.allclose(out_old.logits, out_new.logits, atol=atol), (
        f"Logit mismatch: max |diff| = {diff:.3e}"
    )


# ── Test: no bias ─────────────────────────────────────────────────────────────

class TestNoBias:
    """With all bias types disabled, KHop(k=0) must equal GraphLlama."""

    def test_single_graph(self):
        torch.manual_seed(0)
        bias_cfg = GraphAttnBiasConfig()  # all False

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)

        # No bias params in either model → state dicts are structurally identical.
        new_model.load_state_dict(old_model.state_dict(), strict=True)

        batch = _make_batch(num_nodes=3, tokens_per_node=5)

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)

    def test_larger_graph(self):
        torch.manual_seed(1)
        bias_cfg = GraphAttnBiasConfig()

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)
        new_model.load_state_dict(old_model.state_dict(), strict=True)

        batch = _make_batch(num_nodes=7, tokens_per_node=3, seed=7)

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)


# ── Test: SPD bias ────────────────────────────────────────────────────────────

class TestSPDBias:
    """KHop(k=0) with SPD bias equals GraphLlama after weight transfer."""

    def _path_graph_spd(self, num_nodes: int) -> torch.Tensor:
        spd = torch.zeros(1, num_nodes, num_nodes, dtype=torch.long)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    spd[0, i, j] = abs(i - j)
        return spd

    def test_same_outputs(self):
        torch.manual_seed(0)
        bias_cfg = GraphAttnBiasConfig(spd=True, max_spd=8)

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)

        new_model.load_state_dict(old_model.state_dict(), strict=False)
        _transfer_bias_weights(old_model, new_model, bias_cfg)

        num_nodes = 4
        batch = _make_batch(num_nodes=num_nodes, tokens_per_node=4)
        batch['shortest_path_dists'] = self._path_graph_spd(num_nodes)

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)

    def test_large_distances_clamped(self):
        """Distances beyond max_spd are clamped — both models must clamp identically."""
        torch.manual_seed(2)
        bias_cfg = GraphAttnBiasConfig(spd=True, max_spd=4)

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)

        new_model.load_state_dict(old_model.state_dict(), strict=False)
        _transfer_bias_weights(old_model, new_model, bias_cfg)

        num_nodes = 5
        batch = _make_batch(num_nodes=num_nodes, tokens_per_node=3)
        batch['shortest_path_dists'] = self._path_graph_spd(num_nodes)  # max dist = 4

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)


# ── Test: RRWP bias ───────────────────────────────────────────────────────────

class TestRRWPBias:
    """KHop(k=0) with RRWP bias equals GraphLlama after weight transfer."""

    def test_same_outputs(self):
        torch.manual_seed(0)
        max_rw_steps = 4
        bias_cfg = GraphAttnBiasConfig(rrwp=True, max_rw_steps=max_rw_steps)

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)

        new_model.load_state_dict(old_model.state_dict(), strict=False)
        _transfer_bias_weights(old_model, new_model, bias_cfg)

        num_nodes = 3
        batch = _make_batch(num_nodes=num_nodes, tokens_per_node=4)
        torch.manual_seed(1)
        batch['rrwp'] = torch.rand(1, num_nodes, num_nodes, max_rw_steps)

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)


# ── Test: Magnetic Laplacian bias (m=0) ───────────────────────────────────────

class TestMagneticBias:
    """
    KHop(k=0) with Magnetic bias equals GraphLlama after weight transfer (m=0).

    This is the critical case: for full eigenvectors (m=0) the old model's
    h_avg divisor (num_nodes) equals the new model's divisor (valid.sum()),
    so outputs must be numerically identical.
    """

    def _magnetic_batch(self, graph: nx.DiGraph, m: int = 0) -> dict:
        """Build a batch containing magnetic eigenvectors computed from `graph`."""
        num_nodes = graph.number_of_nodes()
        V_list, lam_list = get_magnetic_laplacian_coords([graph], m=m)
        V_batch   = torch.tensor(V_list[0],   dtype=torch.float32).unsqueeze(0)
        lam_batch = torch.tensor(lam_list[0], dtype=torch.float32).unsqueeze(0)

        batch = _make_batch(num_nodes=num_nodes, tokens_per_node=4)
        batch['magnetic'] = (V_batch, lam_batch)
        return batch

    def _reinit_proj(self, old_model, new_model, seed: int):
        """Replace zero-init final projection with random weights, mirrored to both models."""
        torch.manual_seed(seed)
        with torch.no_grad():
            for i in range(len(old_model.model.layers)):
                old_proj = old_model.model.layers[i].self_attn.magnetic_bias_proj[2]
                new_proj = new_model.model.layers[i].self_attn.graph_bias.bias_modules[0].proj[2]
                torch.nn.init.normal_(old_proj.weight)
                torch.nn.init.normal_(old_proj.bias)
                new_proj.weight.data.copy_(old_proj.weight.data)
                new_proj.bias.data.copy_(old_proj.bias.data)

    def test_same_outputs_path_graph(self):
        """Path graph with full eigenvectors (m=0)."""
        torch.manual_seed(0)
        bias_cfg = GraphAttnBiasConfig(magnetic=True, magnetic_dim=8)

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)

        new_model.load_state_dict(old_model.state_dict(), strict=False)
        _transfer_bias_weights(old_model, new_model, bias_cfg)
        self._reinit_proj(old_model, new_model, seed=42)

        G = nx.path_graph(4, create_using=nx.DiGraph)
        batch = self._magnetic_batch(G, m=0)

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)

    def test_same_outputs_complete_graph(self):
        """Complete directed graph with full eigenvectors (m=0)."""
        torch.manual_seed(2)
        bias_cfg = GraphAttnBiasConfig(magnetic=True, magnetic_dim=8)

        old_model = _make_old_model(bias_cfg)
        new_model = _make_new_model(bias_cfg, k_hop=0)

        new_model.load_state_dict(old_model.state_dict(), strict=False)
        _transfer_bias_weights(old_model, new_model, bias_cfg)
        self._reinit_proj(old_model, new_model, seed=99)

        G = nx.complete_graph(4, create_using=nx.DiGraph)
        batch = self._magnetic_batch(G, m=0)

        with torch.no_grad():
            out_old = old_model(input_graph_batch=batch)
            out_new = new_model(input_graph_batch=batch)

        _assert_logits_equal(out_old, out_new)
