"""
Isolation tests for the Strategy-B registered attention functions.

The backbone-agnostic refactor registers `gtlm_eager` / `gtlm_sdpa` / `gtlm_flex`
into HF's `ALL_ATTENTION_FUNCTIONS`; the live model dispatches to them via
`config._attn_implementation` (Strategy B). The full v1↔v2 parity suite exercises
them end-to-end; these tests pin them down *in isolation* — the §3a spike turned
into a permanent regression test:

  1. the functions are actually registered, and
  2. driven through the registry with a `module._graph_ctx` installed, they read
     the per-batch context + per-layer `graph_bias` and reproduce, bit-for-bit,
     the Strategy-A `graph_attention_dispatch` path on identical q/k/v.
"""

import pytest
import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from src.models import GTLMLlamaConfig, GTLMLlamaAttention
from src.models.structural_mask import build_dense_structural_mask, expand_node_to_token_bias
from src.models.dispatch import graph_attention_dispatch
from src.models.context import GraphContext

DTYPE = torch.float32
DEVICE = torch.device("cpu")

# token → node layout: nodes 0,1,2 are prefix; node 3 is the (causal) prompt node.
NODE_IDS = torch.tensor([[0, 0, 1, 1, 2, 3]])     # (B=1, kv_len=6)
N_NODES = 4
PROMPT_NODE = torch.tensor([3])


def _config(**overrides):
    cfg = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=64,
        checkpoint_graph_bias=False,
    )
    cfg.update(overrides)
    return GTLMLlamaConfig(**cfg)


def _spd_feature():
    # symmetric shortest-path distances over the 4 nodes (path-graph-ish)
    spd = torch.tensor([[[0, 1, 2, 3],
                          [1, 0, 1, 2],
                          [2, 1, 0, 1],
                          [3, 2, 1, 0]]])         # (1, 4, 4) long
    return spd


def _make_module(with_bias):
    cfg = _config(spd=with_bias, max_spd=8, k_hop=0, graph_attn_impl="sdpa")
    attn = GTLMLlamaAttention(cfg, layer_idx=0).to(DTYPE).eval()
    if with_bias:
        # deterministic, clearly non-zero bias weights
        with torch.no_grad():
            for m in attn.graph_bias.bias_modules:
                m.weights.normal_(mean=0.0, std=0.5)
    return attn


def _qkv(module):
    torch.manual_seed(0)
    B = 1
    H = module.config.num_attention_heads
    Hkv = module.config.num_key_value_heads
    d = module.head_dim
    L = NODE_IDS.shape[1]
    q = torch.randn(B, H, L, d, dtype=DTYPE)
    k = torch.randn(B, Hkv, L, d, dtype=DTYPE)
    v = torch.randn(B, Hkv, L, d, dtype=DTYPE)
    return q, k, v


def _make_ctx(module, with_bias, cache):
    L = NODE_IDS.shape[1]
    structural_mask = build_dense_structural_mask(
        node_ids=NODE_IDS, prompt_node=PROMPT_NODE,
        pad_mask=torch.ones_like(NODE_IDS), k_hop_mask=None,
        k_hop=0, q_len=L, kv_len=L, dtype=DTYPE, device=DEVICE,
    )
    return GraphContext(
        node_ids=NODE_IDS,
        num_nodes=torch.tensor([N_NODES]),
        cache=cache,
        features={
            "spd": _spd_feature() if with_bias else None,
            "laplacian": None, "rwse": None, "rrwp": None, "magnetic": None,
        },
        structural_mask=structural_mask,
        block_mask=None,
        node_ids_flex=None,
    )


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_gtlm_fn_matches_strategy_a_dispatch(impl, with_bias):
    module = _make_module(with_bias)
    q, k, v = _qkv(module)
    L = NODE_IDS.shape[1]

    # Reference (Strategy A): build the bias + token expansion + dispatch by hand,
    # with its own fresh cache so it computes independently of the gtlm_* path.
    ref_ctx = _make_ctx(module, with_bias, cache={})
    node_bias = module.graph_bias(
        dtype=DTYPE, device=DEVICE, num_nodes=ref_ctx.num_nodes,
        spd=ref_ctx.features["spd"], laplacian=None, rwse=None, rrwp=None,
        magnetic=None, k_hop_mask=None, cache_dict=ref_ctx.cache,
    )
    assert (node_bias is not None) == with_bias, "bias presence should track config"
    token_soft_bias = (
        expand_node_to_token_bias(node_bias, NODE_IDS, L, L) if node_bias is not None else None
    )
    ref_out, _ = graph_attention_dispatch(
        impl, module, q, k, v, ref_ctx.structural_mask, token_soft_bias,
        scaling=module.scaling, dropout=0.0,
    )

    # Strategy B: install ctx on the module and dispatch through the HF registry.
    fn_name = {"eager": "gtlm_eager", "sdpa": "gtlm_sdpa"}[impl]
    assert fn_name in ALL_ATTENTION_FUNCTIONS, f"{fn_name} not registered"
    module._graph_ctx = _make_ctx(module, with_bias, cache={})
    out, _ = ALL_ATTENTION_FUNCTIONS[fn_name](
        module, q, k, v, None, scaling=module.scaling, dropout=0.0,
    )

    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


def test_gtlm_fn_requires_graph_ctx():
    module = _make_module(with_bias=True)
    module._graph_ctx = None
    q, k, v = _qkv(module)
    with pytest.raises(RuntimeError, match="graph context"):
        ALL_ATTENTION_FUNCTIONS["gtlm_sdpa"](module, q, k, v, None, scaling=module.scaling)
