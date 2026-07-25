"""
GTLM on an ALiBi backbone (BLOOM).

The point of the adapter is that the graph machinery is not RoPE-specific, so the
tests pin the two claims that would make it false:

  1. **The backbone is unharmed.** With no graph bias and a single-node graph
     (whose structural mask reduces to plain causal), GTLM-BLOOM reproduces stock
     ``BloomForCausalLM`` logits exactly — i.e. the ALiBi rebuild is the pretrained
     bias, not an approximation of it.
  2. **The graph bias reaches the scores.** Turning a bias on changes the logits
     and its parameters receive gradients.

Plus the ALiBi-specific behaviour that motivated the adapter (positions come from
GTLM's per-node ``position_ids``, not from serialization order), decoding, and the
refusal of the unwired flex backend.
"""

import pytest
import torch

from transformers.models.bloom.modeling_bloom import BloomForCausalLM, build_alibi_tensor
from transformers.models.bloom.configuration_bloom import BloomConfig

from src.models import GTLMBloomConfig, GTLMBloomForCausalLM

DTYPE = torch.float32

TINY = dict(
    vocab_size=128, hidden_size=32, n_layer=2, n_head=4,
    hidden_dropout=0.0, attention_dropout=0.0,
)

# Two prefix nodes + a prompt node, packed the way GraphCollatorV2 packs them.
NODE_IDS = torch.tensor([[0, 0, 0, 1, 1, 2, 2]])          # (B=1, L=7)
POSITION_IDS = torch.tensor([[0, 1, 2, 0, 1, 0, 1]])      # per-node reset
PROMPT_NODE = torch.tensor([2])
NUM_NODES = torch.tensor([3])
SPD = torch.tensor([[[0, 1, 2],
                     [1, 0, 1],
                     [2, 1, 0]]])                          # (1, 3, 3)


def _gtlm(**bias):
    torch.manual_seed(0)
    cfg = GTLMBloomConfig(k_hop=0, graph_attn_impl="eager",
                          checkpoint_graph_bias=False, **bias, **TINY)
    return GTLMBloomForCausalLM(cfg).eval()


def _graph_batch(input_ids, **overrides):
    batch = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=POSITION_IDS,
        node_ids=NODE_IDS,
        prompt_node=PROMPT_NODE,
        num_nodes=NUM_NODES,
    )
    batch.update(overrides)
    return batch


# ── 1. The pretrained backbone is reproduced exactly ────────────────────────────

def test_matches_stock_bloom_on_a_single_node_graph():
    """No graph bias + one node (== the prompt node) => the structural mask is plain
    causal and the positions are ``arange``, so GTLM-BLOOM must be stock BLOOM."""
    torch.manual_seed(0)
    stock = BloomForCausalLM(BloomConfig(**TINY)).eval()
    gtlm = _gtlm()
    missing, unexpected = gtlm.load_state_dict(stock.state_dict(), strict=False)
    assert not unexpected, unexpected

    L = 7
    input_ids = torch.randint(0, TINY["vocab_size"], (1, L))
    flat = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(L).unsqueeze(0),
        node_ids=torch.zeros(1, L, dtype=torch.long),
        prompt_node=torch.tensor([0]),
        num_nodes=torch.tensor([1]),
    )

    with torch.no_grad():
        ours = gtlm(**flat).logits
        theirs = stock(input_ids=input_ids, attention_mask=flat["attention_mask"]).logits

    assert torch.allclose(ours, theirs, atol=1e-6, rtol=0), \
        f"max |diff| = {(ours - theirs).abs().max().item():.3e}"


# ── 2. The graph bias reaches the attention scores ──────────────────────────────

def test_graph_bias_changes_logits_and_receives_gradients():
    torch.manual_seed(0)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))

    plain = _gtlm()
    biased = _gtlm(spd=True, max_spd=8)
    biased.load_state_dict(plain.state_dict(), strict=False)

    bias_params = [(n, p) for n, p in biased.named_parameters() if "graph_bias" in n]
    assert bias_params, "no graph-bias parameters were built"
    with torch.no_grad():
        # SPDBias is zero-initialised (a run starts from the base model), so give the
        # distance table a signal before asking whether it reaches the scores.
        for i, (_, p) in enumerate(bias_params):
            p.copy_(torch.linspace(-1.0, 1.0, p.numel()).view_as(p) * (i + 1))
        p.requires_grad_(True)

    batch = _graph_batch(input_ids)
    with torch.no_grad():
        base = plain(**batch).logits
        # If the mask substitution had dropped the bias, these would match.
        with_bias = biased(**batch, shortest_path_dists=SPD).logits
    assert not torch.allclose(base, with_bias, atol=1e-6)

    for _, p in bias_params:
        p.requires_grad_(True)

    labels = input_ids.clone()
    biased.train()
    biased(**batch, shortest_path_dists=SPD, labels=labels).loss.backward()
    biased.eval()
    for name, p in bias_params:
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"
    assert any(p.grad.abs().sum() > 0 for _, p in bias_params)


# ── 3. ALiBi is built over GTLM's per-node positions ────────────────────────────

def test_alibi_follows_node_positions_not_sequence_order():
    model = _gtlm()
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    with torch.no_grad():
        model(**_graph_batch(input_ids))

    stack = model.transformer
    assert torch.equal(stack._alibi_positions, POSITION_IDS)

    alibi = stack.build_alibi_tensor(torch.ones(1, 7), TINY["n_head"], DTYPE)
    slopes = stack._slopes(TINY["n_head"], POSITION_IDS.device)
    expected = (slopes.view(1, TINY["n_head"], 1) * POSITION_IDS[:, None, :]).reshape(
        TINY["n_head"], 1, 7).to(DTYPE)
    assert torch.equal(alibi, expected)

    # ...and that is genuinely different from what stock BLOOM would have built
    # from raw sequence order (the whole reason the override exists).
    stock_alibi = build_alibi_tensor(torch.ones(1, 7), TINY["n_head"], DTYPE)
    assert not torch.allclose(alibi, stock_alibi)

    # The slopes themselves are the pretrained ones: a flat arange reproduces stock.
    stack._alibi_positions = torch.arange(7).unsqueeze(0)
    assert torch.equal(
        stack.build_alibi_tensor(torch.ones(1, 7), TINY["n_head"], DTYPE), stock_alibi)


def test_padding_does_not_shift_alibi_of_real_tokens():
    """Right padding carries position 0, so padded keys contribute a zero ALiBi term
    — the same neutrality stock BLOOM gets from its ``* attention_mask`` factor."""
    model = _gtlm()
    padded_positions = torch.cat([POSITION_IDS, torch.zeros(1, 3, dtype=torch.long)], 1)
    model.transformer._alibi_positions = padded_positions
    alibi = model.transformer.build_alibi_tensor(torch.ones(1, 10), TINY["n_head"], DTYPE)
    assert torch.equal(alibi[:, 0, 7:], torch.zeros(TINY["n_head"], 3))


# ── 4. Decoding ─────────────────────────────────────────────────────────────────

def test_generate_extends_positions_and_runs():
    model = _gtlm(spd=True, max_spd=8)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    batch = _graph_batch(input_ids, shortest_path_dists=SPD)

    out = model.generate(**batch, max_new_tokens=4, do_sample=False,
                         pad_token_id=TINY["vocab_size"] - 1)
    assert out.shape == (1, 11)
    # Generated tokens join the prompt node and continue its local position counter
    # (the prompt node holds local 0,1 in the input, so the fed-back tokens are 2,3,4).
    # The last sampled token is never fed back, hence 10 positions for 4 new tokens.
    positions = model.transformer._alibi_positions
    assert positions.shape == (1, 10)
    assert positions[0, -3:].tolist() == [2, 3, 4]
    assert torch.equal(positions[0, :7], POSITION_IDS[0])


# ── 5. The unwired backend is refused, not silently ignored ─────────────────────

def test_flex_backend_is_rejected():
    with pytest.raises(ValueError, match="eager"):
        GTLMBloomForCausalLM(GTLMBloomConfig(graph_attn_impl="flex", **TINY))
