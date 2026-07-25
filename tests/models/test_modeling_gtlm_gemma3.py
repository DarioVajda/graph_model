"""
GTLM on a layer-heterogeneous RoPE backbone (Gemma-3).

Llama gives one RoPE base for every layer and BLOOM gives ALiBi; Gemma-3 gives a
third regime — most layers rotate at a *local* base (``rope_local_base_freq``) and
every ``sliding_window_pattern``-th layer at a *global* one (``rope_theta``). The
tests pin the claims that would make the adapter wrong:

  1. **The backbone is unharmed.** With no graph bias and a single-node graph
     (structural mask reduces to plain causal, positions are ``arange``),
     GTLM-Gemma3 reproduces stock ``Gemma3ForCausalLM`` logits exactly — for any
     length up to ``sliding_window``, which covers every real GraphQA batch
     (packed sequences there top out at ~90 tokens).
  2. **The graph bias reaches the scores.** Turning a bias on changes the logits
     and its parameters receive gradients.

Plus the three deviations the adapter absorbs locally (see the module docstring of
``src/models/modeling_gtlm_gemma3.py``): the dropped sliding window, the refusal to
let Gemma-3 build a ``HybridCache``, and the softcapping guard. Unlike the BLOOM
probe this backbone is Strategy B, so there is no forward override to test — the
dual-RoPE test instead pins that the stock machinery still reaches the layers.
"""

import pytest
import torch

from transformers.cache_utils import DynamicCache, HybridCache
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from src.models import GTLMGemma3Config, GTLMGemma3ForCausalLM

DTYPE = torch.float32

# A small window and pattern so both layer kinds exist and the window is reachable
# within a test-sized sequence (the real gemma-3-1b values are 512 / 6).
TINY = dict(
    vocab_size=128, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, intermediate_size=64,
    sliding_window=4, sliding_window_pattern=2, attention_dropout=0.0,
    query_pre_attn_scalar=8, rope_theta=1_000_000.0, rope_local_base_freq=10_000.0,
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
    cfg = GTLMGemma3Config(k_hop=0, graph_attn_impl="eager",
                           checkpoint_graph_bias=False, **bias, **TINY)
    return GTLMGemma3ForCausalLM(cfg).eval()


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


def _single_node_batch(length):
    """A one-node graph with ``arange`` positions — the degenerate case where GTLM
    must reduce to the stock backbone."""
    input_ids = torch.randint(0, TINY["vocab_size"], (1, length))
    return input_ids, dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(length).unsqueeze(0),
        node_ids=torch.zeros(1, length, dtype=torch.long),
        prompt_node=torch.tensor([0]),
        num_nodes=torch.tensor([1]),
    )


def _paired_models():
    """A stock and a GTLM model sharing one set of backbone weights."""
    torch.manual_seed(0)
    stock = Gemma3ForCausalLM(Gemma3TextConfig(**TINY)).eval()
    gtlm = _gtlm()
    missing, unexpected = gtlm.load_state_dict(stock.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if "graph_bias" not in m], missing
    return stock, gtlm


# ── 1. The pretrained backbone is reproduced exactly ────────────────────────────

@pytest.mark.parametrize("length", [2, 4])
def test_matches_stock_gemma3_on_a_single_node_graph(length):
    """Within the sliding window — the only regime GraphQA ever reaches — GTLM-Gemma3
    must be stock Gemma-3. This is what makes the graph bias the *only* difference
    between the two models."""
    assert length <= TINY["sliding_window"]
    stock, gtlm = _paired_models()
    input_ids, flat = _single_node_batch(length)

    with torch.no_grad():
        ours = gtlm(**flat).logits
        theirs = stock(input_ids=input_ids, attention_mask=flat["attention_mask"]).logits

    assert torch.allclose(ours, theirs, atol=1e-6, rtol=0), \
        f"max |diff| = {(ours - theirs).abs().max().item():.3e}"


def test_sliding_window_is_dropped_beyond_the_window():
    """Past ``sliding_window`` the adapter deliberately stops matching stock Gemma-3:
    the band is defined over packed serialization order and would hide most of the
    graph from the sliding layers. Pinned so the divergence stays a decision rather
    than becoming an accident — and so the boundary is documented in code."""
    stock, gtlm = _paired_models()
    input_ids, flat = _single_node_batch(3 * TINY["sliding_window"])

    with torch.no_grad():
        ours = gtlm(**flat).logits
        theirs = stock(input_ids=input_ids, attention_mask=flat["attention_mask"]).logits

    assert not torch.allclose(ours, theirs, atol=1e-4, rtol=0), \
        "expected the dropped sliding window to change the logits beyond the window"


# ── 2. The graph bias reaches the attention scores ──────────────────────────────

def test_graph_bias_changes_logits_and_receives_gradients():
    torch.manual_seed(0)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))

    plain = _gtlm()
    biased = _gtlm(spd=True, max_spd=8)
    biased.load_state_dict(plain.state_dict(), strict=False)
    # A freshly initialized SPD bias can start near zero; make it visibly non-zero
    # so "the logits changed" tests the wiring rather than the initializer.
    with torch.no_grad():
        for name, param in biased.named_parameters():
            if "graph_bias" in name:
                param.add_(torch.randn_like(param))

    with torch.no_grad():
        before = plain(**_graph_batch(input_ids)).logits
        after = biased(**_graph_batch(input_ids, shortest_path_dists=SPD)).logits
    assert not torch.allclose(before, after, atol=1e-5, rtol=0), \
        "the SPD bias did not reach the attention scores"

    biased.train()
    loss = biased(**_graph_batch(input_ids, shortest_path_dists=SPD, labels=input_ids)).loss
    loss.backward()
    grads = [p.grad for n, p in biased.named_parameters() if "graph_bias" in n]
    assert grads and all(g is not None for g in grads), "graph bias got no gradient"
    assert max(g.abs().max() for g in grads) > 0


# ── 3. Both RoPE bases are live and reach their layers ──────────────────────────

def test_local_and_global_layers_receive_different_rope():
    """The point of this backbone: sliding layers rotate at ``rope_local_base_freq``
    and the others at ``rope_theta``. Strategy B leaves that machinery untouched, so
    this pins that the swap-in of GTLM attention did not collapse the two."""
    gtlm = _gtlm()
    seen = {}

    def capture(idx):
        def hook(module, args, kwargs):
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) > 1:
                pe = args[1]
            seen[idx] = pe
        return hook

    for i, layer in enumerate(gtlm.model.layers):
        layer.self_attn.register_forward_pre_hook(capture(i), with_kwargs=True)

    _, flat = _single_node_batch(4)
    with torch.no_grad():
        gtlm(**flat)

    sliding = [i for i, l in enumerate(gtlm.model.layers) if l.is_sliding]
    globals_ = [i for i, l in enumerate(gtlm.model.layers) if not l.is_sliding]
    assert sliding and globals_, "the tiny config must contain both layer kinds"

    cos_local = seen[sliding[0]][0]
    cos_global = seen[globals_[0]][0]
    assert not torch.allclose(cos_local, cos_global), \
        "local and global layers received the same RoPE — the dual base collapsed"
    # Every layer of a kind shares one base, so this is a per-kind property.
    for i in sliding[1:]:
        assert torch.allclose(seen[i][0], cos_local)
    for i in globals_[1:]:
        assert torch.allclose(seen[i][0], cos_global)


def test_per_node_position_reset_reaches_rope():
    """GTLM's per-node ``position_ids`` — not serialization order — drive the
    rotation, on both bases."""
    gtlm = _gtlm()
    seen = {}

    def hook(module, args, kwargs):
        pe = kwargs.get("position_embeddings")
        seen["cos"] = (pe if pe is not None else args[1])[0]

    gtlm.model.layers[0].self_attn.register_forward_pre_hook(hook, with_kwargs=True)

    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    with torch.no_grad():
        gtlm(**_graph_batch(input_ids))

    cos = seen["cos"][0]                      # (L, head_dim)
    # position_ids = [0,1,2, 0,1, 0,1]: tokens sharing a local position must share a
    # rotation, and token 3 (node 1, position 0) must match token 0 (node 0, pos 0).
    assert torch.allclose(cos[0], cos[3]) and torch.allclose(cos[0], cos[5])
    assert torch.allclose(cos[1], cos[4]) and torch.allclose(cos[1], cos[6])
    assert not torch.allclose(cos[0], cos[1])


# ── 4. Caching: never HybridCache ───────────────────────────────────────────────

def test_eval_forward_returns_a_dynamic_cache():
    """Gemma-3 would build a ``HybridCache`` whose sliding layers hold only
    ``min(sliding_window, max_cache_len)`` keys — at ``L > sliding_window`` they
    return the *last* window of keys while GTLM's structural mask still spans all
    of ``L``, and HF's eager kernel silently slices the mask to the first columns.
    The adapter's pre-hook installs a ``DynamicCache`` instead, matching Llama."""
    gtlm = _gtlm()
    _, flat = _single_node_batch(3 * TINY["sliding_window"])
    with torch.no_grad():
        out = gtlm(**flat, use_cache=True)

    assert isinstance(out.past_key_values, DynamicCache)
    assert not isinstance(out.past_key_values, HybridCache)
    # The whole KV span survives on every layer — the truncation this guards against.
    for layer_idx in range(TINY["num_hidden_layers"]):
        assert out.past_key_values.key_cache[layer_idx].shape[-2] == flat["input_ids"].shape[1]


def test_training_forward_builds_no_cache():
    gtlm = _gtlm().train()
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    out = gtlm(**_graph_batch(input_ids))
    assert out.past_key_values is None


# ── 5. Decoding ─────────────────────────────────────────────────────────────────

def test_generate_extends_positions_and_runs():
    gtlm = _gtlm(spd=True, max_spd=8)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    out = gtlm.generate(**_graph_batch(input_ids, shortest_path_dists=SPD),
                        max_new_tokens=3, do_sample=False)
    assert out.shape == (1, 10)


def test_generation_does_not_request_the_hybrid_cache():
    """``cache_implementation='hybrid'`` is a Gemma-3 config default; a static cache
    sized to the prefill cannot serve the shared decode path, which grows
    ``node_ids`` / ``position_ids`` off ``past_key_values.get_seq_length()``.

    Asserted on the cache the decoder stack actually receives, not on the config:
    ``generate`` deep-copies the generation config, so the adapter's override acts
    on that copy and leaves the caller's object alone (checked below)."""
    gtlm = _gtlm()
    assert gtlm.config.cache_implementation is None
    gtlm.generation_config.cache_implementation = "hybrid"      # simulate a stale one

    seen = []
    gtlm.model.register_forward_pre_hook(
        lambda m, a, kw: seen.append(type(kw.get("past_key_values"))), with_kwargs=True)

    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    gtlm.generate(**_graph_batch(input_ids), max_new_tokens=2, do_sample=False)

    assert seen, "the decoder stack was never called"
    assert all(t is DynamicCache for t in seen), seen
    # The override must not mutate the caller's generation config.
    assert gtlm.generation_config.cache_implementation == "hybrid"


# ── 6. Softcapping is refused, not silently dropped ─────────────────────────────

@pytest.mark.parametrize("field", ["attn_logit_softcapping", "final_logit_softcapping"])
def test_softcapping_is_rejected(field):
    """The shared stack applies neither softcapping site. Gemma-3 sets both to None
    so that is exact; anything else (a Gemma-2 checkpoint, say) must fail loudly
    rather than train a model that is not its pretrained backbone."""
    cfg = GTLMGemma3Config(k_hop=0, graph_attn_impl="eager", **{field: 50.0}, **TINY)
    with pytest.raises(ValueError, match=field):
        GTLMGemma3ForCausalLM(cfg)


# ── 7. Round-trip ───────────────────────────────────────────────────────────────

def test_save_load_round_trip_preserves_logits_and_bias(tmp_path):
    gtlm = _gtlm(spd=True, max_spd=8)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    with torch.no_grad():
        before = gtlm(**_graph_batch(input_ids, shortest_path_dists=SPD)).logits

    gtlm.save_pretrained(tmp_path)
    reloaded = GTLMGemma3ForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(**_graph_batch(input_ids, shortest_path_dists=SPD)).logits

    assert torch.equal(before, after)
    assert reloaded.config.model_type == "gtlm_gemma3"
    assert reloaded.config.cache_implementation is None
    # trust_remote_code: the flat module closure must be bundled, not just the entry.
    bundled = {p.name for p in tmp_path.glob("*.py")}
    assert {"modeling_gtlm_gemma3.py", "causal_lm.py", "attention.py", "bias.py",
            "dispatch.py", "structural_mask.py", "flex_kernel.py", "context.py",
            "config.py", "io.py"} <= bundled, sorted(bundled)
