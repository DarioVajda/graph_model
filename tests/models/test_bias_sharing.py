"""Tests for the layer-grouped magnetic bias (``magnetic_groups``).

``magnetic_groups=G`` replaces the per-layer ``MagneticBias`` with G instances
owned by the causal-LM mixin, layer ``l`` served by group ``l*G//L``. Each group
is computed once per pass by its OWNER layer and read by its FOLLOWERS, and the
tensor is released once the group's last consumer has taken it — so peak
residency is one ``(B,H,N,N)`` tensor rather than G. In backward, followers are
reached before the owner, so the first one rematerialises the value under
``no_grad`` and passes it on as a leaf that requires grad.

That last part is the whole reason this suite exists. HF's non-reentrant
gradient checkpointing requires a region's recompute to save the same tensors its
forward did, and the naive schedules violate it silently or loudly. So the
gradient checks below are run with gradient checkpointing ON, against a
redundant-compute reference that shares the same parameters but recomputes the
bias inside every layer (obviously-correct autograd, no sharing to get wrong).

Pinned here:

  * group assignment is contiguous, uses every group, and is balanced;
  * loss and EVERY parameter gradient match the reference, with gradient
    checkpointing on and off, on the eager (float64) AND flex (float32 — Inductor
    cannot lower FlexAttention in float64) backends;
  * every grouped-bias parameter actually receives a non-zero gradient (a silent
    zero here would make a G sweep measure nothing);
  * gradients are not double-counted (a follower's throwaway leaf must not add a
    second contribution);
  * ``magnetic_groups=L`` reproduces per-layer ``magnetic`` and
    ``magnetic_groups=1`` reproduces ``magnetic_shared``, numerically;
  * eval/generation parity, and that the released tensors are really freed.

Run with:  pytest tests/models/test_bias_sharing.py -v
"""

import gc
import weakref

import pytest
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import MagneticBias, layer_group_map
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from tests.helpers.tiny_model import BASE_CONFIG

DEVICE = torch.device("cpu")
DTYPE = torch.float64

# Flex needs shapes its kernels actually have choices for — head_dim 16 / block 16
# autotunes to an empty choice list. These mirror tests/models/test_flex_attention.py,
# which is the known-good flex configuration in this repo.
FLEX_BLOCK = 128
FLEX_HIDDEN = 128
FLEX_COLLATE = dict(pad_token_id=0, pad_to_block=True, block_size=FLEX_BLOCK,
                    len_buckets=[FLEX_BLOCK], node_buckets=[16])


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

def _batch(node_counts=(6, 8), tokens_per_node=5, seed=0, pad_to_block=None):
    torch.manual_seed(seed)
    items = []
    for n in node_counts:
        item = {
            "num_nodes": n,
            "prompt_node": n - 1,
            "edges": [(i, (i + 1) % n) for i in range(n)],
            "input_ids": [torch.randint(1, 256, (tokens_per_node,)).tolist()
                          for _ in range(n)],
            "magnetic_V": torch.randn(n, n, 2, dtype=DTYPE),
            "magnetic_lambdas": torch.randn(n, dtype=DTYPE),
        }
        item["labels"] = torch.tensor(item["input_ids"][item["prompt_node"]],
                                      dtype=torch.long)
        items.append(item)
    kw = {"pad_to_block": pad_to_block} if pad_to_block else {}
    return GraphCollatorV2(pad_token_id=0, k_hop=0, **kw)([dict(it) for it in items])


def _flex_batch(seed=0):
    """A CUDA fp32 batch on the block-aligned shapes flex kernels support."""
    torch.manual_seed(seed)
    items = []
    for tok_lens, prompt, edges in (([3, 2, 4, 2], 3, [(0, 1), (1, 2), (2, 3), (0, 3)]),
                                    ([2, 3, 2], 0, [(0, 1), (1, 2)])):
        n = len(tok_lens)
        item = {
            "num_nodes": n, "prompt_node": prompt, "edges": edges,
            "input_ids": [torch.randint(1, 256, (l,)).tolist() for l in tok_lens],
            "shortest_path_dists": torch.randint(0, 5, (n, n)),
            "magnetic_V": torch.randn(n, n, 2),
            "magnetic_lambdas": torch.randn(n),
        }
        item["labels"] = torch.tensor(item["input_ids"][prompt], dtype=torch.long)
        items.append(item)
    batch = GraphCollatorV2(k_hop=0, **FLEX_COLLATE)(items)
    return {k: (v.to(device="cuda", dtype=torch.float32) if torch.is_floating_point(v)
                else v.to("cuda")) for k, v in batch.items()}


def _config(num_layers, *, impl="eager", **overrides):
    cfg = dict(BASE_CONFIG)
    cfg["num_hidden_layers"] = num_layers
    cfg.update(k_hop=0, graph_attn_impl=impl, magnetic_dim=8,
               checkpoint_graph_bias=True, spd=True)
    cfg.update(overrides)
    return GTLMLlamaConfig(**cfg)


def _model(num_layers, *, grad_ckpt=True, seed=0, impl="eager", dtype=DTYPE,
           device=DEVICE, **overrides):
    torch.manual_seed(seed)
    model = GTLMLlamaForCausalLM(_config(num_layers, impl=impl, **overrides)).to(device, dtype)
    if grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def _grow_in(modules, seed=7, std=0.1):
    """Randomise the zero-init final projection so the bias is not inert."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for m in modules:
            torch.nn.init.normal_(m.proj[2].weight, std=std)
            torch.nn.init.normal_(m.proj[2].bias, std=std)


def _grouped_model(num_layers, num_groups, *, grad_ckpt=True, impl="eager", dtype=DTYPE,
                   device=DEVICE, **overrides):
    model = _model(num_layers, grad_ckpt=grad_ckpt, impl=impl, dtype=dtype, device=device,
                   magnetic=False, magnetic_groups=num_groups, **overrides)
    _grow_in(model.group_graph_bias)
    return model


def _identical_pair(num_layers, num_groups, *, grad_ckpt=True, impl="eager", dtype=DTYPE,
                    device=DEVICE, **overrides):
    """Two grouped models with bit-identical weights.

    Building them separately (rather than deep-copying) keeps the gradient state
    independent; the assert makes the shared-seed assumption explicit instead of
    load-bearing — module construction order affects the RNG stream, and an
    earlier version of this file silently compared two different backbones.
    """
    kw = dict(grad_ckpt=grad_ckpt, impl=impl, dtype=dtype, device=device, **overrides)
    a = _grouped_model(num_layers, num_groups, **kw)
    b = _grouped_model(num_layers, num_groups, **kw)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"seeded construction diverged on {k}"
    return a, b


def _reference_dispatch(model, num_layers, num_groups):
    """Ground truth: every layer recomputes its group's bias itself.

    No sharing, no caching, no release — plain autograd over parameters used k
    times, which is obviously correct. Whatever the grouped implementation does
    must match this.
    """
    import src.models.dispatch as dispatch_mod
    group_of = layer_group_map(num_layers, num_groups)
    orig = dispatch_mod.compute_node_bias

    def patched(module, ctx, dtype, device):
        cache, ctx.group_bias = ctx.group_bias, None     # suppress the grouped path
        try:
            node_bias = orig(module, ctx, dtype, device)
        finally:
            ctx.group_bias = cache
        g = group_of[module.graph_bias.layer_idx]
        b = model.group_graph_bias[g](
            dtype=dtype, device=device, num_nodes=ctx.num_nodes,
            magnetic=ctx.features["magnetic"]).to(dtype)
        return b if node_bias is None else node_bias + b

    return patched, orig


def _run(model, batch, patched=None, orig=None):
    """Forward + backward; returns (loss, {param_name: grad})."""
    import src.models.dispatch as dispatch_mod
    if patched is not None:
        dispatch_mod.compute_node_bias = patched
    try:
        out = model(**batch)
        out.loss.backward()
    finally:
        if patched is not None:
            dispatch_mod.compute_node_bias = orig
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return out.loss.detach().clone(), grads


def _sync_params(dst, src, mapping):
    """Copy every same-named parameter, then the explicitly mapped ones."""
    with torch.no_grad():
        s = dict(src.named_parameters())
        for name, p in dst.named_parameters():
            if name in s:
                p.copy_(s[name])
        for dst_name, src_name in mapping.items():
            dict(dst.named_parameters())[dst_name].copy_(s[src_name])


# ─── Group assignment ─────────────────────────────────────────────────────────

class TestLayerGroupMap:

    @pytest.mark.parametrize("L,G", [(16, 1), (16, 2), (16, 4), (16, 8), (16, 16),
                                     (32, 8), (8, 3), (16, 5), (12, 5)])
    def test_contiguous_complete_and_balanced(self, L, G):
        m = layer_group_map(L, G)
        assert len(m) == L
        assert sorted(set(m)) == list(range(G)), "every group must own >=1 layer"
        assert m == sorted(m), "groups must be contiguous blocks of layers"
        sizes = [m.count(g) for g in range(G)]
        assert max(sizes) - min(sizes) <= 1, f"unbalanced group sizes {sizes}"
        assert sum(sizes) == L

    def test_endpoints(self):
        assert layer_group_map(16, 16) == list(range(16))    # one group per layer
        assert layer_group_map(16, 1) == [0] * 16            # one group overall

    @pytest.mark.parametrize("L,G", [(16, 0), (16, 17), (4, -1)])
    def test_rejects_out_of_range(self, L, G):
        with pytest.raises(ValueError):
            layer_group_map(L, G)


# ─── Config validation ────────────────────────────────────────────────────────

class TestConfigValidation:

    @pytest.mark.parametrize("conflict", [{"magnetic": True},
                                          {"magnetic_shared": True},
                                          {"magnetic_content": True}])
    def test_mutually_exclusive(self, conflict):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _config(4, magnetic_groups=2, **conflict)

    @pytest.mark.parametrize("G", [0, 5])
    def test_group_count_bounds(self, G):
        if G == 0:
            _config(4, magnetic=False, magnetic_groups=G)      # 0 = disabled, fine
        else:
            with pytest.raises(ValueError, match="num_hidden_layers"):
                _config(4, magnetic=False, magnetic_groups=G)

    def test_module_count_matches_G(self):
        model = _model(8, magnetic=False, magnetic_groups=4)
        assert len(model.group_graph_bias) == 4
        # per-layer magnetic must NOT also be instantiated
        for layer in model.model.layers:
            keys = layer.self_attn.graph_bias._active
            assert "magnetic" not in keys


# ─── The core claim: gradients match a redundant-compute reference ────────────

class TestGradientEquivalence:

    @pytest.mark.parametrize("L,G", [(4, 1), (4, 2), (4, 4), (8, 2), (8, 4), (6, 2), (6, 3)])
    @pytest.mark.parametrize("grad_ckpt", [True, False])
    def test_matches_reference(self, L, G, grad_ckpt):
        batch = _batch()
        model, ref_model = _identical_pair(L, G, grad_ckpt=grad_ckpt)
        patched, orig = _reference_dispatch(ref_model, L, G)
        ref_loss, ref_grads = _run(ref_model, batch, patched, orig)
        loss, grads = _run(model, batch)

        assert torch.allclose(loss, ref_loss, rtol=0, atol=1e-12), (
            f"loss {loss.item():.15f} vs reference {ref_loss.item():.15f}")
        assert set(grads) == set(ref_grads)
        for name in ref_grads:
            assert torch.allclose(grads[name], ref_grads[name], rtol=1e-10, atol=1e-14), (
                f"grad mismatch on {name}: "
                f"max |delta| = {(grads[name] - ref_grads[name]).abs().max().item():.3e}")

    @pytest.mark.parametrize("L,G", [(4, 2), (8, 4)])
    @pytest.mark.parametrize("grad_ckpt", [True, False])
    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="FlexAttention has no CPU backward; needs a GPU node")
    def test_matches_reference_flex(self, L, G, grad_ckpt):
        """Flex is the backend two of the three headline recipes use, and its
        score_mod captures the bias tensor into a compiled kernel — so the
        leaf-vs-graph-tensor distinction has to hold there too.

        float32, not the float64 the eager tests use: Inductor cannot lower
        FlexAttention in float64 (``LoweringException`` on the fp64 query
        buffer), and fp32 is what tests/models/test_flex_attention.py pins parity
        at. Both arms run the identical kernel, so any difference is roundoff.
        """
        flex_dtype = torch.float32
        batch = _flex_batch()
        model, ref_model = _identical_pair(
            L, G, impl="flex", grad_ckpt=grad_ckpt, dtype=flex_dtype, device="cuda",
            hidden_size=FLEX_HIDDEN, intermediate_size=2 * FLEX_HIDDEN,
            flex_block_size=FLEX_BLOCK)
        patched, orig = _reference_dispatch(ref_model, L, G)
        ref_loss, ref_grads = _run(ref_model, batch, patched, orig)
        loss, grads = _run(model, batch)

        assert torch.allclose(loss, ref_loss, rtol=1e-5, atol=1e-6), (
            f"flex loss {loss.item():.8f} vs reference {ref_loss.item():.8f}")
        for name in ref_grads:
            assert torch.allclose(grads[name], ref_grads[name], rtol=1e-4, atol=1e-6), (
                f"flex grad mismatch on {name}: "
                f"max |delta| = {(grads[name] - ref_grads[name]).abs().max().item():.3e}")


# ─── Gradient flow into the bias parameters specifically ─────────────────────

class TestBiasGradientFlow:

    @pytest.mark.parametrize("L,G", [(4, 1), (4, 2), (8, 4), (8, 8)])
    def test_every_group_parameter_gets_nonzero_grad(self, L, G):
        """A silently-zero bias gradient would make a G sweep measure nothing."""
        model = _grouped_model(L, G)
        _run(model, _batch())
        seen = 0
        for name, p in model.named_parameters():
            if not name.startswith("group_graph_bias."):
                continue
            seen += 1
            assert p.grad is not None, f"{name} received no gradient at all"
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
            assert p.grad.abs().max() > 0, f"{name} gradient is identically zero"
        # lambda_lin, deep_set[0], proj[0], proj[2]: 4 modules x (weight, bias)
        assert seen == G * 8, f"expected {G * 8} grouped-bias tensors, saw {seen}"

    @pytest.mark.parametrize("L,G", [(4, 2), (8, 4)])
    def test_no_double_counting(self, L, G):
        """The follower's throwaway leaf must not contribute a second time.

        Scaling a group's contribution by scaling how many layers consume it is
        not something we can isolate directly, so instead compare against the
        reference at TWO different group counts: if followers double-counted, the
        grouped run would differ from the reference by a factor tied to k.
        """
        batch = _batch()
        model, ref_model = _identical_pair(L, G)
        patched, orig = _reference_dispatch(ref_model, L, G)
        _, ref_grads = _run(ref_model, batch, patched, orig)
        _, grads = _run(model, batch)
        for g in range(G):
            key = f"group_graph_bias.{g}.proj.2.weight"
            ratio = grads[key].norm() / ref_grads[key].norm()
            assert abs(ratio.item() - 1.0) < 1e-9, (
                f"group {g} gradient scaled by {ratio.item():.6f} vs reference "
                "— a doubled or dropped contribution")

    def test_gradient_reaches_token_embeddings(self):
        """End-to-end flow: the bias must not detach the graph from the inputs."""
        model = _grouped_model(4, 2)
        _, grads = _run(model, _batch())
        emb = grads["model.embed_tokens.weight"]
        assert emb.abs().max() > 0

    def test_accumulates_over_two_backwards(self):
        """Gradient accumulation must not be corrupted by the release bookkeeping."""
        model = _grouped_model(4, 2)
        b1, b2 = _batch(seed=0), _batch(seed=1)
        _run(model, b1)
        once = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                if p.grad is not None and n.startswith("group_graph_bias.")}
        model.zero_grad(set_to_none=True)
        model(**b1).loss.backward()
        model(**b2).loss.backward()
        twice = {n: p.grad for n, p in model.named_parameters()
                 if p.grad is not None and n.startswith("group_graph_bias.")}
        # accumulating a second, different batch must move every tensor
        assert any((twice[n] - once[n]).abs().max() > 0 for n in once)
        for n in once:
            assert torch.isfinite(twice[n]).all()


# ─── Backward compatibility: the legacy flags as points on the same axis ─────

class TestLegacyEquivalence:
    """``magnetic_groups=L`` is per-layer ``magnetic``; ``magnetic_groups=1`` is
    ``magnetic_shared``. The legacy flags keep their own code paths (and their
    parameter names, so existing checkpoints still load), so this pins that the
    two paths agree numerically rather than assuming it."""

    def test_groups_L_equals_per_layer_magnetic(self):
        L = 4
        legacy = _model(L, magnetic=True)
        per_layer = [layer.self_attn.graph_bias.bias_modules[-1]
                     for layer in legacy.model.layers]
        assert all(isinstance(m, MagneticBias) for m in per_layer)
        _grow_in(per_layer)

        grouped = _grouped_model(L, L)
        # Same-named params (backbone, spd) copied wholesale; the magnetic ones
        # move from legacy's per-layer slot into the matching group.
        _sync_params(grouped, legacy, {
            f"group_graph_bias.{i}.{leaf}":
                f"model.layers.{i}.self_attn.graph_bias.bias_modules.1.{leaf}"
            for i in range(L) for leaf, _ in per_layer[i].named_parameters()})

        batch = _batch()
        legacy_loss, legacy_grads = _run(legacy, batch)
        loss, grads = _run(grouped, batch)

        assert torch.allclose(loss, legacy_loss, rtol=0, atol=1e-12)
        for i, src in enumerate(per_layer):
            for leaf, _ in src.named_parameters():
                a = grads[f"group_graph_bias.{i}.{leaf}"]
                b = legacy_grads[
                    f"model.layers.{i}.self_attn.graph_bias.bias_modules.1.{leaf}"]
                assert torch.allclose(a, b, rtol=1e-10, atol=1e-14), (
                    f"layer {i} / {leaf}: max |delta| = {(a - b).abs().max().item():.3e}")

    def test_groups_1_equals_magnetic_shared(self):
        L = 4
        legacy = _model(L, magnetic=False, magnetic_shared=True)
        _grow_in(legacy.shared_graph_bias)

        grouped = _grouped_model(L, 1)
        _sync_params(grouped, legacy, {
            f"group_graph_bias.0.{leaf}": f"shared_graph_bias.0.{leaf}"
            for leaf, _ in legacy.shared_graph_bias[0].named_parameters()})

        batch = _batch()
        legacy_loss, legacy_grads = _run(legacy, batch)
        loss, grads = _run(grouped, batch)

        assert torch.allclose(loss, legacy_loss, rtol=0, atol=1e-12)
        for leaf, _ in legacy.shared_graph_bias[0].named_parameters():
            a = grads[f"group_graph_bias.0.{leaf}"]
            b = legacy_grads[f"shared_graph_bias.0.{leaf}"]
            assert torch.allclose(a, b, rtol=1e-10, atol=1e-14), (
                f"{leaf}: max |delta| = {(a - b).abs().max().item():.3e}")

    def test_legacy_paths_untouched_when_groups_off(self):
        """magnetic_groups=0 must build exactly the modules it always did."""
        assert _model(4, magnetic=True).group_graph_bias is None
        assert _model(4, magnetic=False, magnetic_shared=True).group_graph_bias is None


# ─── Memory behaviour and eval parity ────────────────────────────────────────

class TestReleaseAndEval:

    def test_group_tensors_are_released_after_forward(self):
        """Peak residency is one tensor, not G: nothing in the graph saves the
        bias, so dropping the cache's reference really frees it."""
        model = _grouped_model(8, 4)
        refs = []
        real_get = model.group_graph_bias[0].__class__.forward

        seen = []

        def spy(self, **kw):
            out = real_get(self, **kw)
            seen.append(weakref.ref(out))
            return out

        model.group_graph_bias[0].__class__.forward = spy
        try:
            out = model(**_batch())
        finally:
            model.group_graph_bias[0].__class__.forward = real_get
        gc.collect()
        alive = [r for r in seen if r() is not None]
        assert seen, "no group bias was computed"
        assert not alive, f"{len(alive)}/{len(seen)} group tensors still resident"
        out.loss.backward()          # and backward still works after the release

    def test_eval_matches_train_forward(self):
        """Eval takes the compute-once-and-hold branch; it must produce the same
        logits as the training-mode forward on the same inputs."""
        model = _grouped_model(8, 4, grad_ckpt=False)
        batch = _batch()
        model.train()
        with torch.no_grad():
            train_logits = model(**batch).logits
        model.eval()
        with torch.no_grad():
            eval_logits = model(**batch).logits
        assert torch.allclose(train_logits, eval_logits, rtol=0, atol=1e-12)

    def test_generation_runs(self):
        """The eval cache is keyed per group and survives decode steps."""
        model = _grouped_model(4, 2, grad_ckpt=False)
        model.eval()
        batch = _batch(node_counts=(5,))
        gen = model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            node_ids=batch["node_ids"], prompt_node=batch["prompt_node"],
            num_nodes=batch["num_nodes"],
            shortest_path_dists=batch.get("shortest_path_dists"),
            magnetic_V=batch.get("magnetic_V"),
            magnetic_lambdas=batch.get("magnetic_lambdas"),
            position_ids=batch.get("position_ids"),
            max_new_tokens=3, do_sample=False)
        assert gen.shape[1] == batch["input_ids"].shape[1] + 3
