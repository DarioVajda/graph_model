"""Correctness gate for ``MagneticPairTrunk`` + ``MagneticNonlinearBias``.

These run BEFORE the training sweep, by design: not one of the failures they catch
announces itself in a training curve. A transposed pool axis silently destroys
directionality; a missed config gate emits no eigenvectors and the run trains with
no bias at all; a fully-padded row NaNs the whole batch's gradients. All of them
read as "the non-linear pooled head didn't help", which is precisely the
conclusion the sweep exists to draw.

Numbering follows ``src/models/NON_LINEAR_BIAS.md`` §7.
"""

import networkx as nx
import pytest
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import BIAS_TYPES, MagneticNonlinearBias, MagneticPairTrunk
from src.utils.magnetic_lap import get_magnetic_laplacian_coords
from src.utils.text_graph_collator_v2 import GraphCollatorV2

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=1024,
    pad_token_id=0, _attn_implementation="eager",
)
_MAG_DIM = 16
_H, _H_KV, _HEAD_DIM, _D_STRUCT = 4, 2, 16, 8
_DEV = torch.device("cpu")


class _Cfg:
    """Minimal bias_config for constructing the trunk / head directly."""
    magnetic_dim = _MAG_DIM
    magnetic_struct_dim = _D_STRUCT
    magnetic_pool = "attn"
    num_key_value_heads = _H_KV
    hidden_size = 64
    bias_self_node = True          # the only setting this arm is ever run at


class _CfgMasked(_Cfg):
    """The diagonal-masked variant, kept only to pin what the factorization can
    and cannot express (§3)."""
    bias_self_node = False


class _CfgUniform(_Cfg):
    magnetic_pool = "uniform"


class _CfgMHA(_Cfg):
    """No GQA — the Bloom case, where per-KV-group degenerates to per-head. Also
    what the axis test needs: it compares the two pools' parameters directly, so
    they have to have the same shape."""
    num_key_value_heads = None


def _items(node_counts=(4, 9, 6), seed=0, m=None):
    """Batch items carrying magnetic eigenvectors, mixed graph sizes.

    ``m=None`` stores the full spectrum (m == N), which is what an ``m=0`` cache
    holds and what forces the collator to pad the smaller graphs — the condition
    test 2 exists to check.
    """
    torch.manual_seed(seed)
    items = []
    for n in node_counts:
        mm = n if m is None else min(m, n)
        item = {
            "num_nodes": n,
            "prompt_node": n - 1,
            "edges": [(i, (i + 1) % n) for i in range(n)],
            "input_ids": [torch.randint(1, 256, (3,)).tolist() for _ in range(n)],
            "shortest_path_dists": torch.randint(0, 6, (n, n)),
            "magnetic_V": torch.randn(n, mm, 2),
            "magnetic_lambdas": torch.randn(mm),
        }
        item["labels"] = torch.tensor(item["input_ids"][item["prompt_node"]],
                                      dtype=torch.long)
        items.append(item)
    return items


def _batch(items, magnetic_m=0):
    return GraphCollatorV2(pad_token_id=0, magnetic_m=magnetic_m)(
        [dict(it) for it in items])


def _mag_inputs(batch):
    return ((batch["magnetic_V"].double(), batch["magnetic_lambdas"].double()),
            batch["num_nodes"])


def _pair(cfg=_Cfg, seed=0, live=True):
    """A ``(trunk, head)`` pair in fp64.

    ``live=True`` lifts ``gamma_in`` off its zero initialisation. Every test about
    what the head COMPUTES has to do this first — otherwise it would pass against
    a module whose output is identically zero, which is the whole point of the
    initialisation and useless as a subject.
    """
    torch.manual_seed(seed)
    trunk = MagneticPairTrunk(_H, _HEAD_DIM, cfg()).double()
    head = MagneticNonlinearBias(_H, _HEAD_DIM, cfg()).double()
    if live:
        with torch.no_grad():
            head.gamma_in.normal_(std=0.5)
            head.gamma_out.normal_(mean=1.0, std=0.2)
    return trunk, head


def _E(trunk, magnetic, num_nodes):
    return trunk(dtype=torch.float64, device=_DEV,
                 magnetic=magnetic, num_nodes=num_nodes)


def _bias(trunk, head, magnetic, num_nodes):
    return head(dtype=torch.float64, device=_DEV, num_nodes=num_nodes,
                pair_features=_E(trunk, magnetic, num_nodes))


# ── 1. Permutation invariance ────────────────────────────────────────────────
#
# The pool is permutation-EQUIVARIANT in theory: E is a linear map of the
# invariant Hermitian contraction, SiLU is pointwise, softmax pooling commutes
# with relabelling. The theory is not taken on trust, because a transposed axis or
# a mis-broadcast mask would break it silently. Fixtures are a star and a cycle,
# whose spectra are degenerate by construction — the condition under which any
# per-eigenvector quantity stops being well defined.

def _relabelled_pair(graph, order_a, order_b):
    """The same graph under two node orderings, as (V, lambdas, order) triples.

    ``nx.to_numpy_array`` uses INSERTION order, not integer labels, so the
    orderings must be built by adding nodes explicitly — relabelling and trusting
    the labels silently produces the same matrix twice and the test passes without
    testing anything. The solver runs in fp32, so ~1e-6 is the floor downstream.
    """
    out = []
    for order in (order_a, order_b):
        h = nx.DiGraph()
        h.add_nodes_from(order)
        h.add_edges_from(graph.edges())
        V, lam = get_magnetic_laplacian_coords(h, q=0.25, use_gpu=False)
        out.append((torch.as_tensor(V).double().unsqueeze(0),
                    torch.as_tensor(lam).double().unsqueeze(0), order))
    return out


@pytest.mark.parametrize("graph,name", [
    (nx.star_graph(4), "star"),        # lambda = [0,1,1,1,2]: a triply degenerate block
    (nx.cycle_graph(6), "cycle"),      # degenerate conjugate pairs
], ids=lambda v: v if isinstance(v, str) else "")
def test_bias_matrix_is_permutation_equivariant(graph, name):
    """The full (H, N, N) bias must permute with its nodes, not change."""
    trunk, head = _pair()
    n = graph.number_of_nodes()
    (Va, la, oa), (Vb, lb, ob) = _relabelled_pair(
        graph, list(range(n)), list(reversed(range(n))))
    num_nodes = torch.tensor([n])

    ba = _bias(trunk, head, (Va, la), num_nodes)[0]
    bb = _bias(trunk, head, (Vb, lb), num_nodes)[0]

    pos_b = {node: i for i, node in enumerate(ob)}
    perm = torch.tensor([pos_b[node] for node in oa])
    diff = (ba - bb[:, perm][:, :, perm]).abs().max().item()
    assert diff < 1e-4, f"{name}: bias moved by {diff} under relabelling"


# ── 15. Degenerate-spectrum invariance ───────────────────────────────────────

@pytest.mark.parametrize("graph,name", [
    (nx.star_graph(4), "star"),
    (nx.cycle_graph(6), "cycle"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_pooled_node_features_are_invariant_under_relabelling(graph, name):
    """z_out and z_in must follow their node, to fp tolerance.

    Stronger than the bias test above and the reason it exists separately: the
    bias is an inner product, so two compensating errors in z_out and z_in could
    cancel there and not here. These are the tensors a future kernel would append
    to Q and K, so they are what has to be invariant.
    """
    trunk, head = _pair()
    n = graph.number_of_nodes()
    (Va, la, oa), (Vb, lb, ob) = _relabelled_pair(
        graph, list(range(n)), list(reversed(range(n))))
    num_nodes = torch.tensor([n])

    qa, ka = head.structural_factors(_E(trunk, (Va, la), num_nodes), num_nodes)
    qb, kb = head.structural_factors(_E(trunk, (Vb, lb), num_nodes), num_nodes)

    pos_b = {node: i for i, node in enumerate(ob)}
    perm = torch.tensor([pos_b[node] for node in oa])
    for got, ref, tag in ((qa, qb, "z_out"), (ka, kb, "z_in")):
        diff = (got - ref[:, :, perm]).abs().max().item()
        assert diff < 1e-4, f"{name}: {tag} moved by {diff} under relabelling"


# ── The pooling axis (NON_LINEAR_BIAS.md §6.1) ───────────────────────────────

def test_the_two_pools_run_on_opposite_axes():
    """z_out pools rows and z_in pools columns — the whole of directionality.

    Given IDENTICAL parameters on both sides, pooling E from the outgoing side
    must equal pooling E-transposed from the incoming side, and vice versa. A pool
    that ran on the same axis twice would satisfy no shape assertion, produce a
    symmetric bias, and quietly delete the one property that makes the magnetic
    Laplacian worth using.
    """
    trunk, head = _pair(_CfgMHA)                   # H_Q == H_KV, so the shapes match
    with torch.no_grad():                          # force the two sides identical
        head.W_attn_in.copy_(head.W_attn_out)
        head.W_val_in.copy_(head.W_val_out)
        head.gamma_in.copy_(head.gamma_out)

    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    E = _E(trunk, magnetic, num_nodes)
    z_out, z_in = head.structural_factors(E, num_nodes)
    z_out_t, z_in_t = head.structural_factors(E.transpose(1, 2).contiguous(), num_nodes)

    assert (z_out - z_in_t).abs().max().item() < 1e-12
    assert (z_in - z_out_t).abs().max().item() < 1e-12
    # ...and on a genuinely asymmetric E the two must actually differ, or the
    # assertions above would be satisfied by a pool that is secretly symmetric.
    assert (E - E.transpose(1, 2)).abs().max().item() > 1e-6, "fixture E is symmetric"
    assert (z_out - z_in).abs().max().item() > 1e-6, "the two pools agree — same axis?"


def test_bias_is_not_symmetric():
    """b(i,j) != b(j,i). The end-to-end reading of the test above."""
    trunk, head = _pair()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    b = _bias(trunk, head, magnetic, num_nodes)
    assert (b - b.transpose(-1, -2)).abs().max().item() > 1e-6


# ── 2. Padded slots (the §2.2 self-loop clause) ──────────────────────────────

def test_padded_rows_are_finite_not_nan():
    """A batch whose node counts span 4 vs 40 — ``test_v2_ragged_magnetic_padding``'s
    spread, and graphqa batches 6 vs 191.

    Every row above a graph's own node count is fully masked except its diagonal.
    Without that clause the softmax is 0/0 and the WHOLE batch's gradients become
    NaN, not just the padded rows'. Backward is asserted, not just forward: a NaN
    that only appears in the backward pass is the expensive version of this bug.
    """
    trunk, head = _pair()
    batch = _batch(_items(node_counts=(4, 40, 7)))
    magnetic, num_nodes = _mag_inputs(batch)

    b = _bias(trunk, head, magnetic, num_nodes)
    assert torch.isfinite(b).all(), "non-finite bias on a ragged batch"
    b.sum().backward()
    for name, p in list(trunk.named_parameters()) + list(head.named_parameters()):
        assert p.grad is None or torch.isfinite(p.grad).all(), f"non-finite grad: {name}"


def test_padded_rows_attend_only_to_themselves():
    """The mask's ``| eye`` term, read directly: a padded row has exactly one open
    entry and it is the diagonal."""
    num_nodes = torch.tensor([4, 40, 7])
    allow = MagneticNonlinearBias._pool_mask(num_nodes, 40, _DEV)
    for b, n in enumerate(num_nodes.tolist()):
        assert allow[b, :n].sum(-1).min().item() == n, "a real row lost partners"
        if n < 40:
            pad = allow[b, n:]
            assert pad.sum(-1).unique().tolist() == [1], "padded row is not a self-loop"
            assert pad.diagonal(offset=n, dim1=0, dim2=1).all()


def test_a_small_graph_is_unaffected_by_a_large_one_in_its_batch():
    """The bias for a 4-node graph must not depend on what else is in the batch.

    This is the failure mode `test_v2_ragged_magnetic_padding` exists for: a
    mis-masked pad slot moves the SMALLER graphs' bias and silently shifts results
    instead of crashing.
    """
    trunk, head = _pair()
    items = _items(node_counts=(4, 40, 7))

    alone = _bias(trunk, head, *_mag_inputs(_batch([items[0]])))
    mixed = _bias(trunk, head, *_mag_inputs(_batch(items)))

    diff = (alone[0] - mixed[0, :, :4, :4]).abs().max().item()
    assert diff < 1e-10, f"padding leaked into the small graph's bias: {diff}"


# ── 9. Zero-init inertness, and that it is not the dead saddle ───────────────

def test_zero_init_bias_is_exactly_zero():
    trunk, head = _pair(live=False)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    assert _bias(trunk, head, magnetic, num_nodes).abs().max().item() == 0.0


def test_zero_init_is_not_a_dead_saddle():
    """gamma_in must have a NON-ZERO gradient at step 0.

    This is what separates a correct one-sided zero-init from zeroing both gains,
    where dB/dgamma_out ∝ gamma_in ⊙ z_in = 0 and symmetrically, so the arm never
    leaves the origin. Everything else is correctly still at zero gradient — it
    sits behind gamma_in and starts moving on the next step. Asserted so the
    asymmetry is on record as a one-step delay and not a defect.
    """
    trunk, head = _pair(live=False)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    _bias(trunk, head, magnetic, num_nodes).sum().backward()

    assert head.gamma_in.grad.abs().max().item() > 0, \
        "gamma_in has zero gradient at step 0 — the arm can never move"
    assert head.gamma_out.grad.abs().max().item() == 0.0
    assert head.W_val_out.grad.abs().max().item() == 0.0
    assert head.W_val_in.grad.abs().max().item() == 0.0


def test_zero_init_logits_match_no_bias_model():
    """End-to-end inertness: feeding the eigenvectors changes nothing at step 0.

    Compared WITHIN one model on purpose — constructing the bias consumes RNG
    draws, so two separately-seeded models get different backbone weights and the
    comparison would be meaningless.
    """
    cfg = GTLMLlamaConfig(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
                          magnetic_struct_dim=_D_STRUCT, bias_self_node=True,
                          graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()

    batch = _batch(_items())
    with torch.no_grad():
        with_mag = model(**batch).logits
        without = model(**{k: v for k, v in batch.items()
                           if not k.startswith("magnetic")}).logits
    assert torch.allclose(with_mag, without, atol=1e-12), \
        (with_mag - without).abs().max().item()


def test_the_trunk_actually_reaches_every_layer():
    """The plumbing test. ``shared_pair_features`` is threaded through a context
    field that no other bias reads, so a missed hop leaves every layer's head with
    ``pair_features=None``, ``forward`` returns None, and the run trains with NO
    bias — the exact silent failure this whole gate is built around. Lifting
    gamma_in must therefore MOVE the logits.
    """
    cfg = GTLMLlamaConfig(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
                          magnetic_struct_dim=_D_STRUCT, bias_self_node=True,
                          graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    assert model.shared_graph_bias_trunk is not None, "no trunk was built"

    batch = _batch(_items())
    with torch.no_grad():
        before = model(**batch).logits
        for mod in model.modules():
            if isinstance(mod, MagneticNonlinearBias):
                mod.gamma_in.normal_(std=0.5)
        after = model(**batch).logits
    assert not torch.allclose(before, after), \
        "lifting gamma_in changed nothing — the head is not wired to the trunk"


def test_parameters_survive_from_pretrained(tmp_path):
    """Every learned tensor must be finite and non-degenerate after `from_pretrained`.

    THE regression test for this arm, and the one whose absence cost a GraphQA
    sweep. Training does not build the model with the constructor — it calls
    `from_pretrained`, which materialises parameters absent from the checkpoint
    and then runs `_init_weights`. That only recognises nn.Linear / nn.Embedding /
    RMSNorm, so a bare nn.Parameter on a custom module keeps whatever the
    materialisation left, and an `nn.init.*` applied AFTER registration does not
    survive. Measured on the real path: W_attn_out, W_attn_in and W_val_in came
    out exactly 0.0 and W_val_out came out NaN, while every direct-constructor
    test passed.

    Why the sweep could not see it: with W_attn == 0 the pool logits are all
    equal, so the learned pool silently degenerates into the uniform ablation —
    the bias is still non-zero, logits still move, and the two arms of the
    experiment become the same model.

    The fixture round-trips a TINY model rather than pulling Llama-1B: the defect
    is in the missing-key materialisation path, which a 2-layer model exercises
    identically.
    """
    kw = dict(magnetic_dim=_MAG_DIM, magnetic_struct_dim=_D_STRUCT,
              bias_self_node=True, graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    # Save a backbone with NO graph bias, then load it into a config that HAS one:
    # every bias tensor is then a missing key, which is exactly the training case.
    GTLMLlamaForCausalLM(GTLMLlamaConfig(**kw)).save_pretrained(tmp_path)
    model = GTLMLlamaForCausalLM.from_pretrained(
        tmp_path, config=GTLMLlamaConfig(magnetic_nonlinear=True, **kw))

    seen = set()
    for name, p in model.named_parameters():
        if not any(t in name for t in ("W_attn", "W_val", "gamma", "trunk")):
            continue
        short = name.rsplit(".", 1)[-1]
        seen.add(short)
        assert torch.isfinite(p).all(), f"{name} holds non-finite values"
        # gamma_in is zero BY DESIGN; every other tensor being all-zero means it
        # was never really initialised.
        if short not in ("gamma_in",) and not short.endswith("bias"):
            assert p.abs().max().item() > 0, \
                f"{name} is identically zero — uninitialised through from_pretrained"

    for want in ("W_attn_out", "W_attn_in", "W_val_out", "W_val_in",
                 "gamma_out", "gamma_in"):
        assert want in seen, f"{want} missing from the loaded model"

    # ...and the pool must not have collapsed to uniform: distinct logits per head
    # are the whole difference between this arm and its own ablation.
    head = next(m for m in model.modules() if isinstance(m, MagneticNonlinearBias))
    assert head.W_attn_out.std().item() > 0, "W_attn_out is constant — pool is uniform"


def test_the_trunk_receives_gradient_through_the_checkpoint_boundary():
    """The load-bearing plumbing property, and the one with no analogue in any
    existing arm.

    E is computed OUTSIDE the gradient-checkpointed region and the pooling that
    consumes it INSIDE, so the trunk's gradient has to arrive through a tensor
    captured in the checkpoint closure. Non-reentrant checkpoint supports that,
    but if it were ever switched to reentrant — or if E were detached anywhere on
    the way — the trunk would silently stop training while every layer's head kept
    learning, and the arm would quietly become "pool a frozen random projection".

    Asserted as an equality against the un-checkpointed model, not merely as
    "non-zero": a wrong-but-non-zero gradient is the failure that looks fine.
    """
    kw = dict(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
              magnetic_struct_dim=_D_STRUCT, bias_self_node=True,
              graph_attn_impl="eager", **_BASE)
    grads = {}
    for ckpt in (True, False):
        torch.manual_seed(0)
        model = GTLMLlamaForCausalLM(
            GTLMLlamaConfig(checkpoint_graph_bias=ckpt, **kw)).double().train()
        for mod in model.modules():                 # lift the gate off zero
            if isinstance(mod, MagneticNonlinearBias):
                with torch.no_grad():
                    mod.gamma_in.normal_(std=0.5)
        model(**_batch(_items())).loss.backward()
        grads[ckpt] = {n: p.grad.clone() for n, p in model.named_parameters()
                       if "shared_graph_bias_trunk" in n and p.grad is not None}

    assert grads[True], "the trunk received NO gradient — it is not training at all"
    assert set(grads[True]) == set(grads[False])
    for n in grads[True]:
        assert grads[True][n].abs().max().item() > 0, f"{n} has an all-zero gradient"
        diff = (grads[True][n] - grads[False][n]).abs().max().item()
        assert diff < 1e-10, f"{n}: checkpoint recompute changed the gradient by {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_flex_matches_eager_for_this_arm():
    """The sweep runs ``graph_attn_impl='flex'``. The arm emits an ordinary dense
    (B,H,N,N) node bias, so flex should need nothing new — but 'should' is what
    this test is for, and the whole sweep runs on the path it checks."""
    from src.utils.text_graph_collator_v2 import GraphCollatorV2 as _C
    kw = dict(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
              magnetic_struct_dim=_D_STRUCT, bias_self_node=True, **_BASE)
    torch.manual_seed(0)
    eager = GTLMLlamaForCausalLM(
        GTLMLlamaConfig(graph_attn_impl="eager", **kw)).cuda().float().eval()
    flex = GTLMLlamaForCausalLM(
        GTLMLlamaConfig(graph_attn_impl="flex", **kw)).cuda().float().eval()
    flex.load_state_dict(eager.state_dict())
    for m in list(eager.modules()) + list(flex.modules()):
        if isinstance(m, MagneticNonlinearBias):
            with torch.no_grad():
                m.gamma_in.normal_(std=0.5)
    flex.load_state_dict(eager.state_dict())

    batch = _C(pad_token_id=0, pad_to_block=True, block_size=128,
               len_buckets=[128], node_buckets=[16])(
        [dict(it) for it in _items(node_counts=(4, 9, 6))])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        oe, of = eager(**batch), flex(**batch)
    ctx = flex.model.layers[0].self_attn._graph_ctx
    assert flex.config._attn_implementation == "gtlm_flex" and ctx.block_mask is not None, \
        "flex path did not run — this would be a trivial eager-vs-eager comparison"
    assert ctx.shared_pair_features is not None, "the trunk did not run under flex"

    mask = batch["attention_mask"].bool()
    diff = (oe.logits[mask] - of.logits[mask]).abs().max().item()
    assert diff < 3e-3, f"flex-vs-eager logit diff {diff}"


# ── 10. No silent no-op (the config gates) ───────────────────────────────────

def test_missing_features_return_none_not_silent_zero():
    _, head = _pair()
    assert head(dtype=torch.float64, device=_DEV, pair_features=None,
                num_nodes=torch.tensor([4])) is None
    trunk, _ = _pair()
    assert trunk(dtype=torch.float64, device=_DEV, magnetic=None,
                 num_nodes=torch.tensor([4])) is None


@pytest.mark.parametrize("clash", [
    {"magnetic": True}, {"magnetic_linear": True}, {"magnetic_hybrid": True},
    {"magnetic_magnitude": True}, {"magnetic_linear_v2": True},
])
def test_model_config_rejects_double_placement(clash):
    with pytest.raises(ValueError):
        GTLMLlamaConfig(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
                        **clash, **_BASE)


def test_model_config_rejects_an_unknown_pool():
    with pytest.raises(ValueError):
        GTLMLlamaConfig(magnetic_nonlinear=True, magnetic_pool="mean", **_BASE)


def test_experiment_config_gates_accept_the_new_arm():
    """Every dataset/collator gate keys on ``uses_magnetic``, not ``magnetic``. A
    gate that missed this arm would emit no eigenvectors, the bias would return
    None, and the run would train with NO graph bias while looking healthy — the
    single most expensive failure mode in this plan, since it reads as a clean
    negative.
    """
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.graphqa.config import RunConfig as GqaCfg

    for Cfg in (KgqaCfg, GqaCfg):
        c = Cfg(magnetic=False, magnetic_nonlinear=True)
        assert c.uses_magnetic, f"{Cfg.__module__}: not in uses_magnetic"
        assert c.collate_magnetic_m == c.magnetic_m
        p = c.bias_params()
        assert p.get("magnetic_nonlinear") is True, p
        # The widths must reach the model config, or it silently builds the head
        # at its own defaults and the run is mislabelled.
        assert p.get("magnetic_struct_dim") == c.magnetic_struct_dim, p
        assert p.get("magnetic_pool") == c.magnetic_pool, p
        assert p.get("magnetic_dim") == c.magnetic_dim, p

    # Same eigenvector bytes as the magnetic arm => same build, or the sweep
    # silently rebuilds a multi-GB dataset per arm.
    assert KgqaCfg(magnetic=False, magnetic_nonlinear=True).data_config_key("webqsp") == \
        KgqaCfg(magnetic=True).data_config_key("webqsp")
    # magnetic_dim is a model width and must NOT fork the cache either: the sweep
    # runs this arm at 64 while every reused baseline was built at 128.
    assert KgqaCfg(magnetic=True, magnetic_dim=64).data_config_key("webqsp") == \
        KgqaCfg(magnetic=True, magnetic_dim=128).data_config_key("webqsp")


def test_experiment_config_rejects_double_placement():
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.graphqa.config import RunConfig as GqaCfg

    for Cfg in (KgqaCfg, GqaCfg):
        with pytest.raises(ValueError):
            Cfg(magnetic=True, magnetic_nonlinear=True).validate()


def test_the_uniform_arm_is_labelled_distinctly():
    """graphqa keys its tables off ``arm``. Two arms sharing one label is the
    `mixed_bias` label trap, which swapped a floor for a ceiling."""
    from src.experiments.graphqa.config import RunConfig as GqaCfg
    learned = GqaCfg(magnetic=False, magnetic_nonlinear=True, magnetic_pool="attn").arm()
    uniform = GqaCfg(magnetic=False, magnetic_nonlinear=True, magnetic_pool="uniform").arm()
    assert learned != uniform, (learned, uniform)
    assert "nonlinear" in learned and "uniform" in uniform


# ── 3. Backward compatibility ────────────────────────────────────────────────

def test_key_off_is_bit_identical():
    """With the key off, nothing about the model changes: no trunk, no new
    parameters. The new BIAS_TYPES entry must not perturb an existing arm."""
    cfg = GTLMLlamaConfig(magnetic=True, magnetic_dim=_MAG_DIM,
                          graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    assert model.shared_graph_bias_trunk is None
    names = {n for n, _ in model.named_parameters()}
    assert not any("gamma_out" in n or "W_val" in n or "trunk" in n for n in names), \
        sorted(n for n in names if "gamma" in n or "W_val" in n or "trunk" in n)


# ── 11. Appended-Q/K parity — what licenses the deferred kernel ──────────────

def test_factorization_reproduces_dense_bias_with_self_node():
    """<z_out[h,i], z_in[g(h),j]> must equal the dense bias on the FULL matrix
    under ``bias_self_node=True`` — the configuration §8 actually runs, and the
    entire premise of the deferred kernel. fp64, CPU, no kernel involved."""
    trunk, head = _pair()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    E = _E(trunk, magnetic, num_nodes)

    dense = head(dtype=torch.float64, device=_DEV, num_nodes=num_nodes, pair_features=E)
    q, k = head.structural_factors(E, num_nodes)
    k = k.repeat_interleave(_H // k.shape[1], dim=1)             # repeat_kv
    recon = torch.einsum("bhid,bhjd->bhij", q, k)

    diff = (dense - recon).abs().max().item()
    assert diff < 1e-10, f"full-matrix factorization mismatch {diff}"


def test_factorization_matches_offdiagonal_when_masked():
    """Under ``bias_self_node=False`` the equivalence can only hold OFF the
    diagonal: an inner product yields <z_out_i, z_in_i> and cannot be forced to
    zero. This is why the arm is run unmasked (§3), and the test pins the cost of
    that choice rather than papering over it."""
    trunk, head = _pair(_CfgMasked)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    E = _E(trunk, magnetic, num_nodes)

    dense = head(dtype=torch.float64, device=_DEV, num_nodes=num_nodes, pair_features=E)
    q, k = head.structural_factors(E, num_nodes)
    recon = torch.einsum("bhid,bhjd->bhij", q, k.repeat_interleave(_H // _H_KV, dim=1))

    n = dense.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=_DEV)
    assert (dense - recon)[..., off].abs().max().item() < 1e-10
    assert dense.diagonal(dim1=-2, dim2=-1).abs().max().item() == 0.0
    assert recon.diagonal(dim1=-2, dim2=-1).abs().max().item() > 1e-6


# ── The GQA group map ────────────────────────────────────────────────────────

def test_key_side_is_per_kv_group_and_matches_repeat_kv():
    """z_in carries H_KV rows, and query head h reads group h // (H_Q/H_KV) —
    HF's ``repeat_kv`` layout. A mismatch here is invisible in the dense path (it
    merely permutes which head gets which key map) and fatal in a kernel."""
    trunk, head = _pair()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    E = _E(trunk, magnetic, num_nodes)
    q, k = head.structural_factors(E, num_nodes)

    assert q.shape[1] == _H and k.shape[1] == _H_KV
    # Distinct key blocks per group, or the per-group capacity is unused.
    assert (k[:, 0] - k[:, 1]).abs().max().item() > 1e-6

    dense = head(dtype=torch.float64, device=_DEV, num_nodes=num_nodes, pair_features=E)
    n_rep = _H // _H_KV
    for h in range(_H):
        ref = torch.einsum("bid,bjd->bij", q[:, h], k[:, h // n_rep])
        assert (dense[:, h] - ref).abs().max().item() < 1e-10, f"head {h} reads the wrong group"


def test_non_divisible_head_counts_raise():
    class Bad(_Cfg):
        num_key_value_heads = 3
    with pytest.raises(ValueError, match="divisible"):
        MagneticNonlinearBias(_H, _HEAD_DIM, Bad())


def test_full_mha_degenerates_to_per_head():
    """Bloom carries no ``num_key_value_heads``; per-group must become per-head
    rather than crash or silently share one key block."""
    _, head = _pair(_CfgMHA)
    assert head.num_kv_heads == _H and head.n_rep == 1


# ── The uniform ablation ─────────────────────────────────────────────────────

def test_uniform_arm_has_no_attention_parameters():
    """Not zeroed — genuinely absent, so the ablation cannot accidentally train a
    pool it is meant not to have."""
    _, head = _pair(_CfgUniform)
    assert head.W_attn_out is None and head.W_attn_in is None
    names = {n for n, _ in head.named_parameters()}
    assert not any("W_attn" in n for n in names), sorted(names)


def test_uniform_pool_weights_are_exactly_one_over_n():
    """Entropy against log n_b is exactly 1.0 for the uniform pool. This is the
    calibration of the diagnostic §5.3 uses to decide whether a learned pool
    actually moved — if the metric cannot report 1.0 where it must, it cannot
    report anything.
    """
    trunk, head = _pair(_CfgUniform)
    head._record_pool_stats = True
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    _bias(trunk, head, magnetic, num_nodes)
    assert set(head._pool_stats) == {"out", "in"}, head._pool_stats
    for d, v in head._pool_stats.items():
        assert abs(v - 1.0) < 1e-9, (d, v)


def test_learned_pool_is_not_uniform():
    """...and the learned pool must report BELOW 1.0 at initialisation. W_attn is
    randomly initialised, so the pool is non-uniform from step 0 — symmetry is
    already broken and the arm does not depend on the gradient to break it."""
    trunk, head = _pair()
    head._record_pool_stats = True
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    _bias(trunk, head, magnetic, num_nodes)
    assert set(head._pool_stats) == {"out", "in"}, head._pool_stats
    for d, v in head._pool_stats.items():
        assert 0.0 < v < 1.0, (d, v)
    # The two directions are separately parameterized, so a single reported number
    # would just be whichever pool ran last — the reason the stat is keyed at all.
    assert head._pool_stats["out"] != head._pool_stats["in"]


# ── The §2.4 range bound ─────────────────────────────────────────────────────

def test_bias_is_bounded_by_the_gain_product():
    """|b| <= max|gamma_out| * max|gamma_in| * d_struct.

    RMSNorm leaves ||z||_2 = sqrt(d_struct) rather than 1, so unlike the magnitude
    channel's single scalar this bound is a product — which is exactly why it is
    logged. A run that diverges must be diagnosable from the logged number.
    """
    trunk, head = _pair()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    b = _bias(trunk, head, magnetic, num_nodes)
    assert b.abs().max().item() <= head.gain_bound + 1e-9


def test_bias_is_invariant_to_the_scale_of_the_value_projection():
    """Rescaling W_val leaves the bias unchanged — the growth mode that diverged.

    This is RMSNorm's real purchase and the property MIXED_BIAS.md §5.8 had to
    engineer by hand: the factor between the pool and the norm enters the bias at
    degree 0, so the head cannot become quartic in its own trunk the way arms 3/4
    were. The scaling is applied to W_val rather than to E because the pool
    weights are deliberately NOT scale-invariant (softmax of a scaled logit is a
    different distribution), so W_val is exactly the factor the norm is meant to
    absorb.

    Both factors are lifted well clear of RMSNorm's eps first. That is not a
    convenience: the invariance is exact only once mean(z^2) >> eps, and the test
    would otherwise be measuring the eps floor instead of the property. The
    companion test below pins what happens in that other regime.
    """
    trunk, head = _pair()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    E = _E(trunk, magnetic, num_nodes)
    kw = dict(dtype=torch.float64, device=_DEV, num_nodes=num_nodes, pair_features=E)

    with torch.no_grad():                       # clear of eps on both sides
        head.W_val_out.mul_(1e4)
        head.W_val_in.mul_(1e4)
    before = head(**kw)
    with torch.no_grad():
        head.W_val_out.mul_(7.0)
        head.W_val_in.mul_(13.0)
    after = head(**kw)

    assert (before - after).abs().max().item() < 1e-9, \
        (before - after).abs().max().item()


def test_the_eps_floor_fails_towards_zero_not_towards_noise():
    """Below RMSNorm's eps the invariance stops being exact — it degrades toward
    an identically-zero bias, never toward a large one.

    Worth pinning because it is the direction that matters. Under GROWTH the norm
    is exact and the bias stays bounded by the gains; under COLLAPSE the eps floor
    shrinks the bias toward 0. Neither regime can produce the unbounded product
    that killed arms 3/4, and this is the assertion that says so.
    """
    trunk, head = _pair()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    E = _E(trunk, magnetic, num_nodes)
    kw = dict(dtype=torch.float64, device=_DEV, num_nodes=num_nodes, pair_features=E)

    with torch.no_grad():
        head.W_val_out.mul_(1e-6)
        head.W_val_in.mul_(1e-6)
    collapsed = head(**kw).abs().max().item()
    assert collapsed < 1e-6, collapsed


# ── 13. Save / load round-trip ───────────────────────────────────────────────

def test_bias_parameters_round_trip(tmp_path):
    """The trunk AND both pooling heads must survive bias_parameters.pt — the file
    the LoRA adapter does not contain. Nothing new is written for this
    (``active_params=("graph_bias",)`` is a substring match), which is exactly why
    the trunk's attribute name has to contain ``graph_bias`` and why that is
    asserted rather than assumed.

    gamma_in especially: it is the gate, so a model that dropped it on load would
    emit an identically-zero bias and every logit comparison would pass.
    """
    from src.models.io import save_bias_parameters, load_bias_parameters

    cfg = GTLMLlamaConfig(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
                          magnetic_struct_dim=_D_STRUCT, bias_self_node=True,
                          graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    for mod in model.modules():
        if isinstance(mod, MagneticNonlinearBias):
            with torch.no_grad():
                mod.gamma_in.normal_(std=0.3)      # the gate: 0 => zero bias
                mod.W_val_out.normal_(std=0.3)

    save_bias_parameters(model, str(tmp_path), ["graph_bias"])
    saved = torch.load(tmp_path / "bias_parameters.pt", map_location="cpu",
                       weights_only=True)
    for want in ("shared_graph_bias_trunk.lambda_lin.weight",
                 "shared_graph_bias_trunk.deep_set.0.weight",
                 "shared_graph_bias_trunk.proj.0.weight",
                 "W_attn_out", "W_val_out", "W_attn_in", "W_val_in",
                 "gamma_out", "gamma_in"):
        assert any(want in k for k in saved), (want, sorted(saved))

    torch.manual_seed(0)                            # identical BACKBONE
    fresh = GTLMLlamaForCausalLM(cfg).double().eval()
    batch = _batch(_items())
    with torch.no_grad():
        before = fresh(**batch).logits
    load_bias_parameters(fresh, str(tmp_path))
    with torch.no_grad():
        after = fresh(**batch).logits
        target = model(**batch).logits

    assert not torch.allclose(before, after), "load was a no-op"
    assert torch.allclose(after, target, atol=1e-10), \
        (after - target).abs().max().item()


# ── 7. Bias regularization ───────────────────────────────────────────────────

def test_weight_decay_classification():
    """The trainer's rule is ``p.ndim >= 2`` (text_graph_trainer_v2.py:82).

    Recorded rather than argued: W_val is 3-D and decays, W_attn and both gammas
    are 2-D and decay too. Every sweep here runs ``bias_weight_decay: 0.0``, so
    nothing depends on the classification — but a later run that turns decay on
    would be decaying the gate, and this is where that is visible.
    """
    cfg = GTLMLlamaConfig(magnetic_nonlinear=True, magnetic_dim=_MAG_DIM,
                          magnetic_struct_dim=_D_STRUCT, graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg)

    decayed = {n for n, p in model.named_parameters() if p.ndim >= 2}
    for n, p in model.named_parameters():
        if not any(t in n for t in ("W_attn", "W_val", "gamma")):
            continue
        assert (n in decayed) == (p.ndim >= 2), (n, p.ndim)
    assert any("gamma_in" in n for n in decayed)


# ── registration ─────────────────────────────────────────────────────────────

def test_registered_in_bias_types():
    assert MagneticNonlinearBias in BIAS_TYPES
    assert MagneticNonlinearBias.config_key == "magnetic_nonlinear"
    # The TRUNK must NOT be registered: it emits (B,N,N,m) pair features, not a
    # (B,H,N,N) bias, so summing it into the shared node bias is a shape error.
    assert MagneticPairTrunk not in BIAS_TYPES
    # The head owns no spectral machinery — that all lives once, on the trunk.
    _, head = _pair()
    for attr in ("proj", "deep_set", "lambda_lin"):
        assert not hasattr(head, attr), attr
    # ...and the trunk owns no final head: MagneticBias's proj[2] is exactly the
    # layer this arm replaces, so building it would put dead weights in the
    # checkpoint and in the weight-decay group.
    trunk, _ = _pair()
    assert len(trunk.proj) == 2 and isinstance(trunk.proj[1], torch.nn.SiLU)
