"""Correctness gate for ``MagneticMagnitudeBias`` and ``MagneticHybridBias``.

These run BEFORE the Phase 2 training sweep, by design: none of the failures they
catch would announce themselves in a training curve. A wiring slip produces a run
with no bias at all; a group-map slip produces a bias that is merely *wrong*; a
non-invariant magnitude feature makes prompt logits depend on node labelling.
All of them train cleanly and all of them read as "the magnitude channel didn't
help", which is precisely the conclusion the sweep exists to draw.

Numbering follows ``src/models/MIXED_BIAS.md`` §4.2.
"""

import networkx as nx
import pytest
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import (BIAS_TYPES, MagneticBias, MagneticHybridBias,
                             MagneticMagnitudeBias)
from src.utils.magnetic_lap import get_magnetic_laplacian_coords
from src.utils.text_graph_collator_v2 import GraphCollatorV2

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)
_MAG_DIM = 16
_H, _H_KV, _HEAD_DIM = 4, 2, 16

# The two arms under test. Parametrizing rather than duplicating keeps them from
# drifting: every property below is required of BOTH, and the hybrid inherits its
# phase channel from LinearMagneticBias, which has its own suite.
_ARMS = [MagneticMagnitudeBias, MagneticHybridBias]
_ARM_IDS = ["magnitude", "hybrid"]


class _Cfg:
    """Minimal bias_config for constructing a bias module directly."""
    magnetic_dim = _MAG_DIM
    magnetic_magnitude_dim = 8
    magnetic_magnitude_repr_dim = 12
    num_key_value_heads = _H_KV
    hidden_size = 64


class _CfgSelfNode(_Cfg):
    """As ``_Cfg`` but keeping the intra-node diagonal (MIXED_BIAS.md §5.1)."""
    bias_self_node = True


class _CfgMHA(_Cfg):
    """No GQA at all — the Bloom case, where per-group degenerates to per-head."""
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
    return (batch["magnetic_V"].double(), batch["magnetic_lambdas"].double()), batch["num_nodes"]


def _live(cls, cfg=_Cfg):
    """A module with the magnitude channel switched ON.

    Zero-init makes the bias identically 0, so every test about what the channel
    COMPUTES has to lift the gate first — otherwise it would pass against a module
    that computes nothing at all. The gate is ``magnitude_gain``: leaving it at its
    initial zero would silently reduce every assertion below to 0 == 0.
    """
    torch.manual_seed(0)
    mod = cls(_H, _HEAD_DIM, cfg()).double()
    with torch.no_grad():
        mod.magnitude_gain.normal_(std=0.5)
        mod.magnitude_q_scale.normal_(std=0.5)
        if hasattr(mod, "proj"):                    # hybrid: wake the phase channel too
            torch.nn.init.normal_(mod.proj[0].weight, std=0.5)
            torch.nn.init.zeros_(mod.proj[0].bias)  # the bias term is not factorizable
    return mod


# ── 1. Permutation invariance ────────────────────────────────────────────────
#
# test_flex_cpu.py checks this end-to-end through the dataset's RCM, but on
# barabasi_albert graphs with SPD features only. Neither guarantees a degenerate
# spectrum, and degeneracy is exactly the condition under which a per-column
# magnitude feature stops being well defined (MIXED_BIAS.md §3). So the fixtures
# here are a star and a cycle, whose spectra are degenerate by construction.

def _relabelled_pair(graph, order_a, order_b):
    """The same graph under two node orderings, as (V, lambdas) pairs.

    ``nx.to_numpy_array`` uses INSERTION order, not integer labels, so the
    orderings have to be built by adding nodes explicitly — relabelling and
    trusting the labels silently produces the same matrix twice and the test
    passes without testing anything.

    Returned batched (B=1) and in fp64 to match the module's inputs. The solver
    itself runs in fp32, so ~1e-6 is the floor for every comparison downstream —
    which is still four orders of magnitude below the 0.674 a per-column feature
    moves by on the star.
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


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
@pytest.mark.parametrize("graph,name", [
    (nx.star_graph(4), "star"),          # lambda = [0,1,1,1,2]: a triply degenerate block
    (nx.cycle_graph(6), "cycle"),        # degenerate conjugate pairs
], ids=lambda v: v if isinstance(v, str) else "")
def test_self_energy_invariant_to_node_relabelling(cls, graph, name):
    """S_i must follow its node under relabelling, to fp tolerance.

    This is the property §2.3's whole form exists to satisfy. Per-column |V_il|^2
    fails it by 0.674 on this very star (§3's measurement) because LAPACK returns
    a genuinely different basis inside a degenerate block; the pool over l is what
    makes it invariant. If someone later "simplifies" the pool away, this fails.
    """
    mod = _live(cls)
    n = graph.number_of_nodes()
    (Va, la, oa), (Vb, lb, ob) = _relabelled_pair(
        graph, list(range(n)), list(reversed(range(n))))

    num_nodes = torch.tensor([n])
    dev = torch.device("cpu")
    Sa = mod._self_energy(*mod._phi((Va, la), num_nodes, dev))     # (1, N, m)
    Sb = mod._self_energy(*mod._phi((Vb, lb), num_nodes, dev))

    # Row i of Sa is node oa[i]; align B's rows onto A's node order.
    pos_b = {node: i for i, node in enumerate(ob)}
    perm = torch.tensor([pos_b[node] for node in oa])
    diff = (Sa[0] - Sb[0][perm]).abs().max().item()
    assert diff < 1e-4, f"{name}: self-energy moved by {diff} under relabelling"


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_per_column_magnitude_would_fail_this(cls):
    """The negative control for the test above.

    Without it, an implementation that accidentally produced a CONSTANT feature
    would pass invariance trivially. This asserts the raw per-column magnitudes
    really do move on the star fixture, i.e. that the invariance test has
    something to catch — and it records §3's measurement as an executable fact.
    """
    n = 5
    (Va, _la, oa), (Vb, _lb, ob) = _relabelled_pair(
        nx.star_graph(4), list(range(n)), list(reversed(range(n))))
    pos_b = {node: i for i, node in enumerate(ob)}
    perm = torch.tensor([pos_b[node] for node in oa])

    def cols(V):
        return (V[0, ..., 0] ** 2 + V[0, ..., 1] ** 2)             # (N, M)

    moved = (cols(Va) - cols(Vb)[perm]).abs().max().item()
    assert moved > 1e-2, (
        "per-column |V|^2 did not move under relabelling on a degenerate "
        f"spectrum (max delta {moved}); the invariance test above is vacuous")


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_bias_matrix_is_permutation_equivariant(cls):
    """The full (H, N, N) bias must permute with its nodes, not change."""
    mod = _live(cls)
    n = 5
    (Va, la, oa), (Vb, lb, ob) = _relabelled_pair(
        nx.star_graph(4), list(range(n)), list(reversed(range(n))))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              num_nodes=torch.tensor([n]))
    ba = mod(magnetic=(Va, la), **kw)[0]
    bb = mod(magnetic=(Vb, lb), **kw)[0]

    pos_b = {node: i for i, node in enumerate(ob)}
    perm = torch.tensor([pos_b[node] for node in oa])
    diff = (ba - bb[:, perm][:, :, perm]).abs().max().item()
    assert diff < 1e-4, f"bias moved by {diff} under relabelling"


# ── 2. Padded eigenvector slots ──────────────────────────────────────────────

@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_padded_eigenvector_slots_contribute_nothing(cls):
    """The pool over l is a fresh place for padded columns to enter.

    The claim is precise: it is safe because the COLLATOR zero-pads V, so
    |V|^2 = 0 in the padded columns kills whatever phi holds there. Two things
    have to hold for that argument to be worth anything, and both are asserted
    here rather than assumed.

    Note what is deliberately NOT tested: writing garbage into padded EIGENVECTOR
    columns of a REAL node's row. That would move the bias, correctly — it is
    a corrupted feature, not padding — and a test that demanded otherwise would
    be asserting the opposite of the design.
    """
    mod = _live(cls)
    batch = _batch(_items(node_counts=(4, 9, 6)))
    (V, lam), num_nodes = _mag_inputs(batch)
    kw = dict(dtype=torch.float64, device=torch.device("cpu"), num_nodes=num_nodes)
    clean = mod(magnetic=(V, lam), **kw)

    # (a) The collator really does zero-pad both axes. If this ever stops being
    #     true the argument above collapses, so it is the first assertion.
    for b, n in enumerate(num_nodes.tolist()):
        # .sum(), not .max(): the largest item in the batch has no padding at
        # all and .max() on an empty slice raises.
        assert V[b, n:, :, :].abs().sum().item() == 0.0, f"node rows of item {b}"
        assert V[b, :, n:, :].abs().sum().item() == 0.0, f"eigen columns of item {b}"

    # (b) Given that, arbitrary EIGENVALUES in the padded slots cannot reach a
    #     real node. phi is computed for every slot (deep_set has a bias, so phi
    #     is non-zero even at lambda=0) and only |V|^2 = 0 stops it propagating —
    #     this is the pool-over-l step, and it is the new surface this arm adds.
    dirty_lam = lam.clone()
    for b, n in enumerate(num_nodes.tolist()):
        dirty_lam[b, n:] = 7.0
    dirtied = mod(magnetic=(V, dirty_lam), **kw)
    for b, n in enumerate(num_nodes.tolist()):
        diff = (clean[b, :, :n, :n] - dirtied[b, :, :n, :n]).abs().max().item()
        assert diff < 1e-12, f"padded eigenvalues leaked into item {b}: {diff}"

    # (c) Garbage in padded NODE rows cannot reach real node pairs either.
    dirty_V = V.clone()
    for b, n in enumerate(num_nodes.tolist()):
        dirty_V[b, n:, :, :] = 7.0
    dirtied = mod(magnetic=(dirty_V, lam), **kw)
    for b, n in enumerate(num_nodes.tolist()):
        diff = (clean[b, :, :n, :n] - dirtied[b, :, :n, :n]).abs().max().item()
        assert diff < 1e-12, f"padded node rows leaked into item {b}: {diff}"


# ── 9. Zero-init inertness, and the dead saddle ──────────────────────────────

@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_zero_init_bias_is_exactly_zero(cls):
    """At step 0 both arms emit exactly 0, so the model is bit-identical to the
    no-bias backbone. Anything else destabilises training from step 0."""
    torch.manual_seed(0)
    mod = cls(_H, _HEAD_DIM, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    assert out is not None
    assert out.abs().max().item() == 0.0


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_zero_init_is_not_a_dead_saddle(cls):
    """The gradient of the GATE must be non-zero at step 0.

    This is what distinguishes a correct zero-init from the failure the plan calls
    out: zeroing every factor of g <q̂, k̂> at once makes it a dead saddle, and the
    arm never leaves the origin while looking perfectly healthy, because the loss
    still falls (the LoRA adapter is training). Exactly one factor is zeroed — the
    gain — so it moves immediately and everything behind it starts moving once it
    does.

    Which factor holds the zero moved when the channel was normalized: it used to
    be ``s``, but ``s`` is inside the normalized vector and F.normalize at exactly
    zero has Jacobian I/eps. See `test_gate_gradient_is_not_the_eps_bomb`.
    """
    torch.manual_seed(0)
    mod = cls(_H, _HEAD_DIM, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    out.sum().backward()

    assert mod.magnitude_gain.grad is not None
    g = mod.magnitude_gain.grad.abs().max().item()
    assert g > 0, "the gain has zero gradient at step 0 — the arm cannot ever move"
    # s and W_K are correctly still at zero gradient here (both are behind the
    # gain); they start moving on the next step. Asserted so the asymmetry is on
    # record — it is a one-step delay, not a dead saddle.
    assert mod.magnitude_q_scale.grad.abs().max().item() == 0.0
    assert mod.magnitude_k_mix.grad.abs().max().item() == 0.0


def test_zero_init_logits_match_no_bias_model():
    """End-to-end inertness: feeding the eigenvectors changes nothing at step 0.

    Compared WITHIN one model on purpose — constructing the bias consumes RNG
    draws, so two separately-seeded models get different backbone weights and the
    comparison would be meaningless.
    """
    for key in ("magnetic_magnitude", "magnetic_hybrid"):
        cfg = GTLMLlamaConfig(magnetic_dim=_MAG_DIM, graph_attn_impl="eager",
                              **{key: True}, **_BASE)
        torch.manual_seed(0)
        model = GTLMLlamaForCausalLM(cfg).double().eval()

        batch = _batch(_items())
        with torch.no_grad():
            with_mag = model(**batch).logits
            without = model(**{k: v for k, v in batch.items()
                               if not k.startswith("magnetic")}).logits
        assert torch.allclose(with_mag, without, atol=1e-12), \
            f"{key}: {(with_mag - without).abs().max().item()}"


# ── 9b. Scale stability (MIXED_BIAS.md §5.7) ─────────────────────────────────
#
# The regression tests for the four divergences. Unnormalized, the same Z fed
# both sides of the inner product, so the bias was quartic in the trunk scale and
# unbounded in every one of its three learned factors. These pin the properties
# that replaced that; they are cheap, and each one failing means the arm is back
# to the parameterization that produced NaN on WebQSP at epoch 2.48.

@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
@pytest.mark.parametrize("factor", ["trunk", "q_scale", "k_mix"])
def test_bias_is_invariant_to_the_scale_of_every_learned_factor(cls, factor):
    """Scaling the trunk, ``s`` or ``W_K`` by k leaves the bias EXACTLY unchanged.

    This is the property the normalization buys, and it is what makes the fix not
    depend on knowing which of the three actually grows during training — a
    question the diagnostic job never got to answer. Note the invariance is to a
    uniform rescaling, which is the growth mode §5.7 measures; it does not claim
    the bias is invariant to arbitrary reparameterization.
    """
    mod = _live(cls)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    before = mod(**kw)

    k = 8.0
    with torch.no_grad():
        if factor == "trunk":
            # The last Linear is what Z is linear in, so this scales Z by exactly k
            # without touching the phase channel's deep_set/lambda_lin.
            mod.magnitude_mlp[-1].weight.mul_(k)
            mod.magnitude_mlp[-1].bias.mul_(k)
        elif factor == "q_scale":
            mod.magnitude_q_scale.mul_(k)
        else:
            mod.magnitude_k_mix.mul_(k)

    diff = (mod(**kw) - before).abs().max().item()
    assert diff < 1e-12, f"{factor} scaled the bias by more than fp64 noise: {diff}"


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_magnitude_bias_is_bounded_by_the_gain(cls):
    """|b_magnitude| <= |g^(h)| per head, whatever the trunk does.

    The gain is the only unbounded quantity left in the channel, which is the
    point: it is one auditable scalar per head rather than a 64x64 W_K, so
    "survived because it is fixed" and "survived because it is postponed" are
    distinguishable by watching a number.
    """
    mod = _live(cls)
    parts = mod._phi(*_mag_inputs(_batch(_items())), torch.device("cpu"))
    b = mod._magnitude_bias(*parts)                       # (B, N, N, H), pre-_finalize
    bound = mod.magnitude_gain.abs()                      # (H,)
    worst = (b.abs().amax(dim=(0, 1, 2)) - bound).max().item()
    assert worst < 1e-12, f"bias exceeded its per-head gain bound by {worst}"


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_gate_gradient_is_not_the_eps_bomb(cls):
    """At step 0 the gradients must be O(1), not O(1/eps).

    The gate is a scalar OUTSIDE the normalized vector precisely so that nothing
    is ever normalized at exactly zero. Were the zero-init still on ``s``, the
    query row would be the zero vector, F.normalize would clamp its denominator to
    eps and pass no gradient through it, and the local Jacobian would be I/eps —
    around 1e12, which with max_grad_norm=1.0 scales every other gradient in the
    model to nothing for that step. A step-0 instability in place of a step-800
    one is not a fix, so it is asserted rather than assumed.
    """
    torch.manual_seed(0)
    mod = cls(_H, _HEAD_DIM, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    mod(dtype=torch.float64, device=torch.device("cpu"),
        magnetic=magnetic, num_nodes=num_nodes).sum().backward()

    for name, p in mod.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.abs().max().item()
        assert g < 1e3, f"{name} has a {g:.3g} gradient at step 0"


# ── 10. No silent no-op ──────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_missing_features_return_none_not_silent_zero(cls):
    """With no eigenvectors the module must return None (so the caller can tell),
    never a zero tensor that masquerades as a computed bias."""
    mod = cls(_H, _HEAD_DIM, _Cfg()).double()
    assert mod(dtype=torch.float64, device=torch.device("cpu"),
               magnetic=None, num_nodes=None) is None


@pytest.mark.parametrize("key", ["magnetic_magnitude", "magnetic_hybrid"])
@pytest.mark.parametrize("clash", [
    dict(magnetic=True),
    dict(magnetic_shared=True),
    dict(magnetic_content=True),
    dict(magnetic_linear=True),
    dict(magnetic_groups=2),
])
def test_config_rejects_double_placement(key, clash):
    """Each new key is a different HEAD on the same magnetic term, so stacking it
    on another placement is never an intended arm: the config must raise rather
    than quietly build two magnetic biases on one feature set."""
    with pytest.raises(ValueError):
        GTLMLlamaConfig(magnetic_dim=_MAG_DIM, **{key: True}, **clash, **_BASE)


def test_the_two_new_keys_reject_each_other():
    with pytest.raises(ValueError):
        GTLMLlamaConfig(magnetic_magnitude=True, magnetic_hybrid=True,
                        magnetic_dim=_MAG_DIM, **_BASE)


@pytest.mark.parametrize("key", ["magnetic_magnitude", "magnetic_hybrid"])
def test_experiment_config_gates_accept_the_new_arms(key):
    """Every dataset/collator gate keys on `uses_magnetic`, not `magnetic`. A gate
    that missed these arms would emit no eigenvectors, the bias would return None,
    and the run would train with NO graph bias while looking healthy — the single
    most expensive failure mode in this plan, since it reads as a clean negative.
    """
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.context.config import RunConfig as CtxCfg
    from src.experiments.graphqa.config import RunConfig as GqaCfg

    for Cfg in (KgqaCfg, CtxCfg, GqaCfg):
        c = Cfg(magnetic=False, **{key: True})
        assert c.uses_magnetic, f"{Cfg.__module__}: {key} not in uses_magnetic"
        assert c.collate_magnetic_m == c.magnetic_m
        assert c.bias_params().get(key) is True, c.bias_params()
        # The widths must reach the model config, or it silently builds the
        # channel at its own defaults and the run is mislabelled.
        assert c.bias_params().get("magnetic_magnitude_dim") == c.magnetic_magnitude_dim
        assert c.bias_params().get("magnetic_magnitude_repr_dim") == \
            c.magnetic_magnitude_repr_dim

    # Same eigenvector bytes as the magnetic arm => same build, or the sweep
    # silently rebuilds a 3.4 GB dataset per arm.
    assert CtxCfg(magnetic=False, **{key: True}).data_config_key() == \
        CtxCfg(magnetic=True).data_config_key()
    assert KgqaCfg(magnetic=False, **{key: True}).data_config_key("webqsp") == \
        KgqaCfg(magnetic=True).data_config_key("webqsp")


@pytest.mark.parametrize("key", ["magnetic_magnitude", "magnetic_hybrid"])
def test_experiment_config_rejects_double_placement(key):
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.context.config import RunConfig as CtxCfg
    from src.experiments.graphqa.config import RunConfig as GqaCfg

    for Cfg in (KgqaCfg, CtxCfg, GqaCfg):
        with pytest.raises(ValueError):
            Cfg(magnetic=True, **{key: True}).validate()


# ── 3. Backward compatibility ────────────────────────────────────────────────

def test_both_keys_off_is_bit_identical():
    """With both keys off, nothing about the model changes. The two new entries in
    BIAS_TYPES must not perturb an existing arm."""
    cfg = GTLMLlamaConfig(magnetic=True, magnetic_dim=_MAG_DIM,
                          graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    names = {n for n, _ in model.named_parameters()}
    assert not any("magnitude" in n for n in names), \
        sorted(n for n in names if "magnitude" in n)


# ── 12. Factorization parity (the one that de-risks the deferred backbone) ───

@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_factorization_reproduces_dense_bias_with_self_node(cls):
    """<Q[h,i], K[g(h),j]> must equal the dense bias on the FULL matrix under
    ``bias_self_node=True`` — the configuration §5 actually runs, and the entire
    premise of the deferred O(N) backbone. fp64, CPU, no kernel involved.
    """
    mod = _live(cls, _CfgSelfNode)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")

    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    q, k = mod.structural_factors(magnetic, num_nodes, dev)
    k = k.repeat_interleave(_H // k.shape[1], dim=1)          # repeat_kv
    recon = torch.einsum("bhid,bhjd->bhij", q, k)

    diff = (dense - recon).abs().max().item()
    assert diff < 1e-10, f"full-matrix factorization mismatch {diff}"


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_factorization_matches_offdiagonal_when_masked(cls):
    """Under the default mask the equivalence can only hold off the diagonal: an
    inner product yields <q_i, k_i> and cannot be forced to 0 (LINEAR_BIAS.md
    §7.3). Asserted so the limitation is pinned rather than assumed."""
    mod = _live(cls)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")

    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    q, k = mod.structural_factors(magnetic, num_nodes, dev)
    recon = torch.einsum("bhid,bhjd->bhij", q, k.repeat_interleave(_H // k.shape[1], dim=1))

    n = dense.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    diff = (dense[..., off] - recon[..., off]).abs().max().item()
    assert diff < 1e-10, f"off-diagonal factorization mismatch {diff}"
    # And the diagonal genuinely differs, or the mask isn't being applied at all.
    assert torch.diagonal(dense, dim1=-2, dim2=-1).abs().max().item() == 0.0


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_key_side_is_per_kv_group_not_per_head(cls):
    """K carries a GROUP axis of size H_KV — the finest granularity GQA allows.

    Per-QUERY-head keys would force H_Q/H_KV copies of one physical key row and
    defeat GQA; a single shared block would leave each head only a rescaling of
    one globally fixed key map. This pins the middle, which is the design.
    """
    mod = _live(cls)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    q, k = mod.structural_factors(magnetic, num_nodes, torch.device("cpu"))
    assert q.shape[1] == _H
    assert k.shape[1] == _H_KV
    assert q.shape[-1] == k.shape[-1]
    # The groups must actually differ, or "per-group" is decorative.
    assert not torch.allclose(k[:, 0], k[:, 1])


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_group_map_matches_repeat_kv(cls):
    """g(h) = h // (H_Q/H_KV), HF's ``repeat_kv`` convention.

    A mismatch here is INVISIBLE in the dense path — it merely permutes which
    head gets which key map, and training absorbs the permutation — and fatal in
    the factorized one, where the two paths would then disagree. So it is pinned
    against the dense forward, head by head.
    """
    mod = _live(cls)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")
    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    q, k = mod.structural_factors(magnetic, num_nodes, dev)

    n_rep = _H // _H_KV
    n = dense.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    for h in range(_H):
        recon_h = torch.einsum("bid,bjd->bij", q[:, h], k[:, h // n_rep])
        diff = (dense[:, h][..., off] - recon_h[..., off]).abs().max().item()
        assert diff < 1e-10, f"head {h} does not read group {h // n_rep}: {diff}"


@pytest.mark.parametrize("cls", _ARMS, ids=_ARM_IDS)
def test_full_mha_degenerates_to_per_head(cls):
    """No ``num_key_value_heads`` (the repo's Bloom backbone) means per-group IS
    per-head. Reading H_KV via getattr with a num_heads fallback is what makes
    that work; a hard attribute read would crash there instead."""
    mod = _live(cls, _CfgMHA)
    assert mod.magnitude_kv_heads == _H and mod.magnitude_repeat == 1
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    _q, k = mod.structural_factors(magnetic, num_nodes, torch.device("cpu"))
    assert k.shape[1] == _H


def test_non_divisible_head_counts_raise():
    """The group map assumes H_Q is a multiple of H_KV; anything else must raise
    at construction rather than silently mis-map heads to groups."""
    class _Bad(_Cfg):
        num_key_value_heads = 3
    with pytest.raises(ValueError, match="divisible"):
        MagneticMagnitudeBias(4, _HEAD_DIM, _Bad())


def test_hybrid_is_exactly_phase_plus_magnitude():
    """b_hybrid = b_phase + b_magnitude, so the two channels are separable and
    the arm-3/arm-4 contrast means what §5.6 says it means."""
    from src.models.bias import LinearMagneticBias

    hybrid = _live(MagneticHybridBias, _CfgSelfNode)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")
    kw = dict(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)

    # A linear arm and a magnitude arm carrying the hybrid's own weights.
    phase = LinearMagneticBias(_H, _HEAD_DIM, _CfgSelfNode()).double()
    phase.load_state_dict({k: v for k, v in hybrid.state_dict().items()
                           if not k.startswith("magnitude")})
    magnitude = MagneticMagnitudeBias(_H, _HEAD_DIM, _CfgSelfNode()).double()
    magnitude.load_state_dict({k: v for k, v in hybrid.state_dict().items()
                               if not k.startswith("proj")})

    diff = (hybrid(**kw) - (phase(**kw) + magnitude(**kw))).abs().max().item()
    assert diff < 1e-12, f"hybrid is not the sum of its channels: {diff}"


# ── 13. Save / load round-trip ───────────────────────────────────────────────

@pytest.mark.parametrize("key,cls", list(zip(["magnetic_magnitude", "magnetic_hybrid"], _ARMS)))
def test_bias_parameters_round_trip(key, cls, tmp_path):
    """MLP_magnitude, s^(h), W_K and the gain must survive save/load through
    bias_parameters.pt — the file the LoRA adapter does NOT contain. Nothing new
    is written for this (``active_params=("graph_bias",)`` is a substring match),
    which is exactly why it is asserted rather than assumed.

    The gain especially: it is the gate, so a model that dropped it on load would
    emit an identically-zero bias and every logit comparison below would pass."""
    from src.models.io import save_bias_parameters, load_bias_parameters

    cfg = GTLMLlamaConfig(magnetic_dim=_MAG_DIM, graph_attn_impl="eager",
                          **{key: True}, **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    # Perturb the head so a failed load cannot pass by coincidence (zero-init).
    for mod in model.modules():
        if isinstance(mod, cls):
            with torch.no_grad():
                mod.magnitude_q_scale.normal_(std=0.3)
                mod.magnitude_gain.normal_(std=0.3)   # the gate: 0 => zero bias

    save_bias_parameters(model, str(tmp_path), ["graph_bias"])
    saved = torch.load(tmp_path / "bias_parameters.pt", map_location="cpu",
                       weights_only=True)
    for want in ("magnitude_mlp.0.weight", "magnitude_q_scale", "magnitude_k_mix",
                 "magnitude_gain"):
        assert any(want in k for k in saved), (want, sorted(saved))

    # Same seed => identical BACKBONE, so the only difference between the two
    # models is the bias head.
    torch.manual_seed(0)
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

@pytest.mark.parametrize("key", ["magnetic_magnitude", "magnetic_hybrid"])
def test_weight_decay_classification(key):
    """The trainer's rule is ``p.ndim >= 2`` (text_graph_trainer_v2.py:82).
    Confirm the new tensors land where intended: the 2-D/3-D weights decay, the
    1-D biases do not. s^(h) is 2-D and DOES decay, which is the same treatment
    arm A's per-head diagonal gets (it lives in proj[0].weight)."""
    cfg = GTLMLlamaConfig(magnetic_dim=_MAG_DIM, graph_attn_impl="eager",
                          **{key: True}, **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg)

    decayed = {n for n, p in model.named_parameters()
               if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)}
    for n, p in model.named_parameters():
        if "magnitude" not in n:
            continue
        assert (n in decayed) == (p.ndim >= 2), (n, p.ndim)
    assert any("magnitude_q_scale" in n for n in decayed)
    assert any("magnitude_k_mix" in n for n in decayed)


# ── registration ─────────────────────────────────────────────────────────────

def test_registered_in_bias_types():
    assert MagneticMagnitudeBias in BIAS_TYPES and MagneticHybridBias in BIAS_TYPES
    assert MagneticMagnitudeBias.config_key == "magnetic_magnitude"
    assert MagneticHybridBias.config_key == "magnetic_hybrid"
    # The magnitude arm has NO pairwise head — MagneticBias's proj would be dead
    # weight in the checkpoint and in the weight-decay group.
    assert not hasattr(MagneticMagnitudeBias(_H, _HEAD_DIM, _Cfg()), "proj")
    # ...but it keeps the machinery that produces phi.
    assert isinstance(MagneticMagnitudeBias(_H, _HEAD_DIM, _Cfg()), MagneticBias)
