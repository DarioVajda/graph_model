"""Correctness gate for ``GatedLinearMagneticBias`` (``--magnetic-linear-v2``).

Same rationale as ``test_linear_magnetic_bias.py``: none of the failures caught
here announce themselves in a training curve. A missed wiring gate produces a run
with NO bias, a wrong gate axis produces a bias that is merely different — and
both train cleanly and read as "the gate did not help", which is precisely the
conclusion the sweep exists to draw — and did draw, so these tests are what
separate "the gate did not help" from "the gate was never wired".

The load-bearing test is ``test_equals_linear_arm_at_init``: this arm is defined
as arm 2 plus a gate that starts at exactly 1, so if the two ever disagree at
initialisation the arm is not what the plan says it is and every comparison
against arm 2's numbers is void.
"""

import pytest
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import BIAS_TYPES, GatedLinearMagneticBias, LinearMagneticBias
from src.utils.text_graph_collator_v2 import GraphCollatorV2

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)
_MAG_DIM = 16


class _Cfg:
    magnetic_dim = _MAG_DIM
    magnetic_gate_repr_dim = 24
    hidden_size = 64


class _CfgSelfNode(_Cfg):
    bias_self_node = True


def _items(node_counts=(4, 9, 6), seed=0, m=None):
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


def _excite(mod, seed=0, gate_std=0.5):
    """Move the module off its zero-init so a parity test cannot pass trivially.

    Both the head AND the gate must be excited: with the gate at its initial zero
    every path below collapses to arm 2 and would agree for the wrong reason.
    """
    torch.manual_seed(seed)
    torch.nn.init.normal_(mod.proj[0].weight, std=0.5)
    torch.nn.init.normal_(mod.proj[0].bias, std=0.5)
    torch.nn.init.normal_(mod.gate_mlp[2].weight, std=gate_std)
    torch.nn.init.normal_(mod.gate_mlp[2].bias, std=gate_std)
    return mod


# ── 1. Registration and config ────────────────────────────────────────────────

def test_registered_in_bias_types():
    """GraphAttentionBias iterates BIAS_TYPES; a class missing from it is simply
    never instantiated, and the run trains with no bias at all."""
    assert GatedLinearMagneticBias in BIAS_TYPES
    assert GatedLinearMagneticBias.config_key == "magnetic_linear_v2"


@pytest.mark.parametrize("clash", [
    dict(magnetic=True),
    dict(magnetic_shared=True),
    dict(magnetic_content=True),
    dict(magnetic_linear=True),
    dict(magnetic_hybrid=True),
    dict(magnetic_magnitude=True),
    dict(magnetic_groups=2),
])
def test_config_rejects_double_placement(clash):
    """This is a different HEAD on the same magnetic term, so stacking it on any
    other placement is never an intended arm."""
    with pytest.raises(ValueError):
        GTLMLlamaConfig(magnetic_linear_v2=True, magnetic_dim=_MAG_DIM,
                        **clash, **_BASE)


def test_experiment_config_gates_accept_v2():
    """Every dataset/collator gate keys on `uses_magnetic`. A gate that missed
    this arm would emit no eigenvectors, the bias would return None, and the run
    would train with NO graph bias while looking perfectly healthy — the single
    failure mode this arm is most exposed to, since it is the newest flag."""
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.context.config import RunConfig as CtxCfg
    from src.experiments.graphqa.config import RunConfig as GqaCfg

    for Cfg in (KgqaCfg, CtxCfg, GqaCfg):
        c = Cfg(magnetic=False, magnetic_linear_v2=True)
        assert c.uses_magnetic, Cfg
        assert c.bias_params().get("magnetic_linear_v2") is True, Cfg
        # The gate width must ride along, or the model silently builds the
        # default-width gate while the run record claims another.
        assert c.bias_params().get("magnetic_gate_repr_dim") == c.magnetic_gate_repr_dim

    # Same eigenvector bytes as the magnetic arm, so it must map to the same data
    # build — otherwise the sweep silently rebuilds the dataset.
    assert (CtxCfg(magnetic=False, magnetic_linear_v2=True).data_config_key()
            == CtxCfg(magnetic=True).data_config_key())
    assert (KgqaCfg(magnetic=False, magnetic_linear_v2=True).data_config_key("webqsp")
            == KgqaCfg(magnetic=True).data_config_key("webqsp"))


# ── 2. Arm-2 equivalence at initialisation ────────────────────────────────────

def test_gate_is_exactly_one_at_init():
    """g = 1 + tanh(0) = 1. The zero-init on the gate's OUTPUT layer is what makes
    this exact rather than approximate."""
    mod = GatedLinearMagneticBias(4, 16, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    parts = mod._phi(magnetic, num_nodes, torch.device("cpu"))
    g = mod._gate(*parts)
    assert (g - 1.0).abs().max().item() == 0.0


def test_equals_linear_arm_at_init():
    """With the gate at 1 this arm IS arm 2, so the two must agree bit-for-bit
    when they carry the same head weights.

    If this ever fails, every comparison of a v2 number against arm 2's numbers is
    meaningless — the arms would differ by something other than the gate.

    The ENTIRE shared trunk is copied across, not just ``proj``: constructing a
    module consumes RNG draws, so two separately-seeded instances get different
    ``lambda_lin``/``deep_set`` weights and therefore different phi. That
    difference has nothing to do with the gate and would make this test fail for
    the wrong reason.
    """
    torch.manual_seed(0)
    v2 = GatedLinearMagneticBias(4, 16, _Cfg()).double()
    arm2 = LinearMagneticBias(4, 16, _Cfg()).double()
    torch.nn.init.normal_(v2.proj[0].weight, std=0.5)
    torch.nn.init.normal_(v2.proj[0].bias, std=0.5)
    arm2.load_state_dict({k: v for k, v in v2.state_dict().items()
                          if not k.startswith("gate_mlp.")})

    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    diff = (v2(**kw) - arm2(**kw)).abs().max().item()
    assert diff < 1e-12, f"v2 differs from arm 2 at gate=1 by {diff}"


def test_zero_init_bias_is_exactly_zero():
    """Both the head and the gate are zero-initialised, so the whole module emits
    exactly 0 at step 0 and cannot destabilise training from the first update."""
    mod = GatedLinearMagneticBias(4, 16, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    assert out is not None
    assert out.abs().max().item() == 0.0


def test_gate_actually_changes_the_bias_once_trained():
    """The converse of the test above: once the gate leaves zero the bias must
    MOVE. A gate wired onto a dead axis would leave it identical to arm 2 forever
    and the sweep would report a null that is really a bug."""
    torch.manual_seed(0)
    v2 = _excite(GatedLinearMagneticBias(4, 16, _Cfg()).double())
    arm2 = LinearMagneticBias(4, 16, _Cfg()).double()
    arm2.load_state_dict({k: v for k, v in v2.state_dict().items()
                          if not k.startswith("gate_mlp.")})

    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    assert (v2(**kw) - arm2(**kw)).abs().max().item() > 1e-6


# ── 3. Folded / unfolded parity ───────────────────────────────────────────────

def test_folded_matches_unfolded_fp64():
    """The folded path pushes both the head AND the per-node gate inside the
    spectral sum, contracting in two einsums where the naive reading needs four
    plus an explicit (B,N,N,2m) cat. The fold is the whole forward, so a wrong
    axis in the gate einsum would go unnoticed without this."""
    mod = _excite(GatedLinearMagneticBias(4, 16, _Cfg()).double())
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    folded = mod(**kw)
    mod.legacy_unfolded = True
    unfolded = mod(**kw)
    diff = (folded - unfolded).abs().max().item()
    assert diff < 1e-11, f"folded/unfolded mismatch {diff}"


def test_gate_indexes_the_query_node_not_the_key_node():
    """g is indexed by i (the QUERY row), which is what makes the collapsed form
    'arm 2 with per-query-node channel weights'. Indexing j instead would silently
    transpose the arm into a different (and untested) model.

    Checked by construction: scale ONE node's gate and assert the change lands in
    that node's ROW and nowhere else.
    """
    mod = _excite(GatedLinearMagneticBias(4, 16, _CfgSelfNode()).double())
    magnetic, num_nodes = _mag_inputs(_batch(_items(node_counts=(6, 6, 6))))
    dev = torch.device("cpu")
    parts = mod._phi(magnetic, num_nodes, dev)

    base = mod._finalize(mod._phase_bias(*parts), dev)
    real_gate = mod._gate
    target = 2                                        # perturb node 2's gate only
    def _patched(V_real, V_imag, phi):
        g = real_gate(V_real, V_imag, phi).clone()
        g[:, target] = g[:, target] * 1.5
        return g
    mod._gate = _patched
    perturbed = mod._finalize(mod._phase_bias(*parts), dev)
    mod._gate = real_gate

    delta = (perturbed - base).abs()                  # (B, H, N, N)
    assert delta[:, :, target, :].max().item() > 1e-6, "query row did not move"
    other = torch.ones(delta.shape[-2], dtype=torch.bool)
    other[target] = False
    assert delta[:, :, other, :].max().item() < 1e-12, "a non-target ROW moved"


# ── 4. Factorization parity ───────────────────────────────────────────────────

def test_factorization_reproduces_dense_bias_everywhere_with_self_node():
    """<Q_struct[i], K_struct[j]> must equal the dense bias on the FULL matrix
    under ``bias_self_node=True`` — the configuration a factorized backbone would
    actually run. The gate rides entirely on the query side, so this is the check
    that it has not leaked into the key side."""
    mod = _excite(GatedLinearMagneticBias(4, 16, _CfgSelfNode()).double())
    torch.nn.init.zeros_(mod.proj[0].bias)      # the beta term is not factorizable
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")

    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    q, k = mod.structural_factors(magnetic, num_nodes, dev)
    recon = torch.einsum("bhim,bjm->bhij", q, k)
    diff = (dense - recon).abs().max().item()
    assert diff < 1e-10, f"full-matrix factorization mismatch {diff}"


def test_factorization_key_side_is_head_free_and_parameter_free():
    """K_struct is [V_R ‖ V_I]: no head axis and no learned parameters at all.

    This is the arm's one structural advantage over the magnitude channel — the
    key block broadcasts across GQA groups instead of needing a per-group copy —
    so it is asserted rather than assumed.
    """
    mod = _excite(GatedLinearMagneticBias(4, 16, _Cfg()).double())
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    q, k = mod.structural_factors(magnetic, num_nodes, torch.device("cpu"))
    assert k.dim() == 3                                  # (B, N, 2M): no head axis
    assert q.shape[1] == 4 and q.shape[-1] == k.shape[-1]
    V_real, V_imag = magnetic[0][..., 0], magnetic[0][..., 1]
    assert torch.allclose(k, torch.cat([V_real, V_imag], dim=-1), atol=1e-12)


def test_rank_is_still_capped_by_two_m():
    """The gate buys per-node adaptivity, NOT rank: the right factor X = [V_R ‖ V_I]
    is shared across channels, so b = (sum_c diag(g[:,c]) X S_c) Xᵀ keeps arm 2's
    ceiling of min(N, 2M). Asserted because the first draft of the plan claimed
    otherwise, and a stated capability nobody checks is how a wrong claim survives
    into a write-up.

    ``beta_h`` is zeroed first: it adds a constant to every entry, i.e. an
    all-ones matrix, which contributes one rank of its own and has nothing to do
    with the spectral part being bounded here.
    """
    cfg = _CfgSelfNode()
    mod = _excite(GatedLinearMagneticBias(4, 16, cfg).double())
    torch.nn.init.zeros_(mod.proj[0].bias)
    # M = 4 eigenvectors on 12-node graphs => rank ceiling 2M = 8 < N = 12.
    magnetic, num_nodes = _mag_inputs(_batch(_items(node_counts=(12, 12)), magnetic_m=4))
    dense = mod(dtype=torch.float64, device=torch.device("cpu"),
                magnetic=magnetic, num_nodes=num_nodes)
    ranks = torch.linalg.matrix_rank(dense, tol=1e-9)
    assert int(ranks.max()) <= 8, f"rank {int(ranks.max())} exceeds 2M = 8"


# ── 5. Scale stability (§8) ───────────────────────────────────────────────────

def test_gate_is_bounded_in_zero_two():
    """tanh is what keeps the shared DeepSets trunk at degree 1 in the bias. This
    is the property whose ABSENCE killed arms 3 and 4 (MIXED_BIAS.md §5.7), so it
    is checked at a trunk scale far past anything training reaches.

    The interval is closed, not open, on purpose: the maths gives (0, 2), but at
    this scale ``tanh`` saturates to exactly ±1 in float64, so the endpoints are
    attained numerically. Boundedness is the property that matters — the trunk
    enters the bias at degree 1 either way.
    """
    mod = GatedLinearMagneticBias(4, 16, _Cfg()).double()
    torch.manual_seed(0)
    torch.nn.init.normal_(mod.gate_mlp[0].weight, std=50.0)
    torch.nn.init.normal_(mod.gate_mlp[2].weight, std=50.0)
    torch.nn.init.normal_(mod.gate_mlp[2].bias, std=50.0)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    parts = mod._phi(magnetic, num_nodes, torch.device("cpu"))
    g = mod._gate(*parts)
    assert g.min().item() >= 0.0 and g.max().item() <= 2.0


def test_bias_bounded_by_twice_arm_two():
    """|b - beta| <= 2 R * sum_c (|W_R[c,h]| + |W_I[c,h]|) * max_l |phi_lc|, the §8
    argument in one assertion.

    R = max_i sum_l |V_il|² is the per-node row energy, and it appears because the
    Cauchy-Schwarz step bounds sum_l |V_il||V_jl| by R, not by 1. For REAL
    eigenvectors R is exactly 1 (mixed_bias/README.md measured 1.0000, min =
    median = max over 7 856 node-rows), which is where §8's factor-of-2 statement
    comes from. This fixture feeds unnormalised ``randn`` columns, so R is far
    from 1 and carrying it explicitly is what makes the test check the inequality
    rather than the fixture.
    """
    mod = _excite(GatedLinearMagneticBias(4, 16, _CfgSelfNode()).double(), gate_std=5.0)
    torch.nn.init.zeros_(mod.proj[0].bias)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")
    V_real, V_imag, phi = mod._phi(magnetic, num_nodes, dev)

    W = mod.proj[0].weight                                   # (H, 2m)
    m = W.shape[1] // 2
    R = (V_real.square() + V_imag.square()).sum(-1).amax(-1)  # (B,) row energy
    phi_max = phi.abs().amax(dim=1)                          # (B, m)
    bound = 2.0 * R[:, None] * torch.einsum(
        'bc,hc->bh', phi_max, W[:, :m].abs() + W[:, m:].abs())
    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    peak = dense.abs().amax(dim=(-1, -2))                    # (B, H)
    assert (peak <= bound + 1e-9).all(), (peak - bound).max().item()


def test_bias_is_at_most_twice_arm_two_channelwise():
    """The §8 claim stated without any assumption on V: with the same head
    weights, v2's bias is bounded by twice arm 2's channel-wise absolute sum,
    because g ∈ [0, 2] scales each channel and nothing else."""
    torch.manual_seed(0)
    v2 = _excite(GatedLinearMagneticBias(4, 16, _CfgSelfNode()).double(), gate_std=5.0)
    torch.nn.init.zeros_(v2.proj[0].bias)
    dev = torch.device("cpu")
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    V_real, V_imag, phi = v2._phi(magnetic, num_nodes, dev)

    W = v2.proj[0].weight
    m = W.shape[1] // 2
    # Per-channel kernel magnitudes, before any gate: sum_c |W_R Re K| + |W_I Im K|.
    reK = (torch.einsum('bil,bjl,blc->bijc', V_real, V_real, phi)
           + torch.einsum('bil,bjl,blc->bijc', V_imag, V_imag, phi))
    imK = (torch.einsum('bil,bjl,blc->bijc', V_imag, V_real, phi)
           - torch.einsum('bil,bjl,blc->bijc', V_real, V_imag, phi))
    ungated = (torch.einsum('bijc,hc->bhij', reK.abs(), W[:, :m].abs())
               + torch.einsum('bijc,hc->bhij', imK.abs(), W[:, m:].abs()))

    dense = v2(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    assert (dense.abs() <= 2.0 * ungated + 1e-9).all(), \
        (dense.abs() - 2.0 * ungated).max().item()


# ── 6. Padding safety and the no-silent-no-op rule ────────────────────────────

def test_missing_features_return_none_not_silent_zero():
    mod = GatedLinearMagneticBias(4, 16, _Cfg()).double()
    assert mod(dtype=torch.float64, device=torch.device("cpu"),
               magnetic=None, num_nodes=None) is None


def test_padded_nodes_are_untouched_by_the_gate():
    """Ragged batches pad V with zeros, so S_i = 0 on a padded row and the gate
    there is a constant — but its BIAS term is not zero once trained, so a padded
    row could pick up a non-zero gate. It must still contribute nothing, because
    V is zero in that row and every path multiplies by V.

    Equivalently: shrinking the batch to one graph must not change that graph's
    bias block.
    """
    mod = _excite(GatedLinearMagneticBias(4, 16, _CfgSelfNode()).double())
    dev = torch.device("cpu")
    items = _items(node_counts=(4, 9, 6))
    kw = dict(dtype=torch.float64, device=dev)

    magnetic, num_nodes = _mag_inputs(_batch(items))
    ragged = mod(magnetic=magnetic, num_nodes=num_nodes, **kw)

    for b, it in enumerate(items):
        n = it["num_nodes"]
        m1, nn1 = _mag_inputs(_batch([it]))
        alone = mod(magnetic=m1, num_nodes=nn1, **kw)
        diff = (ragged[b, :, :n, :n] - alone[0, :, :n, :n]).abs().max().item()
        assert diff < 1e-11, f"graph {b} moved by {diff} when batched"


# ── 7. Save / load round-trip ─────────────────────────────────────────────────

def test_bias_parameters_round_trip(tmp_path):
    """The gate MLP must land in bias_parameters.pt — the file the LoRA adapter
    does NOT contain. A gate that trains but is never saved reloads at g = 1, i.e.
    silently evaluates as arm 2 (see the load_best_model bias bug)."""
    from src.models.io import save_bias_parameters, load_bias_parameters

    cfg = GTLMLlamaConfig(magnetic_linear_v2=True, magnetic_dim=_MAG_DIM,
                          magnetic_gate_repr_dim=24, graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    for mod in model.modules():
        if isinstance(mod, GatedLinearMagneticBias):
            _excite(mod)

    save_bias_parameters(model, str(tmp_path), ["graph_bias"])
    saved = torch.load(tmp_path / "bias_parameters.pt", map_location="cpu",
                       weights_only=True)
    assert any("gate_mlp.2.weight" in k for k in saved), sorted(saved)
    assert any("proj.0.weight" in k for k in saved), sorted(saved)

    torch.manual_seed(0)
    fresh = GTLMLlamaForCausalLM(cfg).double().eval()
    batch = _batch(_items())
    with torch.no_grad():
        before = fresh(**batch).logits
        assert not torch.allclose(before, model(**batch).logits, atol=1e-9)
    load_bias_parameters(fresh, str(tmp_path))
    with torch.no_grad():
        assert torch.allclose(model(**batch).logits, fresh(**batch).logits, atol=1e-12)


# ── 8. It trains ──────────────────────────────────────────────────────────────

def test_gradients_reach_the_gate():
    """A gate with no gradient path is a gate that never leaves 1, and the arm
    would report as arm 2 while claiming to be something else.

    Note the two-step delay this test is NOT asserting away: at step 0 the head is
    zero, so d(loss)/d(gate) is zero and only the head moves; the gate acquires
    gradient once the head is non-zero. That is a delay, not a saddle — which is
    why the head is excited here before the check.
    """
    mod = _excite(GatedLinearMagneticBias(4, 16, _Cfg()).double())
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    out.square().mean().backward()
    for name in ("gate_mlp.0.weight", "gate_mlp.2.weight", "gate_mlp.2.bias"):
        p = dict(mod.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().max().item() > 0, name


def test_gate_has_no_gradient_while_the_head_is_zero():
    """The other half of the two-step story, asserted so the training curve is
    read correctly: at exact zero-init the gate is stationary because the bias
    does not depend on it yet. Seeing the gate flat for the first few steps is
    expected, not a wiring failure."""
    mod = GatedLinearMagneticBias(4, 16, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    (out.sum() + out.square().mean()).backward()
    g = dict(mod.named_parameters())["gate_mlp.2.weight"].grad
    assert g is None or g.abs().max().item() == 0.0
