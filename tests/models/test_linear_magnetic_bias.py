"""Correctness gate for ``LinearMagneticBias`` (``--magnetic-linear``).

These run BEFORE the Phase 2 training sweep, by design: none of the failures they
catch would announce themselves in a training curve. A wiring slip produces a run
with no bias at all; a Psi-indexing slip produces a bias that is merely *wrong*.
Both train cleanly and both read as "linearization hurt", which is precisely the
conclusion the sweep exists to draw. See ``src/models/LINEAR_BIAS.md`` §5.

Numbering follows the doc's test list (9-14).
"""

import networkx as nx
import pytest
import torch

from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.models.bias import BIAS_TYPES, LinearMagneticBias, MagneticBias
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.utils.text_graph_dataset import TextGraphDataset

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)
_MAG_DIM = 16


class _Cfg:
    """Minimal bias_config for constructing a bias module directly."""
    magnetic_dim = _MAG_DIM
    hidden_size = 64


class _CfgSelfNode(_Cfg):
    """As ``_Cfg`` but keeping the intra-node diagonal (LINEAR_BIAS.md §7.3)."""
    bias_self_node = True


def _items(node_counts=(4, 9, 6), seed=0, m=None):
    """Batch items carrying magnetic eigenvectors, mixed graph sizes.

    ``m=None`` stores the full spectrum (m == N), which is what an ``m=0`` cache
    holds and what forces the collator to pad the smaller graphs.
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


# ── 9. Zero-init inertness ────────────────────────────────────────────────────

def test_zero_init_bias_is_exactly_zero():
    """At step 0 the linear head emits exactly 0, so the model is bit-identical
    to the no-bias backbone. Anything else destabilises training from step 0."""
    mod = LinearMagneticBias(4, 16, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    assert out is not None
    assert out.abs().max().item() == 0.0


def test_zero_init_logits_match_no_bias_model():
    """At zero-init the bias must be inert: feeding the eigenvectors changes
    nothing versus withholding them (which makes the module return None).

    Compared WITHIN one model on purpose — constructing the bias consumes RNG
    draws, so two separately-seeded models get different backbone weights and the
    comparison would be meaningless.
    """
    cfg = GTLMLlamaConfig(magnetic_linear=True, magnetic_dim=_MAG_DIM,
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


# ── 10. No silent no-op ───────────────────────────────────────────────────────

def test_missing_features_return_none_not_silent_zero():
    """With no eigenvectors the module must return None (so the caller can tell),
    never a zero tensor that masquerades as a computed bias."""
    mod = LinearMagneticBias(4, 16, _Cfg()).double()
    assert mod(dtype=torch.float64, device=torch.device("cpu"),
               magnetic=None, num_nodes=None) is None


@pytest.mark.parametrize("clash", [
    dict(magnetic=True),
    dict(magnetic_shared=True),
    dict(magnetic_content=True),
    dict(magnetic_groups=2),
])
def test_config_rejects_double_placement(clash):
    """magnetic_linear swaps the HEAD on the magnetic term; stacking it on another
    placement of the same term is never an intended arm, so the config must raise
    rather than quietly build two magnetic biases."""
    with pytest.raises(ValueError):
        GTLMLlamaConfig(magnetic_linear=True, magnetic_dim=_MAG_DIM, **clash, **_BASE)


def test_experiment_config_gates_accept_linear():
    """Every dataset/collator gate keys on `uses_magnetic`, not `magnetic`. A gate
    that missed the linear arm would emit no eigenvectors, the bias would return
    None, and the run would train with NO graph bias while looking healthy."""
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.context.config import RunConfig as CtxCfg

    k = KgqaCfg(magnetic=False, magnetic_linear=True)
    assert k.uses_magnetic and k.collate_magnetic_m == k.magnetic_m
    assert k.bias_params().get("magnetic_linear") is True

    c = CtxCfg(magnetic=False, magnetic_linear=True)
    assert c.uses_magnetic and c.collate_magnetic_m == c.magnetic_m
    assert c.bias_params().get("magnetic_linear") is True
    # The linear arm reads the SAME eigenvector bytes, so it must map to the same
    # build as the magnetic arm — otherwise the sweep silently rebuilds the data.
    assert c.data_config_key() == CtxCfg(magnetic=True).data_config_key()


def test_collate_m_override_must_not_change_data_key():
    """The M-sweep knob is collator-only. If it entered the cache key, every M
    would rebuild the dataset — the exact cost the override exists to avoid."""
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    from src.experiments.context.config import RunConfig as CtxCfg

    for Cfg, key in ((KgqaCfg, lambda c: c.data_config_key("webqsp")),
                     (CtxCfg, lambda c: c.data_config_key())):
        base = Cfg(magnetic=False, magnetic_linear=True)
        trunc = Cfg(magnetic=False, magnetic_linear=True, magnetic_m_collate=16)
        assert key(base) == key(trunc)
        assert trunc.collate_magnetic_m == 16


def test_collate_m_override_rejects_impossible_values():
    from src.experiments.kgqa.config import RunConfig as KgqaCfg
    # Larger than the built m: the collator can only truncate, so this would
    # silently fall back and mislabel the run.
    with pytest.raises(ValueError):
        KgqaCfg(magnetic_linear=True, magnetic_m=128, magnetic_m_collate=256).validate()
    # Set with no magnetic bias at all: silently does nothing.
    with pytest.raises(ValueError):
        KgqaCfg(magnetic=False, magnetic_linear=False, magnetic_m_collate=16).validate()


# ── 11. Folded / unfolded parity ──────────────────────────────────────────────

def test_folded_matches_unfolded_fp64():
    """``_folded_spectral`` pushes proj[0] into phi before the N^2 einsums. For the
    linear head that fold IS the whole forward, so a wrong split point would go
    unnoticed without this. (It was: the base split at out_features, correct only
    because MagneticBias's out == in/2.)"""
    torch.manual_seed(0)
    mod = LinearMagneticBias(4, 16, _Cfg()).double()
    torch.nn.init.normal_(mod.proj[0].weight, std=0.5)
    torch.nn.init.normal_(mod.proj[0].bias, std=0.5)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))

    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    folded = mod(**kw)
    mod.legacy_unfolded = True
    unfolded = mod(**kw)
    assert torch.allclose(folded, unfolded, atol=1e-12), \
        (folded - unfolded).abs().max().item()


def test_mlp_head_folded_parity_still_holds():
    """The split-point fix must not perturb MagneticBias itself."""
    torch.manual_seed(0)
    mod = MagneticBias(4, 16, _Cfg()).double()
    torch.nn.init.normal_(mod.proj[2].weight, std=0.5)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    folded = mod(**kw)
    mod.legacy_unfolded = True
    assert torch.allclose(folded, mod(**kw), atol=1e-12)


# ── 12. Factorization parity (the one that de-risks the deferred backbone) ────

def test_factorization_reproduces_dense_bias_offdiagonal():
    """<Q_struct[i], K_struct[j]> must equal the dense bias off the diagonal.

    This is the entire premise of the deferred O(N) backbone, checked in fp64 on
    CPU with no kernel involved — so it is settled before any backbone work
    starts rather than discovered during it.
    """
    torch.manual_seed(0)
    mod = LinearMagneticBias(4, 16, _Cfg()).double()
    torch.nn.init.normal_(mod.proj[0].weight, std=0.5)
    torch.nn.init.normal_(mod.proj[0].bias, std=0.0)   # bias term is not factorizable
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")

    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    q, k = mod.structural_factors(magnetic, num_nodes, dev)     # (B,H,N,2M), (B,N,2M)
    recon = torch.einsum("bhim,bjm->bhij", q, k)

    n = dense.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    diff = (dense[..., off] - recon[..., off]).abs().max().item()
    assert diff < 1e-10, f"factorization mismatch {diff}"


def test_factorization_reproduces_dense_bias_everywhere_with_self_node():
    """With ``bias_self_node=True`` the equivalence holds on the FULL matrix.

    This is the point of the flag: an inner product gives <q_i, k_i> and cannot be
    forced to 0, so under the default mask the factorization can only ever match
    off-diagonal (test above). Unmasked, the dense path and the factorized path are
    the same function everywhere — which is what the deferred backbone needs, and
    it is checked here in fp64 before any backbone exists.
    """
    torch.manual_seed(0)
    mod = LinearMagneticBias(4, 16, _CfgSelfNode()).double()
    torch.nn.init.normal_(mod.proj[0].weight, std=0.5)
    torch.nn.init.normal_(mod.proj[0].bias, std=0.0)   # bias term is not factorizable
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    dev = torch.device("cpu")

    dense = mod(dtype=torch.float64, device=dev, magnetic=magnetic, num_nodes=num_nodes)
    q, k = mod.structural_factors(magnetic, num_nodes, dev)
    recon = torch.einsum("bhim,bjm->bhij", q, k)

    diff = (dense - recon).abs().max().item()
    assert diff < 1e-10, f"full-matrix factorization mismatch {diff}"


def test_factorization_key_side_is_head_free():
    """K_struct carries no head dimension — the property that makes this
    GQA-native (one structural dictionary broadcast across all query heads)."""
    mod = LinearMagneticBias(4, 16, _Cfg()).double()
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    q, k = mod.structural_factors(magnetic, num_nodes, torch.device("cpu"))
    assert k.dim() == 3                        # (B, N, 2M): no head axis
    assert q.shape[1] == 4 and q.shape[-1] == k.shape[-1]


# ── 13. Save / load round-trip ────────────────────────────────────────────────

def test_bias_parameters_round_trip(tmp_path):
    """The trained head must survive save/load through bias_parameters.pt — the
    file the LoRA adapter does NOT contain."""
    from src.models.io import save_bias_parameters, load_bias_parameters

    cfg = GTLMLlamaConfig(magnetic_linear=True, magnetic_dim=_MAG_DIM,
                          graph_attn_impl="eager", **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    # Perturb the head so a failed load cannot pass by coincidence (zero-init).
    for mod in model.modules():
        if isinstance(mod, LinearMagneticBias):
            torch.nn.init.normal_(mod.proj[0].weight, std=0.3)

    save_bias_parameters(model, str(tmp_path), ["graph_bias"])
    saved = torch.load(tmp_path / "bias_parameters.pt", map_location="cpu",
                       weights_only=True)
    assert any("proj.0.weight" in k for k in saved), sorted(saved)

    # Same seed => identical BACKBONE, so the only difference between the two
    # models is the bias head. Seeding differently would make the comparison
    # about the backbone and the test would say nothing about the round-trip.
    torch.manual_seed(0)
    fresh = GTLMLlamaForCausalLM(cfg).double().eval()

    batch = _batch(_items())
    with torch.no_grad():
        before = fresh(**batch).logits
        target = model(**batch).logits
    # The perturbed head must actually move the logits, or the check below is
    # satisfied by a no-op load.
    assert not torch.allclose(before, target, atol=1e-8)

    assert load_bias_parameters(fresh, str(tmp_path)) is not None
    with torch.no_grad():
        after = fresh(**batch).logits
    assert torch.allclose(after, target, atol=1e-10), (after - target).abs().max().item()


# ── 14. M-truncation equivalence (load-bearing for the whole M-grid) ──────────

def test_collate_truncation_equals_built_at_m():
    """Truncating a stored-m dataset to M at collate time must be bit-identical to
    a dataset built at m=M. Every Phase 2 M-curve is mislabelled if this is false.

    Holds because ``eigh`` returns ascending eigenvalues and both the builder
    (utils/magnetic_lap.py) and the collator truncate by prefix slice.
    """
    M = 5
    stored = _items(m=None)                              # full spectrum, as m=0 caches hold
    # "Built at m=M" = the SAME eigenpairs, pre-sliced by prefix — which is what
    # get_magnetic_laplacian_coords does after eigh returns ascending eigenvalues.
    # (Regenerating them with a different shape would compare different random
    # tensors and prove nothing.)
    pre_truncated = [
        {**it,
         "magnetic_V": it["magnetic_V"][:, :M],
         "magnetic_lambdas": it["magnetic_lambdas"][:M]}
        for it in stored
    ]

    late = _batch(stored, magnetic_m=M)                  # collator truncates
    early = _batch(pre_truncated, magnetic_m=0)          # dataset was built at M

    assert late["magnetic_V"].shape == early["magnetic_V"].shape
    assert torch.equal(late["magnetic_V"], early["magnetic_V"])
    assert torch.equal(late["magnetic_lambdas"], early["magnetic_lambdas"])


def test_phi_normalisation_tracks_truncated_count():
    """``_phi`` averages over VALID eigenvalues, not node count. If it divided by
    N, truncation would rescale phi and the M-sweep would measure that artifact
    instead of truncation."""
    mod = LinearMagneticBias(4, 16, _Cfg()).double()
    dev = torch.device("cpu")
    items = _items(node_counts=(9,), m=None)
    b_full = _batch(items, magnetic_m=0)
    b_trunc = _batch(items, magnetic_m=4)

    (Vf, lf), nf = _mag_inputs(b_full)
    (Vt, lt), nt = _mag_inputs(b_trunc)
    phi_t = mod._phi((Vt, lt), nt, dev)[2]
    # Recomputing phi from the first 4 eigenvalues directly must agree.
    phi_ref = mod._phi((Vf[:, :, :4], lf[:, :4]), torch.tensor([4]), dev)[2]
    assert torch.allclose(phi_t, phi_ref, atol=1e-12)


# ── Structural: permutation invariance with the linear head ───────────────────

def _equiv_items(rcm: bool):
    graphs = []
    for i in range(2):
        g = nx.barabasi_albert_graph(6 + i, 2, seed=i)
        nx.set_node_attributes(g, {nd: "x" for nd in g.nodes()}, "text")
        g.graph["prompt_node"] = i % g.number_of_nodes()
        graphs.append(g)
    ds = TextGraphDataset(graphs, rcm_ordering=rcm)
    ds.compute_shortest_path_distances(use_gpu=False)
    ds.compute_magnetic_lap(q=0.25, use_gpu=False, m=0)

    def toks(o):
        return [1 + (o * 37 + 5) % 200, 1 + (o * 91 + 11) % 200]

    items = []
    for idx in range(len(ds)):
        it = ds[idx]
        n = it["num_nodes"]
        cur_to_orig = {cur: orig for orig, cur in it["original_ids"].items()}
        input_ids = [toks(cur_to_orig[i]) for i in range(n)]
        items.append({
            "num_nodes": n, "prompt_node": it["prompt_node"], "edges": it["edges"],
            "input_ids": input_ids,
            "shortest_path_dists": it["shortest_path_dists"],
            "magnetic_V": it["magnetic_V"], "magnetic_lambdas": it["magnetic_lambdas"],
            "labels": torch.tensor(input_ids[it["prompt_node"]], dtype=torch.long),
        })
    return items


def test_linear_head_is_permutation_invariant():
    """Relabelling nodes must not change prompt logits. The sharpest structural
    check on the linear head: a wrong Psi index or a swapped V_R/V_I half breaks
    this and very little else."""
    cfg = GTLMLlamaConfig(magnetic_linear=True, magnetic_dim=_MAG_DIM,
                          graph_attn_impl="eager", spd=True, max_spd=8, **_BASE)
    torch.manual_seed(0)
    model = GTLMLlamaForCausalLM(cfg).double().eval()
    # Zero-init would make this pass vacuously — the bias must be live.
    for mod in model.modules():
        if isinstance(mod, LinearMagneticBias):
            torch.nn.init.normal_(mod.proj[0].weight, std=0.5)

    def collate(rcm):
        return GraphCollatorV2(pad_token_id=0)([dict(it) for it in _equiv_items(rcm)])

    b0, b1 = collate(False), collate(True)
    with torch.no_grad():
        o0, o1 = model(**b0), model(**b1)

    m0, m1 = b0["labels"] != -100, b1["labels"] != -100
    assert torch.equal(m0, m1)
    diff = (o0.logits[m0] - o1.logits[m1]).abs().max().item()
    assert diff < 1e-6, f"prompt-logit diff under RCM reorder: {diff}"


def test_registered_in_bias_types():
    assert LinearMagneticBias in BIAS_TYPES
    assert LinearMagneticBias.config_key == "magnetic_linear"
    assert not LinearMagneticBias.shared


# ── 15. bias_self_node — the intra-node diagonal (LINEAR_BIAS.md §7.3) ────────

@pytest.mark.parametrize("cls", [MagneticBias, LinearMagneticBias])
def test_diagonal_zeroed_by_default(cls):
    """The default must stay masked: every result recorded before this flag
    existed was measured with b_ii = 0, and silently changing that would
    invalidate the comparison to them."""
    torch.manual_seed(0)
    mod = cls(4, 16, _Cfg()).double()
    _make_live(mod)
    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    out = mod(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    n = out.shape[-1]
    diag = out[..., torch.arange(n), torch.arange(n)]
    assert diag.abs().max().item() == 0.0


@pytest.mark.parametrize("cls", [MagneticBias, LinearMagneticBias])
def test_self_node_keeps_a_live_diagonal(cls):
    """With the flag on, b_ii must be genuinely non-zero — and the OFF-diagonal
    must be untouched, so the flag changes exactly one thing."""
    torch.manual_seed(0)
    mod = cls(4, 16, _Cfg()).double()
    _make_live(mod)
    torch.manual_seed(0)
    mod_self = cls(4, 16, _CfgSelfNode()).double()
    _make_live(mod_self)

    magnetic, num_nodes = _mag_inputs(_batch(_items()))
    kw = dict(dtype=torch.float64, device=torch.device("cpu"),
              magnetic=magnetic, num_nodes=num_nodes)
    masked, unmasked = mod(**kw), mod_self(**kw)

    n = masked.shape[-1]
    ar = torch.arange(n)
    # Only real nodes carry a meaningful diagonal; padded slots are inert.
    live = unmasked[0, :, ar, ar][:, :int(num_nodes[0])]
    assert live.abs().max().item() > 1e-9, "bias_self_node produced a zero diagonal"

    off = ~torch.eye(n, dtype=torch.bool)
    assert torch.allclose(masked[..., off], unmasked[..., off], atol=1e-12), \
        "bias_self_node must not perturb the off-diagonal"


def test_self_node_rejected_alongside_spd():
    """SPDBias has no self-distance row, so the flag would silently cover only
    part of the active biases. It must raise, not half-apply."""
    with pytest.raises(ValueError, match="SPDBias"):
        GTLMLlamaConfig(magnetic_linear=True, magnetic_dim=_MAG_DIM,
                        graph_attn_impl="eager", spd=True, max_spd=8,
                        bias_self_node=True, **_BASE)


def _make_live(mod):
    """Break zero-init so a diagonal assertion cannot pass vacuously."""
    torch.nn.init.normal_(mod.proj[0].weight, std=0.5)
    if len(mod.proj) > 2 and hasattr(mod.proj[2], "weight"):
        torch.nn.init.normal_(mod.proj[2].weight, std=0.5)
