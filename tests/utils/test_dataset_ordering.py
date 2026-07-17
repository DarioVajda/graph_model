"""
Tests for TextGraphDataset RCM node ordering and the persisted ordering label.

Covers:
  * ``_rcm_relabel_graph`` is structure-preserving (isomorphic), remaps the
    prompt node, and keeps ``original_id`` a permutation.
  * the ``rcm_ordering=True`` construction flag and ``apply_rcm_ordering()``
    method both relabel and set ``node_ordering="rcm"``.
  * ``apply_rcm_ordering()`` guards against running after node-indexed features
    exist.
  * the ``node_ordering`` label is persisted across save/load and propagated
    through ``select`` and ``__add__`` (-> "mixed").
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pytest
import torch

from src.utils.text_graph_dataset import TextGraphDataset


def _graphs(n=4, seed=0):
    gs = []
    for i in range(n):
        g = nx.barabasi_albert_graph(7 + i, 2, seed=seed + i)
        nx.set_node_attributes(g, {nd: f"node {nd} graph {i}" for nd in g.nodes()}, "text")
        g.graph["prompt_node"] = i % g.number_of_nodes()
        gs.append(g)
    return gs


def test_rcm_relabel_preserves_structure():
    g = TextGraphDataset(_graphs(1)).graphs[0]          # already int-labeled 0..N-1
    g2 = TextGraphDataset._rcm_relabel_graph(g)

    assert g2.number_of_nodes() == g.number_of_nodes()
    assert g2.number_of_edges() == g.number_of_edges()
    assert nx.is_isomorphic(g, g2)
    # prompt node remapped into range
    assert 0 <= g2.graph["prompt_node"] < g2.number_of_nodes()
    # original_id is a permutation of 0..N-1 (relabeling is a bijection)
    oids = sorted(g2.nodes[i]["original_id"] for i in range(g2.number_of_nodes()))
    assert oids == list(range(g2.number_of_nodes()))


def test_rcm_canonical_node_order():
    """Regression: after RCM relabel the node *insertion* order must equal the
    label order 0..N-1. nx.relabel_nodes keeps the original insertion order, so
    without re-canonicalizing, nx.to_numpy_array (used by SPD/RRWP/magnetic
    feature computation) would be misaligned with the labels the model indexes
    by — silently scrambling every RCM run."""
    g = TextGraphDataset(_graphs(1)).graphs[0]
    g2 = TextGraphDataset._rcm_relabel_graph(g)
    assert list(g2.nodes()) == sorted(g2.nodes()) == list(range(g2.number_of_nodes()))
    # insertion-order adjacency must equal label-order adjacency
    A_insertion = nx.to_numpy_array(g2)
    A_label = nx.to_numpy_array(g2, nodelist=sorted(g2.nodes()))
    assert (A_insertion == A_label).all()


def test_rcm_feature_consistency_spd():
    """End-to-end: SPD computed *after* RCM equals the permuted original SPD, i.e.
    the dataset reorders graph topology and features consistently. This is the
    property the insertion-order bug broke (and that the GPU permutation test
    catches only indirectly)."""
    graphs = _graphs(3)
    a = TextGraphDataset([g.copy() for g in graphs])
    a.compute_shortest_path_distances(use_gpu=False)
    b = TextGraphDataset([g.copy() for g in graphs], rcm_ordering=True)
    b.compute_shortest_path_distances(use_gpu=False)

    for idx in range(len(a)):
        ia, ib = a[idx], b[idx]
        # b's current index -> original node id
        perm = [None] * ib["num_nodes"]
        for orig, cur in ib["original_ids"].items():
            perm[cur] = orig
        permuted_orig = ia["shortest_path_dists"][perm][:, perm]
        assert torch.equal(permuted_orig, ib["shortest_path_dists"]), \
            f"graph {idx}: RCM SPD is not a consistent permutation of the original"


def test_rcm_prompt_node_follows_relabel():
    g = TextGraphDataset(_graphs(1)).graphs[0]
    p_old = g.graph["prompt_node"]
    g2 = TextGraphDataset._rcm_relabel_graph(g)
    # the node that was the prompt keeps its original_id at its new index
    new_p = g2.graph["prompt_node"]
    assert g2.nodes[new_p]["original_id"] == g.nodes[p_old]["original_id"]


def test_init_flag_sets_label_and_reorders():
    plain = TextGraphDataset(_graphs())
    rcm = TextGraphDataset(_graphs(), rcm_ordering=True)
    assert plain.node_ordering == TextGraphDataset.ORDERING_ORIGINAL
    assert not plain.is_rcm_ordered
    assert rcm.node_ordering == TextGraphDataset.ORDERING_RCM
    assert rcm.is_rcm_ordered
    # at least one graph's node order actually changed
    def ids(ds, k):
        g = ds.graphs[k]
        return [g.nodes[i]["original_id"] for i in range(g.number_of_nodes())]
    assert any(ids(plain, k) != ids(rcm, k) for k in range(len(rcm.graphs)))


def test_apply_rcm_ordering_method():
    ds = TextGraphDataset(_graphs())
    ds.apply_rcm_ordering()
    assert ds.is_rcm_ordered
    # text column rebuilt and aligned with the reordered graphs
    assert len(ds._hf_dataset["text"]) == len(ds.graphs)
    g0 = ds.graphs[0]
    assert ds._hf_dataset["text"][0] == [g0.nodes[i]["text"] for i in range(g0.number_of_nodes())]
    assert ds._hf_dataset["prompt_node"][0] == g0.graph["prompt_node"]


def test_apply_rcm_ordering_guards_after_features():
    ds = TextGraphDataset(_graphs())
    ds.compute_laplacian_coordinates(embedding_dim=4)
    with pytest.raises(ValueError, match="before computing node-indexed features"):
        ds.apply_rcm_ordering()


def test_select_carries_ordering():
    ds = TextGraphDataset(_graphs(), rcm_ordering=True)
    sub = ds.select([0, 1])
    assert sub.node_ordering == TextGraphDataset.ORDERING_RCM


def test_add_combines_ordering():
    rcm = TextGraphDataset(_graphs(), rcm_ordering=True)
    rcm2 = TextGraphDataset(_graphs(seed=10), rcm_ordering=True)
    plain = TextGraphDataset(_graphs(seed=20))
    assert (rcm + rcm2).node_ordering == TextGraphDataset.ORDERING_RCM
    assert (rcm + plain).node_ordering == TextGraphDataset.ORDERING_MIXED


@pytest.mark.parametrize("rcm", [False, True])
def test_save_load_roundtrips_ordering(rcm):
    ds = TextGraphDataset(_graphs(), rcm_ordering=rcm)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ds")
        ds.save(path)
        rt = TextGraphDataset.load(path)
    expected = TextGraphDataset.ORDERING_RCM if rcm else TextGraphDataset.ORDERING_ORIGINAL
    assert rt.node_ordering == expected
