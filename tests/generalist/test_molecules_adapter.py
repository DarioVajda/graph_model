"""
T9 — the molecules adapter (`src/generalist/adapters/molecules.py`, DESIGN.md §D3).

The adapter is where the harness meets real chemistry, so the tests are about the
places that meeting can go wrong quietly:

* **graph-to-SMILES targets.** `MOLECULE_GENERALIST.md` §5 says the target is the
  stereo-free canonical SMILES for *both* arms, because a parity word means
  nothing without a neighbour ordering and the graph has none. A target that
  carried stereo would ask the graph arm for information its input does not hold,
  and the whole task would read as a graph-arm failure.
* **The flat twin's matched task is canonicalization** — a randomized SMILES in,
  the canonical one out. If its input were the canonical string the task would be
  a copy and the comparison would be worthless.
* **ChEBI-20's screens.** §6: a heavy-atom cap and an answer for multi-fragment
  molecules, both counted rather than assumed small.
* **Tox21's absent labels.** §1: skipping them silently changes each endpoint's
  effective weight, so the per-endpoint counts are a number the run record
  carries.
* **The partition, at both enforcement points.** A training example over a
  test-role molecule is the leak §3 exists to prevent, and ``load`` re-checks
  what ``build`` promised.

Everything builds from fake corpora of a couple of dozen molecules and the ChEBI
fixture under ``fixtures/chebi20`` — the real CSVs are `test_partition.py`'s job,
and a dataset build big enough to be slow proves nothing this does not.
"""

import json
import os

import pytest

from src.generalist.adapters import get_adapter
from src.generalist.adapters import molecules as M
from src.generalist.adapters._partition import PartitionError
from src.generalist.registry import MOLECULE_PREFIX, Registry, resolve
from src.generalist.schema import Example, validate

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CHEBI_FIXTURE = os.path.join(FIXTURES, "chebi20")

# ─────────────────────────────────────────────────────────────────────────────
# Fake corpora
#
# Two dozen molecules with two dozen distinct Bemis-Murcko scaffolds, each with a
# ring and at least one non-ring atom so `ring_membership` has both classes to
# balance between. Distinct scaffolds are the point: `scaffold_split` pours whole
# scaffold groups, so a pool of substituted benzenes would be ONE group and land
# entirely in one split.
# ─────────────────────────────────────────────────────────────────────────────

POOL_SMILES = (
    "Cc1ccccc1", "CCc1ccncc1", "Cc1ccc2ccccc2c1", "CCc1c[nH]c2ccccc12",
    "Cc1ccc2ncccc2c1", "CCc1ccco1", "CCc1ccsc1", "Cc1cc[nH]c1",
    "CCc1ncc[nH]1", "Cc1cncnc1", "CCc1cc2ccccc2o1", "CC1Cc2ccccc2C1",
    "CCC1CCCCC1", "CCN1CCCCC1", "CCN1CCOCC1", "CCC1CCCO1",
    "CCC1CCCC1", "Cc1ccc(-c2ccccc2)cc1", "Cc1ccc2cc3ccccc3cc2c1",
    "CCc1nc2ccccc2[nH]1", "CCC1CCCCCC1", "CCc1nccs1", "CCc1ncco1",
    "CCN1CCCC1",
)

#: Six molecules, three endpoints, and absent labels in a pattern the test counts
#: by hand. `_label` reads ``None`` as "not measured", which is the rule the
#: adapter's per-endpoint counter reuses rather than restates.
TOX21_SMILES = ("c1ccccc1", "CCCCO", "ClCCl", "Clc1ccccc1", "CC(=O)O", "CCN")
TOX21_ENDPOINTS = ("NR-AR", "SR-MMP", "NR-ER")
TOX21_LABELS = {
    "c1ccccc1":   {"NR-AR": 1.0, "SR-MMP": 0.0, "NR-ER": None},
    "CCCCO":      {"NR-AR": 0.0, "SR-MMP": None, "NR-ER": None},
    "ClCCl":      {"NR-AR": None, "SR-MMP": 1.0, "NR-ER": 0.0},
    "Clc1ccccc1": {"NR-AR": 1.0, "SR-MMP": 0.0, "NR-ER": 1.0},
    "CC(=O)O":    {"NR-AR": 0.0, "SR-MMP": None, "NR-ER": 0.0},
    "CCN":        {"NR-AR": None, "SR-MMP": None, "NR-ER": 1.0},
}
#: labelled per endpoint, by eye: NR-AR 4, SR-MMP 3, NR-ER 4 (of 6 molecules)
TOX21_LABELLED = {"NR-AR": 4, "SR-MMP": 3, "NR-ER": 4}

#: ClinTox — the held-out corpus. Real drugs, deliberately nothing the pool
#: holds, and one of them (nicotine) carries stereo so the key's stereo-freeness
#: is exercised on a molecule that has some.
CLINTOX_SMILES = ("CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
                  "CN1CCC[C@H]1c1cccnc1", "OC(=O)c1ccccc1O")


def _records(smiles_list, targets_for):
    from rdkit import Chem

    out = []
    for smiles in smiles_list:
        mol = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles))
        out.append({"smiles": Chem.MolToSmiles(mol, canonical=True), "mol": mol,
                    "targets": targets_for(smiles)})
    return out


def _fake_load_tier_b(name, raw_dir=None):
    """Stand-in for `data.load_tier_b`, in the shape the real one returns.

    Patched over the *module attribute* in both `data` and `tier_b`: the adapter
    imports it inside each function (so it sees the patch on `data`), while
    `tier_b.build_tier_b_examples` bound it at import time (so it needs its own).
    """
    from src.experiments.molecules.data import TIER_B

    if name in ("bace", "hiv", "bbbp"):
        records = _records(POOL_SMILES, lambda s: {"Class": 1.0})
        spec = TIER_B["bace"]
    elif name == "tox21":
        records = _records(TOX21_SMILES, lambda s: dict(TOX21_LABELS[s]))
        spec = TIER_B["tox21"]
    elif name == "clintox":
        records = _records(CLINTOX_SMILES,
                           lambda s: {"FDA_APPROVED": 1.0, "CT_TOX": 0.0})
        spec = TIER_B["clintox"]
    else:
        raise AssertionError(f"the fake corpora do not cover {name!r}")
    return records, spec, {"parse": 0, "unsupported_bond": 0}


@pytest.fixture(scope="module", autouse=True)
def fake_corpora():
    """Replace the MoleculeNet loader for this whole module.

    Module-scoped so the one dataset build below happens under it; the real CSVs
    are `test_partition.py`'s business and nothing here should be waiting on
    41k HIV molecules to parse.
    """
    from src.experiments.molecules import data, tier_b

    patch = pytest.MonkeyPatch()
    patch.setattr(data, "load_tier_b", _fake_load_tier_b)
    patch.setattr(tier_b, "load_tier_b", _fake_load_tier_b)
    yield
    patch.undo()


def _config(cache_root, **overrides):
    kwargs = dict(
        pool=("bace",), tier_b_corpora=("tox21",), regression_corpora=(),
        tier_a_cap_per_pass=4, tier_a_val_size=2, tier_a_test_size=2,
        g2s_cap_per_pass=3, g2s_val_size=2, g2s_test_size=2, held_out_size=3,
        chebi_dir=CHEBI_FIXTURE, chebi_heavy_atom_cap=20,
        cache_root=str(cache_root), data_seed=0,
    )
    kwargs.update(overrides)
    return M.MoleculeAdapterConfig(**kwargs)


BUILT_TASKS = ("ring_membership", "tox21", "chebi20", "g2s", "bond_path")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One build of five tasks — one per answer kind, plus a held-out one.

    ``ring_membership`` is ``token``, ``tox21`` is ``yesno``, ``chebi20`` is
    ``text``, ``g2s`` is ``smiles``, and ``bond_path`` is the held-out family.
    Between them they cover every code path in `build` and every branch of the
    schema's answer-boundary handling.
    """
    config = _config(tmp_path_factory.mktemp("cache"))
    manifest = M.build(config, tasks=BUILT_TASKS, arms=("graph", "flat"),
                       splits=("train", "held_out"))
    return config, manifest


# ─────────────────────────────────────────────────────────────────────────────
# graph-to-SMILES (MOLECULE_GENERALIST.md §5)
# ─────────────────────────────────────────────────────────────────────────────

STEREO_SMILES = (
    "CN1CCC[C@H]1c1cccnc1",                 # nicotine: one tetrahedral centre
    "C[C@@H](N)C(=O)O",                     # alanine
    "C/C=C/C",                              # trans-2-butene: bond stereo
    "C(=O)(O)[C@@H](N)Cc1ccccc1",           # phenylalanine
)


@pytest.mark.parametrize("smiles", STEREO_SMILES)
def test_g2s_target_is_stereo_free(smiles):
    from rdkit import Chem

    mol = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles))
    target = M.g2s_target(mol)
    assert target is not None
    assert not any(mark in target for mark in M.STEREO_MARKS), target
    # And the input really did carry stereo, or the assertion above is vacuous.
    assert any(mark in Chem.MolToSmiles(mol, canonical=True)
               for mark in M.STEREO_MARKS)


@pytest.mark.parametrize("smiles", STEREO_SMILES + POOL_SMILES[:6])
def test_g2s_target_equals_the_roundtrip_check_expectation(smiles):
    """§5: the target is what `roundtrip_check` at the ``exact`` level compares.

    `data.roundtrip_check` flattens stereo before comparing precisely because the
    graph cannot carry it. If the g2s target and that expectation ever disagreed,
    a model could round-trip its own graph perfectly and still score zero.
    """
    from rdkit import Chem
    from src.experiments.molecules.data import roundtrip_check

    mol = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles))
    ok, level, expected, _got = roundtrip_check(mol, encoding="rich_levi")
    assert level == "exact"
    assert ok, f"{smiles} does not round-trip; it would be excluded, not compared"
    assert M.g2s_target(mol) == expected


def test_g2s_refuses_an_encoding_whose_round_trip_is_not_exact(tmp_path):
    """A graph that does not determine the molecule cannot be asked for it."""
    config = _config(tmp_path, encoding="terse_levi")
    with pytest.raises(M.AdapterBuildError, match="exact"):
        M._draw_g2s(config, "train", 0, [])


def test_g2s_flat_input_is_a_randomized_smiles_of_the_same_molecule(built):
    """The flat twin's task is canonicalization, not copying (§5)."""
    from rdkit import Chem

    config, _manifest = built
    source = M.load(f"{MOLECULE_PREFIX}g2s", "train", "flat", config=config)
    assert len(source) > 0

    randomized_count = 0
    for i in range(len(source)):
        item = source[i]
        text = item["text"][item["prompt_node"]]
        smiles = text.split("\nSMILES: ", 1)[1].split("\nA:", 1)[0]
        answer = source[i]["_schema"]["answer"]

        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        assert Chem.MolToSmiles(mol, canonical=True,
                                isomericSmiles=False) == answer
        if smiles != answer:
            randomized_count += 1
    # `flat_serialize(canonical=False)` is a *randomised* walk; on a handful of
    # tiny molecules one could coincide with the canonical spelling, but not all.
    assert randomized_count > 0, "the flat input is the canonical string, i.e. a copy"


def test_g2s_graph_arm_carries_no_smiles(built):
    """§1: no SMILES anywhere in the graph arm's prompt, or the task is a copy."""
    config, _manifest = built
    source = M.load(f"{MOLECULE_PREFIX}g2s", "train", "graph", config=config)
    assert len(source) > 0
    for i in range(len(source)):
        item = source[i]
        answer = item["_schema"]["answer"]
        assert item["text"][item["question_node"]] == M.G2S_QUESTION
        assert item["text"][item["prompt_node"]] == "\nA: " + answer
        # The flat arm's serialization marker appears nowhere, and neither does
        # the target itself outside the supervised tail: the graph arm has to
        # write the string, not copy it.
        for node, text in enumerate(item["text"]):
            assert "\nSMILES:" not in text
            if node != item["prompt_node"]:
                assert answer not in text


def test_flat_graph_helper_matches_the_molecules_builder():
    """The one duplicated line in this file, pinned against its original.

    `_flat_graph` exists only because `build_flat_example` serializes the
    *canonical* SMILES and g2s needs a randomized one. On a canonical input the
    two must be byte-identical, or the flat arm of g2s would differ from the flat
    arm of every other task by more than its SMILES.
    """
    from rdkit import Chem

    from src.experiments.molecules.data import flat_serialize
    from src.experiments.molecules.dataset import build_flat_example

    mol = Chem.RemoveAllHs(Chem.MolFromSmiles("Cc1ccccc1"))
    cfg = M._run_config(M.MoleculeAdapterConfig(), "chebi20", "flat")
    question, answer = M.G2S_QUESTION, " Cc1ccccc1"

    theirs = build_flat_example(mol, question, answer, cfg)
    mine = M._flat_graph(question, flat_serialize(mol, canonical=True), answer)
    assert mine.nodes[0]["text"] == theirs.nodes[0]["text"]
    assert mine.graph["prompt_node"] == theirs.graph["prompt_node"]


# ─────────────────────────────────────────────────────────────────────────────
# smiles_scores
# ─────────────────────────────────────────────────────────────────────────────

def test_a_stereo_mark_in_a_prediction_is_scored_as_an_error():
    """§5: "emitting them is an error under this target"."""
    target = "CC(N)C(=O)O"
    scores = M.smiles_scores(["C[C@@H](N)C(=O)O"], [target])
    assert scores["stereo_marks_emitted"] == 1.0
    assert scores["exact_match"] == 0.0
    assert scores["roundtrip_match"] == 0.0, (
        "a stereo-bearing prediction canonicalizes to a different string than "
        "the stereo-free target; scoring it as a match would hide the error the "
        "diagnostic is there to count")
    assert scores["validity"] == 1.0        # it parses; it is just wrong


def test_smiles_scores_on_exact_valid_and_invalid_predictions():
    targets = ["CCO", "Cc1ccccc1", "CCN", "CCO"]
    predictions = [
        "CCO",            # exact
        "c1ccccc1C",      # valid, round-trips, not the canonical spelling
        "C(((",           # unparseable
        "CCC",            # valid, parses, wrong molecule
    ]
    scores = M.smiles_scores(predictions, targets)
    assert scores["n"] == 4
    assert scores["validity"] == pytest.approx(3 / 4)
    assert scores["roundtrip_match"] == pytest.approx(2 / 4)
    assert scores["exact_match"] == pytest.approx(1 / 4)
    assert scores["stereo_marks_emitted"] == 0.0


def test_smiles_scores_survives_an_empty_generation_set():
    """A validator that dies here would lose a run that already cost GPU-hours."""
    assert M.smiles_scores([], [])["n"] == 0
    with pytest.raises(ValueError, match="paired"):
        M.smiles_scores(["CCO"], [])


# ─────────────────────────────────────────────────────────────────────────────
# ChEBI-20 (MOLECULE_GENERALIST.md §6)
# ─────────────────────────────────────────────────────────────────────────────

def test_chebi_cap_and_disconnected_screens(tmp_path):
    config = _config(tmp_path, chebi_heavy_atom_cap=20)
    splits, stats = M.load_chebi(config)

    dropped = stats["dropped"]["train"]
    assert dropped["parse"] == 1                # the deliberately broken SMILES
    assert dropped["empty_description"] == 1    # the row with no caption
    assert dropped["disconnected"] == 1         # the nickel chloride hexahydrate
    assert dropped["heavy_atom_cap"] == 1       # the C22 fatty alcohol
    assert stats["kept"]["train"] == 3

    from rdkit import Chem

    for record in splits["train"]:
        assert record["mol"].GetNumHeavyAtoms() <= 20
        assert len(Chem.GetMolFrags(record["mol"])) == 1
        assert record["text"] and not record["text"].startswith(" ")
    assert stats["kept"]["val"] == 3 and stats["kept"]["test"] == 3


def test_a_molecule_that_is_only_hydrogen_is_dropped(tmp_path):
    """ChEBI describes dihydrogen, and after `RemoveAllHs` it is nothing at all.

    Every screen above it passes such a row: `GetMolFrags` counts zero fragments
    rather than two, zero heavy atoms is under any cap, and an empty graph is
    trivially encodable. Its `partition_key` is the empty string, which
    `schema.validate` refuses — 20k examples into a build, on the one task whose
    corpus is large enough for the failure to be expensive.
    """
    for cap in (20, 64):
        stats = M.load_chebi(_config(tmp_path, chebi_heavy_atom_cap=cap))[1]
        assert stats["dropped"]["train"]["no_heavy_atoms"] == 1

    splits, _stats = M.load_chebi(_config(tmp_path, chebi_heavy_atom_cap=64))
    assert all(record["key"] for record in splits["train"])


def test_chebi_cap_is_a_knob_and_is_recorded(tmp_path):
    """§6: the cap is chosen against the size distribution and *recorded*."""
    loose = M.load_chebi(_config(tmp_path, chebi_heavy_atom_cap=64))[1]
    assert loose["dropped"]["train"]["heavy_atom_cap"] == 0
    assert loose["kept"]["train"] == 4
    assert loose["heavy_atoms"]["train"]["max"] == 23


def test_chebi_disconnected_can_be_kept_explicitly(tmp_path):
    """The screen is a decision with a switch, not a hidden filter."""
    config = _config(tmp_path, chebi_heavy_atom_cap=64,
                     chebi_allow_disconnected=True)
    splits, stats = M.load_chebi(config)
    assert stats["dropped"]["train"]["disconnected"] == 0
    assert stats["kept"]["train"] == 5

    from rdkit import Chem

    assert any(len(Chem.GetMolFrags(r["mol"])) > 1 for r in splits["train"])


def test_chebi_answers_are_the_description(built):
    config, _manifest = built
    source = M.load(f"{MOLECULE_PREFIX}chebi20", "train", "graph", config=config)
    assert len(source) == 3
    for i in range(len(source)):
        item = source[i]
        side = item["_schema"]
        assert side["answer_kind"] == "text"
        assert side["question"] == M.CHEBI_QUESTION
        assert side["answer"].startswith("The molecule is")
        # The prompt tail is "\nA: " + answer: the prefix keeps its trailing
        # space and the answer carries none, which is what lets `render` find it.
        assert item["text"][item["prompt_node"]].endswith("\nA: " + side["answer"])


# ─────────────────────────────────────────────────────────────────────────────
# Tier B and its absent labels (MOLECULE_GENERALIST.md §1)
# ─────────────────────────────────────────────────────────────────────────────

def test_tox21_absent_labels_are_counted_per_endpoint():
    """Skipping changes each endpoint's effective weight, so it is a number."""
    from src.experiments.molecules.data import TIER_B

    records, spec, _dropped = _fake_load_tier_b("tox21")
    counts = M._endpoint_label_counts(records, spec)
    assert set(counts) == set(TOX21_ENDPOINTS)
    for endpoint, labelled in TOX21_LABELLED.items():
        assert counts[endpoint]["labelled"] == labelled, endpoint
        assert counts[endpoint]["absent"] == len(TOX21_SMILES) - labelled
    assert TIER_B["tox21"].task_cols == (), (
        "tox21's endpoints are derived from the CSV; if that changed, the "
        "inverse question map below is reading the wrong columns")


def test_tox21_emits_one_example_per_labelled_molecule_endpoint(built):
    """One example per (molecule, endpoint), routed by the endpoint in the question."""
    config, manifest = built
    source = M.load(f"{MOLECULE_PREFIX}tox21", "train", "graph", config=config)
    assert len(source) > 0

    per_endpoint = {}
    for i in range(len(source)):
        side = source[i]["_schema"]
        assert side["answer_kind"] == "yesno"
        assert side["answer"] in (" Yes", " No")
        endpoint = side["meta"]["endpoint"]
        assert endpoint in TOX21_ENDPOINTS
        assert endpoint in side["question"]
        per_endpoint[endpoint] = per_endpoint.get(endpoint, 0) + 1

    stats = manifest["tasks"][f"{MOLECULE_PREFIX}tox21"]["splits"]["train.p0"]
    assert {k: v for k, v in stats["emitted_per_endpoint"].items()} == per_endpoint
    assert sum(per_endpoint.values()) == len(source)
    # And the per-endpoint absent counts travel in the manifest too (§1).
    assert manifest["endpoints"]["tox21"]["NR-AR"]["labelled"] == \
        TOX21_LABELLED["NR-AR"]


def test_the_endpoint_question_map_inverts_tier_bs_templates():
    from src.experiments.molecules.tier_b import question_for

    mapping = M._endpoint_of_question("tox21")
    assert set(mapping.values()) == set(TOX21_ENDPOINTS)
    for endpoint in TOX21_ENDPOINTS:
        assert mapping[question_for("tox21", endpoint)] == endpoint
    clintox = M._endpoint_of_question("clintox")
    assert set(clintox.values()) == {"FDA_APPROVED", "CT_TOX"}


# ─────────────────────────────────────────────────────────────────────────────
# The TaskSource contract (what the mixture codes against)
# ─────────────────────────────────────────────────────────────────────────────

ALL_BUILT = [
    (f"{MOLECULE_PREFIX}ring_membership", "train", "token"),
    (f"{MOLECULE_PREFIX}tox21", "train", "yesno"),
    (f"{MOLECULE_PREFIX}chebi20", "train", "text"),
    (f"{MOLECULE_PREFIX}g2s", "train", "smiles"),
    (f"{MOLECULE_PREFIX}bond_path", "held_out", "token"),
]


@pytest.mark.parametrize("task,split,kind", ALL_BUILT)
@pytest.mark.parametrize("arm", ("graph", "flat"))
def test_every_emitted_example_passes_the_schema_validator(built, task, split,
                                                           kind, arm):
    """D1: an adapter that emits an invalid item fails the build, not the run.

    ``build`` validates as it writes; this re-validates what came *back off
    disk*, which is the half that would catch a save/load path dropping the
    question node or the prompt index.
    """
    config, _manifest = built
    spec = M.task_specs(config)[task]
    source = M.load(task, split, arm, config=config)
    assert len(source) > 0
    assert (source.task, source.split, source.arm, source.pass_id) == \
        (task, split, arm, 0)
    assert source.answer_kind == kind

    for i in range(len(source)):
        item = source[i]
        example = Example.from_item(item, spec, split=split)
        validate(example, spec)
        assert example.arm == arm
        assert example.key
        assert item["ds_label"] == task
        assert item["text"][item["prompt_node"]].endswith(example.answer)
        if arm == "graph":
            assert item["question_node"] >= 0
            assert item["text"][item["question_node"]] == example.question
            assert item["num_nodes"] > 1
        else:
            assert item["num_nodes"] == 1
            assert example.question in item["text"][0]


@pytest.mark.parametrize("task,split,_kind", ALL_BUILT)
def test_lengths_match_the_items(built, task, split, _kind):
    """The bucket table's two vectors, against the items they describe."""
    config, _manifest = built
    source = M.load(task, split, "graph", config=config)
    num_nodes, num_tokens = source.lengths()
    assert len(num_nodes) == len(num_tokens) == len(source)
    for i in range(len(source)):
        item = source[i]
        assert num_nodes[i] == item["num_nodes"]
        assert num_tokens[i] == sum(len(ids) for ids in item["input_ids"])
        assert num_tokens[i] > 0


def test_the_graph_arm_supervises_more_than_one_token_for_a_caption(built):
    """`get_prompt_node_labels` would supervise one token; a caption needs its span."""
    config, _manifest = built
    caption = M.load(f"{MOLECULE_PREFIX}chebi20", "train", "graph", config=config)
    token_task = M.load(f"{MOLECULE_PREFIX}ring_membership", "train", "graph",
                        config=config)

    def supervised(item):
        return sum(1 for label in item["labels"].tolist() if label != -100)

    assert supervised(token_task[0]) == 1
    assert supervised(caption[0]) > 5
    # The span sits at the tail of the prompt node, and nothing before it.
    labels = caption[0]["labels"].tolist()
    ids = caption[0]["input_ids"][caption[0]["prompt_node"]]
    assert len(labels) == len(ids)
    first = next(i for i, label in enumerate(labels) if label != -100)
    assert labels[first:] == list(ids[first:])
    assert set(labels[:first]) <= {-100}


# ─────────────────────────────────────────────────────────────────────────────
# The partition, at both enforcement points (D3.3)
# ─────────────────────────────────────────────────────────────────────────────

def test_built_keys_hold_the_role_their_split_requires(built):
    config, _manifest = built
    part = M.partition(config)
    for task, split, _kind in ALL_BUILT:
        source = M.load(task, split, "graph", config=config)
        for key in source.keys():
            assert part.is_role(key, split), (task, split, key)


def test_held_out_examples_are_over_held_out_molecules(built):
    """§4: the zero-shot number must be over molecules training never saw."""
    config, _manifest = built
    part = M.partition(config)
    source = M.load(f"{MOLECULE_PREFIX}bond_path", "held_out", "graph",
                    config=config)
    clintox = {M.partition_key(r["mol"])
               for r in _fake_load_tier_b("clintox")[0]}
    assert set(source.keys()) <= clintox
    assert set(source.keys()) <= part.keys("held_out")


def test_build_refuses_a_train_draw_over_a_non_train_key(built):
    """The build-time half of the enforcement, on a draw that violates Rule 2."""
    config, _manifest = built
    part = M.partition(config)
    stolen = sorted(part.keys("held_out"))[0]
    draws = [(None, "Question: ?", " Yes", [], stolen, {})]
    with pytest.raises(PartitionError, match="leak"):
        M._check_roles(part, "ring_membership", "train", draws)
    # …and a well-formed draw passes.
    M._check_roles(part, "ring_membership", "train",
                   [(None, "Q", " Yes", [], sorted(part.keys("train"))[0], {})])


def test_load_refuses_a_train_split_holding_a_non_train_key(built, tmp_path):
    """The load-time half (D3.3): a sample of keys is re-checked against the roles.

    Simulated by rewriting one key in a copy of the artifact's sidecar, which is
    exactly the shape of the failure it exists for: a cache built under one
    partition being read under another.
    """
    import shutil

    config, _manifest = built
    corrupted_root = tmp_path / "corrupted"
    shutil.copytree(config.cache_root, corrupted_root)
    bad_config = _config(corrupted_root)
    assert bad_config.build_version() == config.build_version()

    path = bad_config.source_path("ring_membership", "train", "graph", 0)
    sidecar_path = path + ".schema.json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    part = M.partition(bad_config)
    sidecar["records"][0]["key"] = sorted(part.keys("held_out"))[0]
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)

    with pytest.raises(PartitionError, match="role"):
        M.load(f"{MOLECULE_PREFIX}ring_membership", "train", "graph",
               config=bad_config, check_keys=1000)


def test_generator_test_examples_come_from_test_role_molecules(built):
    """§3 Rule 3: a generator's test set is scaffold-novel by construction."""
    config, _manifest = built
    M.build(config, tasks=("ring_membership",), arms=("graph",),
            splits=("test", "val"))
    part = M.partition(config)
    for split in ("val", "test"):
        source = M.load(f"{MOLECULE_PREFIX}ring_membership", split, "graph",
                        config=config)
        assert len(source) > 0
        for key in source.keys():
            assert part.is_role(key, split)
        assert not (set(source.keys()) & part.keys("train"))


# ─────────────────────────────────────────────────────────────────────────────
# Held-out tasks, the registry, and load's refusals (D2.1)
# ─────────────────────────────────────────────────────────────────────────────

def test_a_held_out_task_cannot_be_built_for_a_training_split(built):
    config, _manifest = built
    for task in ("bond_path", "longest_chain", "clintox"):
        assert M.splits_for(task) == ("held_out",)
        with pytest.raises(M.AdapterBuildError, match="held out"):
            M.load(f"{MOLECULE_PREFIX}{task}", "train", "graph", config=config)


def test_the_registry_refuses_a_held_out_task_in_a_mixture(built):
    """D2.1's second enforcement point, over the specs this adapter registers."""
    from src.generalist.registry import RegistryError, is_held_out

    config, _manifest = built
    registry = M.register_molecule_tasks(Registry(), config)
    for task in ("bond_path", "longest_chain", "clintox"):
        assert is_held_out(registry.get(f"{MOLECULE_PREFIX}{task}"))
    for task in M.TIER_A_TRAIN_TASKS + tuple(config.tier_b_corpora):
        assert not is_held_out(registry.get(f"{MOLECULE_PREFIX}{task}"))
    with pytest.raises(RegistryError, match="held out"):
        resolve(registry, [{"name": f"{MOLECULE_PREFIX}bond_path", "weight": 1.0}],
                tokens_per_step=1000)


def test_the_registry_carries_what_the_build_measured(built):
    """D2: `train_size` and `mean_tokens` are properties of the built data."""
    config, _manifest = built
    for arm in ("graph", "flat"):
        specs = M.task_specs(config, arm=arm)
        chebi = specs[f"{MOLECULE_PREFIX}chebi20"]
        assert chebi.train_size == 3
        assert chebi.mean_tokens and chebi.mean_tokens > 0
        assert chebi.build_version == config.build_version()
        assert chebi.answer_kind == "text" and chebi.max_new_tokens == 256

    graph_tokens = M.task_specs(config, arm="graph")[
        f"{MOLECULE_PREFIX}ring_membership"].mean_tokens
    flat_tokens = M.task_specs(config, arm="flat")[
        f"{MOLECULE_PREFIX}ring_membership"].mean_tokens
    assert graph_tokens > flat_tokens, (
        "a Levi graph is many more tokens than one SMILES string; if these were "
        "equal, tokens_per_step would resolve to the same batch for both arms")

    specs = M.task_specs(config)
    assert specs[f"{MOLECULE_PREFIX}g2s"].answer_kind == "smiles"
    assert specs[f"{MOLECULE_PREFIX}g2s"].kind == "generator"
    assert specs[f"{MOLECULE_PREFIX}tox21"].answer_kind == "yesno"
    assert specs[f"{MOLECULE_PREFIX}tox21"].kind == "corpus"
    assert specs[f"{MOLECULE_PREFIX}g2s"].question_template == M.G2S_QUESTION


def test_a_held_out_task_is_measured_off_its_held_out_split(built):
    """So an `adapt` fork can resolve a mixture over it (D6).

    `held_out` is the only split a held-out task has and the only one anything
    ever trains on. Without a measurement there, `registry.resolve` has no
    ``mean_tokens`` to turn a token budget into examples and the one fork that is
    allowed to train the task cannot be planned — a gap that would only surface
    at the moment the fork was submitted.
    """
    from src.generalist.registry import resolve as resolve_mixture

    config, _manifest = built
    spec = M.task_specs(config)[f"{MOLECULE_PREFIX}bond_path"]
    assert spec.mean_tokens and spec.mean_tokens > 0
    assert spec.train_size == config.held_out_size

    registry = M.register_molecule_tasks(Registry(), config)
    name = f"{MOLECULE_PREFIX}bond_path"
    mixture = resolve_mixture(registry, [{"name": name, "weight": 1.0}],
                              tokens_per_step=1000, steps=4,
                              min_examples_per=0, allow_held_out=(name,))
    assert mixture.steps == 4
    assert mixture.entries[0].name == name


def test_load_never_regenerates(built):
    """An absent artifact names the data_prep that should have made it."""
    config, _manifest = built
    with pytest.raises(M.AdapterBuildError, match="data_prep"):
        M.load(f"{MOLECULE_PREFIX}fg_count", "train", "graph", config=config)
    with pytest.raises(M.AdapterBuildError, match="data_prep"):
        M.load(f"{MOLECULE_PREFIX}g2s", "train", "graph", pass_id=7,
               config=config)


def test_the_adapter_resolves_through_the_protocol():
    adapter = get_adapter("molecules")
    assert adapter is M
    for name in ("build", "load", "partition"):
        assert callable(getattr(adapter, name))


def test_a_pass_is_a_fresh_draw(built):
    """D4.2: a generator draws a new pass from the train-role pool, not a repeat."""
    config, _manifest = built
    M.build(config, tasks=("g2s",), arms=("flat",), splits=("train",), passes=2)
    first = M.load(f"{MOLECULE_PREFIX}g2s", "train", "flat", pass_id=0,
                   config=config)
    second = M.load(f"{MOLECULE_PREFIX}g2s", "train", "flat", pass_id=1,
                    config=config)
    assert first.pass_id == 0 and second.pass_id == 1
    assert len(first) == len(second)

    def smiles_of(source):
        out = []
        for i in range(len(source)):
            item = source[i]
            text = item["text"][item["prompt_node"]]
            out.append(text.split("\nSMILES: ", 1)[1].split("\nA:", 1)[0])
        return out

    # Same task, different pass: the randomization is re-seeded, so the flat
    # twin cannot memorise one spelling of a molecule it will meet again (§5).
    assert smiles_of(first) != smiles_of(second)


def test_both_arms_see_the_same_draw(built):
    """The two arms differ in representation and in nothing else."""
    config, _manifest = built
    graph = M.load(f"{MOLECULE_PREFIX}ring_membership", "train", "graph",
                   config=config)
    flat = M.load(f"{MOLECULE_PREFIX}ring_membership", "train", "flat",
                  config=config)
    assert graph.keys() == flat.keys()
    for i in range(len(graph)):
        assert graph[i]["_schema"]["question"] == flat[i]["_schema"]["question"]
        assert graph[i]["_schema"]["answer"] == flat[i]["_schema"]["answer"]
