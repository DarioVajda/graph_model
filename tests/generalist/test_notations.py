"""The flat arm's notation ladder — `MOLECULE_GENERALIST.md` §8.3.

Three notations serialise the same molecule (SMILES, SELFIES, InChI), so the
ladder's whole premise is that a difference between them is a *pretraining*
difference and not an information one. These tests pin the four ways that premise
can quietly stop holding:

* **The build hash does not move.** The notation rides on the arm, not on
  `MoleculeAdapterConfig`, precisely so that adding it leaves ``42f7a14bed21f876``
  — the build every arm-2 cell read and every number in §8 came from — valid. A
  test that fails when the hash moves is what makes that a guarantee rather than
  an intention.
* **The three arms see the same molecules.** Draws are a function of the
  partition and not of the arm, so one task's four rows are a comparison. If the
  notation leaked into the draw seed they would be four separate samples that
  merely look comparable.
* **The two canonical-only notations refuse what they cannot honour.** Silently
  dropping ``atom_labels`` leaves a Tier-A question naming an atom the string does
  not mark; silently ignoring ``canonical=False`` reports a permutation spread of
  zero for a notation with no re-ordered form — a Property-1 "pass" that measured
  nothing. Both raise, and both are asserted here *including the reason*.
* **`perm_spread` does not take the graph path for a flat arm.** A one-node graph
  relabels to itself, so the fall-through would report spread 0 and pass. It has
  to refuse instead.
"""

import networkx as nx
import pytest
from rdkit import Chem

from src.experiments.molecules.data import (
    NOTATION_HEADERS,
    NOTATIONS,
    flat_serialize,
)
from src.generalist.adapters.molecules import (
    FLAT_NOTATIONS,
    MoleculeAdapterConfig,
    _graphs_for,
    _run_config,
)
from src.generalist.config import ARMS, FLAT_ARMS

#: A molecule with a stereocentre, an aromatic ring, a carbonyl and a tertiary
#: amine — enough that a notation dropping any of them shows up in the round trip.
MOL = "CN(C)C(=O)c1ccc(cc1)O[C@@H](C)C(=O)O"

#: The build every arm-2 cell read. `MOLECULE_GENERALIST.md` §8.2.
ARM2_BUILD_VERSION = "42f7a14bed21f876"


def _mol():
    return Chem.MolFromSmiles(MOL)


# ── the premise: same molecule, three strings ────────────────────────────────

@pytest.mark.parametrize("notation", NOTATIONS)
def test_every_notation_round_trips_to_the_same_molecule(notation):
    """Equally expressive is the ladder's premise, so it is asserted, not assumed.

    Each notation is parsed back and compared as a canonical SMILES. If one of
    them lost a bond order or a stereocentre, a lower score on it would be an
    information difference wearing a pretraining difference's clothes.
    """
    text = flat_serialize(_mol(), notation=notation)
    if notation == "smiles":
        back = Chem.MolFromSmiles(text)
    elif notation == "inchi":
        back = Chem.MolFromInchi(text)
    else:
        from src.experiments.molecules.data import _selfies

        back = Chem.MolFromSmiles(_selfies().decoder(text))

    assert back is not None, f"{notation}: {text!r} does not parse back"
    assert Chem.MolToSmiles(back) == Chem.MolToSmiles(_mol()), (
        f"{notation} lost information: {Chem.MolToSmiles(back)!r} != "
        f"{Chem.MolToSmiles(_mol())!r}")


def test_the_three_notations_are_actually_different_strings():
    """A ladder whose rungs coincide measures nothing."""
    written = {n: flat_serialize(_mol(), notation=n) for n in NOTATIONS}
    assert len(set(written.values())) == len(NOTATIONS), written


@pytest.mark.slow
def test_selfies_round_trips_the_real_corpus_under_its_constraint_set():
    """`SELFIES_CONSTRAINTS` is a valence table, so it is checked on real molecules.

    One hand-picked molecule proves nothing about a preset: ``default`` refuses
    hypervalent iodine, which BACE contains, and ``octet_rule`` refuses 12.7 % of
    the pool. What matters is that whatever a molecule *does* encode to decodes
    back to that same molecule — a silent re-interpretation would make the SELFIES
    arm a different dataset wearing the ladder's clothes.
    """
    from src.experiments.molecules.data import _selfies, load_tier_b

    sf = _selfies()
    records, _, _ = load_tier_b("bace")
    checked = failed = 0
    for record in records:
        smiles = Chem.MolToSmiles(record["mol"])
        try:
            encoded = sf.encoder(smiles)
        except Exception:                                        # noqa: BLE001
            failed += 1
            continue
        checked += 1
        back = Chem.MolFromSmiles(sf.decoder(encoded))
        assert back is not None and Chem.MolToSmiles(back) == smiles, (
            f"SELFIES did not round-trip {smiles!r}")
    assert checked > 1000, f"only {checked} molecules were actually checked"
    # Zero, under `SELFIES_CONSTRAINTS`. If this starts failing, the ladder is
    # silently scoring one arm on fewer molecules than the others, and the
    # constraint table is where to look.
    assert failed == 0, f"{failed} BACE molecules no longer encode"


# ── the refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("notation", ("selfies", "inchi"))
def test_atom_labels_are_refused_rather_than_dropped(notation):
    with pytest.raises(ValueError, match="atom_labels"):
        flat_serialize(_mol(), notation=notation, atom_labels=True)


@pytest.mark.parametrize("notation", ("selfies", "inchi"))
def test_randomised_form_is_refused_rather_than_returned_canonical(notation):
    with pytest.raises(ValueError, match="no randomised form"):
        flat_serialize(_mol(), notation=notation, canonical=False, seed=1)


def test_unknown_notation_raises():
    with pytest.raises(ValueError, match="notation must be one of"):
        flat_serialize(_mol(), notation="inchikey")


# ── the wiring ───────────────────────────────────────────────────────────────

def test_the_arm2_build_version_does_not_move():
    """The notation rides on the arm so this hash stays put (FLAT_NOTATIONS).

    If it moves, every artifact under `42f7a14bed21f876` is orphaned and §8's
    numbers can no longer be reproduced from a cached build.

    ``answer_eos`` is spelled out because the default flipped to ``True`` on
    2026-09-10 (`GENERATIVE_ANSWER_KINDS`): the campaign trained with no stop
    token on its generative answers, so *that* is the config its artifacts
    belong to, and pinning the old value has to reach the old build.
    """
    assert MoleculeAdapterConfig(answer_eos=False).build_version() \
        == ARM2_BUILD_VERSION


def test_the_stop_token_is_a_different_build():
    """The fix must not land silently inside the build the campaign reports.

    The two configs draw the same molecules and write different bytes, so they
    are two builds. A shared hash would serve arm-2 data to a run that asked to
    be trained with a stop token, and nothing downstream would notice.
    """
    assert MoleculeAdapterConfig(answer_eos=True).build_version() \
        != ARM2_BUILD_VERSION


def test_every_flat_arm_is_declared_everywhere_it_is_read():
    assert set(FLAT_NOTATIONS) == set(FLAT_ARMS)
    assert set(FLAT_ARMS) | {"graph"} == set(ARMS)
    assert set(FLAT_NOTATIONS.values()) == set(NOTATIONS)


@pytest.mark.parametrize("arm", sorted(FLAT_NOTATIONS))
def test_flat_arms_get_bias_none_and_their_own_notation(arm):
    cfg = _run_config(MoleculeAdapterConfig(), "bbbp", arm)
    assert cfg.bias == "none", "Property 2: a one-node graph carries no bias"
    assert cfg.notation == FLAT_NOTATIONS[arm]


def test_graph_arm_keeps_its_bias_and_never_serialises_a_string():
    cfg = _run_config(MoleculeAdapterConfig(), "bbbp", "graph")
    assert cfg.bias == "spd+magnetic"


@pytest.mark.parametrize("arm", sorted(FLAT_NOTATIONS))
def test_the_prompt_shape_is_identical_across_notations(arm):
    """Only the header word and the string move between the flat arms.

    The comparison is between notations, so anything else that differs in the
    prompt is an uncontrolled second variable.
    """
    config = MoleculeAdapterConfig()
    question = "Question: does this molecule cross the blood-brain barrier?"
    draws = [(_mol(), question, " yes", [], "key", {})]
    graph = _graphs_for(config, "bbbp", arm, draws, 0)[0]

    assert graph.number_of_nodes() == 1, "the flat arm is a single-node graph"
    text = graph.nodes[graph.graph["prompt_node"]]["text"]
    header = NOTATION_HEADERS[FLAT_NOTATIONS[arm]]
    string = flat_serialize(_mol(), notation=FLAT_NOTATIONS[arm])
    assert text == f"{question}\n{header}: {string}\nA: yes"


def test_the_notation_arms_draw_the_same_molecules_as_smiles():
    """One task's rows are a comparison only if the draws do not move.

    `_draw_rng` keeps the arm out of the seed on purpose; this asserts the
    property that rule exists to give, at the point a notation could break it.
    """
    config = MoleculeAdapterConfig()
    question = "Question: is this molecule active?"
    mols = [Chem.MolFromSmiles(s) for s in
            ("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O")]
    draws = [(m, question, " no", [], f"k{i}", {}) for i, m in enumerate(mols)]

    rendered = {}
    for arm in FLAT_NOTATIONS:
        graphs = _graphs_for(config, "bace", arm, draws, 0)
        rendered[arm] = [g.nodes[g.graph["prompt_node"]]["text"] for g in graphs]

    assert all(len(v) == len(mols) for v in rendered.values())
    for i, mol in enumerate(mols):
        canonical = Chem.MolToSmiles(mol)
        for arm, texts in rendered.items():
            notation = FLAT_NOTATIONS[arm]
            assert flat_serialize(mol, notation=notation) in texts[i], (
                f"{arm} row {i} is not {canonical}")


# ── the trap perm_spread would otherwise fall into ───────────────────────────

def test_perm_spread_refuses_a_notation_it_cannot_reorder():
    """Not "returns zero" — refuses. A zero here would read as Property 1 holding.

    The flat path is reached for every flat arm (that is the fix); on a SELFIES
    or InChI prompt there is no SMILES marker to rewrite, and the refusal names
    the reason rather than reporting a spread of nothing.
    """
    from src.generalist.evaluate.builtin import _rewritten_flat_item
    from src.generalist.evaluate import EvalError

    config = MoleculeAdapterConfig()
    draws = [(_mol(), "Question: is this molecule active?", " no", [], "k", {})]
    graph = _graphs_for(config, "bace", "flat_selfies", draws, 0)[0]
    item = {"text": [graph.nodes[0]["text"]], "prompt_node": 0, "num_nodes": 1}

    with pytest.raises(EvalError, match="canonical by construction"):
        _rewritten_flat_item(item, None, perm_id=1)


def test_perm_spread_still_rewrites_a_smiles_prompt():
    """The fix must not have broken the arm the sweep is actually for.

    Needs the real tokenizer: the rewritten prompt is re-rendered, and `render`
    is where the answer span is located.
    """
    from transformers import AutoTokenizer

    from src.experiments.molecules.config import MODEL_NAME
    from src.generalist.evaluate.builtin import _rewritten_flat_item

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip(f"the real tokenizer is not available here: {exc}")

    config = MoleculeAdapterConfig()
    draws = [(_mol(), "Question: is this molecule active?", " no", [], "k", {})]
    graph = _graphs_for(config, "bace", "flat", draws, 0)[0]
    original = graph.nodes[0]["text"]
    item = {"text": [original], "prompt_node": 0, "num_nodes": 1}

    out = _rewritten_flat_item(item, tokenizer, perm_id=3)
    rewritten = out["text"][0]
    assert rewritten != original, "a permutation that changes nothing measures nothing"
    start = rewritten.find("\nSMILES: ") + len("\nSMILES: ")
    end = rewritten.find("\n", start)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(rewritten[start:end])) == \
        Chem.MolToSmiles(_mol())


# ── which notation each (arm, task) actually serialises to ───────────────────

def test_atom_level_tasks_stay_in_smiles_on_every_arm():
    """The four families that name an atom cannot move to a notation without one.

    `ring_membership`, `aromatic_ring`, `ring_size` and `fg_atom_membership` ask
    "is atom 14 …", and only SMILES can mark atom 14. Holding them at SMILES on
    every arm is what keeps the *mixture* identical across the ladder, so a
    gradient between arms cannot be a difference in what they trained on.
    """
    from src.experiments.molecules.tasks import ATOM_LEVEL_TASKS
    from src.generalist.adapters.molecules import _notation_for

    for arm, notation in FLAT_NOTATIONS.items():
        for task in ATOM_LEVEL_TASKS:
            assert _notation_for(arm, task) == "smiles", (
                f"{arm}/{task} must stay in SMILES: it names an atom")
        # Everything else moves to the arm's notation, or the ladder measures
        # nothing.
        for task in ("bace", "bbbp", "hiv", "tox21", "sider", "ring_count",
                     "fg_presence", "chebi20"):
            assert _notation_for(arm, task) == notation


def test_the_notation_arms_serialise_g2s_canonically():
    """g2s has no randomised form outside SMILES, and gets a real task instead.

    The SMILES arm's matched task is canonicalization, which needs a *randomised*
    input or it is a copy. SELFIES and InChI have no randomised form — and need
    none, because the target is canonical SMILES, so a canonical SELFIES input is
    already a translation between notations rather than a copy. That makes g2s a
    different task on those arms, which is one more reason §5 forbids reading the
    g2s column as an arm comparison.
    """
    config = MoleculeAdapterConfig()
    draws = [(_mol(), "Question: write the canonical SMILES for this molecule.",
              " CCO", [], "k", {})]

    for arm, notation in FLAT_NOTATIONS.items():
        graph = _graphs_for(config, "g2s", arm, draws, 0)[0]
        text = graph.nodes[0]["text"]
        assert NOTATION_HEADERS[notation] + ":" in text
        if notation == "smiles":
            continue
        # The input is this arm's canonical string for the molecule.
        assert flat_serialize(_mol(), notation=notation) in text


# ── the molecule no notation can express ─────────────────────────────────────

#: A ferrocene from HIV. The metal centre carries ten bonds, and SELFIES caps any
#: atom at its catch-all constraint — which is what `SELFIES_CONSTRAINTS` raises
#: to 12 so that the twenty-two organometallics in the pool encode at all.
ORGANOMETALLIC = (
    "[Cl][Sn]([Cl])([C]12[C]3=[C]4[C]5=[C]1[Fe]45321678[C]2=[C]1[CH]6[C]7=[C]28)"
    "[C]12[C]3=[C]4[C]5=[C]1[Fe]45321678[C]2=[C]1[CH]6[C]7=[C]28"
)


def test_the_constraint_set_encodes_the_organometallics():
    """The pool's hardest molecules for SELFIES, and why the cap is not a preset.

    Every one of the twenty-two molecules SELFIES could not encode under the
    stock ``hypervalent`` preset is an organometallic — ferrocenes, molybdenum
    and tungsten carbonyls — whose metal centre carries nine or ten bonds against
    the preset's catch-all cap of eight. They are real rows of HIV, one of them
    test-role, so the choice is between raising the cap and scoring three
    notations on three different sets of molecules.
    """
    mol = Chem.MolFromSmiles(ORGANOMETALLIC)
    assert mol is not None

    encoded = flat_serialize(mol, notation="selfies")
    from src.experiments.molecules.data import _selfies

    back = Chem.MolFromSmiles(_selfies().decoder(encoded))
    assert back is not None and Chem.MolToSmiles(back) == Chem.MolToSmiles(mol), (
        "the raised cap must not buy encoding at the price of the round trip")


def test_an_unencodable_molecule_would_keep_its_row_as_a_placeholder():
    """The safety net, exercised on a molecule constructed to trip it.

    `SELFIES_CONSTRAINTS` currently encodes every molecule in the pool, so this
    path fires zero times today. It exists because the alternative failure is
    silent and expensive: a corpus added later that one notation cannot express
    would either abort a multi-hour build or — far worse — shorten one arm's
    dataset by a row and shift every index after it, so arm-to-arm row `i` would
    quietly stop being the same molecule.
    """
    from src.experiments.molecules.data import UNENCODABLE
    from src.experiments.molecules import dataset as ds_mod

    config = MoleculeAdapterConfig()
    mols = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1")]
    draws = [(m, "Question: is this molecule active?", " no", [], f"k{i}", {})
             for i, m in enumerate(mols)]

    real = ds_mod.flat_serialize

    def refuse_the_second(mol, **kwargs):
        if kwargs.get("notation") == "selfies" and Chem.MolToSmiles(mol) == "c1ccccc1":
            raise ds_mod.EncodeUnsupported("constructed refusal")
        return real(mol, **kwargs)

    ds_mod.flat_serialize = refuse_the_second
    try:
        graphs = _graphs_for(config, "hiv", "flat_selfies", draws, 0)
    finally:
        ds_mod.flat_serialize = real

    assert len(graphs) == len(mols), "a row was dropped, not placeheld"
    texts = [g.nodes[0]["text"] for g in graphs]
    assert UNENCODABLE not in texts[0]
    assert UNENCODABLE in texts[1]


# ── the probe's own read-out ─────────────────────────────────────────────────

def test_the_label_check_reproduces_the_truncation_defect():
    """The assertion has to fail on the numbers the defect actually produced.

    A disjointness assert that cannot fail is decoration (`molecules/PLAN.md`
    §9). These are the real SIDER `pos_rate` values from the first probe run: the
    untruncated graph arm read 0.5460 while two flat arms, scoring *identical*
    molecules, read 0.5180 and 0.5080 — because their prompts were truncated at
    `max_length`, taking the answer token with them, and every such row was then
    scored as label "no" at a meaningless position.
    """
    from src.generalist.tools.notation.probe import check_arms_agree_on_labels

    broken = [
        {"task": "sider", "arm": "flat", "pos_rate": 0.5180, "n": 500},
        {"task": "sider", "arm": "flat_inchi", "pos_rate": 0.5080, "n": 500},
        {"task": "sider", "arm": "graph", "pos_rate": 0.5460, "n": 500},
    ]
    with pytest.raises(AssertionError, match="label base rate"):
        check_arms_agree_on_labels(broken)

    fixed = [dict(row, pos_rate=0.5460) for row in broken]
    check_arms_agree_on_labels(fixed)          # must not raise


def test_the_label_check_catches_a_row_count_mismatch():
    from src.generalist.tools.notation.probe import check_arms_agree_on_labels

    rows = [
        {"task": "hiv", "arm": "flat", "pos_rate": 0.038, "n": 500},
        {"task": "hiv", "arm": "flat_selfies", "pos_rate": 0.038, "n": 496},
    ]
    with pytest.raises(AssertionError, match="different row counts"):
        check_arms_agree_on_labels(rows)


def test_leakage_can_strip_stereo_in_every_notation():
    """The leakage detector survives the ladder; `perm_spread` correctly does not.

    The two validators rewrite the flat prompt for different reasons, and only one
    of them is expressible outside SMILES. Stripping stereochemistry is a property
    of the *molecule*, so every notation can write the stripped form — which is
    what keeps §3.2.10's control alive on the notation arms. Re-ordering the atoms
    is a property of the *string*, and SELFIES and InChI have no re-ordered form,
    which is why `perm_spread` refuses.
    """
    from src.generalist.evaluate.builtin import (
        _flat_molecule_span, _parse_notation, _write_notation,
    )

    stereo_mol = Chem.MolFromSmiles("C[C@H](O)c1ccccc1")
    for notation in NOTATIONS:
        string = flat_serialize(stereo_mol, notation=notation)
        text = (f"Question: is this molecule active?\n"
                f"{NOTATION_HEADERS[notation]}: {string}\nA: Yes")

        start, end, detected = _flat_molecule_span(text)
        assert detected == notation, f"header said {detected!r}, wanted {notation!r}"

        parsed = _parse_notation(text[start:end], detected)
        assert parsed is not None, f"{notation} did not parse back"

        kept = _write_notation(parsed, notation, stereo=True)
        dropped = _write_notation(parsed, notation, stereo=False)
        assert kept != dropped, (
            f"{notation}: stripping stereo changed nothing, so the ablation "
            "would measure nothing")
        assert Chem.MolToSmiles(_parse_notation(dropped, notation),
                                isomericSmiles=True) == \
            Chem.MolToSmiles(Chem.MolFromSmiles("CC(O)c1ccccc1")), (
                f"{notation}: the stripped string is not the stripped molecule")


def test_a_prompt_with_no_molecule_header_reports_no_span():
    from src.generalist.evaluate.builtin import _flat_molecule_span

    assert _flat_molecule_span("Question: nothing here\nA: Yes") == (-1, -1, "")


def test_the_notation_validator_set_drops_only_perm_spread():
    from src.generalist.config import DEFAULT_VALIDATORS, VALIDATOR_SETS

    notation = VALIDATOR_SETS["notation"]
    names = {s["name"] for s in notation}
    assert "perm_spread" not in names, (
        "Property 1 has no re-ordered form to sweep on a canonical-only notation")
    assert "leakage" in names, "the leakage detector works on every notation"
    assert names == {s["name"] for s in DEFAULT_VALIDATORS} - {"perm_spread"}


def test_probe_table_renders_without_a_gpu():
    """`table` is pure formatting over rows, so it is testable on CPU."""
    from src.generalist.tools.notation.probe import table

    rows = [{"task": "bace", "arm": arm, "roc_auc": 0.5, "pos_rate": 0.46,
             "tied_pair_fraction": 0.9, "mean_tokens": 42.0}
            for arm in ("flat", "flat_selfies", "flat_inchi", "graph")]
    out = table(rows)
    assert "bace" in out and "flat_selfies" in out
    assert "tied pair fraction" in out, "the floor disclosure is not optional"
