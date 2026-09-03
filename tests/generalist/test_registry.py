"""
T6 — the registry (`src/generalist/registry.py`, DESIGN.md §D2).

What has to hold, and why each one is here rather than left to a report:

* a **held-out** task named in a training mixture fails at ``validate``, before
  any data is built, whichever of the two enforcement sources declares it
  (D2.1);
* a **sub-threshold share** fails, because a task that is in the config and
  absent from the gradient is the ``--magnetic-groups`` bug (`PLAN.md` §10);
* the **budget and step count** come out as documented on a fixture whose
  arithmetic is checkable by hand — that is the number
  `MOLECULE_GENERALIST.md` §2 says the registry computes rather than takes;
* the **snapshot and hash** are insertion-order independent and carry no
  callables, because they go into every checkpoint and a hash that moved
  between two processes would make the resume's mixture-change detection noise.
"""

import pytest

from src.generalist.registry import (
    Mixture,
    Registry,
    RegistryError,
    TaskSpec,
    is_held_out,
    molecule_held_out_names,
    resolve,
)

# ─────────────────────────────────────────────────────────────────────────────
# The fixture: two corpora of different sizes and passes, and one generator.
#
# Chosen so every number below is exact.
#
#   available(bace)   = 3 passes x 1000 = 3000     share 0.25
#   available(chebi)  = 2 passes x 2000 = 4000     share 0.25
#   g2s is a generator: it draws a fresh pass every time and never binds.
#
#   budget = min(3000 / 0.25, 4000 / 0.25) = 12000, bound by mol/bace
#   per task = 3000 / 3000 / 6000
#   share-weighted mean tokens = .25*100 + .25*200 + .5*50 = 100
#   examples/step at 1000 tokens/step = 1000 / 100 = 10
#   steps = 12000 / 10 = 1200
# ─────────────────────────────────────────────────────────────────────────────

SPECS = (
    dict(name="mol/bace", domain="molecules", adapter="molecules", kind="corpus",
         answer_kind="yesno", metric="roc_auc", passes=3, train_size=1000,
         mean_tokens=100.0),
    dict(name="mol/chebi20", domain="molecules", adapter="molecules",
         kind="corpus", answer_kind="text", metric="bleu2", passes=2,
         train_size=2000, mean_tokens=200.0, max_new_tokens=128),
    dict(name="mol/g2s", domain="molecules", adapter="molecules",
         kind="generator", answer_kind="smiles", metric="roundtrip_match",
         passes=1, cap_per_pass=5000, mean_tokens=50.0, max_new_tokens=96),
)

MIXTURE = [
    {"name": "mol/bace", "weight": 1.0},
    {"name": "mol/chebi20", "weight": 1.0},
    {"name": "mol/g2s", "weight": 2.0},
]

TOKENS_PER_STEP = 1000


def registry(order=None, **patch):
    """The fixture registry. ``order`` permutes registration; ``patch`` edits one spec."""
    specs = {s["name"]: dict(s) for s in SPECS}
    for name, changes in patch.items():
        specs[name.replace("__", "/")].update(changes)
    names = order or list(specs)
    return Registry(TaskSpec(**specs[n]) for n in names)


def resolved(**kwargs):
    return resolve(registry(**kwargs), MIXTURE, TOKENS_PER_STEP)


# ─────────────────────────────────────────────────────────────────────────────
# Budget, shares, steps
# ─────────────────────────────────────────────────────────────────────────────


def test_shares_normalise_to_one():
    mixture = resolved()
    assert mixture.shares == pytest.approx(
        {"mol/bace": 0.25, "mol/chebi20": 0.25, "mol/g2s": 0.5})
    assert sum(mixture.shares.values()) == pytest.approx(1.0)


def test_budget_is_set_by_the_binding_corpus():
    mixture = resolved()
    assert mixture.binding_task == "mol/bace"
    assert mixture.budget_examples == 12000
    assert mixture.per_task_examples == {
        "mol/bace": 3000, "mol/chebi20": 3000, "mol/g2s": 6000}


def test_no_corpus_exceeds_its_pass_cap():
    """The property the min rule exists for: three passes means at most three."""
    mixture = resolved()
    for entry in mixture.entries:
        if entry.available is not None:
            assert entry.examples <= entry.available


def test_the_generator_never_bounds_the_budget():
    """Halving the generator's cap does not move the budget or the steps."""
    base = resolved()
    capped = resolve(registry(mol__g2s={"cap_per_pass": 10}), MIXTURE,
                     TOKENS_PER_STEP)
    assert capped.budget_examples == base.budget_examples
    assert capped.steps == base.steps
    assert capped.entries[-1].available is None


def test_examples_per_step_and_steps():
    mixture = resolved()
    assert mixture.mean_tokens == pytest.approx(100.0)
    assert mixture.examples_per_step == pytest.approx(10.0)
    assert mixture.steps == 1200
    assert mixture.tokens_per_step == TOKENS_PER_STEP


def test_tokens_per_step_moves_the_steps_not_the_budget():
    """D4.4: the token budget derives the batch, it does not change the data."""
    doubled = resolve(registry(), MIXTURE, 2 * TOKENS_PER_STEP)
    assert doubled.budget_examples == 12000
    assert doubled.examples_per_step == pytest.approx(20.0)
    assert doubled.steps == 600


def test_a_longer_task_buys_fewer_examples_per_step():
    """A captioning task in the mixture is why mean_tokens cannot be a constant."""
    longer = resolve(registry(mol__chebi20={"mean_tokens": 600.0}), MIXTURE,
                     TOKENS_PER_STEP)
    assert longer.mean_tokens == pytest.approx(200.0)
    assert longer.examples_per_step == pytest.approx(5.0)
    assert longer.steps == 2400


def test_mixture_entries_are_sorted_and_carry_the_resolved_draw():
    mixture = resolved()
    assert [e.name for e in mixture.entries] == \
        ["mol/bace", "mol/chebi20", "mol/g2s"]
    by_name = {e.name: e for e in mixture.entries}
    assert by_name["mol/bace"].passes == 3
    assert by_name["mol/chebi20"].available == 4000
    assert by_name["mol/g2s"].cap_per_pass == 5000
    assert isinstance(mixture, Mixture)


def test_a_mixture_entry_overrides_the_spec_default():
    """One registry, several configs: passes and weight are per-run."""
    mixture = resolve(registry(), [
        {"name": "mol/bace", "weight": 1.0, "passes": 6},
        {"name": "mol/chebi20", "weight": 1.0},
        {"name": "mol/g2s", "weight": 2.0, "cap_per_pass": 99},
    ], TOKENS_PER_STEP)
    # bace now offers 6000, so chebi's 4000 binds: budget = 4000 / 0.25.
    assert mixture.binding_task == "mol/chebi20"
    assert mixture.budget_examples == 16000
    assert {e.name: e.cap_per_pass for e in mixture.entries}["mol/g2s"] == 99


def test_the_mixture_table_names_the_binding_task():
    table = resolved().table()
    assert "mol/bace" in table and "12000 examples over 1200 steps" in table


# ─────────────────────────────────────────────────────────────────────────────
# What resolve refuses
# ─────────────────────────────────────────────────────────────────────────────


def test_an_unregistered_task_fails():
    with pytest.raises(RegistryError, match="mol/nosuch: not registered"):
        resolve(registry(), MIXTURE + [{"name": "mol/nosuch", "weight": 1.0}],
                TOKENS_PER_STEP)


def test_the_molecules_declaration_holds_a_task_out_on_its_own():
    """No ``held_out`` flag on the spec: `molecules/data.py` is the other source."""
    reg = registry()
    reg.register(TaskSpec(name="mol/bond_path", domain="molecules",
                          adapter="molecules", kind="generator",
                          answer_kind="token", mean_tokens=60.0))
    assert is_held_out(reg.get("mol/bond_path"))
    with pytest.raises(RegistryError, match="mol/bond_path: held out"):
        resolve(reg, MIXTURE + [{"name": "mol/bond_path", "weight": 1.0}],
                TOKENS_PER_STEP)


def test_an_explicit_held_out_flag_holds_a_task_out_on_its_own():
    """The flag is a second, independent enforcement point.

    All three of the molecule holdouts now live in the molecules tuples, so this
    uses a task that is in neither: a campaign must be able to hold something out
    without amending the molecules package, which is the whole reason
    :func:`is_held_out` ORs the two sources.
    """
    reg = registry()
    reg.register(TaskSpec(name="mol/atom_count", domain="molecules",
                          adapter="molecules", kind="generator",
                          answer_kind="token", held_out=True, mean_tokens=60.0))
    assert "mol/atom_count" not in molecule_held_out_names()
    assert is_held_out(reg.get("mol/atom_count"))
    with pytest.raises(RegistryError, match="mol/atom_count: held out"):
        resolve(reg, MIXTURE + [{"name": "mol/atom_count", "weight": 1.0}],
                TOKENS_PER_STEP)


def test_longest_chain_is_held_out_by_the_molecules_tuples():
    """It joined ``HELD_OUT_TIER_A_TASKS`` on 2026-09-02 (MOLECULE_GENERALIST.md §4)."""
    reg = registry()
    reg.register(TaskSpec(name="mol/longest_chain", domain="molecules",
                          adapter="molecules", kind="generator",
                          answer_kind="token", mean_tokens=60.0))
    assert "mol/longest_chain" in molecule_held_out_names()
    assert is_held_out(reg.get("mol/longest_chain"))
    with pytest.raises(RegistryError, match="mol/longest_chain: held out"):
        resolve(reg, MIXTURE + [{"name": "mol/longest_chain", "weight": 1.0}],
                TOKENS_PER_STEP)


def test_clintox_is_held_out():
    reg = registry()
    reg.register(TaskSpec(name="mol/clintox", domain="molecules",
                          adapter="molecules", kind="corpus",
                          answer_kind="yesno", train_size=1000,
                          mean_tokens=90.0))
    with pytest.raises(RegistryError, match="mol/clintox: held out"):
        resolve(reg, MIXTURE + [{"name": "mol/clintox", "weight": 1.0}],
                TOKENS_PER_STEP)


def test_molecule_held_out_names_mirrors_the_molecules_package():
    names = molecule_held_out_names()
    assert "mol/bond_path" in names
    assert "mol/clintox" in names
    assert all(n.startswith("mol/") for n in names)


def test_a_sub_threshold_share_fails():
    """D2.2: one example per 1000 steps is the floor for "this task trains"."""
    reg = registry()
    reg.register(TaskSpec(name="mol/tox21", domain="molecules",
                          adapter="molecules", kind="corpus",
                          answer_kind="yesno", train_size=50000,
                          mean_tokens=100.0))
    # At ~10 examples/step a share under 1e-4 buys under one example per 1000
    # steps; 1e-7 of the weight is far under it and 1e-2 is comfortably over.
    with pytest.raises(RegistryError, match=r"mol/tox21: share .* under one example"):
        resolve(reg, MIXTURE + [{"name": "mol/tox21", "weight": 1e-7}],
                TOKENS_PER_STEP)
    resolve(reg, MIXTURE + [{"name": "mol/tox21", "weight": 1e-2}],
            TOKENS_PER_STEP)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan")])
def test_a_non_positive_weight_fails(weight):
    with pytest.raises(RegistryError, match="mol/bace: weight"):
        resolve(registry(), [{"name": "mol/bace", "weight": weight}] + MIXTURE[1:],
                TOKENS_PER_STEP)


def test_missing_mean_tokens_fails():
    with pytest.raises(RegistryError, match="mol/chebi20: mean_tokens"):
        resolve(registry(mol__chebi20={"mean_tokens": None}), MIXTURE,
                TOKENS_PER_STEP)


def test_missing_train_size_on_a_corpus_fails():
    with pytest.raises(RegistryError, match="mol/bace: train_size"):
        resolve(registry(mol__bace={"train_size": None}), MIXTURE,
                TOKENS_PER_STEP)


def test_a_generator_only_mixture_has_no_budget():
    with pytest.raises(RegistryError, match="budget: the mixture has no corpus"):
        resolve(registry(), [{"name": "mol/g2s", "weight": 1.0}], TOKENS_PER_STEP)


def test_a_duplicated_task_fails():
    with pytest.raises(RegistryError, match="mol/bace: listed twice"):
        resolve(registry(), MIXTURE + [{"name": "mol/bace", "weight": 1.0}],
                TOKENS_PER_STEP)


def test_an_empty_mixture_fails():
    with pytest.raises(RegistryError, match="mixture: empty"):
        resolve(registry(), [], TOKENS_PER_STEP)


@pytest.mark.parametrize("tokens", [0, -1, None])
def test_a_non_positive_tokens_per_step_fails(tokens):
    with pytest.raises(RegistryError, match="tokens_per_step"):
        resolve(registry(), MIXTURE, tokens)


def test_a_bad_spec_fails_at_construction():
    with pytest.raises(RegistryError, match="kind must be one of"):
        TaskSpec(name="mol/x", domain="molecules", adapter="molecules",
                 kind="stream")
    with pytest.raises(RegistryError, match="answer_kind must be one of"):
        TaskSpec(name="mol/x", domain="molecules", adapter="molecules",
                 answer_kind="rhyme")
    with pytest.raises(RegistryError, match="loss_norm must be one of"):
        TaskSpec(name="mol/x", domain="molecules", adapter="molecules",
                 loss_norm="per_batch")


def test_registering_a_name_twice_fails():
    reg = registry()
    with pytest.raises(RegistryError, match="mol/bace: already registered"):
        reg.register(TaskSpec(name="mol/bace", domain="molecules",
                              adapter="molecules"))


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot and hash
# ─────────────────────────────────────────────────────────────────────────────


def test_snapshot_and_hash_are_insertion_order_independent():
    forward = registry(order=["mol/bace", "mol/chebi20", "mol/g2s"])
    backward = registry(order=["mol/g2s", "mol/chebi20", "mol/bace"])
    assert forward.snapshot() == backward.snapshot()
    assert forward.hash() == backward.hash()
    assert forward.names() == ["mol/bace", "mol/chebi20", "mol/g2s"]
    assert list(forward.snapshot()["tasks"]) == forward.names()


def test_snapshot_excludes_the_callable_and_records_that_it_exists():
    reg = registry()
    plain = reg.snapshot()["tasks"]["mol/g2s"]
    assert "verify" not in plain
    assert plain["has_verify"] is False

    with_verify = Registry([
        TaskSpec(**{**SPECS[2], "verify": lambda pred, ex: True}),
    ])
    entry = with_verify.snapshot()["tasks"]["mol/g2s"]
    assert "verify" not in entry
    assert entry["has_verify"] is True
    # The callable's identity must not reach the hash: two equivalent registries
    # built in two processes have to agree.
    other = Registry([TaskSpec(**{**SPECS[2], "verify": lambda pred, ex: False})])
    assert with_verify.hash() == other.hash()


def test_snapshot_is_json_serialisable_and_carries_the_build_version():
    import json

    reg = registry(mol__bace={"build_version": "abc123"})
    text = json.dumps(reg.snapshot(), sort_keys=True)
    assert '"build_version": "abc123"' in text
    assert reg.snapshot()["tasks"]["mol/chebi20"]["eval_splits"] == ["val", "test"]


def test_a_changed_spec_changes_the_hash():
    """D5.4 detects a mixture change by this hash; it has to actually move."""
    assert registry().hash() != registry(mol__bace={"build_version": "v2"}).hash()
    assert registry().hash() != registry(mol__bace={"weight": 3.0}).hash()


def test_mixture_hash_is_stable_across_config_order():
    a = resolve(registry(), MIXTURE, TOKENS_PER_STEP)
    b = resolve(registry(), list(reversed(MIXTURE)), TOKENS_PER_STEP)
    assert a.hash() == b.hash()
    assert a.budget_examples == b.budget_examples

    changed = resolve(registry(), [
        {"name": "mol/bace", "weight": 2.0},
        {"name": "mol/chebi20", "weight": 1.0},
        {"name": "mol/g2s", "weight": 2.0},
    ], TOKENS_PER_STEP)
    assert changed.hash() != a.hash()


# ─────────────────────────────────────────────────────────────────────────────
# The `steps` override (smoke runs)
# ─────────────────────────────────────────────────────────────────────────────


def test_a_step_count_overrides_the_budget_rule():
    """A smoke run fixes the steps; the finite sources stop setting the budget."""
    mixture = resolve(registry(), MIXTURE, TOKENS_PER_STEP, steps=200)
    assert mixture.steps == 200
    # 200 steps x 10 examples/step, and the shares are unchanged.
    assert mixture.budget_examples == 2000
    assert mixture.per_task_examples == {
        "mol/bace": 500, "mol/chebi20": 500, "mol/g2s": 1000}
    assert mixture.binding_task == "steps=200"
    assert mixture.examples_per_step == pytest.approx(10.0)
    # The pass caps are still reported; they just do not bind any more.
    assert {e.name: e.available for e in mixture.entries}["mol/bace"] == 3000


def test_a_step_count_admits_a_generator_only_mixture():
    """The one refusal the override lifts: nothing finite has to be in the mixture."""
    with pytest.raises(RegistryError, match="budget: the mixture has no corpus"):
        resolve(registry(), [{"name": "mol/g2s", "weight": 1.0}], TOKENS_PER_STEP)

    mixture = resolve(registry(), [{"name": "mol/g2s", "weight": 1.0}],
                      TOKENS_PER_STEP, steps=50)
    assert mixture.steps == 50
    assert mixture.examples_per_step == pytest.approx(20.0)   # 1000 / 50 tokens
    assert mixture.budget_examples == 1000
    # The sub-threshold floor is still on by default, and still lowerable: over 50
    # steps a 1e-5 share buys nothing, which is not what that check is about.
    reg = registry()
    reg.register(TaskSpec(name="mol/tox21", domain="molecules", adapter="molecules",
                          kind="corpus", answer_kind="yesno", train_size=50000,
                          mean_tokens=100.0))
    tiny = MIXTURE + [{"name": "mol/tox21", "weight": 1e-5}]
    with pytest.raises(RegistryError, match=r"mol/tox21: share .* under one example"):
        resolve(reg, tiny, TOKENS_PER_STEP, steps=200)
    assert resolve(reg, tiny, TOKENS_PER_STEP, steps=200,
                   min_examples_per=0).steps == 200


def test_registry_lookup_helpers():
    reg = registry()
    assert "mol/bace" in reg
    assert len(reg) == 3
    assert [s.name for s in reg] == reg.names()
    assert reg.get("mol/bace").metric == "roc_auc"
    with pytest.raises(RegistryError, match="mol/nope: not registered"):
        reg.get("mol/nope")
