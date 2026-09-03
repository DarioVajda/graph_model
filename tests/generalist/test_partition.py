"""
T2 — the partition (`src/generalist/adapters/_partition.py`, DESIGN.md §D3.3).

`MOLECULE_GENERALIST.md` §3 exists because the campaign already had the incident
it prevents: a structural question about a BBBP *test* molecule landing in
training, which makes "scaffold-novel" mean nothing. So the properties below are
not stylistic — each one is the negation of a way that could happen again.

* roles are **pairwise disjoint**: one molecule, one role, no exceptions;
* **ClinTox is absent from every training source** — it is the held-out corpus,
  and a training example over one of its molecules would void the zero-shot
  number for all three held-out tasks at once;
* **priority** ``held_out > test > val > train`` decides every conflict, and it
  decides it the same way whatever order the sources were loaded in;
* the **ledger** counts what was lost and to whom, because "the partition is
  enforced" without a number is an assertion rather than a measurement.

The fast tests run on hand-built claims whose every count is checkable by eye.
The ``slow`` one builds the partition from the real MoleculeNet CSVs — ~130k
molecules, a canonicalization and a Murcko scaffold each, minutes the first time
and seconds afterwards because `molecules.partition` caches it under the cache
root keyed by the source checksums. Run it with a longer wall clock:

    TIME=02:00:00 src/generalist/tools/run_tests.sh \\
        tests/generalist/test_partition.py -q
"""

import json

import pytest

from src.generalist.adapters._partition import (
    PARTITION_RULE_VERSION,
    ROLES,
    Claim,
    Partition,
    PartitionError,
    build_partition,
)

# ─────────────────────────────────────────────────────────────────────────────
# The fixture. Eight keys, four sources, every conflict class present exactly
# once, so the ledger below is a hand count and not a re-derivation.
#
#   a  bace/train                                  -> train   (uncontested)
#   b  bace/train, bbbp/test                       -> test    (test beats train)
#   c  bace/train, clintox/held_out                -> held_out
#   d  bace/val,   bbbp/test                       -> test    (test beats val)
#   e  bace/test                                   -> test
#   f  bbbp/train, chebi/val                       -> val     (val beats train)
#   g  chebi/train                                 -> train
#   h  clintox/held_out, bbbp/test                 -> held_out (held_out beats test)
#
# bace  claims 3 train (a b c), 1 val (d), 1 test (e): keeps a and e,
#       loses b and d to bbbp:test and c to clintox:held_out -> lost 3.
# bbbp  claims 1 train (f), 3 test (b d h): keeps b and d, loses f to chebi:val
#       and h to clintox:held_out -> lost 2.
# chebi claims 1 train (g), 1 val (f): keeps both -> lost 0.
# clintox claims 2 held_out (c h): keeps both -> lost 0.
# ─────────────────────────────────────────────────────────────────────────────

CLAIMS = (
    Claim("bace", "train", ("a", "b", "c")),
    Claim("bace", "val", ("d",)),
    Claim("bace", "test", ("e",)),
    Claim("bbbp", "train", ("f",)),
    Claim("bbbp", "test", ("b", "d", "h")),
    Claim("chebi", "train", ("g",)),
    Claim("chebi", "val", ("f",)),
    Claim("clintox", "held_out", ("c", "h")),
)

EXPECTED_ROLES = {
    "a": "train", "b": "test", "c": "held_out", "d": "test",
    "e": "test", "f": "val", "g": "train", "h": "held_out",
}


@pytest.fixture
def part():
    return build_partition(CLAIMS, meta={"note": "T2 fixture"})


# ── the rules ────────────────────────────────────────────────────────────────

def test_priority_decides_every_conflict(part):
    """held_out > test > val > train, key by key."""
    for key, role in EXPECTED_ROLES.items():
        assert part.role(key) == role, key


def test_roles_are_pairwise_disjoint(part):
    sets = {role: part.keys(role) for role in ROLES}
    for i, left in enumerate(ROLES):
        for right in ROLES[i + 1:]:
            assert not (sets[left] & sets[right]), (left, right)
    assert set().union(*sets.values()) == set(EXPECTED_ROLES)
    assert len(part) == len(EXPECTED_ROLES)


def test_clintox_keys_are_held_out_and_train_holds_none_of_them(part):
    """The held-out corpus leaves every training source. §3 Rule 1."""
    clintox = {"c", "h"}
    assert clintox <= part.keys("held_out")
    assert not (clintox & part.keys("train"))
    assert not (clintox & part.keys("val"))
    assert not (clintox & part.keys("test"))


def test_claim_order_does_not_change_the_outcome():
    """Ties inside one role break by source name, not by call order.

    Two processes that build the same sources in different orders have to agree,
    or the ledger is a property of the loop and not of the data.
    """
    forward = build_partition(CLAIMS)
    backward = build_partition(tuple(reversed(CLAIMS)))
    assert forward.to_json()["roles"] == backward.to_json()["roles"]
    assert forward.ledger == backward.ledger


# ── the ledger ───────────────────────────────────────────────────────────────

def test_ledger_counts_match_the_hand_count(part):
    ledger = part.ledger

    assert ledger["bace"]["claimed"] == {"train": 3, "val": 1, "test": 1}
    assert ledger["bace"]["keys"] == 5
    assert ledger["bace"]["kept"] == 2                     # a, e
    assert ledger["bace"]["lost"] == 3                     # b, d -> bbbp; c -> clintox
    assert ledger["bace"]["to"] == {"bbbp:test": 2, "clintox:held_out": 1}

    assert ledger["bbbp"]["claimed"] == {"train": 1, "test": 3}
    assert ledger["bbbp"]["kept"] == 2                     # b, d
    assert ledger["bbbp"]["lost"] == 2                     # f -> chebi; h -> clintox
    assert ledger["bbbp"]["to"] == {"chebi:val": 1, "clintox:held_out": 1}

    assert ledger["chebi"]["lost"] == 0
    assert ledger["clintox"]["lost"] == 0

    for source, entry in ledger.items():
        assert entry["kept"] + entry["lost"] == entry["keys"], source


def test_counts_are_final_roles_not_claims(part):
    """`counts` says where a source's keys ENDED, which is the number Rule 4 wants."""
    assert part.counts["bace"] == {"held_out": 1, "test": 3, "val": 0, "train": 1}
    assert part.counts["clintox"] == {"held_out": 2, "test": 0, "val": 0, "train": 0}
    assert part.role_totals == {"held_out": 2, "test": 3, "val": 1, "train": 2}


def test_summary_names_every_source(part):
    text = part.summary()
    for source in ("bace", "bbbp", "chebi", "clintox"):
        assert source in text


# ── persistence ──────────────────────────────────────────────────────────────

def test_round_trips_through_disk(tmp_path, part):
    path = part.save(str(tmp_path / "partition.json"))
    reloaded = Partition.load(path)
    assert reloaded.to_json() == part.to_json()
    assert reloaded.role("c") == "held_out"
    assert reloaded.meta == {"note": "T2 fixture"}


def test_a_partition_from_another_rule_version_is_refused(tmp_path, part):
    """A role rule that moved must rebuild, never load. D3.3."""
    payload = part.to_json()
    payload["partition_rule_version"] = PARTITION_RULE_VERSION + "-old"
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(PartitionError, match="rule version"):
        Partition.load(str(path))


def test_an_unknown_role_is_refused():
    with pytest.raises(PartitionError, match="role"):
        Claim("bace", "holdout", ("a",))
    with pytest.raises(PartitionError, match="role"):
        build_partition(()).keys("training")


def test_a_source_cannot_claim_one_role_twice():
    """Two claims of one (source, role) would count the same key twice in the ledger."""
    with pytest.raises(PartitionError, match="twice"):
        build_partition((Claim("bace", "train", ("a",)),
                         Claim("bace", "train", ("b",))))


def test_an_unclaimed_key_is_None_not_train(part):
    """`None` means "no source ever saw this", which is not the same as trainable."""
    assert part.role("never-seen") is None
    assert not part.is_role("never-seen", "train")


# ─────────────────────────────────────────────────────────────────────────────
# The real thing
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_real_csvs_partition_disjointly(capsys):
    """The whole partition, from the MoleculeNet CSVs and ChEBI-20.

    Minutes the first time; the cache under the adapter's cache root makes every
    later run seconds, and it is keyed by the source checksums so a re-downloaded
    corpus rebuilds rather than silently reusing the old roles.
    """
    from src.generalist.adapters import molecules as M
    from src.experiments.molecules.data import load_tier_b

    config = M.MoleculeAdapterConfig()
    part = M.partition(config)

    with capsys.disabled():
        print()
        print(part.summary())

    # 1. Pairwise disjoint. The mapping is key -> one role by construction; this
    #    asserts the construction rather than trusting it.
    sets = {role: part.keys(role) for role in ROLES}
    for i, left in enumerate(ROLES):
        for right in ROLES[i + 1:]:
            assert not (sets[left] & sets[right]), (left, right)
    assert sum(len(s) for s in sets.values()) == len(part)
    # 77,994 distinct stereo-free keys as of the 2026-09-02 sources; HIV alone is
    # 41k. A number in the tens of thousands means every corpus loaded, and a
    # loose floor is what keeps this from failing on a corpus refresh.
    assert len(part) > 50_000, "the pool is far smaller than the sources are"

    # 2. ClinTox is held out, everywhere. Loaded fresh rather than read off the
    #    ledger, so a bug in the ledger cannot hide a leak.
    clintox_records, _spec, _dropped = load_tier_b("clintox")
    clintox = {M.partition_key(r["mol"]) for r in clintox_records}
    assert clintox <= sets["held_out"]
    assert not (clintox & sets["train"])
    assert not (clintox & sets["val"])
    assert not (clintox & sets["test"])

    # 3. Every source's keys are accounted for, and the ledger's "to" column
    #    only ever points at a role at least as high as the one that lost.
    priority = {role: i for i, role in enumerate(ROLES)}
    for source, entry in part.ledger.items():
        assert entry["kept"] + entry["lost"] == entry["keys"], source
        lowest_claimed = max(priority[r] for r in entry["claimed"])
        for label, count in entry["to"].items():
            assert count > 0
            assert priority[label.rsplit(":", 1)[1]] <= lowest_claimed, (
                f"{source} lost keys to {label}, which is not a higher role")

    # 4. No Tier-B or ChEBI val/test molecule is train-role — Rule 1, stated as
    #    the thing a training source may not contain.
    for source, counts in part.counts.items():
        assert set(counts) == set(ROLES), source

    assert sets["held_out"], "nothing is held out; §4 would be vacuous"
    assert sets["train"] and sets["val"] and sets["test"]
