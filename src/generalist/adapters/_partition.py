"""
D3.3 — one molecule, one role.

`MOLECULE_GENERALIST.md` §3: the Tier-A generators and graph-to-SMILES draw
their molecules from the Tier-B corpora, so without a single rule across sources
a structural question about a BBBP *test* molecule lands in training and the
scaffold split stops meaning "structurally novel". The campaign already had
exactly that incident once (`molecules/PLAN.md` §3.2.10), which is why this is a
separate object built before any example exists rather than a filter applied
while generating.

The rule, in full:

* **Key** — the stereo-free canonical SMILES. Stereo-free on purpose: two
  stereoisomers have identical graphs up to the parity words, so keying on the
  isomeric string would let near-identical graphs straddle the train/test line.
  Both isomers share one role and each keeps its own labels.
* **Claims** — every source declares, for each key it holds, the role it would
  like it to have: a Tier-B corpus claims by its own scaffold split, ClinTox
  claims ``held_out`` for everything, the regression sets claim ``train`` for
  everything (unlabeled pool, `MOLECULE_GENERALIST.md` §1).
* **Priority** — ``held_out > test > val > train``. The highest claim wins, and
  the key gets that one role everywhere.
* **Ledger** — how many keys each source lost to a higher claim, and to which
  claimant. That number is the one Rule 4 says goes into the run record; without
  it "the partition is enforced" is an assertion rather than a measurement.

Ties inside one role are broken by source name, so the ledger is a function of
the inputs and not of the order the caller happened to build them in.

This module is free of RDKit and torch: the key function lives in the molecules
adapter, which is the only thing that knows what a molecule is. What is here is
the role algebra, and it is domain-agnostic on purpose — the trunk's other
adapters get the same object for free.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

#: Highest priority first. `MOLECULE_GENERALIST.md` §3 Rule 1.
ROLES = ("held_out", "test", "val", "train")

#: The training roles a mixture may draw from. Exactly one, but named rather
#: than written as a literal at each of the three enforcement points.
TRAIN_ROLE = "train"

#: Bumped when the rules above change. Part of the build version (D3.2), so a
#: rule change cannot silently reuse a cached partition.
PARTITION_RULE_VERSION = "1"

_PRIORITY = {role: i for i, role in enumerate(ROLES)}


class PartitionError(ValueError):
    """A partition that cannot be built, or a role that does not exist."""


@dataclass(frozen=True)
class Claim:
    """One source asking for one role over a set of keys.

    ``source`` is a stable name that appears in the ledger and the run record —
    ``"tier_b/bace"``, ``"chebi20"``, ``"clintox"``. ``keys`` may repeat across
    claims; that is the whole point.
    """

    source: str
    role: str
    keys: tuple

    def __post_init__(self):
        if self.role not in _PRIORITY:
            raise PartitionError(
                f"{self.source}: role {self.role!r} is not one of {ROLES}")


class Partition:
    """``key -> role``, plus the per-source counts and the overlap ledger.

    Persisted as ``partition.json`` next to the built data and re-checked at
    load time (D3.3). It is the object both enforcement points consult: ``build``
    refuses to emit a training example whose key is not ``train``-role, and
    ``load`` re-checks a sample.
    """

    def __init__(self, roles: dict, counts: dict, ledger: dict, meta: dict = None):
        self._roles = dict(roles)
        self._counts = {s: dict(c) for s, c in counts.items()}
        self._ledger = {s: dict(l) for s, l in ledger.items()}
        self.meta = dict(meta or {})
        self._by_role = None

    # ── lookups ──────────────────────────────────────────────────────────────

    def role(self, key: str):
        """This key's role, or ``None`` if no source ever claimed it.

        ``None`` is not the same as ``"train"``: a key nobody claimed is a key
        the partition has never seen, and emitting a training example for it
        would mean the pool and the partition were built from different sources.
        """
        return self._roles.get(key)

    def keys(self, role: str) -> set:
        if role not in _PRIORITY:
            raise PartitionError(f"role {role!r} is not one of {ROLES}")
        if self._by_role is None:
            self._by_role = {r: set() for r in ROLES}
            for key, assigned in self._roles.items():
                self._by_role[assigned].add(key)
        return self._by_role[role]

    def is_role(self, key: str, role: str) -> bool:
        return self._roles.get(key) == role

    @property
    def counts(self) -> dict:
        """``{source: {role: n}}`` — the FINAL role of each of a source's keys.

        Not what the source claimed: what it got. The difference between the two
        is the ledger, and reading only this one would make an overlap invisible.
        """
        return self._counts

    @property
    def ledger(self) -> dict:
        """``{source: {claimed, kept, lost, to}}`` — Rule 4's overlap record."""
        return self._ledger

    @property
    def role_totals(self) -> dict:
        return {role: len(self.keys(role)) for role in ROLES}

    def __len__(self) -> int:
        return len(self._roles)

    def __contains__(self, key) -> bool:
        return key in self._roles

    def __repr__(self) -> str:
        totals = ", ".join(f"{r}={n}" for r, n in self.role_totals.items())
        return f"<Partition {len(self._roles)} keys: {totals}>"

    def summary(self) -> str:
        """The table the run record carries (D3.3, Rule 4)."""
        rows = [f"  partition: {len(self._roles)} keys  "
                + "  ".join(f"{r} {n}" for r, n in self.role_totals.items()),
                "  source                 held_out    test     val   train    lost",
                "  " + "-" * 62]
        for source in sorted(self._counts):
            c = self._counts[source]
            lost = self._ledger.get(source, {}).get("lost", 0)
            rows.append(
                f"  {source:<20} {c.get('held_out', 0):>9} {c.get('test', 0):>7} "
                f"{c.get('val', 0):>7} {c.get('train', 0):>7} {lost:>7}")
        return "\n".join(rows)

    # ── persistence ──────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        return {
            "partition_rule_version": PARTITION_RULE_VERSION,
            "roles": self._roles,
            "counts": self._counts,
            "ledger": self._ledger,
            "role_totals": self.role_totals,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Partition":
        got = obj.get("partition_rule_version")
        if got != PARTITION_RULE_VERSION:
            raise PartitionError(
                f"partition.json was written under rule version {got!r} and this "
                f"code is {PARTITION_RULE_VERSION!r}; rebuild it rather than "
                "mixing two role rules in one run.")
        return cls(obj["roles"], obj["counts"], obj["ledger"], obj.get("meta"))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f)
        os.replace(tmp, path)          # a half-written partition is never loadable
        return path

    @classmethod
    def load(cls, path: str) -> "Partition":
        with open(path) as f:
            return cls.from_json(json.load(f))


def build_partition(claims, meta: dict = None) -> Partition:
    """Resolve overlapping :class:`Claim`s into one role per key.

    Claims are sorted by ``(role priority, source name)`` and walked highest
    first; the first claim to reach a key owns it. Sorting by source name rather
    than by call order is what makes the ledger reproducible: two processes that
    build the same sources in different orders get byte-identical output.
    """
    claims = list(claims)
    seen = set()
    for claim in claims:
        if not isinstance(claim, Claim):
            raise PartitionError(f"expected a Claim, got {claim!r}")
        if (claim.source, claim.role) in seen:
            raise PartitionError(
                f"{claim.source}: claims {claim.role!r} twice; merge the key sets "
                "so the ledger counts each key once")
        seen.add((claim.source, claim.role))

    ordered = sorted(claims, key=lambda c: (_PRIORITY[c.role], c.source))

    winner = {}                                   # key -> (source, role)
    for claim in ordered:
        for key in claim.keys:
            if key not in winner:
                winner[key] = (claim.source, claim.role)

    roles = {key: role for key, (_source, role) in winner.items()}

    counts, ledger = {}, {}
    for claim in ordered:
        source_counts = counts.setdefault(claim.source, {r: 0 for r in ROLES})
        entry = ledger.setdefault(
            claim.source,
            {"keys": 0, "claimed": {}, "kept": 0, "lost": 0, "to": {}})
        keys = set(claim.keys)
        entry["claimed"][claim.role] = len(keys)
        for key in keys:
            source, role = winner[key]
            source_counts[role] += 1
            if source == claim.source and role == claim.role:
                entry["kept"] += 1
            else:
                entry["lost"] += 1
                label = f"{source}:{role}"
                entry["to"][label] = entry["to"].get(label, 0) + 1

    for source, entry in ledger.items():
        # Summed rather than counted over a union: every source here splits its
        # own keys disjointly across the roles it claims (a scaffold split, a
        # corpus split), so the sum is the key count. A source that claimed one
        # key under two roles would overcount, and would be describing a split
        # that is already broken on its own terms.
        entry["keys"] = sum(entry["claimed"].values())

    return Partition(roles, counts, ledger, meta)
