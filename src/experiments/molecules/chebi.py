"""
Tier C — ChEBI-20 molecule captioning (`PLAN.md` §1).

`PLAN.md` §1 listed ChEBI-20 as a tier of this benchmark suite and then deferred
it to the generalist, because the generalist was the only harness that could
train a generative task. That made the *specialist* number live in
`src/generalist/`, beside a mixture and a cross-source partition it does not
need. This module brings it back where it belongs: one task, one corpus, the
benchmark's own split, scored the way its published baselines are scored.

**The split is ChEBI-20's own, and that is a deliberate difference from the
generalist's build.** The generalist applies D3.3's cross-source partition, under
which a ChEBI *training* molecule that some MoleculeNet corpus claims as test is
withheld — 2,184 of them on the cap-128 build. That rule is right for a model
scored on BACE/BBBP/HIV as well as on captions; it is a handicap no published
ChEBI-20 baseline pays, and it is wrong for a specialist reported only on
ChEBI-20. Here the three files *are* the split.

**Coverage is part of the metric.** The benchmark's test split is 3,300
molecules; a build keeps fewer, because a molecule over ``heavy_atom_cap`` heavy
atoms has no graph we can encode. :func:`excluded_cids` names the ones it drops
so they can be charged as misses rather than quietly leaving the denominator —
see `TODO.md` §3. Every drop is a counted number, as `load_tier_b` treats its
own.
"""

from __future__ import annotations

import os

from .data import RAW_DIR, is_encodable

CHEBI_DIR = os.path.join(RAW_DIR, "chebi20")

#: The three tab-separated files under ``ChEBI-20_data/`` in blender-nlp/MolT5.
CHEBI_FILES = {"train": "train.txt", "val": "validation.txt", "test": "test.txt"}

#: The benchmark's own split sizes. Hard-coded because the denominator of a
#: published metric is a property of the benchmark and not of our build — the
#: whole point of the coverage accounting is that the two had drifted apart.
BENCHMARK_SIZES = {"train": 26407, "val": 3301, "test": 3300}

#: The task name on the `dataset.ALL_TASKS` axis.
CHEBI_TASK = "chebi20"

#: The question every ChEBI example carries. One fixed string: the task is
#: "describe this molecule", and the molecule is the only thing that varies.
CHEBI_QUESTION = "Describe this molecule."

#: Heavy-atom ceiling. 128 is the 99th percentile of the test split (q0.99 = 135,
#: max = 383) and admits 3,261 of 3,300; admitting the whole tail would put a
#: 536-atom molecule through an N^2 bias for the sake of 39 rows.
DEFAULT_HEAVY_ATOM_CAP = 128


def partition_key(mol) -> str:
    """The stereo-free canonical SMILES.

    Two stereoisomers have identical graphs up to the parity words, so keying on
    the isomeric string would let near-identical graphs straddle the train/test
    line. Kept here (rather than imported from the generalist adapter) so this
    module has no dependency on the generalist package; the two definitions are
    the same expression and `tests/experiments/test_chebi.py` pins that.
    """
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _screen(smiles, heavy_atom_cap, allow_disconnected):
    """``(mol, reason)`` — the mol if it passes every filter, else why it did not."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "parse"
    # `RemoveAllHs` before every check, as `load_tier_b` does: an explicit `[H]`
    # beside the parent's own hydrogen count double-counts the hydrogen.
    mol = Chem.RemoveAllHs(mol)
    if not allow_disconnected and len(Chem.GetMolFrags(mol)) > 1:
        return None, "disconnected"
    if mol.GetNumHeavyAtoms() > heavy_atom_cap:
        return None, "heavy_atom_cap"
    if not mol.GetNumHeavyAtoms():
        # ChEBI describes some entries that are hydrogen and nothing else —
        # dihydrogen, the hydron. `RemoveAllHs` leaves them with no atoms at all,
        # which passes every filter above. What it is not is a molecule.
        return None, "no_heavy_atoms"
    if not is_encodable(mol)[0]:
        return None, "unsupported_bond"
    return mol, None


def _rows(split, chebi_dir):
    """``(cid, smiles, description)`` per line of the benchmark file."""
    path = os.path.join(chebi_dir, CHEBI_FILES[split])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing. ChEBI-20 is the three tab-separated files "
            "(CID, SMILES, description) under ChEBI-20_data/ in the MolT5 "
            "repository, blender-nlp/MolT5.")
    out = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        if not header.lower().startswith("cid"):
            raise ValueError(
                f"{path}: expected a 'CID\\tSMILES\\tdescription' header, got "
                f"{header[:60]!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                out.append((parts[0] if parts else "", "", ""))
                continue
            out.append((parts[0], parts[1], "\t".join(parts[2:]).strip()))
    return out


def load_chebi(heavy_atom_cap: int = DEFAULT_HEAVY_ATOM_CAP,
               allow_disconnected: bool = True,
               chebi_dir: str = CHEBI_DIR,
               splits=("train", "val", "test")):
    """``(splits, stats)`` — the screened benchmark, split by its own files.

    ``splits`` is ``{split: [{"cid", "mol", "key", "text"}, ...]}``. Every drop is
    a counted number rather than a silent skip:

    ``parse``              RDKit cannot read the SMILES.
    ``unsupported_bond``   a bond with no faithful text encoding (`is_encodable`).
    ``heavy_atom_cap``     over ``heavy_atom_cap`` heavy atoms.
    ``disconnected``       more than one fragment, and they are not allowed.
    ``empty_description``  no caption; nothing to supervise.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out, stats = {}, {"kept": {}, "dropped": {}, "heavy_atoms": {}}
    for split in splits:
        dropped = {k: 0 for k in ("parse", "unsupported_bond", "heavy_atom_cap",
                                  "no_heavy_atoms", "disconnected",
                                  "empty_description")}
        kept, sizes = [], []
        for cid, smiles, text in _rows(split, chebi_dir):
            if not text:
                dropped["empty_description"] += 1
                continue
            mol, reason = _screen(smiles, heavy_atom_cap, allow_disconnected)
            if mol is None:
                dropped[reason] += 1
                continue
            kept.append({"cid": cid, "mol": mol, "key": partition_key(mol),
                         "text": text})
            sizes.append(mol.GetNumHeavyAtoms())
        out[split] = kept
        stats["kept"][split] = len(kept)
        stats["dropped"][split] = dropped
        stats["heavy_atoms"][split] = {
            "mean": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max": max(sizes) if sizes else 0,
        }
    stats["molecules"] = sum(stats["kept"].values())
    stats["distinct_keys"] = len({r["key"] for s in out.values() for r in s})
    return out, stats


def excluded_cids(split: str,
                  heavy_atom_cap: int = DEFAULT_HEAVY_ATOM_CAP,
                  allow_disconnected: bool = True,
                  chebi_dir: str = CHEBI_DIR) -> list:
    """``[{"cid", "reason"}, ...]`` for molecules the build drops from ``split``.

    The complement of what :func:`load_chebi` keeps, by construction: both walk
    the same rows through :func:`_screen`. A rescorer charges each of these as an
    empty prediction against its real caption, which is what keeps the reported
    denominator the benchmark's own.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = []
    for cid, smiles, text in _rows(split, chebi_dir):
        if not text:
            out.append({"cid": cid, "reason": "empty_description"})
            continue
        mol, reason = _screen(smiles, heavy_atom_cap, allow_disconnected)
        if mol is None:
            out.append({"cid": cid, "reason": reason})
    return out


def reference_captions(split: str, chebi_dir: str = CHEBI_DIR) -> dict:
    """``{cid: description}`` straight from the benchmark file.

    An excluded molecule is never materialised by the build, so its reference has
    to come from the same place the benchmark's own denominator does.
    """
    return {cid: text for cid, _s, text in _rows(split, chebi_dir) if text}


def build_chebi_examples(heavy_atom_cap: int = DEFAULT_HEAVY_ATOM_CAP,
                         allow_disconnected: bool = True,
                         chebi_dir: str = CHEBI_DIR,
                         answer_prefix: str = " "):
    """``(splits, stats)`` as ``{split: [(mol, question, answer), ...]}``.

    The shape `tier_b.build_tier_b_examples` returns, so `dataset.py` can treat
    Tier C like any other task. The answer carries a leading space for the same
    reason every other tier's does: it is the supervised token boundary.
    """
    loaded, stats = load_chebi(heavy_atom_cap, allow_disconnected, chebi_dir)
    out = {}
    for split, records in loaded.items():
        out[split] = [(r["mol"], CHEBI_QUESTION, answer_prefix + r["text"])
                      for r in records]
    stats["split_sizes"] = {k: len(v) for k, v in out.items()}
    return out, stats
