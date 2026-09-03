"""
D3.1 — the molecules package as schema Examples.

Every task the molecule generalist trains on (`MOLECULE_GENERALIST.md` §1) is
registered here under ``mol/``: nine Tier-A generators, five Tier-B corpora,
ChEBI-20 captioning, graph-to-SMILES, and the three held-out tasks. The adapter
owns *which molecules* and *which splits*; the molecules package owns *what a
node says*, and this file never restates it. `mol_to_graph`, `attach_question`,
`build_graph_example`, `build_flat_example`, `build_tier_b_examples`,
`_build_split_graphs`, `TASK_GENERATORS`, `flat_serialize` and `roundtrip_check`
are all called, not copied: a second rendering of "carbon aromatic ring deg3 H1"
would drift from the one 90 sweeps were measured on, and every graph/flat
comparison downstream would carry that drift as an uncontrolled difference.

Three things are genuinely new here, and each is new because nothing in the
molecules package could have needed it:

* **The partition** (`_partition.py`, `MOLECULE_GENERALIST.md` §3). The molecules
  campaign runs one corpus at a time, so a per-corpus scaffold split was enough.
  A mixture drawing structural questions from every corpus at once is not: one
  molecule, one role, across all sources.
* **graph-to-SMILES** (§5). The target is the *stereo-free* canonical SMILES for
  both arms, because the graph carries parity words without the neighbour
  ordering that would give them meaning. The flat twin's matched task is
  canonicalization — randomized SMILES in, canonical SMILES out.
* **ChEBI-20** (§6). A captioning corpus with its own three splits, a heavy-atom
  cap and an explicit answer for multi-fragment molecules.

**Two spellings of the answer boundary, one byte sequence.** The molecules
package writes ``"\\nA:" + answer`` and every `tasks.py` answer carries its own
leading space, so a Tier-A prompt ends ``"\\nA: Yes"``. The schema's multi-token
kinds (``text``, ``smiles``) hold an answer with *no* leading space, so the
prompt has to end ``"\\nA: " + answer``. Both are satisfied by passing
``" " + answer`` to the molecules builders and storing the bare answer on the
Example: the bytes are identical either way, and `schema.render` locates the
span against the stored answer. This is the only place the two conventions meet
and it is why they are described here rather than in either file.

Nothing at module scope pulls torch or RDKit; the imports that do are inside the
functions that need them, so ``validate`` mode still resolves a config on the
login node.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass

from ..registry import MOLECULE_PREFIX, Registry, TaskSpec
from ..schema import SCHEMA_VERSION, Example, SchemaError, render, validate
from ._partition import (
    PARTITION_RULE_VERSION,
    ROLES,
    Claim,
    Partition,
    PartitionError,
    build_partition,
)

DOMAIN = "molecules"
ADAPTER_NAME = "molecules"

#: Bumped when this file changes what it writes to disk for a fixed config.
#: Part of the build version, so a code change cannot silently reuse a cache.
ADAPTER_VERSION = "1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_MOLECULES_DIR = os.path.join(_REPO_ROOT, "src", "experiments", "molecules")

#: Where the MoleculeNet CSVs live (the molecules package's own ``raw_data``),
#: and where the ChEBI-20 split files are downloaded to. Both are gitignored —
#: re-fetchable corpora, not repo content.
#:
#: There is deliberately **no** ``raw_dir`` knob on the config. `load_tier_b`
#: takes one, but `tier_b.build_tier_b_examples` — which produces the actual
#: Tier-B examples — does not, so a config that redirected one and not the other
#: would build a partition from one corpus and examples from another. One
#: directory, and a test that wants small inputs replaces `load_tier_b` itself.
RAW_DIR = os.path.join(_MOLECULES_DIR, "raw_data")
CHEBI_DIR = os.path.join(RAW_DIR, "chebi20")

#: Default cache root. ``results/`` is never committed (DESIGN.md §1), which is
#: the right home for tens of gigabytes of built graphs.
DEFAULT_CACHE_ROOT = os.path.join(_REPO_ROOT, "src", "generalist", "results", "data")

# ── the task table ───────────────────────────────────────────────────────────

#: The nine Tier-A families that train. `MOLECULE_GENERALIST.md` §1.
TIER_A_TRAIN_TASKS = (
    "ring_membership", "aromatic_ring", "ring_size", "ring_count",
    "fg_presence", "fg_count", "fg_atom_membership",
    "stereo_potential", "stereo_assigned",
)

#: Held out permanently (§4). All three are also declared in `molecules/data.py`
#: (``HELD_OUT_TIER_A_TASKS``, ``HELD_OUT_DATASETS``), and every spec here
#: additionally carries ``held_out=True`` — `registry.is_held_out` refuses a task
#: either source names, so the declaration survives one of them being edited.
HELD_OUT_TIER_A_TASKS = ("bond_path", "longest_chain")
HELD_OUT_CORPORA = ("clintox",)

#: The five Tier-B corpora that train. BACE / BBBP / HIV are the headline;
#: Tox21 and SIDER are gradient, never an anchor-table number (§1).
TIER_B_CORPORA = ("bace", "bbbp", "hiv", "tox21", "sider")

#: Regression sets. Excluded as *tasks* — the margin readout cannot score a
#: number — but their molecules are unlabeled train-role pool (§1).
REGRESSION_CORPORA = ("esol", "freesolv", "lipo")

CHEBI_TASK = "chebi20"
G2S_TASK = "g2s"

#: `MOLECULE_GENERALIST.md` §5, verbatim. No atom labels.
G2S_QUESTION = "Question: write the canonical SMILES for this molecule."
CHEBI_QUESTION = "Describe this molecule."


class AdapterBuildError(ValueError):
    """A build that cannot proceed. The message names the task and the cause."""


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoleculeAdapterConfig:
    """Everything that changes what this adapter writes to disk.

    The generalist ``RunConfig`` embeds one of these; it is a dataclass of its
    own so the build version (D3.2) can be a hash of exactly this object plus the
    source checksums, with no run-level fields (output dir, seed, Slurm) leaking
    into a data cache key.
    """

    # ── graph encoding: passed straight through to the molecules RunConfig ────
    encoding: str = "rich_levi"
    stereo_tags: bool = True
    question_node: str = "on"
    ordering: str = "rcm"
    magnetic_q: float = 0.25
    magnetic_m: int = 0
    #: A model knob, not a data one: SPD is stored unclamped and the bias module
    #: clamps at read time. Carried here so one config feeds both, and left OUT
    #: of the build version so moving the clamp does not rebuild 40k graphs.
    max_spd: int = 32

    model_name: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 512

    # ── molecule pools ───────────────────────────────────────────────────────
    #: Corpora the Tier-A generators and graph-to-SMILES draw from. The default
    #: is the train-role union of everything that is not held out (§3), which is
    #: wider than the molecules campaign's `DEFAULT_POOL`: the generalist's pool
    #: is the whole partition, not one experiment's five corpora.
    pool: tuple = TIER_B_CORPORA + REGRESSION_CORPORA
    #: Corpora that contribute Tier-B yes/no tasks.
    tier_b_corpora: tuple = TIER_B_CORPORA
    #: Corpora that enter the partition as unlabeled train-role molecules.
    regression_corpora: tuple = REGRESSION_CORPORA

    # ── generator sizing ─────────────────────────────────────────────────────
    tier_a_cap_per_pass: int = 4000
    tier_a_val_size: int = 500
    tier_a_test_size: int = 1000
    g2s_cap_per_pass: int = 4000
    g2s_val_size: int = 500
    g2s_test_size: int = 1000
    #: Examples per held-out Tier-A task. Drawn from held_out-role molecules, so
    #: the zero-shot number is over molecules training never saw in any form.
    held_out_size: int = 1000

    # ── ChEBI-20 ─────────────────────────────────────────────────────────────
    chebi_dir: str = CHEBI_DIR
    #: §6: larger molecules hit the node budget. 64 heavy atoms is a Levi graph
    #: of roughly 135 nodes, which is inside the range the campaign has measured.
    #: The number of molecules this drops is recorded, never assumed small.
    chebi_heavy_atom_cap: int = 64
    #: §6: ChEBI-20 carries salts and multi-fragment molecules. SPD between two
    #: components is the ``max_spd`` clamp, which is the same value as "very far
    #: apart within one component" — so a disconnected graph tells the bias
    #: channel a fragment boundary and a long path apart. Dropped and counted by
    #: default; flip this to keep them once that conflation is measured.
    chebi_allow_disconnected: bool = False

    # ── seeds and cache ──────────────────────────────────────────────────────
    data_seed: int = 0
    cache_root: str = DEFAULT_CACHE_ROOT

    def validate(self) -> "MoleculeAdapterConfig":
        from ...experiments.molecules.data import ENCODINGS, QUESTION_NODE_MODES, TIER_B

        if self.encoding not in ENCODINGS:
            raise AdapterBuildError(
                f"encoding must be one of {ENCODINGS}, got {self.encoding!r}")
        if self.question_node not in QUESTION_NODE_MODES:
            raise AdapterBuildError(
                f"question_node must be one of {QUESTION_NODE_MODES}, got "
                f"{self.question_node!r}")
        for name in tuple(self.pool) + tuple(self.tier_b_corpora) + tuple(
                self.regression_corpora):
            if name not in TIER_B:
                raise AdapterBuildError(
                    f"unknown molecule source {name!r} (have {sorted(TIER_B)})")
        if self.chebi_heavy_atom_cap < 1:
            raise AdapterBuildError("chebi_heavy_atom_cap must be >= 1")
        return self

    # ── versioning (D3.2) ────────────────────────────────────────────────────

    def source_digests(self) -> dict:
        """sha256 of every raw file this config reads. Part of the build version.

        A corpus re-downloaded with different content is a different dataset, and
        it is the one input change that leaves no trace in any config field.
        """
        from ...experiments.molecules.data import TIER_B

        from ...experiments.molecules.data import RAW_DIR as MOL_RAW_DIR

        names = sorted(set(tuple(self.pool) + tuple(self.tier_b_corpora)
                           + tuple(self.regression_corpora) + HELD_OUT_CORPORA))
        out = {}
        for name in names:
            path = os.path.join(MOL_RAW_DIR, TIER_B[name].filename)
            out[f"tier_b/{name}"] = _file_digest(path)
        for split in _CHEBI_FILES:
            out[f"chebi20/{split}"] = _file_digest(
                os.path.join(self.chebi_dir, _CHEBI_FILES[split]))
        return out

    def partition_version(self) -> str:
        """Hash of everything that moves a key's ROLE.

        Deliberately narrower than the build version: the partition costs a
        canonicalization and a Murcko scaffold for ~130k molecules, and an
        encoding change does not move a single role. Keyed separately so the
        expensive half is not rebuilt for the cheap half's sake.
        """
        return _hash({
            "partition_rule_version": PARTITION_RULE_VERSION,
            "adapter_version": ADAPTER_VERSION,
            "sources": self.source_digests(),
            "pool": sorted(self.pool),
            "tier_b_corpora": sorted(self.tier_b_corpora),
            "regression_corpora": sorted(self.regression_corpora),
            "chebi_heavy_atom_cap": self.chebi_heavy_atom_cap,
            "chebi_allow_disconnected": self.chebi_allow_disconnected,
        })

    def build_version(self) -> str:
        """Hash of every input that changes a built example's bytes (D3.2)."""
        payload = asdict(self)
        for drop in ("cache_root", "chebi_dir", "max_spd"):
            payload.pop(drop, None)
        payload["pool"] = sorted(self.pool)
        payload["tier_b_corpora"] = sorted(self.tier_b_corpora)
        payload["regression_corpora"] = sorted(self.regression_corpora)
        return _hash({
            "adapter_version": ADAPTER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "partition_version": self.partition_version(),
            "config": payload,
        })

    # ── cache paths ──────────────────────────────────────────────────────────

    def partition_path(self) -> str:
        return os.path.join(self.cache_root, "partitions",
                            f"{self.partition_version()}.json")

    def build_dir(self) -> str:
        return os.path.join(self.cache_root, self.build_version())

    def manifest_path(self) -> str:
        return os.path.join(self.build_dir(), "manifest.json")

    def source_path(self, task: str, split: str, arm: str, pass_id: int) -> str:
        stem = f"{split}.{arm}.p{int(pass_id)}"
        return os.path.join(self.build_dir(), _task_dir(task), stem)


def _task_dir(task: str) -> str:
    return task.replace("/", "_")


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()[:16]


def _file_digest(path: str) -> str:
    if not os.path.exists(path):
        raise AdapterBuildError(
            f"{path} is missing. MoleculeNet CSVs come from `molecules/PLAN.md` M0; "
            "ChEBI-20 is the three tab-separated files of blender-nlp/MolT5's "
            "ChEBI-20_data/ (CID, SMILES, description).")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# The partition (D3.3, MOLECULE_GENERALIST.md §3)
# ─────────────────────────────────────────────────────────────────────────────

def partition_key(mol) -> str:
    """The stereo-free canonical SMILES. §3: one key, one role, both isomers.

    Two stereoisomers have identical graphs up to the parity words, so keying on
    the isomeric string would let near-identical graphs straddle the train/test
    line. Each isomer keeps its own labels; they share a role.
    """
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def partition(config: MoleculeAdapterConfig, rebuild: bool = False) -> Partition:
    """Every key this adapter will ever emit, with its one role.

    Cached at ``<cache_root>/partitions/<partition_version>.json``: the scan is
    a canonicalization and a Murcko scaffold over every molecule in every source
    (~130k), which is minutes, and it is a pure function of the raw files.
    """
    config.validate()
    path = config.partition_path()
    if not rebuild and os.path.exists(path):
        return Partition.load(path)

    claims, meta = _partition_claims(config)
    part = build_partition(claims, meta)
    part.save(path)
    return part


def _partition_claims(config: MoleculeAdapterConfig):
    """Build the claim list of §3 rules 1–4, and the meta a run record carries."""
    from ...experiments.molecules.data import load_tier_b, scaffold_split

    claims, meta = [], {"sources": {}, "endpoints": {}}

    # Tier B and the regression pool: each corpus claims by its own scaffold
    # split, which is what makes "structurally novel" mean anything downstream.
    tier_b = sorted(set(tuple(config.tier_b_corpora) + tuple(config.pool)))
    for name in tier_b:
        if name in HELD_OUT_CORPORA:
            continue
        records, spec, dropped = load_tier_b(name)
        keys = [partition_key(r["mol"]) for r in records]
        if name in config.regression_corpora:
            # §1: no anchor and no yes/no readout, so they are not tasks — but
            # their molecules are perfectly good unlabeled pool, at train role
            # unless a higher claim takes them.
            claims.append(Claim(f"tier_b/{name}", "train", tuple(keys)))
        else:
            tr, va, te = scaffold_split([r["smiles"] for r in records])
            for role, idx in (("train", tr), ("val", va), ("test", te)):
                claims.append(Claim(f"tier_b/{name}", role,
                                    tuple(keys[i] for i in idx)))
            meta["endpoints"][name] = _endpoint_label_counts(records, spec)
        meta["sources"][f"tier_b/{name}"] = {
            "molecules": len(records), "distinct_keys": len(set(keys)),
            "dropped": dropped}

    # ClinTox: held out entirely (§4). Every one of its molecules leaves every
    # training source, whichever split of whichever corpus it also sits in.
    for name in HELD_OUT_CORPORA:
        records, _spec, dropped = load_tier_b(name)
        keys = [partition_key(r["mol"]) for r in records]
        claims.append(Claim(name, "held_out", tuple(keys)))
        meta["sources"][name] = {"molecules": len(records),
                                 "distinct_keys": len(set(keys)),
                                 "dropped": dropped}

    # ChEBI-20 keeps its own three splits and folds into the same partition (§6).
    chebi, chebi_stats = load_chebi(config)
    for role in ("train", "val", "test"):
        claims.append(Claim("chebi20", role,
                            tuple(r["key"] for r in chebi[role])))
    meta["sources"]["chebi20"] = chebi_stats

    return claims, meta


def _endpoint_label_counts(records, spec) -> dict:
    """Per-endpoint labelled / absent counts over one corpus.

    §1: "Tox21's ~16k absent labels are skipped at the (molecule, endpoint)
    level; the per-endpoint example counts go in the run record, because skipping
    changes each endpoint's effective weight silently otherwise." The skip rule
    is `tier_b._label` — imported rather than restated, so this count and the
    examples that are actually emitted can never disagree about what NaN means.
    """
    from ...experiments.molecules.tier_b import _label

    if not records:
        return {}
    columns = spec.task_cols or tuple(records[0]["targets"])
    out = {}
    for endpoint in columns:
        labelled = sum(1 for r in records
                       if _label(r["targets"].get(endpoint)) is not None)
        out[endpoint] = {"labelled": labelled, "absent": len(records) - labelled}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ChEBI-20 (MOLECULE_GENERALIST.md §6)
# ─────────────────────────────────────────────────────────────────────────────

_CHEBI_FILES = {"train": "train.txt", "val": "validation.txt", "test": "test.txt"}


def load_chebi(config: MoleculeAdapterConfig):
    """The three ChEBI-20 splits, screened. Returns ``(splits, stats)``.

    ``splits`` is ``{"train"|"val"|"test": [{"cid", "mol", "key", "text"}, ...]}``
    and every drop is a counted number rather than a silent skip, exactly as
    `load_tier_b` treats its parse and bond-type failures:

    ``parse``              RDKit cannot read the SMILES.
    ``unsupported_bond``   a bond with no faithful text encoding (`is_encodable`).
    ``heavy_atom_cap``     over ``chebi_heavy_atom_cap`` heavy atoms (§6).
    ``disconnected``       more than one fragment, and they are not allowed (§6).
    ``empty_description``  no caption; nothing to supervise.

    ``RemoveAllHs`` before every check, as `load_tier_b` does: an explicit ``[H]``
    beside the parent's own hydrogen count double-counts the hydrogen and was the
    single ``rich_levi`` round-trip failure at M0.
    """
    from rdkit import Chem, RDLogger

    from ...experiments.molecules.data import is_encodable

    RDLogger.DisableLog("rdApp.*")

    splits, stats = {}, {"kept": {}, "dropped": {}, "heavy_atoms": {}}
    for split, filename in _CHEBI_FILES.items():
        path = os.path.join(config.chebi_dir, filename)
        if not os.path.exists(path):
            raise AdapterBuildError(
                f"{path} is missing. ChEBI-20 is the three tab-separated files "
                "(CID, SMILES, description) under ChEBI-20_data/ in the MolT5 "
                "repository, blender-nlp/MolT5.")
        dropped = {"parse": 0, "unsupported_bond": 0, "heavy_atom_cap": 0,
                   "no_heavy_atoms": 0, "disconnected": 0, "empty_description": 0}
        kept, sizes = [], []
        with open(path, encoding="utf-8") as f:
            header = f.readline()
            if not header.lower().startswith("cid"):
                raise AdapterBuildError(
                    f"{path}: expected a 'CID\\tSMILES\\tdescription' header, got "
                    f"{header[:60]!r}")
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    dropped["empty_description"] += 1
                    continue
                cid, smiles, description = parts[0], parts[1], "\t".join(parts[2:])
                description = description.strip()
                if not description:
                    dropped["empty_description"] += 1
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    dropped["parse"] += 1
                    continue
                mol = Chem.RemoveAllHs(mol)
                if not config.chebi_allow_disconnected and \
                        len(Chem.GetMolFrags(mol)) > 1:
                    dropped["disconnected"] += 1
                    continue
                if mol.GetNumHeavyAtoms() > config.chebi_heavy_atom_cap:
                    dropped["heavy_atom_cap"] += 1
                    continue
                if not mol.GetNumHeavyAtoms():
                    # ChEBI describes some entries that are hydrogen and nothing
                    # else — dihydrogen, the hydron. `RemoveAllHs` leaves them
                    # with no atoms at all, which passes every filter above:
                    # `GetMolFrags` counts zero fragments rather than two, zero
                    # is under any cap, and an empty graph is trivially
                    # encodable. What it is not is a molecule. Its
                    # `partition_key` is the empty string, and `schema.validate`
                    # refuses that — correctly, and about 20k examples into the
                    # build. The floor belongs at the same place as the cap.
                    dropped["no_heavy_atoms"] += 1
                    continue
                if not is_encodable(mol)[0]:
                    dropped["unsupported_bond"] += 1
                    continue
                kept.append({"cid": cid, "mol": mol, "key": partition_key(mol),
                             "text": description})
                sizes.append(mol.GetNumHeavyAtoms())
        splits[split] = kept
        stats["kept"][split] = len(kept)
        stats["dropped"][split] = dropped
        stats["heavy_atoms"][split] = {
            "mean": (sum(sizes) / len(sizes)) if sizes else 0.0,
            "max": max(sizes) if sizes else 0,
        }
    stats["molecules"] = sum(stats["kept"].values())
    stats["distinct_keys"] = len({r["key"] for s in splits.values() for r in s})
    return splits, stats


# ─────────────────────────────────────────────────────────────────────────────
# graph-to-SMILES (MOLECULE_GENERALIST.md §5)
# ─────────────────────────────────────────────────────────────────────────────

def g2s_target(mol):
    """The stereo-free canonical SMILES, or ``None`` if it is not a fixed point.

    §5: the target is ``MolToSmiles(mol, canonical=True, isomericSmiles=False)``
    for **both** arms, because a parity word is only meaningful relative to a
    neighbour ordering and the graph has none. `roundtrip_check` compares
    stereo-flattened strings at the ``exact`` level for exactly this reason.

    ``None`` when re-parsing the target does not reproduce it. The schema
    requires a ``smiles`` answer to equal its own canonicalization — otherwise
    exact match would score against one arbitrary spelling out of many — so a
    molecule that is not a fixed point is excluded and counted rather than
    emitted as an unreachable target.
    """
    from rdkit import Chem

    target = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    reparsed = Chem.MolFromSmiles(target)
    if reparsed is None:
        return None
    if Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=False) != target:
        return None
    return target


#: Characters that can only be stereo in a SMILES string: tetrahedral parity and
#: the two directional bond marks. Under the §5 target, emitting any of them is
#: an error, and how often the model does is a recorded diagnostic.
STEREO_MARKS = ("@", "/", "\\")


def smiles_scores(predictions, targets) -> dict:
    """The three §5 metrics plus the stereo diagnostic, in order of what matters.

    ``validity``          RDKit parses the output.
    ``roundtrip_match``   the output canonicalizes to the target. Canonicalized
                          **as given**, stereo included, so a prediction that
                          carries ``@`` fails: under this target stereo is an
                          error, not a harmless extra (§5). A stereo-free
                          prediction is unaffected — with no stereo to write, the
                          isomeric and stereo-free canonical forms are one string.
    ``exact_match``       the strict proxy: canonical atom ordering is an RDKit
                          ranking the model may or may not learn to reproduce.
    ``stereo_marks_emitted``  the diagnostic §5 asks to record.

    An empty input returns zeros rather than raising: a validator that dies on an
    empty generation set would lose a run that already cost GPU-hours.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    predictions, targets = list(predictions), list(targets)
    if len(predictions) != len(targets):
        raise ValueError(
            f"smiles_scores: {len(predictions)} predictions and {len(targets)} "
            "targets; they are paired, so the lengths must match")
    n = len(predictions)
    if not n:
        return {"validity": 0.0, "roundtrip_match": 0.0, "exact_match": 0.0,
                "stereo_marks_emitted": 0.0, "n": 0}

    valid = roundtrip = exact = stereo = 0
    for prediction, target in zip(predictions, targets):
        prediction = (prediction or "").strip()
        if any(mark in prediction for mark in STEREO_MARKS):
            stereo += 1
        if prediction == target:
            exact += 1
        mol = Chem.MolFromSmiles(prediction) if prediction else None
        if mol is None:
            continue
        valid += 1
        if Chem.MolToSmiles(mol, canonical=True) == target:
            roundtrip += 1
    return {"validity": valid / n, "roundtrip_match": roundtrip / n,
            "exact_match": exact / n, "stereo_marks_emitted": stereo / n, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Registry (D2) — the specs this adapter owns
# ─────────────────────────────────────────────────────────────────────────────

def task_specs(config: MoleculeAdapterConfig, arm: str = "graph") -> dict:
    """``{name: TaskSpec}`` for everything under ``mol/``.

    ``train_size``, ``mean_tokens`` and ``build_version`` are properties of the
    built data, not knobs, so they come from the build manifest when one exists
    and stay ``None`` when it does not — ``registry.resolve`` refuses a mixture
    whose tasks have no ``mean_tokens``, which is the right failure for "you
    asked to train before you ran ``data_prep``".

    ``arm`` selects which arm's mean length fills the spec. The two arms differ
    by several times on Tier A (a Levi graph against one SMILES string), so
    ``tokens_per_step`` resolves to a different example count for each and a
    single shared number would be wrong for both.
    """
    from ...experiments.molecules.tier_b import QUESTION_TEMPLATES, question_for

    manifest = read_manifest(config)
    version = config.build_version()

    def measured(name):
        entry = (manifest.get("tasks", {}) or {}).get(name, {})
        by_arm = entry.get("arms", {}).get(arm, {})
        return by_arm.get("train_size"), by_arm.get("mean_tokens")

    specs = {}

    def add(name, **kwargs):
        train_size, mean_tokens = measured(name)
        specs[name] = TaskSpec(name=name, domain=DOMAIN, adapter=ADAPTER_NAME,
                               build_version=version, train_size=train_size,
                               mean_tokens=mean_tokens, **kwargs)

    # Tier A: free labels, exact answers, fresh every pass (§2 "Passes").
    for task in TIER_A_TRAIN_TASKS:
        add(f"{MOLECULE_PREFIX}{task}", kind="generator", answer_kind="token",
            metric="exact_match", passes=1,
            cap_per_pass=config.tier_a_cap_per_pass,
            question_template=_tier_a_question_template(task))

    for task in HELD_OUT_TIER_A_TASKS:
        add(f"{MOLECULE_PREFIX}{task}", kind="generator", answer_kind="token",
            metric="exact_match", held_out=True, passes=1,
            cap_per_pass=config.held_out_size, eval_splits=("held_out",),
            question_template=_tier_a_question_template(task))

    # Tier B: yes/no, scored by the logit margin (`molecules/evaluate.py`).
    for task in config.tier_b_corpora:
        template = QUESTION_TEMPLATES[task]
        add(f"{MOLECULE_PREFIX}{task}", kind="corpus", answer_kind="yesno",
            metric="roc_auc", passes=3,
            question_template=template if isinstance(template, str) else None)

    for task in HELD_OUT_CORPORA:
        add(f"{MOLECULE_PREFIX}{task}", kind="corpus", answer_kind="yesno",
            metric="roc_auc", held_out=True, passes=1,
            eval_splits=("held_out",),
            question_template=question_for(task, "CT_TOX"))

    add(f"{MOLECULE_PREFIX}{CHEBI_TASK}", kind="corpus", answer_kind="text",
        metric="bleu2", passes=3, max_new_tokens=256,
        question_template=CHEBI_QUESTION)

    add(f"{MOLECULE_PREFIX}{G2S_TASK}", kind="generator", answer_kind="smiles",
        metric="roundtrip_match", passes=1,
        cap_per_pass=config.g2s_cap_per_pass, max_new_tokens=256,
        question_template=G2S_QUESTION)

    return specs


def _tier_a_question_template(task: str) -> str:
    """The family's question, taken from its generator's own docstring source.

    Tier-A questions are formatted per example (they name an atom, or a
    functional group), so there is no single literal to read off `tasks.py`. What
    the registry records is the family's shape, which is what "the model is
    routed by the question alone" means for a generator.
    """
    return {
        "ring_membership": "Question: is atom {atom} part of a ring?",
        "aromatic_ring": "Question: is atom {atom} part of an aromatic ring?",
        "ring_size": "Question: what is the size of the smallest ring containing "
                     "atom {atom}?",
        "ring_count": "Question: how many rings does this molecule have?",
        "fg_presence": "Question: does this molecule contain a {group}?",
        "fg_count": "Question: how many {group}s does this molecule contain?",
        "fg_atom_membership": "Question: is atom {atom} part of a {group}?",
        "stereo_potential": "Question: how many atoms in this molecule could be "
                            "stereocenters?",
        "stereo_assigned": "Question: how many stereocenters in this molecule "
                           "have a defined configuration?",
        "bond_path": "Question: how many bonds separate atom {a} and atom {b}?",
        "longest_chain": "Question: how many carbon atoms are in the longest "
                         "unbranched chain of non-ring carbons?",
    }[task]


def register_molecule_tasks(registry: Registry, config: MoleculeAdapterConfig,
                            arm: str = "graph") -> Registry:
    """Put every ``mol/`` task into ``registry``. D2's single table."""
    for spec in task_specs(config, arm=arm).values():
        registry.register(spec)
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# The molecules RunConfig this adapter drives the package with
# ─────────────────────────────────────────────────────────────────────────────

def _run_config(config: MoleculeAdapterConfig, task: str, arm: str):
    """A molecules ``RunConfig`` for one (task, arm).

    Built rather than re-implemented: `build_graph_example` and
    `build_flat_example` read `cfg.encoding`, `cfg.stereo_tags`,
    `cfg.question_node` and `cfg.task`, and going through them is what keeps one
    source of truth for node text. ``validate()`` is deliberately not called —
    ``chebi20`` and ``g2s`` are not molecules-package tasks, and the only thing
    the builders ask of ``cfg.task`` is whether it is atom-level.
    """
    from ...experiments.molecules.config import RunConfig

    return RunConfig(
        task=task, arm=arm, encoding=config.encoding,
        stereo_tags=config.stereo_tags, question_node=config.question_node,
        bias="none" if arm == "flat" else "spd+magnetic",
        max_spd=config.max_spd, magnetic_q=config.magnetic_q,
        magnetic_m=config.magnetic_m, model_name=config.model_name,
        ordering=config.ordering, data_seed=config.data_seed,
    )


def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    cached = _TOKENIZERS.get(model_name)
    if cached is None:
        cached = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZERS[model_name] = cached
    return cached


_TOKENIZERS = {}


def _draw_rng(config: MoleculeAdapterConfig, task: str, split: str, pass_id: int):
    """The draw's RNG. **Arm is not in the seed, on purpose.**

    Both arms must see the same molecules and the same questions, or the
    comparison carries an uncontrolled difference. `random.Random` seeded with a
    string is stable across processes and interpreter runs, which `hash()` of a
    tuple is not.
    """
    return random.Random(f"molecules|{config.data_seed}|{task}|{split}|{pass_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Drawing — one function per family, all returning the same shape
# ─────────────────────────────────────────────────────────────────────────────
#
# A *draw* is ``(mol, question, answer, named_atoms, key, meta)`` with ``answer``
# already in the molecules package's convention (leading space) and ``meta``
# JSON-serialisable. Drawing is separated from building because it is the half
# that must be arm-independent: the same molecules and the same questions go into
# both arms, and only the input representation differs.


def _pool(config: MoleculeAdapterConfig, part: Partition):
    """The generator pool, split by role. Deterministically ordered.

    A molecule is in exactly one of these lists, which is what makes the two
    enforcement points of §3 Rules 2 and 3 structural rather than a filter
    applied after the fact: a generator handed the ``train`` list *cannot* emit
    a test-role example.
    """
    from ...experiments.molecules.data import load_tier_b

    by_role = {role: [] for role in ROLES}
    seen = set()
    for name in config.pool:
        records, _spec, _dropped = load_tier_b(name)
        for record in records:
            key = partition_key(record["mol"])
            if key in seen:
                continue            # one molecule, one role, one pool entry
            seen.add(key)
            role = part.role(key)
            if role is None:
                raise AdapterBuildError(
                    f"{name}: key {key[:40]!r} is in the pool and not in the "
                    "partition; the two were built from different sources")
            by_role[role].append((record["mol"], key))
    return by_role


def _held_out_pool(config: MoleculeAdapterConfig, part: Partition):
    """Molecules for the held-out Tier-A tasks: the ``held_out`` role, i.e. ClinTox.

    §4 holds out two *question families* and one *corpus*. Running the two
    families over held_out-role molecules makes the zero-shot number a statement
    about molecules training never saw in any form, rather than one about
    molecules it saw under a different question — the stronger of the two claims,
    and the one the enforcement rule ("a held-out example's key must be
    held_out-role") already forces.
    """
    from ...experiments.molecules.data import load_tier_b

    out, seen = [], set()
    for name in HELD_OUT_CORPORA:
        records, _spec, _dropped = load_tier_b(name)
        for record in records:
            key = partition_key(record["mol"])
            if key in seen or not part.is_role(key, "held_out"):
                continue
            seen.add(key)
            out.append((record["mol"], key))
    return out


def _draw_tier_a(config, task, split, pass_id, pool):
    """``n`` Tier-A examples for one family, from a role-uniform molecule list.

    Mirrors `dataset.generate_examples`: shuffle the pool and walk it, consuming
    molecules **without replacement** within a pass, so no molecule is used twice
    until every usable one has been used once. Repeats across passes are counted
    (`stats["repeats"]`) rather than left invisible, and a
    `SINGLE_EXAMPLE_TASKS` family refuses to repeat a molecule at all.

    The loop is here rather than in `generate_examples` for one reason: that
    function returns graphs, and the adapter needs the *molecule* behind each
    example — the partition key is per example, and §3 Rule 2 is only enforceable
    if every emitted example can name the molecule it came from. Every piece of
    content is still the package's: the generator is `TASK_GENERATORS[task]`, the
    single-example rule is `dataset.SINGLE_EXAMPLE_TASKS`, and the graph is built
    by `build_graph_example` / `build_flat_example` downstream.
    """
    from ...experiments.molecules.dataset import SINGLE_EXAMPLE_TASKS
    from ...experiments.molecules.tasks import TASK_GENERATORS

    n = _tier_a_size(config, split)
    generator = TASK_GENERATORS[task]
    single = task in SINGLE_EXAMPLE_TASKS
    rng = _draw_rng(config, task, split, pass_id)

    draws, emitted, used = [], set(), set()
    stats = {"attempts": 0, "repeats": 0, "answers": {},
             "pool": len(pool), "requested": n}
    while len(draws) < n:
        order = list(range(len(pool)))
        rng.shuffle(order)
        produced = 0
        for i in order:
            if len(draws) >= n:
                break
            mol, key = pool[i]
            stats["attempts"] += 1
            made = generator(mol, rng)
            if made is None:
                continue
            question, answer, named = made
            if (i, question, answer) in emitted:
                stats["repeats"] += 1
            emitted.add((i, question, answer))
            used.add(i)
            produced += 1
            stats["answers"][answer] = stats["answers"].get(answer, 0) + 1
            draws.append((mol, question, answer, list(named), key,
                          {"family": task, "pool_index": i}))
        if len(draws) >= n:
            break
        if single:
            raise AdapterBuildError(
                f"{task!r} yields one example per molecule, but the {split} pool "
                f"has {produced} usable molecules of {len(pool)} and {n} examples "
                "were requested. Repeating a molecule would put an identical "
                "example in the split twice; widen `pool` or lower the size.")
        if produced == 0:
            raise AdapterBuildError(
                f"{task}: the generator refused every molecule in the {split} pool "
                f"({len(pool)} molecules).")
    stats["molecules"] = len(used)
    return draws, stats


def _tier_a_size(config, split):
    return {"train": config.tier_a_cap_per_pass, "val": config.tier_a_val_size,
            "test": config.tier_a_test_size,
            "held_out": config.held_out_size}[split]


def _draw_tier_b(config, task, split, part):
    """Tier-B (molecule, endpoint) yes/no examples, filtered by the partition.

    `build_tier_b_examples` does the whole job — the scaffold split, one example
    per labelled ``(molecule, endpoint)`` with the endpoint substituted into the
    question, missing labels skipped rather than imputed, and split membership
    decided per *molecule* so no structure spans two splits.

    What is added here is §3 Rule 1: a molecule this corpus would train on, which
    another source holds at a higher role, is dropped from the **train** split
    and counted. Val and test keep theirs whatever role they hold — a ClinTox
    molecule sitting in BACE's test set is not trained on either way, and
    removing it would shrink the split the published anchors are quoted on.
    """
    from ...experiments.molecules.tier_b import build_tier_b_examples

    splits, stats = build_tier_b_examples(task)
    endpoints = _endpoint_of_question(task)

    draws = []
    dropped = 0
    per_endpoint = {}
    for mol, question, answer in splits[split]:
        key = partition_key(mol)
        if split == "train" and not part.is_role(key, "train"):
            dropped += 1
            continue
        endpoint = endpoints.get(question)
        per_endpoint[endpoint] = per_endpoint.get(endpoint, 0) + 1
        draws.append((mol, question, answer, [], key,
                      {"corpus": task, "endpoint": endpoint}))
    stats = dict(stats)
    stats["partition_dropped"] = dropped
    stats["emitted_per_endpoint"] = per_endpoint
    stats["emitted"] = len(draws)
    return draws, stats


def _endpoint_of_question(task: str) -> dict:
    """``question text -> endpoint``, inverted from `tier_b.question_for`.

    The question is the only thing that names the endpoint, so inverting the
    template is how an emitted example gets its endpoint back into ``meta`` and
    into the per-endpoint counts §1 asks the run record to carry.
    """
    from ...experiments.molecules.data import TIER_B
    from ...experiments.molecules.tier_b import QUESTION_TEMPLATES, question_for

    template = QUESTION_TEMPLATES[task]
    if isinstance(template, dict):
        columns = tuple(template)
    else:
        columns = TIER_B[task].task_cols
        if not columns:
            columns = _corpus_columns(task)
    return {question_for(task, endpoint): endpoint for endpoint in columns}


def _corpus_columns(task: str) -> tuple:
    """The endpoint columns of a multi-endpoint corpus, by `load_tier_b`'s rule."""
    from ...experiments.molecules.data import TIER_B, load_tier_b

    records, spec, _dropped = load_tier_b(task)
    if spec.task_cols:
        return tuple(spec.task_cols)
    return tuple(c for c in records[0]["targets"]
                 if c not in (TIER_B[task].smiles_col, "mol_id"))


def _draw_held_out_corpus(config, task, part):
    """Every labelled ``(molecule, endpoint)`` of a held-out corpus, as one split.

    ClinTox never trains, so its own scaffold split is meaningless here: the
    whole corpus is the held-out evaluation set.
    """
    from ...experiments.molecules.tier_b import build_tier_b_examples

    splits, stats = build_tier_b_examples(task)
    endpoints = _endpoint_of_question(task)
    draws = []
    for name in ("train", "val", "test"):
        for mol, question, answer in splits[name]:
            key = partition_key(mol)
            draws.append((mol, question, answer, [], key,
                          {"corpus": task, "endpoint": endpoints.get(question),
                           "source_split": name}))
    stats = dict(stats)
    stats["emitted"] = len(draws)
    return draws, stats


def _draw_chebi(config, split, part, chebi=None):
    """ChEBI-20 captioning draws. Answer is the description, kind ``text``.

    The description carries no leading space; the ``" " + answer`` handed to the
    builders is what makes the prompt tail ``"\\nA: The molecule is …"`` — see the
    module docstring.
    """
    if chebi is None:
        chebi, _stats = load_chebi(config)
    records = chebi[split]
    draws, dropped = [], 0
    for record in records:
        if split == "train" and not part.is_role(record["key"], "train"):
            dropped += 1
            continue
        draws.append((record["mol"], CHEBI_QUESTION, " " + record["text"], [],
                      record["key"], {"cid": record["cid"]}))
    return draws, {"available": len(records), "partition_dropped": dropped,
                   "emitted": len(draws)}


def _draw_g2s(config, split, pass_id, pool):
    """graph-to-SMILES draws: one example per molecule per pass (§5).

    Molecules whose graph fails `roundtrip_check` at the ``exact`` level are
    excluded and counted — the graph does not determine them, so the task would
    be asking for information the input does not carry. The encoding itself must
    be an ``exact``-level one for the same reason, and a weaker one is refused
    rather than silently producing an ill-posed task.
    """
    from ...experiments.molecules.data import ROUNDTRIP_LEVEL, roundtrip_check

    level = ROUNDTRIP_LEVEL[config.encoding]
    if level != "exact":
        raise AdapterBuildError(
            f"graph-to-SMILES needs an encoding whose round trip is exact; "
            f"{config.encoding!r} is declared {level!r} in `data.ROUNDTRIP_LEVEL`, "
            "so the graph does not determine the molecule and the target is not a "
            "function of the input.")

    n = {"train": config.g2s_cap_per_pass, "val": config.g2s_val_size,
         "test": config.g2s_test_size}[split]
    rng = _draw_rng(config, G2S_TASK, split, pass_id)
    order = list(range(len(pool)))
    rng.shuffle(order)

    draws = []
    stats = {"pool": len(pool), "requested": n, "dropped": {
        "roundtrip": 0, "not_canonical": 0}}
    for i in order:
        if len(draws) >= n:
            break
        mol, key = pool[i]
        ok, _level, _expected, _got = roundtrip_check(
            mol, encoding=config.encoding, stereo_tags=config.stereo_tags)
        if not ok:
            stats["dropped"]["roundtrip"] += 1
            continue
        target = g2s_target(mol)
        if target is None:
            stats["dropped"]["not_canonical"] += 1
            continue
        draws.append((mol, G2S_QUESTION, " " + target, [], key,
                      {"pool_index": i}))
    stats["emitted"] = len(draws)
    return draws, stats


# ─────────────────────────────────────────────────────────────────────────────
# Building — draws become graphs, graphs become a TextGraphDataset
# ─────────────────────────────────────────────────────────────────────────────

def _flat_graph(question: str, smiles: str, answer: str):
    """The flat arm's single-node graph, for a SMILES this adapter chose itself.

    Byte-identical to `dataset.build_flat_example`; it exists only because that
    function serializes the *canonical* SMILES, and graph-to-SMILES needs a
    randomized one with stereo (§5: the flat twin's task is canonicalization).
    `test_molecules_adapter.py` pins the two against each other on a canonical
    case, so the duplication cannot drift.
    """
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(0, text=f"{question}\nSMILES: {smiles}\nA:{answer}",
                   kind="prompt")
    graph.graph["prompt_node"] = 0
    return graph


def _graphs_for(config, task, arm, draws, pass_id):
    """Turn draws into arm-appropriate nx graphs, through the molecules package."""
    from ...experiments.molecules.data import flat_serialize
    from ...experiments.molecules.dataset import build_flat_example, build_graph_example

    cfg = _run_config(config, task, arm)
    graphs = []
    for i, (mol, question, answer, named, _key, _meta) in enumerate(draws):
        if arm == "graph":
            graphs.append(build_graph_example(mol, question, answer, named, cfg))
        elif task == G2S_TASK:
            # A fresh randomization every pass, so the flat twin cannot memorise
            # one spelling of a molecule it will see again (§5).
            smiles = flat_serialize(
                mol, canonical=False,
                seed=(config.data_seed * 1_000_003 + pass_id * 7919 + i))
            graphs.append(_flat_graph(question, smiles, answer))
        else:
            graphs.append(build_flat_example(mol, question, answer, cfg))
    return graphs


def _answer_kind(task: str) -> str:
    if task in TIER_A_TRAIN_TASKS or task in HELD_OUT_TIER_A_TASKS:
        return "token"
    if task == CHEBI_TASK:
        return "text"
    if task == G2S_TASK:
        return "smiles"
    return "yesno"


def _labels_fn(tokenizer, answer_kind: str, max_length: int):
    """The dataset's ``labels`` column: `schema.render`'s span, not a copy of it.

    `dataset.get_prompt_node_labels` supervises the prompt node's *last* token,
    which is right for the single-token kinds and wrong for a caption. So the
    span comes from `render` itself, on a one-node stub holding the prompt text:
    labels are aligned to the prompt node's tokens and depend on nothing else in
    the graph, so the stub gives the same answer as the full example at the cost
    of one tokenizer call instead of fifty.

    The answer is read back out of the prompt text (the tail after the last
    ``"\\nA:"``) rather than passed in, because `compute_labels` hands its
    callable one row and no index. It differs from the stored answer by a leading
    space for the multi-token kinds, and that cannot move the span: the extra
    space either merges into the answer's first token — in which case both
    spellings break the prefix comparison at the same position — or tokenizes
    alone, in which case the shorter prefix simply runs out at that position.
    Both land on the same ``answer_start``.
    """
    from ..schema import ANSWER_PREFIX

    def get_labels(example):
        """One row -> its ``labels`` list, aligned to the prompt node's tokens."""
        text = example["text"][example["prompt_node"]]
        index = text.rfind(ANSWER_PREFIX)
        if index < 0:
            raise SchemaError(
                f"prompt node {text[:80]!r} has no {ANSWER_PREFIX!r}; the "
                "supervised span cannot be located")
        answer = text[index + len(ANSWER_PREFIX):]
        stub = Example(
            task="_", domain=DOMAIN, split="train", arm="flat",
            graph={"text": [text], "prompt_node": 0, "num_nodes": 1},
            question="_", answer=answer, answer_kind=answer_kind, key="_")
        return render(stub, tokenizer, max_length=max_length).labels

    return get_labels


def _materialise(config, task, split, arm, pass_id, draws, spec):
    """Build, featurize, validate and save one ``(task, split, arm, pass)``.

    Featurization is `dataset.prepare_dataset`'s, minus its Tier-A/Tier-B
    branching: tokenize, labels, SPD, magnetic Laplacian, cast to fp32. Both
    feature families always, so one artifact serves every bias arm; on the flat
    arm they are 1x1 tensors, which is free and keeps the two arms' pipelines
    byte-identical downstream.
    """
    from ...utils import TextGraphDataset

    tokenizer = _tokenizer(config.model_name)
    answer_kind = spec.answer_kind
    graphs = _graphs_for(config, task, arm, draws, pass_id)

    ds = TextGraphDataset(graphs, rcm_ordering=(config.ordering == "rcm"))
    ds.tokenize(tokenizer, max_length=config.max_length)
    ds.compute_labels(_labels_fn(tokenizer, answer_kind, config.max_length),
                      num_proc=1)
    ds.compute_shortest_path_distances()
    ds.compute_magnetic_lap(q=config.magnetic_q, m=config.magnetic_m)
    ds.cast_float_features_to_fp32()

    name = f"{MOLECULE_PREFIX}{task}"
    records, num_nodes, num_tokens = [], [], []
    for i, (_mol, question, answer, _named, key, meta) in enumerate(draws):
        stored = answer if answer_kind in ("token", "yesno") else answer[1:]
        record = {"question": question, "answer": stored, "key": key,
                  "meta": dict(meta)}
        item = _item(ds, i, name)
        example = Example(task=name, domain=DOMAIN, split=split, arm=arm,
                          graph=item, question=question, answer=stored,
                          answer_kind=answer_kind, key=key, meta=record["meta"])
        validate(example, spec)
        records.append(record)
        num_nodes.append(int(item["num_nodes"]))
        num_tokens.append(sum(len(ids) for ids in item["input_ids"]))

    path = config.source_path(task, split, arm, pass_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save(path)
    sidecar = {
        "task": name, "molecules_task": task, "split": split, "arm": arm,
        "pass_id": int(pass_id), "domain": DOMAIN, "answer_kind": answer_kind,
        "build_version": config.build_version(),
        "partition_path": config.partition_path(),
        "records": records, "num_nodes": num_nodes, "num_tokens": num_tokens,
    }
    with open(_sidecar_path(path), "w") as f:
        json.dump(sidecar, f)
    return sidecar


def _sidecar_path(path: str) -> str:
    return path + ".schema.json"


def _item(ds, i: int, task: str) -> dict:
    """One ``TextGraphDataset`` item, with the question node's index restored.

    ``question_node`` is a *graph-level* attribute — `TextGraphDataset` keeps it
    on the pickled nx graph and never puts it in the Arrow table, so an item read
    straight off the dataset does not carry it. The schema needs it: with
    ``question_node: on`` the question lives in its own prefix node and is *not*
    in the prompt text, so without this the validator would look for the question
    in the prompt and fail every graph-arm example.
    """
    item = dict(ds[i])
    item["ds_label"] = task
    item["question_node"] = int(ds.graphs[i].graph.get("question_node", -1))
    return item


# ─────────────────────────────────────────────────────────────────────────────
# The TaskSource the mixture batches over
# ─────────────────────────────────────────────────────────────────────────────

class MoleculeTaskSource:
    """One built ``(task, split, arm, pass)``. See `adapters.TaskSource`.

    Holds the `TextGraphDataset` and the schema fields beside it; ``__getitem__``
    puts the two back together as ``Example.to_item()``, which is the dict
    ``GraphCollatorV2`` collates and the mixture routes by.
    """

    def __init__(self, dataset, sidecar: dict, path: str):
        self._ds = dataset
        self._records = sidecar["records"]
        self._num_nodes = sidecar["num_nodes"]
        self._num_tokens = sidecar["num_tokens"]
        self.path = path
        self.task = sidecar["task"]
        self.molecules_task = sidecar["molecules_task"]
        self.split = sidecar["split"]
        self.arm = sidecar["arm"]
        self.pass_id = int(sidecar["pass_id"])
        self.domain = sidecar["domain"]
        self.answer_kind = sidecar["answer_kind"]
        self.build_version = sidecar["build_version"]
        if len(self._records) != len(self._ds):
            raise AdapterBuildError(
                f"{path}: {len(self._records)} schema records beside "
                f"{len(self._ds)} graphs; the artifact and its sidecar disagree")

    @property
    def dataset(self):
        """The underlying `TextGraphDataset`, for callers that want it whole."""
        return self._ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> dict:
        record = self._records[i]
        example = Example(
            task=self.task, domain=self.domain, split=self.split, arm=self.arm,
            graph=_item(self._ds, i, self.task), question=record["question"],
            answer=record["answer"], answer_kind=self.answer_kind,
            key=record["key"], meta=record["meta"])
        return example.to_item()

    def example(self, i: int) -> Example:
        """The :class:`Example` behind item ``i``, for validators and reports."""
        from ..schema import Example as _Example

        return _Example.from_item(self[i], None, split=self.split)

    def keys(self) -> list:
        return [record["key"] for record in self._records]

    def lengths(self) -> tuple:
        """``(num_nodes, num_tokens)``, measured at build time.

        Stored rather than computed on demand: the mixture builds its bucket
        table over every source before the first step, and a full pass over the
        Arrow table per source at startup is minutes of a GPU job spent counting.
        """
        return list(self._num_nodes), list(self._num_tokens)

    def __repr__(self) -> str:
        return (f"<MoleculeTaskSource {self.task} {self.split}/{self.arm} "
                f"p{self.pass_id} n={len(self)}>")


# ─────────────────────────────────────────────────────────────────────────────
# D3's three functions
# ─────────────────────────────────────────────────────────────────────────────

#: The config ``load`` uses when it is not given one. Set by :func:`configure`
#: and by :func:`build`. The D3 protocol is ``load(task, split, arm)`` — no
#: config — so the cache root and build version have to reach it somehow; an
#: explicit ``config=`` keyword overrides this for anything that has one to hand.
_ACTIVE_CONFIG = None


def configure(config: MoleculeAdapterConfig) -> MoleculeAdapterConfig:
    """Make ``config`` the one :func:`load` resolves paths against."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config.validate()
    return _ACTIVE_CONFIG


def _resolve(config):
    if config is not None:
        return config
    if _ACTIVE_CONFIG is None:
        raise AdapterBuildError(
            "no adapter config: call molecules.configure(cfg) (or build(cfg)) "
            "before load(), or pass config= explicitly")
    return _ACTIVE_CONFIG


def splits_for(task: str) -> tuple:
    """Which splits a task may be built for.

    A held-out task admits ``held_out`` and nothing else, in both enforcement
    points (D2.1): the schema refuses the example, and this refuses the build.
    """
    if task in HELD_OUT_TIER_A_TASKS or task in HELD_OUT_CORPORA:
        return ("held_out",)
    return ("train", "val", "test")


def all_tasks(config: MoleculeAdapterConfig) -> tuple:
    return (TIER_A_TRAIN_TASKS + HELD_OUT_TIER_A_TASKS
            + tuple(config.tier_b_corpora) + HELD_OUT_CORPORA
            + (CHEBI_TASK, G2S_TASK))


def build(config: MoleculeAdapterConfig, roles: Partition = None, tasks=None,
          arms=("graph", "flat"), splits=None, passes: int = 1,
          rebuild: bool = False) -> dict:
    """Materialise every requested ``(task, split, arm, pass)`` on disk.

    ``roles`` is the :class:`Partition`; it is computed if not given. ``tasks``
    defaults to everything this adapter owns, and is the knob a data_prep job
    uses to build one corpus at a time.

    Two refusals, both structural rather than checked after the fact
    (`MOLECULE_GENERALIST.md` §3 Rules 2 and 3):

    * a **train** example whose key is not ``train``-role, and
    * a **held_out** example whose key is not ``held_out``-role.

    Generators are handed a role-filtered pool, so neither can happen by
    construction; the assertion runs anyway, because "cannot happen" is what the
    2026-08 leak was also believed to be.
    """
    config.validate()
    configure(config)
    part = roles if roles is not None else partition(config)
    specs = task_specs(config)
    tasks = tuple(tasks) if tasks is not None else all_tasks(config)
    arms = tuple(arms)

    pools = None
    chebi = None
    manifest = read_manifest(config)
    manifest.setdefault("build_version", config.build_version())
    manifest.setdefault("partition_version", config.partition_version())
    manifest.setdefault("tasks", {})
    manifest["partition"] = {"counts": part.counts, "ledger": part.ledger,
                             "role_totals": part.role_totals,
                             "path": config.partition_path()}
    manifest["endpoints"] = part.meta.get("endpoints", {})
    manifest["chebi"] = part.meta.get("sources", {}).get("chebi20", {})

    _assert_single_token_answers(config)

    for task in tasks:
        spec = specs[f"{MOLECULE_PREFIX}{task}"]
        allowed = splits_for(task)
        wanted = tuple(s for s in allowed if splits is None or s in splits)
        entry = manifest["tasks"].setdefault(
            f"{MOLECULE_PREFIX}{task}", {"arms": {}, "splits": {}})
        for split in wanted:
            n_passes = passes if (split == "train" and spec.kind == "generator") else 1
            for pass_id in range(n_passes):
                if pools is None and _needs_pool(task):
                    pools = _pool(config, part)
                if chebi is None and task == CHEBI_TASK:
                    chebi, _chebi_stats = load_chebi(config)
                built = {arm: _sidecar_path(config.source_path(
                    task, split, arm, pass_id)) for arm in arms}
                if rebuild or not all(os.path.exists(p) for p in built.values()):
                    draws, stats = _draws(config, task, split, pass_id, part,
                                          pools, chebi)
                    _check_roles(part, task, split, draws)
                    entry["splits"][f"{split}.p{pass_id}"] = stats
                for arm in arms:
                    if rebuild or not os.path.exists(built[arm]):
                        sidecar = _materialise(config, task, split, arm,
                                               pass_id, draws, spec)
                    else:
                        with open(built[arm]) as f:
                            sidecar = json.load(f)
                    # `mean_tokens` and `train_size` are properties of the built
                    # data (D2), so they are read off the artifact whether this
                    # call produced it or found it — a warm cache must leave the
                    # registry as complete as a cold one.
                    #
                    # A held-out task has no train split, and `held_out` is the
                    # split its one sanctioned training run reads: an `adapt` fork
                    # (D6) trains on exactly that split, and `registry.resolve`
                    # needs `mean_tokens` before it can turn a token budget into
                    # examples. Measuring it here is what makes that fork
                    # resolvable; nothing else ever trains a held-out task.
                    measured_split = "held_out" if spec.held_out else "train"
                    if split == measured_split and pass_id == 0:
                        tokens = sidecar["num_tokens"]
                        entry["arms"][arm] = {
                            "train_size": len(tokens),
                            "mean_tokens": (sum(tokens) / len(tokens))
                            if tokens else None,
                        }

    write_manifest(config, manifest)
    return manifest


def _needs_pool(task: str) -> bool:
    return (task in TIER_A_TRAIN_TASKS or task in HELD_OUT_TIER_A_TASKS
            or task == G2S_TASK)


def _draws(config, task, split, pass_id, part, pools, chebi):
    """Dispatch to the family's draw function. One place, so `build` stays flat."""
    if task in TIER_A_TRAIN_TASKS:
        return _draw_tier_a(config, task, split, pass_id,
                            pools[_role_for(split)])
    if task in HELD_OUT_TIER_A_TASKS:
        return _draw_tier_a(config, task, "held_out", pass_id,
                            _held_out_pool(config, part))
    if task in HELD_OUT_CORPORA:
        return _draw_held_out_corpus(config, task, part)
    if task == CHEBI_TASK:
        return _draw_chebi(config, split, part, chebi)
    if task == G2S_TASK:
        return _draw_g2s(config, split, pass_id, pools[_role_for(split)])
    return _draw_tier_b(config, task, split, part)


def _role_for(split: str) -> str:
    """§3 Rules 2 and 3: train draws from ``train``, test from ``test``."""
    return split


def _check_roles(part: Partition, task: str, split: str, draws) -> None:
    """The build-time half of the enforcement (D3.3)."""
    if split not in ("train", "held_out"):
        return
    for _mol, _question, _answer, _named, key, _meta in draws:
        if not part.is_role(key, split):
            raise PartitionError(
                f"{task}/{split}: key {key[:60]!r} has role "
                f"{part.role(key)!r}, not {split!r}. "
                + ("A training example over a test-role molecule is the leak "
                   "MOLECULE_GENERALIST.md §3 exists to prevent."
                   if split == "train" else
                   "A held-out example must be over a held_out-role molecule, or "
                   "the zero-shot number is over molecules training saw."))


def _assert_single_token_answers(config) -> None:
    """`tasks.py`'s answer vocabulary is one token, under this tokenizer.

    `dataset.prepare_dataset` asserts it before every build for the same reason:
    last-token supervision would not cover a two-token answer, and the logit
    margin would compare *first* tokens, which may not distinguish the classes.
    """
    from ...experiments.molecules.tasks import ANSWER_VOCAB

    tokenizer = _tokenizer(config.model_name)
    for answer in ANSWER_VOCAB:
        n = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
        if n > 2:
            raise AdapterBuildError(
                f"answer {answer!r} tokenizes to {n} tokens under "
                f"{config.model_name}; last-token supervision would not cover it")


def load(task: str, split: str, arm: str, pass_id: int = 0, config=None,
         check_keys: int = 200) -> MoleculeTaskSource:
    """The built source for one ``(task, split, arm, pass)``. Never regenerates.

    A resume that rebuilt a pass would be a resume that changes the data under
    the sampler, so an absent artifact is an error naming the ``data_prep`` that
    should have produced it rather than a silent build.

    ``check_keys`` re-checks that many randomly chosen keys against the
    partition, which is D3.3's load-time half of the enforcement. It is cheap
    (one dict lookup each) and it is the check that catches a cache built under
    one partition being read under another.
    """
    config = _resolve(config)
    name = task[len(MOLECULE_PREFIX):] if task.startswith(MOLECULE_PREFIX) else task
    if split not in splits_for(name):
        raise AdapterBuildError(
            f"{task}: split {split!r} is not one of {splits_for(name)}"
            + (" — this task is held out and never trains (D2.1)."
               if splits_for(name) == ("held_out",) else ""))

    path = config.source_path(name, split, arm, pass_id)
    sidecar_path = _sidecar_path(path)
    if not os.path.exists(sidecar_path):
        raise AdapterBuildError(
            f"{path} has not been built. Run data_prep for {task} "
            f"({split}/{arm}, pass {pass_id}) — load never generates, because a "
            "resume that rebuilt a pass would change the data under the sampler.")
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    from ...utils import TextGraphDataset

    source = MoleculeTaskSource(TextGraphDataset.load(path), sidecar, path)
    if check_keys and split in ("train", "held_out"):
        _recheck_keys(config, source, split, check_keys)
    return source


def _recheck_keys(config, source: MoleculeTaskSource, split: str, n: int) -> None:
    part = partition(config)
    keys = source.keys()
    rng = random.Random(f"recheck|{source.task}|{split}|{source.pass_id}")
    sample = keys if len(keys) <= n else rng.sample(keys, n)
    for key in sample:
        if not part.is_role(key, split):
            raise PartitionError(
                f"{source.task}/{split}: built key {key[:60]!r} has role "
                f"{part.role(key)!r} under the partition at "
                f"{config.partition_path()}. The artifact and the partition were "
                "built from different sources; delete the build and rerun "
                "data_prep.")


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def read_manifest(config: MoleculeAdapterConfig) -> dict:
    path = config.manifest_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_manifest(config: MoleculeAdapterConfig, manifest: dict) -> str:
    path = config.manifest_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp, path)
    return path
