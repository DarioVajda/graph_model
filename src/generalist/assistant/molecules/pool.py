"""The molecule pool `pipeline/build.py` draws from, and each molecule's fact sheet.

Molecules and their fact sheets come from the §3 partition, sampled stratified by
source and heavy-atom count — the roles and the label splits are the part of the
pipeline that was never in question, and `_sources_and_labels`, `_captions` and
`_stratified` below are that sampling unchanged from the build the intent design
replaced.

`MoleculePool` is the molecule domain's subject source. The contract
`pipeline/build.py` relies on is four methods:

    by_role(roles)            {role: [entry, ...]}, the whole pool
    sample(entries, n, rng)   n entries drawn from one role
    sheet(entry, rng)         (subject_id, subject_string, [Fact]), or None
    fields(entry)             the per-row fields written beside the intent

`sample` and `sheet` are the only two that draw from ``rng``, and the build calls
them in a fixed order — every `sample` for a role, then `sheet` once per entry —
so a seed reproduces a build.
"""

from __future__ import annotations


#: Heavy-atom bands the pool is stratified over, alongside the source corpus.
SIZE_BUCKETS = ((0, 20), (21, 28), (29, 38), (39, 10000))


def _bucket(n: int) -> str:
    for low, high in SIZE_BUCKETS:
        if low <= n <= high:
            return f"{low}-{high}"
    return "other"


def _sources_and_labels(config, roles_wanted):
    """``(molecules, labels)`` — the pool by role with its source, and Tier-B labels.

    One pass over every corpus, because both halves come from the same records:
    `load_tier_b` gives the molecule and the corpus it came from, and
    `build_tier_b_examples` gives the labelled (molecule, endpoint) pairs the
    trunk actually trained on.
    """
    from ...adapters.molecules import (TIER_B_CORPORA, _endpoint_of_question,
                                       partition, partition_key)
    from ....experiments.molecules.data import load_tier_b
    from ....experiments.molecules.tier_b import build_tier_b_examples

    print("partition...", flush=True)
    part = partition(config)
    molecules, seen = [], set()
    for name in config.pool:
        print(f"pool {name}...", flush=True)
        records, _spec, _dropped = load_tier_b(name)
        for record in records:
            key = partition_key(record["mol"])
            if key in seen:
                continue
            seen.add(key)
            role = part.role(key)
            if role in roles_wanted:
                molecules.append({"mol": record["mol"], "key": key,
                                  "source": name, "role": role})

    labels = {}
    for corpus in TIER_B_CORPORA:
        print(f"labels {corpus}...", flush=True)
        splits, _stats = build_tier_b_examples(corpus)
        # The endpoint is recovered from the question the trunk was trained on,
        # so the fact sheet names the assay in the same words the trunk saw.
        endpoints = _endpoint_of_question(corpus)
        for split, examples in splits.items():
            for mol, question, answer in examples:
                key = partition_key(mol)
                labels.setdefault(key, []).append({
                    "corpus": corpus,
                    "endpoint": endpoints.get(question, ""),
                    "split": split,
                    "label": answer.strip().lower().startswith("yes")})
    return molecules, labels


def _captions(config) -> dict:
    """``key -> ChEBI-20 caption``, over every split."""
    from ...adapters.molecules import load_chebi

    chebi, _stats = load_chebi(config)
    out = {}
    for split, records in chebi.items():
        for record in records:
            out.setdefault(record["key"], (record["text"], split))
    return out


def _stratified(molecules, n, rng):
    """``n`` molecules, spread over (source, size bucket) in the pool's proportions.

    Largest-remainder rather than rounding each cell independently: rounding
    gives a total that misses ``n`` by however many cells there are, and the
    shortfall would land wherever the loop happened to stop.
    """
    cells = {}
    for entry in molecules:
        heavy = entry["mol"].GetNumHeavyAtoms()
        entry["heavy_atoms"] = heavy
        cells.setdefault((entry["source"], _bucket(heavy)), []).append(entry)

    total = sum(len(v) for v in cells.values())
    if total < n:
        raise SystemExit(f"the pool holds {total} molecules and {n} were asked for")
    exact = {cell: n * len(v) / total for cell, v in cells.items()}
    take = {cell: min(int(value), len(cells[cell]))
            for cell, value in exact.items()}
    short = n - sum(take.values())
    order = sorted(cells, key=lambda c: exact[c] - take[c], reverse=True)
    i = 0
    while short > 0:
        cell = order[i % len(order)]
        if take[cell] < len(cells[cell]):
            take[cell] += 1
            short -= 1
        i += 1

    out = []
    for cell, count in take.items():
        out.extend(rng.sample(cells[cell], count))
    rng.shuffle(out)
    return out


class MoleculePool:
    """The §3 pool with its Tier-B labels and ChEBI-20 captions, for one config.

    A test-role molecule's sheet may carry labels and a caption from any split; a
    train-role molecule's carries only the train split's, so nothing a test
    question could be graded on reaches training through a fact sheet.
    """

    def __init__(self, config):
        adapter = config.adapter_config()
        self._adapter = adapter
        self._labels = None
        self._captions = None
        self._molecules = None

    def by_role(self, roles) -> dict:
        roles = set(roles)
        self._molecules, self._labels = _sources_and_labels(self._adapter, roles)
        self._captions = _captions(self._adapter)
        return {role: [m for m in self._molecules if m["role"] == role]
                for role in sorted(roles)}

    def sample(self, entries, n, rng):
        return _stratified(entries, n, rng)

    def sheet(self, entry, rng):
        from .sheet import canonical_smiles, fact_sheet

        tier_b = []
        for record in self._labels.get(entry["key"], []):
            if entry["role"] == "train" and record["split"] != "train":
                continue
            tier_b.append((record["corpus"], record["endpoint"], record["label"]))
        caption, caption_split = self._captions.get(entry["key"], ("", ""))
        if entry["role"] == "train" and caption_split != "train":
            caption = ""

        sheet = fact_sheet(entry["mol"], rng=rng, tier_b=tier_b, caption=caption)
        if not sheet:
            return None
        return entry["key"], canonical_smiles(entry["mol"]), sheet

    def fields(self, entry) -> dict:
        return {"key": entry["key"], "role": entry["role"],
                "source": entry["source"], "heavy_atoms": entry["heavy_atoms"]}
