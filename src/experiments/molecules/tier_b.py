"""
Tier B — MoleculeNet property prediction (PLAN.md §1 Tier B).

Three things distinguish it from Tier A, and each is a design decision rather
than a detail:

1. **Scaffold split, not random.** Molecules are grouped by Bemis-Murcko scaffold
   and no scaffold spans two splits, so the test set is structurally novel. This
   is what the published baselines do and it is why their numbers are so much
   lower than random-split ones would be.

2. **One example per `(molecule, endpoint)`, with the endpoint named in the
   QUESTION node.** Tox21 has 12 endpoints and SIDER 27; the alternative is 39
   classifier heads. Naming the endpoint in text is the GTLM-native choice and it
   makes the multi-endpoint sets *free* multi-task data for PLAN.md §4 arm 2.
   Missing labels (NaN) are skipped rather than imputed — MoleculeNet's
   multi-endpoint sets are sparsely labelled and imputing would invent positives.

3. **Split membership follows the molecule, not the example.** Every endpoint of
   one molecule lands in the same split, or the scaffold guarantee is void: the
   same structure would appear in train and test under a different question.
"""

from __future__ import annotations

import math

from .data import TIER_B, load_tier_b, scaffold_split

TIER_B_TASKS = tuple(TIER_B)

#: How each endpoint becomes a question. ``{endpoint}`` is substituted with a
#: readable form of the column name. Phrasing is deliberately plain: the question
#: is the only thing telling the trunk which task it is being asked.
QUESTION_TEMPLATES = {
    "bace": "Question: does this molecule inhibit human beta-secretase 1?",
    "bbbp": "Question: does this molecule penetrate the blood-brain barrier?",
    "hiv": "Question: does this molecule inhibit HIV replication?",
    "tox21": "Question: is this molecule active in the {endpoint} toxicity assay?",
    "clintox": {
        "FDA_APPROVED": "Question: is this molecule an FDA-approved drug?",
        "CT_TOX": "Question: did this molecule fail clinical trials for toxicity?",
    },
    "sider": "Question: does this molecule cause side effects in the category "
             "{endpoint}?",
}

#: Regression sets — out of scope for the yes/no margin readout; they need the
#: `numeric_text` path (PLAN.md §8 M3). Listed so `validate` can say so clearly
#: rather than producing a nonsense binary label.
REGRESSION_TASKS = ("esol", "freesolv", "lipo")


def _readable(endpoint: str) -> str:
    """Endpoint column name as it should appear in the question.

    Near-identity on purpose. Tox21's columns are assay *codes* ("NR-AR" =
    nuclear receptor, androgen receptor) that a chemistry-pretrained model has
    seen verbatim; lowercasing and de-hyphenating them into "nr ar" destroys that
    and gains nothing. SIDER's columns are already natural language
    ("Hepatobiliary disorders"). Only underscores need touching.
    """
    return endpoint.replace("_", " ").strip()


def question_for(task: str, endpoint: str) -> str:
    template = QUESTION_TEMPLATES[task]
    if isinstance(template, dict):
        return template[endpoint]
    return template.format(endpoint=_readable(endpoint))


def _label(value):
    """MoleculeNet stores 0/1 with NaN for 'not measured'. Returns None for NaN."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return bool(int(float(value)))
    except (TypeError, ValueError):
        return None


def build_tier_b_examples(task, yes=" Yes", no=" No"):
    """Return ``(splits, stats)`` where ``splits`` is ``{"train"|"val"|"test":
    [(mol, question, answer), ...]}``.

    Split membership is decided per *molecule* (see the module docstring), then
    every labelled endpoint of that molecule is emitted into the same split.
    """
    if task in REGRESSION_TASKS:
        raise NotImplementedError(
            f"{task!r} is a regression set; the yes/no margin readout does not "
            "apply. It needs the numeric_text path (PLAN.md §7.2 / M3).")

    records, spec, dropped = load_tier_b(task)
    task_cols = spec.task_cols or tuple(
        c for c in records[0]["targets"] if c not in (spec.smiles_col, "mol_id"))

    train_idx, val_idx, test_idx = scaffold_split([r["smiles"] for r in records])
    membership = {}
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in indices:
            membership[i] = name

    splits = {"train": [], "val": [], "test": []}
    stats = {"dropped": dropped, "molecules": len(records),
             "endpoints": len(task_cols), "unlabelled": 0,
             "positives": 0, "negatives": 0}

    for i, record in enumerate(records):
        split = membership[i]
        for endpoint in task_cols:
            label = _label(record["targets"].get(endpoint))
            if label is None:
                stats["unlabelled"] += 1
                continue
            stats["positives" if label else "negatives"] += 1
            splits[split].append(
                (record["mol"], question_for(task, endpoint), yes if label else no))

    stats["split_sizes"] = {k: len(v) for k, v in splits.items()}
    return splits, stats
