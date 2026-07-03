"""
Dataset loading + label-balance reporting for the expressiveness experiment.
"""

import fcntl
import os
import random

from ....utils import TextGraphDataset
from ..config import MAGNETIC_Q
from .data_gen import create_and_save_dataset, dataset_path_and_size


def calculate_label_distribution(dataset, tokenized_labels):
    """Yes/No answer split over ``dataset``.

    Only the final label token (the Yes/No answer) is needed, so read the stored
    ``labels`` straight from the Arrow table. Iterating the dataset normally
    (``for ex in dataset``) routes through ``TextGraphDataset.__getitem__``, which
    reconstructs every graph's O(N^2) shortest-path + magnetic features just to
    look at one token — a few minutes at N~1000, ~an hour at N~2000. The column
    read is memory-mapped and builds no features.
    """
    yes_id, no_id = tokenized_labels[0][0], tokenized_labels[1][0]
    yes_count = 0
    no_count = 0
    for label in dataset._hf_dataset["labels"]:
        answer = label[-1]  # last label token is the Yes/No answer
        if answer == yes_id:
            yes_count += 1
        elif answer == no_id:
            no_count += 1
    total = yes_count + no_count
    yes_percentage = (yes_count / total) * 100 if total > 0 else 0
    no_percentage = (no_count / total) * 100 if total > 0 else 0
    return yes_percentage, no_percentage


def dataset_stored_m(ds, sample=32):
    """Magnetic truncation actually stored in ``ds``: ``0`` if the eigenvectors are
    full (``m_eff == N`` for every sampled graph), else the truncation cap ``K``
    (the max stored ``m_eff``). Only a sample of examples is read — the Arrow table
    is memory-mapped, so this is cheap even for the largest datasets.

    The stored ``m`` is not recorded in metadata; it is implicit in each example's
    ``magnetic_V`` shape ``(N, m_eff, 2)`` (see ``TextGraphDataset.__getitem__``).
    """
    n = len(ds)
    if n == 0:
        return 0
    idxs = range(n) if n <= sample else random.sample(range(n), sample)
    cap, truncated = 0, False
    for i in idxs:
        v = ds[i].get("magnetic_V")
        if v is None:
            return 0
        N, m_eff = v.shape[0], v.shape[1]
        cap = max(cap, m_eff)
        if m_eff < N:
            truncated = True
    return cap if truncated else 0


def _check_magnetic_m(ds, requested, path):
    """Raise if the dataset has fewer magnetic eigenvectors than ``requested``.

    The collator silently uses ``min(stored, requested)``, so a request larger than
    what is stored would quietly run with fewer eigenvectors than asked for (``0``
    means "all ``N``" — the largest possible). Fail fast instead.
    """
    stored = dataset_stored_m(ds)
    if stored == 0:                       # full eigenvectors -> any request is fine
        return
    if requested == 0 or requested > stored:
        want = "0 (all N)" if requested == 0 else requested
        raise ValueError(
            f"Dataset at {path} stores only m={stored} magnetic eigenvectors, but "
            f"magnetic_m={want} was requested (the collator would silently use {stored}). "
            f"Request magnetic_m <= {stored}, or delete the dataset to regenerate with a larger m.")


def load_or_create_dataset(dataset_size, easy, model_name, min_nodes, max_nodes,
                           ordering="rcm", magnetic_q=MAGNETIC_Q, magnetic_m=0):
    """Load a `.gtds` dataset, generating it (with the scalable label scheme + node
    ordering) if absent. The ordering is part of the artifact name, so an RCM
    request never silently reuses an original-order artifact; a flock around
    creation lets parallel jobs that all find it missing build it exactly once.

    The requested ``magnetic_m`` is checked against what the dataset actually stores
    (filenames do not encode it) so a too-large request errors instead of silently
    truncating — see :func:`_check_magnetic_m`."""
    path, _ = dataset_path_and_size(dataset_size, easy=easy, min_nodes=min_nodes, max_nodes=max_nodes, ordering=ordering)
    if not os.path.exists(path):
        # flock so concurrent jobs (e.g. sbatch per-config) don't race to build the
        # same artifact: the first builds it, the rest wait then find it present.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(path):
                print(f"Dataset not found at {path}. Creating new dataset...")
                create_and_save_dataset(
                    dataset_size=dataset_size, min_nodes=min_nodes, max_nodes=max_nodes,
                    model_name=model_name, easy=easy, magnetic_q=magnetic_q,
                    magnetic_m=magnetic_m, ordering=ordering,
                )
    ds = TextGraphDataset.load(path)
    if ds.node_ordering != ordering:
        raise ValueError(
            f"Dataset at {path} has node_ordering={ds.node_ordering!r} but {ordering!r} "
            f"was requested — stale artifact. Delete it to regenerate.")
    _check_magnetic_m(ds, magnetic_m, path)
    print(f"Loaded dataset from {path} with {len(ds)} examples (ordering={ds.node_ordering}).")
    return ds
