"""
Dataset loading + label-balance reporting for the expressiveness experiment.
"""

import os

from ...utils import TextGraphDataset
from .data_gen import create_and_save_dataset, dataset_path_and_size


def calculate_label_distribution(dataset, tokenized_labels):
    yes_count = 0
    no_count = 0
    for example in dataset:
        label = example["labels"][-1].item()  # last label token is the Yes/No answer
        if label == tokenized_labels[0][0]:
            yes_count += 1
        elif label == tokenized_labels[1][0]:
            no_count += 1
    total = yes_count + no_count
    yes_percentage = (yes_count / total) * 100 if total > 0 else 0
    no_percentage = (no_count / total) * 100 if total > 0 else 0
    return yes_percentage, no_percentage


def load_or_create_dataset(dataset_size, easy, bias_params, model_name, min_nodes, max_nodes, spectral_dims, ordering="rcm"):
    """Load a `.gtds` dataset, generating it (with the scalable label scheme + node
    ordering) if absent. The ordering is part of the artifact name, so an RCM
    request never silently reuses an original-order artifact."""
    path, _ = dataset_path_and_size(dataset_size, easy=easy, min_nodes=min_nodes, max_nodes=max_nodes, ordering=ordering)
    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Creating new dataset...")
        create_and_save_dataset(
            dataset_size=dataset_size, min_nodes=min_nodes, max_nodes=max_nodes,
            spectral_dims=spectral_dims, model_name=model_name,
            max_rrwp_steps=bias_params["max_rw_steps"], max_rwse_steps=bias_params["max_rw_steps"],
            easy=easy, magnetic_q=bias_params["magnetic_q"], ordering=ordering,
        )
    ds = TextGraphDataset.load(path)
    if ds.node_ordering != ordering:
        raise ValueError(
            f"Dataset at {path} has node_ordering={ds.node_ordering!r} but {ordering!r} "
            f"was requested — stale artifact. Delete it to regenerate.")
    print(f"Loaded dataset from {path} with {len(ds)} examples (ordering={ds.node_ordering}).")
    return ds
