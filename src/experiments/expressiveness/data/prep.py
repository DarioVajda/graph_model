"""
``data_prep`` mode — build the train + eval datasets for one config.

Dataset preparation is the experiment's own concern (the sweep runner knows
nothing about datasets). Routed to from ``__main__`` when ``--mode data_prep``;
for a multi-size sweep, run the sweep once in ``data_prep`` mode (each config
builds its own datasets; the flock in ``load_or_create_dataset`` makes parallel
jobs build each artifact exactly once) before running it again in ``train`` mode.

Large N is host-RAM heavy (the N×N SPD feature), so prep these off the GPU —
e.g. an sbatch job with ``--mem=224G`` for the N=2000 datasets.
"""

from .datasets import load_or_create_dataset


def run_data_prep_mode(cfg):
    """Ensure this config's train + eval `.gtds` exist (build if missing)."""
    for which, size in (("train", cfg.train_dataset_size), ("eval", cfg.eval_dataset_size)):
        print(f"[data_prep] {which}: size={size} n{cfg.min_nodes}-{cfg.max_nodes} "
              f"m={cfg.magnetic_m} ordering={cfg.ordering}")
        load_or_create_dataset(
            size, cfg.is_easy, cfg.model_name, cfg.min_nodes, cfg.max_nodes,
            ordering=cfg.ordering, magnetic_q=cfg.magnetic_q, magnetic_m=cfg.magnetic_m)
    print("[data_prep] done.")
