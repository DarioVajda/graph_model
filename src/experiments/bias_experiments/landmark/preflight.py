"""Dry-run every landmark sweep run through the real parser + validator.

    python -m src.experiments.bias_experiments.landmark.preflight

Why this exists (same reasoning as `linear_bias/preflight.py`): a sweep that
*parses* but resolves to the wrong thing runs to completion and produces numbers
that look like results. For this experiment the specific catastrophe is a
landmark arm whose `landmark` column is missing or whose flag does not reach the
model — that trains cleanly, scores like the floor, and reads as "the landmark
bias does not work", which is precisely the conclusion the sweep exists to test.

Checks, per resolved run:
  1. the generated flags parse (`build_parser`) and validate (`config_from_args`);
  2. the arm gets the bias it claims, and ONLY that bias (`bias_params()`);
  3. `bias_self_node` reaches the model on every biased arm — the mask lives in
     `_finalize`, so a dropped flag makes an arm byte-identical to its masked twin;
  4. the dimension bookkeeping is right: 2M for magnetic_linear, 3k for landmark,
     and the matched cells really do match;
  5. every run maps to ONE data cache key, it exists on disk, and — for landmark
     arms — that cache actually carries the `landmark` column at the k the run
     slices to.
"""

from __future__ import annotations

import argparse
import json
import os

from sweep.execute import render_flags
from sweep.expand import load_and_expand

from ...kgqa.__main__ import build_parser, config_from_args
from ...kgqa.process_dataset import OUTPUT_ROOT
from ...graphqa.__main__ import build_parser as gq_build_parser
from ...graphqa.__main__ import config_from_args as gq_config_from_args

_HERE = os.path.dirname(__file__)
_CONFIGS = [
    os.path.join(_HERE, "configs", "039_webqsp_smoke.jsonc"),
    os.path.join(_HERE, "configs", "040_webqsp_dimsweep.jsonc"),
    os.path.join(_HERE, "configs", "041_webqsp_floor.jsonc"),
    os.path.join(_HERE, "configs", "042_webqsp_landmark_norm.jsonc"),
    os.path.join(_HERE, "configs", "043_webqsp_landmark_lowlr.jsonc"),
    os.path.join(_HERE, "configs", "044_webqsp_magnetic_lowlr.jsonc"),
    os.path.join(_HERE, "configs", "045_graphqa_landmark.jsonc"),
]


# The two experiments this directory sweeps differ in parser, cache layout, split
# names and matched-dimension grid — but in nothing else this file checks, so they
# are two rows of a table rather than two code paths.
#
# The GRID differs because the datasets do: WebQSP averages ~500 nodes and matches
# magnetic_linear's 2M against landmark's 3k at {24,48,96}; GraphQA averages 12.9
# nodes, where 96 anchor dims on a 13-node graph is meaningless, so it runs
# {12,24,48}.
_EXPERIMENTS = {
    "kgqa": dict(
        build_parser=build_parser, config_from_args=config_from_args,
        splits=("train", "dev", "test"),
        cache=lambda cfg: os.path.join(OUTPUT_ROOT,
                                       cfg.data_config_key(cfg.train_datasets[0])),
        dims_grid={24, 48, 96},
        # One build serves the whole sweep.
        caches_expected=lambda runs: 1,
    ),
    "graphqa": dict(
        build_parser=gq_build_parser, config_from_args=gq_config_from_args,
        splits=("train", "validation", "test"),
        cache=lambda cfg: cfg.dataset_dir(),
        dims_grid={12, 24, 48},
        # One build PER TASK, so the count is the task axis's width — checking for
        # a single key here would fail a correct 3-task sweep.
        caches_expected=lambda runs: len({r.get("task") for r in runs}),
    ),
}


def _experiment_of(runs) -> str:
    """Which experiment a sweep config targets, read off the runs themselves.

    `task` is GraphQA's required axis and `dataset` is KGQA's, so this is derived
    from the config rather than from a filename convention that could drift.
    """
    has_task = any("task" in r for r in runs)
    has_dataset = any("dataset" in r for r in runs)
    if has_task and not has_dataset:
        return "graphqa"
    if has_dataset and not has_task:
        return "kgqa"
    raise SystemExit(
        f"cannot tell which experiment this config targets "
        f"(task={has_task}, dataset={has_dataset}) — preflight would check it "
        f"against the wrong parser and pass for the wrong reasons.")


def _landmark_meta(cache_dir: str, split: str = "train"):
    p = os.path.join(cache_dir, f"{split}.gtds", "metadata.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=_CONFIGS)
    a = ap.parse_args(argv)

    failures, checked = [], 0
    for path in a.configs:
        meta, runs = load_and_expand(path)
        exp = _EXPERIMENTS[_experiment_of(runs)]
        print(f"\n=== {meta['name']}  ({len(runs)} runs, {_experiment_of(runs)})")
        keys = set()

        for i, run in enumerate(runs):
            flags = render_flags(run)
            try:
                cfg = exp["config_from_args"](exp["build_parser"]().parse_args(flags))
            except SystemExit as e:
                failures.append(f"{meta['name']}[{i}] argparse rejected flags: {e}")
                continue
            except ValueError as e:
                failures.append(f"{meta['name']}[{i}] validation failed: {e}")
                continue
            checked += 1

            bp = cfg.bias_params()
            want_lm = bool(run.get("landmark"))
            want_lin = bool(run.get("magnetic_linear"))

            # (2) exactly the claimed bias, nothing else
            if want_lm and not bp.get("landmark"):
                failures.append(f"{meta['name']}[{i}] landmark arm has no landmark bias: {bp}")
            if want_lin and not bp.get("magnetic_linear"):
                failures.append(f"{meta['name']}[{i}] magnetic arm has no linear bias: {bp}")
            if not want_lm and bp.get("landmark"):
                failures.append(f"{meta['name']}[{i}] non-landmark arm got landmark: {bp}")
            if not want_lin and bp.get("magnetic_linear"):
                failures.append(f"{meta['name']}[{i}] non-magnetic arm got magnetic_linear: {bp}")

            # (3) the diagonal flag must reach the model on every biased arm
            want_self = bool(run.get("bias_self_node"))
            if want_self != bool(bp.get("bias_self_node")):
                failures.append(
                    f"{meta['name']}[{i}] bias_self_node={want_self} did not survive "
                    f"to bias_params: {bp}")
            if (want_lm or want_lin) and not want_self:
                failures.append(
                    f"{meta['name']}[{i}] a factorized arm is running MASKED; neither "
                    "bias can express the diagonal, so unmasked is the only "
                    "configuration the deferred backbone can run.")

            if want_lm and meta["name"][:3] in ("042", "043") and not cfg.landmark_norm:
                failures.append(
                    f"{meta['name']}[{i}] landmark_norm is OFF in the sweep whose "
                    "entire purpose is the normalization — this would silently "
                    "reproduce 040.")

            # 043 and 044 exist to sample BELOW the {5e-3, 2e-2} bracket they
            # inherited. Leaving a bracket value in place would reproduce the old
            # sweep under a new name and read as "lowering the LR changed nothing".
            # 044 is the fairness control for 043, so it must clear the same bar —
            # a 044 that quietly ran at the old LR would make magnetic look
            # untuned relative to landmark, which is the exact asymmetry it exists
            # to remove.
            if meta["name"][:3] in ("043", "044") and cfg.bias_lr >= 5e-3:
                failures.append(
                    f"{meta['name']}[{i}] bias_lr={cfg.bias_lr:g} is not below 042's "
                    "bracket; this sweep's whole purpose is the sub-bracket range.")

            # (4) dimension bookkeeping
            if want_lm:
                k = cfg.landmark_k_collate or cfg.landmark_k
                dims = 3 * k
                if cfg.landmark_channels != 3:
                    failures.append(f"{meta['name']}[{i}] channels={cfg.landmark_channels}")
            elif want_lin:
                dims = 2 * (cfg.magnetic_m_collate or cfg.magnetic_m)
            else:
                dims = 0
            if (want_lm or want_lin) and dims not in exp["dims_grid"]:
                failures.append(
                    f"{meta['name']}[{i}] struct dims={dims} is off the matched grid "
                    f"{sorted(exp['dims_grid'])} — the two curves would not share "
                    "an x-axis.")

            # (5) one cache, present, and carrying what a landmark arm needs
            cache = exp["cache"](cfg)
            keys.add(cache)
            if not os.path.isdir(cache):
                failures.append(f"{meta['name']}[{i}] cache missing (would rebuild): {cache}")
            elif want_lm:
                for split in exp["splits"]:
                    md = _landmark_meta(cache, split)
                    if not md or "landmark_k" not in md:
                        failures.append(
                            f"{meta['name']}[{i}] {split}: cache has no landmark column. "
                            "Run add_landmark_column.py — otherwise this trains with NO "
                            "bias and reads as a clean negative.")
                    elif md["landmark_k"] < (cfg.landmark_k_collate or cfg.landmark_k):
                        failures.append(
                            f"{meta['name']}[{i}] {split}: stored landmark_k="
                            f"{md['landmark_k']} < requested "
                            f"{cfg.landmark_k_collate or cfg.landmark_k}")
                    elif md.get("landmark_d_max") != cfg.landmark_d_max:
                        failures.append(
                            f"{meta['name']}[{i}] {split}: stored d_max="
                            f"{md.get('landmark_d_max')} != cfg {cfg.landmark_d_max}; "
                            "the symbol alphabet would be misread.")

        want_keys = exp["caches_expected"](runs)
        if len(keys) != want_keys:
            failures.append(
                f"{meta['name']} resolves to {len(keys)} cache dirs, expected "
                f"{want_keys}: {sorted(keys)}")
        else:
            for k in sorted(keys):
                print(f"    cache: {os.path.basename(k.rstrip('/'))}")

    print(f"\nchecked {checked} resolved runs")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in dict.fromkeys(failures):
            print(f"  - {f}")
        raise SystemExit(1)
    print("preflight OK — safe to submit")


if __name__ == "__main__":
    main()
