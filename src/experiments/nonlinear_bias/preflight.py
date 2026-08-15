"""Dry-run every run through the real parser + validator before submission.

    python -m src.experiments.nonlinear_bias.preflight

Why this exists: a sweep that fails on arg parsing burns a queue slot per run and
reports as N crashed jobs; worse, a sweep that *parses* but resolves to the wrong
thing (features not emitted, dataset silently rebuilt under a new cache key, a
head flag dropped between the parser and the RunConfig) runs to completion and
produces numbers that look like results. Both are cheap to catch here and
expensive to catch later — the second has already happened twice on this project:
``--magnetic-linear`` was added to graphqa's parser but not to its
``RunConfig(...)`` call, and 45 jobs silently trained the DEFAULT arm.

This arm adds two failure modes of its own, both silent:

  * **the pool knob.** ``magnetic_pool`` selects between the learned pool and the
    ablation. If it does not reach the model config, BOTH arms build the same
    module and the sweep is six duplicate cells reported as two arms — a fake
    "the ablation matches" result.
  * **d_struct.** ``magnetic_struct_dim`` is appended head width. If it does not
    reach the model, the head silently builds at its own getattr default and the
    run is mislabelled about the one cost that matters.

Adapted from ``mixed_bias/preflight.py``; the checks it shares are the ones whose
failure is invisible in a training curve.
"""

from __future__ import annotations

import argparse
import os
import sys

from sweep.execute import render_flags
from sweep.expand import load_and_expand

_CONFIGS = [
    ("src.experiments.kgqa",
     "src/experiments/nonlinear_bias/configs/032_webqsp_nonlinear.jsonc"),
    ("src.experiments.graphqa",
     "src/experiments/nonlinear_bias/configs/033_graphqa_nonlinear.jsonc"),
    ("src.experiments.graphqa",
     "src/experiments/nonlinear_bias/configs/034_graphqa_nonlinear_hot.jsonc"),
    ("src.experiments.kgqa",
     "src/experiments/nonlinear_bias/configs/035_webqsp_nonlinear_hot.jsonc"),
]

# Every placement of the magnetic term. The whole point of this file is that a
# gate enumerating a SUBSET of these is the failure mode, so the list lives in
# exactly one place and every check below derives from it.
_HEADS = ("magnetic", "magnetic_linear", "magnetic_magnitude", "magnetic_hybrid",
          "magnetic_linear_v2", "magnetic_nonlinear")

# Widths that must reach the model whenever this arm is on. Both are NOT free:
# magnetic_struct_dim is head width, magnetic_dim is the shared pair tensor.
_NONLINEAR_WIDTHS = ("magnetic_struct_dim", "magnetic_dim", "magnetic_pool")


def _load_entrypoints(module):
    if module == "src.experiments.kgqa":
        from ..kgqa.__main__ import build_parser, config_from_args
    else:
        from ..graphqa.__main__ import build_parser, config_from_args
    return build_parser, config_from_args


def _data_key(module, cfg, root):
    if module == "src.experiments.kgqa":
        return os.path.join(root, cfg.data_config_key(cfg.train_datasets[0]))
    return cfg.dataset_dir()          # graphqa's is already absolute


def _data_root(module):
    if module == "src.experiments.kgqa":
        from ..kgqa.process_dataset import OUTPUT_ROOT
        return OUTPUT_ROOT
    return None


def _expected_builds(module, runs):
    """How many distinct dataset builds this sweep should touch.

    One, except on graphqa, whose cache is keyed per (graph_type, task) — there
    the count is the number of tasks swept, and anything else means an arm forked
    the cache.
    """
    if module != "src.experiments.graphqa":
        return 1
    return len({(r.get("graph_type"), r.get("task")) for r in runs})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--strict-cache", action="store_true", default=True,
                   help="fail if a run's dataset cache is missing (default on)")
    a = p.parse_args(argv)

    failures = []
    for module, path in _CONFIGS:
        meta, runs = load_and_expand(path)
        build_parser, config_from_args = _load_entrypoints(module)
        root = _data_root(module)
        print(f"\n=== {meta['name']}  ({len(runs)} runs, {module})")

        keys, pools = set(), set()
        for i, run in enumerate(runs):
            flags = render_flags(run)
            try:
                args = build_parser().parse_args(flags)
                cfg = config_from_args(args)
            except SystemExit as e:
                failures.append(f"{meta['name']}[{i}] argparse rejected the flags: {e}")
                continue
            except ValueError as e:
                failures.append(f"{meta['name']}[{i}] config validation failed: {e}")
                continue

            bp = cfg.bias_params()
            wanted = [h for h in _HEADS if run.get(h)]
            if len(wanted) > 1:
                failures.append(
                    f"{meta['name']}[{i}] requests {wanted} — two heads on one term.")
                continue

            # (2) the arm gets EXACTLY the head it claims. Both directions matter:
            # a missing head is a bias-free run that reads as a clean negative, and
            # an extra head is two biases stacked on one feature set.
            got = [h for h in _HEADS if bp.get(h)]
            if wanted and got != wanted:
                failures.append(
                    f"{meta['name']}[{i}] {wanted[0]} arm resolved to {got or 'NO head'}: {bp}")
            if not wanted and got:
                failures.append(f"{meta['name']}[{i}] no-bias arm got a magnetic bias: {bp}")

            # (2b) the self-node arm must reach the model. The mask lives in
            # _finalize, so a dropped flag produces a run byte-identical to its
            # masked twin — two arms with the same numbers and no error. This arm
            # is only ever run unmasked, so anything else is a config slip.
            if not bp.get("bias_self_node"):
                failures.append(
                    f"{meta['name']}[{i}] bias_self_node did not reach bias_params. This "
                    "arm is a pure inner product and CANNOT express a zeroed diagonal; "
                    "running it masked prices a configuration no kernel could run.")

            # (3) THE new silent failure: the widths and the pool knob. Without
            # this, the two arms build the identical module and the sweep reports
            # six duplicate cells as an arm and its ablation.
            if wanted == ["magnetic_nonlinear"]:
                for w in _NONLINEAR_WIDTHS:
                    if bp.get(w) != getattr(cfg, w):
                        failures.append(
                            f"{meta['name']}[{i}] {w} is {getattr(cfg, w)!r} on the config "
                            f"but {bp.get(w)!r} in bias_params — the model would build the "
                            "head at a setting the record does not name.")
                pools.add(cfg.magnetic_pool)
            elif any(k in bp for k in _NONLINEAR_WIDTHS if k != "magnetic_dim"):
                failures.append(
                    f"{meta['name']}[{i}] non-nonlinear arm carries this arm's widths: {bp}")

            # (4) the silent no-op: a magnetic arm whose collator emits nothing.
            # graphqa is excluded because 0 means "keep ALL eigenvectors" there
            # (its cache is built at magnetic_m=0), the opposite of kgqa.
            zero_means_all = module == "src.experiments.graphqa"
            if wanted and cfg.collate_magnetic_m <= 0 and not zero_means_all:
                failures.append(
                    f"{meta['name']}[{i}] {wanted[0]} arm but collate_magnetic_m=0 — the "
                    "collator would emit no eigenvectors, the trunk would return None, and "
                    "the run would train with no bias while looking healthy.")
            if wanted and not cfg.uses_magnetic:
                failures.append(
                    f"{meta['name']}[{i}] {wanted[0]} is set but uses_magnetic is False — "
                    "every dataset/collator gate reads that property.")

            # (4b) the flag reached the config at all. A parser that accepts a flag
            # while config_from_args drops it yields a run that trains the DEFAULT
            # arm and reports as the requested one.
            for flag in (*_HEADS, *_NONLINEAR_WIDTHS, "magnetic_m_collate", "spd",
                         "bias_lr", "bias_self_node", "num_epochs", "seed"):
                if flag in run and getattr(cfg, flag, None) != run[flag]:
                    failures.append(
                        f"{meta['name']}[{i}] {flag}={run[flag]!r} was requested but the "
                        f"config holds {getattr(cfg, flag, None)!r} — the flag parses but "
                        f"is not forwarded by config_from_args.")

            try:
                keys.add(_data_key(module, cfg, root))
            except FileNotFoundError as e:
                failures.append(f"{meta['name']}[{i}] no built dataset: {e}")

        # (5) both pool settings must actually be present, or the ablation is
        # missing and a positive result cannot be attributed.
        if pools != {"attn", "uniform"}:
            failures.append(
                f"{meta['name']}: pool settings resolved to {sorted(pools)} — both 'attn' "
                "and 'uniform' must be present or there is no ablation.")

        # (6) the expected number of builds, each of which must already exist
        expected = _expected_builds(module, runs)
        if len(keys) != expected:
            failures.append(
                f"{meta['name']}: runs resolve to {len(keys)} different builds "
                f"({sorted(keys)}) — expected {expected}.")
        for k in keys:
            status = "present" if os.path.isdir(k) else "MISSING"
            print(f"  data build: {os.path.basename(k)}\n              -> {status}")
            if status == "MISSING" and a.strict_cache:
                failures.append(f"{meta['name']}: dataset absent at {k}")

        arms = {(r.get("magnetic_nonlinear"), r.get("magnetic_pool"),
                 r.get("bias_lr"), r.get("bias_self_node")) for r in runs}
        print(f"  arms: {len(arms)}  runs: {len(runs)}  parsed+validated OK")

    print()
    if failures:
        print(f"PREFLIGHT FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT PASSED — every run parses, validates, gets exactly its intended "
          "head at the intended widths and pool, and reuses an existing dataset cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
