"""Dry-run every Phase 2 run through the real parser + validator before submission.

    python -m src.experiments.bias_experiments.linear_bias.preflight

Why this exists: a sweep of 42 jobs that fails on arg parsing burns a queue slot
per run and reports as 42 crashed jobs; worse, a sweep that *parses* but resolves
to the wrong thing (features not emitted, dataset silently rebuilt under a new
cache key) runs to completion and produces numbers that look like results. Both
failure modes are cheap to catch here and expensive to catch later.

Checks, per resolved run:
  1. the generated flags parse (``build_parser``) and validate (``config_from_args``);
  2. the arm actually gets the bias it claims (``bias_params()``);
  3. eigenvector features are emitted whenever a magnetic term is on — the
     silent-no-op that would make an arm look like a clean negative;
  4. every run maps to the SAME data cache key, and that key already exists on
     disk, so no run triggers a surprise multi-hour dataset rebuild.
"""

from __future__ import annotations

import argparse
import os
import sys

from sweep.execute import render_flags
from sweep.expand import load_and_expand

_CONFIGS = [
    ("src.experiments.kgqa", "src/experiments/bias_experiments/linear_bias/configs/010_webqsp_linear.jsonc"),
    ("src.experiments.context", "src/experiments/bias_experiments/linear_bias/configs/011_context4k_linear.jsonc"),
    ("src.experiments.context", "src/experiments/bias_experiments/linear_bias/configs/012_context4k_linear_long.jsonc"),
    ("src.experiments.graphqa", "src/experiments/bias_experiments/linear_bias/configs/013_graphqa_linear.jsonc"),
    ("src.experiments.kgqa", "src/experiments/bias_experiments/linear_bias/configs/014_webqsp_magdim256.jsonc"),
    ("src.experiments.kgqa", "src/experiments/bias_experiments/linear_bias/configs/015_webqsp_selfnode.jsonc"),
    ("src.experiments.context", "src/experiments/bias_experiments/linear_bias/configs/016_context4k_selfnode.jsonc"),
    ("src.experiments.graphqa", "src/experiments/bias_experiments/linear_bias/configs/017_graphqa_selfnode.jsonc"),
]


def _load_entrypoints(module):
    if module == "src.experiments.kgqa":
        from ...kgqa.__main__ import build_parser, config_from_args
    elif module == "src.experiments.graphqa":
        from ...graphqa.__main__ import build_parser, config_from_args
    else:
        from ...context.__main__ import build_parser, config_from_args
    return build_parser, config_from_args


def _data_key(module, cfg, root):
    """The build this run will actually READ.

    For context that is ``resolved_data_root`` (which accepts a superset build),
    not the raw key — the arms that drop SPD resolve to the richer existing build
    rather than demanding a rebuild of a strict subset.
    """
    if module == "src.experiments.kgqa":
        return os.path.join(root, cfg.data_config_key(cfg.train_datasets[0]))
    if module == "src.experiments.graphqa":
        # graphqa caches per (graph_type, task), so a multi-task sweep legitimately
        # reads several builds — see _expected_builds.
        return cfg.dataset_dir()
    return cfg.resolved_data_root(root)


def _data_root(module):
    if module == "src.experiments.kgqa":
        from ...kgqa.process_dataset import OUTPUT_ROOT
    elif module == "src.experiments.graphqa":
        return None                     # dataset_dir() is already absolute
    else:
        from ...context.process_dataset import OUTPUT_ROOT
    return OUTPUT_ROOT


def _expected_builds(module, runs):
    """How many distinct dataset builds this sweep should touch.

    One, everywhere except graphqa, whose cache is keyed per task — there the
    count is the number of tasks swept, and anything else means an arm forked the
    cache.
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

        keys = set()
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
            want_mag = bool(run.get("magnetic"))
            want_lin = bool(run.get("magnetic_linear"))

            # (2) the arm gets the bias it claims
            if want_lin and not bp.get("magnetic_linear"):
                failures.append(f"{meta['name']}[{i}] magnetic_linear arm has no linear bias: {bp}")
            if want_mag and not bp.get("magnetic"):
                failures.append(f"{meta['name']}[{i}] magnetic arm has no magnetic bias: {bp}")
            if not (want_mag or want_lin) and (bp.get("magnetic") or bp.get("magnetic_linear")):
                failures.append(f"{meta['name']}[{i}] no-bias arm got a magnetic bias: {bp}")

            # (2b) the self-node arm must actually reach the model. The mask lives
            # in _finalize, so a dropped flag produces a run that is byte-identical
            # to its masked twin — two arms with the same numbers and no error.
            want_self = bool(run.get("bias_self_node"))
            if want_self and not bp.get("bias_self_node"):
                failures.append(
                    f"{meta['name']}[{i}] bias_self_node arm did not reach bias_params: {bp}")
            if not want_self and bp.get("bias_self_node"):
                failures.append(
                    f"{meta['name']}[{i}] masked arm has bias_self_node set: {bp}")

            # (3) THE silent no-op: a magnetic arm whose collator emits nothing.
            # graphqa is excluded because 0 means "keep ALL eigenvectors" there
            # (its cache is built at magnetic_m=0), the opposite of kgqa/context
            # where 0 means "emit none".
            zero_means_all = module == "src.experiments.graphqa"
            if (want_mag or want_lin) and cfg.collate_magnetic_m <= 0 and not zero_means_all:
                failures.append(
                    f"{meta['name']}[{i}] magnetic arm but collate_magnetic_m=0 — the "
                    "collator would emit no eigenvectors and the bias would return None, "
                    "producing a bias-free run that looks like a clean negative.")
            if not (want_mag or want_lin) and cfg.collate_magnetic_m:
                failures.append(f"{meta['name']}[{i}] no-bias arm still requests eigenvectors")

            # (3b) the flag reached the config at all. A parser that accepts
            # --magnetic-linear while config_from_args drops it yields a run that
            # trains the DEFAULT arm and reports as the requested one.
            for flag in ("magnetic_linear", "magnetic_m_collate", "magnetic", "spd",
                         "magnetic_dim", "bias_lr", "bias_self_node"):
                if flag in run and getattr(cfg, flag, None) != run[flag]:
                    failures.append(
                        f"{meta['name']}[{i}] {flag}={run[flag]!r} was requested but the "
                        f"config holds {getattr(cfg, flag, None)!r} — the flag parses but "
                        f"is not forwarded by config_from_args.")

            try:
                keys.add(_data_key(module, cfg, root))
            except FileNotFoundError as e:
                failures.append(f"{meta['name']}[{i}] no built dataset: {e}")

        # (4) the expected number of builds, each of which must already exist
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

        arms = {(r.get("spd"), r.get("magnetic"), r.get("magnetic_linear"),
                 r.get("magnetic_m_collate"), r.get("bias_self_node")) for r in runs}
        print(f"  arms: {len(arms)}  runs: {len(runs)}  parsed+validated OK")

    print()
    if failures:
        print(f"PREFLIGHT FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT PASSED — every run parses, validates, gets its intended bias, "
          "and reuses one existing dataset cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
