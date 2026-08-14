"""Dry-run every Phase 2 run through the real parser + validator before submission.

    python -m src.experiments.mixed_bias.preflight

Why this exists: a sweep of 87 jobs that fails on arg parsing burns a queue slot
per run and reports as 87 crashed jobs; worse, a sweep that *parses* but resolves
to the wrong thing (features not emitted, dataset silently rebuilt under a new
cache key, a head flag dropped between the parser and the RunConfig) runs to
completion and produces numbers that look like results. Both failure modes are
cheap to catch here and expensive to catch later — the second one has already
happened once on this project: --magnetic-linear and --magnetic-m-collate were
added to graphqa's parser but not to its RunConfig(...) call, so 45 submitted
jobs silently trained the DEFAULT arm.

Checks, per resolved run:
  1. the generated flags parse (``build_parser``) and validate (``config_from_args``);
  2. the arm gets EXACTLY the head it claims, and no other (``bias_params()``);
  3. the magnitude widths reach the model config, so an arm cannot silently run
     at the module's own defaults while being recorded as running at 64/256;
  4. eigenvector features are emitted whenever a magnetic term is on — the
     silent-no-op that would make an arm look like a clean negative;
  5. every run maps to the expected data cache key, and that key already exists
     on disk, so no run triggers a surprise multi-hour dataset rebuild.
"""

from __future__ import annotations

import argparse
import os
import sys

from sweep.execute import render_flags
from sweep.expand import load_and_expand

_CONFIGS = [
    ("src.experiments.kgqa", "src/experiments/mixed_bias/configs/018_webqsp_mixed.jsonc"),
    ("src.experiments.graphqa", "src/experiments/mixed_bias/configs/019_graphqa_mixed.jsonc"),
    ("src.experiments.context", "src/experiments/mixed_bias/configs/020_context4k_mixed.jsonc"),
    ("src.experiments.kgqa", "src/experiments/mixed_bias/configs/022_webqsp_magnitude_repro.jsonc"),
    ("src.experiments.kgqa", "src/experiments/mixed_bias/configs/023_webqsp_mixed_arms34.jsonc"),
    ("src.experiments.graphqa", "src/experiments/mixed_bias/configs/024_graphqa_linear_v2.jsonc"),
    ("src.experiments.kgqa", "src/experiments/mixed_bias/configs/025_webqsp_linear_v2.jsonc"),
]

# Every placement of the magnetic term. The whole point of this file is that a
# gate enumerating a SUBSET of these is the failure mode, so the list lives in
# exactly one place and every check below derives from it.
_HEADS = ("magnetic", "magnetic_linear", "magnetic_magnitude", "magnetic_hybrid",
          "magnetic_linear_v2")

# Heads that build the magnitude channel and therefore must carry its widths.
_MAGNITUDE_HEADS = ("magnetic_magnitude", "magnetic_hybrid")


def _load_entrypoints(module):
    if module == "src.experiments.kgqa":
        from ..kgqa.__main__ import build_parser, config_from_args
    elif module == "src.experiments.graphqa":
        from ..graphqa.__main__ import build_parser, config_from_args
    else:
        from ..context.__main__ import build_parser, config_from_args
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
        from ..kgqa.process_dataset import OUTPUT_ROOT
    elif module == "src.experiments.graphqa":
        return None                     # dataset_dir() is already absolute
    else:
        from ..context.process_dataset import OUTPUT_ROOT
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
            wanted = [h for h in _HEADS if run.get(h)]
            if len(wanted) > 1:
                failures.append(
                    f"{meta['name']}[{i}] requests {wanted} — two heads on one term; "
                    "the bundle is malformed.")
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

            # (3) the magnitude widths reach the model. Without this an arm could
            # run at the bias module's own getattr defaults while the record says
            # 64/256 — a mislabelled run, which is worse than a crashed one.
            if wanted and wanted[0] in _MAGNITUDE_HEADS:
                for w in ("magnetic_magnitude_dim", "magnetic_magnitude_repr_dim"):
                    if bp.get(w) != getattr(cfg, w):
                        failures.append(
                            f"{meta['name']}[{i}] {w} is {getattr(cfg, w)!r} on the config "
                            f"but {bp.get(w)!r} in bias_params — the model would build the "
                            "magnitude channel at a width the record does not name.")
            elif any(k.startswith("magnetic_magnitude_") for k in bp):
                failures.append(
                    f"{meta['name']}[{i}] non-magnitude arm carries magnitude widths: {bp}")

            # (4) THE silent no-op: a magnetic arm whose collator emits nothing.
            # graphqa is excluded because 0 means "keep ALL eigenvectors" there
            # (its cache is built at magnetic_m=0), the opposite of kgqa/context
            # where 0 means "emit none".
            zero_means_all = module == "src.experiments.graphqa"
            if wanted and cfg.collate_magnetic_m <= 0 and not zero_means_all:
                failures.append(
                    f"{meta['name']}[{i}] {wanted[0]} arm but collate_magnetic_m=0 — the "
                    "collator would emit no eigenvectors and the bias would return None, "
                    "producing a bias-free run that looks like a clean negative.")
            if not wanted and cfg.collate_magnetic_m:
                failures.append(f"{meta['name']}[{i}] no-bias arm still requests eigenvectors")
            if wanted and not cfg.uses_magnetic:
                failures.append(
                    f"{meta['name']}[{i}] {wanted[0]} is set but uses_magnetic is False — "
                    "every dataset/collator gate reads that property, so this arm would "
                    "train with no graph bias at all.")

            # (4b) the flag reached the config at all. A parser that accepts
            # --magnetic-hybrid while config_from_args drops it yields a run that
            # trains the DEFAULT arm and reports as the requested one.
            for flag in (*_HEADS, "magnetic_m_collate", "spd", "magnetic_dim", "bias_lr",
                         "bias_self_node", "magnetic_magnitude_dim",
                         "magnetic_magnitude_repr_dim", "num_epochs"):
                if flag in run and getattr(cfg, flag, None) != run[flag]:
                    failures.append(
                        f"{meta['name']}[{i}] {flag}={run[flag]!r} was requested but the "
                        f"config holds {getattr(cfg, flag, None)!r} — the flag parses but "
                        f"is not forwarded by config_from_args.")

            try:
                keys.add(_data_key(module, cfg, root))
            except FileNotFoundError as e:
                failures.append(f"{meta['name']}[{i}] no built dataset: {e}")

        # (5) the expected number of builds, each of which must already exist
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

        arms = {tuple(r.get(h) for h in _HEADS) + (r.get("magnetic_m_collate"),
                                                   r.get("bias_self_node"),
                                                   r.get("bias_lr")) for r in runs}
        print(f"  arms: {len(arms)}  runs: {len(runs)}  parsed+validated OK")

    print()
    if failures:
        print(f"PREFLIGHT FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT PASSED — every run parses, validates, gets exactly its intended "
          "head at the intended widths, and reuses an existing dataset cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
