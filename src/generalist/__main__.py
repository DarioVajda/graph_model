"""
D8.1 — the command line: ``validate``, ``data_prep``, ``train``, ``resume``,
``fork``, ``eval``.

    python3 -m src.generalist validate  --config src/generalist/configs/001_molecule_generalist.jsonc
    python3 -m src.generalist data_prep --config <cfg>
    python3 -m src.generalist train     --config <cfg>
    python3 -m src.generalist resume    --from latest --config <cfg>
    python3 -m src.generalist fork      --from <ckpt> --mode anneal --config <cfg>
    python3 -m src.generalist eval      --checkpoint <ckpt> --config <cfg>
    python3 -m src.generalist --init my_run          # write a config under configs/

This file is a dispatcher and nothing else: every mode resolves a
:class:`~src.generalist.config.RunConfig`, hands it to `wiring.py`, and prints.
Anything that decides something belongs in one of those two.

**The subcommand is optional and defaults to ``train``.** The sweep runner
invokes an experiment as ``python -m <module> --key value …`` with no
subcommand (`sweep/README.md`), so an argv whose first token is a flag is read
as a ``train``. That is what makes ``python3 -m sweep src.generalist <cfg>``
work against the same program that ``chain.sh`` calls with an explicit mode.

**Flags are generated from the dataclass.** One flag per ``RunConfig`` field,
built by :func:`add_config_flags`, so a field added there is swept, chained and
overridable without being wired up in three places. Every one defaults to
``None`` and only the ones actually passed override the config file — with
argparse's usual ``default=<the dataclass default>`` a flag nobody typed would
silently overwrite the file.

``validate`` runs on a login node: it resolves the config, builds the registry,
checks the partition and prints the mixture table and the step budget, without
importing torch or transformers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields

from .config import (
    CONFIGS_DIR,
    ConfigError,
    RunConfig,
    load_config_file,
    shell_assignments,
    write_template,
)
from . import wiring

MODES = ("validate", "data_prep", "train", "resume", "fork", "eval")

#: Fields no flag is generated for. ``selection`` is a dict (D7.4 refuses it on a
#: training run anyway, so there is nothing to type); the rest are paths and
#: bookkeeping that the runner passes under its own names.
NO_FLAG_FIELDS = ("selection",)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_config_flags(parser: argparse.ArgumentParser) -> None:
    """One flag per ``RunConfig`` field, all defaulting to ``None``."""
    group = parser.add_argument_group(
        "run config", "overrides the --config file; see src/generalist/config.py")
    defaults = RunConfig()
    for spec in fields(RunConfig):
        if spec.name in NO_FLAG_FIELDS:
            continue
        current = getattr(defaults, spec.name)
        note = f"default {current!r}"
        if spec.type in ("bool", bool):
            group.add_argument(_flag(spec.name), dest=spec.name, default=None,
                               action=argparse.BooleanOptionalAction, help=note)
        elif spec.type in ("int", int):
            group.add_argument(_flag(spec.name), dest=spec.name, default=None,
                               type=int, help=note)
        elif spec.type in ("float", float):
            group.add_argument(_flag(spec.name), dest=spec.name, default=None,
                               type=float, help=note)
        else:
            group.add_argument(_flag(spec.name), dest=spec.name, default=None,
                               help=note)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="a .jsonc run config (the same file the sweep runner takes).")
    parser.add_argument("--runs-jsonl", default=None,
                        help="(runner) JSONL to append this run's record to.")
    parser.add_argument("--run-id", default=None,
                        help="(runner) this run's name within the sweep.")
    parser.add_argument("--sweep-id", default=None,
                        help="(runner) the sweep this run belongs to.")
    add_config_flags(parser)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.generalist",
        description="The generalist harness (DESIGN.md D8). The mode defaults to "
                    "'train' when the first argument is a flag.")
    p.add_argument("--init", nargs="?", const="generalist", default=None,
                   metavar="NAME",
                   help="write a run/sweep config to configs/<NAME>.jsonc and exit.")
    subs = p.add_subparsers(dest="mode", metavar="MODE")

    validate = subs.add_parser(
        "validate", help="resolve the config, check the partition, print the "
                         "mixture table and the step budget. No GPU.")
    add_common(validate)
    validate.add_argument("--json", action="store_true",
                          help="also print the resolved config as JSON.")
    validate.add_argument("--print-shell", action="store_true",
                          help="print only the chain script's GEN_* shell "
                               "assignments and exit (no registry, no adapter).")

    prep = subs.add_parser("data_prep", help="build every split of the config's tasks.")
    add_common(prep)
    prep.add_argument("--arms", default=None,
                      help="comma-joined arms to build (default: the config's arm).")
    prep.add_argument("--only", default=None,
                      help="comma-joined task names to build (default: the "
                           "mixture's tasks plus the held-out ones).")
    prep.add_argument("--rebuild", action="store_true",
                      help="rebuild artifacts that already exist.")

    train = subs.add_parser("train", help="a fresh run: [warmup, stable] for the budget.")
    add_common(train)

    resume = subs.add_parser("resume", help="continue a run from a checkpoint (D5.4).")
    add_common(resume)
    resume.add_argument("--from", dest="from_", required=True, metavar="CKPT",
                        help="a checkpoint directory, or 'latest' for the newest "
                             "complete one under the run's output dir.")

    fork_p = subs.add_parser("fork", help="branch a checkpoint (D6).")
    add_common(fork_p)
    fork_p.add_argument("--from", dest="from_", required=True, metavar="CKPT",
                        help="the parent checkpoint to fork from.")
    fork_p.add_argument("--mode", dest="fork_mode", required=True,
                        choices=("anneal", "admit", "adapt"),
                        help="which fork this is.")
    fork_p.add_argument("--fork-config", default=None, metavar="PATH",
                        help="the fork's own .jsonc (criterion, target, "
                             "decay_steps, …).")
    fork_p.add_argument("--decay-steps", type=int, default=None,
                        help="anneal: length of the decay segment (default 10%% "
                             "of the parent's steps).")
    fork_p.add_argument("--run-dir", default=None,
                        help="where the child run lands (default: beside the parent).")
    fork_p.add_argument("--dry-run", action="store_true",
                        help="plan the fork and print it; write nothing, train nothing.")

    ev = subs.add_parser("eval", help="run validators on a checkpoint. No training.")
    add_common(ev)
    ev.add_argument("--checkpoint", required=True,
                    help="the checkpoint directory to score.")
    # Not `--validators`: that flag is generated from the RunConfig field of the
    # same name, which names the *set*. This one narrows that set to a few of its
    # members for one scoring pass, and the two are not the same question.
    ev.add_argument("--only-validators", dest="only_validators", default=None,
                    help="comma-joined validator names to run (default: every "
                         "validator the config's set contains).")
    ev.add_argument("--out", default=None,
                    help="where the eval record is written (default: inside the "
                         "checkpoint).")
    return p


def normalise_argv(argv) -> list:
    """Insert the default subcommand when argv starts with a flag."""
    argv = list(argv)
    if not argv:
        return argv
    if argv[0] in MODES or argv[0] in ("-h", "--help") or argv[0].startswith("--init"):
        return argv
    if argv[0].startswith("-"):
        return ["train"] + argv
    return argv


def config_from_args(args) -> RunConfig:
    """``RunConfig`` defaults, then the ``--config`` file, then explicit flags."""
    values = {}
    if getattr(args, "config", None):
        values.update(load_config_file(args.config))
    for spec in fields(RunConfig):
        given = getattr(args, spec.name, None)
        if given is not None:
            values[spec.name] = given
    # The sweep runner names the run; keeping its name is what makes a sweep's
    # runs.jsonl line and this run's output directory refer to the same thing.
    if getattr(args, "run_id", None):
        values["run_name"] = args.run_id
    config = RunConfig(**values)
    return config.validate()


def _names(raw):
    return tuple(s.strip() for s in (raw or "").split(",") if s.strip())


# ─────────────────────────────────────────────────────────────────────────────
# validate
# ─────────────────────────────────────────────────────────────────────────────

def _print_shares(config: RunConfig) -> None:
    """The mixture table when the budget is not resolvable yet.

    Shares need only the config; the budget and the step count additionally need
    ``mean_tokens`` and ``train_size``, which are properties of the *built* data
    (D2) and do not exist before ``data_prep``. Printing what is known and saying
    plainly what is not beats either inventing a token length or refusing to
    check a config until a 40-minute build has run.
    """
    entries = config.mixture_entries()
    total = sum(float(e["weight"]) for e in entries)
    print("  task                       share    passes")
    print("  " + "-" * 44)
    for entry in sorted(entries, key=lambda e: -float(e["weight"])):
        print(f"  {entry['name']:<24} {float(entry['weight']) / total:7.4f} "
              f"{int(entry.get('passes', 1)):>8d}")


def _print_partition(config: RunConfig, adapter_config) -> None:
    from .adapters._partition import Partition

    path = adapter_config.partition_path()
    if not os.path.exists(path):
        print(f"partition: not built yet ({path}); data_prep builds it first.")
        return
    part = Partition.load(path)
    print(f"partition: {path}")
    for line in part.summary().splitlines():
        print("  " + line)


def mode_validate(config: RunConfig, args) -> int:
    if args.print_shell:
        # The chain script's path: the config is already validated, and building
        # the registry here would put RDKit and the raw CSVs between a submission
        # and the queue for no gain.
        print(shell_assignments(config))
        return 0

    registry, adapter_config = wiring.build_registry(config)

    print(f"run           {config.run_name}  ({config.arm} arm)")
    print(f"output_dir    {config.run_dir()}")
    print(f"config_hash   {config.config_hash()}")
    print(f"build_version {adapter_config.build_version()}")
    print(f"registry      {len(registry)} tasks, hash {registry.hash()[:16]}")
    print()
    _print_partition(config, adapter_config)
    print()

    missing = wiring.unbuilt_tasks(registry, config)
    if missing:
        print("mixture (shares only — these tasks have no build manifest, so the "
              "example budget and the step count cannot be computed yet):")
        _print_shares(config)
        print(f"  unbuilt: {', '.join(missing)}")
        print("  run data_prep, then validate again for the budget.")
    else:
        mixture = wiring.resolve_mixture(config, registry)
        print("mixture:")
        print(mixture.table())
        print(f"  mixture_hash {mixture.hash()[:16]}")
        passes = wiring.passes_needed(mixture, registry)
        needed = wiring.generator_passes(config, mixture, registry)
        print(f"  generator passes to build: {needed} "
              f"({', '.join(f'{k}={v}' for k, v in sorted(passes.items()))})")
    print()

    from .evaluate import build_validators
    print("validators:")
    for validator in build_validators(config.validator_specs()):
        print(f"  {validator.name:<14} {validator.cadence:<14} "
              f"v{validator.protocol_version}  {dict(validator.options)}")

    if args.json:
        print()
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True, default=str))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# data_prep
# ─────────────────────────────────────────────────────────────────────────────

def mode_data_prep(config: RunConfig, args) -> int:
    from .adapters import molecules
    from .registry import MOLECULE_PREFIX, is_held_out

    registry, adapter_config = wiring.build_registry(config)
    arms = _names(args.arms) or (config.arm,)

    if args.only:
        names = _names(args.only)
    else:
        names = [e["name"] for e in config.mixture_entries()]
        names += [spec.name for spec in registry if is_held_out(spec)]
    bare = tuple(dict.fromkeys(
        n[len(MOLECULE_PREFIX):] if n.startswith(MOLECULE_PREFIX) else n
        for n in names))

    print(f"data_prep: {len(bare)} tasks x {len(arms)} arms -> "
          f"{adapter_config.build_dir()}")
    part = molecules.partition(adapter_config)
    print(part.summary())

    # Two rounds, because the two quantities depend on each other: the number of
    # generator passes a run consumes comes from the resolved mixture, and the
    # mixture cannot resolve until a build has measured `mean_tokens`. The first
    # round makes the resolve possible, the second builds the passes it asks for.
    # Generators never bound the budget (D4.2), so the second resolve would give
    # the same answer and a third round is never needed.
    passes = wiring.generator_passes(config)
    molecules.build(adapter_config, roles=part, tasks=bare, arms=tuple(arms),
                    passes=passes, rebuild=args.rebuild)

    registry, _ = wiring.build_registry(config, adapter_config)
    if not wiring.unbuilt_tasks(registry, config):
        mixture = wiring.resolve_mixture(config, registry)
        needed = wiring.generator_passes(config, mixture, registry)
        if needed > passes:
            print(f"data_prep: the mixture consumes {needed} generator passes; "
                  f"building the {needed - passes} missing ones")
            molecules.build(adapter_config, roles=part, tasks=bare,
                            arms=tuple(arms), passes=needed,
                            rebuild=args.rebuild)
        print()
        print(wiring.resolve_mixture(config, registry).table())
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# train / resume
# ─────────────────────────────────────────────────────────────────────────────

def _record(config: RunConfig, args, payload: dict) -> None:
    """One line in ``runs.jsonl``, and one ``run.json`` in the run directory."""
    from .lineage import append_line

    payload = dict(payload)
    payload.setdefault("run_name", config.run_name)
    payload.setdefault("config_hash", config.config_hash())
    if getattr(args, "sweep_id", None):
        payload.setdefault("sweep_id", args.sweep_id)
    path = getattr(args, "runs_jsonl", None) or config.runs_jsonl()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    append_line(path, json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 default=str))
    with open(os.path.join(config.run_dir(), "run.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def _write_log_history(config: RunConfig, run) -> None:
    """Persist ``log_history`` once training returns, ``end`` metrics included.

    Every other cadence survives on its own: a ``steps:`` or ``milestone``
    firing is logged during training and the next checkpoint's
    ``trainer_state.json`` carries it. The ``end`` event has no such carrier.
    It fires from ``on_train_end``, which HF runs *after* the last checkpoint is
    written and after the progress bar has been closed, so the metrics go into
    ``state.log_history``, are never printed, and are never saved — the 200-step
    smoke computed ``perm_spread`` and ``per_example`` in full and left no trace
    of either. D7.1 says a validator's metrics are written to the run record;
    this is the file that makes that true for the ones that only ever fire at
    the end.

    Written whole rather than appended to: on a resume HF restores the parent
    checkpoint's history first, so the last chunk's copy is the run's history,
    not that chunk's.
    """
    history = list(getattr(run.trainer.state, "log_history", ()) or ())
    if not history:
        return
    path = os.path.join(config.run_dir(), "log_history.json")
    with open(path, "w") as fh:
        json.dump(history, fh, indent=2, sort_keys=True, default=str)
    print(f"train: wrote {len(history)} log entries to {path}")


def mode_train(config: RunConfig, args) -> int:
    from . import checkpoint as ckpt_mod

    run_dir = config.run_dir()
    existing = ckpt_mod.latest(run_dir) if os.path.isdir(run_dir) else None
    if existing is not None:
        raise SystemExit(
            f"train: {run_dir} already holds a complete checkpoint ({existing}). "
            "A fresh train would restart the schedule and the sampler from zero "
            "beside it; use `resume --from latest` to continue it, or point "
            "--output-dir somewhere else.")

    run = wiring.build_run(config)
    os.makedirs(run_dir, exist_ok=True)
    print(run.mixture.table())
    _record(config, args, {"mode": "train", "run": run_dir,
                           "steps": run.max_steps,
                           "mixture_hash": run.mixture.hash(),
                           "budget_examples": run.mixture.budget_examples})
    output = run.trainer.train()
    _write_log_history(config, run)
    print(f"train: finished at step {run.trainer.state.global_step}; "
          f"{output.metrics if output is not None else ''}")
    return 0


def mode_resume(config: RunConfig, args) -> int:
    run = wiring.build_run(config)
    print(f"resume: {config.run_dir()} from {args.from_}")
    output = run.trainer.resume(args.from_)
    _write_log_history(config, run)
    print(f"resume: finished at step {run.trainer.state.global_step}; "
          f"{output.metrics if output is not None else ''}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# fork
# ─────────────────────────────────────────────────────────────────────────────

def load_fork_config(path, args, config: RunConfig) -> dict:
    """The fork's own config, with the CLI's overrides and the run's defaults.

    ``min_factor`` comes from the run config's ``lr_min`` unless the fork sets
    one: `MOLECULE_GENERALIST.md` §7 says an anneal decays to ``lr/10`` and that
    is a property of the *recipe*, so it should not have to be restated in every
    fork config.
    """
    from .evaluate import check_selection

    out = {}
    if path:
        from sweep.expand import load_config
        loaded = load_config(path)
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: a fork config must be a JSON object")
        out.update(loaded)
    if args.decay_steps is not None:
        out["decay_steps"] = int(args.decay_steps)
    out.setdefault("tokens_per_step", config.tokens_per_step)
    out.setdefault("seed", config.seed)
    if args.fork_mode == "anneal":
        out.setdefault("min_factor", config.decay_min_factor())
    # D7.4, applied before anything is written: a fork may select, but never on
    # a key naming the test split.
    check_selection(out.get("selection"), mode=args.fork_mode)
    return out


def mode_fork(config: RunConfig, args) -> int:
    from .evaluate import build_validators
    from .fork import fork, plan_fork
    from .lineage import Lineage

    fork_config = load_fork_config(args.fork_config, args, config)
    registry, adapter_config = wiring.build_registry(config)

    if args.dry_run:
        plan = plan_fork(args.from_, args.fork_mode, fork_config,
                         registry=registry, run_dir=args.run_dir)
        print(json.dumps(plan.to_json(), indent=2, sort_keys=True, default=str))
        return 0

    validators = build_validators(config.validator_specs())
    mixture = (None if wiring.unbuilt_tasks(registry, config)
               else wiring.resolve_mixture(config, registry))
    eval_sets = wiring.build_eval_sets(config, registry, mixture,
                                       wiring.splits_wanted(validators),
                                       adapter_config)
    lineage = Lineage(config.lineage_dir())
    trainer_factory, validate = wiring.fork_callables(
        config, registry=registry, adapter_config=adapter_config,
        validators=validators, eval_sets=eval_sets, lineage=lineage)

    result = fork(args.from_, args.fork_mode, fork_config, registry=registry,
                  run_dir=args.run_dir, results_dir=config.lineage_dir(),
                  lineage=lineage, trainer_factory=trainer_factory,
                  validate=validate, runs_jsonl=config.runs_jsonl())
    print(json.dumps(result.to_json(), indent=2, sort_keys=True, default=str))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# eval
# ─────────────────────────────────────────────────────────────────────────────

def mode_eval(config: RunConfig, args) -> int:
    from . import checkpoint as ckpt_mod
    from .fork import load_start_weights

    ckpt = os.path.abspath(args.checkpoint)
    state = ckpt_mod.verify(ckpt)
    step = int(state.get("step") or 0)
    out_dir = os.path.abspath(args.out or os.path.join(ckpt, "eval"))

    run = wiring.build_run(config, output_dir=out_dir, fire_validators=False)
    load_start_weights(run.trainer, ckpt)

    result = wiring.run_evaluation(
        run, step, event="manual", only=(_names(args.only_validators) or None),
        scratch_dir=os.path.join(out_dir, "scratch"))

    os.makedirs(out_dir, exist_ok=True)
    record = {"mode": "eval", "checkpoint": ckpt, "step": step,
              "arm": config.arm, "config_hash": config.config_hash(),
              "checkpoint_state": {k: state.get(k) for k in
                                   ("step", "mixture_hash", "schema_version",
                                    "config_hash", "architecture_hash")},
              "eval": result.record()}
    path = os.path.join(out_dir, f"eval_step{step}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps(result.record(), indent=2, sort_keys=True, default=str))
    print(f"eval: wrote {path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

MODE_FUNCTIONS = {
    "validate": mode_validate,
    "data_prep": mode_data_prep,
    "train": mode_train,
    "resume": mode_resume,
    "fork": mode_fork,
    "eval": mode_eval,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(normalise_argv(argv))

    if getattr(args, "init", None) is not None:
        path = write_template(args.init, CONFIGS_DIR)
        print(f"Wrote {path}\n"
              f"Edit it, then:  python3 -m src.generalist validate --config {path}")
        return 0
    if not args.mode:
        parser.print_help()
        return 2

    config = config_from_args(args)
    return MODE_FUNCTIONS[args.mode](config, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
