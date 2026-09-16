"""The zero-shot notation probe — what each input representation is worth before any training.

`MOLECULE_GENERALIST.md` §8.3. Scores the **untrained** backbone on the
five Tier-B property sets, once per flat notation (SMILES, SELFIES, InChI) and
once on the graph arm, and reports ROC-AUC from the same yes/no margin readout
every property number in that document was produced by.

**What it is for.** §8 reads the flat arm's property-prediction lead as a SMILES
pretraining prior the graph arm cannot reach. The three notations determine the
same molecule, so a difference between them is not an information difference —
it is a difference in how much of each notation the backbone read. This probe
measures the *floor* of that ladder: what each representation is worth with no
training at all.

**The graph row is a floor, not a zero.** The graph arm's node text is English
(``carbon aromatic ring deg2 H1``), which the backbone reads perfectly well, so a
score above chance is the expected outcome. What the graph arm has no prior for
is the structural channel, and this probe runs with ``bias: none`` so that
channel is absent rather than random — a randomly initialised bias would add
noise to a floor measurement and make it harder to read, not more honest.

**One model serves all four arms.** A flat arm is a single-node graph, where every
structural bias is identically zero (Property 2), and the graph floor wants no
bias either. So a single ``bias: none`` backbone with no adapter is the correct
model for every row here, and there is nothing trained anywhere in this file.

**Indices match §8.** ``max_samples`` defaults to 500, which is what the campaign's
``in_mixture`` validator used, and `eval_indices` is the same fixed seeded
subsample — so a zero-shot row and a trained row are over the same molecules.

Usage (GPU, through Slurm — never on the login node):

    python3 -m src.generalist.tools.notation_probe --out results/notation_probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

#: The five property sets of §1's table. BACE / BBBP / HIV are the headline
#: three; Tox21 and SIDER are the multi-endpoint diagnostics.
DEFAULT_TASKS = ("bace", "bbbp", "hiv", "tox21", "sider")

#: `flat` is SMILES and is already built by the campaign; the other two build
#: here. `graph` is the floor row.
DEFAULT_ARMS = ("flat", "flat_selfies", "flat_inchi", "graph")

#: §8's `in_mixture` cap, so the rows are over the same molecules as the trained
#: numbers this floor is read against.
DEFAULT_MAX_SAMPLES = 500


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--out", default=None,
                   help="directory for probe.json (default: results/notation_probe)")
    p.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                   help="0 scores the whole split")
    p.add_argument("--impl", default="v2-flex")
    p.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--build-only", action="store_true",
                   help="materialise the sources and stop (CPU; no model is loaded)")
    p.add_argument("--checkpoint", default=None,
                   help="score a TRAINED checkpoint instead of the untrained "
                        "backbone; requires --run-config for its recipe")
    p.add_argument("--run-config", default=None,
                   help="the run config the checkpoint was trained under")
    p.add_argument("--cell", default=None,
                   help="which cell of a multi-cell --run-config the checkpoint "
                        "was trained under")
    return p.parse_args(argv)


def build_sources(config, tasks, arms, log=print) -> None:
    """Materialise the test split of every ``(task, arm)`` this probe scores.

    Cheap and idempotent: `build` skips anything already on disk, so the SMILES
    and graph arms are found rather than rebuilt. The draws are a deterministic
    function of the partition, so the notation arms get the *same molecules in
    the same order* as the arm-2 build — which is what makes the four rows of one
    task a comparison rather than four separate samples.
    """
    from ..adapters import molecules

    start = time.time()
    molecules.build(config, tasks=tuple(tasks), arms=tuple(arms),
                    splits=("test",), passes=1)
    log(f"[build] {len(tasks)} tasks x {len(arms)} arms, test split, "
        f"{time.time() - start:.0f}s")


def build_probe_model(impl, model_name, max_length, device=None):
    """Base Llama with **no bias and no adapter** — the untrained model itself.

    ``bias_params`` is empty, so the graph-attention path adds nothing to the
    attention logits, and `select_active_params` is deliberately not called: there
    is no LoRA to attach when nothing is being trained. `build_model` freezes
    every parameter, which is what this probe wants anyway.
    """
    import torch

    from ...experiments.expressiveness.training.dispatch import (
        build_collator, build_model,
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model(
        impl, model_name, {}, k_hop=0, k_hop_directed=False, device=device,
        flex_compile_mode="max-autotune-no-cudagraphs")
    model.eval()
    pad = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
           else tokenizer.eos_token_id)
    collator = build_collator(impl, tokenizer, pad, k_hop=0, k_hop_directed=False,
                              magnetic_m=0)
    return model, tokenizer, collator, device


def excluded_rows(config, task, arms, max_length) -> dict:
    """Row indices no arm may score, and why. Two reasons, both silent otherwise.

    **Truncation, which is the one that bites.** A flat arm is a single node, so
    its whole prompt is one text and `max_length` truncates it *from the right* —
    taking the answer with it. `render` then supervises "the prompt node's last
    token", which for a truncated row is some mid-molecule token, and the margin
    readout duly reports `y_true` = no and a margin read at a position that means
    nothing. Nothing raises. It shows up only as a `pos_rate` that disagrees
    between arms scoring identical molecules, which is how it was found: on SIDER
    the untruncated graph arm reads 0.5460 while the flat arms read 0.5180 and
    0.5080. The longer notations truncate more often, so this is *arm-asymmetric*
    — exactly the shape that would otherwise look like a notation effect.

    The graph arm is unaffected: it has hundreds of short nodes and `max_length`
    is per node, so no atom text comes near it.

    **Un-encodable molecules**, the `UNENCODABLE` placeholder. Currently never
    fires — under `SELFIES_CONSTRAINTS` all 53,921 Tier-B molecules encode and
    round-trip and InChI never fails — but it is checked rather than assumed.

    Both are unioned across arms and dropped from *all* of them, so the four rows
    of one task stay a comparison rather than four different samples.
    """
    from ...experiments.molecules.data import UNENCODABLE
    from ..adapters import molecules

    truncated, unencodable = set(), set()
    for arm in arms:
        if arm not in molecules.built_arms(config, task, "test"):
            # A build that only carries the arms it trains — the instruct
            # campaign builds `graph` and `flat` and no notation ladder — has
            # nothing to exclude for the arms it never made. Skipping is right
            # here and would be wrong for the ladder: there the union over all
            # four arms is the whole point, and a missing arm would silently
            # shrink it. What keeps both true is that the caller passes the arms
            # that exist, and this only forgives the ones that do not.
            continue
        if arm == "graph":
            # Immune by construction, and skipped rather than scanned: a graph
            # row is hundreds of atom texts, `max_length` is per node, and the
            # longest atom text the encoding produces is a dozen tokens. Scanning
            # it would mean reading every row's SPD and magnetic matrices —
            # minutes of Arrow reads — to confirm an arithmetic fact.
            continue
        source = molecules.load(task, "test", arm, 0, config)
        # A flat arm is one node, so the per-example token count measured at
        # build time *is* the prompt node's length, and it is already in the
        # sidecar. No row has to be materialised to find the truncated ones.
        _nodes, tokens = source.lengths()
        truncated.update(i for i, n in enumerate(tokens) if n >= max_length)
        # The prompt text comes off the stored graphs, which `load` already
        # brought into memory — one node per row on a flat arm.
        for i, graph in enumerate(source.dataset.graphs):
            node = graph.graph.get("prompt_node", 0)
            if UNENCODABLE in graph.nodes[node]["text"]:
                unencodable.add(i)
    return {"truncated": sorted(truncated),
            "unencodable": sorted(unencodable),
            "all": sorted(truncated | unencodable)}


def _score_indices(model, tokenizer, collator, source, spec, indices, device,
                   batch_tokens) -> dict:
    """`score_source` on an explicit index list.

    `score_source` chooses its own subsample, and this probe has to hand the same
    one to four arms *after* removing the shared exclusions — so it calls the
    yes/no scorer underneath directly. Everything else about the readout is
    identical, which is what keeps a zero-shot row comparable with §8's trained
    ones.
    """
    from ..evaluate.scorers import METRIC_KEYS, _score_yesno

    if spec.answer_kind != "yesno":
        raise ValueError(
            f"{spec.name}: this probe scores yes/no property sets, not "
            f"{spec.answer_kind!r}")
    if not indices:
        return {k: (0 if k == "n" else float("nan")) for k in METRIC_KEYS["yesno"]}
    return _score_yesno(model, tokenizer, collator, source, indices, device,
                        batch_size=8, per_endpoint=True, batch_tokens=batch_tokens)


def build_trained_model(run_config_path, checkpoint, out_dir, cell=None):
    """A trained checkpoint, loaded through the harness that produced it.

    The same path `__main__.mode_eval` takes — `wiring.build_run` then
    `fork.load_start_weights` — so the model is assembled with the run's own LoRA
    and bias configuration rather than a reconstruction of it. `fire_validators`
    is off: this tool does its own scoring, with the exclusions the default
    validators do not apply.

    Scoring a trained model here rather than through `eval` is the point: the
    `in_mixture` validator has no notion of the truncated rows §8 documents, so
    its property numbers are over a slightly different set of molecules on each
    arm. These are over the same set on every arm, which is what a ladder needs.
    """
    from .. import wiring
    from ..config import RunConfig, load_config_file
    from ..fork import load_start_weights

    config = RunConfig(**load_config_file(run_config_path, cell)).validate()
    run = wiring.build_run(config, output_dir=out_dir, fire_validators=False)
    load_start_weights(run.trainer, checkpoint)
    # `Run` carries these directly; reaching through the trainer gets a
    # `tokenizer` that recent transformers no longer sets.
    run.model.eval()
    return run.model, run.tokenizer, run.collator, config, run.device


def score(config, tasks, arms, model, tokenizer, collator, device,
          max_samples, batch_tokens, log=print) -> list:
    """One row per ``(task, arm)``: the metrics plus the size disclosures."""
    import torch

    from ..adapters import molecules
    from ..evaluate.scorers import eval_indices

    specs = molecules.task_specs(config)
    rows = []
    # **The exclusion is over every arm of the ladder, not the arms being
    # scored.** A trained checkpoint reads one notation, so scoring it alone
    # would drop only the rows *its own* arm truncates — and the four arms would
    # then be scored on four slightly different sets of molecules, which is the
    # one thing the ladder cannot survive. Computing the union over
    # `DEFAULT_ARMS` makes every run, whenever it is scored and whatever it
    # reads, drop the same rows. `check_arms_agree_on_labels` is what caught the
    # per-arm version: BBBP read a base rate of 0.5271 on SELFIES against 0.5245
    # everywhere else, because SELFIES alone had dropped its one truncated row.
    dropped = {task: excluded_rows(config, task, DEFAULT_ARMS, config.max_length)
               for task in tasks}
    for task, why in dropped.items():
        if why["all"]:
            log(f"[exclude] {task}: {len(why['all'])} row(s) dropped from every "
                f"arm — {len(why['truncated'])} truncated at max_length, "
                f"{len(why['unencodable'])} un-encodable")

    for arm in arms:
        for task in tasks:
            source = molecules.load(task, "test", arm, 0, config)
            spec = specs[f"{molecules.MOLECULE_PREFIX}{task}"]
            # The same seeded subsample every other number in this campaign
            # uses, minus the shared exclusions — computed once per task, so all
            # four arms score exactly the same molecules in the same order.
            keep = set(eval_indices(len(source), max_samples or None))
            keep -= set(dropped[task]["all"])
            indices = sorted(keep)
            start = time.time()
            with torch.no_grad():
                out = _score_indices(model, tokenizer, collator, source, spec,
                                     indices, device, batch_tokens)
            # Tokens per example is a required disclosure for this ladder
            # (`MOLECULE_GENERALIST.md` §8.3): a notation can score low because the backbone read less
            # of it, or because it is simply longer to attend over, and the two
            # are different causes.
            _nodes, tokens = source.lengths()
            row = {
                "task": task, "arm": arm,
                "roc_auc": out.get("roc_auc"),
                # The positive-class rate of the scored split. A score without
                # its floor is uninterpretable (`molecules/PLAN.md` §8.1), and on
                # an untrained model the distance from 0.5 is the whole reading.
                "pos_rate": out.get("pos_rate"),
                "accuracy": out.get("accuracy"),
                "n_distinct": out.get("n_distinct"),
                "tied_pair_fraction": out.get("tied_pair_fraction"),
                "n": out.get("n"),
                "n_source": len(source),
                "n_excluded": len(dropped[task]["all"]),
                "n_truncated": len(dropped[task]["truncated"]),
                "n_unencodable": len(dropped[task]["unencodable"]),
                "mean_tokens": (sum(tokens) / len(tokens)) if tokens else None,
                "max_tokens": max(tokens) if tokens else None,
                "seconds": round(time.time() - start, 1),
                "metrics": out,
            }
            rows.append(row)
            log(f"[score] {arm:13s} {task:6s} roc_auc="
                f"{_fmt(row['roc_auc'])} pos_rate={_fmt(row['pos_rate'])} "
                f"tied={_fmt(row['tied_pair_fraction'])} "
                f"n={row['n']} mean_tokens={_fmt(row['mean_tokens'], 1)} "
                f"({row['seconds']}s)")
    check_arms_agree_on_labels(rows)
    return rows


def check_arms_agree_on_labels(rows) -> None:
    """Every arm of one task must see the same labels. Asserted, not just read.

    `pos_rate` is ``y_true.mean()`` over the scored rows, and the four arms score
    the same molecules in the same order — so it is a property of the task, not
    of the arm, and any disagreement means the arms are not scoring what this
    probe claims they are. This is the check that would have caught the
    truncation defect on its first run instead of after it had produced a table:
    SIDER read 0.5460 on the graph arm against 0.5180 and 0.5080 on two flat
    ones, which is the whole bug in one number.

    `molecules/PLAN.md` §9: a quantity that is only ever read and never asserted
    on has no error-detecting surface.
    """
    by_task: dict = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    for task, group in sorted(by_task.items()):
        rates = {r["arm"]: r["pos_rate"] for r in group}
        values = [v for v in rates.values() if v is not None]
        if len(values) > 1 and (max(values) - min(values)) > 1e-9:
            raise AssertionError(
                f"{task}: the arms disagree on the label base rate {rates}. "
                "They score identical molecules, so this cannot differ — some "
                "rows are being scored at the wrong position (truncation), or "
                "the arms are not aligned. Do not report these numbers.")
        counts = {r["arm"]: r["n"] for r in group}
        if len(set(counts.values())) > 1:
            raise AssertionError(
                f"{task}: the arms scored different row counts {counts}")


def _fmt(value, places: int = 4) -> str:
    return "  n/a " if value is None else f"{float(value):.{places}f}"


def table(rows) -> str:
    """The zero-shot read-out (`MOLECULE_GENERALIST.md` §8.3), arms as columns."""
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != "flat", a))
    # DEFAULT_TASKS order where known, then anything a caller added, so a custom
    # --tasks does not make the read-out raise.
    tasks = sorted({r["task"] for r in rows},
                   key=lambda t: (DEFAULT_TASKS.index(t)
                                  if t in DEFAULT_TASKS else len(DEFAULT_TASKS), t))
    by = {(r["task"], r["arm"]): r for r in rows}

    width = max(14, max((len(a) for a in arms), default=14))
    head = f"{'task':8s}" + "".join(f"{a:>{width}s}" for a in arms)
    lines = ["", "zero-shot ROC-AUC (untrained backbone, no bias, no adapter)",
             head, "-" * len(head)]
    for task in tasks:
        cells = "".join(f"{_fmt(by.get((task, a), {}).get('roc_auc')):>{width}s}"
                        for a in arms)
        lines.append(f"{task:8s}{cells}")
    lines.append("")
    lines.append("tied pair fraction (an untrained model can emit one margin for "
                 "every row; 1.0 makes the AUROC above meaningless)")
    for task in tasks:
        cells = "".join(
            f"{_fmt(by.get((task, a), {}).get('tied_pair_fraction')):>{width}s}"
            for a in arms)
        lines.append(f"{task:8s}{cells}")
    lines.append("")
    lines.append("mean tokens per example")
    for task in tasks:
        cells = "".join(
            f"{_fmt(by.get((task, a), {}).get('mean_tokens'), 1):>{width}s}"
            for a in arms)
        lines.append(f"{task:8s}{cells}")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = _args(argv)
    from ...experiments.molecules.data import SELFIES_CONSTRAINTS
    from ..adapters import molecules

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    unknown = [a for a in arms if a not in ("graph",) + tuple(molecules.FLAT_NOTATIONS)]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}")

    config = molecules.MoleculeAdapterConfig(
        model_name=args.model_name, max_length=args.max_length).validate()
    out_dir = os.path.abspath(
        args.out or os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "results", "notation_probe"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"build_version {config.build_version()}  "
          f"partition {config.partition_version()}")
    if args.checkpoint:
        # **Do not build when scoring a checkpoint.** `build` rewrites the shared
        # `manifest.json` through a single `manifest.json.tmp`, so two scoring
        # jobs running at once race on that path and one dies with a
        # `FileNotFoundError` after the other's `os.replace` — the same
        # check-then-act shape as the fork races in `DESIGN.md` §D9, one level up.
        # Nothing needs building here anyway: a trained checkpoint implies its
        # sources exist, and `load` says so plainly if they do not.
        print("scoring a checkpoint: sources must already exist, not building")
    else:
        build_sources(config, tasks, arms)
    if args.build_only:
        print("build-only: sources materialised, no model loaded")
        return 0

    run_name = None
    if args.checkpoint:
        if not args.run_config:
            raise SystemExit("--checkpoint needs --run-config: a trained model is "
                             "assembled from the recipe it was trained under")
        model, tokenizer, collator, trained_config, device = build_trained_model(
            args.run_config, os.path.abspath(args.checkpoint),
            os.path.join(out_dir, "scratch"), args.cell)
        run_name = trained_config.run_name
        # A trained model reads one notation. Scoring it on another arm is a
        # cross-notation *transfer* measurement, which is a different question
        # from the ladder's, so the default is its own arm and anything else has
        # to be asked for.
        if args.arms == ",".join(DEFAULT_ARMS):
            arms = (trained_config.arm,)
            print(f"checkpoint arm: scoring {arms[0]!r} only")
    else:
        model, tokenizer, collator, device = build_probe_model(
            args.impl, args.model_name, args.max_length)
    rows = score(config, tasks, arms, model, tokenizer, collator, device,
                 args.max_samples, args.batch_tokens)

    record = {
        "probe": "notation_trained" if args.checkpoint else "notation_zero_shot",
        "section": ("MOLECULE_GENERALIST.md §8.3 (trained)" if args.checkpoint
                    else "MOLECULE_GENERALIST.md §8.3 (zero-shot)"),
        "build_version": config.build_version(),
        "partition_version": config.partition_version(),
        "model_name": args.model_name,
        "impl": args.impl,
        "max_samples": args.max_samples,
        "trained": bool(args.checkpoint),
        "checkpoint": args.checkpoint,
        "run_config": args.run_config,
        "cell": args.cell,
        "bias": "none",
        "adapter": None,
        "selfies_constraints": SELFIES_CONSTRAINTS,
        "rows": rows,
    }
    stem = f"trained_{run_name}.json" if args.checkpoint else "probe.json"
    path = os.path.join(out_dir, stem)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    print(table(rows))
    print(f"\nnotation_probe: wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
