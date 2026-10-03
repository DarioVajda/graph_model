"""What a checkpoint says when it is asked, and whether the answer was licensed.

Step 11's scorer, and the reason `mol/assistant` can be registered as an ordinary
text corpus. The answers in this set were **composed** from RDKit facts and only
re-voiced, so every row carries the statements its reply was supposed to make.
That is what makes a free-form reply scorable with no reference string — and it
is why `TaskSpec.metric` is `bleu2` and *this* is the measurement. An overlap
score against one voicing of an answer whose `brief` varies row by row cannot
tell responsive from unresponsive, preserved from dropped, or licensed from
invented, and quoting it as if it could would be the same error as quoting an
uncalibrated filter.

Three modes, and the middle one is deliberately not a new instrument:

    --mode generate  a checkpoint's replies over the composed test split,
                     written in exactly the shapes `pipeline/judge.py` reads
    (then run the judge over them, unmodified)
    --mode score     the judged verdicts and the pattern checks, per twist and
                     per task

**The judge runs unmodified.** It was calibrated against the writer's output on
the same three axes wanted here, so `generate` writes `batches/`, `ask/` and
`voice/` rather than teaching a second copy of it what a decline looks like. The
upstream build's batch files were deleted; the render dict is rebuilt from the
sidecar, which carries the accept pass's renaming of it in full.

    GPU=1 src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.analysis.score \\
        --mode generate --config .../008_molecule_generalist_instruct.jsonc \\
        --cell molecule_generalist_instruct_graph_s0 \\
        --checkpoint .../replay_anneal15_graph_s0/anneal/checkpoint-12255 \\
        --out .../results/assistant/score/control_s0
    GPU=1 VENV_BIN=.venv_writer/bin CONTAINER=.../nemo_26.04.sqsh \\
    GPU_CONSTRAINT='GPU_BRD:B200|GPU_BRD:B300' \\
    src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.pipeline.judge \\
        --model .../gemma-4-31B-it --batches .../control_s0/batches \\
        --asks .../control_s0/ask --voiced .../control_s0/voice \\
        --out .../control_s0/judged
    src/generalist/tools/launch/run_py.sh -m src.generalist.assistant.analysis.score \\
        --mode score --out .../control_s0

**Compare, do not quote.** The judge's precision as a filter measured 0.231 and
the three fixes that followed make that number stale. Its bias is a property of
the judge, so two checkpoints read on the same rows share it and it cancels in
the difference: a delta against the control is reportable where the level is not.
Recalibrate with `calibrate.py` before any absolute rate ships.

**The zero-shot control is owed, and it is a build rather than a flag.** The
demonstrations are disconnected graph components, so a model can answer by
copying the nearest one instead of reading the molecule, and scoring the same
rows with the shots stripped is what separates the behaviour from the shortcut —
the eval-time half of the 50/50 polarity rule `pipeline/compose.py` enforces at
build time. It is not a switch here because the shots are baked into the graph at
`_draw_assistant`, so the control needs its own artifact; it belongs beside the
flat control in §9.4's plan. A flag here that quietly generated from the
shot-bearing graphs anyway would be the unwired-argparse failure this repo has
already paid for once.
"""

import argparse
import glob
import json
import os
import statistics
import sys

#: The composed row is the accept pass's renaming of the render dict, so this
#: inverts it exactly for the six fields that were renamed.
#:
#: `ask` is the one field no composed row carries, and the two keys read off it
#: are both recoverable: a row is unanswerable exactly when its twist says so,
#: and `constraint` is only ever tested as `constraint and verdict`, so a row
#: with a verdict is a row with a constraint. Reconstructed rather than guessed —
#: if either mapping stops holding, the judge quietly stops being told that a
#: decline was wanted, which is the single largest error it made on the build.
def render_of(meta: dict) -> dict:
    """The render dict the judge and the accept pass expect, from a sidecar."""
    return {
        "statements": meta["statements"],
        "answers": meta.get("answers"),
        "verdict": meta.get("verdict"),
        "gloss": meta.get("gloss"),
        "skeleton": meta.get("skeleton"),
        "reply": meta.get("rendered_reply"),
        "turns": meta.get("turns"),
        "ask": {"answerable": meta.get("twist") != "unanswerable",
                "constraint": bool(meta.get("verdict"))},
    }


def _write_jsonl(path: str, rows) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_jsonl(path: str) -> list:
    out = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# generate
# ─────────────────────────────────────────────────────────────────────────────

def mode_generate(args) -> int:
    from ...adapters import molecules as adapter
    from ...config import RunConfig, load_config_file
    from ...evaluate.scorers import generate_predictions
    from ...fork import load_start_weights
    from ... import wiring

    config = RunConfig(**load_config_file(args.config, args.cell)).validate()
    registry, adapter_config = wiring.build_registry(config)
    spec = registry.get(args.task)

    run = wiring.build_run(config, output_dir=os.path.join(args.out, "scratch"),
                           fire_validators=False)
    load_start_weights(run.trainer, args.checkpoint)

    source = adapter.load(args.task, args.split, config.arm, pass_id=0,
                          config=adapter_config)
    n = len(source) if not args.limit else min(args.limit, len(source))
    indices = list(range(n))
    print(f"{n} {args.split} rows from {source.path}", flush=True)

    predictions, targets = generate_predictions(
        run.model, run.tokenizer, run.collator, source, indices,
        max_new_tokens=args.max_new_tokens or (spec.max_new_tokens or 160),
        device=run.device)

    os.makedirs(args.out, exist_ok=True)
    for sub in ("batches", "ask", "voice"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    examples, asks, voices, rows = [], [], [], []
    for i, prediction, target in zip(indices, predictions, targets):
        meta = source.example(i).meta
        row_id = meta["id"]
        examples.append({"id": row_id, "render": render_of(meta)})
        # The question as the model was given it, which is `question_text`'s
        # output and therefore carries the few-shot pointer line. The judge is
        # asked whether the reply answered *that*, not the bare question.
        asks.append({"id": row_id, "turn": source.example(i).question})
        voices.append({"id": row_id, "reply": prediction})
        rows.append({"id": row_id, "reply": prediction, "target": target,
                     "twist": meta.get("twist"), "task": meta.get("task"),
                     "brief": meta.get("brief"), "shots": len(meta["shots"]),
                     "source": meta.get("source")})

    stem = f"{args.split}-batch-0000"
    with open(os.path.join(args.out, "batches", f"{stem}.json"), "w") as handle:
        json.dump({"role": args.split, "batch": 0, "examples": examples}, handle)
    _write_jsonl(os.path.join(args.out, "ask", f"{stem}.jsonl"), asks)
    _write_jsonl(os.path.join(args.out, "voice", f"{stem}.jsonl"), voices)
    _write_jsonl(os.path.join(args.out, "generations.jsonl"), rows)

    with open(os.path.join(args.out, "generate.json"), "w") as handle:
        json.dump({"checkpoint": args.checkpoint, "task": args.task,
                   "split": args.split, "arm": config.arm, "n": n,
                   "build_version": source.build_version,
                   "source": source.path}, handle, indent=1)
    lengths = [len(r["reply"]) for r in rows]
    print(f"wrote {len(rows)} replies to {args.out}; "
          f"reply chars mean {statistics.mean(lengths):.0f} "
          f"max {max(lengths)}, {sum(1 for l in lengths if l == 0)} empty")
    print("next: assistant.pipeline.judge --batches {0}/batches --asks {0}/ask "
          "--voiced {0}/voice --out {0}/judged".format(args.out))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# score
# ─────────────────────────────────────────────────────────────────────────────

#: The deterministic half. Every one of these is a check the accept pass already
#: runs on the *writer*, reused verbatim on the model — they cost nothing, they
#: cannot disagree with themselves between two readings, and they were built
#: against exactly this failure surface.
def pattern_checks(render: dict, reply: str, fmt: str) -> dict:
    from ..pipeline.accept import (declines, dropped_statements, format_met,
                                   states_the_gloss, states_the_verdict)

    out = {"dropped": dropped_statements(render, reply, fmt),
           "format_met": format_met(reply, fmt, render.get("skeleton"))}
    if render["ask"].get("constraint") and render.get("verdict"):
        out["verdict_stated"] = states_the_verdict(render, reply, fmt)
    if render.get("gloss"):
        out["gloss_stated"] = states_the_gloss(render, reply)
    if render["ask"].get("answerable") is False:
        out["declined"] = declines(reply)
    return out


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


def mode_score(args) -> int:
    generations = {r["id"]: r for r in
                   _load_jsonl(os.path.join(args.out, "generations.jsonl"))}
    batch_paths = sorted(glob.glob(os.path.join(args.out, "batches", "*.json")))
    renders = {}
    for path in batch_paths:
        with open(path) as handle:
            for example in json.load(handle)["examples"]:
                renders[example["id"]] = example["render"]

    judged = {}
    for path in sorted(glob.glob(os.path.join(args.out, "judged", "*.jsonl"))):
        for row in _load_jsonl(path):
            judged[row["id"]] = row
    if not judged:
        print(f"no judged rows under {args.out}/judged — the pattern checks "
              "below stand alone, and the three entailment axes are unmeasured.",
              file=sys.stderr)

    per_row, by_twist, by_task, by_shots = [], {}, {}, {}
    for row_id, generation in generations.items():
        render = renders[row_id]
        fmt = (generation.get("brief") or {}).get("format", "")
        checks = pattern_checks(render, generation["reply"], fmt)
        verdict = judged.get(row_id, {})
        entry = {
            "id": row_id, "twist": generation.get("twist"),
            "task": generation.get("task"), "shots": generation.get("shots"),
            "n_dropped": len(checks["dropped"]),
            "n_statements": len(render["statements"]),
            "format_met": checks["format_met"],
            "verdict_stated": checks.get("verdict_stated"),
            "gloss_stated": checks.get("gloss_stated"),
            "declined": checks.get("declined"),
            "responsive": verdict.get("responsive"),
            "preserved": verdict.get("preserved"),
            "added": verdict.get("added"),
            "judged": bool(verdict),
        }
        # "Clean" is the conjunction of everything that was actually measured on
        # this row, so a row the judge did not reach is not silently counted as
        # passing its three axes.
        flags = [entry["n_dropped"] == 0, entry["format_met"]]
        for key in ("verdict_stated", "gloss_stated", "declined"):
            if entry[key] is not None:
                flags.append(entry[key])
        for key in ("responsive", "preserved"):
            if entry[key] is not None:
                flags.append(entry[key])
        if entry["added"] is not None:
            flags.append(not entry["added"])
        entry["clean"] = all(flags)
        per_row.append(entry)
        for bucket, key in ((by_twist, entry["twist"]), (by_task, entry["task"]),
                            (by_shots, entry["shots"])):
            bucket.setdefault(key, []).append(entry)

    def summarise(rows):
        judged_rows = [r for r in rows if r["judged"]]
        return {
            "n": len(rows),
            "clean": _rate(sum(1 for r in rows if r["clean"]), len(rows)),
            "statements_dropped": _rate(
                sum(r["n_dropped"] for r in rows),
                sum(r["n_statements"] for r in rows)),
            "format_met": _rate(sum(1 for r in rows if r["format_met"]), len(rows)),
            "n_judged": len(judged_rows),
            "responsive": _rate(sum(1 for r in judged_rows if r["responsive"]),
                                len(judged_rows)),
            "preserved": _rate(sum(1 for r in judged_rows if r["preserved"]),
                               len(judged_rows)),
            "added": _rate(sum(1 for r in judged_rows if r["added"]),
                           len(judged_rows)),
        }

    report = {
        "overall": summarise(per_row),
        "by_twist": {k: summarise(v) for k, v in sorted(
            by_twist.items(), key=lambda kv: str(kv[0]))},
        "by_task": {k: summarise(v) for k, v in sorted(
            by_task.items(), key=lambda kv: str(kv[0]))},
        "by_shot_count": {str(k): summarise(v) for k, v in sorted(
            by_shots.items(), key=lambda kv: (kv[0] is None, kv[0]))},
    }
    for name in ("declined", "verdict_stated", "gloss_stated"):
        scoped = [r for r in per_row if r[name] is not None]
        if scoped:
            report[name] = {"n": len(scoped),
                            "rate": _rate(sum(1 for r in scoped if r[name]),
                                          len(scoped))}

    _write_jsonl(os.path.join(args.out, "per_row.jsonl"), per_row)
    with open(os.path.join(args.out, "score.json"), "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print(json.dumps(report, indent=1, sort_keys=True))
    print("\nThese are a level, and the judge's own precision is stale. Quote a "
          "difference against the control, or recalibrate first.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("generate", "score"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--cell")
    parser.add_argument("--checkpoint")
    parser.add_argument("--task", default="mol/assistant")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "generate":
        for required in ("config", "checkpoint"):
            if not getattr(args, required):
                parser.error(f"--mode generate needs --{required}")
        return mode_generate(args)
    return mode_score(args)


if __name__ == "__main__":
    sys.exit(main())
