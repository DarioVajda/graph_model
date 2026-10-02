"""The judge pass: is the reply responsive, and did the re-voicing preserve it?

Step 5 of §9.4's order of work, and the only filter left that is not a decision
procedure. Everything a pattern can decide with certainty — the anchors are
named, the statements survive, the format brief is met — lives in
`accept.py`. What remains is two entailment questions:

* **responsive** — does the reply answer the message that was actually sent, as
  opposed to a neighbouring question;
* **preserved** — does the reply still *state* each rendered statement, rather
  than merely reuse its vocabulary. The accept pass checks this over content
  words, which is sound against a dropped clause and blind to a negated one.
* **added** — does the reply assert anything about the molecule that no statement
  licenses. The whole design exists to make this unreachable, so a nonzero rate
  here is a finding about the renderer or the voice prompt.

A 31B instruct model does these better than any pattern, and it is a different
task in kind from writing, so its blind spots are not the writer's. It is still
a filter, and **an unmeasured filter is the mistake this section has made four
times** — so its verdicts do not count until `analysis/calibrate.py` has scored
them against 100 hand-read rows, and its precision and recall ship with the set.

    VENV_BIN=.venv_writer/bin CONTAINER=/shared/workspace/povejmo/containers/nemo_26.04.sqsh \
    GPU=1 GPU_CONSTRAINT='GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100' \
    src/generalist/tools/run_py.sh -m src.generalist.assistant.pipeline.judge \
        --model .../gemma-4-31B-it --batches .../v5 \
        --asks .../v5/ask --voiced .../v5/voice --out .../v5/judged

Greedy, not sampled: a filter that gives a different verdict on a second reading
of the same row cannot be calibrated on the first.

The system prompt is the domain's `judge_system` (`--domain`, molecules by
default); the four-line verdict format it asks for is parsed here.
"""

import argparse
import glob
import json
import os
import re
import sys

from ..domain import get_domain

_VERDICT = re.compile(r"^\s*(RESPONSIVE|PRESERVED|ADDED)\s*:\s*(yes|no)\b",
                      re.IGNORECASE | re.MULTILINE)
_NOTE = re.compile(r"^\s*NOTE\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def judge_prompt(example, turn: str, reply: str) -> str:
    rendered = example["render"]
    lines = [f"USER MESSAGE: {turn}", "", f"REPLY: {reply}", "",
             "STATEMENTS the reply was supposed to make:"]
    lines += [f"  - {s}" for s in rendered["statements"]]
    # The decision answers the constraint, not a fact, so it is not one of the
    # statements. Withheld, the judge scored a `decide` reply both ways wrong: a
    # reply that never decided was PRESERVED, because every statement was there,
    # and one that did decide was ADDED, because the verdict is an assertion
    # about the molecule that no statement licenses.
    if rendered["ask"].get("constraint") and rendered.get("verdict"):
        # "given anywhere" is doing real work. `states_the_verdict` took four
        # rewrites to learn that a decision is rarely a sentence of its own — it
        # is the leading "No." of "No. It contains no ketone.", or item 1 of a
        # numbered list — and the judge made the same mistake the same way,
        # refusing rows for a missing DECISION that the reply's own answer gave.
        lines += ["", "DECISION the reply was also supposed to give: "
                  + ("Yes — it meets the constraint."
                     if rendered["verdict"] == "yes"
                     else "No — it does not meet the constraint.")
                  + " It counts as given anywhere in the reply, including as a "
                    "bare yes or no, or as one item of a list; it does not need "
                    "a sentence of its own."]
    # Same again for the gloss: it is a definition, not a fact about the
    # molecule, so a judge that only sees the statements reads a kept explanation
    # as an addition and a dropped one as fully preserved.
    if rendered.get("gloss"):
        lines += ["", "EXPLANATION the reply was also supposed to give: "
                  + rendered["gloss"]]
    if rendered["ask"].get("answerable") is False:
        # Spelling out what a decline looks like, because the shape it takes in a
        # schema is the single largest error the judge made on the final build.
        # 39 of its 59 ADDED refusals were a `fill_record` whose schema had a
        # field for the property nobody can compute, answered `null` — which is
        # the only way to decline inside a schema the question itself fixed, and
        # which the judge read as a supplied value every time. Five more were the
        # prose version, where it called the refusal itself unlicensed content.
        lines += ["", "NOTE: the reply was also supposed to decline part of the "
                  "question. Declining is not an addition, and answering what it "
                  "should have declined is. A reply that says it does not have "
                  "the value has declined. Where the question fixes a JSON "
                  "schema, a null in that field is the decline — the only one "
                  "available — and is not a supplied value."]
    return "\n".join(lines)


def parse_verdict(text: str) -> dict:
    """The three verdicts and the note, or `{}` if the judge did not answer."""
    found = {key.lower(): value.lower() == "yes"
             for key, value in _VERDICT.findall(text or "")}
    if len(found) != 3:
        return {}
    note = _NOTE.search(text or "")
    found["note"] = note.group(1).strip() if note else ""
    return found


def _load_jsonl(directory: str, field: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if field in row:
                        out[row["id"]] = row[field]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--asks", required=True)
    parser.add_argument("--voiced", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", default="*batch-*.json")
    # 96 truncated the judge mid-sentence on the final build, and the rows it
    # cost were ones where it had reasoned its way round to the right answer and
    # ran out of room to record it — the NOTE line stopped at "This preserves the
    # number" and the verdict never arrived. The four lines asked for are short;
    # the ceiling only has to be wide enough that a judge which pads them can
    # still reach the end.
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--domain", default=None,
                        help="the assistant domain the batches were built for "
                             "(default: molecules)")
    args = parser.parse_args(argv)
    system = get_domain(args.domain).judge_system

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ...tools.check_chat_template import gate

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gate(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    turns = _load_jsonl(args.asks, "turn")
    replies = _load_jsonl(args.voiced, "reply")

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.batches, args.glob)))
    paths = [p for i, p in enumerate(paths) if i % args.shards == args.shard]
    if not paths:
        raise SystemExit(f"no batch files matched {args.glob} in {args.batches}")

    counts = {"responsive": 0, "preserved": 0, "added": 0, "judged": 0,
              "unparsed": 0}
    for path in paths:
        with open(path) as handle:
            batch = json.load(handle)
        examples = [e for e in batch["examples"]
                    if e["id"] in turns and e["id"] in replies]

        rows = []
        for start in range(0, len(examples), args.group):
            group = examples[start:start + args.group]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": judge_prompt(
                     e, turns[e["id"]], replies[e["id"]])}],
                tokenize=False, add_generation_prompt=True) for e in group]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**encoded, do_sample=False,
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
            for example, sequence in zip(group, out):
                text = tokenizer.decode(
                    sequence[encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True)
                verdict = parse_verdict(text)
                row = {"id": example["id"], "raw": text.strip()}
                if verdict:
                    row.update(verdict)
                    counts["judged"] += 1
                    for key in ("responsive", "preserved", "added"):
                        counts[key] += bool(verdict[key])
                else:
                    counts["unparsed"] += 1
                rows.append(row)

        name = os.path.basename(path).replace(".json", ".jsonl")
        with open(os.path.join(args.out, name), "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"{os.path.basename(path)}: {len(rows)} judged", flush=True)

    total = max(counts["judged"], 1)
    print(f"\n{counts['judged']} judged, {counts['unparsed']} unparsed")
    for key in ("responsive", "preserved", "added"):
        print(f"  {key:11} {counts[key]:6}  {counts[key] / total:.3f}")
    print("\nThese rates mean nothing until analysis/calibrate.py has scored the "
          "judge against hand labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
