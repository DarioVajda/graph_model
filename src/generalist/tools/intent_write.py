"""The two writer calls. Neither asks the model to know any chemistry.

Step 4 of §9.4's order of work.

    --pass ask     write the person's turn from the situation and the ask
    --pass voice   re-voice the rendered reply under the style brief

**They are two calls and not one on purpose.** The `ask` pass is never shown the
answer — only what is being asked for — so a question that states its own answer
is not caught, it is unreachable. Folding the two into one generation to save a
forward pass would hand the model the reply while it writes the question and give
that guarantee away.

The `voice` pass is shown the statements verbatim and told not to add to them. It
is performing a string transformation on supplied content, which is what an
instruct model at this size does near-perfectly, rather than composing a sentence
about a molecule, which is where it errs.

    VENV_BIN=.venv_writer/bin CONTAINER=/shared/workspace/povejmo/containers/nemo_26.04.sqsh \
    GPU=1 GPU_CONSTRAINT='GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100' \
    src/generalist/tools/run_py.sh -m src.generalist.tools.intent_write \
        --model /shared/workspace/povejmo/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/b9ea41a2887d8607f594846523f94c6cc75ac8a4 \
        --batches .../v5 --out .../v5/ask --pass ask

`VENV_BIN` and `CONTAINER` travel together: the venv's python is a symlink into
that image and exists nowhere else, so the wrong container fails as a bad path.
"""

import argparse
import glob
import json
import os
import re
import sys

ASK_SYSTEM = (
    "You write the user's side of a conversation with a chemistry assistant. "
    "You are given who the person is, what they are doing, and what they want to "
    "know. Write only their message.\n\n"
    "Rules:\n"
    "- Write what the person would actually type. Their situation shapes the "
    "wording; do not state the situation.\n"
    "- Ask for exactly what is listed under WANTS, nothing more, and in the "
    "order it is listed. Do not ask about any other property, and do not invent "
    "one.\n"
    "- Name every anchor under ANCHORS exactly as written, including the atom "
    "index and its element in brackets, and use no other identifier for an atom "
    "or a group than the ones given there.\n"
    "- The person and the assistant are already looking at the same structure. "
    "Refer to it as 'this molecule', 'this compound' or 'it'. Never invent a "
    "name, a code, a label or a SMILES string for it, and never invent an atom "
    "reference that is not under ANCHORS.\n"
    "- You do not know the answer and must not guess one, imply one, or ask a "
    "question whose wording assumes one.\n"
    "- One to three sentences. No preamble, no sign-off, no quotation marks."
)

VOICE_SYSTEM = (
    "You rewrite a chemistry assistant's reply in a given voice. You are given "
    "the user's message, a plain draft of the reply, and a style. Rewrite the "
    "draft.\n\n"
    "Rules:\n"
    "- Say everything the draft says. Every statement under STATEMENTS must "
    "still be stated, and where there is a DECISION or an EXPLANATION line the "
    "reply must carry that as well — it is what the user asked for and it is "
    "not one of the statements.\n"
    "- Add nothing. No extra facts, no reasons, no chemistry the draft does not "
    "contain, no offers of further help.\n"
    "- Do not mention the draft, the statements, or that you were given "
    "anything.\n"
    "- Follow the style exactly. Where the style asks for one word or for JSON, "
    "the value alone is the whole reply and the statements are what it comes "
    "from — do not restate them in a sentence.\n"
    "- Output only the rewritten reply."
)


def _ask_prompt(example) -> str:
    intent, rendered = example["intent"], example["render"]
    ask = rendered["ask"]
    wants = ask.get("asks") or []
    if not wants:
        # `wants` alone is a task name ("the value", "a comparison") and carries
        # no subject, so a writer handed it invents one — bond lengths,
        # electronegativities, a residue called Arg124. A brief with no `asks` is
        # a build bug, and it fails here rather than reaching the model.
        raise ValueError(f"{example['id']}: ask carries no asks to write from")
    lines = [
        f"PERSON: {intent['situation']['persona']}",
        f"DOING: {intent['situation']['context']}",
        "WANTS: " + "; ".join(wants),
    ]
    # Atom references only. A group name is already inside its ask phrase
    # ("whether it contains a nitrile"), and listing it again as an anchor reads
    # as the subject of the question: "Does the nitrile contain a nitrile?"
    anchors = [a for a in ask.get("anchors", []) if a.lower().startswith("atom ")]
    if anchors and not ask.get("underspecified"):
        lines.append("ANCHORS: " + "; ".join(anchors))
    if ask.get("claim"):
        lines.append(
            "THEY BELIEVE: " + ask["claim"]
            + "  (State this as their own belief and ask for it to be checked. "
              "Do not say whether it is right.)")
    if ask.get("constraint"):
        lines.append(
            "THEIR CONSTRAINT: " + ask["constraint"]
            + "  (State the constraint exactly as it is written here — do not "
              "flip it, drop a 'not' or add one — and ask whether this one "
              "meets it. The answer was worked out against this wording.)")
    if ask.get("answerable") is False:
        lines.append("NOTE: they do not know whether this can be answered; ask "
                     "plainly.")
    if ask.get("underspecified"):
        lines.append(
            "IMPORTANT: their message must be ambiguous about which atom they "
            "mean — do NOT name the atom, and do not include ANCHORS. Ask the "
            "question in a way that leaves the atom unsaid.")
    fmt = intent["style"]["format"]
    if fmt != "prose":
        lines.append(f"THEY ALSO ASK FOR THE ANSWER AS: {fmt}")
        if rendered.get("skeleton"):
            lines.append("SCHEMA: " + json.dumps(rendered["skeleton"],
                                                 sort_keys=True))
    return "\n".join(lines)


def _voice_prompt(example, turn: str) -> str:
    intent, rendered = example["intent"], example["render"]
    style = intent["style"]
    # Under the clarification twist the reply answers the *third* turn, not the
    # first. Handed only the underspecified opening, a model re-asks for the
    # clarification it has already been given.
    written = list(rendered["turns"])
    if written and written[0]["text"] is None:
        written[0] = dict(written[0], text=turn)
    if len(written) > 2:
        lines = ["CONVERSATION SO FAR:"]
        lines += [f"  {t['role']}: {t['text']}" for t in written[:-1]]
    else:
        lines = [f"USER SAID: {turn}"]
    lines += [
        "",
        f"DRAFT REPLY: {rendered['reply']}",
        "",
        "STATEMENTS (each must still be stated):",
    ]
    lines += [f"  - {s}" for s in rendered["statements"]]
    # A `decide` reply owes an answer to the constraint, and that answer is not
    # one of the statements — it is about the constraint, not about a fact. A
    # writer told only that "every statement must still be stated" therefore
    # drops it and is doing as it was asked: 16 of 37 non-terse `decide` replies
    # stated every fact and never said whether the molecule qualified. Naming it
    # here is the fix; the accept pass checking for it is the backstop.
    if rendered["ask"].get("constraint") and rendered.get("verdict"):
        lines += ["", "DECISION (must be stated too): "
                  + ("Yes — it meets the constraint."
                     if rendered["verdict"] == "yes"
                     else "No — it does not meet the constraint.")]
    # The same omission with the same cause, on the task whose name is the thing
    # being dropped. The gloss is about the *concept*, not about the molecule, so
    # it is not a statement and a writer told only to state the statements leaves
    # it out — which it did in 92% of v6's explain rows.
    if rendered.get("gloss"):
        lines += ["", "EXPLANATION (must be included too): " + rendered["gloss"]]
    lines += [
        "",
        f"STYLE: {style['register']}; {style['length']}; {style['format']}",
    ]
    if rendered.get("skeleton"):
        lines.append("SCHEMA (keys and types exactly): "
                     + json.dumps(rendered["skeleton"], sort_keys=True))
    if rendered["ask"].get("answerable") is False:
        lines.append("NOTE: the draft declines part of the question. Keep the "
                     "decline; do not answer what it declines.")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """The model's reply, less the wrappers instruct models add anyway."""
    text = (text or "").strip()
    text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    for prefix in ("USER:", "User:", "REPLY:", "Reply:", "ASSISTANT:",
                   "Assistant:", "Message:", "MESSAGE:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) > 1 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pass", dest="which", required=True,
                        choices=("ask", "voice"))
    parser.add_argument("--asks", default=None,
                        help="the ask pass's output; required for --pass voice")
    parser.add_argument("--glob", default="*batch-*.json")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--group", type=int, default=16,
                        help="prompts per generate() call")
    parser.add_argument("--writer", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .check_chat_template import gate

    writer = args.writer or os.path.basename(args.model.rstrip("/"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gate(tokenizer)                       # §9.4's hard gate, before any weights
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    system = ASK_SYSTEM if args.which == "ask" else VOICE_SYSTEM
    asks = {}
    if args.which == "voice":
        if not args.asks:
            raise SystemExit("--pass voice needs --asks pointing at the ask pass")
        for path in sorted(glob.glob(os.path.join(args.asks, "*.jsonl"))):
            with open(path) as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        asks[row["id"]] = row["turn"]

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.batches, args.glob)))
    paths = [p for i, p in enumerate(paths) if i % args.shards == args.shard]
    if not paths:
        raise SystemExit(f"no batch files matched {args.glob} in {args.batches}")

    written = skipped = 0
    for path in paths:
        with open(path) as handle:
            batch = json.load(handle)
        examples = batch["examples"]
        if args.which == "voice":
            examples = [e for e in examples if e["id"] in asks]
            skipped += len(batch["examples"]) - len(examples)

        rows = []
        for start in range(0, len(examples), args.group):
            group = examples[start:start + args.group]
            bodies = [_ask_prompt(e) if args.which == "ask"
                      else _voice_prompt(e, asks[e["id"]]) for e in group]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": body}],
                tokenize=False, add_generation_prompt=True) for body in bodies]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**encoded, do_sample=True,
                                     temperature=args.temperature, top_p=0.95,
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
            for example, sequence in zip(group, out):
                text = _clean(tokenizer.decode(
                    sequence[encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True))
                if not text:
                    continue
                row = {"id": example["id"], "writer": writer}
                row["turn" if args.which == "ask" else "reply"] = text
                rows.append(row)

        name = os.path.basename(path).replace(".json", ".jsonl")
        with open(os.path.join(args.out, name), "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        written += len(rows)
        print(f"{os.path.basename(path)}: {len(rows)}/{len(examples)}",
              flush=True)

    print(f"{written} rows written by {writer} on pass {args.which}"
          + (f", {skipped} skipped for having no turn" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
