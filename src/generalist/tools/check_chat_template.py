"""Check that a writer's chat template tokenizes to the control tokens it names.

A chat template is a string, and a string that names a token the tokenizer does
not have is not an error — it is text. The turn markers then reach the model as
ordinary words, the prompt is not in the shape the instruction tuning was done
in, and nothing anywhere reports a problem: generation still runs, the replies
still parse, and the only symptom is a model that follows instructions worse
than it should for reasons that look like the model's fault.

So before trusting any measurement of how well a writer follows a brief, render
one chat and check that every special marker in the rendered string is a single
token id.

    RUNMOD=src.generalist.tools.check_chat_template src/generalist/tools/run_cli.sh \
        --model /shared/workspace/povejmo/models/hf_models/gemma-4-31B
"""

import argparse
import re
import sys

#: Anything that looks like a control marker in a rendered chat.
MARKER = re.compile(r"<[^<>\s]{1,40}>|<\|[^<>\s]{1,40}\||\|[^<>\s]{1,40}\|>")


def gate(tokenizer) -> None:
    """Refuse a tokenizer whose chat template writes markers it cannot tokenize.

    The failure this exists for is silent: three builds of §9.4 ran a *base*
    checkpoint under a hand-added Gemma-3 template, so `<start_of_turn>` reached
    the model as ordinary text and nothing anywhere said so. A marker that is not
    a single token id is that failure, whatever the path was called.
    """
    chat = [{"role": "system", "content": "SYSTEM_TEXT"},
            {"role": "user", "content": "USER_TEXT"}]
    rendered = tokenizer.apply_chat_template(chat, tokenize=False,
                                             add_generation_prompt=True)
    split = []
    for marker in dict.fromkeys(MARKER.findall(rendered)):
        pieces = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if len(pieces) != 1:
            split.append((marker, len(pieces)))
    if split:
        detail = ", ".join(f"{m!r} -> {n} tokens" for m, n in split)
        raise SystemExit(
            f"chat template gate: {len(split)} marker(s) reach the model as "
            f"plain text ({detail}). This is the checkpoint/template mismatch "
            f"that produced three builds of unusable yield numbers; fix the "
            f"model path or its template rather than passing a flag.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    chat = [{"role": "system", "content": "SYSTEM_TEXT"},
            {"role": "user", "content": "USER_TEXT"}]
    rendered = tokenizer.apply_chat_template(chat, tokenize=False,
                                             add_generation_prompt=True)
    print("=== rendered ===")
    print(repr(rendered))

    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    print(f"\n=== {len(ids)} tokens ===")
    for i in ids:
        print(f"  {i:>7}  {tokenizer.decode([i])!r}")

    print("\n=== markers ===")
    bad = 0
    for marker in dict.fromkeys(MARKER.findall(rendered)):
        pieces = tokenizer(marker, add_special_tokens=False)["input_ids"]
        ok = len(pieces) == 1
        bad += not ok
        print(f"  {marker!r:24} -> {len(pieces)} token(s) {pieces}"
              f"  {'OK' if ok else 'SPLIT — reaches the model as plain text'}")
    print(f"\n{bad} marker(s) are not single tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
