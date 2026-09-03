"""Does the generation prompt end exactly where the answer begins?

The generative scorers cut the prompt node's tokens at `answer_start` — the
first non-`-100` position in `labels` — and generate from there, so a
generative metric is only a measurement of the model if that cut is in the right
place. Off by a token and the model is asked to continue a prompt it never
trained on, and the metric that comes back is a real number about nothing. The
smoke run's `mol/g2s` validity was 0.0 throughout, which is what a floored 1B
model looks like and also what a misplaced cut looks like, and nothing in the
run separated the two.

This separates them, with no model and no GPU. For each example of a built
source it checks that

  * the tokens at and after the cut decode to the stored answer, and
  * the tokens before it do not already contain the answer,

and for a ``smiles`` task it additionally scores the stored answers against
themselves. That oracle pass is the ceiling of the task: `smiles_scores` reads
``roundtrip_match`` by canonicalising the prediction and comparing it to the
target *as stored*, so if the stored answers are not already canonical then no
prediction can score, however good the model is. Both numbers should be 1.0, and
if either is not, the generative metrics for that task are bounded by it.

    src/generalist/tools/answer_boundary.py --task mol/g2s --split test
    src/generalist/tools/answer_boundary.py --task mol/chebi20 --split test \\
        --config src/generalist/configs/004_smoke_probe.jsonc
"""

import argparse
import sys

#: Enough to be conclusive; the check is over identical machinery for every row.
DEFAULT_LIMIT = 200


def main() -> int:
    from src.generalist import wiring
    from src.generalist.adapters import molecules as adapter
    from src.generalist.config import RunConfig, load_config_file
    from src.generalist.evaluate.scorers import answer_start
    from src.generalist.schema import SIDECAR_KEY

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="src/generalist/configs/000_smoke.jsonc")
    parser.add_argument("--task", default="mol/g2s")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config)).validate()
    registry, adapter_config = wiring.build_registry(config)
    spec = registry.get(args.task)
    source = adapter.load(args.task, args.split, config.arm, pass_id=0,
                          config=adapter_config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    n = min(len(source), args.limit)
    print(f"{args.task} {args.split}/{config.arm}: {len(source)} rows, "
          f"checking {n}, answer_kind {spec.answer_kind}")

    exact, leaked, answers = 0, 0, []
    shown = 0
    for i in range(n):
        item = source[i]
        answer = (item.get(SIDECAR_KEY) or {}).get("answer", "")
        answers.append(answer)
        tokens = list(item["input_ids"][int(item["prompt_node"])])
        start = answer_start(item)
        after = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip()
        before = tokenizer.decode(tokens[:start], skip_special_tokens=True)
        if after == answer.strip():
            exact += 1
        elif shown < 3:
            shown += 1
            print(f"  row {i}: after the cut {after!r} != answer {answer!r}")
        if answer and answer.strip() in before:
            leaked += 1
    print(f"  supervised span decodes to the answer: {exact}/{n}")
    print(f"  answer already present in the prompt:  {leaked}/{n}")

    if spec.answer_kind == "smiles":
        from src.generalist.adapters.molecules import smiles_scores

        oracle = smiles_scores(answers, answers)
        print("  oracle (stored answers scored against themselves):")
        for key in sorted(oracle):
            print(f"    {key} = {oracle[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
