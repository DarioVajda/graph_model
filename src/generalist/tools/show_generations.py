"""Print what a checkpoint actually generates for a task. Nulls, verified.

A generative metric of exactly 0.0 has two causes that look identical in a
metrics file: a model with nothing to say, and a scorer comparing the wrong two
strings. `answer_boundary.py` rules out the second for the *prompt*; this rules
it out for the *prediction*, by running the same `generate_predictions` the
scorers run and printing the pairs it produced.

The smoke checkpoint scored `bleu2`, `bleu4`, `rouge_l` and `meteor` all exactly
0.0 on 32 ChEBI captions. Exactly zero ROUGE-L across 32 long English captions is
not what a bad model usually looks like — a bad model still emits "the" and
"molecule" — so it is worth a look rather than a shrug.

    GPU=1 src/generalist/tools/show_generations.py --task mol/chebi20 \\
        --checkpoint <ckpt> --config src/generalist/configs/004_smoke_probe.jsonc
"""

import argparse
import sys


def main() -> int:
    from src.generalist import wiring
    from src.generalist.config import RunConfig, load_config_file
    from src.generalist.evaluate.scorers import eval_indices, generate_predictions
    from src.generalist.fork import load_start_weights

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config)).validate()
    registry, adapter_config = wiring.build_registry(config)
    spec = registry.get(args.task)

    run = wiring.build_run(config, output_dir=args.checkpoint + "/scratch",
                           fire_validators=False)
    load_start_weights(run.trainer, args.checkpoint)

    from src.generalist.adapters import molecules as adapter

    source = adapter.load(args.task, args.split, config.arm, pass_id=0,
                          config=adapter_config)
    indices = eval_indices(len(source), args.n)
    predictions, targets = generate_predictions(
        run.model, run.tokenizer, run.collator, source, indices,
        max_new_tokens=args.max_new_tokens or (spec.max_new_tokens or 64),
        device=run.device)

    for i, (prediction, target) in enumerate(zip(predictions, targets)):
        print(f"\n--- {i} (row {indices[i]}) ---")
        print(f"  target     [{len(target)} chars]: {target[:300]!r}")
        print(f"  prediction [{len(prediction)} chars]: {prediction[:300]!r}")
    empty = sum(1 for p in predictions if not p.strip())
    print(f"\nempty predictions: {empty}/{len(predictions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
