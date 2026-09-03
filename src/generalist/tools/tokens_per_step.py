"""What `tokens_per_step` gives a target examples/step on a built arm?

D4.4 sets the batch in tokens, not in examples, so that a mixture whose tasks
render to very different lengths still takes a roughly constant amount of
compute per step. That is the right default, and it is exactly wrong when the
thing being compared is two arms of the same task: a flat BACE example is a
SMILES string and a graph BACE example is a Levi graph with a prefix per node,
so a shared token budget hands the flat arm several times the batch and the
comparison stops being a comparison.

This prints each mixture task's measured `mean_tokens` for the config's arm,
the share-weighted mean `resolve` actually divides by, and the
`tokens_per_step` that lands on a stated examples/step. Run it once per arm
after `data_prep` and write the answer into the config:

    src/generalist/tools/tokens_per_step.py --config <cfg> --examples-per-step 32

`mean_tokens` is a property of the built data — the adapter measures it during
`data_prep` — so this is only answerable after a build, and it is why the flat
twin of a graph config cannot state its token budget until its arm exists.
"""

import argparse
import sys


def main() -> int:
    from src.generalist import wiring
    from src.generalist.config import RunConfig, load_config_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--examples-per-step", type=float, default=32.0)
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config)).validate()
    registry, _ = wiring.build_registry(config)

    missing = wiring.unbuilt_tasks(registry, config)
    if missing:
        print(f"not built for arm {config.arm}: {', '.join(missing)}")
        return 1

    mixture = wiring.resolve_mixture(config, registry)
    print(f"config {args.config}")
    print(f"  arm {config.arm}, mixture {config.mixture}")
    print(f"  {'task':<24} {'share':>8} {'mean_tokens':>12}")
    for entry in sorted(mixture.entries, key=lambda e: -e.share):
        print(f"  {entry.name:<24} {entry.share:8.4f} {entry.mean_tokens:12.1f}")
    print(f"  weighted mean_tokens {mixture.mean_tokens:.2f}")
    print(f"  at tokens_per_step {mixture.tokens_per_step}: "
          f"{mixture.examples_per_step:.2f} examples/step, {mixture.steps} steps")

    want = args.examples_per_step
    print(f"\n  for {want:g} examples/step: "
          f"tokens_per_step {int(round(want * mixture.mean_tokens))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
