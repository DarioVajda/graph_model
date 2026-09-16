"""Build the extra graph-to-SMILES passes an `anneal` fork needs.

`mol/g2s` is a generator, so its passes are files on disk rather than a re-walk
of one corpus, and `load` never generates — an anneal that runs off the end of
the built passes fails at the moment it is wanted rather than building what it
needs. The trunk's `data_prep` builds exactly the passes the *trunk* consumes;
the decay adds ~10 % more examples on top, of which g2s takes its 0.15 share, so
one or two more passes are owed. Arm 2 covered this by building both arms to 14
(`configs/forks/anneal_molecule_generalist.jsonc`); the notation arms of
`MOLECULE_GENERALIST.md` §8.3 came out of their build at 12, and this closes the gap.

Cheap and idempotent: `build` skips every pass already on disk.

    src/generalist/tools/run_py.sh src/generalist/tools/build_g2s_tail.py \\
        --arms flat_selfies,flat_inchi --passes 14
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", required=True)
    parser.add_argument("--passes", type=int, default=14)
    args = parser.parse_args(argv)

    from src.generalist.adapters import molecules

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    config = molecules.MoleculeAdapterConfig().validate()
    print(f"build_version {config.build_version()}")
    print(f"g2s -> {args.passes} passes for {arms}")

    molecules.build(config, tasks=("g2s",), arms=arms, passes=args.passes)

    for arm in arms:
        built = sum(
            1 for _ in __import__("glob").glob(
                config.source_path("g2s", "train", arm, 0).replace(
                    ".p0", ".p*") + ".schema.json"))
        print(f"  {arm}: {built} train passes on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
