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

**Pass the same `--config` and `--cell` the run uses.** The adapter's build
directory is keyed by `build_version()`, a hash over the adapter settings, so a
bare `MoleculeAdapterConfig()` builds a correct set of passes into a directory
no run reads — the build reports success and the run fails at the same step it
failed at before. Going through `wiring.build_registry` resolves the same
adapter config the run resolves, and the printed `build_version` must match the
directory in the run's error.

    src/generalist/tools/launch/run_py.sh src/generalist/tools/campaign/build_g2s_tail.py \\
        --config src/generalist/configs/probes/008_molecule_generalist_instruct.jsonc \\
        --cell molecule_generalist_instruct_graph_s0 \\
        --arms graph --passes 32
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="the run's config; without it the "
                                         "adapter defaults are used, which is "
                                         "almost never the directory a run reads")
    parser.add_argument("--cell")
    parser.add_argument("--arms", required=True)
    parser.add_argument("--passes", type=int, default=14)
    args = parser.parse_args(argv)

    from src.generalist.adapters import molecules

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if args.config:
        from src.generalist import wiring
        from src.generalist.config import RunConfig, load_config_file

        run_config = RunConfig(**load_config_file(args.config, args.cell)).validate()
        _registry, config = wiring.build_registry(run_config)
    else:
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
