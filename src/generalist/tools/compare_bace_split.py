"""Are the harness's BACE splits the specialist's BACE splits?

The cross-check compares an AUROC from this harness against `026`'s 0.8220. That
comparison is only meaningful if the two are scored on the same molecules, and
`validate`'s partition table says they may not be: it reports BACE as
held_out 1 / test 153 / val 151 / train 1178 over 1485 keys, against the
specialist's 1210 / 151 / 152 over 1513 records.

Both sides call `molecules.data.scaffold_split` over the same `load_tier_b`
records, so every difference comes from what the harness does after the split:

  * it keys on the stereo-free canonical SMILES (`partition_key`), so duplicate
    molecules in the raw file collapse to one key — and BACE has duplicates;
  * §3's cross-source partition gives each key exactly one role over *all*
    sources, so a BACE molecule that ClinTox or ChEBI also claims leaves BACE.

This prints the resulting difference per split rather than assuming it is small,
and separately prints whether any molecule the harness scores at test was a
*training* molecule for the specialist, which is the difference that would
actually invalidate the comparison.

Run it on a compute node — it imports RDKit and reads the built artifacts:

    src/generalist/tools/run_py.sh src/generalist/tools/compare_bace_split.py
"""

import sys


def main() -> int:
    from src.experiments.molecules.data import load_tier_b, scaffold_split
    from src.generalist.adapters.molecules import partition_key

    records, _spec, dropped = load_tier_b("bace")
    keys = [partition_key(r["mol"]) for r in records]
    train_idx, val_idx, test_idx = scaffold_split([r["smiles"] for r in records])
    spec_keys = {"train": {keys[i] for i in train_idx},
                 "val": {keys[i] for i in val_idx},
                 "test": {keys[i] for i in test_idx}}
    print(f"bace: {len(records)} records ({dropped}), {len(set(keys))} distinct keys")
    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        print(f"  specialist {split:<5} {len(idx):>5} rows  "
              f"{len(spec_keys[split]):>5} distinct keys")

    from src.generalist import wiring
    from src.generalist.adapters import molecules as adapter
    from src.generalist.config import RunConfig, load_config_file

    config = RunConfig(**load_config_file(
        "src/generalist/configs/probes/002_cross_check_bace_graph.jsonc")).validate()
    _registry, adapter_config = wiring.build_registry(config)

    for split in ("train", "val", "test"):
        try:
            source = adapter.load("mol/bace", split, config.arm, pass_id=0,
                                  config=adapter_config)
        except Exception as exc:                                  # noqa: BLE001
            print(f"\nharness {split}: not built ({type(exc).__name__}: {exc})")
            continue
        got = set(source.keys())
        mine = spec_keys[split]
        print(f"\nharness {split}: {len(source)} rows, {len(got)} distinct keys")
        print(f"  in both:         {len(got & mine)}")
        print(f"  harness only:    {len(got - mine)}")
        print(f"  specialist only: {len(mine - got)}")
        for other in ("train", "val", "test"):
            if other == split:
                continue
            crossed = got & spec_keys[other]
            if crossed:
                print(f"  !! {len(crossed)} of the harness's {split} keys are in "
                      f"the specialist's {other} split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
