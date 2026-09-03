"""Per-molecule detail behind `perm_spread`'s one `margin_spread_max` number.

The validator reports the *max* over molecules of the margin's spread across
relabellings, and compares it against the bf16 margin quantum. That is the right
assertion to run every time — Property 1 says the spread is zero on the graph
arm, and a max is what catches the one molecule that breaks it — but when it
fails, one number cannot say whether the cause is structural (the model is
reading something order-dependent) or numerical (float non-associativity in the
attention reductions, scaled up by a confident model's larger logits).

The two look different here. Structural non-equivariance concentrates: a handful
of molecules move a lot, the rest not at all, and the movement does not track
the margin's magnitude. Float noise spreads: most molecules move by a quantum or
two, none by much more, and the spread grows with |margin| because a relative
error on a bigger number is a bigger absolute one.

``--control`` is the decisive form. It holds the permutation at 0 — every view is
the molecule exactly as built, no relabelling anywhere — and varies only the
order the molecules are batched in, which changes each one's padding and its
neighbours in the reduction. Anything the control measures is float noise by
construction, because the model was handed identical inputs. A permuted spread
that does not exceed the control's is not evidence against Property 1.

    GPU=1 src/generalist/tools/run_py.sh src/generalist/tools/perm_spread_detail.py \\
        --config src/generalist/configs/002_cross_check_bace_graph.jsonc \\
        --checkpoint <ckpt> --task mol/bace [--control]
"""

import argparse
import sys


def main() -> int:
    import numpy as np

    from src.generalist import wiring
    from src.generalist.adapters import molecules as adapter
    from src.generalist.config import RunConfig, load_config_file
    from src.generalist.evaluate.builtin import _PermutedSource
    from src.generalist.evaluate.scorers import eval_indices, teacher_forced
    from src.generalist.fork import load_start_weights
    from src.experiments.molecules.evaluate import (
        answer_token_ids, make_margin_preprocessor,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="mol/bace")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-molecules", type=int, default=200)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--control", action="store_true",
                        help="identity permutation throughout; vary only the "
                             "batching order")
    args = parser.parse_args()

    config = RunConfig(**load_config_file(args.config)).validate()
    _registry, adapter_config = wiring.build_registry(config)
    run = wiring.build_run(config, output_dir=args.checkpoint + "/scratch",
                           fire_validators=False)
    load_start_weights(run.trainer, args.checkpoint)

    source = adapter.load(args.task, args.split, config.arm, pass_id=0,
                          config=adapter_config)
    indices = eval_indices(len(source), args.n_molecules)
    yes_id, no_id = answer_token_ids(run.tokenizer)
    preprocess = make_margin_preprocessor(yes_id, no_id)

    rng = np.random.default_rng(0)
    margins = []
    for p in range(args.n_permutations):
        # In control mode the molecule is always permutation 0 — the item as
        # built — and `order` is what changes: which eight go in a batch together
        # and therefore what each one is padded to.
        order = (rng.permutation(len(indices)) if args.control and p
                 else np.arange(len(indices)))
        view = _PermutedSource(source, config.arm, 0 if args.control else p,
                               run.tokenizer, [indices[j] for j in order])
        preds, _labels = teacher_forced(run.model, run.collator, view,
                                        range(len(view)), device=run.device,
                                        batch_size=args.batch_size,
                                        preprocess=preprocess)
        row = np.empty(len(indices), dtype=np.float64)
        row[order] = preds[:, 0] - preds[:, 1]
        margins.append(row)
    margins = np.asarray(margins, dtype=np.float64)          # (perms, molecules)

    spread = margins.max(axis=0) - margins.min(axis=0)
    magnitude = np.abs(margins).mean(axis=0)
    quantum = 0.125                                          # bf16 at this scale

    print(f"arm={config.arm} molecules={margins.shape[1]} "
          f"views={margins.shape[0]} "
          f"mode={'control (identity, re-batched)' if args.control else 'permuted'}")
    print(f"|margin| mean {magnitude.mean():.4f}  max {magnitude.max():.4f}")
    print(f"spread   mean {spread.mean():.4f}  max {spread.max():.4f}  "
          f"({spread.max() / quantum:.1f} quanta)")
    hist = np.bincount(np.rint(spread / quantum).astype(int))
    for q, count in enumerate(hist):
        if count:
            print(f"  spread = {q} quantum(a): {count} molecule(s)")
    moving = spread > quantum
    print(f"molecules above one quantum: {int(moving.sum())} / {len(spread)}")
    if moving.any():
        ratio = spread[moving] / np.maximum(magnitude[moving], 1e-9)
        print(f"  their |margin| mean {magnitude[moving].mean():.4f} vs "
              f"{magnitude[~moving].mean():.4f} for the rest")
        print(f"  spread/|margin| mean {ratio.mean():.4f} max {ratio.max():.4f}")
    order = np.argsort(-spread)[:10]
    print("worst molecules (row, spread, |margin|):")
    for j in order:
        print(f"  {indices[j]:>6d}  {spread[j]:.4f}  {magnitude[j]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
