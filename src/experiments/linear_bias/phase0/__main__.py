"""Phase 0 driver — run the offline linearization measurements over an M-grid.

    python -m src.experiments.linear_bias.phase0 \
        --job src/experiments/bias_sharing/results/002_webqsp_g_sweep/jobs/<...>.sh \
        --run-dir checkpoints/kgqa/002_webqsp_g_sweep_0000_seed0_magnetic_groups0 \
        --out src/experiments/linear_bias/results/p0_webqsp_seed0.json

Reports, per ``M`` and per layer, the R^2 of the best linear head against the
trained MLP head (P0a) and the trained bias's rank spectrum (P0b). P0c lives in
``attention.py`` because it needs a GPU and a full model forward.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .measure import (FitScore, LinearFit, compute_phi, mlp_head, rank_spectrum,
                      spectral_pairs)
from .recipe import (config_from_job, iter_magnetic_batches, load_magnetic_weights,
                     resolve_checkpoint)


def _sample_pairs(n_nodes: int, n_pairs: int, gen: torch.Generator, device):
    """Random off-diagonal node pairs. The diagonal is excluded because
    ``_finalize`` (bias.py:211) zeroes it in the trained bias, and the factorized
    form cannot represent that zeroing anyway (LINEAR_BIAS.md §7.3) — fitting to
    it would price a constraint neither arm actually pays.

    Sampled on the CPU generator (so a run is reproducible from ``--seed``
    independently of the device) and moved once.
    """
    if n_nodes < 2:
        return None
    rows = torch.randint(0, n_nodes, (n_pairs,), generator=gen)
    offs = torch.randint(1, n_nodes, (n_pairs,), generator=gen)
    cols = (rows + offs) % n_nodes
    return rows.to(device), cols.to(device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", required=True, help="sweep job script to replay")
    p.add_argument("--run-dir", required=True, help="checkpoint run directory")
    p.add_argument("--m-grid", default="8,16,32,64,128")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--pairs", type=int, default=20000,
                   help="sampled node pairs per graph per layer (P0a)")
    p.add_argument("--layers", default="", help="comma list; default = all")
    p.add_argument("--spectrum-graphs", type=int, default=4,
                   help="graphs used for the full-matrix rank spectrum (P0b)")
    p.add_argument("--split", default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    m_grid = [int(x) for x in a.m_grid.split(",") if x]
    device = torch.device(a.device)
    t0 = time.time()

    kind, cfg = config_from_job(a.job)
    ckpt = resolve_checkpoint(a.run_dir)
    weights = {l: {k: v.to(device) for k, v in w.items()}
               for l, w in load_magnetic_weights(ckpt).items()}
    layers = ([int(x) for x in a.layers.split(",") if x]
              or sorted(weights))
    print(f"[phase0] device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"[phase0] {kind} recipe: magnetic_m={cfg.magnetic_m} "
          f"magnetic_dim={cfg.magnetic_dim} max_spd={cfg.max_spd}")
    print(f"[phase0] checkpoint: {ckpt}  ({len(weights)} magnetic layers)")
    print(f"[phase0] M grid: {m_grid}   layers: {len(layers)}")

    # One pass over the data; every M reuses the same batches so the grid is a
    # comparison of truncations, not of samples.
    batches = [b.to(device) for b in iter_magnetic_batches(
        kind, cfg, batch_size=a.batch_size, n_batches=a.batches,
        split=a.split, seed=a.seed)]
    print(f"[phase0] loaded {len(batches)} batches in {time.time()-t0:.1f}s")

    # M is per batch, not global: the collator emits min(stored_m, batch max node
    # count), so a batch of small graphs carries fewer eigenvector columns.
    stored_m = max(mb.lambdas.shape[1] for mb in batches)
    n_heads = weights[layers[0]]["proj.2.weight"].shape[0]
    d_mag = weights[layers[0]]["proj.2.weight"].shape[1]

    # Padded eigenvector slots must be exactly zero, or truncation to M > n would
    # silently mix garbage into the fit. This is the leak
    # tests/models/test_v2_ragged_magnetic_padding.py guards; assert it here too
    # because Phase 0 reads the features directly, bypassing the model.
    for mb in batches:
        m_b = mb.lambdas.shape[1]
        for b, n in enumerate(mb.num_nodes.tolist()):
            if n >= m_b:
                continue
            tail = max(mb.V_real[b, :, n:].abs().max().item(),
                       mb.V_imag[b, :, n:].abs().max().item())
            if tail > 0:
                raise RuntimeError(
                    f"Non-zero padded eigenvector columns (max {tail:.3e}) for a "
                    f"graph with {n} nodes; M-truncation would be unsound.")

    results = {
        "meta": {
            "kind": kind, "job": a.job, "checkpoint": ckpt,
            "recipe_magnetic_m": cfg.magnetic_m, "magnetic_dim": cfg.magnetic_dim,
            "stored_m": stored_m, "n_heads": n_heads, "d_mag": d_mag,
            "n_batches": len(batches), "batch_size": a.batch_size,
            "pairs_per_graph": a.pairs, "split": a.split, "seed": a.seed,
        },
        "p0a": {}, "p0b": {},
    }

    for m_trunc in m_grid:
        if m_trunc > stored_m:
            print(f"[phase0] M={m_trunc} exceeds stored m={stored_m}; skipped")
            continue
        tm = time.time()
        trunc = [b.truncate(m_trunc) for b in batches]
        fits = {l: LinearFit(2 * d_mag, n_heads, device=device) for l in layers}
        scores = {l: FitScore(n_heads, device=device) for l in layers}

        def _each_graph(fn):
            """Apply ``fn(layer, feats, target)`` over every (batch, graph, layer).

            Pairs are redrawn identically in both passes because the generator is
            reseeded per pass, so pass 2 scores W on exactly the pairs it was fit
            on (an in-sample R^2, which is what "can this family represent that
            function" asks).
            """
            g = torch.Generator().manual_seed(a.seed)
            for mb in trunc:
                for l in layers:
                    w = weights[l]
                    phi = compute_phi(w, mb.lambdas, mb.num_nodes)  # (B, M, d_mag)
                    for b, n in enumerate(mb.num_nodes.tolist()):
                        pairs = _sample_pairs(int(n), a.pairs, g, device)
                        if pairs is None:
                            continue
                        rows, cols = pairs
                        feats = spectral_pairs(mb.V_real[b], mb.V_imag[b], phi[b], rows, cols)
                        fn(l, feats, mlp_head(w, feats))

        _each_graph(lambda l, f, t: fits[l].update(f, t))
        solved = {l: fits[l].solve() for l in layers}
        # Second pass: residuals measured against the data, never reconstructed
        # from moments (see FitScore's docstring — the moment form cancels away
        # to nonsense when the Gram is rank-deficient).
        _each_graph(lambda l, f, t: scores[l].update(f, t, solved[l]))

        per_layer = {}
        for l in layers:
            r2 = scores[l].r2()
            per_layer[str(l)] = {
                "r2_mean": float(r2.mean()), "r2_min": float(r2.min()),
                "r2_median": float(r2.median()),
                "resid_frac_of_std": float(scores[l].resid_frac_of_std().mean()),
                "bias_std": float(scores[l].target_std().mean()),
                "gram_cond": fits[l].condition_number(),
                "degenerate_heads": int(scores[l].is_degenerate().sum()),
                "n_pairs": scores[l].count,
            }
        agg = torch.tensor([v["r2_mean"] for v in per_layer.values()])
        res = torch.tensor([v["resid_frac_of_std"] for v in per_layer.values()])
        results["p0a"][str(m_trunc)] = {"per_layer": per_layer,
                                        "r2_mean_over_layers": float(agg.mean()),
                                        "r2_worst_layer": float(agg.min()),
                                        "resid_mean_over_layers": float(res.mean())}
        print(f"[phase0] M={m_trunc:>3}  R2 mean {agg.mean():.4f}  "
              f"worst layer {agg.min():.4f}  resid/std {res.mean():.3f}  "
              f"({time.time()-tm:.1f}s)")

    # ── P0b: full-matrix rank spectrum at the recipe's own M ──────────────────
    spec = {}
    mb = batches[0]
    for l in layers:
        w = weights[l]
        phi = compute_phi(w, mb.lambdas, mb.num_nodes)
        rows_all = []
        for b, n in enumerate(mb.num_nodes.tolist()):
            n = int(n)
            if n < 4:
                continue
            ar = torch.arange(n, device=device)
            ii, jj = torch.meshgrid(ar, ar, indexing="ij")
            feats = spectral_pairs(mb.V_real[b], mb.V_imag[b], phi[b],
                                   ii.reshape(-1), jj.reshape(-1))
            bias = mlp_head(w, feats).reshape(n, n, n_heads)
            # cap = 2M for THIS graph's eigenvector count, which is what the
            # factorization would actually be able to reach on it.
            cap = 2 * min(mb.lambdas.shape[1], n)
            for h in range(n_heads):
                rows_all.append(rank_spectrum(bias[:, :, h], cap=cap))
            if len(rows_all) >= a.spectrum_graphs * n_heads:
                break
        if rows_all:
            spec[str(l)] = {
                k: float(torch.tensor([r[k] for r in rows_all], dtype=torch.float64).mean())
                for k in ("energy_in_cap", "rank_90", "rank_99")
            } | {"n_singular": rows_all[0]["n_singular"], "cap": rows_all[0]["cap"]}
    results["p0b"] = spec

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[phase0] wrote {a.out}  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
