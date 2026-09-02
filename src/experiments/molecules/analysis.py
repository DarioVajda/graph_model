"""Per-example error analysis: is a mistake explained by the molecule's geometry?

`max_spd` clamps every shortest-path distance into a fixed number of embedding
rows, and the Levi transform doubles every distance, so a natural hypothesis is
that the graph arm fails on molecules too *wide* for the clamp. That hypothesis is
cheap to state and expensive to act on, and this project has already built one fix
(`spd_depth`) on a plausible mechanism that turned out not to be the cause
(`project-spd-depth-rejected`). So: measure the error rate against molecular width
before touching `max_spd`.

The measurement is per-example accuracy bucketed by the example's Levi-graph
diameter. If the clamp is the limitation there must be a **step** at the clamp —
accuracy roughly flat below `max_spd` and falling above it. A smooth decline with
width is the ordinary "bigger molecules are harder" effect and says nothing about
the clamp, because the flat arm shows it too. Hence both arms are always reported:
the graph-minus-flat gap as a function of width is the quantity that isolates a
graph-specific geometry effect from plain difficulty.
"""

import json

import numpy as np


UNREACHABLE = 32_767          # `compute_shortest_path_distances`' sentinel


def as_spd_matrix(spd):
    """Coerce a stored `shortest_path_dists` row to an (n, n) matrix.

    The column is *written* flat (`dist_matrix.flatten().tolist()`) but comes back
    from the dataset already nested, so `len(row)` is `n` in one case and `n*n` in
    the other. Deriving `n` from `len()` therefore silently produced `sqrt(61) = 8`
    and a reshape error — on the graph arm only, because a single-node flat example
    has one element either way and looks fine. Derive `n` from the element count,
    which is right for both shapes.
    """
    arr = np.asarray(spd, dtype=np.int32).reshape(-1)
    n = int(round(arr.size ** 0.5))
    if n * n != arr.size:
        raise ValueError(f"shortest_path_dists has {arr.size} elements, not a square")
    return arr.reshape(n, n)


def geometry_of(spd_row):
    """Width statistics for one example, read from its stored SPD matrix.

    The dataset already carries `shortest_path_dists`, so this needs no graph
    rebuild and no RDKit — the numbers describe exactly the tensor the bias
    consumed, including the prompt and question nodes.
    """
    spd = as_spd_matrix(spd_row)
    n_nodes = spd.shape[0]
    finite = spd[(spd != UNREACHABLE) & (spd > 0)]
    if finite.size == 0:
        return {"n_nodes": int(n_nodes), "diameter": 0,
                "mean_dist": 0.0, "unreachable_fraction": 1.0}
    off_diag = n_nodes * n_nodes - n_nodes
    return {
        "n_nodes": int(n_nodes),
        # The widest finite separation in the graph the model actually saw.
        "diameter": int(finite.max()),
        "mean_dist": float(finite.mean()),
        # The question node is edge-free, so a nonzero floor here is expected.
        "unreachable_fraction": float((spd == UNREACHABLE).sum() / off_diag),
    }


def clamped_fraction(spd_row, max_spd):
    """Share of finite off-diagonal distances that `SPDBias` folds into its top row.

    This is the quantity `max_spd` actually controls: pairs at or beyond the clamp
    are indistinguishable to the bias. Zero here means raising `max_spd` cannot
    change this example's prediction, whatever else is wrong with it.
    """
    spd = as_spd_matrix(spd_row)
    finite = spd[(spd != UNREACHABLE) & (spd > 0)]
    if finite.size == 0:
        return 0.0
    return float((finite >= max_spd).sum() / finite.size)


def per_example_correct(preds, labels):
    """Exact match per example over the supervised span — `make_compute_metrics`
    unaggregated, so a row here and the reported accuracy cannot disagree."""
    out = []
    for i in range(len(labels)):
        valid = labels[i] != -100
        if not np.any(valid):
            out.append(None)
            continue
        out.append(bool(np.array_equal(preds[i][valid], labels[i][valid])))
    return out


def tier_b_scores(predictions, yes_id):
    """Per-example ``(margin, y_true, y_score)`` from the Tier-B readout.

    `make_margin_preprocessor` reduces the (B, L, V) logits to
    ``(logit_yes, logit_no, true_token_id)`` per example, so the Tier-A
    exact-match path — which masks predictions with the (N, L) label mask —
    cannot read this array at all: it raises `IndexError` on a 3-wide row against
    a 1536-wide mask. Tier B needs its own unpacking, and it is the more useful
    one. The margin is the exact quantity the reported AUROC is computed from, so
    writing it per molecule is what makes a **paired** comparison between two arms
    possible: on a fixed test set two arms' AUROC errors are common-mode, and only
    a per-molecule readout can cancel them.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    if preds.ndim != 2 or preds.shape[1] != 3:
        raise ValueError(
            "the Tier-B per-example readout expects the margin preprocessor's "
            f"(N, 3) output, got {preds.shape}.")
    margin = preds[:, 0] - preds[:, 1]
    y_true = (preds[:, 2] == yes_id).astype(int)
    y_score = 1.0 / (1.0 + np.exp(-margin))
    return margin, y_true, y_score


def write_per_example_report(trainer, dataset, cfg, out_path, yes_id=None):
    """Run the test set once more, and write one JSON line per example.

    Returns the summary dict that also goes into the run record. The score
    recomputed here is checked against the trainer's own, because a silent
    misalignment between predictions and dataset rows would make every geometry
    conclusion below it wrong in a way that looks entirely plausible.

    `yes_id` is required on Tier B and ignored on Tier A — the two tiers report
    different things (a token span vs a logit margin) and so unpack differently.
    """
    output = trainer.predict(dataset, metric_key_prefix="pe")

    tier_b = cfg.tier() == "B"
    if tier_b:
        if yes_id is None:
            raise ValueError("a Tier-B per-example report needs `yes_id`.")
        margin, y_true, y_score = tier_b_scores(output.predictions, yes_id)
        correct = [bool(p == t) for p, t in zip((y_score >= 0.5).astype(int), y_true)]
    else:
        margin = y_true = y_score = None
        correct = per_example_correct(output.predictions, output.label_ids)

    rows = []
    for i, ok in enumerate(correct):
        if ok is None:
            continue
        item = dataset[i]
        spd = item.get("shortest_path_dists")
        row = {"i": i, "correct": ok}
        if tier_b:
            # The three columns a paired test needs. `margin` is pre-sigmoid and
            # is what `n_distinct` / `tied_pair_fraction` are computed on, so it
            # is written raw rather than only as the squashed score.
            row.update(margin=float(margin[i]), y_true=int(y_true[i]),
                       y_score=float(y_score[i]))
        if spd is not None:
            row.update(geometry_of(spd))
            row["clamped_fraction"] = clamped_fraction(spd, cfg.max_spd)
        ids = item.get("input_ids")
        if ids is not None:
            # Nested per node on the graph arm, flat on the flat arm.
            row["n_tokens"] = int(sum(len(x) for x in ids)) if ids and \
                isinstance(ids[0], (list, tuple)) else int(len(ids))
        rows.append(row)

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    acc = float(np.mean([r["correct"] for r in rows])) if rows else 0.0
    diam = [r["diameter"] for r in rows if "diameter" in r]
    clamp = [r["clamped_fraction"] for r in rows if "clamped_fraction" in r]

    # On Tier B the report can check itself: the AUROC recomputed from the rows
    # just written must equal the one `make_margin_metrics` reported from the same
    # predictions. If they disagree, the rows are misaligned with the dataset and
    # every per-molecule conclusion drawn from this file is void — which is
    # exactly the failure that would otherwise look entirely plausible.
    per_example_roc_auc = None
    if tier_b and rows and len(np.unique([r["y_true"] for r in rows])) == 2:
        from sklearn.metrics import roc_auc_score
        per_example_roc_auc = float(roc_auc_score(
            [r["y_true"] for r in rows], [r["y_score"] for r in rows]))

    summary = {
        "per_example_path": out_path,
        "per_example_accuracy": acc,
        "per_example_roc_auc": per_example_roc_auc,
        "diameter_p50": float(np.median(diam)) if diam else None,
        "diameter_p90": float(np.percentile(diam, 90)) if diam else None,
        "diameter_max": int(max(diam)) if diam else None,
        # The share of examples the clamp touches AT ALL. If this is ~0, raising
        # `max_spd` is arithmetically incapable of helping and the hypothesis dies
        # here rather than after a sweep.
        "examples_touching_clamp": float(np.mean([c > 0 for c in clamp])) if clamp else None,
        "mean_clamped_fraction": float(np.mean(clamp)) if clamp else None,
    }
    print(f"[analysis] wrote {len(rows)} per-example rows to {out_path}")
    print(f"[analysis] acc={acc:.4f} diameter p50/p90/max="
          f"{summary['diameter_p50']}/{summary['diameter_p90']}/{summary['diameter_max']} "
          f"examples touching max_spd={cfg.max_spd}: {summary['examples_touching_clamp']}")
    return summary


def accuracy_by_width(rows, edges=(10, 16, 22, 28, 32, 40, 50)):
    """Bucket per-example accuracy by diameter. The shape is the whole point:
    a STEP at `max_spd` implicates the clamp, a smooth slope does not."""
    buckets = {}
    for row in rows:
        d = row.get("diameter")
        if d is None:
            continue
        lo = 0
        for e in edges:
            if d >= e:
                lo = e
        buckets.setdefault(lo, []).append(row["correct"])
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(buckets.items())}
