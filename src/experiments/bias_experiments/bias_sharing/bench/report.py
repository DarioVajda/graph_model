"""
Turn `bench/speed.py`'s jsonl into the tables quoted in README §6.

    python3 -m src.experiments.bias_experiments.bias_sharing.bench.report
    python3 -m src.experiments.bias_experiments.bias_sharing.bench.report --in <path>.jsonl

Two views:

* **synth** — step time vs node count, one column per `G`, plus the plain-LLM
  floor. `G=0` (legacy per-layer) anchors the speedup column, matching §4.
* **real** — one row per source: the GTLM endpoints against the vanilla LLM at
  the *same* token tensors, so the last column is "what the graph machinery costs
  over a plain LLM at equal sequence length".

A `first_over_median` above 1.5 on any cell is reprinted as a warning: it means a
flex compile or an allocator storm escaped the warm-up passes and the timings for
that cell should not be trusted.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

DEFAULT_IN = os.path.join(os.path.dirname(__file__), "..", "results", "bench", "speed.jsonl")
ARM_ORDER = ["g0", "g1", "g2", "g4", "g8", "g16", "nobias", "llm", "llm_causal"]


def load(path: str) -> tuple[list[dict], list[dict]]:
    timings, fidelity = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            (fidelity if rec.get("kind") == "fidelity" else timings).append(rec)
    # A rerun appends; keep the last record for each cell. The compile mode is part
    # of the key — a comparison run must not clobber the rows it is compared
    # against. `compile_label` is always set by speed.py *now*, but records written
    # before it existed carry only `flex_compile_mode`, which is null for the `llm`
    # arm (a stock LlamaConfig has no such field). Defaulting those to "recipe"
    # silently merged the two runs' floors once already, so an unlabelled row is
    # named loudly rather than guessed.
    unlabelled = []
    latest: dict[tuple, dict] = {}
    for rec in timings:
        label = rec.get("compile_label")
        if label is None:
            label = rec.get("flex_compile_mode")
            if label is None:
                unlabelled.append(rec)
                continue
        latest[(rec["source"], rec.get("n_nodes_target"), rec["arm"], label)] = rec

    if unlabelled:
        print("⚠ dropped rows with no recoverable compile mode — re-run them, or "
              "backfill `compile_label` from the job that wrote them:")
        for rec in unlabelled:
            tag = f'{rec["source"]}' + (f'/N={rec["n_nodes_target"]}'
                                        if rec.get("n_nodes_target") else "")
            print(f'   {tag}/{rec["arm"]}')
    return list(latest.values()), fidelity


def _cell(rec: dict | None) -> str:
    if rec is None:
        return f'{"—":>10}'
    if not rec.get("ok"):
        return f'{rec.get("error", "fail"):>10}'
    return f'{rec["step_ms_median"]:10.1f}'


def compile_mode_table(records: list[dict]) -> None:
    """Autotuned vs plain `torch.compile`: is the long autotune worth its wall time?

    Only flex arms appear — `llm` never compiles, and its row is carried as a
    control: if the floor moved between the two runs, something other than the
    compile mode did.
    """
    modes = {r.get("compile_label", "recipe") for r in records if r["source"] == "synth"}
    if len(modes) < 2:
        return
    order = sorted(modes, key=lambda m: m != "default")     # 'default' first
    by = {(r["n_nodes_target"], r["arm"], r.get("compile_label", "recipe")): r
          for r in records if r["source"] == "synth"}
    nodes = sorted({r["n_nodes_target"] for r in records if r["source"] == "synth"})
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in records
                                        if r["source"] == "synth")]

    for label, field, fmt in (("median step, ms", "step_ms_median", "{:10.1f}"),
                              ("warm-up (compile) wall, s", "warmup_s", "{:10.1f}")):
        print(f"\n## flex compile mode — {label}\n")
        print(f'{"N":>6} {"mode":>26} ' + " ".join(f"{a:>10}" for a in arms))
        for n in nodes:
            for mode in order:
                cells = []
                for a in arms:
                    r = by.get((n, a, mode))
                    cells.append(fmt.format(r[field]) if r and r.get("ok") else f'{"—":>10}')
                print(f'{n:>6} {mode:>26} ' + " ".join(cells))

    if len(order) == 2:
        slow, fast = order[1], order[0]        # autotuned, default
        # Per shape: a pooled ratio divides two medians that may sit on different
        # sequence lengths, which is what produced the spurious 1.38x/1.30x cells
        # in the first pass at N=512.
        print(f'\n  step-time ratio {fast} ÷ {slow}  (>1 = autotuning is worth it)\n')
        print(f'{"N":>6} {"L":>7} ' + " ".join(f"{a:>10}" for a in arms))
        for n in nodes:
            n_lens = sorted({int(L) for a in arms for m in order
                             for L in ((by.get((n, a, m)) or {}).get("per_shape") or {})})
            for L in (n_lens or [None]):
                cells = []
                for a in arms:
                    x = _step(by.get((n, a, fast)), L)
                    y = _step(by.get((n, a, slow)), L)
                    cells.append(f'{x / y:9.3f}x' if x and y else f'{"—":>10}')
                print(f'{n:>6} {str(L or "pooled"):>7} ' + " ".join(cells))


def synth_table(records: list[dict]) -> None:
    rows = [r for r in records if r["source"] == "synth"]
    if not rows:
        return
    # When a compile-mode comparison is present, the main table shows the recipe's
    # own mode; compile_mode_table() owns the side-by-side.
    # The main table always shows the recipe's own compile mode; a comparison run
    # lives only in compile_mode_table(). Grouping on `compile_label` (never null)
    # rather than on the model's resolved mode is what keeps the `llm` rows of two
    # runs from colliding.
    rows = [r for r in rows if r.get("compile_label", "recipe") == "recipe"]
    by = {(r["n_nodes_target"], r["arm"]): r for r in rows}
    nodes = sorted({r["n_nodes_target"] for r in rows})
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]

    # Per-node-count token sampling is i.i.d., so two seeds at the same N can land
    # in different length buckets: N=512 spans L∈{1536,2048}, N=1024 {3072,4096},
    # N=4096 {12288,16384}. Only N=2048 is single-shape. A median pooled across two
    # shapes sits wherever the middle step happens to fall and jumps between arms
    # for reasons that have nothing to do with G — so the pooled column is printed
    # only as a mixed-shape summary, and the per-shape block below is the one to
    # read. Rows without `per_shape` predate that recording and cannot be split.
    mixed = {n for n in nodes
             if len({L for a in arms if (r := by.get((n, a))) and r.get("ok")
                     for L in (r.get("shape", {}).get("distinct_seq_lens") or [])}) > 1}

    print("\n## synth — WebQSP token profile at N nodes, median ms per fwd+bwd step\n")
    print(f'{"N":>6} {"L":>7} {"pad":>5} ' + " ".join(f"{a:>10}" for a in arms))
    for n in nodes:
        shapes = [by[(n, a)] for a in arms if (n, a) in by and by[(n, a)].get("ok")]
        L = f'{shapes[0]["shape"]["seq_len_mean"]:.0f}' if shapes else "?"
        pad = f'{shapes[0]["shape"]["padding_frac"]*100:.0f}%' if shapes else "?"
        flag = " *" if n in mixed else ""
        print(f"{n:>6} {L:>7} {pad:>5} " + " ".join(_cell(by.get((n, a))) for a in arms) + flag)
    if mixed:
        print(f'\n  * pooled across >1 sequence length — NOT comparable across arms; '
              f'use the per-shape block. Affected: {sorted(mixed)}')

    lens = sorted({int(L) for (n, a) in by for L in (by[(n, a)].get("per_shape") or {})})
    if lens:
        print("\n  per sequence length — median ms/step\n")
        print(f'{"N":>6} {"L":>7} ' + " ".join(f"{a:>10}" for a in arms))
        for n in nodes:
            for L in lens:
                cells, any_ok = [], False
                for a in arms:
                    ps = (by.get((n, a)) or {}).get("per_shape", {}).get(str(L))
                    cells.append(f'{ps["median"]:10.1f}' if ps else f'{"—":>10}')
                    any_ok |= ps is not None
                if any_ok:
                    print(f'{n:>6} {L:>7} ' + " ".join(cells))

    # Per shape, for the same reason the table above is: at a mixed-shape N the
    # pooled ratio compares two arms' medians that may sit on different shapes.
    print(f'\n{"":>6} speedup vs G=0 (higher = faster)')
    print(f'{"N":>6} {"L":>7} ' + " ".join(f"{a:>10}" for a in arms))
    for n in nodes:
        n_lens = sorted({int(L) for a in arms
                         for L in ((by.get((n, a)) or {}).get("per_shape") or {})})
        for L in (n_lens or [None]):
            base = _step(by.get((n, "g0")), L)
            if base is None:
                continue
            cells = [f'{base / v:9.2f}x' if (v := _step(by.get((n, a)), L)) else f'{"—":>10}'
                     for a in arms]
            print(f'{n:>6} {str(L or "pooled"):>7} ' + " ".join(cells))

    print(f'\n{"":>6} peak memory, MB')
    print(f'{"N":>6} ' + " ".join(f"{a:>10}" for a in arms))
    for n in nodes:
        cells = []
        for a in arms:
            r = by.get((n, a))
            cells.append(f'{r["peak_mem_mb"]:10.0f}' if r and r.get("ok") else f'{"—":>10}')
        print(f"{n:>6} " + " ".join(cells))


def _step(rec: dict | None, L: int | None) -> float | None:
    """One arm's time at sequence length ``L``, or its pooled median if L is None."""
    if not rec or not rec.get("ok"):
        return None
    if L is None:
        return rec["step_ms_median"]
    ps = (rec.get("per_shape") or {}).get(str(L))
    return ps["median"] if ps else None


# Bias computes per step. The legacy path evaluates the bias 3x per layer: once in
# the forward, once when the decoder layer's checkpoint recomputes, and once more
# for the inner `checkpoint_graph_bias` backward (src/models/dispatch.py:241-245).
# A group costs 3 the same way, except a *single-layer* group, which has no
# follower to rematerialize and so costs 2 — that is why G=L is 2L, not 3L.
def magnetic_computes(arm: str, n_layers: int = 16) -> int | None:
    if arm == "g0":
        return 3 * n_layers
    if not arm.startswith("g"):
        return None
    g = int(arm[1:])
    return (2 if g >= n_layers else 3) * g


def cost_model_table(records: list[dict], n_layers: int = 16) -> None:
    """Fit one per-magnetic-compute cost on (g0, g1); predict the held-out arms.

    `g0` and `g1` differ *only* in magnetic computes — SPD stays per-layer at every
    G — so the pair pins the constant, and g2/g4/g8 are genuine held-out tests of
    the compute counts above. This is the evidence for §1's `3L` and `2L`, and it
    is recomputed here rather than quoted so the README cannot drift from the data.
    """
    rows = [r for r in records
            if r["source"] == "synth" and r.get("compile_label", "recipe") == "recipe"]
    by = {(r["n_nodes_target"], r["arm"]): r for r in rows}
    cells: list[tuple] = []
    for n in sorted({n for n, _ in by}):
        # Only shapes the *fitted pair* both report can be used; deriving the list
        # from all arms picks up shapes that newer rows record and older ones don't,
        # which silently drops every cell. Fall back to the pooled median (flagged
        # `pooled`, and unreliable wherever the N is mixed-shape) when neither
        # endpoint carries per-shape data.
        shared = [set((by.get((n, a)) or {}).get("per_shape") or {}) for a in ("g0", "g1")]
        lens = sorted(int(L) for L in set.intersection(*shared)) if all(shared) else []
        for L in (lens or [None]):
            g0, g1 = _step(by.get((n, "g0")), L), _step(by.get((n, "g1")), L)
            if g0 is None or g1 is None:
                continue
            span = magnetic_computes("g0", n_layers) - magnetic_computes("g1", n_layers)
            k = (g0 - g1) / span
            resid = g1 - magnetic_computes("g1", n_layers) * k
            cells.append((n, L, k, resid, by))

    if not cells:
        return
    held = ["g2", "g4", "g8", "g16"]
    print("\n## cost model — k fitted on (g0, g1), held-out arms predicted\n")
    print(f'{"N":>6} {"L":>7} {"k ms":>8} {"resid ms":>9} ' + " ".join(f"{a:>9}" for a in held))
    for n, L, k, resid, by in cells:
        out = []
        for a in held:
            m = _step(by.get((n, a)), L)
            if m is None:
                out.append(f'{"—":>9}')
                continue
            pred = resid + magnetic_computes(a, n_layers) * k
            out.append(f'{100 * (pred - m) / m:+8.1f}%')
        print(f'{n:>6} {str(L or "pooled"):>7} {k:8.2f} {resid:9.0f} ' + " ".join(out))
    print("  (error of the prediction vs measurement; g16 uses 2L computes, the rest 3G)")


def gather_scaling_table(records: list[dict], n_layers: int = 16) -> None:
    """Split the non-magnetic bias cost into SPD *compute* and the per-score gather.

    `nobias -> g1` prices everything the bias costs at maximal sharing; subtracting
    the 3 magnetic computes leaves SPD plus flex's `node_bias[b,h,node[q],node[k]]`
    lookup. Those two scale differently, and the synthetic grid separates them for
    free: token counts are sampled i.i.d., so each N lands in **two** length
    buckets. Holding N fixed and varying L is a natural experiment —

      * SPD *compute* is O(N²) and does not depend on L at all;
      * the gather runs once per attention score, so it is O(L²) at fixed N.

    Fitting ``cost(L) = A + B·L²`` per N therefore attributes A to SPD and B·L² to
    the gather. B coming out roughly constant across N is the check that the split
    is real: cost per token-pair should not care how many nodes there are.

    Only N values with two shapes can be fitted (two points, two unknowns), so A
    carries no error bars — read B, which is overdetermined across N, as the result.
    """
    rows = [r for r in records
            if r["source"] == "synth" and r.get("compile_label", "recipe") == "recipe"]
    by = {(r["n_nodes_target"], r["arm"]): r for r in rows}
    out = []
    for n in sorted({n for n, _ in by}):
        pts = []
        for L in sorted({int(L) for a in ("g0", "g1", "nobias")
                         for L in ((by.get((n, a)) or {}).get("per_shape") or {})}):
            g0, g1, nb = (_step(by.get((n, a)), L) for a in ("g0", "g1", "nobias"))
            if None in (g0, g1, nb):
                continue
            k = (g0 - g1) / (magnetic_computes("g0", n_layers)
                             - magnetic_computes("g1", n_layers))
            pts.append((L, g1 - nb - magnetic_computes("g1", n_layers) * k, g1))
        if len(pts) >= 2:
            (l1, y1, _), (l2, y2, g1_hi) = pts[0], pts[-1]
            b = (y2 - y1) / (l2 ** 2 - l1 ** 2)
            out.append((n, l2, y1 - b * l1 ** 2, b, b * l2 ** 2, g1_hi))
    if not out:
        return

    print("\n## non-magnetic bias: SPD compute vs the per-score gather\n")
    print(f'{"N":>6} {"L":>7} {"SPD (A) ms":>11} {"B ns/pair":>10} '
          f'{"gather ms":>10} {"of g1":>7}')
    for n, L, a, b, gather, g1 in out:
        print(f'{n:>6} {L:>7} {a:>11.0f} {b * 1e6:>10.1f} {gather:>10.0f} '
              f'{100 * gather / g1:>6.0f}%')
    print("  B ~constant across N is the evidence: gather cost is per token-pair,")
    print("  independent of node count. A is exactly determined (2 points) — treat")
    print("  a small negative A as 'consistent with zero', not as a measurement.")


def decomposition_table(records: list[dict]) -> None:
    """Split the distance from the plain-LLM floor into its three causes.

    ``G`` only shares the *magnetic* term, so sharing cannot touch the rest:

      llm_causal -> nobias   the graph structural mask (causality relaxed to
                             bidirectional across the prefix, so the prefix block
                             is a full square) plus flex replacing fused sdpa.
      nobias     -> g1       everything the bias costs at maximal sharing: the
                             per-layer SPD term, 3 magnetic computes, and flex's
                             per-score gather of node_bias.
      g1         -> g0       the magnetic term alone, unshared (45 extra computes).

    The floor is `llm_causal`, not `llm`: an attention_mask containing zeros makes
    transformers build an explicit 4D mask, which costs sdpa its is_causal fast
    path. Against the padded floor `nobias` came out *faster* than a plain LLM,
    which is not physical. Rows are per sequence length — a median pooled across
    the two length buckets each N lands in is not comparable across arms.
    """
    rows = [r for r in records
            if r["source"] == "synth" and r.get("compile_label", "recipe") == "recipe"]
    by = {(r["n_nodes_target"], r["arm"]): r for r in rows if r.get("ok")}
    floor_arm = "llm_causal" if any(a == "llm_causal" for _, a in by) else "llm"
    cells = []
    for n in sorted({n for n, _ in by}):
        lens = sorted({int(L) for a in ("g0", "g1", "nobias", floor_arm)
                       for L in ((by.get((n, a)) or {}).get("per_shape") or {})})
        for L in (lens or [None]):
            vals = [_step(by.get((n, a)), L)
                    for a in (floor_arm, "nobias", "g1", "g0")]
            if None not in vals:
                cells.append((n, L, *vals))
    if not cells:
        return

    print(f"\n## where the gap to a plain LLM comes from (synth, ms; floor = {floor_arm})\n")
    print(f'{"N":>6} {"L":>7} {"floor":>8} {"+mask/kernel":>13} {"+bias@G=1":>11} '
          f'{"+unshared mag":>14} {"= g0":>9}   {"g1/floor":>9}')
    for n, L, floor, nb, g1, g0 in cells:
        print(f'{n:>6} {str(L or "pooled"):>7} {floor:>8.0f} {nb - floor:>+13.0f} '
              f'{g1 - nb:>+11.0f} {g0 - g1:>+14.0f} {g0:>9.0f}   {g1 / floor:>8.2f}x')


def real_table(records: list[dict]) -> None:
    rows = [r for r in records if r["source"] != "synth"]
    if not rows:
        return
    by = defaultdict(dict)
    for r in rows:
        by[r["source"]][r["arm"]] = r
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]

    print("\n## real cached batches — median ms per fwd+bwd step\n")
    print(f'{"source":>8} {"B":>3} {"L":>7} {"pad":>5} {"backend":>8} '
          + " ".join(f"{a:>10}" for a in arms))
    for source in ("webqsp", "graphqa", "context"):
        if source not in by:
            continue
        arms_here = by[source]
        ok = [r for r in arms_here.values() if r.get("ok")]
        if not ok:
            continue
        ref = ok[0]
        print(f'{source:>8} {ref["batch_size"]:>3} {ref["shape"]["seq_len_mean"]:>7.0f} '
              f'{ref["shape"]["padding_frac"]*100:>4.0f}% '
              f'{arms_here.get("g0", ref)["backend"]:>8} '
              + " ".join(_cell(arms_here.get(a)) for a in arms))

    # Per-shape, because real batches span several sequence lengths and a pooled
    # median is not comparable across arms (see time_arm's note).
    if any("per_shape" in r for r in rows):
        print("\n  per sequence length — median ms/step (mixed-shape pooling is not "
              "comparable across arms)\n")
        print(f'{"source":>8} {"L":>6} ' + " ".join(f"{a:>10}" for a in arms))
        for source in ("webqsp", "graphqa", "context"):
            shapes = sorted({int(L) for r in by.get(source, {}).values()
                             for L in r.get("per_shape", {})})
            for L in shapes:
                cells = []
                for a in arms:
                    ps = (by[source].get(a) or {}).get("per_shape", {}).get(str(L))
                    cells.append(f'{ps["median"]:10.1f}' if ps else f'{"—":>10}')
                print(f'{source:>8} {L:>6} ' + " ".join(cells))

    # Two floors, because they answer different questions. `llm` is fed the same
    # padded tensors GTLM gets, so it is the honest same-input comparison — but an
    # attention_mask with zeros forces transformers to build an explicit 4D mask,
    # and sdpa given one cannot take the is_causal fast path. `llm_causal` drops
    # the mask to recover that path, giving the best case a plain LLM reaches at
    # this sequence length. Ratios against `llm` alone understate GTLM's overhead.
    llm_arms = [a for a in ("llm", "llm_causal") if any(a in v for v in by.values())]
    num = [a for a in arms if a not in ("llm", "llm_causal")]
    for floor_arm in llm_arms:
        note = ("same padded inputs as GTLM"
                if floor_arm == "llm" else "no padding mask — is_causal fast path")
        print(f'\n  GTLM overhead vs `{floor_arm}` ({note})\n')
        print(f'{"source":>8} {"backend":>9} {"floor ms":>9} '
              + " ".join(f"{a:>10}" for a in num))
        for source in ("webqsp", "graphqa", "context"):
            base = by.get(source, {}).get(floor_arm)
            if not base or not base.get("ok"):
                continue
            floor = base["step_ms_median"]
            cells = [f'{r["step_ms_median"] / floor:9.2f}x'
                     if (r := by[source].get(a)) and r.get("ok") else f'{"—":>10}'
                     for a in num]
            print(f'{source:>8} {base["backend"]:>9} {floor:>9.1f} ' + " ".join(cells))


def warnings(records: list[dict]) -> None:
    bad = [r for r in records
           if r.get("ok") and (r.get("first_over_median") or 0) > 1.5]
    if not bad:
        return
    print("\n⚠ cells where the first timed step exceeded 1.5x the median — a compile "
          "or allocator storm escaped warm-up:")
    for r in bad:
        tag = f'{r["source"]}' + (f'/N={r["n_nodes_target"]}' if r["n_nodes_target"] else "")
        print(f'   {tag}/{r["arm"]}: first/median = {r["first_over_median"]:.2f}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="path", default=DEFAULT_IN)
    args = p.parse_args(argv)

    records, fidelity = load(args.path)
    if fidelity:
        f = fidelity[-1]
        print("token fidelity, prefix-node tokens (synthetic vs WebQSP):")
        for name in ("webqsp", "synthetic"):
            s = f[name]
            print(f'  {name:>10}  mean {s["mean"]:.3f}  sd {s["std"]:.3f}  '
                  f'median {s["median"]:.0f}  p90 {s["p90"]:.0f}  max {s["max"]}')
        print(f'  prompt-node mean: webqsp {f["prompt_mean"]["webqsp"]:.2f} '
              f'vs synthetic {f["prompt_mean"]["synthetic"]:.2f}')

    synth_table(records)
    cost_model_table(records)
    gather_scaling_table(records)
    decomposition_table(records)
    compile_mode_table(records)
    real_table(records)
    warnings(records)
    failed = [r for r in records if not r.get("ok")]
    if failed:
        print("\nfailed cells:")
        for r in failed:
            tag = f'{r["source"]}' + (f'/N={r["n_nodes_target"]}' if r["n_nodes_target"] else "")
            print(f'   {tag}/{r["arm"]}: {r["error"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
