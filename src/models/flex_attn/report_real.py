"""
One consolidated table across every ``bench_real`` record.

``bench_real`` writes one ``{experiment}.md`` per experiment, each a stack of
per-arm tables. For the rebuttal we need the opposite view: one row per arm, the
methods as columns, so the TAG and GraphQA regimes can be read against each
other. That is what this produces.

    python -m src.models.flex_attn.report_real \
        --results-dir src/models/flex_attn/results_h100_tag

Records are keyed by (experiment, arm, gradient_checkpointing, pad_mode,
len_bucket_multiple); a re-run of the same key replaces the earlier record, so
re-running an arm to fix it does not leave a stale row behind.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def _key(r: dict) -> tuple:
    """What makes two records the *same* configuration.

    dtype and batch_size belong here: a bf16/B=1 normalization run of an arm is a
    different measurement from its fp32/B=4 paper-setting run, and merging the two
    would silently overwrite one with the other.
    """
    d, c = r["data"], r["config"]
    return (r.get("experiment", "tag"), r["dataset"], c["dtype"], d["batch_size"],
            c["gradient_checkpointing"],
            d.get("pad_mode", "bucket"), d.get("len_bucket_multiple", 512))


def load_records(results_dir: str) -> list[dict]:
    """Every record in the directory, merged by key.

    Records sharing a key are merged **per method**, last write wins. Plain
    last-write-wins on the whole record would let a later run that measured only
    two arms erase the four-arm record it shares a key with — which is exactly
    what happens when a reference arm is added to an already-measured config.
    """
    out: dict[tuple, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("kind") not in ("real_inputs", "tag_real"):
                    continue
                k = _key(r)
                prev = out.get(k)
                if prev is None:
                    out[k] = r
                    continue
                merged = {m["method"]: m for m in prev["methods"]}
                for m in r["methods"]:
                    old = merged.get(m["method"])
                    # Later wins, EXCEPT that a successful measurement is never
                    # replaced by a failed one — a crashed or cancelled job would
                    # otherwise erase good data it happens to share a key with.
                    if old is not None and old.get("ok") and not m.get("ok"):
                        continue
                    merged[m["method"]] = m
                r = dict(r)
                r["methods"] = list(merged.values())
                out[k] = r
    return list(out.values())


METHOD_COLS = ("eager", "flex", "flex-nobias", "sdpa-graphmask", "sdpa")


def render(records: list[dict]) -> str:
    rows = []
    for r in records:
        m = {x["method"]: x for x in r["methods"] if x.get("ok")}
        bad = {x["method"]: x.get("error") for x in r["methods"] if not x.get("ok")}
        d, c = r["data"], r["config"]

        # A stray allocator stall shows up as mean >> trimmed mean. Records
        # written before that bug was fixed carry no trimmed mean at all, so they
        # cannot be checked — mark both cases rather than quietly mixing them in.
        stale = any(x.get("step_ms_trimmed_mean") is None for x in m.values())
        susp = any(x["step_ms_mean"] / x["step_ms_trimmed_mean"] > 1.10
                   for x in m.values() if x.get("step_ms_trimmed_mean"))
        flag = " ⚠stale" if stale else (" ⚠stall" if susp else "")

        def cell(name):
            if name in m:
                return f"{m[name]['step_ms_mean']:.1f}"
            return f"**{bad[name]}**" if name in bad else "—"

        e = m.get("eager", {}).get("step_ms_mean")
        f = m.get("flex", {}).get("step_ms_mean")
        s = m.get("sdpa", {}).get("step_ms_mean")
        real_L = d["real_tokens_total"] / d["n_batches"] / d["batch_size"]
        rows.append((
            real_L,
            r.get("experiment", "tag"), r["dataset"] + flag,
            f"{real_L:.0f}", f"{d['seq_len_mean']:.0f}",
            # Tokens actually pushed through per step. Omitting this is what made
            # a B=4 graphqa row look "shorter" than a B=1 TAG row while doing the
            # same amount of work.
            f"{d['seq_len_mean'] * d['batch_size']:.0f}",
            c["dtype"], str(d["batch_size"]),
            d.get("pad_mode", "bucket"),
            "on" if c["gradient_checkpointing"] else "off",
            cell("eager"), cell("flex"), cell("flex-nobias"),
            cell("sdpa-graphmask"), cell("sdpa"),
            f"{e / f:.2f}×" if e and f else "—",
            f"{f / s:.2f}×" if f and s else "—",
        ))
    rows.sort(key=lambda t: (t[0], t[5], t[6]))   # by true sequence length

    head = ("| exp | arm | L real | L padded | tok/step | dtype | B | pad | gc | "
            "eager | flex | flex-nobias | sdpa+mask | sdpa | "
            "flex vs eager | flex vs sdpa |")
    sep = "|" + "---|" * 16
    body = ["| " + " | ".join(r[1:]) + " |" for r in rows]
    return "\n".join([
        "# Real-input benchmark — consolidated",
        "",
        "Step latency in ms (mean over timed passes), **sorted by true sequence",
        "length**. `L real` is the mean unpadded token count per graph; `L padded`",
        "is the tensor width actually run, after bucketing. `flex vs eager` > 1",
        "means flex is faster.",
        "",
        "**Compare along a row, not down a column.** Rows differ in pad mode and",
        "length ladder, and `pad=batch` rows run the dense arms at their natural",
        "per-batch L, where flex cannot run at all.",
        "",
        "**Absolute times are comparable across rows only at equal `dtype` and",
        "`B`.** GraphQA's paper recipe is fp32/B=4 and TAG's is bf16/B=1, so a",
        "graphqa row can show a larger absolute cost than a TAG row with longer",
        "`L real` — it is pushing `tok/step` tokens through fp32 arithmetic. Use",
        "the bf16/B=1 graphqa rows for any cross-experiment reading; the ratio",
        "columns are within-row and always valid.",
        "",
        "The arms, in order of how much graph machinery they carry:",
        "",
        "| arm | mask | bias | kernel |",
        "|---|---|---|---|",
        "| `sdpa` | plain causal | — | fused flash — the theoretical floor |",
        "| `sdpa+mask` | GTLM's | — | dense SDPA |",
        "| `flex-nobias` | GTLM's | — | flex block-sparse |",
        "| `eager` | GTLM's | full | dense |",
        "| `flex` | GTLM's | full | flex block-sparse |",
        "",
        "`sdpa` is deliberately the most favourable baseline: plain causal is",
        "flash-eligible, and block skipping is precisely what GTLM gives up by",
        "construction. `sdpa+mask` prices the mask shape alone — on Cora GTLM's",
        "mask admits 0.70 of the L×L matrix against plain causal's 0.50.",
        "",
        "⚠stall — some arm's mean exceeds its trimmed mean by >10%, the signature",
        "of a stray allocator stall. ⚠stale — written before that bug was fixed,",
        "so it cannot be checked; treat as indicative only.",
        "",
        head, sep, *body, "",
    ])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="src/models/flex_attn/results_h100_tag")
    p.add_argument("--out", default=None,
                   help="write here instead of stdout (default: {results-dir}/summary.md)")
    a = p.parse_args(argv)

    records = load_records(a.results_dir)
    if not records:
        raise SystemExit(f"no bench_real records under {a.results_dir}")
    text = render(records)
    out = a.out or os.path.join(a.results_dir, "summary.md")
    with open(out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"\n[report] wrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
