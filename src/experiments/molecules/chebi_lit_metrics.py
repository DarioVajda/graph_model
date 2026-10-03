"""ChEBI-20 caption metrics under the PUBLISHED protocol, from dumped predictions.

`evaluate/captions.py` scores captions with hand-written BLEU/ROUGE/METEOR, and
that is the right instrument for every comparison this project makes between its
own arms: one definition, no dependencies, pinned by hand-computed cases. It is
the wrong instrument for a number placed beside MolT5, for three reasons that
compound:

* **Tokenizer.** Ours is a regex over lowercased words. Theirs is
  ``BertTokenizerFast`` on ``allenai/scibert_scivocab_uncased`` — word pieces, so
  ``dihydroxybenzoate`` is five tokens to them and one to us. That moves every
  n-gram precision.
* **METEOR.** Ours is the exact-match stage alone (`DESIGN.md` §D7.3): no Porter
  stem, no WordNet synonyms. It is a *lower bound* on the published definition,
  and the gap is not a constant.
* **ROUGE-L.** Ours is a corpus-level LCS; theirs is the mean of per-sentence
  F-measures from ``google-research/rouge_score`` over the **raw strings**, which
  the reference implementation reaches without going through its own tokenizer.

So this rescores the JSONL `chebi_score.py` dumps, offline, on CPU,
against the reference implementation
(``blender-nlp/MolT5:evaluation/text_translation_metrics.py``, read 2026-09-17):

    BLEU-2   nltk.translate.bleu_score.corpus_bleu(weights=(.5, .5))
    BLEU-4   the same with weights=(.25, .25, .25, .25)
    ROUGE-L  mean over rows of rouge_scorer.score(prediction, target).fmeasure
    METEOR   mean over rows of nltk meteor_score([target_tokens], prediction_tokens)

with BLEU and METEOR on SciBERT word pieces with ``[PAD]``/``[CLS]``/``[SEP]``
filtered, and ROUGE on the raw strings.

**COVERAGE IS PART OF THE METRIC HERE.** The benchmark's test split is 3,300
molecules and our build keeps fewer — the heavy-atom cap and the disconnected
filter drop the large and multi-fragment end. A mean over the kept rows is a
score on an easier benchmark that shares a name with ChEBI-20. ``--denominator
benchmark`` (the default) charges every excluded molecule as an empty prediction,
which scores zero on all four metrics and is exactly what the system would earn
if it were asked; ``--denominator built`` reproduces the kept-rows-only number so
the two can be printed together and the cost of the cap is visible rather than
assumed.

Usage (CPU, no GPU, no Slurm needed):

    python3 -m src.experiments.molecules.chebi_lit_metrics \\
        src/generalist/results/chebi/*-test.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

#: The reference implementation's tokenizer, and its truncation length.
SCIBERT = "allenai/scibert_scivocab_uncased"
TRUNC = 512

#: Filtered from the word-piece stream before scoring, as the reference does.
SPECIAL = ("[PAD]", "[CLS]", "[SEP]")

METRICS = ("bleu2", "bleu4", "rouge_l", "meteor")


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reports", nargs="+",
                   help="the .json sidecars written by chebi_score.py "
                        "(the .jsonl beside each is what is read).")
    p.add_argument("--denominator", default="benchmark",
                   choices=("benchmark", "built", "both"),
                   help="benchmark charges excluded molecules as misses.")
    p.add_argument("--out", default=None,
                   help="where to write the summary json (default: beside the "
                        "first report).")
    return p.parse_args(argv)


def tokenize(tokenizer, text: str) -> list:
    """SciBERT word pieces with the special tokens filtered, as the reference does."""
    ids = tokenizer(text or "", truncation=True, max_length=TRUNC,
                    padding="max_length")["input_ids"]
    return [t for t in tokenizer.convert_ids_to_tokens(ids) if t not in SPECIAL]


def score(pairs) -> dict:
    """The four published metrics over ``[(prediction, target), ...]``.

    An empty prediction is kept rather than skipped: it is how an excluded
    molecule is charged, and every metric here is defined on it (BLEU's clipped
    counts are zero, ROUGE's F is zero, METEOR's is zero).
    """
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(SCIBERT)
    # No stemmer: the reference constructs `RougeScorer` with the default
    # `use_stemmer=False`, and turning it on raises every ROUGE by a point or
    # two. Matching their call is the whole point of this module.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=False)

    references, hypotheses, meteors, rouges = [], [], [], {"1": [], "2": [], "l": []}
    for prediction, target in pairs:
        gt_tokens = tokenize(tokenizer, target)
        out_tokens = tokenize(tokenizer, prediction)
        references.append([gt_tokens])
        hypotheses.append(out_tokens)
        meteors.append(meteor_score([gt_tokens], out_tokens))
        row = scorer.score(prediction or "", target)
        rouges["1"].append(row["rouge1"].fmeasure)
        rouges["2"].append(row["rouge2"].fmeasure)
        rouges["l"].append(row["rougeL"].fmeasure)

    return {
        "bleu2": corpus_bleu(references, hypotheses, weights=(.5, .5)),
        "bleu4": corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25)),
        "rouge_1": statistics.fmean(rouges["1"]),
        "rouge_2": statistics.fmean(rouges["2"]),
        "rouge_l": statistics.fmean(rouges["l"]),
        "meteor": statistics.fmean(meteors),
        "n": len(pairs),
    }


def reference_captions(split: str, chebi_dir: str = None) -> dict:
    """``{cid: description}`` straight from the benchmark file.

    The build never materialises an excluded molecule, so its reference caption
    has to come from the same place the benchmark's own denominator does — the
    three tab-separated files under ``ChEBI-20_data``. ``chebi_dir`` comes from
    the report's sidecar, so the captions are read from the file that build
    filtered rather than from wherever the default happens to point.
    """
    from .chebi import CHEBI_DIR, reference_captions as _captions

    return _captions(split, chebi_dir or CHEBI_DIR)


def pairs_for(report_path: str, denominator: str) -> tuple:
    """``(pairs, meta)`` for one report, with the excluded rows charged or not.

    **An excluded molecule is charged as an empty prediction against its REAL
    caption**, not as a pair of empties. The difference is not cosmetic: corpus
    BLEU's brevity penalty is computed on summed reference and hypothesis
    lengths, so a pair of empties adds nothing to either and the molecule escapes
    the penalty it should incur, while NLTK's ``modified_precision`` still charges
    an empty n-gram denominator against it. The score that comes out is an
    artifact of the padding rather than a measurement. Charging the true
    reference is what "the system was asked for this caption and produced
    nothing" actually means, and it is the only version of the number that a
    published row is comparable with.

    **How the charge reaches each metric**, because they do not take it the same
    way. ROUGE-L and METEOR are means over rows, so an excluded molecule costs
    exactly its share. BLEU is corpus-level: an empty hypothesis adds nothing to
    the clipped n-gram counts, so the charge arrives through the brevity penalty
    ``exp(1 - r/c)``, where the excluded molecule's reference still lands in ``r``
    and nothing lands in ``c``. That is proportionate at the coverage levels this
    section reports — 411 of 3,300 missing costs BLEU about 13 %, 39 of 3,300
    about 1 % — and it is savage at low coverage, which is correct and is worth
    knowing before reading a number off a build that keeps half the split.
    """
    with open(report_path) as fh:
        meta = json.load(fh)
    rows_path = report_path[:-len(".json")] + ".jsonl"
    pairs = []
    with open(rows_path) as fh:
        for line in fh:
            row = json.loads(line)
            pairs.append((row["prediction"], row["target"]))

    if denominator == "benchmark":
        missing = meta["n_benchmark"] - len(pairs)
        if missing < 0:
            raise SystemExit(
                f"{report_path}: {len(pairs)} scored rows against a benchmark "
                f"split of {meta['n_benchmark']}; the report is not of the whole "
                "split and the denominator cannot be the benchmark's")
        captions = reference_captions(meta["split"], meta.get("chebi_dir"))
        excluded = meta.get("excluded_cids") or []
        if len(excluded) != missing:
            raise SystemExit(
                f"{report_path}: {missing} rows short of the benchmark split but "
                f"{len(excluded)} CIDs listed as excluded; the two have to be the "
                "same molecules or the denominator is guesswork")
        for cid in excluded:
            if cid not in captions:
                raise SystemExit(
                    f"{report_path}: excluded CID {cid!r} has no caption in the "
                    "benchmark file; it cannot be charged against its reference")
            pairs.append(("", captions[cid]))
    return pairs, meta


def main(argv=None) -> int:
    args = _args(argv)
    wanted = (("benchmark", "built") if args.denominator == "both"
              else (args.denominator,))

    out = {}
    for report in sorted(args.reports):
        for denominator in wanted:
            pairs, meta = pairs_for(report, denominator)
            result = score(pairs)
            # `run_name` is derived rather than required: `chebi_score.py`'s
            # sidecar identifies a run by its checkpoint path, and the run's name
            # is that path's parent directory. Falling back keeps this tool able
            # to read both its own sidecars and the older `chebi_report.py` ones.
            run_name = meta.get("run_name") or os.path.basename(
                os.path.dirname(meta["checkpoint"]))
            key = f"{run_name}|{denominator}"
            out[key] = dict(result, run_name=run_name, arm=meta["arm"],
                            seed=meta.get("seed"),
                            model_name=meta.get("model_name"),
                            denominator=denominator,
                            n_scored=meta["n_scored"],
                            n_benchmark=meta["n_benchmark"],
                            coverage=meta.get("coverage"),
                            checkpoint=meta["checkpoint"])
            print(f"{run_name:<40s} {denominator:<9s} "
                  + "  ".join(f"{m}={result[m]:.4f}" for m in METRICS)
                  + f"  n={result['n']}")

    summary = aggregate(out)
    if summary:
        print(f"\n{'group':<34s}" + "".join(f"{m:>18s}" for m in METRICS) + f"{'seeds':>7s}")
        for key in sorted(summary):
            row = summary[key]
            print(f"{key:<34s}"
                  + "".join(f"{row[m]['mean']:>11.4f} ±{row[m]['sd']:.4f}"
                            for m in METRICS)
                  + f"{row['seeds']:>7d}")

    target = args.out or os.path.dirname(os.path.abspath(args.reports[0]))
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, "lit_metrics.json")
    with open(path, "w") as fh:
        json.dump({"per_run": out, "summary": summary}, fh, indent=2, sort_keys=True)
    print(f"\nwrote {path}")
    return 0


def aggregate(per_run: dict) -> dict:
    """Mean and sd over seeds, grouped by ``(run family, denominator)``.

    The family is the run name with its ``_s<seed>`` suffix removed, which is how
    every campaign in this repository names a seed — so three cells of one arm
    group without the caller naming them. A group of one reports ``sd`` 0.0 and
    ``seeds`` 1, and the seed count is printed beside every row precisely so that
    a one-seed row cannot be read as a three-seed one.
    """
    import re

    groups = {}
    for row in per_run.values():
        # A cell is named `<sweep>_000N_seedN`, so the family is the name with
        # that suffix removed; the older `_sN` spelling is still handled.
        family = re.sub(r"_\d{4}_seed\d+$", "", row["run_name"])
        family = re.sub(r"_s\d+$", "", family)
        groups.setdefault(f"{family}|{row['denominator']}", []).append(row)

    out = {}
    for key, rows in groups.items():
        entry = {"seeds": len(rows),
                 "seed_ids": sorted(r["seed"] for r in rows if r["seed"] is not None),
                 "arm": rows[0]["arm"], "model_name": rows[0]["model_name"],
                 "coverage": rows[0]["coverage"],
                 "n_scored": rows[0]["n_scored"],
                 "n_benchmark": rows[0]["n_benchmark"]}
        for metric in METRICS + ("rouge_1", "rouge_2"):
            values = [r[metric] for r in rows]
            entry[metric] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        out[key] = entry
    return out


if __name__ == "__main__":
    raise SystemExit(main())
