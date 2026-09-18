"""
The ChEBI-20 literature-protocol rescorer (`tools/chebi_lit_metrics.py`).

`molecules/PLAN.md` §9's standing rule is that a quantity which is only ever
*read* has no error-detecting surface, and that every instrument gets a test
which fails when the instrument is wrong — **including one that reproduces the
defect**. The defect this module exists to prevent is a caption score quoted
against a published row while it was measured on a subset of the benchmark, so
what is pinned here is the denominator, not the metric arithmetic (that belongs
to NLTK and `rouge_score`, which are the reference implementations and are not
ours to re-verify).

Three properties, and the second is the one that was actually got wrong:

* a report that covers the whole benchmark is unchanged by the ``benchmark``
  denominator, so the accounting is inert where it should be inert;
* an excluded molecule is charged **against its real caption**, not as a pair of
  empty strings — the empty-pair version lets a missing molecule escape BLEU's
  brevity penalty while still charging it an empty n-gram denominator, which
  moves the score in a direction that has nothing to do with the model;
* a report whose excluded-CID list does not account for the shortfall is
  **refused** rather than silently padded, because a denominator nobody can
  enumerate is guesswork.

The captions are three lines of a fake benchmark file, so nothing here reads
ChEBI-20 off disk.
"""

import json

import pytest

from src.experiments.molecules import chebi_lit_metrics as lit


def _write(tmp_path, rows, *, n_benchmark, excluded_cids, split="test"):
    """A `chebi_report.py` pair of artifacts, as this module expects to read them."""
    stem = tmp_path / "run-checkpoint-1-test"
    with open(f"{stem}.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"run_name": "run_s0", "arm": "graph", "seed": 0, "model_name": "m",
            "checkpoint": "c", "split": split, "n_scored": len(rows),
            "n_built": len(rows), "n_benchmark": n_benchmark,
            "coverage": len(rows) / n_benchmark,
            "excluded_reasons": {}, "excluded_cids": excluded_cids}
    with open(f"{stem}.json", "w") as fh:
        json.dump(meta, fh)
    return f"{stem}.json"


def _rows(pairs):
    return [{"cid": str(i), "key": "C", "heavy_atoms": 5, "bucket": "0-20",
             "prediction": p, "target": t,
             "prediction_chars": len(p), "target_chars": len(t)}
            for i, (p, t) in enumerate(pairs)]


@pytest.fixture
def captions(monkeypatch):
    """A fake benchmark file: CID -> caption, patched in place of ChEBI-20's."""
    table = {"900": "The molecule is a very long reference caption about a steroid.",
             "901": "The molecule is another long reference caption about an acid."}
    monkeypatch.setattr(lit, "reference_captions",
                        lambda split, chebi_dir=None: table)
    return table


def test_full_coverage_makes_the_benchmark_denominator_inert(tmp_path, captions):
    """Nothing excluded -> the two denominators are the same set of pairs."""
    rows = _rows([("a caption", "a caption"), ("another", "another one")])
    report = _write(tmp_path, rows, n_benchmark=2, excluded_cids=[])

    built, _ = lit.pairs_for(report, "built")
    benchmark, _ = lit.pairs_for(report, "benchmark")
    assert built == benchmark


def test_an_excluded_molecule_is_charged_against_its_real_caption(tmp_path, captions):
    """The defect: charging ``("", "")`` instead of ``("", <its caption>)``.

    The pair that goes in has to carry the reference, or BLEU's brevity penalty
    never sees the tokens the model failed to produce. Asserting on the pair is
    what makes this test fail if the padding is reverted, which asserting on the
    score alone would not do reliably — the two disagree by an amount that
    depends on the corpus.
    """
    rows = _rows([("a caption", "a caption")])
    report = _write(tmp_path, rows, n_benchmark=3, excluded_cids=["900", "901"])

    pairs, meta = lit.pairs_for(report, "benchmark")

    assert len(pairs) == meta["n_benchmark"] == 3
    charged = pairs[1:]
    assert charged == [("", captions["900"]), ("", captions["901"])]
    assert all(prediction == "" for prediction, _ in charged)
    assert all(target for _, target in charged), \
        "an excluded molecule charged against an empty reference escapes BLEU's " \
        "brevity penalty; it must be charged against its real caption"


def test_charging_the_reference_lowers_the_score(tmp_path, captions):
    """And the direction is down, which is the whole point of the accounting."""
    rows = _rows([("a caption", "a caption")])
    report = _write(tmp_path, rows, n_benchmark=3, excluded_cids=["900", "901"])

    built = lit.score(lit.pairs_for(report, "built")[0])
    benchmark = lit.score(lit.pairs_for(report, "benchmark")[0])

    for metric in ("bleu2", "rouge_l", "meteor"):
        assert benchmark[metric] < built[metric], metric
    # ROUGE-L and METEOR are means over rows, so a third of the split missing
    # costs exactly a third.
    assert benchmark["rouge_l"] == pytest.approx(built["rouge_l"] / 3, rel=1e-9)
    assert benchmark["meteor"] == pytest.approx(built["meteor"] / 3, rel=1e-9)


def test_an_unaccounted_shortfall_is_refused(tmp_path, captions):
    """A denominator nobody can enumerate is guesswork, so it is an error."""
    rows = _rows([("a caption", "a caption")])
    report = _write(tmp_path, rows, n_benchmark=3, excluded_cids=["900"])

    with pytest.raises(SystemExit, match="excluded"):
        lit.pairs_for(report, "benchmark")


def test_a_report_longer_than_the_benchmark_is_refused(tmp_path, captions):
    rows = _rows([("a", "a"), ("b", "b"), ("c", "c")])
    report = _write(tmp_path, rows, n_benchmark=2, excluded_cids=[])

    with pytest.raises(SystemExit, match="whole"):
        lit.pairs_for(report, "benchmark")


def test_an_excluded_cid_with_no_caption_is_refused(tmp_path, captions):
    rows = _rows([("a caption", "a caption")])
    report = _write(tmp_path, rows, n_benchmark=2, excluded_cids=["not-a-cid"])

    with pytest.raises(SystemExit, match="no caption"):
        lit.pairs_for(report, "benchmark")


def test_aggregate_groups_seeds_and_reports_the_count():
    """Three seeds of one arm group; a group of one says so rather than implying three."""
    def run(name, seed, bleu2):
        return {"run_name": name, "seed": seed, "arm": "graph", "model_name": "m",
                "denominator": "benchmark", "coverage": 1.0, "n_scored": 10,
                "n_benchmark": 10, "bleu2": bleu2, "bleu4": 0.1,
                "rouge_l": 0.2, "meteor": 0.3, "rouge_1": 0.4, "rouge_2": 0.5}

    summary = lit.aggregate({
        "a": run("chebi_specialist_graph_s0", 0, 0.40),
        "b": run("chebi_specialist_graph_s1", 1, 0.44),
        "c": run("chebi_specialist_graph_s2", 2, 0.48),
        "d": run("other_arm_s0", 0, 0.10),
    })

    graph = summary["chebi_specialist_graph|benchmark"]
    assert graph["seeds"] == 3
    assert graph["seed_ids"] == [0, 1, 2]
    assert graph["bleu2"]["mean"] == pytest.approx(0.44)
    assert graph["bleu2"]["sd"] == pytest.approx(0.04)
    assert summary["other_arm|benchmark"]["seeds"] == 1
    assert summary["other_arm|benchmark"]["bleu2"]["sd"] == 0.0
