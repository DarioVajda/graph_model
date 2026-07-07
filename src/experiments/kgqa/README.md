# KGQA experiment

Feeding KG subgraphs directly into a single GTLM model, replacing
GNN-RAG's GNN-reasoner + LLM-reader pipeline. Starting on **SR-WebQSP**.

## Goal

Two benchmarks, both under **SR retrieval** (Zhang et al.'s subgraph retriever, the same
inputs GNN-RAG's SR variant consumes): **WebQSP** (1-2 hop) and **CWQ / ComplexWebQuestions**
(1-4 hop). Data pipelines for both exist (`data/sr-webqsp`, `data/sr-cwq`); only WebQSP has
been trained so far.

**The win condition is matched-input, not leaderboard SOTA.** The claim this experiment
exists to test: *at fixed SR inputs and matched model size, one graph-native GTLM ≥
GNN-RAG's GNN-reasoner + LLM-reader pipeline, and > the same base LLM reading the subgraph
as serialized text.* Retrieval is deliberately held fixed — it is a confound, not a
contribution; every point gained by swapping retrievers would be unattributable to the
graph architecture. Concretely, in order:

1. **Beat retrieval-matched SR-GNN-RAG** (~78.9 WebQSP Hits@1; we are at 75.4). This is
   the apples-to-apples pipeline-vs-single-model comparison the setup was designed for.
2. **Beat the text-serialization ablation**: same base LLM, same SR subgraph flattened to
   triples in the prompt, no structural biases. This isolates whether the graph attention
   biases do anything — the result that transfers to the rest of the repo. *(Not yet run.)*
3. **One scale run (3B/8B)** before drawing architecture conclusions: at 1B, plain
   text-RAG (RPO-RAG) gets 69.8 F1 and gains +11.5 F1 going to 8B — part of our gap may
   be reader capacity, not graph handling.
4. Only after 1–3 are won: **demonstrate retrieval portability** by plugging in one
   stronger public retriever (e.g. SubgraphRAG's) as a second input condition. This lifts
   the coverage ceiling and yields competitive headline numbers without claiming
   retrieval as a contribution.

**Non-goal:** chasing the 2026 leaderboard (agentic/interactive-KG systems, GPT-4-class
readers, own retrievers — see [Published SOTA landscape](#published-sota-landscape-as-of-2026-07)).
Those gains come from retrieval quality, multi-turn KG access and reader scale, which are
orthogonal to the GTLM thesis; the SR answer-coverage ceiling (below) bounds our setting
regardless.

The direct predecessor is **GNN-RAG** — GTLM's aim is to match/beat it while collapsing its
GNN-reasoner + LLM-reader into a single model. Published leaderboard numbers below (Hits@1 / F1,
higher = better); the last column is our best run to date.

| Benchmark | Metric | RoG | GNN-RAG | GNN-RAG + RA | Best published (2026) | **Ours — best** |
|---|---|---:|---:|---:|---:|---:|
| **WebQSP** | Hits@1 | 80.0 | 80.6 | 82.8 | 91.6 | **75.4** |
| **WebQSP** | F1 | 70.8 | 71.3 | 73.5 | 88.6 | **66.9** |
| **CWQ** | Hits@1 | 57.8 | 61.7 | 62.8 | 79.6 | — *(not run)* |
| **CWQ** | F1 | 56.2 | 59.4 | 60.4 | 74.2 | — *(not run)* |

- **Ours** = best `baseline`-sweep config (Llama-3.2-1B, k_hop 0, lr 1e-4); test set, verbatim
  GNN-RAG scoring. WebQSP only — no CWQ runs yet. See [Results so far](#results-so-far).
- **GNN-RAG / GNN-RAG+RA / RoG**: from GNN-RAG Table 2 (Mavromatis & Karypis, 2024). Those use
  *combined* GNN retrievers; the **retrieval-matched** SR-only GNN-RAG is ~**78.9** WebQSP Hits@1
  — the fairest single baseline given our SR inputs (still ~3.5 pts above ours).
- **Best published (2026)**: WebQSP Hits@1 = TRACE (91.6, GPT-4.1 agentic); WebQSP F1 and both
  CWQ cells = GraphWalker (Qwen2.5-7B SFT+RL; reports EM, ≈Hits@1). Metric caveats and the full
  method list are in [Published SOTA landscape](#published-sota-landscape-as-of-2026-07) below.
  These methods run their **own retrieval/agentic KG access**, so they are *not* bounded by our
  SR answer-coverage ceiling and upper-bound the field loosely, not our setting.
- Our SR-retrieval **answer-coverage ceiling** (below) caps WebQSP at Hits@1 ≤ 89.7 / F1 ≤ 87.3 —
  the headroom any GTLM on these inputs is competing for. Note current SOTA already presses
  against it: the field has moved past what SR retrieval can support.

### Published SOTA landscape (as of 2026-07)

Numbers and metric caveats below; **what each method actually is and how it works lives in
[SOTA.md](SOTA.md)** (method summaries grouped by family, kept out of this README on purpose).

**Metric warning:** many KGQA papers label as "Hits@1" what is actually **Hit** (any gold
substring appears anywhere in the generated text — the laxest metric; SubgraphRAG's appendix
documents this mislabeling). Columns below use each paper's numbers sorted into the metric they
*actually* compute; blank = not reported. Our own eval logs true Hits@1, Hit and F1 separately,
so compare column-to-column.

| Method (year) | Base LLM | WQSP Hit | WQSP H@1 | WQSP F1 | CWQ Hit | CWQ H@1 | CWQ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| GNN-RAG + RA (2024) | Llama-2-7B | 90.7 | 82.8 | 73.5 | 68.7 | 62.8 | 60.4 |
| GCR (2024) | Llama-3.1-8B + GPT-4o-mini | 92.2 | 82.9¹ | 74.1 | 75.8 | 59.1¹ | 61.7 |
| SubgraphRAG (ICLR '25) | retriever + GPT-4o-mini | 90.1 | 84.3¹ | 77.5 | 62.0 | 60.7¹ | 54.1 |
| KG-R1 (2025) | Qwen2.5-3B, RL | | 82.8 | | | 65.3 | |
| ReKnoS (2025) | Qwen3-235B | | | | | 65.6 | |
| KnowCoder-A1 (2025) | Qwen2.5-Coder-7B, RL | | 80.1 | 77.2 | | 75.7² | 68.3 |
| TRACE (2026) | GPT-4.1, agentic | | 91.6³ | 81.7 | | 76.9³ | 72.9 |
| GraphWalker (2026) | Qwen2.5-7B, SFT+RL | | 91.5² | 88.6 | | 79.6² | 74.2 |
| RPO-RAG (2026) | Llama-3.1-8B | 89.9 | | 81.3 | 72.3 | | 64.5 |
| **RPO-RAG (2026)** | **Llama-3.2-1B** | **82.3** | | **69.8** | **60.3** | | **50.4** |
| PathISE (2026) | Llama-3.1-8B + GPT-4o/4.1 | 91.6 | 86.8 | 81.3 | 71.9 | 63.4 | 61.5 |
| *ChatKBQA (2024)* ⁴ | Llama-2-13B, SP | | *86.4* | *83.5* | | *86.0* | *81.3* |
| *PGDA-KGQA (2025)* ⁴ | Llama-2-7B/13B, SP | | *89.0* | *86.3* | | *87.1* | *83.1* |

¹ GCR/SubgraphRAG H@1 cells are PathISE's re-evaluation (GPT-4o backend) — the papers
  themselves only report Hit + F1.
² EM (exact match of the answer set / RHits@1) — closest to, but not literally, Hits@1.
³ TRACE calls it Hits@1 but sits in the ToG lineage where "Hits@1" is typically Hit;
  unverified, treat as upper bound.
⁴ *Italics* = semantic-parsing methods scored **with oracle (gold) topic entities** — they
  generate a logical form and execute it on Freebase. Not comparable to end-to-end retrieval
  systems (entity linking is given for free); listed because they are the absolute
  benchmark-leaderboard tops, especially on CWQ.

Reading it for our purposes:

- **True-Hits@1 SOTA** (verified metric, no oracle): **PathISE 86.8** WebQSP / **KG-R1 65.3** CWQ
  (GraphWalker's EM 91.5/79.6 likely exceeds these but under a slightly different match rule).
- **F1 SOTA** (no oracle): **GraphWalker 88.6** WebQSP / **74.2** CWQ — a 7B model with SFT+RL
  and agentic multi-turn KG access, i.e. *not* a single-pass reader over a fixed subgraph.
- **The most informative anchor for us is RPO-RAG's Llama-3.2-1B row** (bolded): same base
  model as ours, single-pass RAG-style reading. Hit 82.3 / F1 69.8 WebQSP vs our Hit 80.4 /
  F1 66.9 — we are ~2–3 pts behind a 2026 preference-optimized text-RAG pipeline at equal
  model size, with a different (their own) retriever.
- Everything above ~87 WebQSP Hits@1 exceeds our capped SR ceiling (89.7 uncapped-input /
  87.3 F1) — those systems retrieve better than SR or query the KG interactively. Beating
  them from SR inputs is impossible by construction; the honest target for GTLM is the
  retrieval-matched comparison plus closing the gap to the SR ceiling.

> **NOTE:** This README is a temporary stub holding the answer-coverage measurements
> below. A full rewrite (task description, data-prep pipeline, usage) is pending.

## Running (sweep workflow)

The experiment is a standalone single-run program driven by the generic `sweep`
runner. **Run everything from the repo root** — both the dataset paths and
`results_dir` are repo-root-relative.

```bash
# 1. Scaffold a sweep config, then edit its axes / scalars.
python3 -m src.experiments.kgqa --init my_sweep
#    -> src/experiments/kgqa/configs/my_sweep.jsonc

# 2. Build the .gtds datasets for every data config the sweep references
python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/my_sweep.jsonc

# 3. Train (flip "mode" back to "train").
python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/my_sweep.jsonc

# 4. Aggregate the runs once the (sbatch) jobs finish.
python3 -m sweep.report src/experiments/kgqa/results/my_sweep
```

For a single config / quick iteration, invoke the experiment directly (bypassing
the sweep runner):

```bash
python3 -m src.experiments.kgqa --mode data_prep                 # build this config's datasets
python3 -m src.experiments.kgqa --lora-r 16 --k-hop 2            # train one config
python3 -m src.experiments.kgqa --max-steps 4 --gen-max-samples 8   # smoke test
```

Standalone train runs (no `--runs-jsonl`) append their record to
`src/experiments/kgqa/results/train_runs.jsonl`.

See `configs/example.jsonc` for an annotated template.

## Results so far

All completed sweeps, merged (test set, 1628 questions, sorted by test F1; per-sweep
reports live in `results/<sweep>/report.md`). Fixed across every run: Llama-3.2-1B,
lora_r 16, max_nodes 512, n_max 20, versions 8, one B200. The sweeps also differ in
seed (42 vs 0) and checkpoint selection (`baseline`: 128 dev samples; `relmode_khop`:
full 246-graph dev split), so cross-sweep deltas under ~1 F1 are noise.

| sweep | k_hop | rel_mode | lr | bias_lr | epochs | train time | test F1 | Hits@1 | Hit | F1 strict | dev F1 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | last_1 | 1e-4 | 5e-3 | 15 | 1h 49m | **66.90** | **75.43** | **80.41** | 65.27 | 66.58 |
| relmode_khop | 0 | last_1 | 5e-5 | 1e-2 | 30 | 3h 25m | 66.45 | 73.03 | 79.73 | 64.85 | 67.32 |
| baseline | 0 | last_1 | 3e-4 | 5e-3 | 15 | 2h 02m | 65.88 | 73.65 | 79.91 | 64.35 | 67.81 |
| relmode_khop | 0 | last_2 | 5e-5 | 1e-2 | 30 | 3h 43m | 65.61 | 73.77 | 79.79 | 64.26 | 67.02 |
| relmode_khop | 5 | last_2 | 5e-5 | 1e-2 | 30 | 3h 36m | 61.64 | 69.53 | 75.86 | 59.91 | 63.64 |
| relmode_khop | 5 | last_1 | 5e-5 | 1e-2 | 30 | 3h 28m | 60.51 | 68.61 | 76.66 | 59.08 | 62.05 |
| baseline | 2 | last_1 | 1e-4 | 5e-3 | 15 | 1h 45m | 48.76 | 58.17 | 65.72 | 47.63 | 48.02 |
| baseline | 2 | last_1 | 3e-4 | 5e-3 | 15 | 1h 49m | 47.60 | 56.88 | 64.07 | 47.63 | 46.39 |

Takeaways:

- **k=0 runs plateau at 66.5 ± 0.7 test F1** across both lrs, both horizons, both
  bias lrs and both rel_modes. The 30-epoch runs converge (dev F1 flat from step
  ~7000/9600), so training time is exhausted at this configuration — and the
  plateau is reachable in under 2 GPU-hours.
- **k_hop gate: 0 > 5 >> 2.** k=2 collapses because the prompt node (edges only to
  topic entities) sits 5 *Levi* hops from a 2-KG-hop answer — the generating node
  cannot attend to any answer. k=5 restores reachability (no collapse) but still
  costs 5–6 F1: hard gating hurts even when everything is reachable.
- **rel_mode is a wash at k=0** (mixed signs, within noise) despite 49% of
  questions carrying a within-subgraph relation-text collision under `last_1` —
  the model resolves the ambiguity from graph context. `last_2` costs ~+32%
  tokens (~+15 min here); keep `last_1`.

These bound how well *any* model can do given SR retrieval — the input either contains
the answer or it doesn't. WebQSP reports **macro** metrics (per-question, averaged over
questions), so the macro rows are the operative ceilings; micro is diagnostic only.

Measured from `data/sr-webqsp/{train,dev,test}.json` (data-format v2). Each cell
is **uncapped / capped**:

- **uncapped** — the gold's `kb_id` occurs anywhere in the raw `subgraph.tuples`:
  the pure SR-retrieval ceiling, before any data prep of ours.
- **capped** — the gold survives the actual pipeline graph
  (`select_triples(max_nodes=512)` → Levi → CVT collapse) **and** has a scoreable
  text (its `text`, else a literal `kb_id` — dates/numbers/codes; see
  `answer_text`): exactly the `present_answer_texts` criterion that decides what
  the built `.gtds` can supervise.

"Perfect precision" = model emits only correct, present golds; `N_max=20` =
generation capped at 20 answers.

Reproduce with:
```
python3 -m src.experiments.kgqa --mode data_prep --analyse-dataset
``` 
(see `analyse_dataset.py`; prints these tables and saves `coverage_analysis.json`
next to the built splits).

| Ceiling (uncapped / capped) | **test** (n=1628) | train (n=2826) | dev (n=246) | Bounds |
|---|---|---|---|---|
| ≥1 gold present per question | **91.1% / 89.7%** | 92.6% / 91.0% | 89.8% / 88.2% | **Hits@1** |
| Recall — macro (avg per-q present/total) | 89.2% / 86.4% | 90.5% / 87.6% | 86.7% / 84.2% | per-q recall |
| Recall — micro (Σpresent/Σtotal) | 63.3% / 53.8% | 56.9% / 47.4% | 34.0% / 32.2% | answer-instance recall *(diagnostic)* |
| **F1 — macro**, perfect precision, uncapped | **89.6% / 87.3%** | 91.0% / 88.5% | 87.1% / 85.0% | **macro-F1 (WebQSP metric)** |
| Recall — macro, cap N_max=20 | 86.4% / 83.9% | 87.9% / 85.3% | 85.2% / 83.1% | — |
| F1 — macro, cap N_max=20 | **87.4% / 85.3%** | 88.9% / 86.8% | 86.1% / 84.1% | macro-F1 under our cap |

**Reading it:**
- Operative test ceilings for models trained/scored on the built dataset:
  **Hits@1 ≤ 89.7%**, **macro-F1 ≤ 87.3%** (→ **85.3%** under the N_max=20 cap).
  Against raw SR retrieval: 91.1% / 89.6% (→ 87.4%).
- micro ≪ macro: entirely the enumeration tail (6.8% of questions have >20 golds,
  up to 3688). Micro weights every (q, answer) pair equally so those questions dominate; it is
  **not** a benchmark ceiling — don't optimize for it.
- The N_max=20 cap costs only ~2 macro-F1 points → cheap.
- All rows assume perfect precision, so real achievable numbers are strictly below.
- GNN-RAG's SR Hits@1 (~78.9) sits ~11 pts under even the capped ceiling — that gap is the
  graph-reasoning headroom GTLM targets (genuine reasoning, not retrieval failure).

### Why the two ceilings differ (drop decomposition)

Per question, the first gate that removes its last present gold. **Train** keeps
only the answerable questions (there is nothing to supervise otherwise);
**dev/test keep every answered question** — the non-answerable ones as
empty-target rows that score ~0 — so all eval metrics use the full benchmark
denominators out of the box (no post-hoc correction needed).

| Questions | train | dev | **test** |
|---|---|---|---|
| **answerable** (supervisable; = train's kept rows) | 2573 | 217 | **1460** |
| answer not in SR subgraph (retrieval failure) | 209 | 25 | 145 |
| retrieved, no scoreable answer text | 0 | 0 | 0 |
| lost to the `max_nodes=512` cap | 10 | 0 | 3 |
| lost to CVT collapse | 34 | 4 | 20 |
| **total answered** (dev/test `.gtds` size = eval denominator) | 2826 | 246 | **1628** |

- The `max_nodes=512` cap is nearly free: 3/1628 test questions (0.2 pt). Raising it is
  not the lever — an *uncapped* build recovers only those 3 while the N×N features
  (SPD, magnetic, attention bias) grow quadratically. `max_nodes` stays **512**.
- The bulk (145/1628 test, 8.9%) is SR retrieval failure — the answer is nowhere in the
  retrieved subgraph. GNN-RAG faces the identical bound; unfixable in data prep.
- The old "retrieved but no `text`" bucket (24 test questions) is gone: those golds are
  *literals* (dates, numbers, currency codes) whose `kb_id` is the answer string itself,
  and `answer_text` now falls back to it — the same string the graph shows for the node.
- CVT-collapse losses are answers that are *unnamed* single-parent mediator nodes. Under
  v1 naming (`entities_names.json` only) they would read "unnamed entity" even if kept, so
  they are effectively unanswerable anyway; a naming v2 (broader, answer-independent alias
  source) would both name them and stop their collapse.

### Evaluation parity with GNN-RAG

Benchmark comparability is exact, not approximate:

- `evaluate.py`'s primary metrics are **verbatim ports of GNN-RAG's**
  `llm/src/qa_prediction/evaluate_results.py` (normalized-substring `match`,
  their F1/Hits@1/Hit definitions). Our stricter exact-set variants are logged
  with a `_strict` suffix.
- Gold lists (`graph['gold_answers']`) mirror RoG/GNN-RAG's `answer` lists:
  verified **identical by question id on 1628/1628 test questions** against
  `rmanluo/RoG-webqsp` (their test split is exactly our 1628 answered questions).
  Like theirs, golds with no name anywhere stay as raw-mid placeholders that never
  match — deflating recall for us exactly as it does for them.
- `entities_names.json` (node naming) is itself the file shipped in GNN-RAG's
  `llm/` folder.

### Built-split token lengths

Total tokens per stored example (sum over all node texts of one graph, ≈3 tokens/node;
the train split stores `versions`=8 answer-order augmentations per question):

| Split (examples) | mean | p50 | p75 | p90 | p95 | p99 | max | tokens/node |
|---|---|---|---|---|---|---|---|---|
| train (n=20584) | 331 | 162 | 375 | 923 | 1272 | 1701 | 2414 | 3.04 |
| dev (n=246) | 332 | 155 | 396 | 907 | 1309 | 1606 | 1859 | 3.04 |
| test (n=1628) | 353 | 179 | 393 | 1032 | 1324 | 1656 | 2259 | 3.01 |

Answer-set sizes: median 1, mean 11.2; 52% single-answer, 31% 2–5, 10.5% 6–20, 6.8% >20.
