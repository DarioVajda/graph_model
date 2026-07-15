# KGQA — status summary + CWQ expansion plan

## Where the experiment stands (summary of executed work, through 2026-07-11)

*(This section condenses the previous TODO.md — the full D0–D5 pre-scale plan
with per-arm tables and decision logs is in git history; headline numbers also
live in the README's [Results so far].)*

**Goal:** one graph-native model (GTLM) answering questions directly from
SR-retrieved KG subgraphs, replacing GNN-RAG's GNN-reasoner + LLM-reader
pipeline. The win condition is **retrieval-matched**: beat SR-only GNN-RAG
(~78.9 Hits@1 / 71.3 F1 on WebQSP) at the same inputs and model size.

**What's been established on WebQSP (Llama-3.2-1B):**

- **Data-format v3 was the biggest lever** (+4.6 F1 over the 66.5 plateau):
  newline answer delimiter (commas appear inside 6.5% of golds) + naming v2
  (76 test golds were collapsing as presumed CVTs because their mids were
  missing from the name dict). Best graph run: **72.50 F1 / 78.75 Hits@1** —
  F1 ahead of retrieval-matched GNN-RAG, Hits@1 within noise.
- **Hits@1 structurally under-credits a set generator** (A2 probe): Hit/F1 are
  the honest primary metrics.
- **The flat text-serialization control BEATS the graph arm at 1B**
  (D2/D2b, the central attribution result): collapsed-serialization flat
  **74.93 ± 0.23** test F1 vs graph-native **72.55 ± 0.22** at byte-identical
  supervision (3 seeds each, frozen recipe). The CVT-collapse hypothesis was
  rejected (collapse helps BOTH arms); the deficit is attributable to the
  graph representation/biases at this scale. Both arms beat retrieval-matched
  GNN-RAG; flat beats it on both headline metrics.
- **Instruct backbone + chat formatting is a modest, real win** (D3:
  +0.5 F1 / +1.0 Hits@1; requires weights AND formatting together).
- **Recall-side calibration levers are exhausted** (D0 autopsy + D4 boundary
  loss = null): the ~10-pt under-generation pool folds into the scale
  hypothesis, as does miss_copied (~9 pts, LoRA-capacity-invariant per C1).
- **Frozen recipe for the D5 scale run (3B/8B, both arms):**
  Llama-3.2-3B/8B-Instruct, prompt_style chat, r64, lr 1e-4 (bias_lr 5e-3
  graph), n_max 50, gen 256, dfv3, SR/cap512, 15 ep, full-dev selection.
  D5 has NOT run yet.

**Standing constraints (locked):** natural greedy decoding, no inference-time
interventions; no target-format gimmicks (no answer-count prefixes); retrieval
fixed at SR.

**CWQ groundwork landed 2026-07-11** (see `sr_records.py` and the README's
naming section):

- The sr-cwq int→mid decode is solved and verified: SR's `ent2id.pickle`
  (56.4M entities, irreproducible-from-source enumeration order; obtained via
  SR issue #8's Drive folder) → extracted to `data/cwq_ent_id2mid.txt`;
  16,959/16,960 in-record ground-truth pairs match (the miss is the cid=-1
  sentinel).
- `sr_records.load_sr_records(dataset, split)` normalizes CWQ records to the
  WebQSP shape (trimmed 753k-id decode map auto-built/cached); all record
  consumers (graph + flat data prep, coverage analysis, naming builder) go
  through it. WebQSP passthrough verified byte-identical.
- `RunConfig.dataset` knob (`webqsp`|`cwq`) + `--dataset` flag; cache keys
  `sr-{dataset}_…` (pre-existing keys unchanged).
- Naming v2 rebuilt WITH CWQ: 598,524 → **840,194** entries, append-only
  (existing naming-v2 node texts and caches untouched); CWQ test gold answer
  mids named 98.3% → 99.2%.
- CWQ scale facts: train/dev/test = 27,639/3,519/3,531 questions; subgraphs
  mean 273 tuples (p50 106, max ~3.9k) vs WebQSP's 85; answers median 1 /
  mean 2.0; topic entities mean 1.57; 1–4 hops.

---

# TODO — CWQ expansion plan (2026-07-11)

## Global goals

1. **Every entry point works with both benchmarks** (WebQSP + CWQ, both under
   SR retrieval): graph data prep, flat data prep, training (both arms),
   generative evaluation, coverage analysis, sweep reports, error analysis.
2. **Configurable dataset roles:** train on one, the other, or both together
   (plain concat); evaluate on one, the other, or both — each eval dataset
   always scored **separately**, never pooled.
3. **Per-dataset metric namespaces, always:** every logged metric carries its
   dataset (`eval_webqsp_f1`, `test_cwq_hits1`, …), including single-dataset
   runs. Merging results across benchmarks is meaningless; the naming makes
   it impossible by accident.
4. **Strict metrics become opt-in** (`log_strict_metrics`, default False).
5. **End state of this plan:** the retrieval-matched CWQ comparison (our 1B
   arms vs SR-only GNN-RAG on CWQ) plus a transfer/regression matrix for
   mixed training — with WebQSP reproducibility intact throughout. The 3B/8B
   scale run (D5) stays a separate gate and inherits this machinery.

## Locked decisions (discussion 2026-07-11)

- **Mixed training = plain concat** of the built train splits (CWQ-heavy
  ~10:1). Per-dataset eval is the regression guard; a balancing/mixing knob is
  added only if WebQSP measurably regresses under concat.
- **Checkpoint selection** (multi-eval runs): both dev F1s logged separately;
  the selection metric defaults to their **mean**; a `selection_dataset` knob
  (`webqsp`|`cwq`, default None = mean) overrides it to a single benchmark.
- **Metric names are always dataset-prefixed from now on.** Deliberate naming
  break: legacy unprefixed names in old logs/records read as
  `X == webqsp_X` (all history to date is WebQSP-only).
- **CWQ `max_nodes` = 1024 provisionally**; the final cap is chosen from the
  E1.1 coverage-vs-cost table. WebQSP stays at 512. `n_max` reviewed with the
  same data (CWQ answer sets are small — the 50 default is likely moot).
- **Cache layout stays per-dataset** (one `.gtds`/flat cache per dataset per
  data config; concat happens at load time — no combined artifacts on disk).
- Scope: 1B runs only in this plan; D5 (scale) afterwards, on both benchmarks.

### Addendum (2026-07-12)

- **Recipe updates from the regularization round** fold into every E4 run:
  the graph-bias weight-decay bug is fixed (bias matrices now decayed;
  neutral-to-positive), and `lora_dropout` **0.15** — the only reg lever that
  helped — replaces the 0.05 default.
- **CWQ epochs come down** (user call, 2026-07-12): CWQ is data-rich, so the
  E4.2 probe starts from a SHORT schedule (~5 epochs, not 15) and only extends
  if dev F1 is still climbing.
- **CWQ builds use `versions=1`**: answer sets are near-singleton (median 1),
  so the answer-order augmentation is a no-op on most questions while 8×-ing
  the train build — the ver8 build OOM-killed at 250G. WebQSP keeps ver8
  (its answer sets are large; and its historical caches stay byte-identical).
- **CWQ generative eval needs a ≥80 GB GPU at cap1024** (found by the E3.5
  smokes 2026-07-12): the eager-path generation prefill materializes the
  token-level bias + attention weights at (H=32, q≈kv≈7k tokens) in fp32 —
  ~12 GiB on top of what flex's shape-family buffers already hold; an
  A100-40GB OOMs. B300/A100-80GB are fine. (A chunked-prefill or bf16-bias
  path would lift this; not needed for the current plan.)
- **Training order: CWQ-only first (E4.2–E4.3), mixed after (E4.4)** — the
  retrieval-matched CWQ headline needs the CWQ-only cell (GNN-RAG trains
  per-benchmark, Table 14), and the CWQ schedule must be frozen before a mixed
  run is interpretable. Because the recipe moved (wd fix + lora_dropout 0.15 +
  short schedule), the **webqsp-only transfer-matrix cells get re-run under the
  updated recipe** (~2 GPU-h/run — cheap) rather than reusing D2b history.

## E0 — groundwork ✅ DONE 2026-07-11

- [x] **E0.1** sr-cwq int→mid decode (`data/cwq_ent_id2mid.txt`, verified) +
  provenance documented in `sr_records.py`.
- [x] **E0.2** `sr_records.load_sr_records` adapter; all record consumers
  routed through it; WebQSP passthrough byte-identical.
- [x] **E0.3** `dataset` config knob + per-dataset cache keys (existing keys
  unchanged).
- [x] **E0.4** naming v2 rebuilt with CWQ (append-only, 840,194 entries);
  README updated.

## E1 — CWQ data readiness (CPU-only, no training)

*Everything here is cheap and decision-loaded: it fixes the caps, the eval
denominators, and the baseline number before any GPU hour is spent.*

- [x] **E1.1 coverage-ceiling analysis** ✅ 2026-07-12 — README has the tables
  (builds at `ver1`/`nmax50`, sweep `018_cwq_coverage`). **Decisions: CWQ
  `max_nodes` = 1024** (knee: 512→1024 = +0.9 ceiling pts, 1024→2048 = +0.5 at
  ~2× cost), **`n_max` = 50 stands** (capped == uncapped on every split),
  flat `seq_len` set from the flat cache's token distribution (below).
  Test pipeline ceiling at 1024: **79.9 Hits@1 / 79.0 F1**; retrieval failure
  dominates the gap (682/3531 not retrieved).
  <details>original item follows</details>
  Original: coverage-ceiling analysis at `max_nodes` ∈ {512, 1024, 2048}:
  `--dataset cwq --mode data_prep --analyse-dataset` per cap (data prep once
  per cap; the analysis rides along). Produce the README-style tables:
  uncapped/capped Hits@1 + macro-F1 ceilings, drop decomposition (retrieval
  failure vs cap loss vs CVT collapse vs no-text), built-split token lengths.
  - **Decision out:** final CWQ `max_nodes` (knee of coverage vs quadratic
    SPD/magnetic cost; provisional 1024) and CWQ `n_max`.
  - Also read off: **flat-arm `seq_len`** — CWQ subgraphs are ~3× WebQSP's,
    so the 4096 that covered 99.6% of WebQSP will not cover CWQ; measure the
    serialized-token distribution at the chosen cap and set `seq_len` (or the
    truncation policy) accordingly.
  - Record the tables in the README (new CWQ subsection mirroring WebQSP's).
- [x] **E1.2 eval-parity check vs `rmanluo/RoG-cwq`** ✅ 2026-07-12
  (`check_rog_parity.py`, now scripted for both benchmarks): 3,531/3,531 test
  ids common (= the pinned eval denominator; every test record is answered);
  gold lists identical on 3,471; all 60 diffs benign (57 RoG-duplicate-only —
  ours dedupe, marginally conservative; 3 a `:m.` prefix artifact that
  normalizes away). Documented in the README parity section.
- [x] **E1.3 pin the retrieval-matched CWQ baseline** ✅ 2026-07-12: GNN-RAG
  paper **Table 15 row (d)** (GNN-RAG reading the SR sparse subgraph):
  CWQ **Hit 60.6 / Hits@1 55.6 / F1 53.3** — far below the combined-retriever
  61.7/59.4 the README previously showed (SR *hurts* GNN-RAG on CWQ:
  disconnected sparse subgraphs break their path extraction). Bonus finding:
  the same row's WebQSP F1 is **69.8**, not the 71.3 we'd been citing (71.3 is
  Table 2's dense-retriever figure) — our 72.5 leads the retrieval-matched
  baseline by 2.7, not 1.2. README + SOTA.md updated.

## E2 — multi-dataset config surface

- [x] **E2.1 dataset-role knobs** ✅ 2026-07-12 (`train_datasets`/`eval_datasets`/
  `selection_dataset` + `--dataset` alias kept as sugar; validation in
  `RunConfig.validate`): replace the (day-old) `dataset` field with
  `train_datasets` + `eval_datasets` (each a non-empty subset of
  `{webqsp, cwq}`), CLI flags (comma-separated or repeated), validation, and
  `selection_dataset` (None = mean of eval-dev F1s; must be ∈ `eval_datasets`
  when set). Migrate `--dataset X` → both knobs = `[X]` (no deprecation
  period needed — it shipped yesterday).
- [x] **E2.2 per-dataset data knobs** ✅ 2026-07-12 (`max_nodes`/`n_max`/AND
  `versions` scalar-or-mapping via `_per_dataset`; resolved values in cache
  keys — WebQSP keys byte-stable, test-pinned): `max_nodes` (and `n_max` if E1.1 says
  so) become dataset-resolvable — a scalar applies to all datasets, a
  `{webqsp: 512, cwq: 1024}` mapping resolves per dataset. Cache keys remain
  per-dataset and embed the *resolved* values, so WebQSP keys stay byte-stable
  regardless of CWQ settings. `data_prep` mode builds every (dataset ×
  resolved-config) cache the run references.
- [x] **E2.3 loaders** ✅ 2026-07-12 (concat via `torch ConcatDataset` in both
  arms, single-dataset path byte-identical; eval always per-dataset dicts): train loader concatenates the per-dataset built train
  splits (each with its own `versions` augmentation as built); dev/test
  loaders return one dataset per call — the trainer/evaluator iterates
  `eval_datasets`, never a pooled set. Applies to both arms (graph `.gtds`
  and flat `.jsonl`).

## E3 — trainer, evaluation, logging

- [x] **E3.1 per-dataset evaluation loop + prefixed metrics** ✅ 2026-07-12
  (`PerDatasetEvalMixin` shared by both trainers; str-keyed recursion keeps
  HF's per-dataset dataloader caches; `eval_sel_f1` drives selection):
  `KGQAGraphTrainer.evaluate` (and the flat trainer's equivalent) scores each
  eval dataset separately and logs `eval_{ds}_{metric}` /
  `test_{ds}_{metric}` for every metric currently logged. The selection
  metric is computed as `eval_sel_f1` = mean of the per-dataset
  `eval_{ds}_f1` (or the `selection_dataset` one) and drives
  `metric_for_best_model`. Answer parsing/scoring per dataset is unchanged
  (verbatim GNN-RAG ports).
- [x] **E3.2 strict metrics opt-in** ✅ 2026-07-12 (`log_strict_metrics`,
  default False; code path kept): `log_strict_metrics: bool = False`
  config knob; `evaluate.py` computes/returns the `_strict` variants only
  when enabled (code path kept). Default runs log 3 metrics × datasets
  instead of 6 ×.
- [x] **E3.3 in-training eval cost control** ✅ 2026-07-12 (`eval_indices`:
  gen_eval_samples now takes a FIXED seeded subsample, not first-n; probe/
  headline configs use 256): CWQ dev is 3,519 questions
  (14× WebQSP's 246) — full-dev generative eval per checkpoint is too slow.
  Use the existing `gen_eval_samples` cap for in-training selection on CWQ
  (fixed subsample, seeded for comparability across checkpoints); final
  dev/test scoring stays full-split. Document the chosen cap in the sweep
  configs.
- [x] **E3.4 records + reports** ✅ 2026-07-12 (records carry dataset roles +
  `eval_{ds}_*`/`test_{ds}_*` blocks via `result_block`; generic sweep.report
  renders the prefixed columns; legacy note in README): `runs.jsonl` records carry per-dataset
  result blocks + the dataset-role config; `sweep.report` aggregates the
  prefixed names (one table per eval dataset). Add the legacy-name note
  (unprefixed == webqsp) where old records are read.
- [x] **E3.5 tests + smokes** ✅ 2026-07-12 (unit: tests/test_kgqa_multidataset.py,
  85 total pass; smokes: webqsp graph + flat verified on A100-40, cwq +
  mixed rerun on A100-80GB after the gen-prefill OOM noted above): unit tests for metric prefixing, selection
  mean/override, concat loading, per-dataset knob resolution, strict-off
  default. Then three `--max-steps 4` smoke runs: webqsp-only (the
  regression guard — same pipeline as all history), cwq-only, both+both.

## E4 — CWQ + mixed 1B runs (B300)

*Budget realism: CWQ train is ~10× WebQSP's examples at ~2–3× per-example
cost (bigger graphs / longer flat sequences). A WebQSP graph run ≈ 2 GPU-h;
a CWQ run at 15 epochs would be ~40–60 GPU-h. CWQ is data-rich — epochs come
down. Plan single-seed exploratory runs first, 3 seeds only for headline
cells.*

- [x] **E4.1 data prep** ✅ 2026-07-12 at the E1-chosen caps: CWQ graph + flat
  caches (train/dev/test) built (the E3.5 smokes and the E4.2 probe ran off
  them).
- [x] **E4.2 epoch/convergence probe** ✅ 2026-07-13 (`020_cwq_epoch_probe`,
  6 ep, graph arm): dev-256 F1 peaked **0.583 @ epoch 4.78**, flat through 6
  (last three readings .565/.570/.567) — no extension warranted; already ~5
  pts above the E1.3 baseline. Eval loss RISES from epoch ~2 while F1 climbs
  (overconfidence on confidently-wrong answers): eval loss is not a selection
  signal here — select on generative dev F1 (as the pipeline does). The
  post-training full-split eval was host-RAM OOM-killed (MaxRSS 100.6G vs
  96G; flex shape-buffer growth over full-split shapes suspected) → **CWQ
  runs ask `mem: 200G`**; best ckpt survived
  (`cwq_epoch_probe_0000/checkpoint-14000`), no rescoring needed — E4.3
  supersedes. Schedule: probe validated 6; user call 2026-07-13 raised the
  headline runs to **8 epochs** to stress the short/noisy plateau.
- [x] **E4.3 CWQ-only headline arms** (3 seeds each, frozen 1B recipe
  adapted per E4.2): graph-native + flat (collapsed serialization — the D2b
  winner). Evaluate on CWQ. This is the retrieval-matched CWQ result vs
  E1.3's SR-only GNN-RAG baseline.
  **SUBMITTED 2026-07-13: slurm array job 111530** (`022_cwq_headline`, 6
  tasks, 8 ep, mem 200G, B300; ~24–28h/run). Aggregate:
  `python3 -m sweep.report src/experiments/kgqa/results/cwq_headline`.
  **DONE 2026-07-15** — all 6 runs clean (200G held; full-split test evals
  completed). Test means ± sd over 3 seeds:
  | arm | F1 | Hits@1 | Hit* |
  |---|---|---|---|
  | flat (collapsed) | **.5857 ± .0041** | **.6072 ± .0052** | **.6387 ± .0062** |
  | graph-native | .5401 ± .0099 | .5605 ± .0100 | .5933 ± .0087 |
  | SR-GNN-RAG (Table 15d) | .533 | .556 | .606 (Hit) |
  Readout: flat clearly beats the retrieval-matched baseline (+5.3 F1);
  graph-native is only at baseline parity (+0.7 F1, within seed noise; Hit*
  .593 vs baseline Hit .606). Flat–graph gap **widened** vs WebQSP: 4.6 F1
  (CWQ) vs 2.4 (WebQSP) — the flat-beats-graph dynamic did NOT flip on
  3×-larger subgraphs; it strengthened. Dev-256 in-training bests for graph
  (.580–.597) overstated final test by ~4–6 pts (selection-max inflation);
  flat dev→test drop was ~3 pts. Graph seed spread ~2.4× flat's.
- [ ] **E4.4 mixed-training arms** (concat webqsp+cwq; 3 seeds; both arms):
  evaluate on BOTH benchmarks separately. Fill the transfer matrix —
  {webqsp-only, cwq-only, mixed} × {eval webqsp, eval cwq} — from E4.3 +
  existing WebQSP results + these runs. Decision out: does mixed training
  help, hurt, or wash per benchmark? (If WebQSP regresses under concat,
  revisit the balancing-knob decision.)
- [ ] **E4.5 CWQ error analysis**: rerun the E2-style decomposition
  **Degree slice DONE 2026-07-15** (`degree_slice.py`, per-question rerun of
  all 6 headline ckpts, full test): flat−graph gap is +4.1–4.2 F1 across the
  three lower max_ent_deg quartiles and +6.0 [4.3, 7.7] in the top quartile
  (deg 170–632); same pattern by n_nodes (+3.8→+6.0). So hub-binding adds
  ~2 pts in the hardest quartile (suggestive, ~1.6σ on the diff-of-diffs)
  but the DOMINANT component is a ~4-pt uniform deficit present even on
  tiny low-degree graphs. Per-question F1 corr .86; both-zero 27%, both-
  perfect 36%; flat strictly better 21.8% vs graph 11.1%, flat's wins NOT
  degree-concentrated (median deg 58 vs 61 overall). Hard zeros (all
  seeds): graph 33.1% / flat 30.2%.
  (SR-ceiling floor / set-calibration / miss_copied) on the best CWQ run —
  CWQ is 1–4 hop, so the reasoning-error share is expected to grow; this is
  the number that frames the D5 scale hypothesis on CWQ.
- [ ] **E4.6 gate → D5**: freeze the multi-dataset recipe (datasets, caps,
  schedule, selection) and update the D5 plan to run both arms × both
  benchmarks at 3B/8B. Update README headline tables (CWQ "Ours" column) and
  memory.

## Execution order + rough budget

E1 (CPU, ~an afternoon incl. three CWQ data-prep builds) → E2+E3 (code, no
GPU; the E3.5 smokes are minutes) → E4.1–E4.2 (~1 build + 1 probe run) →
E4.3–E4.4 (headline: 4 cells × 3 seeds; ~150–250 GPU-h depending on the
E4.2 schedule — the dominant cost) → E4.5–E4.6 (analysis, cheap).
