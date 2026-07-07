# TODO: close the gap to retrieval-matched SR-GNN-RAG (~78.9 Hits@1 / 71.3 F1)

Roadmap written 2026-07-07, after the `baseline` + `relmode_khop` sweeps and the
error analysis. Everything needed to execute it is named here or in the repo —
no other context required.

## Context: where we are and how we got here

Best converged run = **test F1 66.5 / Hits@1 73.3 / Hit 79.9**
(`relmode_khop_0000`: k_hop 0, rel_mode last_1, 30 epochs; best checkpoint at
`checkpoints/kgqa/relmode_khop_0000_k_hop0_rel_modelast_1/checkpoint-8000`).
Metric names: repo `test_hits1` = papers' Hits@1 (first parsed answer),
`test_hit_star` = papers' "Hit" (any gold matched); scoring is a verbatim port
of GNN-RAG's — see README → "Evaluation parity with GNN-RAG".

Measured and EXHAUSTED (do not revisit for accuracy — README → Results table):
training time (30-epoch runs converge, dev F1 flat from step ~7000/9600),
rel_mode last_2 (wash at k=0), k-hop gating (0 > 5 ≫ 2; k=2 collapses because
the prompt node sits 5 *Levi* hops from a 2-KG-hop answer), lr in [5e-5, 3e-4].
The k=0 plateau is 66.5 ± 0.7 F1 across five configurations, reachable in
<2 GPU-hours (15 epochs).

**Error-analysis artifacts** (all under `results/error_analysis/`):

- `dump_generations.py` — GPU script; regenerated all dev+test questions from
  the checkpoint above (greedy, eager, 128 new tokens, prompt truncated at
  "Answer:"). Also the reference recipe for LOADING a trained checkpoint:
  `GTLMLlamaConfig.from_pretrained(ckpt)` → backbone
  `GTLMLlamaForCausalLM.from_pretrained(model_name, config=...)` →
  `PeftModel.from_pretrained(model, ckpt)` →
  `src.models.io.load_bias_parameters(model, ckpt)`.
- `generations.jsonl` — 1874 rows: `{split, idx, question, golds, gen_text}`;
  `idx` indexes the built split (`TextGraphDataset.load(<cache>/<split>)`).
- `analyze_errors.py` — offline; rejoins rows with graphs, scores, buckets.
- `buckets.jsonl` — per-question rows incl. `bucket`, parsed `preds`,
  `preds_in_graph`, `n_gold_present`, `f1/hits1/hit`.
- `report.md` — the full findings.

The 33.4 missing macro-F1 points on test decompose as:

| bucket | F1 pts | fix |
|---|---:|---|
| retrieval_miss (205 qs; ~60 are OUR pipeline's losses — see A1) | 10.75 | naming v2 + cap audit (→ Phase B) |
| set calibration (partial_precision 5.4 / partial_both 5.3 / partial_recall 2.6) | 13.31 | A2 diagnostic → training-side fix; n_max/gen-len arms (→ B5) |
| miss_copied (copied real node texts, wrong ones = reasoning) | 8.11 | capacity probe (→ Phase C) |
| miss_fabricated (true hallucination) | 1.35 | nothing — non-problem, validates no-bolt-ons |

The "~60" derivation: 205 test questions have no gold generatable from the
BUILT graph, but only 145 are raw SR-retrieval failures (README →
answer-coverage ceilings), so ~60 lose the answer inside OUR pipeline
(cap / CVT collapse / naming).

Two orthogonal findings: the **comma-in-gold bug** — 6.5% of test questions
have a comma inside a gold answer ("Return J. Meigs, Jr."); targets are
comma-joined and predictions comma-split, so those sets are unlearnable AND
unparseable; mean F1 there 0.477 vs 0.666 overall ≈ ~1.2 F1 pts (→ B1) — and
the **ordering loss** — 107 qs score Hit but not Hits@1, the whole
Hits@1↔Hit gap (measured, not "fixed": see A2(b)).

**Standing constraints** (user decisions, do not relitigate): stay at
Llama-3.2-1B; no bolt-ons — headline numbers come from plain autoregressive
generation with NO test-time post-processing (likelihood thresholds/reranking
are diagnostics only, per A2); retrieval stays SR — the win condition is the
retrieval-matched comparison (SR-GNN-RAG ~78.9 Hits@1), not the leaderboard.

**Ops**: run everything from the repo root with `.venv` activated. Sweeps:
`python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/<name>.jsonc`,
then `python3 -m sweep.report src/experiments/kgqa/results/<name>`. One-off GPU
jobs follow the sbatch pattern in `results/error_analysis/job.sh` + the
submit line in git history (container srun → `sweep/slurm_launch.sh` with
ABSOLUTE paths). Built datasets live in `processed_datasets/`; the current
(v2) cache is `sr-webqsp_meta-llama-Llama-3.2-1B_vlast_1_cap512_nmax20_ver8_
spd64_magq0.25m128_len1024_rcm1_seed42_dfv2`.

---

## Phase A — probes, no training (~1 day, <1 h GPU)

- [ ] A1 **Audit the ~60 pipeline-lost questions** (offline, no GPU): take the
  `bucket == "retrieval_miss"` rows of `results/error_analysis/buckets.jsonl`,
  join back to raw `data/sr-webqsp/test.json` (JSONL; match on the `question`
  string, or rebuild the idx→record mapping the way `process_dataset.py`
  iterates the file). For each question whose gold `kb_id` IS in raw
  `subgraph.tuples` but has no generatable text in the built graph, classify:
  (i) gold node dropped by the `max_nodes=512` cap (`select_triples`),
  (ii) merged away by CVT collapse, or (iii) node present but its text is
  "unnamed entity" / a literal kb_id (name missing from `entities_names.json`).
  Output: counts per cause → decides what naming v2 (B2) actually implements.
- [ ] A2 **Likelihood scoring — DIAGNOSTIC ONLY** (decided 2026-07-07:
  test-time thresholds/reranking conflict with the native-generation thesis
  and will NOT become an inference mode; run once to learn, not to ship).
  One GPU job, modeled on `dump_generations.py` (same checkpoint loading):
  for every parsed answer in `buckets.jsonl` (`preds` field, ~7.5k candidates),
  teacher-forced score = length-normalized logP(answer tokens | prompt node
  truncated at "Answer:") — single forwards, no generation. Offline readouts:
  - [ ] (a) **calibration separability**: on `partial_precision` questions, do
        correct predictions score above the plausible-sibling errors (AUC /
        overlap)? If yes → over-generation is a calibration failure that
        TRAINING (e.g. stronger end-of-set/EOS supervision) can fix natively;
        if no → that class of training fix is off the table.
  - [ ] (b) **rerank delta**: re-order each question's parsed answers by score,
        recompute Hits@1 → upper-bounds what ordering is worth. Expected
        small — greedy already ≈ emits the highest-likelihood answer first;
        shuffled training (versions=8) makes first-position likelihood a
        marginal-confidence estimate, so the readout is meaningful either way.
  - [ ] Framing note for the paper/README: a set-generation model is
        structurally penalized by an ordered metric — Hit and F1 are the
        honest primary metrics; report Hits@1 for comparability.

## Phase B — data-format v3 + attribution sweep (~2–3 days)

- [ ] B1 **Delimiter fix — newline separator** (decided 2026-07-07 over
  comma-stripping and semicolon): in `process_dataset.py` the answer part of
  the prompt-node target becomes `"\n".join(answers)` (currently `", ".join`);
  in `evaluate.py::parse_answer_list` split on `"\n"` (currently `","`).
  Rationale: newline is collision-free (0/14,336 test golds contain one, vs
  12.7% for comma, 14 for semicolon), tokenizes as its own single token
  (verified: `Baku\nGanja` → `...aku, \n, G...`), and matches GNN-RAG's own
  generation format exactly. The `"\nAnswer:"` truncation anchor
  (`ANSWER_DELIM`) is a different string and stays. At smoke time, eyeball
  generations for `\n\n` drift (the parser filters empty segments regardless).
- [ ] B2 **Naming v2**, guided by A1: for cause (iii), merge a broader
  mid→name alias source into `entities_names.json` (candidates: RoG /
  GNN-RAG / SubgraphRAG public entity-name dumps); touch cap/collapse logic
  only if A1 shows causes (i)/(ii) are material. Kills "unnamed entity" node
  texts (currently the fallback in `resolve_entity_text`).
- [ ] B3 Bump `DATA_FORMAT_VERSION = 3` in `config.py` (one bump covers B1+B2 —
  it invalidates all `_dfv2` caches), then re-run the coverage analysis:
  `python3 -m src.experiments.kgqa --mode data_prep --analyse-dataset`
  and confirm the capped ceilings moved (expect up to ~+3.7 Hit on test;
  update the README ceiling table).
- [ ] B4 Rebuild datasets: v3 at `n_max=20` AND `n_max=50` (recall arm —
  questions with >5 present golds average 12.8 preds vs more golds; NOT a
  gen-length artifact, only 14/1628 generations hit the 128-token cap).
  `n_max` is in the cache key, so these are two builds (~10 min each on one
  GPU; see `results/dataprep_last2/job.sh` for the job pattern).
- [ ] B5 **Attribution sweep** at the cheap operating point (15 epochs,
  lr 1e-4, bias_lr 5e-3, k_hop 0, batch 2 × accum 4, full-dev checkpoint
  selection: `gen_eval_samples: null`, `gen_max_samples: null`; eval_steps 200
  ≈ 24 selection points over ~4800 steps): axes
  `data {v2-control, v3} × seed {0,1,2}` = 6 runs × ~2 h; optional
  `+ v3/n_max=50` arm (3 more, with `gen_max_new_tokens: 256`). Buys v3
  attribution AND the seed-noise bar we have never measured (retroactively
  calibrates every past "within noise" claim). Plain autoregressive
  generation only (per A2). NOTE: the sweep runner has no "data version" key —
  v2-control vs v3 differ only via `DATA_FORMAT_VERSION`, so either pin the
  v2 arm to the existing `_dfv2` cache with an explicit config knob, or run
  it as a second 3-run sweep before bumping the version; decide at
  implementation.
- [ ] B6 Update the README: Results table (all runs), goal table
  ("Ours — best"), and ceiling table (from B3).

## Phase C — capacity probe (parallel with B5's queue time)

- [ ] C1 `lora_r` [8, 64] on v3/n_max=20 data, same operating point as B5
  (2 runs × ~2 h; the r=16 control comes from B5's v3 arm). Target bucket:
  miss_copied (8.1 pts — wrong-node selection). If r=64 doesn't move that
  bucket, that is the cleanest evidence the residual gap is backbone scale →
  feeds the planned one-off 3B/8B scale run (README goal #3), not more 1B
  tuning.
- [ ] C2 **Re-run the error analysis on the best v3 model**: repoint
  `CHECKPOINT` in `results/error_analysis/dump_generations.py` (and the cache
  path in `analyze_errors.py` — v3 dir name), rerun both, and diff the bucket
  table against `results/error_analysis/report.md`. The bucket diff — not
  headline F1 — is the readout on which interventions worked and what remains.

## Expected landing zone (if estimates hold)

+1.2 (delimiter) + ~2 (naming F1) from pure data fixes → **F1 ≈ 69–70**,
approaching GNN-RAG's 71.3; closing the rest rides on what A2's calibration
readout says about training-side set-calibration fixes, and on C1. Hit ceiling
rises 89.7 → ~93 capped. (No test-time post-processing points in this
projection — excluded per the A2 decision.)

## Deferred (decided, not forgotten)

- **Text-serialization ablation** (README goal #2) — same base LLM, subgraph
  as flat triples in the prompt, no graph biases. Slot after B5 so it compares
  against v3 data.
- **Scale run 3B/8B** (goal #3) — after C1 says whether capacity is the
  binding constraint.
- **Retriever portability / union** (goal #4) — only after goals 1–3; raises
  the 91.1 uncapped Hit ceiling; pressures max_nodes=512 / O(N²) bias.
- **k-hop gating** — accuracy question closed (0 > 5 ≫ 2); revisit only as an
  efficiency lever (prompt-global mask + small k) if large-N inputs arrive.
- **CWQ** — untouched (`data/sr-cwq` exists); start after WebQSP v3 settles.

## Open decisions

1. B5 budget: 6 runs (attribution + seeds) vs 9 (+ n_max=50 arm) — ~12 vs ~18
   GPU-hours.
2. Whether B1+B2 land as one arm (cheaper, joint attribution) or get a quick
   ablation split if the v3 delta surprises.
3. B5's v2-control mechanics (pin old cache vs pre-bump sweep) — see B5 note.
