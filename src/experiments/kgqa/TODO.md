# KGQA (SR-WebQSP) — results summary

**The goal:** train a single graph-native model (GTLM) to answer questions directly from retrieved knowledge-graph subgraphs, replacing GNN-RAG's two-stage GNN-reasoner + LLM-reader pipeline. Success is measured against *retrieval-matched* SR-GNN-RAG (~78.9 Hits@1 / 71.3 F1) — same retrieval, so any gap is attributable to the reasoning architecture rather than better retrieval.

**Data-pipeline fixes were the single biggest lever.** Two problems were diagnosed by tracing exactly where the model's 33-point F1 gap actually came from: (1) roughly 6.5% of gold answers contain an internal comma, which collided with the comma-separated answer format used both for training targets and for parsing generations — these answers were structurally unlearnable and unscoreable; and (2) 76 test questions were losing their gold answer not to genuine retrieval failure but because the entity's name was missing from the naming dictionary, which caused the pipeline to mistake real named entities for anonymous graph mediators and collapse them away. Fixing the delimiter (switching to a newline separator) and closing the naming gap (merging in a public Freebase name dump) together raised test F1 from a 66.5 plateau to **71.1 on average across three seeds, and 72.5 on the best run** — a gain of +4.6 points, several times larger than the seed-to-seed noise measured in the same sweep (±0.4–1.0 points). This also raised the ceiling on how well any model could possibly do on this data (the fraction of questions where the answer is even present and named correctly rose from ~87.5% to ~92%).

**Two diagnostic probes clarified what's *not* worth chasing further.** First, scoring how confident the model was in each candidate answer (without changing inference) showed that reordering answers by confidence barely moves the "first answer correct" metric (Hits@1) — meaning that metric's gap to GNN-RAG is a structural artifact of scoring a multi-answer generator on a "first guess" basis, not a real weakness; F1 and "any correct answer found" (Hit) are the metrics that actually reflect quality. Second, comparing a small vs. much larger LoRA adapter showed that the "wrong node selected" error category was completely unmoved by extra capacity — meaning that particular failure mode is bounded by the base model's size (currently a 1B-parameter Llama), not by how it's fine-tuned. That result is the evidence needed to justify testing a larger backbone (3B/8B) as the next step, rather than more tuning at 1B.

**Net result: the model now matches or exceeds the target baseline.** Best run: F1 72.5 (vs. GNN-RAG's 71.3 — ahead) and Hits@1 78.75 (vs. ~78.9 — within noise). A detailed before/after error breakdown showed the remaining ~27 points of lost F1 split roughly three ways: an unavoidable retrieval-ceiling floor (~7.7 pts, not fixable without changing retrieval), a "how many answers to generate" calibration issue that flipped from over-generating to under-generating once the data fixes landed (~10 pts, and shown to be trainable rather than fundamental), and genuine wrong-answer reasoning errors (~9 pts, now understood to require a bigger model rather than more training). Those three categories now define the next steps: recall-focused calibration training, a scale-up experiment, and a planned ablation comparing this graph-native approach against simply feeding the same subgraph as flattened text — the comparison that would show whether the graph structure itself is earning its keep.

---

# TODO — pre-scale plan (2026-07-08) — EXECUTED 2026-07-08/09

**Objective:** lock the best 1B recipe and land the two attribution controls —
text-serialization and instruct-vs-base — on WebQSP, so that the one-off 3B/8B
scale run (README goal #3) tests a frozen, fully-attributed recipe with both
arms. CWQ stays parked until WebQSP is nailed down.

## Executive summary of the executed plan (2026-07-09, ~50 GPU-h)

All of D0–D4 ran overnight (details + decision logs in each section; full
per-arm table: `results/error_analysis/prescale_summary.py`). Headlines:

1. **The text-serialization control WINS at 1B: flat 73.37 ± 0.21 test F1 vs
   graph-native 71.56 ± 0.25 at nmax20 (3 seeds each), widening to
   flat 75.03 vs graph 72.55 ± 0.22 at the frozen nmax50 recipe** —
   byte-identical supervision, same LoRA/lr/schedule; flat result
   independently audited (fresh reload + offline rescore, exact match).
   The graph attention biases *cost* 1.8–2.5 F1 at this scale. This reframes
   the D5 scale run: the question is no longer "does the graph arm keep its
   edge" but whether graph structure closes its deficit with backbone
   capacity. Both arms beat retrieval-matched SR-GNN-RAG (71.3 F1); the flat
   nmax50 run beats it on BOTH headline metrics (75.0 F1, 79.6 Hits@1).
2. **Best 1B graph recipe (D1):** r64 / lr 1e-4 / bias_lr 5e-3 / **n_max 50**
   — confirmed at 3 seeds: 72.55 ± 0.22 (r64+nmax50 stack; +1.0 over
   nmax20). Capacity not monotone (r128 worse).
3. **Instruct backbone (D3): modest, real win** — instruct+chat 72.11 ± 0.34
   vs base 71.57 ± 0.31 (+1.0 Hits@1); the instruct+plain control shows the
   gain needs chat formatting + weights together. Scale run backbone:
   3B/8B-Instruct, prompt_style chat.
4. **Set-calibration lever (D4): null.** Boundary-token loss re-weighting
   (w=4, renormalized) moved nothing (F1 71.36 ± 0.09; Hit* slightly down).
   Combined with D0.1 (missed golds mostly score low; stop margins soft but
   emphasis doesn't fix identification): the 10.3-pt under-generation pool is
   a reasoning/scale limitation, not a trainable stopping decision.
5. **Diagnostics (D0):** training targets don't teach early stopping (D0.2);
   the autopsy artifacts (boundary_scores.jsonl, missed_gold_scores.jsonl)
   are reusable on any future checkpoint.

**Current 1B state of the art on this benchmark, ours:** flat text arm with
the COLLAPSED (relation-composition) serialization at the frozen nmax50
recipe — **test F1 74.93 ± 0.23 / Hits@1 79.71 / Hit* 84.52** (3 seeds; best
single run 75.26). The collapse 2×2 (D2b, user-requested) rejected the
hypothesis that the CVT collapse causes GTLM's deficit: collapse helps BOTH
arms; the flat−graph gap (+2.4 collapsed / +3.0 uncollapsed) is attributable
to the graph-native representation itself at 1B.

**Standing constraints (locked):**
- Generation stays natural greedy autoregressive decoding — we are evaluating
  the models' native capabilities. No inference-time interventions (no EOS
  suppression, no likelihood cut-offs, no reranking).
- No target-format gimmicks (e.g. answer-count prefixes): predicting the count
  up-front would force the model to solve the whole set in one step and then
  constrain itself to it — answers are predicted one-by-one.
- Retrieval stays fixed at SR; `max_nodes` stays 512.

**What we're attacking** (error_analysis C2 decomposition of the best v3 run's
27.37 missing F1 pts): SR ceiling 7.65 (unfixable at fixed retrieval) ·
set calibration 10.29 (now recall-side: the model under-generates, 6.26 preds
vs 9.85 present golds on multi-answer questions) · miss_copied 9.15 (invariant
to LoRA capacity per C1 → backbone scale) · fabricated 0.31 (solved).

Reference numbers: best v3 run (attribution_v3_0004, seed 1) test F1 72.50 /
Hits@1 78.75 / Hit 82.74; v3 3-seed mean F1 71.11 ± 1.0; r64 single-seed 71.90.

## D0 — diagnostics on existing checkpoints (no training, GPU-light) ✅ DONE 2026-07-08

*Goal: decide whether the 10.3-pt recall-side calibration pool is a trainable
stopping-decision problem or the same reasoning limit as miss_copied — this
verdict alone decides whether D4 exists.*

- [x] **D0.1 stopping-decision autopsy** (`error_analysis/stop_autopsy.py` +
  `stop_autopsy_report.py`; teacher-forced pass over all 1628 test questions of
  attribution_v3_0004/checkpoint-4000; 6 095 emitted answers + 3 505 missed
  present-golds scored). **VERDICT: (a)-partial — D4 ACTIVATES.** The evidence:
  - The missed-gold POOL scores well below emitted answers (median logp_norm
    −1.31 vs −0.35; only 20% clear the emitted p25) → the *bulk of the missed
    mass* is identification failure, same family as miss_copied (scale-bound).
  - BUT the stop decision is demonstrably soft exactly where under-generation
    happens: final-EOS margin logP(eos)−logP(nl) has median **2.75 nats on
    under-generating questions vs 11.1 on fully-recalled ones** (p25 = 1.0 nat);
    P(continue) at the premature stop reaches 0.27–0.38 in the upper quartile.
  - Per-question: **64% of under-generating questions (277/432) hold at least
    one missed gold scoring ≥ p25 of emitted answers**, and 121 of those also
    stopped with margin < 2 nats → a quarter-to-half of the 10.3-pt pool is
    plausibly trainable stopping calibration; the rest folds into the
    scale-run hypothesis.
- [x] **D0.2 training-target audit** (`error_analysis/target_audit.py`, CPU).
  **The training data does NOT systematically teach early stopping**: stored
  target size == eval-side present-gold count for ~94% of train questions;
  the mismatch is confined to 148/2607 (5.7%) truncation-bound questions at
  n_max 20 (4 156 answers lost) and halves at n_max 50 (63 questions, 1 624
  lost). The n_text−n_kbid gap is 0 for 95% of questions. Under-generation
  affects far more questions than the 6% with short targets → not data-taught,
  consistent with D0.1's calibration reading (and a mild extra argument for
  the n_max-50 arm in D1).

## D1 — hyperparameter retune under v3 (sweep `retune_v3`)

*Goal: the current operating point (lr 1e-4, r16, 15 ep) was tuned on the v2
comma-format landscape; v3 changed the loss surface. Also settle whether the
r64 gain (+1.6 F1, single seed) is real. ~7 runs × ~2 h on B300.*

All arms dfv3, n_max 20 unless noted, k_hop 0, last_1, 15 epochs, full-dev
checkpoint selection (eval_steps 200):

- [x] **D1.1** sweep ran 2026-07-08/09 (`configs/retune_v3.jsonc`, 7 runs).
  Test F1 (dev F1) per arm, all r64/lr1e-4/bias_lr5e-3/nmax20 unless noted:
  | arm | test F1 | dev F1 |
  |---|---|---|
  | r64 seed 1 | 71.29 | 72.26 |
  | r64 seed 2 | 71.51 | 71.81 |
  | lr 5e-5 (s0) | 71.28 | 72.02 |
  | lr 2e-4 (s0) | 69.57 | 71.02 |
  | bias_lr 1e-2 (s0) | 69.18 | 71.22 |
  | r128 (s0) | 70.22 | 72.57 |
  | **n_max 50 (s0)** | **72.35** | **73.29** |
- [x] **D1.2 decision (2026-07-09):** new default = **r64 / lr 1e-4 /
  bias_lr 5e-3 / n_max 50** (argmax dev F1 73.29; also best test 72.35 —
  r64 and n_max 50 STACK, consistent with D0.2's truncation finding).
  r64 verdict at 3 seeds (nmax20): mean 71.57 ± 0.31 vs r16's 71.11 ± 1.0 —
  the capacity gain is real but modest (+0.5); capacity is NOT monotone past
  64 (r128 dev-test gap: dev 72.57 / test 70.22). lr stays 1e-4 (5e-5 flat,
  2e-4 clearly worse); bias_lr stays 5e-3.
  **Reconciliation:** D2/D3/D4 arms were launched (decision logged below) at
  the presumptive winner r64/lr1e-4/**nmax20** — those comparisons remain
  internally consistent (all arms matched on data config); n_max is a
  data-side knob orthogonal to the attribution questions. Follow-up sweep
  `frozen_nmax50` closes the gap: graph seeds {1,2} at nmax50 (3-seed set
  with retune_v3_0006) + flat control at nmax50 seed 0, so the frozen D5
  recipe has a matched pair.
  **frozen_nmax50 results (2026-07-09):** graph nmax50 = {72.35, 72.86,
  72.45} → **3-seed mean 72.55 ± 0.22** (vs nmax20's 71.56 ± 0.25: the
  nmax50 gain replicates, +1.0). Winner recipe confirmed at 3 seeds.
- **No longer-epoch arm** — wandb dev-F1 curves for the v3 runs are flat over
  the entire second half of the 15-epoch schedule (converged; eval loss even
  drifts up while F1 holds).

## D2 — text-serialization ablation (README goal #2)

*Goal: same base LLM, same capped SR subgraph flattened to triples in the
prompt, no structural biases — the control that isolates whether the graph
attention biases do anything. This is the result that transfers to the rest of
the repo, and the scale run is only conclusive if this arm exists too.*

**Feasibility settled (measured 2026-07-08):** serializing the SAME
`select_triples(max_nodes=512)` subgraphs as `head | relation | tail` lines
(last_1 verbalization, entities_names v2 texts, Llama-3.2-1B tokenizer) gives
train mean 734 / p99 3 637 / max 4 499 tokens and test mean 761 / p99 3 547 /
max 4 714 — **seq-len 4096 covers 99.6 %+ of questions** (truncate the ~10
outliers). ≈2.2× the graph model's token count → est. 3–5× the ~2 h graph run,
6–10 GPU-h per run. No trick needed.

- [x] **D2.1 flat data-prep mode** (`flat_data.py`, `--mode flat_data_prep`):
  serializes the *raw selected triples* (post-`select_triples`, pre-Levi /
  pre-CVT-collapse) with the same entity texts, as
  `{question}\n{h | r | t per line}\nAnswer: {answers}`. **Fairness verified
  2026-07-08: all 22 730 rows (train 8-version layout included) carry
  byte-identical questions/targets/gold lists to the graph caches** — the same
  `present_answer_texts` on the same built Levi graph, the same augmentation
  RNG stream. Only the input representation differs.
- [x] **D2.2 trainer path** (`flat_train.py`, `--mode flat_train`): plain HF
  causal-LM + LoRA via the same `select_active_params`/`cfg.lora_config()`,
  sdpa attention, seq-len 4096 (outliers drop trailing triple lines), same
  scoring functions / schedule / greedy decoding / full-dev eval_f1 selection;
  appends to the same runs.jsonl schema (`arm: flat_text`). Smoke run passed
  end-to-end (4 steps, generation + parsing + record, rc=0).
- [ ] **D2.3 runs**: launched 2026-07-08 (`configs/flat_v3.jsonc`, job 110138):
  lr {1e-4, 2e-4} × r64 × seed 0. **Decision logged:** launched at the
  PRESUMPTIVE D1 winner (r64 / lr 1e-4 / n_max 20) without waiting ~1.5 h for
  D1 — r64 was the only arm with prior evidence (+0.8 F1 at seed 0); if D1
  disagrees, the discrepancy gets logged and the headline arms re-run.
  If the flat arm lands within ~2 F1 of the graph arm → extend to 3 seeds.
  **Results (2026-07-09, seed 0):** lr 1e-4 → **test F1 73.10** / Hits@1 77.83
  / Hit* 83.91 (dev F1 75.22); lr 2e-4 → test F1 72.96. The flat control
  BEATS the seed-matched graph arm (r64 seed 0: 71.90) and the best graph run
  overall (72.50) at identical supervision — at 1B the graph attention biases
  are NOT earning their keep on this benchmark. lr verdict: keep 1e-4 (probe
  2e-4 slightly worse). **3-seed extension launched** (`flat_v3_seeds`,
  seeds {1,2}) for the real comparison.
  **3-seed verdict (2026-07-09 04:30):** flat test F1 {73.10, 73.61, 73.40} →
  **mean 73.37 ± 0.21** vs graph r64 3-seed **71.57 ± 0.31** (same lr/r/data).
  The seed bars don't overlap: **at 1B the flat text serialization BEATS the
  graph-native arm by +1.8 F1** at byte-identical supervision. The graph
  attention biases are not just "not earning their keep" — they cost F1 at
  this scale. (Both arms still beat retrieval-matched GNN-RAG's 71.3 F1 —
  with the standing caveat that our dfv3 data fixes are not in their published
  number; the flat-vs-graph comparison is the controlled claim.)
  A matched pair at the frozen nmax50 recipe is running (`frozen_nmax50`).
  **Independently audited (2026-07-09, `error_analysis/audit_flat.py`):**
  fresh checkpoint reload outside the trainer + offline rescore over all 1628
  test questions reproduces F1 0.7310 / H@1 0.7783 / Hit* 0.8391 exactly;
  per-question generations in `audit_flat_generations.jsonl`.
  **Matched pair at the frozen nmax50 recipe (2026-07-09):** flat nmax50
  seed 0 → **test F1 75.03 / Hits@1 79.61 / Hit* 84.58** vs graph nmax50
  3-seed 72.55 ± 0.22. The flat advantage WIDENS at nmax50 (+2.5 F1) — the
  extra recall headroom is exploited better without the graph biases. This
  run beats retrieval-matched GNN-RAG on BOTH headline metrics
  (F1 75.0 vs 71.3; Hits@1 79.6 vs ~78.9) and is the new best 1B result.

## D3 — instruct base model (Llama-3.2-1B-Instruct)

*Goal: test whether instruction-tuned weights + chat formatting beat the base
model at identical graph inputs, before committing the scale run to a backbone
variant. Design locked 2026-07-08:*

- **Graph nodes: unchanged** (plain text, `add_special_tokens=False`). The
  graph is a new modality with no template to fit into; we teach the model to
  read it as-is.
- **Prompt node = ONE node holding the full chat-templated string**
  (turns-as-nodes deliberately deferred — don't change graph structure and
  base model at the same time):
  `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{PINNED_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answers, newline-joined}<|eot_id|>`
- **System turn: pinned minimal constant** (e.g. "Answer the question using
  the provided knowledge graph context.") — never the template default, whose
  auto-inserted "Today Date" would make cache content depend on build date.
- **BOS kept verbatim mid-sequence**: from `<|begin_of_text|>` onward the
  stream matches the SFT distribution exactly; the BOS doubles as the
  delimiter between graph modality and chat turn. (The graph prefix is
  off-distribution either way; LoRA adapts.)
- **Label/generation boundary**: the assistant-header token sequence
  (`<|start_header_id|>assistant<|end_header_id|>\n\n`) replaces `"Answer:"`
  as `question_end`. `AnswerLabelMasker` and `_find_prefix_len` match
  arbitrary token subsequences — no logic changes, only the constant.
- **EOS**: the instruct tokenizer's `eos_token_id` IS `<|eot_id|>`, so
  `tokenize(add_eos=True)` already appends the right terminator; unanswerable
  eval rows keep the existing empty-target convention (labels = terminator
  only). The special tokens are in-vocab, so the templated string tokenizes
  correctly as a plain node text — no new machinery.

Steps:

- [x] **D3.1 config**: `prompt_style` on `RunConfig` (None = auto: chat iff
  Instruct), `resolved_prompt_style` / `question_end_str` properties,
  `PINNED_SYSTEM_PROMPT` + `ASSISTANT_HEADER` constants; cache-key suffix
  `_pschat` only for non-plain styles so every existing cache keeps its name.
- [x] **D3.2 pipeline**: `chat_prompt_text` + chat branch in `add_prompt_node`;
  `question_end` derived from `cfg.question_end_str` in `process_dataset.py`
  and `train.py`. Verified on real records: supervised span decodes to exactly
  the newline-joined answers + `<|eot_id|>`.
- [x] **D3.3 unit tests** (`tests/test_kgqa_prompt_style.py`, 13 tests):
  masking/generation-cut at the assistant header, special-token round-trips,
  standalone-"\n" boundaries after the header, unanswerable labels =
  terminator only, header-subsequence unambiguity, plain-style dfv3 golden.
- [x] **D3.4 data prep**: both instruct caches built on GPU (`_pschat` +
  instruct-plain; configs/instruct_dataprep.jsonc). Smoke run submitted before
  the arms (confirm `generate` stops on `<|eot_id|>`, parses on newlines).
- [x] **D3.5 arms** (`configs/instruct_v3.jsonc`) — results 2026-07-09:
  | arm | test F1 | test H@1 |
  |---|---|---|
  | instruct+chat s0 | 71.97 | 78.32 |
  | instruct+chat s1 | 72.57 | 78.32 |
  | instruct+chat s2 | 71.77 | 79.30 |
  | **instruct+chat mean** | **72.10 ± 0.34** | **78.65** |
  | instruct+plain s0 | 71.12 | 77.52 |
  | base r64 3-seed (reference) | 71.57 ± 0.31 | 77.68 |
- [x] **D3.6 decision (2026-07-09):** instruct+chat wins by +0.5 F1 and +1.0
  Hits@1 over the base 3-seed set — a modest but seed-consistent gain. The
  instruct+plain control (71.12, BELOW the seed-matched base 71.90) shows the
  gain requires chat formatting + instruct weights TOGETHER; instruct weights
  under plain formatting actually hurt. **Default backbone for the scale run:
  Llama-3.2-3B/8B-Instruct with prompt_style chat.** (Flat+chat formatting is
  still deferred; must be built for the D5 text arm — see flat_data.py's
  NotImplementedError guard.)

## D4 — recall-side set calibration (ACTIVATED by D0.1 verdict (a)-partial)

*Goal: recover part of the 10.3-pt under-generation pool strictly training-side
— generation logic and target format untouched.*

**Lever decision (2026-07-08):** boundary-token loss re-weighting, implemented
as `boundary_loss_weight` (`KGQAGraphTrainer.compute_loss` →
`boundary_weighted_loss`, unit-tested in `tests/test_kgqa_boundary_loss.py`).
Supervised tokens equal to the dfv3 `"\n"` separator get weight w; weights are
renormalized so the mean weight over supervised tokens is 1 — total loss scale
(effective lr) unchanged, only per-token emphasis shifts toward the
continue-vs-stop decision. Chosen over `versions`-scaling because it attacks
exactly the soft stop margins D0.1 measured (median 2.75 nats), touches no
data, and is trivially attributable. Weight w=4 (options considered: 2 — risks
being invisible against ±1.0 seed noise given boundary tokens are only ~5–9%
of supervised mass; 4 — a strong, still-renormalized nudge; picked 4).
Plan: w=4 × 3 seeds at the D1-winner config vs the D1 r64 3-seed control.

Status 2026-07-09 00:40: smoke #1 crashed (a module-level helper had landed
mid-class and detached `evaluate` from `KGQAGraphTrainer` → no generative
metrics → KeyError 'eval_f1'; running sweeps were unaffected — they loaded the
pre-edit module). Fixed + regression test
(`test_trainer_class_keeps_its_methods`); smoke #2 clean end-to-end.
`calib_v3` (w=4 × seed {0,1,2}, r64/lr 1e-4) launched. Instruct arms
(`instruct_v3`) launched at the same time; instruct smoke confirmed
`<|eot_id|>` stopping + newline parsing (F1 0.24 on 8 samples after 4 steps).

**VERDICT (2026-07-09 05:15): NULL.** w=4 (3 seeds) vs the r64 control
(3 seeds), same config otherwise:
| metric | w=4 | control |
|---|---|---|
| test F1 | 71.36 ± 0.09 | 71.56 ± 0.25 |
| test Hits@1 | 77.33 ± 0.35 | 77.66 ± 0.33 |
| test Hit* (recall side) | 81.98 ± 0.29 | 82.29 ± 0.33 |
| test F1_strict | 70.95 ± 0.17 | 71.05 ± 0.22 |
No metric moved — including Hit*, the recall-side one the lever targeted;
if anything the lever costs a hair. Boundary-token loss emphasis does not
convert D0.1's soft stop margins into recovered answers at 1B: the model
learns the boundary distribution fine; what it lacks is identification of
further answers (consistent with D0.1's pool-level reading, where 80% of
missed golds scored below the emitted p25). **The full 10.3-pt
under-generation pool folds into the scale-run hypothesis.** No further
training-side calibration levers planned.

## D2b — collapse 2×2 ablation (user-requested 2026-07-09, running)

*Hypothesis (user): the CVT collapse — not the graph biases — may be what
holds GTLM below the flat serialization. The collapse is the single
structural asymmetry between the arms beyond the biases themselves.*

Design: cross `cvt_collapse` over both arms at the frozen recipe
(r64 / lr 1e-4 / n_max 50 / gen 256), 3 seeds per new cell
(`configs/collapse_2x2.jsonc`; knob: `--cvt-collapse/--no-cvt-collapse`,
None = arm default so all existing caches keep their keys; unit tests in
`tests/test_kgqa_cvt_collapse.py`):

**RESULTS (2026-07-09, test F1, 3 seeds per cell):**

|  | collapsed input | uncollapsed input | Δ collapse |
|---|---|---|---|
| GTLM | **72.55 ± 0.22** | 71.60 ± 0.46 | **+0.95** |
| flat LLM | **74.93 ± 0.23** | 74.59 ± 0.67 | +0.34 |
| Δ flat−graph | **+2.38** | +2.99 | |

**VERDICT: the collapse hypothesis is REJECTED — decisively.** The CVT
collapse does not hold GTLM back; it *helps* both arms (removing it makes
GTLM ~1 F1 WORSE), and the flat−graph gap persists in both columns
(+2.4 collapsed, +3.0 uncollapsed). The deficit is attributable to the
graph-native representation/biases themselves at this scale, not to the
collapse preprocessing. Secondary metrics agree (collapsed flat: H@1 79.71,
Hit* 84.52 — all cells' orderings identical).

Bonus findings: (1) the collapsed (relation-composition) serialization is
now the best-measured arm overall — **74.93 ± 0.23** with a tight bar, best
single run 75.26; the earlier flat-raw 75.03 was that arm's high seed
(3-seed 74.59 ± 0.67). (2) Collapse helps text too — composing relations
through mediators is a better serialization than raw "unnamed entity" lines.
**The frozen D5 flat arm should therefore use the COLLAPSED serialization.**

Implementation notes: the uncollapsed GTLM arm needs no cap change
(`select_triples` always budgeted PRE-collapse Levi nodes, so uncollapsed
graphs still fit 512); the collapsed-flat arm serializes the collapsed Levi
graph with relation-composition lines (`p | rel0 rel1 | leaf` — the exact
text equivalent of the graph contraction; verified on real records, e.g.
"sacha baron cohen | arrest source | …"). Targets are collapse-invariant
(collapse never removes a nameable answer node — unit-tested + parity check
against stored caches before launch).

## D5 — gate: the scale run (NOT run — excluded from this execution by design)

**The frozen 1B recipe (2026-07-09):** backbone Llama-3.2-3B/8B-**Instruct**,
prompt_style **chat**, lora_r **64**, lr **1e-4** (bias_lr 5e-3 for the graph
arm), **n_max 50** / gen 256, dfv3, SR/cap512, 15 epochs, full-dev selection.

Run **one 3B/8B experiment with BOTH arms (graph-native +
text-serialization)**. D2's verdict re-poses the question: at 1B the flat arm
is AHEAD by 1.8 F1, so the scale run now tests whether graph structure closes
its deficit (or the deficit widens) with backbone capacity — and the flat arm
is the stronger 1B baseline to beat (73.37 ± 0.21). Then unpark CWQ.

**Prerequisites still open for D5:**
- flat + chat formatting (`flat_data.py` currently guards it with
  NotImplementedError; the D3 chat design transfers directly — triples block
  ahead of `<|begin_of_text|>`, question in the user turn).
- ~~`frozen_nmax50` results~~ — landed 2026-07-09: graph 72.55 ± 0.22,
  flat 75.03 (see D1/D2 sections). Optionally add 2 flat-nmax50 seeds
  before freezing the D5 baseline number.

**Rough budget:** D0 ≈ 1 GPU-h · D1 ≈ 14 · D2 ≈ 10–20 · D3 ≈ 8 · D4 ≈ 12 if
activated → ~35–55 GPU-h pre-scale, all single-B300 jobs.
**Actual spend:** ≈ 50 GPU-h (7 D1 + 2+2+1 flat + 2 dataprep + 4 instruct +
3 calib + 3 frozen_nmax50 runs + smokes/diagnostics), all B300.
