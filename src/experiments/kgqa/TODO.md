# KGQA (SR-WebQSP) — results summary

**The goal:** train a single graph-native model (GTLM) to answer questions directly from retrieved knowledge-graph subgraphs, replacing GNN-RAG's two-stage GNN-reasoner + LLM-reader pipeline. Success is measured against *retrieval-matched* SR-GNN-RAG (~78.9 Hits@1 / 71.3 F1) — same retrieval, so any gap is attributable to the reasoning architecture rather than better retrieval.

**Data-pipeline fixes were the single biggest lever.** Two problems were diagnosed by tracing exactly where the model's 33-point F1 gap actually came from: (1) roughly 6.5% of gold answers contain an internal comma, which collided with the comma-separated answer format used both for training targets and for parsing generations — these answers were structurally unlearnable and unscoreable; and (2) 76 test questions were losing their gold answer not to genuine retrieval failure but because the entity's name was missing from the naming dictionary, which caused the pipeline to mistake real named entities for anonymous graph mediators and collapse them away. Fixing the delimiter (switching to a newline separator) and closing the naming gap (merging in a public Freebase name dump) together raised test F1 from a 66.5 plateau to **71.1 on average across three seeds, and 72.5 on the best run** — a gain of +4.6 points, several times larger than the seed-to-seed noise measured in the same sweep (±0.4–1.0 points). This also raised the ceiling on how well any model could possibly do on this data (the fraction of questions where the answer is even present and named correctly rose from ~87.5% to ~92%).

**Two diagnostic probes clarified what's *not* worth chasing further.** First, scoring how confident the model was in each candidate answer (without changing inference) showed that reordering answers by confidence barely moves the "first answer correct" metric (Hits@1) — meaning that metric's gap to GNN-RAG is a structural artifact of scoring a multi-answer generator on a "first guess" basis, not a real weakness; F1 and "any correct answer found" (Hit) are the metrics that actually reflect quality. Second, comparing a small vs. much larger LoRA adapter showed that the "wrong node selected" error category was completely unmoved by extra capacity — meaning that particular failure mode is bounded by the base model's size (currently a 1B-parameter Llama), not by how it's fine-tuned. That result is the evidence needed to justify testing a larger backbone (3B/8B) as the next step, rather than more tuning at 1B.

**Net result: the model now matches or exceeds the target baseline.** Best run: F1 72.5 (vs. GNN-RAG's 71.3 — ahead) and Hits@1 78.75 (vs. ~78.9 — within noise). A detailed before/after error breakdown showed the remaining ~27 points of lost F1 split roughly three ways: an unavoidable retrieval-ceiling floor (~7.7 pts, not fixable without changing retrieval), a "how many answers to generate" calibration issue that flipped from over-generating to under-generating once the data fixes landed (~10 pts, and shown to be trainable rather than fundamental), and genuine wrong-answer reasoning errors (~9 pts, now understood to require a bigger model rather than more training). Those three categories now define the next steps: recall-focused calibration training, a scale-up experiment, and a planned ablation comparing this graph-native approach against simply feeding the same subgraph as flattened text — the comparison that would show whether the graph structure itself is earning its keep.

---

# TODO — pre-scale plan (2026-07-08)

**Objective:** lock the best 1B recipe and land the two attribution controls —
text-serialization and instruct-vs-base — on WebQSP, so that the one-off 3B/8B
scale run (README goal #3) tests a frozen, fully-attributed recipe with both
arms. CWQ stays parked until WebQSP is nailed down.

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

## D0 — diagnostics on existing checkpoints (no training, GPU-light)

*Goal: decide whether the 10.3-pt recall-side calibration pool is a trainable
stopping-decision problem or the same reasoning limit as miss_copied — this
verdict alone decides whether D4 exists.*

- [ ] **D0.1 stopping-decision autopsy** (extends `error_analysis/score_candidates.py`;
  one teacher-forced pass over the 1628 test generations of the best v3
  checkpoint). At every answer boundary (each `\n` and the final EOS) record
  P(stop) vs P(continue); additionally score every *missed* present-gold as a
  continuation appended to the emitted prefix (length-normalized logP, as in A2).
  - **Verdict (a)** — missed golds are high-likelihood but the stop decision
    won → under-generation is a calibration failure → **D4 activates**.
  - **Verdict (b)** — missed golds score low → the model doesn't identify
    them; same limitation as miss_copied → **delete D4**, fold the pool into
    the scale-run hypothesis.
- [ ] **D0.2 training-target audit** (CPU only): distribution of stored target
  set sizes vs present-gold counts across the 8 `versions` (n_max truncation
  interaction) — verify the training data itself doesn't systematically teach
  early stopping, and size how many questions are affected.

## D1 — hyperparameter retune under v3 (sweep `retune_v3`)

*Goal: the current operating point (lr 1e-4, r16, 15 ep) was tuned on the v2
comma-format landscape; v3 changed the loss surface. Also settle whether the
r64 gain (+1.6 F1, single seed) is real. ~7 runs × ~2 h on B300.*

All arms dfv3, n_max 20 unless noted, k_hop 0, last_1, 15 epochs, full-dev
checkpoint selection (eval_steps 200):

- [ ] **D1.1** scaffold the sweep config (pattern: `configs/attribution_v3.jsonc`):
  - `lora_r 64 × seed {1, 2}` — 3-seed confirmation of the capacity gain
  - `lr {5e-5, 2e-4} × r64 seed 0` — lr re-centering (1e-4 exists)
  - `bias_lr 1e-2 × r64 seed 0`
  - `lora_r 128 × seed 0` — is capacity still monotone past 64?
  - `n_max 50 × r64 seed 0` — cross the two individually-positive arms
- [ ] **D1.2** report + decision: new default = argmax mean dev F1;
  r64 verdict at 3 seeds.
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

- [ ] **D2.1 flat data-prep mode** (`--prompt-format flat` or sibling script):
  serialize the *raw selected triples* (post-`select_triples`, pre-Levi /
  pre-CVT-collapse — the collapse is part of OUR graph processing, so the
  honest "subgraph as text" control skips it) with the same entity texts, as
  `{question}\n{h | r | t per line}\nAnswer: {answers}` — identical
  newline-joined targets, identical label masking on "Answer:".
- [ ] **D2.2 trainer path**: plain HF causal-LM + LoRA (no graph collator, no
  biases), flash/sdpa attention, seq-len 4096; reuse `evaluate.py` verbatim
  (same metrics, same gold lists); append to the same runs.jsonl schema.
- [ ] **D2.3 runs**: 1 seed at the D1-winner budget (same lora_r; lr may need
  its own value — add one lr probe arm). If it lands within ~2 F1 of the graph
  arm, extend to 3 seeds for a real comparison.

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

- [ ] **D3.1 config**: add `prompt_style: "plain" | "chat"` to `RunConfig` —
  data-affecting → goes into `data_config_key()` (model_name is already in the
  key, so instruct builds cache separately regardless). Default: `"chat"` iff
  `"Instruct" in model_name`, explicit override allowed (needed for the
  instruct+plain control arm). Add the `PINNED_SYSTEM_PROMPT` constant.
- [ ] **D3.2 pipeline**: chat branch in `add_prompt_node` (build the templated
  string with the pinned system turn); derive `question_end` from
  `prompt_style` in both `process_dataset.py` and `train.py`.
- [ ] **D3.3 unit tests** (pattern: `tests/test_probes_flags.py`): masking
  starts exactly after the assistant header; generation cut ends exactly at
  the assistant header; special tokens round-trip as single ids under
  `add_special_tokens=False`; unanswerable-row labels; plain style unchanged
  (dfv3 golden).
- [ ] **D3.4 smoke run** (`--max-steps 4 --gen-max-samples 8`): confirm
  `generate` stops on `<|eot_id|>`, decoded text parses on newlines.
- [ ] **D3.5 arms** (at the D1-winner training config): `instruct+chat × seed
  {0,1,2}` (headline) + `instruct+plain × seed 0` (isolates weights vs
  formatting) vs the existing base 3-seed v3 arm. ~4 runs ≈ 8 GPU-h.
- [ ] **D3.6 decision**: if instruct wins, it becomes the default backbone for
  D2's 3-seed extension and the scale run (Llama-3.2-3B-Instruct / 8B-Instruct).

## D4 — recall-side set calibration (CONDITIONAL on D0 verdict (a))

*Goal: recover part of the 10.3-pt under-generation pool strictly training-side
— generation logic and target format untouched.* Candidate levers, to be chosen
by what D0.1 actually shows: re-weighting the loss on the boundary token (the
newline-vs-EOS decision), and/or scaling `versions` with answer-set size so
large sets contribute proportionally more full-set stop examples. One lever vs
control, 3 seeds. If D0 returns verdict (b), this section is deleted.

## D5 — gate: the scale run

When D1–D3 are in (D4 optional): freeze the 1B recipe — backbone variant,
prompt_style, lora_r, lr, n_max — and run **one 3B/8B experiment with BOTH
arms (graph-native + text-serialization)**, which is what makes the
architecture claim testable at scale (RPO-RAG gains +11.5 F1 from 1B→8B on
text alone; the question is whether the graph arm keeps its edge). Then unpark
CWQ.

**Rough budget:** D0 ≈ 1 GPU-h · D1 ≈ 14 · D2 ≈ 10–20 · D3 ≈ 8 · D4 ≈ 12 if
activated → ~35–55 GPU-h pre-scale, all single-B300 jobs.
