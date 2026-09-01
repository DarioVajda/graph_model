# GraphQA refactor — progress & handoff

Status as of 2026-07-17. This file is the resume point: what the refactor is, what
is done and verified, the one open blocker, and the remaining steps.

## Goal

Bring the GraphQA experiment onto the generic `sweep` runner, matching the
`src/experiments/template` contract (one `RunConfig`, one standalone single-run
program, one JSONL record per run), the same shape kgqa / probes already use.

## Decisions taken

1. **v2 model only.** The legacy v0 path was removed, not kept behind a flag. v0 and
   v2-eager are pinned numerically equivalent at fp32 (loss + every bias gradient +
   base-weight gradients) by `tests/models/test_modeling_gtlm_llama_v2.py` and, at the
   wide within-batch node spread GraphQA's incidence graphs produce, by the new
   `tests/models/test_v2_ragged_magnetic_padding.py`. Dropping v0 also gains k-hop
   masking for free (which `graphqa_mag_khop` previously forked onto legacy v1 to get).
   - `impl` defaults to `v2-eager` (the parity anchor + right choice for these short
     sequences: ~35 tokens standard / ~150 incidence — flex only pays off far longer).
   - `dtype` defaults to `fp32` (the dtype the parity is proven at; bf16 is a real
     numerical change, opt-in).
2. **Official validation split.** GraphQA ships 1000 train / 500 val / 500 test per
   task; the old pipeline built only train+test and carved the last 15% off train.
   Now the official val split is used for the 9 reported tasks. The 2 non-reported
   tasks without an official val (`disconnected_nodes`, `node_classification`) fall
   back to carving `val_fraction` off train.
3. **Checkpoint selection on `eval_em_accuracy`**, not `eval_loss` (the metric actually
   reported; loss and accuracy diverge late in training).
4. **Dataset cache key** tags only non-default feature knobs, so the existing ~11 GB
   cache and `graphqa_mag_khop`'s hardcoded path stay valid; any deviation
   (`magnetic_q`, `max_rw_steps`, `magnetic_m`, `max_length`, `model_name`) gets a
   tagged sibling dir.
5. **GPU brand**: smoke + study accept `["B200:1","B300:1"]` (drain onto whatever is
   idle). Note this makes `train_runtime_s` non-comparable across brands — accuracy
   unaffected.

> Both protocol changes (2 and 3) mean new numbers are NOT bit-comparable to the
> originally published GraphQA numbers. This is deliberate and documented in README.md.

## What was built

New / rewritten in `src/experiments/graphqa/`:
- `config.py` — `RunConfig` (every knob), `validate()`, `bias_params()`,
  `lora_config()`, `dataset_dir()` (the cache rule), `arm()`, `has_official_val()`.
- `data.py` — `load_data(cfg)` → (train, val, test); `run_data_prep_mode(cfg)`;
  `required_splits(cfg)` (config-derived, NOT filesystem-derived — see note below).
- `process_dataset.py` — kept all graph-construction logic; replaced the `__main__`
  orchestration with `build_split(cfg, split, tokenizer)`; `question_end` is now
  derived from the tokenizer (`ANSWER_PREFIX = "A:"`), not the hardcoded `[32, 25]`.
- `train.py` — `run_train_mode(cfg, …)`, builds GTLM v2 directly (kgqa-style),
  `GraphCollatorV2` + `GraphTrainerV2` + shared `make_compute_metrics`; one JSONL
  record. Dropped the duplicated `init_model` / `select_active_params` /
  `print_trainable_parameters` / `PreprocessLogitsEM` / `compute_exact_match`.
- `__main__.py` — sweep-contract argparse program (`build_parser`,
  `config_from_args`, `--init`, `TEMPLATE`, dispatcher on `--mode {train,data_prep}`).
- `_io.py`, `__init__.py`.
- `configs/` — `000_example`, `001_data_prep`, `002_smoke` (plumbing check),
  `003_ablation` (the 135-run study: 5 arms × 9 tasks × 3 seeds), `004_canary`
  (1 real run on the easiest task — does it learn?).
- `analysis/prep_table.py` — the paper's LaTeX tables from `runs.jsonl` structured
  fields (was: regex-parsing run names against `results.json`).
- `README.md` — rewritten (quick start, layout, protocol, cache, the two protocol
  changes vs published numbers).

Deleted: `full_experiment.py`, `run_experiment.sbatch`, old `prep_table.py`.

New tests:
- `tests/experiments/test_graphqa_flags.py` — sweep-contract round-trip (render → parse
  → config), unwired-feature rejection, v0 rejection, 10 tests.
- `tests/models/test_v2_ragged_magnetic_padding.py` — v0/v2 parity at a 4-vs-40 node
  spread (the magnetic-m zero-padding path), with a negative control, 6 tests.

## Verified

- Full suite: **263 passed / 11 skipped** (11 are GPU tests, no driver on the login node).
- `graphqa_mag_khop` still imports and loads (its `load_graphqa_datasets` import intact).
- 135 runs expand + parse + validate; `arm` bundle label does not leak as a `--arm` flag.
- All 18 official validation splits built (data_prep sweep, CPU, local); prep idempotent;
  default cache resolves onto the existing on-disk train/test.
- `prep_table` arithmetic verified against known synthetic inputs; thin (<3-seed) cells
  print `?`.
- `sweep.report` renders `report.md` for the smoke sweep.
- **GPU smoke (002)** completed on a B300, logged well-formed records (0.0 accuracy is
  EXPECTED at 4 steps — EM is all-or-nothing and warmup has barely lifted the LR).
- **GPU canary (004)** trained node_count to `eval_em_accuracy = 1.0` (from 0.0 → 0.366
  → 0.892 → 1.0), proving the pipeline learns to ceiling. `eval_loss` 6.7e-06.

## OPEN BLOCKER — `load_best_model_at_end` drops graph-bias weights

**Do not launch the 135-run study until this is fixed.**

`GraphTrainerV2.save_model` writes a checkpoint in two parts: the LoRA adapter via HF's
`super().save_model()` (→ `adapter_model.safetensors`) and the trainable graph-bias
tensors via our own `save_bias_parameters()` (→ `bias_parameters.pt`). HF's
`_load_best_model` (transformers 4.50.3) only knows the adapter — for a PEFT model it
calls `model.load_adapter(best_ckpt, active_adapter)` and nothing else. So with
`load_best_model_at_end=True`, the reloaded model is the **best-checkpoint adapter +
end-of-training graph-bias weights** — a pair that never existed in training. The
post-`train()` dev/test eval reports that mismatched model, so the final number is
**understated** (training + on-disk checkpoints are fine; only the reported metric is wrong).

**Measured, not inferred.** Loading canary `checkpoint-80` (recorded `best_metric=1.0`)
two ways on the same val split:

| load | eval_em_accuracy | eval_loss |
|---|---|---|
| adapter + `bias_parameters.pt` (full) | **1.0** | 0.0136 |
| adapter only (HF's path) | **0.652** | 0.257 |

The run itself logged 0.838 (best adapter + partially-trained end-of-run bias; the exact
value isn't reproducible because `save_total_limit=1` deleted the final checkpoint).
Severity scales with how far the bias drifts after the best checkpoint: pathological when
best is early (canary: step 80/620), negligible when it lands late.

**This is a pre-existing bug in shared code, not introduced by this refactor.** It affects
every experiment using LoRA + `active_params` + `load_best_model_at_end`: **graphqa, kgqa,
probes, expressiveness, tag_benchmarks** — all report an understated best-checkpoint metric
by an unmeasured, run-dependent amount. Their published numbers are real lower bounds (never
inflated); a fix will *raise* them and break comparability with what's already recorded.

**The fix already exists but is never called:** `src/models/io.py` has
`load_bias_parameters()` (mirror of `save_bias_parameters`). Likely fix: override
`_load_best_model` in `GraphTrainerV2` to call
`load_bias_parameters(self.model, self.state.best_model_checkpoint)` after the adapter
loads (or make checkpoints self-contained in `_save_checkpoint`).

**Decision pending:** scope of the fix (shared trainer vs graphqa-only workaround vs
document-only), because it reaches into other experiments' published results. The options
are laid out above; the kgqa blast radius has to be understood before choosing between
them (structurally confirmed affected; magnitude unmeasured — no kgqa checkpoint currently
on disk to cheaply measure, would need a fresh multi-hour run).

## Remaining steps

1. **Resolve the reload bug** (blocker above) — settle the fix scope, then:
   - Override `_load_best_model` in `src/utils/text_graph_trainer_v2.py` to restore bias
     params (or equivalent seam).
   - Add a regression test: train a few steps, force a known-best early checkpoint, assert
     the reloaded model reproduces its recorded score. Put it where the shared trainer is
     tested (`tests/utils/` or `tests/models/`).
   - If shared-trainer scope: flag that kgqa / probes / expressiveness / tag_benchmarks
     past numbers were understated; re-running them is separate work.
2. **Re-run the canary (004)** and confirm `test_accuracy` now lands where val says
   (~1.0), i.e. the reported number matches the best checkpoint.
3. **Launch `003_ablation`** (135 runs, sbatch) only after 1–2 pass.
4. `sweep.report` + `analysis/prep_table` over `003_ablation` results → paper tables.
5. Commit. Nothing is committed yet — deletions are staged via `git rm`; everything else
   is working-tree. Suggested commit boundary: the graphqa refactor + the two new tests;
   the shared-trainer bug fix is arguably its own commit given its cross-experiment reach.

## Landmines / notes for whoever resumes

- **`required_splits` is config-derived, not filesystem-derived** (data.py). The raw
  `hf_dataset/` is gitignored, so a filesystem-derived answer collapses to `[]` on a fresh
  checkout and turns a clear "run data prep" error into an obscure loader traceback. The
  declarative `TASKS_WITHOUT_OFFICIAL_VAL` list in config.py was verified against the actual
  download for all 11 tasks — keep them in sync.
- **Do not run training/eval on the login node** — its NVIDIA driver is too old (CUDA init
  fails). Everything GPU goes through Slurm + the container. Reuse `sweep/slurm_launch.sh`
  (`slurm_launch.sh <label> <job_script.sh>`); the container only mounts `/shared`, so scratch
  must live under `/shared` (e.g. `$HOME/.cache/...`), not `/tmp`.
- **`bias_params()` includes `magnetic_q`** as provenance only — no bias module reads it
  (the charge is consumed at data-prep time when eigenvectors are computed). Matches the
  historical v0 `BIAS_PARAMS` and kgqa's convention.
- The cached datasets log `Legacy dataset detected (no metadata.json). Defaulting
  per_graph_versions to 1` — correct for GraphQA, just means the cache predates the metadata
  format. Harmless.
- The proof script for the reload bug lives at `$HOME/.cache/graphqa_reload_proof/` (outside
  the repo). The canary checkpoint used to measure it is
  `./checkpoints/graphqa/004_canary_0000/checkpoint-80`.
