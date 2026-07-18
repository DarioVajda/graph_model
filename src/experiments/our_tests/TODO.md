# our_tests — TODO

## Context

`our_tests` (Family Tree + KG-QA synthetic tasks) is one of the three experiments in
the paper (alongside `tag_benchmarks` and `graphqa`). Two things need doing here: a
**refactor** onto the shared `sweep` utility, and a **correctness re-run** of the paper
numbers. Do the refactor **first** — it makes the re-run just a sweep config instead of a
manual, error-prone effort, and it's what makes the results clean and reproducible.

## 1. Refactor onto the shared `sweep` utility (do this first)

The codebase now runs thorough experiments through the generic `sweep` runner
(`python3 -m sweep <experiment_module> <config.jsonc>`), with `graphqa` as the reference
implementation. `our_tests` still uses its legacy bespoke structure (hand-rolled argparse
in `__main__.py`, the `modeling_gtlm_llama_v0` model, manual result bookkeeping) and
should be migrated to match.

Target structure, mirroring `src/experiments/graphqa/`:

- **`config.py`** — a `RunConfig` dataclass enumerating every run parameter (incl. the
  `spd` / `rrwp` / `magnetic` bias toggles) plus task/graph-type enums, so ablation arms
  are just config axes.
- **`__main__.py`** — a self-contained single-run argparse entrypoint that runs *one*
  resolved config and logs *one* record. It should know nothing about sweeps or job
  submission; the `sweep` runner invokes it once per resolved config, rendering each
  config key to the matching flag. Keep a `--mode` router (`train` / `data_prep`) and an
  `--init` sweep-template writer, as in `graphqa`.
- **`configs/*.jsonc`** — JSONC sweep configs (scalars = fixed, `[a, b, c]` = axis,
  `[{…}, {…}]` = bundle). At minimum: a smoke config and the ablation config
  (`spd/rrwp/magnetic` arms × seeds).
- **`_io.py` / `data.py`** — split dataset building (`data_prep.py`, `data_load.py`,
  `family_tree_*`) and result IO to match the `graphqa` module layout.
- Move to the current model impl (align with `graphqa`'s `impl` selection) instead of
  `modeling_gtlm_llama_v0`, and confirm the fixed `GraphTrainer` bias-reload path is used
  (this is also what closes the reload bug described in §2).

Doing the refactor first means the correctness re-run in §2 is just a sweep config, and
the `our_tests` ablation table drops out in the same comparable format as `graphqa`'s.

## 2. Correctness re-run (blocking — paper numbers are currently understated)

`our_tests` uses the same evaluation path that produced the GTLM reload bug in `graphqa`:

- `__main__.py` / `train_llm_baseline.py` set `load_best_model_at_end=True`, and
- it trains through the shared `GraphTrainer` with `active_params` (graph-bias params
  saved separately) — the exact precondition for the bug.

Before the 2026-07-17 fix, HF's `_load_best_model` restored only the LoRA adapter from
the best checkpoint and left the bias params at their end-of-run values, so the reported
metrics pair a best-step adapter with a mismatched bias. **The paper's `our_tests` GTLM
numbers are therefore understated by an unmeasured amount and must be regenerated on the
fixed pipeline.** (`tag_benchmarks` is *not* affected — its paper numbers came from a
manual best-checkpoint selection + separate load-and-test job, so adapter and bias come
from the same checkpoint. Leave `tag_benchmarks` alone.)

### Ablation plan for the re-run

Decouple the two questions and do **not** gate any arm on how another arm turns out:

- **Base (Standard) arm — unconditional.** Required purely to correct the understated
  paper numbers. Nothing to do with the ablation.
- **`w/o SPD` arm — unconditional, report either way.** This is the arm that resolves
  the cross-paper SPD justification question (GraphQA showed SPD ~neutral; we need to
  know whether it helps here). Whatever it shows, it must inform the SPD wording in the
  paper — no running-and-burying.
- **`w/o RRWP` and `w/o Magnetic` arms — editorial, not conditional on the SPD result.**
  They test *different* biases and say nothing about SPD, so "SPD helped, therefore run
  them" is a non-sequitur. Run them if we want a complete `our_tests` ablation table as
  a paper contribution.
- **Recommended:** if training on these synthetic datasets is cheap, run **all four arms
  in one sweep upfront** rather than staging. It's barely more compute than the base
  re-run we already owe, and it avoids any appearance of outcome-gated experimentation
  (running more arms only after an early result looked favorable). Stage
  (base + `w/o SPD` first) only if the dataset turns out to be expensive.

Seeds: match the `graphqa` ablation convention (`{42, 43, 44}`, sample std / n−1) so the
tables are directly comparable.

## Notes

- Keep the LLM / RGLM baselines (`train_llm_baseline.py`, LLaGA/RGLM prep) working through
  the migration — only the GTLM path needs the bias-reload fix, but the baselines should
  slot into the same sweep/config scheme.
- `tag_benchmarks` needs no re-run; only `graphqa` (already corrected) and `our_tests`
  (this file) were affected by the reload bug.
