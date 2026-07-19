# our_tests — TODO

## Context

`our_tests` (Family Tree + KG-QA synthetic tasks) is one of the three experiments in
the paper (alongside `tag_benchmarks` and `graphqa`). Three things need doing here, in
order: a **refactor** onto the shared `sweep` utility, a **`question_node` probe** ported
from `kgqa`, and a **correctness re-run** of the paper numbers. Do the refactor **first** —
it makes everything downstream a sweep config instead of a manual, error-prone effort, and
it's what makes the results clean and reproducible. The `question_node` probe sits in the
middle because its outcome decides which encoding the final ablation table is built on:
running the ablation first and the probe after would mean either re-running the ablation or
publishing it on a known-suboptimal encoding.

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
  (this is also what closes the reload bug described in §3).

Doing the refactor first means the correctness re-run in §3 is just a sweep config, and
the `our_tests` ablation table drops out in the same comparable format as `graphqa`'s.

## 2. `question_node` probe (do after the refactor, before the ablation)

Port the `question_node` feature from `kgqa` and measure it on the `our_tests` tasks
before committing the ablation table to an encoding.

**What it is.** Today the prompt node carries both question and answer in one node
(`f"Q: {q}\nA: {a}"` — `data_prep.py:41`, `family_tree_prep.py:64`). `question_node`
moves the question text out into its own dedicated QUESTION *prefix* node, so the
model's bidirectional-prefix mask lets every graph token attend to the question. Graph
token representations become question-conditioned (task-specific) instead of
question-blind. See `kgqa/process_dataset.py:308` (`add_prompt_node`) and
`kgqa/config.py:71` for the reference implementation.

**Scope here: two values only, `off` and `isolated`.** `kgqa` also has `all` and
`topics`, which differ only in the QUESTION node's directed OUT-edge set (those edges
feed the SPD/magnetic bias features; token visibility comes from the mask either way).
`isolated` gives QUESTION no edges at all. It was the best-performing `kgqa` arm and is
the cleanest probe of the mask effect alone, so it is the only variant worth carrying
over for now. `off` must stay byte-identical to the historical single-prompt-node format.

**The target-node hazard — the thing to be careful about.** When the question moves out,
the prompt node must **not** be left empty-before-the-answer. If it were, the first answer
token would have no in-node predecessor token to be predicted from, making it ambiguous
which token conditions the first generated one. **The target node must always retain a
non-empty prefix.** Keep the existing `"A: "` delimiter as that prefix — the prompt node
becomes `f"A: {a}"` while QUESTION holds the question text. This mirrors what `kgqa`
already does (`process_dataset.py:339`: `text = f"Answer:{suffix}"`), so the precedent is
proven, not speculative.

Consequences to handle:
- **Label masking still works unchanged in principle** — `GetGraphLabels` finds the
  `question_end` token subsequence and masks through it. With the prompt node reduced to
  `f"A: {a}"`, `"A:"` is still present and still the delimiter. Verify the match lands at
  the right index in the shortened node, and add a test.
- **Derive `question_end` from the tokenizer**, not the hardcoded `[32, 25]`
  (`data_prep.py:158`, `family_tree_prep.py`). `graphqa` already made this change; the
  hardcoded pair silently encodes one tokenizer's ids.
- **`question_node` is a data-prep-time knob**, so it belongs in the dataset cache key
  (as in `kgqa`'s `data_config_key()`) — each value needs its own built dataset.
- Both task families (`family` and `kg_qa`) need the change; they have separate prep
  paths that duplicate the prompt-node construction.

**Deliverable:** a small sweep config (`question_node` ∈ {`off`, `isolated`} × seeds
`{42, 43, 44}`) on the standard arm with all biases on. Whichever wins becomes the fixed
`question_node` value for the §3 ablation, and the comparison itself is reportable.

**The `off` arm IS the §3 base re-run — do not run it twice.** Standard encoding, all
biases on, three seeds, on the fixed pipeline: that is precisely the correctness re-run
§3 owes. It is scheduled here, and §3's base arm is satisfied by it.

**What the `off` arm should show (it is NOT a parity check against the paper).** It runs
the historical *encoding* but on the *fixed* reload path, so it should come out **higher**
than the published number, by the amount the reload bug was costing:

- `off` > paper → expected. The delta is the bug's magnitude on `our_tests`.
- `off` ≈ paper → investigate rather than celebrate. It would mean the best checkpoint
  always landed late enough that the bias barely drifted. Possible, but check before
  assuming.

Reproducing the paper number exactly would mean the fix did nothing, which contradicts
§3's premise. The actual refactor-didn't-change-anything check is cheap and separate: the
byte-identical-graphs test for `question_node="off"`, plus the v0/v2 numerical parity
already pinned in `tests/models/`. Do not use a training run for that purpose.

## 3. Correctness re-run (blocking — paper numbers are currently understated)

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

- **Base (Standard) arm — already run in §2.** This is the `question_node="off"` arm of
  the probe (standard encoding, all biases on, seeds `{42, 43, 44}`), which doubles as the
  correction of the understated paper numbers. Reuse those records; do not re-run.
  This holds whichever encoding wins: both §2 arms are "standard + all biases on",
  differing only in `question_node`, so the ablation's base cell at the winning encoding
  is always one of the two §2 arms verbatim. §3 therefore only needs the three `w/o *`
  arms — 3 arms × 3 seeds, with the base cell reused from §2.
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

All four arms run at the `question_node` value chosen in §2, held fixed across arms.

## Notes

- The reload fix from §3 has **already landed in shared code**: both
  `src/utils/text_graph_trainer.py:133` (the v1 `GraphTrainer` `our_tests` uses today) and
  `text_graph_trainer_v2.py:171` now override `_load_best_model` to restore bias params.
  §3 is therefore about regenerating numbers, not writing the fix.
- Keep the LLM / RGLM baselines (`train_llm_baseline.py`, LLaGA/RGLM prep) working through
  the migration — only the GTLM path needs the bias-reload fix, but the baselines should
  slot into the same sweep/config scheme.
- `tag_benchmarks` needs no re-run; only `graphqa` (already corrected) and `our_tests`
  (this file) were affected by the reload bug.
