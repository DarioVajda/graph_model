# Generalist harness — design

**Status (2026-09-02):** design, nothing built. `PLAN.md` says what the generalist programme is and
in what order its decisions land; this document says what the *code* has to do, file by file, so
that the first consumer — `MOLECULE_GENERALIST.md` — can run on it and the trunk (`PLAN.md` §5)
can run on the same code later without a rewrite. The two documents are kept apart on purpose:
`PLAN.md` changes when a decision changes, this one changes when the harness does.

**Scope of the first build.** Everything the molecule generalist needs and nothing it does not:
schema, registry, one adapter, the mixture, a resumable WSD trainer with branching, an evaluation
plugin system, and the tests that make each of those trustworthy. Trunk-only machinery — the
forgetting loss, the admission fork's regression gate, multi-domain adapters — is *designed for*
here (the seams exist) but built when the trunk needs it. §9 lists what is deferred and why.

---

## 0. What already exists and is reused

The repo has run ~90 molecule sweeps and hundreds of runs elsewhere on machinery that works. The
harness does not replace any of it.

| Need | Reused as-is | Where |
|---|---|---|
| model construction, LoRA + bias parameter selection, collator | `build_model`, `select_active_params`, `build_collator` | `src/experiments/expressiveness/training/dispatch.py` |
| two-LR optimizer (LoRA vs bias), bias-aware save / best-model reload / resume reload | `GraphTrainerV2` | `src/utils/text_graph_trainer_v2.py` |
| bias tensors beside the PEFT adapter | `save_bias_parameters`, `load_bias_parameters` | `src/models/io.py` |
| graph example storage, merging, node ordering, structural features | `TextGraphDataset` (`+`, `assign_label`, `ds_label`) | `src/utils/text_graph_dataset.py` |
| named multi-dataset eval with per-dataset metric prefixes and a pinned selection dataset | the `eval_dataset={name: ds}` pattern and `PerDatasetEvalMixin` | `src/experiments/kgqa/train.py` |
| exact-match and margin-AUROC scorers | `make_compute_metrics`, `make_margin_metrics`, `make_margin_preprocessor`, `answer_token_ids` | `src/utils`, `src/experiments/molecules/evaluate.py` |
| molecule data: generators, Tier-B loaders, scaffold split, encodings, round-trip test | the whole `molecules` package | `src/experiments/molecules/` |
| sweeps, sbatch, arrays, `runs.jsonl`, `report.md` | the `sweep` runner | `sweep/` |
| step timing and peak memory | `StepMemCallback` | `expressiveness/training/instrumentation.py` |

Three things the existing trainers *cannot* do, and which are the reason this harness exists:
train on a weighted mixture of sources with per-source loss accounting; run a horizon-free
schedule that survives a resume and a mixture change; and branch a checkpoint into a decayed,
reportable model without ending the parent. Everything in §D4–D6 is about those three.

---

## 1. Layout

`src/generalist/`, sibling of `src/experiments/`, not a setuptools package (`PLAN.md` §1). Run as
`python -m src.generalist …`, swept as `python -m sweep src.generalist configs/….jsonc`.

```
src/generalist/
├── PLAN.md                    # the programme
├── DESIGN.md                  # this document
├── MOLECULE_GENERALIST.md     # first consumer
├── __main__.py                # argparse: modes train | resume | fork | eval | data_prep | validate
├── config.py                  # RunConfig (one knob = one place) + ForkConfig
├── schema.py                  # D1: Example, validator, answer kinds
├── registry.py                # D2: TaskSpec, held-out enforcement, mixture resolution
├── adapters/
│   ├── __init__.py            # Adapter protocol
│   └── molecules.py           # D3: molecules package → schema, the partition
├── mixture.py                 # D4: weights → per-step draw plan, resumable sampler, batching
├── trainer.py                 # D5: GeneralistTrainer(GraphTrainerV2), per-task loss, schedule
├── schedule.py                # D5: horizon-free WSD segments, re-warm, ratio
├── checkpoint.py              # D5: what a checkpoint contains, write/verify/restore
├── lineage.py                 # D6: results/lineage.json
├── fork.py                    # D6: anneal | admit | adapt
├── evaluate/
│   ├── __init__.py            # D7: Validator protocol, scheduler, registry of validators
│   ├── builtin.py             # D7: the validators listed in §D7.3
│   ├── adaptation.py          # D7: steps-to-target
│   └── report.py              # run record assembly
├── configs/                   # .jsonc for the sweep runner
├── results/                   # runs.jsonl, per_example/, lineage.json — never committed
└── tests → tests/generalist/  # §T
```

The build order is §8. `schema.py` and `registry.py` are first because every other file imports
them and `PLAN.md` §1 already names them as the two that carry the design risk.

---

## D1. Schema — one example format

Every training and evaluation item, from every adapter, is one `Example`. The validator runs on
every item at adapter build time and on a sample at load time; an adapter that emits an invalid
item fails the build, not the run.

```
Example
  task:        str        registry key, e.g. "mol/ring_size", "mol/bace", "mol/chebi20", "mol/g2s"
  domain:      str        "molecules" for everything here
  split:       str        "train" | "val" | "test" | "held_out"
  arm:         str        "graph" | "flat"
  graph:       TextGraph  the existing TextGraphDataset item: node text, edges, prompt_node,
                          question node, and whatever bias features the build computed
  question:    str        the exact question text (also present in the graph's question node)
  answer:      str        the target text, already formatted
  answer_kind: str        "token" | "yesno" | "text" | "smiles"   (§D1.1)
  key:         str        partition key — canonical isomeric SMILES here; opaque elsewhere
  meta:        dict       adapter-owned; molecule id, endpoint name, symmetry classes, …
```

### D1.1 Answer kinds decide scoring, loss masking and generation

| kind | loss span | eval | generation |
|---|---|---|---|
| `token` | the answer tokens | teacher-forced exact match on the answer span | none |
| `yesno` | the single answer token | logit margin `yes − no` → ROC-AUC, AP, `tied_pair_fraction`, `n_distinct` | none |
| `text` | the answer tokens | greedy generation, BLEU-2/4, ROUGE-L, METEOR | `max_new_tokens` per task |
| `smiles` | the answer tokens | greedy generation, validity / round-trip / canonical exact | per task |

The kind is a property of the *task*, held in the registry, and copied onto the example so a
batch can be scored without a registry lookup. A task has exactly one kind.

### D1.2 Formatting is the schema's job, not the adapter's

Chat formatting (D3: instruct weights + chat template, both or neither) and the answer-boundary
convention live in `schema.render(example, tokenizer)`, which returns `input_ids`, `labels` with
the loss span masked, and the answer start offset. Adapters emit *text*; the schema decides how
text becomes tokens. This is the single biggest lever found on KGQA (format v3, +4.6 F1) and it
must be one function, version-stamped, not eight copies.

### D1.3 The validator

`schema.validate(example, spec)` checks: required fields present; `split` is one the spec allows
(a held-out task admits only `held_out`); `answer_kind` matches the spec; for `yesno` the answer
is exactly one of the two configured words; for `smiles` the answer parses under RDKit and equals
its own canonicalization; `key` non-empty; the graph's question node carries `question`
verbatim. The graph itself is validated by `TextGraphDataset` already.

---

## D2. Registry — what is in the mixture, at what weight, from which build

`registry.py` holds one `TaskSpec` per task and is the *only* place that answers those three
questions. A run's registry state is serialised into the run record and into every checkpoint.

```
TaskSpec
  name:          "mol/bace"
  domain:        "molecules"
  adapter:       "molecules"                 # adapters/<name>.py
  kind:          "corpus" | "generator"      # PLAN §3.2, §5: pass caps differ
  answer_kind:   "yesno"
  held_out:      bool                        # refuses any split but held_out
  weight:        float                       # example share, absolute; mixture.py normalises
  loss_norm:     "per_example" | "per_token" # D7a; default per_example
  passes:        int                         # corpus: max passes; generator: fresh per pass
  cap_per_pass:  int | None                  # generator: examples per pass
  metric:        str                         # primary, e.g. "roc_auc"
  verify:        callable | None             # (prediction, example) -> bool, PLAN §3.2
  max_new_tokens: int | None                 # text / smiles kinds
  build_version: str                         # hash of the adapter's inputs (§D3.2)
  eval_splits:   ("val", "test")             # which splits the in-mixture validator scores
```

### D2.1 Held-out enforcement is in two places on purpose

The molecules package refuses to *build* `bond_path`, `longest_chain` and ClinTox without
`held_out_eval` (`data.py`). The registry refuses to put a `held_out: true` task into a training
mixture, and refuses a training example whose `key` appears in any held-out or test role of the
partition (§D3.3). A mixture config that names a held-out task fails at `validate`, before any
data is built.

### D2.2 Resolving a mixture

`registry.resolve(config) -> Mixture` turns the config's task list and weights into normalised
shares, the budget in examples (from the finite sources' pass caps, `MOLECULE_GENERALIST.md`
§2), and the step count at the configured tokens-per-step. It fails if any task's share would
round to fewer than one example per 1000 steps — a task silently contributing nothing is the
`--magnetic-groups` class of bug (`PLAN.md` §10) and it is caught here rather than in a report.

---

## D3. Adapters — existing package → schema

An adapter is a module with three functions and no training logic:

```
build(config, roles) -> None        # materialise every split of every task it owns, on disk
load(task, split, arm) -> Dataset   # a TextGraphDataset-backed dataset of Examples
partition(config) -> Partition      # the role assignment for every key it will ever emit
```

### D3.1 `adapters/molecules.py`

Wraps the molecules package. Tier-A generation goes through `generate_examples`, Tier B through
`prepare_tier_b_graphs` and the scaffold split, encodings through `build_graph_example` /
`build_flat_example`. Two things are new:

* **graph-to-SMILES** — a generator over the train-role pool, one example per molecule per pass,
  target `Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)` for both arms
  (`MOLECULE_GENERALIST.md` §5: the graph carries parity words without the neighbour order that
  gives them meaning, so stereo cannot be a target); the flat arm's input is a randomized SMILES
  from `flat_serialize(canonical=False, seed=…)`, stereo included, which the model must learn to
  drop. Molecules whose graph fails `roundtrip_check` at the `exact` level are excluded from this
  task and counted.
* **ChEBI-20** — a corpus loader with its own three splits, a heavy-atom cap, and the
  disconnected-graph check (`MOLECULE_GENERALIST.md` §6).

The adapter imports the scoring helpers from `molecules/evaluate.py`; it does not re-implement
them. The margin readout and its bf16 quantisation caveat (`project-gtlm-margin-quantization`)
live there and a second copy would drift.

### D3.2 Build versioning

`build_version` is a hash over: the source CSV checksums, the encoding and its options,
`stereo_tags`, `question_node`, the partition rule version, the generator seed, and the schema
version. The dataset cache path includes it. A registry entry whose `build_version` differs from
the one a checkpoint was trained on is a *mixture change* (§D5.4) — legal, but it forces a re-warm
and a lineage entry.

### D3.3 The partition

`Partition` is a mapping `key -> role` plus per-source counts and the overlap ledger. Built once
per `build_version` from the raw sources, before any example is generated, following
`MOLECULE_GENERALIST.md` §3 rules 1–4 with priority `held_out > test > val > train`. Persisted as
`partition.json` next to the built data and re-checked at load time (a sample of training keys is
asserted absent from the test and held-out sets). The disjointness test (§T2) builds it from the
CSVs in the test environment and asserts the role sets are pairwise disjoint.

---

## D4. Mixture and sampler

### D4.1 The draw plan is a pure function of `(mixture, seed, step)`

Resumability requires that step *k* draws the same examples whether the process has been running
since step 0 or was restored at step *k − 1*. So the sampler holds no mutable state that is not in
the checkpoint: it is seeded by `(mixture hash, seed)`, and the composition of step *k* is
computed from *k* alone (per-task counts by a deterministic multinomial with a step-indexed
stream, then per-task cursors into a per-pass permutation). The checkpoint stores the cursor
vector; `sampler.state_dict()` / `load_state_dict()` round-trip it. §T4 asserts that a run
resumed at step *k* produces the same batch keys at *k + 1* as an uninterrupted run.

### D4.2 Generators refresh per pass

A generator task exposes `pass_id`; the adapter's `load(task, "train", arm, pass_id=p)` returns a
fresh draw of `cap_per_pass` examples from the train-role pool under seed `(build seed, p)`.
Builds are cached on disk per pass so a resume does not regenerate. Corpus tasks get a fresh
permutation per pass and stop at `passes`.

### D4.3 Batching and loss accounting (D7)

Batches are **mixed**, bucketed by `(node count, token length)` within a task-agnostic bucket
table — homogeneous batches would make per-task gradient noise a function of task share, which
would confound the mixture-weight readout. Two-level normalisation (D7a): each example's loss is
divided by its own loss-span length (per-example), and the batch loss is the *mean over examples*,
so a task's gradient share equals its example share in expectation. Under gradient accumulation
and DDP the example count is taken across the whole optimizer step, not per micro-batch — the
known accumulation-normalisation footgun — and §T3 pins it.

`per_token` tasks (none in the first build) contribute their span-summed loss divided by the
mean span length of the batch, so that the *task-level* share still matches the example share.

The trainer logs, at every logging step, per-task: examples seen, loss, and the fraction of the
summed gradient norm attributable to that task's examples (measured by per-task backward on a
sampled step every `grad_share_every` steps, not every step). The plumbing smoke run asserts that
the measured share tracks the configured weight within tolerance. That assertion is the reason
this instrumentation exists and it is on by default.

### D4.4 Tokens-per-step

`RunConfig.tokens_per_step` fixes the effective batch in tokens (D7b). The dataloader packs
buckets to that budget; `batch_size` is derived, not configured. Changing it is a config change
with a lineage entry, never a silent effect of a different node's memory.

---

## D5. Trainer, schedule, checkpoints, resume

### D5.1 `GeneralistTrainer(GraphTrainerV2)`

Keeps `create_optimizer`'s two parameter groups (LoRA at `lr`, bias at `bias_lr`) and the bias-aware
`save_model` / `_load_best_model` / `_load_from_checkpoint`. Adds: the mixture dataloader (D4),
per-task loss accounting (D4.3), the schedule (D5.2), the extended checkpoint (D5.3), the
validator hooks (D7), and the `eval_dataset={name: ds}` pattern from kgqa for per-task metrics.
`load_best_model_at_end` is **off** — selection is a fork's job (D6), and Tier-B val has been
measured to anti-rank arms (`molecules/PLAN.md` §8.4).

### D5.2 Schedule: horizon-free WSD

HF's `warmup_stable_decay` requires `warmup + stable + decay == num_training_steps` and silently
runs the remainder at `min_lr` otherwise (`PLAN.md` §5). The trunk has no horizon, so the schedule
is our own `LambdaLR` built from **segments**:

```
Schedule = [Segment(kind, steps, lr_start, lr_end), ...]
  warmup    linear 0 → lr over `warmup_steps`
  stable    constant lr, steps = None (open-ended)
  rewarm    linear from `rewarm_from * lr` → lr over `rewarm_steps`
  decay     cosine or linear lr → lr_min over `decay_steps`
```

* A **training** run is `[warmup, stable]`. It ends when the budget in examples is consumed, or
  never (trunk), and its LR at the end is the stable LR.
* A **resume** appends `rewarm` if any discontinuity is detected (D5.4) and continues `stable`.
* An **anneal fork** (D6) appends `decay` and stops.
* The bias group's LR is `ratio × lr` with `ratio = bias_lr / lr` held constant through every
  segment; a per-segment `ratio_end` exists so the trunk can decay the ratio toward 1 later
  (`PLAN.md` §5) without a new mechanism.

The schedule's position is stored as `(segment index, step within segment)` in the checkpoint;
`global_step` alone is not enough once segments have been appended.

### D5.3 What a checkpoint contains

Written by `checkpoint.write(dir)`, verified by `checkpoint.verify(dir)` before the write is
marked complete (a `COMPLETE` marker is the last file written; a directory without it is never
resumed from):

| file | contents | why |
|---|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | LoRA (PEFT's own format) | as today |
| `bias_parameters.pt` | the graph-bias tensors | as today; the reload bug of 2026-07-17 is why it is separate and why `verify` checks its norm against `state.json` |
| `optimizer.pt`, `scheduler.json` | AdamW moments for both groups; schedule segments + position | resuming with fresh moments runs the first steps at an effectively huge LR (`PLAN.md` §5) |
| `sampler.json` | D4.1 cursor vector, pass ids | deterministic resume |
| `rng.pt` | torch / cuda / numpy / python RNG states | bit-exact resume (§T4) |
| `state.json` | global step, examples per task, tokens seen, bias-norm fingerprint, config hash, architecture hash, mixture hash, registry snapshot, schema version, eval-protocol version | everything a resume or a fork must check |
| `lineage.json` (fragment) | parent checkpoint, mode, config diff | D6 |

`save_total_limit` keeps the last *N* complete checkpoints plus every checkpoint a fork was taken
from (they are marked `pinned` in `state.json` and exempt from rotation).

### D5.4 Resume semantics

`python -m src.generalist resume --from <ckpt|latest> [--config <new.jsonc>]`

1. Refuse if `COMPLETE` is absent, if `architecture hash` differs (parameter shapes would not
   load), or if `schema version` differs (labels would be masked differently — a silent metric
   shift, not a crash).
2. Restore model, bias tensors, optimizer, schedule, sampler, RNG. Assert the restored bias norm
   equals the fingerprint in `state.json` to 1e-6 relative; a mismatch means the
   `bias_parameters.pt` / adapter pairing is wrong and the run aborts before a step.
3. Detect discontinuities: `mixture hash` changed, `tokens_per_step` changed, `lr` or `bias_lr`
   changed, hardware class changed (recorded from `torch.cuda.get_device_name`). Any of them
   appends a `rewarm` segment (D5.2) and writes a lineage entry naming the cause. None of them is
   an error; all of them are recorded.
4. If nothing changed, continue the `stable` segment from the stored position. The next batch's
   keys equal what the uninterrupted run would have drawn (§T4).

`resume --from latest` resolves to the newest complete checkpoint under the run's output dir,
which is what an sbatch chain uses (§D8.3).

---

## D6. Branching and lineage

`python -m src.generalist fork --from <ckpt> --mode anneal|admit|adapt --config <fork.jsonc>`

A fork copies the checkpoint into a *new* run directory with its own `runs.jsonl` line and a
`lineage.json` entry `{child, parent, parent_step, mode, config_diff, created}`. The parent is
pinned (D5.3) and continues untouched.

| mode | what it does | ends with |
|---|---|---|
| `anneal` | appends a `decay` segment of `decay_steps` (default 10 % of the parent's steps so far) to `lr_min`, trains on the parent's mixture, runs the full validator set at the end | the reportable model for that milestone |
| `admit` | adds a candidate task to the mixture at a configured weight, appends `rewarm`, trains a fixed budget, runs the regression suites and applies the four-part criterion from `PLAN.md` §5, written into the fork config *before* it runs | pass / fail in the run record; the fork is discarded either way |
| `adapt` | trains on **one held-out task only**, from the parent *and* from base Llama with identical config, evaluating every `eval_steps`, and records steps-to-target for a configured target | the adaptation-efficiency number, `PLAN.md` §3.3 |

The molecule generalist uses `anneal` (once, at the end) and `adapt` (three held-out tasks × two
starting points × seeds). `admit` is built as far as its config and lineage go; its regression
suites are the trunk's validators and land with the trunk.

`results/lineage.json` is append-only and is the file `PLAN.md` §3.4 says to fix the fields of
before the first admission. The fields above are that fixing.

---

## D7. Evaluation as plugins

### D7.1 The `Validator` protocol

```
class Validator(Protocol):
    name: str
    cadence: "steps:<n>" | "milestone" | "end" | "manual"
    needs: set[str]          # {"model", "tokenizer", "eval_sets", "train_sampler", "base_model"}
    def run(self, ctx: EvalContext) -> dict[str, float | list | str]
```

`EvalContext` carries the model, tokenizer, the registry, the per-task eval datasets for the
splits the validator asks for, the current step and schedule position, and a scratch directory.
A validator returns a flat metric dict; keys are namespaced `<validator>/<task>/<metric>` by the
harness, logged to the trainer's `log_history`, written to the run record, and (when a
`selection` is configured for forks) used for it.

Validators are registered by name in `evaluate/__init__.py` and enabled per run in the config:

```jsonc
"validators": [
  {"name": "in_mixture",  "cadence": "steps:500"},
  {"name": "held_out",    "cadence": "milestone"},
  {"name": "bias_norm",   "cadence": "steps:500"},
  {"name": "grad_share",  "cadence": "steps:200"},
  {"name": "base_exact",  "cadence": "milestone"},
  {"name": "perm_spread", "cadence": "end"},
  {"name": "throughput",  "cadence": "steps:50"}
]
```

A validator that raises is logged and skipped; it never loses a training run that already cost
GPU-hours (the `_per_example` contract in `molecules/train.py`). A validator that returns a key
it did not declare fails the plumbing smoke, so metric names cannot drift silently.

### D7.2 Protocol versioning

Every validator declares `protocol_version`. Generation config (greedy, per-task
`max_new_tokens`), answer extraction and metric implementation live in the validator, not the
trainer, and the set of `(validator, version)` pairs goes into `state.json` and the run record.
A checkpoint evaluated under a different version than it was trained beside is flagged in the
report. This is `PLAN.md` §3.4's "eval protocol, version-stamped as files".

### D7.3 Built-in validators (first build)

| name | what | notes |
|---|---|---|
| `in_mixture` | per task, on `val` and `test`: the metric for its `answer_kind` (D1.1) | teacher-forced for `token`/`yesno`, generative for `text`/`smiles`; per-endpoint for Tox21 / SIDER, plus `tied_pair_fraction`, `n_distinct`, `pos_rate` on every `yesno` task |
| `held_out` | zero-shot on every `held_out` task | same scorers; never trains |
| `bias_norm` | L2 norm of the bias tensors | `feedback-verify-nulls-are-real`; also the resume fingerprint |
| `grad_share` | measured per-task gradient share vs configured weight | D4.3; the smoke run asserts on it |
| `base_exact` | adapters off, logits on a fixed text batch equal base Llama's to bf16 tolerance | Property 2; meaningless under D4 arm B/C and says so |
| `perm_spread` | flat arm: AUROC spread over 10 randomized SMILES per test molecule, stratified by symmetry class; graph arm: asserted ≤ 1e-4 | `molecules/PLAN.md` §6 |
| `throughput` | wall-clock s/it, peak GB, tokens/s | `feedback-throughput-metric`: s/it, not `step_ms_mean` |
| `per_example` | the molecules per-example error / geometry report on `test` | wraps `analysis.write_per_example_report` |

`adaptation` (D6 `adapt`) is a fork mode, not a validator, because it trains.

### D7.4 Selection

Training runs do not select. A fork may declare `selection: {"metric": "<key>", "split": "val"}`
for its own purposes (e.g. `adapt` needs a target), and the harness refuses any selection key
containing `test`. The report shows the annealed-final number as the headline and the best-val
number beside it with its selection gain, the way `molecules/train.py` records `_last` beside the
selected score.

---

## D8. Command line, sweeps, Slurm

### D8.1 Modes

| mode | does |
|---|---|
| `validate` | resolve the config, build the registry, check the partition, print the mixture table and step budget; no GPU |
| `data_prep` | run every adapter's `build` for the config's tasks and arms; CPU job |
| `train` | a fresh run: `[warmup, stable]` for the resolved budget |
| `resume` | D5.4 |
| `fork` | D6 |
| `eval` | run a set of validators on a checkpoint, no training; writes an `eval` record |

`--init <name>` writes a sweep config under `configs/`, as the template experiment does.

### D8.2 One `RunConfig`, one place

Every knob is a field with a default, validated in `validate()`. Groups: backbone and bias
architecture (as `molecules/config.py`, plus the D2 sharing and D6 dtype fields when they land);
tasks and weights (`tasks: [{name, weight, passes, cap_per_pass}]`); `tokens_per_step`,
`loss_norm` default; schedule (`warmup_steps`, `rewarm_steps`, `lr`, `bias_lr`, `lr_min`);
checkpointing (`save_steps`, `save_total_limit`); validators; seed and data seed. The config hash
in `state.json` is over this object with `run_name`, `output_dir` and Slurm fields excluded.

### D8.3 Slurm chain

`frida` caps walltime at 7 days and the trunk is longer; the molecule generalist is not, but it
runs on the same path so the chain is proven on a cheap run first (`PLAN.md` §10). One sbatch
script per chunk, `--time` sized to the *window* (`feedback-fit-jobs-to-window`), body
`python -m src.generalist resume --from latest` after the first chunk's `train`, requeue-safe
because every chunk ends with a complete checkpoint and a chunk killed mid-write leaves no
`COMPLETE` marker. The `sweep` runner's `execution.sbatch` block is reused for single runs;
chaining adds `chain: {chunks: N, dependency: afterany}` which the runner expands to N dependent
jobs. Shared inductor cache across chunks (`project-ddp-flex-bucketing`).

Job scripts live under `/shared`, never in the node-local scratch (`feedback-submit-to-slurm`).

---

## T. Tests

`tests/generalist/`. Each row is a test file; the harness is not "built" until every row passes
on CPU with a tiny model except where noted.

| | test | asserts |
|---|---|---|
| T1 | `test_schema.py` | validator accepts every adapter's sample and rejects each malformed field; `render` masks exactly the answer span; format version pinned by a golden token sequence |
| T2 | `test_partition.py` | roles pairwise disjoint on the real CSVs; ClinTox and held-out keys absent from every training source; priority order on conflict; overlap ledger counts match a hand count on a fixture |
| T3 | `test_loss_accounting.py` | per-task gradient share equals example share on a synthetic 3-task batch; unchanged under accumulation 1 vs 4 and under 1 vs 2 ranks (CPU gloo) |
| T4 | `test_resume.py` | train 6 steps; train 3, checkpoint, resume 3: identical batch keys, loss trajectory and parameters to bf16 tolerance; a missing `COMPLETE` is refused; a bias-norm mismatch aborts |
| T5 | `test_schedule.py` | segment LR values at boundaries; `rewarm` appended on each discontinuity class and not otherwise; bias ratio constant; position survives a checkpoint |
| T6 | `test_registry.py` | held-out task in a training mixture fails `validate`; sub-threshold share fails; budget and step count computed as documented |
| T7 | `test_fork.py` | `anneal` ends at `lr_min`; parent pinned and untouched; lineage entry fields; `adapt` runs from parent and base with identical configs |
| T8 | `test_validators.py` | each built-in declares its keys and returns exactly them; a raising validator does not abort training; protocol version in the record |
| T9 | `test_molecules_adapter.py` | graph-to-SMILES target is stereo-free and equals the stereo-flattened `roundtrip_check` expectation; flat input is a valid randomized SMILES of the same molecule; a stereo mark in a prediction is scored as an error; ChEBI cap and disconnected check; Tox21 absent-label counts |
| T10 | `test_smoke_gpu.py` *(GPU, Slurm)* | 1B, three maximally different tasks (`yesno`, `token`, `smiles`), 200 steps: `grad_share` within tolerance, no task at zero, resume mid-run, one `anneal` fork, every validator ran |

The cross-check in `MOLECULE_GENERALIST.md` (a single-task mixture reproduces the molecules
trainer within seed noise) is not a unit test; it is a run, recorded in that file.

---

## 8. Build order

- [ ] **D1** `schema.py` + T1
- [ ] **D2** `registry.py` + T6
- [ ] **D3** `adapters/molecules.py`, partition, graph-to-SMILES, ChEBI-20 + T2, T9
- [ ] **D4** `mixture.py` + T3
- [ ] **D5** `schedule.py`, `checkpoint.py`, `trainer.py` + T4, T5
- [ ] **D7** `evaluate/` protocol and built-ins + T8
- [ ] **D6** `fork.py`, `lineage.py` + T7
- [ ] **D8** `__main__.py`, `config.py`, sweep config, chain script
- [ ] **T10** smoke run on Slurm; per-source loss curves read; mixture weights revised if needed
- [ ] cross-check run (`MOLECULE_GENERALIST.md` checklist)
- [ ] arm 2

Estimated at roughly a week of building before T10, on the strength of how much §0 reuses.

---

## 9. Deferred, with the seam that keeps it possible

| deferred | seam already in the design |
|---|---|
| `forgetting.py` (KL-to-base on text batches) | a `text` domain adapter emitting single-node graphs is just another task; the teacher is `base_exact`'s adapters-off path |
| the admission regression gate | `admit` mode exists; the four suites are validators registered by name |
| adapters for graphqa / kgqa / tag / relbench / clrs | the `Adapter` protocol; each is a `build / load / partition` triple |
| D4 arm B/C (unfrozen `W_q`/`W_k`, full fine-tune) | `active_params` and the optimizer groups; `base_exact` reports itself meaningless when the backbone moves |
| ZeRO / sharded optimizer state | `checkpoint.py` owns the optimizer file; sharding changes its writer, not its contract |
| DDP mixed batches with cross-rank example counts | T3 already runs at 2 ranks on CPU |

## 10. Open decisions

* **`tokens_per_step`** for the molecule generalist. Molecules are short (Levi N ~ 52, few hundred
  tokens); the value is chosen from the T10 smoke's measured s/it, not from a round number.
* **`decay_steps`** for the anneal fork: 10 % of parent steps is the default; the smoke run says
  whether the annealed model has settled by then.
* **Heavy-atom cap** for ChEBI-20 and whether `max_spd 32` moves — both wait on the clamp sweep
  named in `molecules/PLAN.md` §8.4.3 and §9.
