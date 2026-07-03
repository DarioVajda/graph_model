# TODO: refactor the KGQA experiment to work with the `sweep` runner

Goal: make `python3 -m sweep src.experiments.kgqa configs/<file>.jsonc` work exactly
like it does for `src.experiments.expressiveness` — the reference integration to
imitate throughout.

## The contract the sweep runner imposes (what "works with sweep" means)

The runner (`sweep/execute.py`) is experiment-agnostic and never imports the
experiment. It expands a JSONC config into flat per-run parameter dicts and invokes
the experiment once per run as a subprocess:

```
python -m src.experiments.kgqa <flags> --runs-jsonl <path> --run-name <name> --sweep-id <id>
```

So the experiment must be a **standalone argparse program that runs exactly ONE
configuration** and satisfies these rules (see `render_flags` in `sweep/execute.py`):

1. **Flag naming**: every config key `some_key` is rendered as `--some-key`
   (underscores → hyphens). Argparse maps hyphens back to `args.some_key`, so config
   keys map 1:1 to parser flags.
2. **Booleans** are rendered as `--key` / `--no-key` → every bool flag must use
   `argparse.BooleanOptionalAction` (NOT `store_true`).
3. **Lists** are rendered comma-joined as a single value (`--key v1,v2`) → list
   flags need a comma-splitting `type=` function (see `_int_list` in
   `expressiveness/__main__.py`).
4. **`None`** values are omitted entirely → a key whose default is `None` means
   "feature off / use internal default".
5. **Bookkeeping flags**: must accept `--runs-jsonl`, `--run-name`, `--sweep-id`
   (the runner always appends them, in every mode — including data_prep).
6. **Result logging**: at the end of a run, append ONE JSON line
   (hyperparameters + result metrics + `sweep_id`/`sweep_run`) to the
   `--runs-jsonl` path. That is what `sweep/report.py` aggregates into `report.md`.
7. Unknown keys in a config become unknown flags → argparse errors → run FAILS.
   This is the desired fail-fast behavior; additionally a `RunConfig.validate()`
   should reject *accepted-but-unsupported* combinations with a clear message.
8. Conventions (not hard requirements, but match expressiveness): a `--mode
   {train,data_prep}` router, an `--init <name>` sweep-template writer into
   `configs/`, a `results_dir` pointing at the experiment's own `results/` dir, and
   a fallback `results/train_runs.jsonl` for standalone (non-sweep) runs.

## Known incompatibilities in the current kgqa code (why each step below exists)

- `__main__.py` and `process_dataset.py` use **underscore flags** (`--model_name`)
  and **`store_true` booleans** (`--k_hop_directed`, `--no_gradient_checkpointing`,
  `--rcm/--no_rcm`) — both violate rules 1–2.
- `--active_params` uses `nargs="+"` — incompatible with comma-joined rendering
  (rule 3).
- `--wandb_project` defaults to `"GraphLLM"` with a `"none"` string sentinel —
  should default to `None` (= no tracking, flag omitted), like expressiveness.
- **`--run_name` collides with the runner's reserved `--run-name`** (same argparse
  dest after hyphenation). The user-facing knob must be dropped; the sweep's
  `--run-name` becomes the run identity.
- No `--runs-jsonl` / `--sweep-id` flags, and no JSONL record is written — results
  only go to stdout/wandb (rules 5–6).
- Data prep is a separate program (`process_dataset.py`) with its **own** argparse
  and its own defaults dict; `__main__.py` calls `load_data()` with no overrides,
  so a sweep could never vary data-config keys (`rel_mode`, `max_nodes`, …).
- `process_dataset` uses `base_model`, `__main__` uses `model_name` — same thing,
  two names. Also `spd_cutoff` (data) vs `BIAS_PARAMS["max_spd"]` (model) are two
  knobs for one concept, both 64 today; and `magnetic_m` is silently taken from
  `BIAS_PARAMS["magnetic_dim"]` in `__main__.py` (line ~106) while
  `process_dataset` has its own `magnetic_m` default — they only agree by
  coincidence (both 128).
- `--seed` is baked into the dataset **cache key** (`config_key`) AND used as the
  training seed → sweeping the training seed would force a full dataset rebuild
  per seed.
- Checkpoints go to `./checkpoints/kgqa/{run_name}` where the default `run_name`
  does not include the seed → two runs differing only in seed (or two parallel
  `per_config` sbatch jobs) would write to the same directory.
- No `--max-steps` knob → no cheap smoke-test runs.
- `test.py` is a stale copy of the *template* experiment (docstring says
  "template", imports the old `...train.eval.evaluate_checkpoint` stack).
- `train.sbatch` hand-rolls what `execution: {mode: "sbatch", ...}` now does.

---

## Step-by-step plan

### Phase 1 — single source of truth: `config.py` with a `RunConfig` dataclass

- [ ] 1.1 Create `src/experiments/kgqa/config.py` modeled on
  `expressiveness/config.py`. Define one `RunConfig` dataclass holding EVERY knob
  both entry points read, merging today's `process_dataset.DEFAULTS`, the
  `__main__.parse_args` defaults, and `BIAS_PARAMS`:
  - **what to run**: `mode: str = "train"` (`"train" | "data_prep"`).
  - **data-prep keys** (these determine the `.gtds` cache directory):
    `rel_mode="last_1"`, `max_nodes=512`, `n_max=20`, `versions=8` (rename of the
    cryptic `k` — do NOT keep a key named `k`, it reads like `k_hop`),
    `max_length=1024`, `rcm=True`, `data_seed=42`.
  - **shared model/bias keys**: `model_name="meta-llama/Llama-3.2-1B"` (unify
    `base_model`/`model_name` under `model_name`), `spd=True`, `max_spd=64`
    (single knob; data prep uses it as the SPD cutoff, the model as the bucket
    cap — drop `spd_cutoff`), `magnetic=True`, `magnetic_dim=128`,
    `magnetic_q=0.25`, `magnetic_m=128` (single knob used by BOTH data prep and
    the model/collator — kill the `magnetic_dim`-as-`magnetic_m` aliasing in
    `__main__.py`).
  - **train keys**: `num_epochs=5`, `batch_size=2`, `accumulation_steps=4`,
    `lr=3e-4`, `bias_lr=5e-3` (rename `learning_rate`/`bias_learning_rate` to
    match expressiveness), `eval_steps=100` (rename `eval_every`),
    `max_steps=-1` (NEW — quick smoke tests), `seed=42` (training seed —
    now decoupled from `data_seed`), `lora_r=16`, `k_hop=2`,
    `k_hop_directed=False`, `graph_attn_impl="flex"`, `dtype="bf16"`,
    `gradient_checkpointing=True` (positive-sense bool; replaces the inverted
    `--no_gradient_checkpointing`), `active_params=("graph_bias",)`,
    `num_workers=<pick>` (optional but recommended — expose
    `dataloader_num_workers` like expressiveness does).
  - **generative-eval keys**: `gen_max_new_tokens=128`, `gen_max_samples=None`.
  - **tracking**: `wandb_project=None` (None = off; drop the `"none"` sentinel).
- [ ] 1.2 Give `RunConfig` the helper methods the entry points need:
  - `bias_params()` → the dict currently hardcoded as `BIAS_PARAMS`, built from
    `spd/max_spd/magnetic/magnetic_dim/magnetic_q`.
  - `lora_config()` → the dict currently inlined in `__main__.main` (`None` when
    `lora_r == 0`).
  - `data_config_key()` → replaces `process_dataset.config_key(cfg)`; built ONLY
    from data-affecting fields (`model_name`, `rel_mode`, `max_nodes`, `n_max`,
    `versions`, `max_spd`, `magnetic_q`, `magnetic_m`, `rcm`, `data_seed`,
    `max_length`). **Decision needed**: keeping the exact current key format
    preserves existing caches in `processed_datasets/`; renaming fields changes
    the string. Either replicate the old format exactly, or accept a one-time
    rebuild — pick one consciously and note it.
  - `validate()` → fail fast on: `rel_mode` not in `{last_1,last_2,full}`,
    `graph_attn_impl` not in `{flex,eager}`, `dtype` not in `{bf16,fp32}`,
    `lora_r < 0`, `n_max < 1`, neither `spd` nor `magnetic` enabled, etc.
    Return `self` so `__main__` can chain (`config_from_args(args).validate()`).
- [ ] 1.3 Copy the 15-line `append_jsonl` helper from `expressiveness/_io.py` into
  `src/experiments/kgqa/_io.py` (it is package-private there; experiments stay
  independent — same pattern as everything else in this repo).

### Phase 2 — rewrite `__main__.py` as the standalone single-run program

- [ ] 2.1 Rewrite `parse_args` → `build_parser()`, one flag per `RunConfig` field,
  defaults pulled from a `RunConfig()` instance (so defaults live in exactly one
  place). Follow the rendering rules:
  - all flags hyphenated: `--model-name`, `--lora-r`, `--k-hop`, `--rel-mode`,
    `--max-nodes`, `--gen-max-samples`, …
  - bools via `argparse.BooleanOptionalAction`: `--k-hop-directed`, `--rcm`,
    `--gradient-checkpointing`, `--spd`, `--magnetic`.
  - `--active-params` with a comma-splitting `type=` (e.g.
    `lambda s: [x for x in s.split(",") if x.strip()]`), NOT `nargs="+"`.
  - `--gen-max-samples` default `None` (omitted flag = full dev set — already the
    semantics; `None` handling comes free per rule 4).
  - REMOVE the user-facing `--run_name` flag entirely (reserved for the runner).
  - ADD the three bookkeeping flags exactly as expressiveness does:
    `--runs-jsonl`, `--run-name`, `--sweep-id`, all default `None`.
  - ADD `--mode {train,data_prep}` and `--init [NAME]`.
- [ ] 2.2 Add `config_from_args(args) -> RunConfig` (mirror
  `expressiveness/__main__.py::config_from_args`), ending in `.validate()`.
- [ ] 2.3 Add the `TEMPLATE` JSONC string + `_do_init(name)` writing to
  `src/experiments/kgqa/configs/<name>.jsonc`. The template must contain:
  - `"name"`, `"results_dir": "src/experiments/kgqa/results"`, and a full
    `"execution"` block (copy the sbatch block from the expressiveness template —
    partition/account/gpus/container are cluster facts that carry over; adjust
    `time`/`mem` to KGQA's needs, cf. the old `train.sbatch`: 8 CPUs / 64G / 12h).
    - Include `execution.sbatch.max_concurrent` (a per-config concurrency cap): with
      `granularity: per_config` it submits the runs as one throttled Slurm job array
      (`--array=0-(K-1)%N`) so at most N configs run at once, each freed slot picking
      up the next pending config. **This flag did NOT exist when the expressiveness
      experiment was built** — the expressiveness sbatch template predates it and
      submits one independent job per config with no cap. We explicitly want it here:
      KGQA runs are heavier (a 1B model + LoRA vs. the small expressiveness models),
      so bounding how many land on the cluster at once matters. Set a sensible
      default in the template (e.g. `"max_concurrent": 4`) and comment it.
  - example axes that make sense here: `"lora_r": [8, 16]`, `"k_hop": [0, 2]`,
    `"seed": [0, 1, 2]`; a bundle example for data configs, e.g.
    `"data_profile": [{"rel_mode": "last_1", "max_nodes": 512}, ...]`;
  - all fixed scalars with comments, `"wandb_project": null`.
- [ ] 2.4 Slim `main()` to a dispatcher (mirror expressiveness):
  `--init` → write template and exit; `mode == "data_prep"` →
  `run_data_prep_mode(cfg)`; else `run_train_mode(cfg, runs_jsonl=args.runs_jsonl,
  run_name=args.run_name, sweep_id=args.sweep_id)`. Move ALL the current
  model/trainer construction out of `__main__.py` into `train.py` (Phase 4).

### Phase 3 — make data prep a mode of the same config

- [ ] 3.1 In `process_dataset.py`: keep the whole pipeline (naming, Levi, CVT
  collapse, `AnswerLabelMasker`, `process_split`, …) but make it consume a
  `RunConfig` instead of its own argparse namespace. Replace `DEFAULTS`,
  `config_key`, and its `parse_args`/`main` with a
  `run_data_prep_mode(cfg, splits=("train","dev","test"))` driver that:
  - resolves `out_dir = OUTPUT_ROOT / cfg.data_config_key()`,
  - **skips splits whose `.gtds` directory already exists** (idempotency — with
    `granularity: per_config` many parallel sbatch jobs may share one data
    config; expressiveness solves this with an flock in
    `load_or_create_dataset`. Minimum viable: skip-if-exists + document "run a
    data_prep sweep first"; better: an `fcntl.flock` around the build like
    expressiveness's `datasets.py`),
  - uses `cfg.data_seed` (not `cfg.seed`) for the augmentation RNG,
  - still writes `config.json` provenance into `out_dir`.
  - Note: `use_gpu` was a data-prep-only CLI switch; either promote it to a
    `RunConfig` field (fine — train just ignores it) or hardcode `True`.
- [ ] 3.2 Update `load_data.py`: `load_data(cfg)` takes the `RunConfig` and
  resolves `OUTPUT_ROOT / cfg.data_config_key()`; keep the helpful
  FileNotFoundError, but point it at the new command
  (`python -m src.experiments.kgqa --mode data_prep <flags>`). Drop the
  `**overrides`/`SimpleNamespace` machinery — the config object IS the override
  mechanism now.
- [ ] 3.3 Keep `python -m src.experiments.kgqa.process_dataset` working (optional
  but cheap): a thin `main()` that builds a parser from the same
  `build_parser()`/`config_from_args` and calls `run_data_prep_mode` — or delete
  the entry point and update the README/docstrings that mention it (they're in
  `__main__.py`, `load_data.py`, and `process_dataset.py` docstrings).

### Phase 4 — `train.py`: run ONE config, log ONE record

- [ ] 4.1 Create `src/experiments/kgqa/train.py` with
  `run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None)` containing
  the body of today's `__main__.main()`:
  - derive the internal run identity from the sweep bookkeeping:
    `run_name = f"{sweep_id}_{run_name}"` when provided, else the old pattern
    **plus the seed** (`kgqa_lora{r}_khop{k}_{impl}_s{seed}`) — this fixes both
    the checkpoint-collision and wandb-name-collision problems.
  - `output_dir = f"./checkpoints/kgqa/{run_name}"` (now unique per run).
  - use `cfg.bias_params()` everywhere `BIAS_PARAMS` was used; use
    `cfg.magnetic_m` for the collators (NOT `magnetic_dim`).
  - pass `max_steps=cfg.max_steps` and (if added) `dataloader_num_workers` into
    `TrainingArguments`.
  - `report_to = "wandb" if cfg.wandb_project else "none"`; call
    `set_wandb_project(cfg.wandb_project)` BEFORE building the trainer (today it
    is set after — works only by accident of lazy wandb init), and keep the
    `wandb.finish()` at the end.
  - keep the final `trainer.evaluate(test_dataset, metric_key_prefix="test")`.
- [ ] 4.2 Decide the fate of `save_run_metadata` (old stack, `src/train/trainer.py`):
  under a sweep the resolved config is already saved to
  `<sweep_dir>/resolved/<run>.json`, and the JSONL record (4.3) carries the
  hyperparameters — recommend dropping the call and the `...train` import (keep
  `select_active_params`, `print_trainable_parameters`, `get_device`).
- [ ] 4.3 Add `_save_train_record(...)` (mirror expressiveness
  `training/train.py::_save_train_record`) appending via `_io.append_jsonl` to
  `runs_jsonl`, with fallback
  `src/experiments/kgqa/results/train_runs.jsonl` for standalone runs. Record:
  - bookkeeping: `mode`, `sweep_id`, `sweep_run`, `run_name`;
  - hyperparameters: `model_name`, `lora_r`, `k_hop`, `k_hop_directed`,
    `graph_attn_impl`, `dtype`, `lr`, `bias_lr`, `num_epochs`, `batch_size`,
    `accumulation_steps`, `max_steps`, `seed`, and the data keys (`rel_mode`,
    `max_nodes`, `n_max`, `versions`, `magnetic_m`, `data_seed`);
  - results: best-checkpoint dev metrics (`eval_f1`, `eval_hits1`,
    `eval_hit_star`, teacher-forced `eval_em_accuracy` if present) AND the test
    metrics (`test_f1`, `test_hits1`, `test_hit_star`). The trainer's final
    `evaluate()` returns dicts — capture both instead of only printing them.
  - Write the record inside a `try/finally`-ish flow AFTER test eval; a run that
    crashes mid-train simply has no line (that's how the runner counts failures).
- [ ] 4.4 `evaluate.py` needs no interface changes (`KGQAGraphTrainer` /
  `generative_eval` are internal) — just re-point imports if module paths move.

### Phase 5 — housekeeping

- [ ] 5.1 Delete `train.sbatch` — its role is replaced by the config's
  `execution: {"mode": "sbatch", ...}` block. (Carry its resource numbers and the
  creds/cwd notes into the TEMPLATE comments before deleting.)
- [ ] 5.2 Fix or delete `test.py` (stale template copy; wrong docstring, old-stack
  `evaluate_checkpoint` import that doesn't match the GTLM/KGQA generative eval).
  If checkpoint re-evaluation is wanted, rewrite it around `generative_eval` +
  `load_data(cfg)`; otherwise delete and note "re-run eval via a sweep config"
  in the README.
- [ ] 5.3 `mkdir` conventions: create `configs/` (check in one example sweep, e.g.
  `configs/example.jsonc` from the template) and let `results/` be created by the
  runner. Extend `src/experiments/kgqa/.gitignore` with `results/`, `logs/`
  (decide whether sweep outputs should be committed — expressiveness currently
  leaves its `results/` untracked-but-present; match whatever you want going
  forward, just decide once).
- [ ] 5.4 Update `README.md`: replace the two "Run:" snippets with the sweep
  workflow, run FROM THE REPO ROOT (both dataset paths and `results_dir` are
  repo-root-relative — same caveat as the expressiveness README):
  ```bash
  python3 -m src.experiments.kgqa --init my_sweep
  # edit src/experiments/kgqa/configs/my_sweep.jsonc
  python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/my_sweep.jsonc  # with "mode": "data_prep" first
  python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/my_sweep.jsonc  # then "mode": "train"
  python3 -m sweep.report src/experiments/kgqa/results/my_sweep                      # after sbatch jobs finish
  ```
  Keep the answer-coverage-ceiling tables untouched.

### Phase 6 — verification (do these in order)

- [ ] 6.1 `python3 -m src.experiments.kgqa --help` — all flags hyphenated, bools
  show `--x/--no-x`, the three bookkeeping flags present, no `--run-name` other
  than the runner's.
- [ ] 6.2 Round-trip unit test (add next to `tests/test_sweep_expansion.py`, e.g.
  `tests/test_kgqa_flags.py`): for a representative resolved-config dict, assert
  `build_parser().parse_args(render_flags(params))` reproduces every value
  (import `render_flags` from `sweep.execute`). This pins the contract so future
  flag edits can't silently break sweeps.
- [ ] 6.3 Smoke sweep, local: write `configs/smoke.jsonc` with
  `execution.mode: "local"`, tiny knobs (`max_steps: 4`, `eval_steps: 2`,
  `gen_max_samples: 8`, `num_epochs: 1`, `wandb_project: null`) and one 2-value
  axis (e.g. `"k_hop": [0, 2]`). Run `"mode": "data_prep"` first (or reuse an
  existing cache by matching its config), then `"mode": "train"`. Verify the
  sweep dir contains `config.jsonc`, `resolved/*.json`, `logs/*.log`,
  `runs.jsonl` with 2 lines, and a `report.md` whose columns include the F1/Hits
  metrics.
- [ ] 6.4 sbatch dry run: same config with `execution.mode: "sbatch"` +
  `"dry_run": true` — inspect `sbatch_commands.sh` and `jobs/*.sh` for correct
  flags/paths, then (optionally) submit one real job and aggregate with
  `python3 -m sweep.report <sweep_dir>`.
- [ ] 6.5 Standalone regression: `python3 -m src.experiments.kgqa --max-steps 4
  --gen-max-samples 8` with no bookkeeping flags — confirm it still trains and
  appends to `results/train_runs.jsonl` (the fallback path).

## Open decisions (make them at Phase 1, they ripple through everything)

1. **Cache-key compatibility** (1.2): replicate the old `config_key` string to
   keep existing `processed_datasets/` caches, or accept a one-time rebuild.
2. **`data_seed` vs `seed` split** (1.1): required for cheap seed sweeps; means
   old caches keyed on `seed=42` correspond to `data_seed=42`.
3. **Where sweep results live / git policy** (5.3).
4. **Keep or delete** `process_dataset` as a standalone entry point (3.3) and
   `test.py` (5.2).
