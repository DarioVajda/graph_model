# Template experiment

A minimal, **copy-me** experiment that trains `GraphLlama` on a tiny synthetic
graph task (predict a prompt node's out-degree from its neighbourhood). Its real
job is to be the reference layout for a new experiment that plugs into the generic
[`sweep`](../../../sweep) runner: one standalone single-run program, one
`RunConfig` for all the knobs, one JSONL record per run.

Everything is run **from the repo root**.

## Layout

| File | Responsibility |
|------|----------------|
| `config.py` | `RunConfig` dataclass — **every** knob lives here, once. `validate()`, `bias_params()`, `lora_config()`. |
| `data.py` | `load_data(cfg, tokenizer)` — builds the dataset and computes the features `cfg` enables. **Replace this for a real task.** |
| `train.py` | `run_train_mode(cfg, …)` — builds the model, trains one config, appends one JSONL record. |
| `__main__.py` | The standalone argparse program: `build_parser`, `config_from_args`, `--init`, and a thin `main()` dispatcher. |
| `test.py` | Re-evaluate a saved checkpoint on the test split. |
| `_io.py` | `append_jsonl` (package-private). |
| `configs/` | Sweep configs (JSONC). `--init` writes one here. |
| `results/` | Per-sweep output dirs + the standalone `train_runs.jsonl` fallback. |

## Run it standalone (single config, quick iteration)

```bash
# one config with defaults
python3 -m src.experiments.template --k-hop 2 --seed 0

# a fast smoke test (cap the optimizer steps, tiny dataset, no tracking)
python3 -m src.experiments.template --max-steps 4 --num-epochs 1 --num-samples 24

# turn features / LoRA on and off
python3 -m src.experiments.template --no-magnetic --no-lora --k-hop 0
```

`python3 -m src.experiments.template --help` lists every flag. A standalone run
appends its record to `src/experiments/template/results/train_runs.jsonl` and
saves checkpoints under `./checkpoints/template/<run_name>`.

Re-evaluate a checkpoint (pass the same data/bias flags you trained with):

```bash
python3 -m src.experiments.template.test --checkpoint-path ./checkpoints/template/<run_name> --include-f1
```

## Run it with `sweep` (many configs)

```bash
# 1. write a sweep template into configs/
python3 -m src.experiments.template --init my_sweep

# 2. edit src/experiments/template/configs/my_sweep.jsonc  (set axes, execution mode)

# 3. run every resolved config (locally, sequential)
python3 -m sweep src.experiments.template src/experiments/template/configs/my_sweep.jsonc

# 4. aggregate the runs.jsonl into a report (auto-written after a local sweep;
#    run manually after sbatch jobs finish)
python3 -m sweep.report src/experiments/template/results/my_sweep
```

In the config, a **list** value becomes a sweep axis and a **`[{…}, {…}]`** list
of objects is a *bundle* of keys that vary together (see the comments in the
generated template). Every config key maps 1:1 to a CLI flag.

### On the cluster (sbatch)

Set `execution.mode: "sbatch"` in the config. With `granularity: "per_config"`
each config is its own job; `max_concurrent` caps how many run at once by
submitting them as a throttled Slurm job array (`--array=0-(K-1)%N`) — Slurm keeps
N running and starts the next config as each slot frees up:

```jsonc
"execution": {
  "mode": "sbatch",
  "sbatch": {
    "granularity": "per_config",
    "max_concurrent": 4,          // omit for no cap
    "partition": "frida", "account": "povejmo", "gpus": "B200:1",
    "container": "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"
    // "dry_run": true            // write sbatch_commands.sh without submitting
  }
}
```

## Adapting the template to a new experiment

Copy this directory to `src/experiments/<your_experiment>/`, then:

1. **`config.py`** — edit the `RunConfig` fields to your knobs. Keep `validate()`
   honest: reject unsupported combinations (the sweep runner will happily pass any
   key through, so this is where you draw the line). `WIRED_FEATURES` /
   `UNWIRED_FEATURES` document which bias features your data actually produces.
2. **`data.py`** — replace `load_data` with your dataset. Compute only the features
   your config enables so the data matches `cfg.bias_params()`.
3. **`train.py`** — adjust the record fields in `_save_train_record` to whatever
   `sweep.report` should aggregate (the metrics your `training_run` returns).
4. **`__main__.py`** — the `TEMPLATE` string is what `--init` writes; update its
   axes and scalars. Add one `p.add_argument` per new `RunConfig` field.

### The sweep contract (why `__main__.py` looks the way it does)

The runner renders each config value to a flag, so the parser must match:

| Config value | Rendered as | Parser requirement |
|--------------|-------------|--------------------|
| `some_key: 3` | `--some-key 3` | flag is `--some-key` (underscores → hyphens) |
| `flag: true/false` | `--flag` / `--no-flag` | `action=argparse.BooleanOptionalAction` (not `store_true`) |
| `xs: [a, b]` | `--xs a,b` | a comma-splitting `type=` (see `_str_list`) |
| `k: null` | *(omitted)* | default `None` means "off / internal default" |

Also required: accept `--runs-jsonl`, `--run-name`, `--sweep-id` (the runner always
appends them), and write exactly **one** JSONL line at the end of a run. Unknown
config keys become unknown flags and fail the run fast — that is intended.

`tests/test_template_flags.py` pins this contract with a render → parse → config
round-trip; keep an equivalent test when you adapt the template.
