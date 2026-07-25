# GraphQA

Trains GTLM on the [GraphQA benchmark](https://huggingface.co/datasets/baharef/GraphQA):
nine algorithmic questions over small graphs (count the nodes, find a shortest path,
check for a cycle, ...), each posed as a text-attributed graph whose prompt node carries
the question and answer. The experiment compares two graph encodings and ablates the
three graph-bias features.

Everything is run **from the repo root**.

## Quick start

```bash
python3 -m src.experiments.graphqa.download                                                  # raw data (once)
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/001_data_prep.jsonc # build datasets (once, CPU)
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/002_smoke.jsonc     # 4-step plumbing check
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/004_canary.jsonc    # 1 real run: does it learn?
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/003_ablation.jsonc  # the 135-run study
python3 -m sweep.report src/experiments/graphqa/results/003_ablation                         # aggregate
python3 -m src.experiments.graphqa.analysis.prep_table src/experiments/graphqa/results/003_ablation   # paper tables
```

Side probe — the same recipe on an **ALiBi** backbone (bloom-1b1) instead of Llama-3,
to show the GTLM stack is not tied to RoPE (`src/models/modeling_gtlm_bloom.py`):

```bash
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/007_bloom_alibi_data_prep.jsonc  # BLOOM-tokenized cache
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/008_bloom_alibi_canary.jsonc     # 1 run: does it learn?
python3 -m sweep src.experiments.graphqa src/experiments/graphqa/configs/009_bloom_alibi.jsonc            # 27 runs: 9 tasks x 3 seeds
python3 -m sweep.report src/experiments/graphqa/results/009_bloom_alibi                                   # aggregate
```

Same nine reported tasks and the same recipe as `003_ablation` (standard encoding,
`question_node: "off"`), with the full graph bias on in every run — this is a
demonstration that GTLM trains on an ALiBi base model, not a second ablation.

The backbone follows from `model_name` (`config.BACKBONES`), which also picks the LoRA
target modules and the tokenizer the cache is keyed by. Absolute accuracy sits below
the Llama numbers for reasons unrelated to positional encoding (bloom-1b1 has ~680M
non-embedding parameters to Llama-3.2-1B's ~975M, and is multilingual).

The two checks answer different questions, and the order matters. `002_smoke` runs 4
steps and **reports `test_accuracy: 0.0` even when everything is correct** — exact match
over the whole answer span is all-or-nothing, and warmup has barely lifted the LR off
zero by step 4. It proves a run completes and logs a well-formed record (read `eval_loss`,
not accuracy). `004_canary` trains one real config on the easiest task at the full recipe,
where a healthy pipeline lands high — that is what catches a labels/collator/bias bug the
smoke is blind to. Run both before committing to the 135-run sweep.

A single config, for quick iteration:

```bash
python3 -m src.experiments.graphqa --task shortest_path --graph-type standard
python3 -m src.experiments.graphqa --no-magnetic --seed 43          # an ablation arm
python3 -m src.experiments.graphqa --max-steps 4 --num-epochs 1     # smoke test
python3 -m src.experiments.graphqa --help                           # every flag
```

A standalone run appends its record to `results/train_runs.jsonl` and checkpoints under
`./checkpoints/graphqa/<run_name>`.

## Layout

| File | Responsibility |
|------|----------------|
| `config.py` | `RunConfig` — **every** knob lives here, once. `validate()`, `bias_params()`, `lora_config()`, `dataset_dir()`. |
| `data.py` | `load_data(cfg)` → train/val/test; `run_data_prep_mode(cfg)` builds + caches. |
| `process_dataset.py` | Raw json → text-attributed graphs (edge parsing, incidence transform, prompt node, features). |
| `load_dataset.py` | Merged multi-task loading; kept for `graphqa_mag_khop`. |
| `train.py` | `run_train_mode(cfg, …)` — trains one config, appends one JSONL record. |
| `__main__.py` | The argparse program: `build_parser`, `config_from_args`, `--init`, a thin dispatcher. |
| `download.py` | Fetch the raw dataset from HuggingFace. |
| `analysis/prep_table.py` | The paper's LaTeX tables, from a sweep's `runs.jsonl`. |
| `configs/` | Sweep configs (JSONC). |
| `results/` | Per-sweep output dirs + the standalone `train_runs.jsonl` fallback. |

## Protocol

| | |
|---|---|
| Splits | The benchmark's own: 1000 train / 500 validation / 500 test per task |
| Selection | Best **validation** exact match, evaluated every `eval_steps`; that checkpoint is reloaded and scored **once** on test |
| Reported metric | Test exact match, mean ± sample std over seeds {42, 43, 44} |
| Model / recipe | Llama-3.2-1B, `v2-eager`, fp32, LoRA r=16 (α=2r, dropout 0.05), lr 3e-5, bias_lr 5e-3, 20 epochs, batch 4 × accum 8 |
| Encodings (`graph_type`) | `standard` (one node per vertex) and `incidence` (bipartite Levi graph: one node per vertex **and** per edge) |
| Ablation arms | `base`, `no-spd`, `no-rrwp`, `no-magnetic` — standard encoding only |

The study is 5 arms × 9 tasks × 3 seeds = 135 runs. The arms are a **bundle** rather than
a free cross of `graph_type` × bias flags: incidence is only run with every bias on, so a
plain cartesian product would spend 3/8 of the jobs on cells the study never included.

### Two things changed relative to the originally published numbers

Both are deliberate; new results are **not** bit-comparable to the old ones.

1. **The official validation split is now used.** The benchmark ships 1000/500/500 per
   task, but the old pipeline built only train and test and then carved the last 15% off
   train to select on. That trained on 850 examples instead of 1000 and selected on a
   150-example signal drawn from the training distribution. All nine reported tasks ship
   a validation file. (`disconnected_nodes` and `node_classification` do not; they fall
   back to carving `--val-fraction` off train. Neither is reported.)
2. **Selection is on validation accuracy, not validation loss.** The old runs chose the
   lowest-val-loss checkpoint while reporting exact match — loss and accuracy routinely
   diverge late in training, as the model grows more confident rather than more correct.

## The dataset cache

Data prep is a separate, CPU-only sweep mode; training never builds datasets (it raises
and tells you to run prep). Built splits live in `processed_datasets/<graph_type>/<task>/`
and are reused, never rebuilt.

The feature-generation knobs (`model_name`, `max_length`, `magnetic_q`, `magnetic_m`,
`max_rw_steps`) are part of the cache identity: a config that changes one of them resolves
to a **tagged** sibling directory, so it can never silently train on features built with a
different value. A config using the defaults resolves to the historical untagged path, so
the existing (large) cache stays valid and `graphqa_mag_khop`, which hardcodes that layout,
keeps working.

The bias flags are deliberately *not* part of the cache key — every split carries spd, rrwp
and magnetic features regardless of which arm consumes them, so all four ablation arms share
one built dataset.

## Model implementation

This experiment runs the **v2** GTLM stack only; the legacy v0 path it used historically was
removed. v0 and v2-eager are pinned numerically equivalent at fp32 — same loss, same graph-bias
gradients, same base-weight gradients — by `tests/models/test_modeling_gtlm_llama_v2.py`, and
by `tests/models/test_v2_ragged_magnetic_padding.py` at the wide within-batch node spread this
experiment's incidence graphs produce (where v2's collator zero-pads the magnetic eigenvectors
of the smaller graphs and v0's ragged collator did not). Dropping v0 also gains k-hop masking,
which `graphqa_mag_khop` previously had to fork onto legacy v1 to get.

Two defaults follow from that parity rather than from taste:

- `impl` defaults to `v2-eager` — the implementation the parity is proven for, and the right
  one for these sequence lengths (~35 tokens standard, ~150 incidence; `v2-flex` and its
  autotune only pay off on far longer sequences).
- `dtype` defaults to `fp32` — the dtype the parity is proven at. `bf16` halves memory and is
  what the flex benchmarks measured, but it is a genuine numerical change, not a free speedup.
