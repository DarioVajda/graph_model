## Text-Attributed Graph (TAG) Benchmarks

Node classification on Cora, PubMed, OGBN-Arxiv and Reddit. Each node's k-hop
neighbourhood becomes one text-attributed graph; the target node carries the question
and the answer, and supervision is the answer span only, scored by exact match.

The experiment plugs into the generic [`sweep`](../../../sweep) runner: one standalone
single-run program, one `RunConfig` holding every knob, one JSONL record per run.
Everything is run **from the repo root**.

| File | Responsibility |
|------|----------------|
| `config.py` | `RunConfig` — every knob, once. `validate()`, `bias_params()`, `dataset_dir()`. |
| `data.py` | `run_data_prep_mode(cfg)` builds + caches the splits; `load_data(cfg)` returns `(train, val, test)`. |
| `train.py` | `run_train_mode(cfg, …)` — trains one config, appends one JSONL record. |
| `__main__.py` | The argparse program: `build_parser`, `config_from_args`, `--init`, a thin `main()`. |
| `test.py` | Re-score a saved checkpoint on a test split. |

### 1. Raw data

Download the raw datasets per the [RGLM repository](https://github.com/zhongjian-zhang/RGLM)
([Google Drive](https://drive.google.com/drive/folders/1aPlqxTUjRPUCNlRS-OpaRToEhZb61ffu)),
and unpack them into `raw_data/<dataset>/processed_data.pt`.

### 2. Build a dataset

```bash
python3 -m src.experiments.tag_benchmarks --mode data_prep \
    --dataset cora --max-neighbors 60 --text-mapping target_abstract \
    --samples-per-node 16
```

This writes `processed_data/cora_hops2_neighbors60_target_abstract/{train,val,test}/`
as `<first>-<last>.gtds` chunks. Splits are the benchmark's own `train_mask` /
`val_mask` / `test_mask`. Every build computes **all** structural features (SPD, RRWP,
magnetic), so bias-ablation arms share one built dataset.

`--text-mapping` is the experiment's main data axis — how much text a node contributes
as a function of its distance from the target:

| Mapping | Datasets | Param |
|---|---|---|
| `all_titles`, `all_abstracts` | cora, pubmed, ogbn-arxiv | — |
| `target_abstract` | " | — (abstract only for the target) |
| `neighbor_abstracts` | " | — (abstracts within 1 hop) |
| `random_abstracts` | " | `--text-mapping-param p` (abstract w.p. `p**dist`) |
| `full_text`, `more_target_text` | reddit | — |
| `truncated_text` | reddit | `--text-mapping-param <chars>` |

### 3. Train

```bash
python3 -m src.experiments.tag_benchmarks \
    --dataset cora --max-neighbors 60 --text-mapping target_abstract \
    --samples-per-node 16 --num-epochs 10 --lora-r 64 --lr 1e-5 --bias-lr 1e-2
```

`--help` lists every flag. A standalone run appends its record to
`results/train_runs.jsonl` and checkpoints to `./checkpoints/tag_benchmarks/<run_name>`.

### 4. Sweep

```bash
python3 -m src.experiments.tag_benchmarks --init my_sweep     # write configs/my_sweep.jsonc
python3 -m sweep src.experiments.tag_benchmarks src/experiments/tag_benchmarks/configs/my_sweep.jsonc
python3 -m sweep.report src/experiments/tag_benchmarks/results/my_sweep
```

Run the file once with `"mode": "data_prep"` to build every dataset it references, then
again with `"mode": "train"`. Set `execution.mode: "sbatch"` to run on the cluster.

> **The login node is for submitting jobs, not running them.** Anything that loads a
> model belongs on a compute node via `sbatch`.

## Protocol

Train on train; evaluate + checkpoint every `eval_steps`; reload the **best-validation**
checkpoint (`load_best_model_at_end`); score it on **test exactly once**. The reported
number is that checkpoint's test exact match (`test_accuracy` in the JSONL).

This is a deliberate change. The pre-refactor loop set the same best-val machinery up
and then ignored it: it scored the three surviving checkpoints on the **test** split and
reported the best of them — i.e. it selected on test. **Numbers from this code are
therefore expected to be lower than previously reported ones**, which were
optimistically biased. Any existing benchmark table needs re-running, not re-formatting.

`val_subsample` (default 500) strides the val split down for the in-training evals that
drive selection — full val is 542–29,799 graphs and is scored every `eval_steps`. The
striding (contiguous blocks of 50, evenly spread) is preserved verbatim from the old
loader, so selection behaviour is unchanged.

## Notes on the v2 migration

The experiment runs the **v2** stack (`GTLMLlamaForCausalLM`) only.

* **`impl` defaults to `v2-flex`.** TAG sequences are long (529–2,757 average tokens, see
  `processed_data/README.md`), which is where flex pays off: it captures the small
  `(B,H,N,N)` node bias and gathers it inside the kernel rather than materializing the
  `(B,H,L,L)` token-level bias, and it is bias-agnostic so RRWP rides along free.
  `v2-eager` is the parity/debug path, not the fast one. (This is the opposite default
  from `graphqa`, whose sequences are ~35–150 tokens.)
* **`dtype` defaults to `bf16`**, matching kgqa and probes. `graphqa` defaults to fp32
  because that is where v0↔v2 parity is proven and its runs are too short to care; here
  the cost is real. `--dtype fp32` is a genuine numerical change away from that default.
* **The historical runs used v0 + `sdpa`**, not v0 + eager — and the repo's parity test
  only covered eager. `tests/models/test_modeling_gtlm_llama_v2.py::test_v2_eager_matches_v0_sdpa`
  now pins v0-sdpa == v2-eager (logits and loss, every bias combination) so that gap is
  closed rather than assumed.
* **`laplacian` and `rwse` are rejected**, not silently accepted. The old `BIAS_PARAMS`
  enabled both by default, but data prep never computed their features: the collator
  emitted `(B, N, 0)` tensors and a `cdist` over a zero-width embedding is identically
  zero. They contributed nothing but parameters. `validate()` now says so. To use them,
  extend `data.py` to call `compute_laplacian_coordinates` / `compute_rwse`.
* **The training seed is real.** The old loop used `int(time.time()) % 65535`, so no run
  was ever reproducible.

## The dataset cache

`dataset_dir()` reproduces the historical directory names, so the existing cache
(up to 90,941 ogbn-arxiv train subgraphs) is reused as-is. A config whose
feature-generation knobs match `_CACHE_DEFAULTS` resolves to the untagged historical
path; changing `magnetic_q`, `max_rw_steps`, `max_length` or the tokenizer resolves to a
tagged sibling instead of silently reusing the wrong data.

**`samples_per_node` is the exception.** It is baked into a built dataset (as
`per_graph_versions`) but the old naming scheme never encoded it, and the existing caches
disagree — cora/60 used 16, cora/30 and reddit/15 used 4, pubmed/60 used 2,
ogbn-arxiv/60 used 1. Tagging on it would orphan every directory, so instead it is
**cross-checked against the cache's `metadata.json` on load** and a mismatch is a hard
error. Pass `--samples-per-node` explicitly for `data_prep`; for `train` it may be
omitted, in which case whatever the cache holds is adopted and recorded in the run's
JSONL line.

Two legacy directories are unreachable by the naming scheme and are effectively
orphaned: `ogbn-arxiv_hops2_neighbors30` (built before the mapping suffix existed) and
`cora_hops2_neighbors60_target_abstract_SMALL`.

The former `pubmed_hops2_neighbors30` was a third such orphan, but its contents were
confirmed to be the `target_abstract` mapping (target node carries title + abstract,
every neighbour title-only) at `samples_per_node=4`, so it was renamed to the canonical
`pubmed_hops2_neighbors30_target_abstract` and is now reached normally. `ogbn-arxiv_hops2_neighbors30`
is presumably recoverable the same way if its mapping is ever confirmed.
