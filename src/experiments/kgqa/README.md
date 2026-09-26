# KGQA experiment

Feeding KG subgraphs directly into a single GTLM model, replacing
GNN-RAG's GNN-reasoner + LLM-reader pipeline. Starting on **SR-WebQSP**.

## Goal

Two benchmarks, both under **SR retrieval** (Zhang et al.'s subgraph retriever, the same
inputs GNN-RAG's SR variant consumes): **WebQSP** (1-2 hop) and **CWQ / ComplexWebQuestions**
(1-4 hop). Data pipelines for both exist (`data/sr-webqsp`, `data/sr-cwq`); only WebQSP has
been trained so far.

**The win condition is matched-input, not leaderboard SOTA.** The claim this experiment
exists to test: *at fixed SR inputs and matched model size, one graph-native GTLM ≥
GNN-RAG's GNN-reasoner + LLM-reader pipeline, and > the same base LLM reading the subgraph
as serialized text.* Retrieval is deliberately held fixed — it is a confound, not a
contribution; every point gained by swapping retrievers would be unattributable to the
graph architecture. Concretely, in order:

1. **Beat retrieval-matched SR-GNN-RAG** (paper Table 15(d): WebQSP **78.9 Hits@1 /
   69.8 F1**, CWQ **55.6 Hits@1 / 53.3 F1**; we are at 79.1 / 74.1 WebQSP after the
   question-node arm — **ahead on both: F1 +4.3, Hits@1 +0.15**; on CWQ the graph arm
   is at baseline parity (57.0 / 54.7 best seed). The 71.3 F1 cited here
   previously is Table 2's *dense-retriever* GNN-RAG. Per the A2 diagnostic, Hits@1
   structurally under-credits a set generator: Hit/F1 are the honest primary
   metrics). This is the apples-to-apples pipeline-vs-single-model comparison the
   setup was designed for. **Won on WebQSP.**
2. **Beat the text-serialization ablation**: same base LLM, same SR subgraph flattened to
   triples in the prompt, no structural biases. This isolates whether the graph attention
   biases do anything — the result that transfers to the rest of the repo. *(Run 2026-07:
   flat currently wins — WebQSP 74.9 vs 74.1 F1, CWQ 58.6 vs 54.0. The question-node arm
   closed ~⅓ of the WebQSP gap; see [Results so far](#results-so-far).)*
3. **One scale run (3B/8B)** before drawing architecture conclusions: at 1B, plain
   text-RAG (RPO-RAG) gets 69.8 F1 and gains +11.5 F1 going to 8B — part of our gap may
   be reader capacity, not graph handling.
4. Only after 1–3 are won: **demonstrate retrieval portability** by plugging in one
   stronger public retriever (e.g. SubgraphRAG's) as a second input condition. This lifts
   the coverage ceiling and yields competitive headline numbers without claiming
   retrieval as a contribution. Sized in
   [Baseline retrieval ceilings](#baseline-retrieval-ceilings-sr-vs-the-rog-2-hop-pool-2026-09-23):
   the SR ∪ RoG-pool union reaches 90.2 Hits@1 on CWQ against 80.7 for either
   alone, so the headroom is in combining retrievers, not in picking a better one.

**Non-goal:** chasing the 2026 leaderboard (agentic/interactive-KG systems, GPT-4-class
readers, own retrievers — see [Published SOTA landscape](#published-sota-landscape-as-of-2026-07)).
Those gains come from retrieval quality, multi-turn KG access and reader scale, which are
orthogonal to the GTLM thesis; the SR answer-coverage ceiling (below) bounds our setting
regardless.

The direct predecessor is **GNN-RAG** — GTLM's aim is to match/beat it while collapsing its
GNN-reasoner + LLM-reader into a single model. Published leaderboard numbers below (Hits@1 / F1,
higher = better); the last column is our best run to date.

| Benchmark | Metric | RoG | GNN-RAG | GNN-RAG + RA | Best published (2026) | **Ours — best** |
|---|---|---:|---:|---:|---:|---:|
| **WebQSP** | Hits@1 | 80.0 | 80.6 | 82.8 | 91.6 | **79.1** |
| **WebQSP** | F1 | 70.8 | 71.3 | 73.5 | 88.6 | **74.1** |
| **CWQ** | Hits@1 | 57.8 | 61.7 | 62.8 | 79.6 | **57.0** |
| **CWQ** | F1 | 56.2 | 59.4 | 60.4 | 74.2 | **54.7** |

- **Ours** = best graph-native (GTLM) run per benchmark, test set, verbatim GNN-RAG
  scoring. WebQSP: `question_node_webqsp` seed 2, `isolated` mode (021 recipe +
  question node; test Hit 83.6; arm mean 73.5 ± 0.8 F1). CWQ: `cwq_headline` seed 2
  (test Hit 60.2; arm mean 54.0 ± 1.0 F1). The flat text-serialization *control*
  is still higher on both (74.9 / 58.6 F1) — see [Results so far](#results-so-far).
- **`graph_construction: "triplet"` is excluded from this column by decision, not
  by oversight.** It is the highest-scoring WebQSP arm (74.49 ± 0.50 mean F1, config
  034 — above the 74.1 cell above), but one node per raw triple hands the model the
  flat serialization's content in the flat serialization's own units, sidestepping
  the graph-native encoding this work exists to test. It stays a content-fair
  *control*; do not promote it into "Ours".
- **The WebQSP row is SPD+MagLap.** Adding RRWP (the paper's third encoding) does
  not change it: 72.74 ± 1.03 F1 / 78.11 ± 1.31 Hits@1, i.e. flat-to-slightly-below
  at 13× the storage — see
  [RRWP encoding parity](#rrwp-encoding-parity-2026-07-25-complete--one-negative-result).
- **GNN-RAG / GNN-RAG+RA / RoG**: from GNN-RAG Table 2 (Mavromatis & Karypis, 2024). Those use
  *dense/combined* GNN retrievers; the **retrieval-matched** baseline is GNN-RAG reading the
  SR sparse subgraph — paper **Table 15 row (d)** (verified against the PDF 2026-07-12):
  WebQSP **Hit 83.4 / Hits@1 78.9 / F1 69.8**, CWQ **Hit 60.6 / Hits@1 55.6 / F1 53.3**.
  These are the fairest single baselines given our SR inputs (our WebQSP F1 is ahead 72.5 vs
  69.8; Hits@1 0.15 short). NB the 71.3 F1 this README previously paired with the 78.9 is
  Table 2's dense-retriever figure, not the SR row — SR notably *hurts* GNN-RAG on CWQ
  (61.3 → 55.6 Hits@1 vs dense; Table 15 attributes it to disconnected sparse subgraphs
  breaking their shortest-path extraction).
- **Best published (2026)**: WebQSP Hits@1 = TRACE (91.6, GPT-4.1 agentic); WebQSP F1 and both
  CWQ cells = GraphWalker (Qwen2.5-7B SFT+RL; reports EM, ≈Hits@1). Metric caveats and the full
  method list are in [Published SOTA landscape](#published-sota-landscape-as-of-2026-07) below.
  These methods run their **own retrieval/agentic KG access**, so they are *not* bounded by our
  SR answer-coverage ceiling and upper-bound the field loosely, not our setting.
- Our SR-retrieval **answer-coverage ceiling** (below) caps WebQSP at Hits@1 ≤ 90.9 / F1 ≤ 89.1
  (data-format v3; pipeline losses now ≈0, so this is within 0.2 of raw SR itself) — the headroom
  any GTLM on these inputs is competing for. Note current SOTA already presses against it: the
  field has moved past what SR retrieval can support.

### Published SOTA landscape (as of 2026-07)

Numbers and metric caveats below; **what each method actually is and how it works lives in
[SOTA.md](SOTA.md)** (method summaries grouped by family, kept out of this README on purpose).

**Metric warning:** many KGQA papers label as "Hits@1" what is actually **Hit** (any gold
substring appears anywhere in the generated text — the laxest metric; SubgraphRAG's appendix
documents this mislabeling). Columns below use each paper's numbers sorted into the metric they
*actually* compute; blank = not reported. Our own eval logs true Hits@1, Hit and F1 separately,
so compare column-to-column.

| Method (year) | Base LLM | WQSP Hit | WQSP H@1 | WQSP F1 | CWQ Hit | CWQ H@1 | CWQ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| GNN-RAG + RA (2024) | Llama-2-7B | 90.7 | 82.8 | 73.5 | 68.7 | 62.8 | 60.4 |
| GCR (2024) | Llama-3.1-8B + GPT-4o-mini | 92.2 | 82.9¹ | 74.1 | 75.8 | 59.1¹ | 61.7 |
| SubgraphRAG (ICLR '25) | retriever + GPT-4o-mini | 90.1 | 84.3¹ | 77.5 | 62.0 | 60.7¹ | 54.1 |
| KG-R1 (2025) | Qwen2.5-3B, RL | | 82.8 | | | 65.3 | |
| ReKnoS (2025) | Qwen3-235B | | | | | 65.6 | |
| KnowCoder-A1 (2025) | Qwen2.5-Coder-7B, RL | | 80.1 | 77.2 | | 75.7² | 68.3 |
| TRACE (2026) | GPT-4.1, agentic | | 91.6³ | 81.7 | | 76.9³ | 72.9 |
| GraphWalker (2026) | Qwen2.5-7B, SFT+RL | | 91.5² | 88.6 | | 79.6² | 74.2 |
| RPO-RAG (2026) | Llama-3.1-8B | 89.9 | | 81.3 | 72.3 | | 64.5 |
| **RPO-RAG (2026)** | **Llama-3.2-1B** | **82.3** | | **69.8** | **60.3** | | **50.4** |
| PathISE (2026) | Llama-3.1-8B + GPT-4o/4.1 | 91.6 | 86.8 | 81.3 | 71.9 | 63.4 | 61.5 |
| *ChatKBQA (2024)* ⁴ | Llama-2-13B, SP | | *86.4* | *83.5* | | *86.0* | *81.3* |
| *PGDA-KGQA (2025)* ⁴ | Llama-2-7B/13B, SP | | *89.0* | *86.3* | | *87.1* | *83.1* |

¹ GCR/SubgraphRAG H@1 cells are PathISE's re-evaluation (GPT-4o backend) — the papers
  themselves only report Hit + F1.
² EM (exact match of the answer set / RHits@1) — closest to, but not literally, Hits@1.
³ TRACE calls it Hits@1 but sits in the ToG lineage where "Hits@1" is typically Hit;
  unverified, treat as upper bound.
⁴ *Italics* = semantic-parsing methods scored **with oracle (gold) topic entities** — they
  generate a logical form and execute it on Freebase. Not comparable to end-to-end retrieval
  systems (entity linking is given for free); listed because they are the absolute
  benchmark-leaderboard tops, especially on CWQ.

Reading it for our purposes:

- **True-Hits@1 SOTA** (verified metric, no oracle): **PathISE 86.8** WebQSP / **KG-R1 65.3** CWQ
  (GraphWalker's EM 91.5/79.6 likely exceeds these but under a slightly different match rule).
- **F1 SOTA** (no oracle): **GraphWalker 88.6** WebQSP / **74.2** CWQ — a 7B model with SFT+RL
  and agentic multi-turn KG access, i.e. *not* a single-pass reader over a fixed subgraph.
- **The most informative anchor for us is RPO-RAG's Llama-3.2-1B row** (bolded): same base
  model as ours, single-pass RAG-style reading. Hit 82.3 / F1 69.8 WebQSP vs our Hit 83.6 /
  F1 74.1 (question-node arm) — **we now exceed the 2026 preference-optimized text-RAG pipeline
  at equal model size** (+1.3 Hit, +4.3 F1), with a different (their own) retriever.
- Everything above ~89 WebQSP Hits@1 exceeds our capped SR ceiling (90.9 / 89.1 F1, data v3)
  — those systems retrieve better than SR or query the KG interactively. Beating
  them from SR inputs is impossible by construction; the honest target for GTLM is the
  retrieval-matched comparison plus closing the gap to the SR ceiling.

## Reproducing from scratch

Everything below is **run from the repo root** — dataset paths and `results_dir`
are repo-root-relative. The pipeline is four ordered stages: **setup → acquire
raw data → build the name dictionary → build datasets → train/evaluate**. Later
stages hard-depend on earlier ones (data prep reads the name dictionary, which
reads the raw subgraphs), so run them in order the first time.

### 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # pip-tools lock; edit requirements.in to change deps

# Llama-3.2-1B is a GATED model: accept its license on the HF model page, then
# authenticate. wandb is only needed if a config sets "wandb_project".
huggingface-cli login                        # (in-house: `bash login.sh` does hf + wandb)
```

### 2. Acquire the raw SR subgraphs + seed name dictionary

Both come from GNN-RAG's public release (Mavromatis & Karypis, 2024). The
[`gnn/`](https://github.com/cmavro/GNN-RAG/tree/main/gnn) README points to the
Drive folder for "the datasets", and the
[`llm/`](https://github.com/cmavro/GNN-RAG/tree/main/llm) README points to the
same folder for `entities_names.json`:

- **`data.zip`** — SR-retrieved subgraphs, ~1.5 GB
- **`entities_names.json`** — 560k Freebase-mid → name seed dictionary

```bash
KGQA=src/experiments/kgqa
pip install gdown

# GNN-RAG Drive folder (canonical source):
gdown --folder 1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp -O /tmp/gnnrag

unzip /tmp/gnnrag/data.zip -d "$KGQA"         # -> $KGQA/data/{sr-webqsp,sr-cwq,webqsp,CWQ}/
cp    /tmp/gnnrag/entities_names.json "$KGQA" # -> $KGQA/entities_names.json (the "v1" seed)
```

Only `data/sr-webqsp/` is consumed for WebQSP (`data/sr-cwq/` for the not-yet-run
CWQ arm); `data/webqsp/`, `data/CWQ/`, and the `*.npy` embeddings are GNN-RAG's
GNN-training inputs and go unused here. Everything under `data/` and both
`entities_names*.json` files are git-ignored (they are large / externally sourced).

### 3. Build the name dictionary (naming v2)

`entities_names.json` from the release misses ~76 test golds, which then collapse
as presumed CVT mediators (see [drop decomposition](#why-the-two-ceilings-differ-drop-decomposition)).
Naming v2 extends it in place with Freebase-native aliases — in-subgraph
`type.object.name` triples plus the FB5M name dump — and backs the seed up to
`entities_names.v1.json`:

```bash
# FB5M Freebase name dump (naming v2's third source):
curl -L -o "$KGQA/data/FB5M.name.txt.bz2" \
  https://raw.githubusercontent.com/castorini/BuboQA-data/master/FB5M.name.txt.bz2

python3 -m src.experiments.kgqa.analysis.build_entities_names_v2   # 560k -> 598.5k entries
```

With the sr-cwq splits and the `data/cwq_ent_id2mid.txt` decode table also present
(see `sr_records.py` — CWQ subgraphs are int-coded and need SR's `ent2id.pickle`
vocabulary to name anything), the rebuild additionally names CWQ mids: → 840.2k
entries. Rebuilds seed from the current file and are **append-only**, so existing
naming-v2 node texts never change and every naming-v2 cache stays valid.

Both dictionaries are kept, and which one a run reads is the explicit
`naming_version` knob (`1` = `entities_names.v1.json`, `2` = `entities_names.json`),
which is part of the dataset cache key. So the legacy-naming control arm rebuilds
reproducibly from a clean clone, in any order — nothing depends on *when* you ran
this step.

> **Two different "v2"s.** `data_format_version` (dfv2/dfv3) versions the *pipeline
> semantics*; `naming_version` versions the *name dictionary* — and the numbers run
> opposite ways. Data-format **v3** bundles naming **v2**, so a faithful dfv2 control
> arm pins **both** (`data_format_version: 2` + `naming_version: 1`), as
> `005_attribution_v3.jsonc` does.

### 4. Build the datasets and train

The experiment is a standalone single-run program driven by the generic `sweep`
runner. The headline results come from [`configs/005_attribution_v3.jsonc`](configs/005_attribution_v3.jsonc)
(annotated in-file). The two-step workflow — **data_prep once, then train** — is
idempotent; data prep is cheap (no GPU):

```bash
CFG=$KGQA/configs/005_attribution_v3.jsonc

# a) Build every .gtds the config references. Set "mode": "data_prep" in $CFG, then:
python3 -m sweep src.experiments.kgqa "$CFG"

# b) Train. Flip "mode" back to "train", then:
python3 -m sweep src.experiments.kgqa "$CFG"

# c) Aggregate once the (sbatch) jobs finish -> results/attribution_v3/report.md
python3 -m sweep.report "$KGQA/results/attribution_v3"
```

`005_attribution_v3.jsonc` runs on Slurm (`"mode": "sbatch"`, B300 GPU); flip
`execution.mode` to `"local"` to run on the current machine. The best single run
(v3, `n_max` 20, seed 1) reproduces **72.5 F1 / 78.75 Hits@1**; see
[Results so far](#results-so-far).

### Single-config / quick iteration

To bypass the sweep runner (one config, direct CLI flags):

```bash
python3 -m src.experiments.kgqa --mode data_prep                     # build this config's datasets
python3 -m src.experiments.kgqa --lora-r 16 --k-hop 0 --lr 1e-4      # train one config
python3 -m src.experiments.kgqa --max-steps 4 --gen-max-samples 8    # smoke test
python3 -m src.experiments.kgqa --init my_sweep                      # scaffold a new sweep config
```

Standalone train runs (no `--runs-jsonl`) append their record to
`results/train_runs.jsonl`. Every CLI flag maps 1:1 to a `RunConfig` field
(`config.py`); see `configs/000_example.jsonc` for an annotated sweep template and
the [Running](#running-sweep-workflow) note below.

## Running (sweep workflow)

The generic `sweep` runner expands a JSONC config (scalars fixed, lists swept,
bundles varied together) into one run per resolved config, invoking
`python3 -m src.experiments.kgqa` per run. A key appears in exactly one place and
maps 1:1 to a CLI flag. **Run from the repo root** — paths are repo-root-relative.

```bash
python3 -m src.experiments.kgqa --init my_sweep                        # -> configs/my_sweep.jsonc
python3 -m sweep src.experiments.kgqa src/experiments/kgqa/configs/my_sweep.jsonc  # data_prep, then train
python3 -m sweep.report src/experiments/kgqa/results/my_sweep          # aggregate
```

The `.gtds` cache directory is keyed only by data-affecting fields
(`RunConfig.data_config_key`), so runs differing only in training config
(seed, lr, k_hop, …) share one built dataset.

## Results so far

### Data-format v3 sweeps (2026-07-08)

`attribution_v3` (data {v2-control, v3, v3/n_max=50} × seed {0,1,2}) and
`capacity_lora` (lora_r {8,64}), all at the cheap operating point: k_hop 0,
last_1, lr 1e-4, bias_lr 5e-3, 15 epochs, full-dev checkpoint selection
(eval_steps 200), B300. Test set, 1628 questions:

| arm | lora_r | n_max | test F1 (seeds 0/1/2) | mean F1 | Hits@1 | Hit |
|---|---:|---:|---|---:|---:|---:|
| v2-control | 16 | 20 | 66.81 / 66.73 / 65.93 | **66.49 ± 0.4** | 74.42 | 80.08 |
| **v3** | 16 | 20 | 70.26 / **72.50** / 70.56 | **71.11 ± 1.0** | 77.48 | 82.15 |
| v3, recall arm | 16 | 50 | 72.12 / 71.46 / 70.80 | **71.46 ± 0.5** | 77.60 | 81.84 |
| v3, capacity | 8 | 20 | 69.40 (seed 0) | 69.40 | 76.23 | 80.41 |
| v3, capacity | 64 | 20 | 71.90 (seed 0) | 71.90 | 77.76 | 82.74 |

Takeaways:

- **Data-format v3 (newline answer delimiter + naming v2) is worth +4.6 test F1**
  (66.5 → 71.1) at identical training config — 5–10× the seed-noise bar, which this
  sweep measured for the first time (±0.4 v2 / ±1.0 v3). The v2-control landing on
  the historical 66.5 plateau validates the comparison.
- **Best run** (v3, n_max 20, seed 1): **72.50 F1 / 78.75 Hits@1 / 82.74 Hit** —
  F1 above retrieval-matched SR-GNN-RAG (69.8, Table 15(d); the previously-cited 71.3
  is their dense-retriever F1), Hits@1 within 0.15 of it (78.9).
- **n_max=50** buys ~+0.35 mean F1 (within noise of n_max=20's spread), flat Hits@1 —
  the enumeration tail is not the binding constraint.
- **LoRA capacity is monotone** (r8 69.4 < r16 70.3 < r64 71.9 at seed 0): +1.6 F1
  for r64 over control vs ±1.0 noise — suggestive; the miss_copied bucket diff
  (error analysis v3) is the deciding readout for the 3B/8B scale-run decision.

### Regularization probes (2026-07-11/12)

32-run campaign (`reg_probes` / `reg_combo` / `reg_round2`) probing whether the
graph-bias channel's total lack of regularization drives the memorization seen
in the train-slice diagnostic (train-fit 96.7 vs test 72.6). Full design,
per-arm tables and verdict: `TODO_reg.md`. Bottom line vs the frozen-recipe
control (test F1 72.55 ± 0.22):

- **`lora_dropout 0.15` is the only winner (+0.62)**; 0.25 overshoots (−0.8).
  Carried into the scale/CWQ recipe.
- **Every graph-side regularizer is neutral-to-catastrophic** (coherent
  spectral dropout −0.3; droppath −1.8/−3.9 at 0.05/0.1; per-layer eigvec
  dropout −11.9; element-wise bias dropout −35). Train-slice shows they *do*
  cut memorization (96.7 → 89–94) but test falls harder: the structural
  channel carries disproportionately *generalizing* signal. Graph channel
  exonerated; memorization lives in the LoRA/backbone path.
- Weight decay on graph-bias params (an accidental never-decayed group, now
  fixable via `bias_weight_decay`) is flat at 0.1 and mildly harmful at 0.3 —
  the fix stays, the value stays 0.

The negative model-side mechanisms are removed from the codebase after this
campaign; reproducing those arms = checkout tag `reg-probes-2026-07`.

### Bias ablation / `magnetic_shared` affordability (2026-07-13 → 2026-07-14, complete)

`bias_ablation_webqsp` (job 111534, array `0-17%18`): the same 6 bias arms as
the probe suite (`none`, `spd`, `magnetic`, `spd+magnetic`, `magnetic_shared`,
`spd+magnetic_shared`) × seeds {0,1,2}, graph-native mode only, on the frozen
WebQSP recipe (data-format v3, r64, lora_dropout 0.15, bias-wd fix, 15 ep,
full-dev selection — everything else pinned to `021_webqsp_recipe_refresh.jsonc`).
Config: [`configs/024_bias_ablation_webqsp.jsonc`](configs/024_bias_ablation_webqsp.jsonc).

`magnetic_shared` and the bias-free `none` arm didn't exist as options in kgqa
before this sweep — `RunConfig` only had independent `spd`/`magnetic` bools and
`validate()` hard-required at least one on. Added `magnetic_shared` (wired into
`bias_params()`; already a first-class field on the shared `GraphConfigMixin`
model config, so no model-side change was needed) and relaxed `validate()` to
reject only `magnetic`+`magnetic_shared` together, not all-off.

**Accuracy** (test set, mean ± std over seeds {0,1,2}):

| arm | F1 | Hits@1 | Hit* | EM |
|---|---:|---:|---:|---:|
| `none` | 0.4762 ± 0.0083 | 0.5604 ± 0.0076 | 0.6454 ± 0.0077 | 0.2189 ± 0.0038 |
| `spd` | 0.6360 ± 0.0032 | 0.7099 ± 0.0010 | 0.7885 ± 0.0029 | 0.3081 ± 0.0053 |
| `magnetic_shared` | 0.6971 ± 0.0047 | 0.7615 ± 0.0082 | 0.8077 ± 0.0048 | 0.3581 ± 0.0010 |
| `magnetic` | 0.7066 ± 0.0022 | 0.7717 ± 0.0029 | 0.8186 ± 0.0014 | 0.3722 ± 0.0066 |
| `spd+magnetic_shared` | 0.7175 ± 0.0047 | 0.7750 ± 0.0075 | 0.8225 ± 0.0048 | 0.3690 ± 0.0012 |
| `spd+magnetic` | **0.7278 ± 0.0033** | **0.7819 ± 0.0022** | **0.8266 ± 0.0006** | **0.3855 ± 0.0040** |

**Speed** (median steady-state train-step rate off the tqdm bars in the Slurm
logs, converted to wall-clock s/it; three samples per arm, one per seed — noisy,
since runs 3-17 shared nodes with 3-5 concurrent array tasks and each other,
unlike the first 3 which ran close to solo):

| arm | median s/it |
|---|---:|
| `none` | ~0.72 |
| `magnetic_shared` | ~0.73 |
| `spd` | ~0.82 |
| `spd+magnetic_shared` | ~0.92 |
| `spd+magnetic` | ~1.05 |
| `magnetic` | ~1.17 |

**Verdicts:**

1. **Ablation** — every bias arm beats `none` by a wide margin (+0.16 to +0.25
   F1); this is the dominant effect, dwarfing differences between bias types.
   `magnetic` alone (0.707 F1) outperforms `spd` alone (0.636 F1) by ~7 pp —
   the magnetic term carries more of the signal than SPD on this task. Combining
   both gives the best result (`spd+magnetic`, 0.728 F1), but the marginal gain
   over `magnetic` alone is small (+0.021 F1) — `spd` mostly reinforces
   `magnetic` rather than adding an independent signal.
2. **`magnetic_shared` affordability** — costs a small, consistent accuracy tax
   vs. its per-layer counterpart (`magnetic_shared` 0.697 vs. `magnetic` 0.707 F1;
   `spd+magnetic_shared` 0.718 vs. `spd+magnetic` 0.728 F1 — about 1 pp F1 in
   both cases), in exchange for a real speed win: ~0.73 vs. ~1.17 s/it standalone
   (~38% faster), consistent with the probe suite's finding that `magnetic_shared`
   runs at roughly SPD cost. **Recommendation:** switch the frozen recipe to
   `magnetic_shared` if throughput matters more than the last ~1 pp of F1;
   keep per-layer `magnetic` if squeezing peak accuracy is the priority. No
   peak-GPU-memory instrumentation exists in kgqa's `train.py` (unlike the
   probes' sweep), so the speed comparison above is wall-clock only.

### bias_lr bracket (2026-07-15, complete)

`bias_lr_webqsp` (job 112102): does a hotter bias-group lr improve the
graph-native arm? Motivated by the 024 ablation — the soft bias is the sole
topology carrier in graph mode, yet transmits structure slightly worse than
flat text — and by 5e-3 being inherited, never bracketed. bias_lr ∈
{1e-2, 3e-2} × seeds {0,1,2}, everything else pinned to
`021_webqsp_recipe_refresh`'s graph arm; 021's graph runs are the 5e-3
baseline. Config: [`configs/025_bias_lr_webqsp.jsonc`](configs/025_bias_lr_webqsp.jsonc).

| bias_lr | test F1 | test Hits@1 |
|---|---:|---:|
| 5e-3 (021 baseline) | 0.7207 ± 0.0016 | — |
| 1e-2 | 0.7229 ± 0.0081 | 0.7758 ± 0.0068 |
| 3e-2 | 0.5596 ± 0.0381 | 0.6630 ± 0.0327 |

**Verdict: bracket closed, keep 5e-3.** 1e-2 is a wash (+0.2 pp F1, within
seed noise, with 5× the seed spread); 3e-2 is destructive (−16 pp F1, high
variance — the bias MLP destabilizes training). The graph arm's
structure-transmission deficit vs. flat (0.721 vs. 0.749 F1) is not a
bias-lr problem; whatever closes it has to come from elsewhere (see the
question-node arm).

### Question-node ablation (2026-07-15, complete) — current best graph arm

`question_node_webqsp` (job 112227): the mechanism-3 fix from the
why-flat-beats-graph analysis. The structural mask provably blocks graph→prompt
attention, so the graph was encoded *question-agnostically* while the flat
control's question-first serialization gives every context token
question-conditioned compute. The `question_node` knob moves the question text
out of the PROMPT node into its own QUESTION prefix node, which the existing
bidirectional-prefix mask exposes to every graph token — question-conditioned
encoding with zero model changes. Swept QUESTION's directed out-edge set
(`all` = edge to every base-graph node, `topics` = to topic entities,
`isolated` = no edges) × seeds {0,1,2}; everything else pinned to
`021_webqsp_recipe_refresh`, whose runs are the controls. Configs:
[`configs/028_question_node_data_prep.jsonc`](configs/028_question_node_data_prep.jsonc) (cache builds),
[`configs/029_question_node_webqsp.jsonc`](configs/029_question_node_webqsp.jsonc).

| arm | test F1 | Hits@1 | Hit |
|---|---:|---:|---:|
| flat control (021) | 0.7490 ± 0.0003 | 0.7995 | 0.8501 |
| **`isolated`** | **0.7351 ± 0.0076** | 0.7803 | 0.8325 |
| `all` | 0.7287 ± 0.0086 | 0.7793 | 0.8280 |
| `topics` | 0.7284 ± 0.0022 | 0.7783 | 0.8272 |
| graph control (021, no question node) | 0.7207 ± 0.0016 | 0.7729 | — |

**Verdicts:**

1. **The question node is worth +0.8 to +1.4 test F1** over the old
   construction — all 9 runs landed at or above the best control seed. It closes
   ~⅓–½ of the graph-vs-flat gap; flat still leads by ~1.4 F1.
2. **The edge modes are statistically indistinguishable, with the edge-free
   `isolated` on top**: the gain comes from the question *text* being visible
   during graph encoding, not from wiring the question into the SPD/magnetic
   structure (the `all` hub, if anything, costs a few tenths).
3. **Best single run** (`isolated`, seed 2): **74.07 F1 / 79.05 Hits@1 / 83.60
   Hit** — the best graph-native result to date and the first to clear
   retrieval-matched SR-GNN-RAG (69.8 F1 / 78.9 Hits@1) on both primary metrics.
   Dev F1 plateaued by epoch ~11–14, so longer training is not the next lever;
   the residual deficit vs flat is attributed to the other two diagnosed
   mechanisms (per-node position reset, duplicate keys), pointing at the
   superset arm (flat serialization + node-span graph biases) next.

### Position/order + capacity probes (2026-07-16, complete) — three negative results

Follow-up to the question-node ablation's closing note, which attributed the
residual flat-vs-graph deficit to "the other two diagnosed mechanisms
(per-node position reset, duplicate keys)." Three independent probes, each
reusing an existing run as its control (no re-runs of settings already on
record), 3 seeds per new arm, all WebQSP. Full per-seed data, reasoning, and
the mechanism writeup that motivated these arms: `TODO.md`.

| probe | question | control | new-arm result | verdict |
|---|---|---:|---:|---|
| **Flat-order shuffle** (`configs/030_flat_shuffle_diag.jsonc`) | does flat's edge come from a retrieval-order → RoPE-position signal? | 0.7490 ± 0.0003 (021 flat) | **0.7525 ± 0.0035** (`flat_shuffle_lines=true`) | premise falsified — scrambling the order cost nothing |
| **`magnetic_dim` capacity** (`configs/031_magnetic_dim_sweep.jsonc`) | is the bias MLP's width a binding constraint? | 0.7351 ± 0.0076 (dim 128) | 0.7382 / 0.7347 / 0.7300 (dims 32/64/256) | capacity ruled out — flat, no trend |
| **SPD-depth position encoding** (`configs/032_node_position_spd_depth.jsonc`) | does a RoPE-visible, graph-structure-derived position (replacing the per-node reset-to-0) close the gap? | 0.7351 ± 0.0076 (`reset`, isolated arm) | **0.6412 ± 0.0037** (`node_position_mode=spd_depth`) | fix rejected — clear −9.4 F1 regression |

**Verdicts:**

1. **The retrieval-order hypothesis is dead.** Destroying flat's triple order
   (a deterministic per-question shuffle, independent RNG stream from the
   answer-order augmentation so targets stay byte-identical) left F1
   unchanged, if anything marginally higher. Whatever flat is doing better
   than graph, it isn't leaning on serial position as a distance proxy.
2. **Capacity isn't the lever.** All four `magnetic_dim` widths land in a
   tight 0.730–0.738 band with no monotonic trend — consistent with the
   graph-bias weights already converging well (message-passing-like attention
   patterns observed in other benchmarks).
3. **The position-encoding fix (`node_position_mode="spd_depth"`) is a clear
   regression, not a fix** — `STRIDE × shortest-path-distance-from-prompt`
   offsets, tested end-to-end with new backward-compatibility and
   permutation-equivariance proofs (`tests/test_node_position_encoding.py`;
   both invariants hold to float64 numerical precision). The revealing detail:
   **Hits@1 barely moved** (0.7827 vs. 0.7803 control) while **F1 collapsed**
   — top-1 answer selection stayed intact, multi-answer recall broke. Likely
   mechanism: pushing prefix nodes to large, STRIDE-scaled RoPE distances from
   the query weakens the model's ability to aggregate the *full* answer set,
   not its ability to find *one* good answer.

**Net effect: RoPE-visible relative position was not the missing
ingredient.** The "per-node position reset" mechanism from the question-node
write-up above is now a *narrowed*, not confirmed, hypothesis — the bias
channel may still be undersized/underdriven relative to what it's replacing
(per `TODO_reg.md`'s finding that it already carries real, generalizing
signal), but "give it a RoPE-shaped distance signal" was the wrong lever.
Both new knobs (`flat_shuffle_lines`, `node_position_mode`) stay in the
codebase as tested, reversible, documented negative results; defaults are
unchanged (`False` / `"reset"`).

### RRWP encoding parity (2026-07-25, complete) — one negative result

The GTLM paper uses **SPD + RRWP + MagLap** throughout, but the KGQA arm was
built with RRWP dropped on an unverified assumption that its storage cost was
prohibitive. That left the WebQSP results a strict *subset* of the paper's
encoding — an inconsistency worth closing before quoting them externally.
Configs: [`configs/038_rrwp_data_prep.jsonc`](configs/038_rrwp_data_prep.jsonc)
(cache build), [`configs/039_rrwp_webqsp.jsonc`](configs/039_rrwp_webqsp.jsonc)
(job 117232). `rrwp` / `max_rw_steps` default OFF in `RunConfig` and `_rw{T}`
joins `data_config_key()` only when enabled, so every pre-existing cache stayed
valid.

039 is 029's `isolated` arm with **only** `rrwp` added — a genuine one-variable
change against a published 3-seed baseline:

| arm | test F1 | Hits@1 | Hit* |
|---|---:|---:|---:|
| `isolated`, SPD+MagLap (029) | **0.7351 ± 0.0076** | 0.7803 ± 0.0115 | 0.8325 |
| `isolated`, +RRWP (039) | 0.7274 ± 0.0103 | 0.7811 ± 0.0131 | 0.8274 |

Per-seed F1, paired by seed: 73.81→73.03, 72.65→71.60, 74.07→73.60 (paired delta
−0.77 ± 0.29).

**Verdicts:**

1. **RRWP does not help WebQSP.** F1 −0.77, Hits@1 flat (+0.08). All three seeds
   moved the same direction with a tight delta spread, but the delta is *under*
   the within-arm seed noise (±0.8–1.0) — read this as neutral-to-slightly-
   negative, not as a demonstrated regression. What it is not is an improvement.
2. **The storage assumption was wrong; the decision it produced was right.**
   Measured, not estimated: the RRWP column is n²×16 float32 = 41 GB, taking a
   built WebQSP config from 3.4 GB to **42 GB** (13×). The multiplier is dtype
   and depth, not graph size — 16 float32 per node pair where SPD stores one
   int16 over the identical n² footprint. Prep is cheap in time (~13 min) but
   training runs 30–60% slower. So: 13× the disk and a wall-clock tax for no
   accuracy gain.
3. **The encoding subset was not hiding anything.** The "these numbers
   underperform because the encoding was crippled" reading is closed off with
   data. Both configurations sit above retrieval-matched SR-GNN-RAG (69.8 F1)
   and below the flat control (74.9); RRWP does not touch that gap.

Ops note: training needs **≥192 G** (seed 2 peaked at MaxRSS 156 G; 128 G would
have been OOM-killed), and the prep peaked at 132 G — the 64 G that older prep
configs use is not enough. `TextGraphDataset.compute_rrwp` now sizes its Arrow
`writer_batch_size` from the largest graph: at n=512 × 16 steps a row is 4.2 M
floats, so the default 1000-row writer batch would have overflowed Arrow's
2³¹-element array cap.

### Base-model scale: Llama-3.1-8B on WebQSP (2026-09-23, complete)

README goal #3, the one scale run before architecture conclusions. Config
[`configs/044_webqsp_8b_scale.jsonc`](configs/044_webqsp_8b_scale.jsonc) (job
165295; cache from [`043`](configs/043_webqsp_8b_data_prep.jsonc)) is 029's
`isolated` arm with only the backbone swapped to Llama-3.1-8B base: lr 1e-4
and bias_lr 5e-3 kept, 15 epochs, graph arm only, one seed. It ran as 4-rank
DDP at 1 x 2, which is the same effective batch and schedule as 029's 2 x 4.

| arm | test F1 | Hits@1 | Hit* | EM |
|---|---:|---:|---:|---:|
| 1B, `isolated` (029, 3 seeds) | 0.7351 ± 0.0076 | 0.7803 | 0.8325 | — |
| 1B flat control (021) | 0.7490 ± 0.0003 | 0.7995 | 0.8501 | — |
| **8B, `isolated` (044, seed 0)** | **0.7607** | **0.8084** | **0.8440** | 0.4140 |

**Verdicts:**

1. **Scale helps the graph arm: +2.6 test F1**, about 3.4x the 1B seed spread.
   This is one seed, so treat it as a clear effect whose exact size is not yet
   pinned down. Hits@1 80.8 now clears SR-GNN-RAG (78.9) by ~2 points, not 0.15.
2. **The 8B graph arm beats the 1B flat control** (76.1 vs 74.9). That is not
   the goal-#2 comparison, which needs flat at 8B too and was not run. It does
   show that part of the 1B graph-vs-flat gap was reader capacity.
3. **Learning is faster, not just higher.** Dev F1 reached 70 by epoch 2.5,
   where the 1B took about 10 epochs. It was still rising at epoch 14.7 (best
   dev 77.9 at the last eval), so 15 epochs slightly undertrains the 8B.
4. **Cost:** 3 h 44 min on 4x B300, including a ~20 min cold flex compile and
   the final test eval. Training ran at ~2.1 s/step. That is ~4.2x the 1B's
   GPU-time per training example, well below the ~6x that parameter count
   predicts. Flex autotune logs `No valid triton configs ... out of resource`
   at head_dim 128. The message is harmless: the oversized candidates are
   skipped and the ones that fit are used.

**Follow-up: lr, epochs and adapter regularization (2026-09-24, complete).**
044's dev curve pointed at lr 1e-4 being high for 8B: dev F1 sat in a noisy 64-72
band while the cosine schedule held lr near its peak, and it gained only once lr
fell below ~6e-5. Dev loss rose from epoch 1.2 on, and train loss ended at 0.18
against the 1B's 0.33. Configs:
[`047`](configs/047_webqsp_8b_seeds_lr.jsonc) (seeds and lr),
[`049`](configs/049_webqsp_8b_e22.jsonc) (22 epochs),
[`050`](configs/050_webqsp_8b_lr5e5_e30.jsonc) (lr 5e-5, 30 epochs),
[`051`](configs/051_webqsp_8b_regularization.jsonc) (dropout and rank). All keep
8 examples per optimizer step; only the rank count and the GPU type vary.

| lr | epochs | lora_r | lora_dropout | seeds | test F1 | Hits@1 | Hit* |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1e-4 | 15 | 64 | 0.15 | 0, 1 | 0.7637 ± 0.0030 | 0.8096 | 0.8445 |
| 1e-4 | 22 | 64 | 0.15 | 0 | 0.7637 | 0.8120 | 0.8471 |
| 1e-4 | 15 | 64 | 0.25 | 0 | 0.7670 | 0.8047 | 0.8471 |
| 1e-4 | 15 | 16 | 0.15 | 0 | 0.7780 | 0.8170 | 0.8538 |
| **5e-5** | 15 | 64 | 0.15 | 0, 1 | **0.7749 ± 0.0041** | **0.8231** | **0.8529** |
| 5e-5 | 30 | 64 | 0.15 | 0 | 0.7768 | 0.8176 | 0.8550 |
| 5e-5 | 15 | 16 | 0.15 | 0 | 0.7683 | 0.8206 | 0.8569 |

**Verdicts:**

1. **lr 5e-5 beats 1e-4 by +1.1 test F1.** Both 5e-5 seeds land above both 1e-4
   seeds. With two seeds per side the effect is consistent, but its size is not
   yet tight. The 8B graph arm now sits +4.0 over the 1B graph arm (73.5) and
   +2.6 over the 1B flat control (74.9).
2. **Epochs are not the lever.** 22 epochs at 1e-4 lands exactly on the 15-epoch
   mean. 30 epochs at 5e-5 lands +0.2 over its 15-epoch mean, inside the noise:
   dev F1 reached ~76-77 by epoch 12 and then held a 75-79 band for 18 epochs
   without turning over. 15 epochs is enough at either lr.
3. **Smaller adapter, yes; more dropout, no.** lora_r 16 at lr 1e-4 matches the
   5e-5 arm (+1.7 over its same-seed control, one seed). lora_dropout 0.25 is
   +0.6, inside the 1e-4 seed spread. Both effective changes reduce how hard
   the 8B fits the train set.
4. **The two gains do not stack.** lr 5e-5 with r16
   ([`052`](configs/052_webqsp_8b_lr5e5_r16.jsonc)) lands at 76.83, about 1 F1
   under either parent at the same seed (77.90, 77.80). Dev F1 plateaued at
   76-78 from epoch ~4 on, so the run is not undertrained on schedule: lowering
   either knob captures the gain, and lowering both costs a little. lr 5e-5 at
   r64 stays the recipe.
5. **Host memory grows with run length at 8B.** 050's first attempt was
   OOM-killed at epoch ~17 on 256G (4 ranks), partway through writing a
   checkpoint. 049's first attempt stalled at epoch ~20 on 160G (2 ranks) until
   the NCCL allreduce timed out after 30 min. Both resumed cleanly with
   `--resume-from` at double the memory. Size runs past 15 epochs at ≥128G per
   rank.

CWQ at 8B, same recipe at 042's 8 epochs:
[`configs/046_cwq_8b_scale.jsonc`](configs/046_cwq_8b_scale.jsonc) (job 165393,
lr 1e-4) and [`configs/048_cwq_8b_lr5e5.jsonc`](configs/048_cwq_8b_lr5e5.jsonc)
(job 165487, lr 5e-5), with the cache from
[`045`](configs/045_cwq_8b_data_prep.jsonc), plus
[`053`](configs/053_cwq_8b_lr3e5.jsonc) (job 165853, lr 3e-5). Measured at
~2.2 s/step on 4 GPUs, so ~21-24 h per run including the test eval.

| lr | epochs | seed | best dev-512 F1 | test F1 | Hits@1 | Hit* |
|---:|---:|---|---:|---:|---:|---:|
| 1e-4 | 8 | 0 | 64.6 (ep 6.5) | 0.5903 | 0.6109 | 0.6403 |
| **5e-5** | 8 | 0 | 68.1 (ep 6.5) | **0.6206** | **0.6449** | **0.6729** |
| 3e-5 | 8 | 0 | 65.7 (ep 8.0) | 0.6032 | 0.6270 | 0.6565 |

At lr 1e-4 the 8B graph arm lands +3.8 F1 over the 1B graph arm on the same
recipe (042, 55.2 over 3 seeds) and +0.4 over the 1B flat control (58.6). That
is the first CWQ run where the graph arm is not below flat. It sits 0.4 F1 and
0.6 Hits@1 under dense-retrieval GNN-RAG (59.4 / 61.7) and +5.7 F1 over the
retrieval-matched SR-only GNN-RAG (53.3). Dev-512 plateaued at 64-65 from
epoch 6 and ran ~5.5 over test, in line with the 1B selection inflation.

lr 5e-5 adds another +3.0 F1 on top, the same direction as WebQSP but three
times the size. It ran 2-5 dev F1 ahead of lr 1e-4 at every eval from epoch
2 on. At 62.06 F1 / 64.49 Hits@1 it is +6.9 F1 over the 1B graph arm, +3.5
over the 1B flat control, and above both dense-retrieval GNN-RAG (59.4 / 61.7)
and GNN-RAG + RA (60.4 / 62.8), on the sparser SR retrieval. One seed per lr.

lr 3e-5 turns back down: 60.32 F1, -1.7 under 5e-5, though still over 1e-4.
It trailed 5e-5 on dev at every eval, flattened at 65-66 from epoch 4.5, and
never reached 5e-5's 67-68 band, so the 8-epoch budget is not what held it
back. lr 5e-5 is the CWQ optimum on this grid, as on WebQSP.

### Data-format v2 sweeps (historical)

All v2-era sweeps, merged (test set, 1628 questions, sorted by test F1; per-sweep
reports live in `results/<sweep>/report.md`). Fixed across every run: Llama-3.2-1B,
lora_r 16, max_nodes 512, n_max 20, versions 8, one B200. The sweeps also differ in
seed (42 vs 0) and checkpoint selection (`baseline`: 128 dev samples; `relmode_khop`:
full 246-graph dev split), so cross-sweep deltas under ~1 F1 are noise.

| sweep | k_hop | rel_mode | lr | bias_lr | epochs | train time | test F1 | Hits@1 | Hit | F1 strict | dev F1 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | last_1 | 1e-4 | 5e-3 | 15 | 1h 49m | **66.90** | **75.43** | **80.41** | 65.27 | 66.58 |
| relmode_khop | 0 | last_1 | 5e-5 | 1e-2 | 30 | 3h 25m | 66.45 | 73.03 | 79.73 | 64.85 | 67.32 |
| baseline | 0 | last_1 | 3e-4 | 5e-3 | 15 | 2h 02m | 65.88 | 73.65 | 79.91 | 64.35 | 67.81 |
| relmode_khop | 0 | last_2 | 5e-5 | 1e-2 | 30 | 3h 43m | 65.61 | 73.77 | 79.79 | 64.26 | 67.02 |
| relmode_khop | 5 | last_2 | 5e-5 | 1e-2 | 30 | 3h 36m | 61.64 | 69.53 | 75.86 | 59.91 | 63.64 |
| relmode_khop | 5 | last_1 | 5e-5 | 1e-2 | 30 | 3h 28m | 60.51 | 68.61 | 76.66 | 59.08 | 62.05 |
| baseline | 2 | last_1 | 1e-4 | 5e-3 | 15 | 1h 45m | 48.76 | 58.17 | 65.72 | 47.63 | 48.02 |
| baseline | 2 | last_1 | 3e-4 | 5e-3 | 15 | 1h 49m | 47.60 | 56.88 | 64.07 | 47.63 | 46.39 |

Takeaways:

- **k=0 runs plateau at 66.5 ± 0.7 test F1** across both lrs, both horizons, both
  bias lrs and both rel_modes. The 30-epoch runs converge (dev F1 flat from step
  ~7000/9600), so training time is exhausted at this configuration — and the
  plateau is reachable in under 2 GPU-hours.
- **k_hop gate: 0 > 5 >> 2.** k=2 collapses because the prompt node (edges only to
  topic entities) sits 5 *Levi* hops from a 2-KG-hop answer — the generating node
  cannot attend to any answer. k=5 restores reachability (no collapse) but still
  costs 5–6 F1: hard gating hurts even when everything is reachable.
- **rel_mode is a wash at k=0** (mixed signs, within noise) despite 49% of
  questions carrying a within-subgraph relation-text collision under `last_1` —
  the model resolves the ambiguity from graph context. `last_2` costs ~+32%
  tokens (~+15 min here); keep `last_1`.

These bound how well *any* model can do given SR retrieval — the input either contains
the answer or it doesn't. WebQSP reports **macro** metrics (per-question, averaged over
questions), so the macro rows are the operative ceilings; micro is diagnostic only.

Measured from `data/sr-webqsp/{train,dev,test}.json` (**data-format v3** — newline
delimiter + naming v2; the v2 tables live in the `_dfv2` caches'
`coverage_analysis.json`). Each cell is **uncapped / capped**:

- **uncapped** — the gold's `kb_id` occurs anywhere in the raw `subgraph.tuples`:
  the pure SR-retrieval ceiling, before any data prep of ours.
- **capped** — the gold survives the actual pipeline graph
  (`select_triples(max_nodes=512)` → Levi → CVT collapse) **and** has a scoreable
  text (its `text`, else a literal `kb_id` — dates/numbers/codes; see
  `answer_text`): exactly the `present_answer_texts` criterion that decides what
  the built `.gtds` can supervise.

"Perfect precision" = model emits only correct, present golds; `N_max=20` =
generation capped at 20 answers.

Reproduce with:
```
python3 -m src.experiments.kgqa --mode data_prep --analyse-dataset
``` 
(see `analyse_dataset.py`; prints these tables and saves `coverage_analysis.json`
next to the built splits).

| Ceiling (uncapped / capped) | **test** (n=1628) | train (n=2826) | dev (n=246) | Bounds |
|---|---|---|---|---|
| ≥1 gold present per question | **91.1% / 90.9%** | 92.6% / 92.3% | 89.8% / 89.8% | **Hits@1** |
| Recall — macro (avg per-q present/total) | 89.2% / 88.6% | 90.5% / 89.7% | 86.7% / 86.4% | per-q recall |
| Recall — micro (Σpresent/Σtotal) | 63.3% / 54.5% | 56.9% / 47.9% | 34.0% / 32.6% | answer-instance recall *(diagnostic)* |
| **F1 — macro**, perfect precision, uncapped | **89.6% / 89.1%** | 91.0% / 90.3% | 87.1% / 86.9% | **macro-F1 (WebQSP metric)** |
| Recall — macro, cap N_max=20 | 86.4% / 86.0% | 87.9% / 87.4% | 85.2% / 85.2% | — |
| F1 — macro, cap N_max=20 | **87.4% / 87.1%** | 88.9% / 88.5% | 86.1% / 86.1% | macro-F1 under our cap |

**Reading it:**
- Operative test ceilings for models trained/scored on the built dataset:
  **Hits@1 ≤ 90.9%**, **macro-F1 ≤ 89.1%** (→ **87.1%** under the N_max=20 cap).
  Against raw SR retrieval: 91.1% / 89.6% (→ 87.4%). Pipeline losses are now ≈0:
  the capped ceilings sit within 0.2–0.5 pts of raw SR (v2 had a 1.4-pt gap).
- On the stricter **text-generatable** criterion (some node text contains a
  normalized gold — what generation can actually copy), the test Hit ceiling moved
  **87.5% (v2) → 92.1% (v3)**: naming v2 recovered 74 test questions. It can exceed
  the kb_id-based row because a gold string can also appear inside another node's text.
- micro ≪ macro: entirely the enumeration tail (6.8% of questions have >20 golds,
  up to 3688). Micro weights every (q, answer) pair equally so those questions dominate; it is
  **not** a benchmark ceiling — don't optimize for it.
- The N_max=20 cap costs only ~2 macro-F1 points → cheap (and the n_max=50 recall arm
  confirmed it empirically: +0.35 F1, within noise).
- All rows assume perfect precision, so real achievable numbers are strictly below.
- GNN-RAG's SR Hits@1 (~78.9) sits ~12 pts under even the capped ceiling — that gap is the
  graph-reasoning headroom GTLM targets (genuine reasoning, not retrieval failure).

### Why the two ceilings differ (drop decomposition)

Per question, the first gate that removes its last present gold. **Train** keeps
only the answerable questions (there is nothing to supervise otherwise);
**dev/test keep every answered question** — the non-answerable ones as
empty-target rows that score ~0 — so all eval metrics use the full benchmark
denominators out of the box (no post-hoc correction needed).

| Questions (data-format v3, v2 in parens) | train | dev | **test** |
|---|---|---|---|
| **answerable** (supervisable; = train's kept rows) | 2607 (2573) | 221 (217) | **1480 (1460)** |
| answer not in SR subgraph (retrieval failure) | 209 | 25 | 145 |
| retrieved, no scoreable answer text | 0 | 0 | 0 |
| lost to the `max_nodes=512` cap | 10 | 0 | 3 |
| lost to CVT collapse | 0 (34) | 0 (4) | **0 (20)** |
| **total answered** (dev/test `.gtds` size = eval denominator) | 2826 | 246 | **1628** |

- The `max_nodes=512` cap is nearly free: 3/1628 test questions (0.2 pt). Raising it is
  not the lever — an *uncapped* build recovers only those 3 while the N×N features
  (SPD, magnetic, attention bias) grow quadratically. `max_nodes` stays **512**.
- The bulk (145/1628 test, 8.9%) is SR retrieval failure — the answer is nowhere in the
  retrieved subgraph. GNN-RAG faces the identical bound; unfixable in data prep.
- The old "retrieved but no `text`" bucket (24 test questions) is gone: those golds are
  *literals* (dates, numbers, currency codes) whose `kb_id` is the answer string itself,
  and `answer_text` now falls back to it — the same string the graph shows for the node.
- CVT-collapse losses are **zero since naming v2** (data-format v3). The A1 audit
  (`results/error_analysis/audit_pipeline_losses.py`) showed the v2 losses were never
  true CVTs: they were real named entities missing from `entities_names.json`, whose
  "unnamed entity" fallback text made `_collapse_cvts` contract them as presumed
  mediators. Naming v2 (`build_entities_names_v2.py`: in-subgraph `type.object.name`
  triples + the FB5M Freebase name dump, 560k → 598.5k entries) names them, which both
  fixes their node text and stops their collapse.

### Evaluation parity with GNN-RAG

Benchmark comparability is exact, not approximate:

- `evaluate.py`'s primary metrics are **verbatim ports of GNN-RAG's**
  `llm/src/qa_prediction/evaluate_results.py` (normalized-substring `match`,
  their F1/Hits@1/Hit definitions). Our stricter exact-set variants are logged
  with a `_strict` suffix.
- Gold lists (`graph['gold_answers']`) mirror RoG/GNN-RAG's `answer` lists:
  verified **identical by question id on 1628/1628 test questions** against
  `rmanluo/RoG-webqsp` (their test split is exactly our 1628 answered questions).
  Like theirs, golds with no name anywhere stay as raw-mid placeholders that never
  match — deflating recall for us exactly as it does for them.
- **CWQ parity** (2026-07-12, `check_rog_parity.py` — the now-scripted version of
  the check above): all **3,531** RoG-cwq test questions are present in sr-cwq by id
  (that count is the pinned eval denominator; every test record is answered). Gold
  lists are identical on 3,471; the 60 diffs are benign — 57 differ only by
  *duplicate* strings in RoG's lists (ours dedupe; marginally conservative for us,
  since their recall denominator counts duplicates), 3 by a RoG-side `:m.0mmyl` vs
  our `m.0mmyl` (identical after their `normalize`). No question is missing a gold.
- `entities_names.json` (node naming) started as the file shipped in GNN-RAG's
  `llm/` folder (560k entries, preserved at `entities_names.v1.json`); naming v2
  extends it to 598.5k with Freebase-native aliases only (in-subgraph
  `type.object.name` triples + the FB5M name dump — see
  `build_entities_names_v2.py`). No per-question answer-text harvesting: gold
  texts still feed ONLY targets/eval, never node text.

### Built-split token lengths

Total tokens per stored example (sum over all node texts of one graph, ≈3 tokens/node;
the train split stores `versions`=8 answer-order augmentations per question).
Measured on the dfv2 build; v3 graphs are marginally larger (fewer collapses,
+34 train questions):

| Split (examples) | mean | p50 | p75 | p90 | p95 | p99 | max | tokens/node |
|---|---|---|---|---|---|---|---|---|
| train (n=20584) | 331 | 162 | 375 | 923 | 1272 | 1701 | 2414 | 3.04 |
| dev (n=246) | 332 | 155 | 396 | 907 | 1309 | 1606 | 1859 | 3.04 |
| test (n=1628) | 353 | 179 | 393 | 1032 | 1324 | 1656 | 2259 | 3.01 |

Answer-set sizes: median 1, mean 11.2; 52% single-answer, 31% 2–5, 10.5% 6–20, 6.8% >20.

### CWQ answer-coverage ceilings (E1.1, 2026-07-12)

Same methodology as the WebQSP tables above (uncapped = gold `kb_id` anywhere in the
raw decoded `subgraph.tuples`; pipeline = survives `select_triples` → Levi → collapse
with a scoreable text). Built at `n_max=50`, `versions=1` (CWQ answer sets are
near-singleton — median 1, mean 2.0 — so answer-order augmentation is a no-op;
`ver1` also keeps the 27.6k-question train build in memory).

**Cap decision — the coverage-vs-cost knee is `max_nodes=1024`** (test-split
pipeline ceilings; uncapped ceiling 80.7 Hits@1 / 79.8 F1):

| `max_nodes` | Hits@1 ceiling | F1 ceiling | questions lost to cap | node count p95 (built) |
|---|---:|---:|---:|---:|
| 512 | 79.0 | 78.1 | 57 | 512 |
| **1024** (chosen) | **79.9** | **79.0** | **24** | **1021** |
| 2048 | 80.4 | 79.5 | 6 | 1754 |

512→1024 buys +0.9/+0.9 ceiling; 1024→2048 buys only +0.5/+0.5 while p95 built
node count grows 1.7× (quadratic SPD/magnetic + attention cost, ~3× at the tail).
WebQSP stays at 512. `n_max=50` is moot on CWQ: capped ceilings equal uncapped on
every split (no enumeration tail).

Per-split ceilings at the chosen cap (uncapped / pipeline, `cap1024`):

| Ceiling | test (n=3531) | train (n=27639) | dev (n=3519) |
|---|---|---|---|
| ≥1 gold present (bounds Hits@1) | 80.7% / 79.9% | 85.2% / 84.8% | 81.6% / 81.3% |
| Recall — macro | 79.5% / 78.7% | 83.8% / 83.3% | 79.8% / 79.5% |
| **F1 — macro, perfect precision** | **79.8% / 79.0%** | 84.1% / 83.7% | 80.2% / 79.9% |

Drop decomposition (test, first failing gate): 2822 answerable, **682 not
retrieved** (19.3% — SR retrieval failure dominates, exactly the mechanism GNN-RAG's
paper blames for SR hurting them on CWQ), 24 lost to the cap, 2 no-text, 1 collapse.
The retrieval-matched target (SR-only GNN-RAG, Table 15(d): 55.6 Hits@1 / 53.3 F1)
sits ~24 pts under this ceiling — far more reasoning headroom than WebQSP offered.

Built-split token lengths at `cap1024` (per stored example; `ver1`, so train has one
row per question):

| Split (examples) | mean | p50 | p75 | p90 | p95 | p99 | max | tokens/node |
|---|---|---|---|---|---|---|---|---|
| train (n=23442) | 1098 | 570 | 1655 | 2968 | 3521 | 4515 | 11146 | 3.23 |
| dev (n=3519) | 1163 | 619 | 1908 | 2931 | 3404 | 4706 | 7312 | 3.20 |
| test (n=3531) | 1086 | 596 | 1626 | 2928 | 3404 | 4268 | 7276 | 3.13 |

**Flat-arm `seq_len` for CWQ = 8192.** Measured on the raw flat serialization at
`cap1024` (prompt + triple lines + target, incl. EOS): test p50 1394 / p95 6597 /
p99 7845 / max 25427; a 8192 cap covers 99.1–99.4% per split (matching WebQSP's
"4096 covers 99.6%" standard), while WebQSP's 4096 would cover only ~76% of CWQ.
Outliers drop trailing triple lines, as on WebQSP. The collapsed serialization
(the D2b winner) is slightly shorter, so 8192 covers it a fortiori.

### Baseline retrieval ceilings: SR vs the RoG 2-hop pool (2026-09-23)

The ceilings above bound our input condition. The baselines that run their own
retrieval are bounded by something else, and no paper in the landscape table
publishes it — GNN-RAG reports no subgraph answer coverage in either the ACL
version or the arXiv one. Measured directly instead, over the `graph` field of
`rmanluo/RoG-{webqsp,cwq}` (the 2-hop neighbourhoods RoG and GNN-RAG retrieve
from), with `_ceilings`' definitions reused verbatim so the rows are comparable:

    python -m src.experiments.kgqa.analysis.rog_pool_ceiling --datasets webqsp cwq

| Input condition (test split) | WQSP H@1 | WQSP F1 | CWQ H@1 | CWQ F1 |
|---|---:|---:|---:|---:|
| RoG / GNN-RAG 2-hop pool | **95.6** | **94.3** | 80.7 | 79.5 |
| SR, uncapped | 91.1 | 89.6 | 80.7 | 79.8 |
| SR, as built (cap512 / cap1024) | 90.9 | 89.1 | 79.9 | 79.0 |

**This is an upper bound on the baselines, not their ceiling**: it measures the
pool they retrieve *from*, and their GNN selects a subset of it. The matching key
also differs — RoG ships name-resolved triples, so gold strings are intersected
with node strings where the SR path intersects mids. Both ask "is the gold a node
of the retrieved graph", and unnamed-mid golds never match on either side.

**WebQSP: their pool is genuinely richer** — 94.3 vs our operative 89.1 F1, a
5.2-point advantage. Any WebQSP comparison against a dense-retrieval baseline is
made from 5 points further back.

**CWQ: the ceilings are the same, and that is not an artifact.** 80.69 vs 80.66
Hits@1 is close enough to look like a shared data source or a bug, so
`analysis/ceiling_agreement.py` checks whether the same *questions* are covered:

| CWQ test, ≥1 gold present | count | share |
|---|---:|---:|
| covered by both | 2513 | 71.2% |
| SR only (pool misses it) | 336 | 9.5% |
| pool only (SR misses it) | 335 | 9.5% |
| neither | 347 | 9.8% |

Two different retrievals whose totals coincide: 21.6% of questions differ in
per-question recall, almost perfectly symmetrically. The same script on WebQSP
disagrees strongly and in one direction (106 pool-only against 32 SR-only,
91.09 / 95.64), which is what shows the comparison is not forced to agree. The
sources are independent — sr-cwq is SR's own retriever over the int-coded
Freebase cache (see `sr_records.py`), not RoG's neighbourhoods.

**The two pools are nowhere near the same size**, which is where the extra
WebQSP coverage comes from (test split, raw retriever output on both sides, no
cap of ours applied):

    python -m src.experiments.kgqa.analysis.subgraph_sizes --datasets webqsp cwq

| Retriever | dataset | nodes (mean / p50 / p95 / max) | triples (mean / p50 / p95 / max) | relations (mean) |
|---|---|---|---|---:|
| SR (raw) | WebQSP | 75 / 29 / 284 / 1182 | 90 / 35 / 356 / 1741 | 10.7 |
| RoG 2-hop pool | WebQSP | 1389 / 1530 / 1995 / 1999 | 4309 / 4415 / 7983 / 10810 | 295.0 |
| SR (raw) | CWQ | 204 / 94 / 854 / 1715 | 268 / 119 / 1017 / 4487 | 14.5 |
| RoG 2-hop pool | CWQ | 1281 / 1481 / 1989 / 1998 | 4273 / 4143 / 9043 / 17049 | 268.6 |

SR is a precision-oriented retriever and it shows: 18× fewer nodes and 48× fewer
triples than the pool on WebQSP, 6× / 16× on CWQ, over ~20× fewer distinct
relations. The two-hop pool is a candidate set, not a reader input — RoG and
GNN-RAG prune it to paths before anything reaches the LLM — so this is not a
prompt-length comparison. It is what the +5.2 WebQSP F1 of ceiling costs: the
pool buys its coverage by being an order of magnitude larger, and on CWQ that
same order of magnitude buys nothing (79.5 vs 79.8 F1).

Note the pool's own ceiling is capped: max nodes 1999 (WebQSP) and 1998 (CWQ),
with p50 already at ~1500, so `rmanluo/RoG-*` is itself truncated at 2000 nodes.
The pool ceilings are therefore an upper bound on a *already-capped* pool, not on
unrestricted 2-hop reachability.

Consequences worth carrying into the write-up:

- The CWQ gap to dense-retrieval GNN-RAG (55.2 vs 59.4 F1) is **not** a coverage
  gap. Both conditions bound any reader at ~79 F1, so that margin is reading.
- GNN-RAG loses 6.1 F1 on CWQ (59.4 → 53.3) swapping dense for SR under an
  unchanged ceiling — their paper's stated mechanism, disconnected sparse
  subgraphs breaking shortest-path extraction. There is no path-extraction stage
  here to break.
- **Union ceilings are far above either retriever**: CWQ 90.2 Hits@1 (+9.5 over
  both), WebQSP 97.6 (+2.0 over the pool, +6.5 over SR). That is the concrete
  size of the prize in [Goal](#goal) step 4, and a better argument for it than
  "a stronger public retriever".

### Entity-redundancy check: flat vs Levi-graph token cost (2026-07-13)

Flat serialization repeats an entity's text once per triple it appears in; the
Levi graph names each entity once regardless of degree — a candidate
explanation for the flat-beats-graph result (D2/D2b) if SR's subgraphs simply
lack the entity reuse needed for that dedup to pay off. Measured directly on
the test split at each dataset's trained `max_nodes` cap (real
`select_triples`/`resolve_entity_text`/`verbalize_relation`, Llama-3.2-1B
tokenizer): the flat/graph entity-token redundancy ratio is **1.99× on WebQSP
vs 2.07× on CWQ** (avg entity degree 2.34× vs 2.48×) — essentially flat despite
CWQ's subgraphs being 3× bigger (67.6 vs 202.3 triples/question), since both
flat and graph token counts scale up by the same factor. So **CWQ does not
test the token-dedup hypothesis**: SR's retriever pulls near-shortest-paths,
which structurally caps entity reuse regardless of hop count, so a real test
needs a hub-heavy retrieval topology (e.g. PPR) instead of just more SR hops —
untested here, and blocked on the O(N²) SPD/magnetic bias cost that already
caps `max_nodes` today. CWQ remains a fair checkpoint for the *other* candidate
mechanism (graph's explicit distance bias vs. flat's serial-position distance
cues degrading over long context — "lost in the middle"), since its flat
sequences are ~3× longer in absolute tokens (598 → 1849/q).
