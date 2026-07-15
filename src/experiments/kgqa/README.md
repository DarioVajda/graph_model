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
   69.8 F1**, CWQ **55.6 Hits@1 / 53.3 F1**; we are at 78.8 / 72.5 WebQSP after
   data-format v3 — **F1 ahead by 2.7, Hits@1 0.15 short**; the 71.3 F1 cited here
   previously is Table 2's *dense-retriever* GNN-RAG. Per the A2 diagnostic, Hits@1
   structurally under-credits a set generator: Hit/F1 are the honest primary
   metrics). This is the apples-to-apples pipeline-vs-single-model comparison the
   setup was designed for.
2. **Beat the text-serialization ablation**: same base LLM, same SR subgraph flattened to
   triples in the prompt, no structural biases. This isolates whether the graph attention
   biases do anything — the result that transfers to the rest of the repo. *(Not yet run.)*
3. **One scale run (3B/8B)** before drawing architecture conclusions: at 1B, plain
   text-RAG (RPO-RAG) gets 69.8 F1 and gains +11.5 F1 going to 8B — part of our gap may
   be reader capacity, not graph handling.
4. Only after 1–3 are won: **demonstrate retrieval portability** by plugging in one
   stronger public retriever (e.g. SubgraphRAG's) as a second input condition. This lifts
   the coverage ceiling and yields competitive headline numbers without claiming
   retrieval as a contribution.

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
| **WebQSP** | Hits@1 | 80.0 | 80.6 | 82.8 | 91.6 | **78.8** |
| **WebQSP** | F1 | 70.8 | 71.3 | 73.5 | 88.6 | **72.5** |
| **CWQ** | Hits@1 | 57.8 | 61.7 | 62.8 | 79.6 | — *(not run)* |
| **CWQ** | F1 | 56.2 | 59.4 | 60.4 | 74.2 | — *(not run)* |

- **Ours** = best `attribution_v3` run (Llama-3.2-1B, data-format v3, k_hop 0, lr 1e-4,
  n_max 20, seed 1; test Hit 82.7); test set, verbatim GNN-RAG scoring. The 3-seed v3 arm
  means are F1 71.1–71.5 / Hits@1 77.5 (±1.0 seed noise). WebQSP only — no CWQ runs yet.
  See [Results so far](#results-so-far).
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
  model as ours, single-pass RAG-style reading. Hit 82.3 / F1 69.8 WebQSP vs our Hit 82.7 /
  F1 72.5 (data-format v3) — **we now exceed the 2026 preference-optimized text-RAG pipeline
  at equal model size** (+0.4 Hit, +2.7 F1), with a different (their own) retriever.
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

python3 -m src.experiments.kgqa.build_entities_names_v2   # 560k -> 598.5k entries
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

### Data-format v3 sweeps (2026-07-08) — current best

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
