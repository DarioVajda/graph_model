## Expressiveness of our Graph-aware Positional Encodings

**This experiment is used to generate the simulated message-passing visualisation
in our paper, and it doubles as the end-to-end validation harness for the
backbone-agnostic GTLM refactor (v0 → v2) and its flex-attention backend.** See
[Running experiments](#running-experiments) for commands.

The model is asked to read a graph it is only given as an *unordered set of node
labels* — no edges in the token stream — and answer a connectivity question. If
the graph-aware positional encodings (shortest-path distance + magnetic
Laplacian) are expressive enough, the frozen LLM can "see" the graph.

### Problem setup
- The model is given $N$ node labels $l_1,\dots,l_N$ (a scalable spreadsheet-style
  scheme ` A`…` Z`, ` AA`, … so any node count is supported).
- The labels form a graph with several connected components.
- It is asked *"Are nodes $l_x$ and $l_y$ connected?"* → *Yes/No*.
- **HARD** variant (used here): a variable number of components, not fully
  connected, directed edges.
- Only the **shortest-path-distance** and **magnetic-Laplacian** biases are
  enabled (`rrwp` and `rwse` are off — see [Engineering](#engineering-decisions-and-why)).

### The three implementations

The same task and the same `.gtds` dataset are run across all three
implementations (`TextGraphDataset` is implementation-agnostic; only the
collator / forward contract / trainer differ):

| Implementation | Model | Collator | Backend |
|---|---|---|---|
| `v0-eager` | `GraphLlamaForCausalLM` (legacy) | `GraphCollator` | eager — the legacy reference |
| `v2-eager` | `GTLMLlamaForCausalLM` | `GraphCollatorV2` | eager — refactor parity |
| `v2-flex` | `GTLMLlamaForCausalLM` | `GraphCollatorV2` (block-padded) | flex — sparse/efficient path |

`sdpa` is only an alias for `eager` (a custom dense bias makes the fused kernels
unusable). Forward *and* backward bit-parity of `v2-eager` vs `v0-eager` is
checked by `tests/test_modeling_gtlm_llama_v2.py` (32 tests).

All results below: frozen Llama-3.2-1B, only the ~173 k graph-bias params (+
LoRA adapters where noted) trainable, bf16. Eval metric: prompt-span exact
match (`em_accuracy`, same metric for v0 and v2). Headlines: **v2 reproduces
v0** (within seed noise), **flex is 2–3.5× faster at half the memory and runs
where eager OOMs**, and **k-hop masking trades a real speed-up for an
accuracy cost that grows with graph size**.

---

### Small-graph accuracy — v0/v2 parity and the k-hop cost

HARD task, 2 000 train / 500 eval graphs of 10–25 nodes
(`2k_hard_n10-25_rcm_dataset.gtds`, `500_hard_n10-25_rcm_dataset.gtds`),
3 epochs, 3 seeds:

| Config | em accuracy (mean ± std) | per-seed |
|---|---|---|
| `v0-eager` k=0 | **0.782 ± 0.055** | 0.844, 0.762, 0.740 |
| `v2-eager` k=0 | **0.731 ± 0.046** | 0.684, 0.734, 0.776 |
| `v2-flex`  k=0 | **0.787 ± 0.027** | 0.766, 0.778, 0.818 |
| `v2-eager` k=1 | **0.689 ± 0.002** | 0.692, 0.688, 0.688 |
| `v2-flex`  k=1 | **0.716 ± 0.005** | 0.714, 0.712, 0.722 |

- **v0 ≈ v2 at k=0 (within noise).** The forward numerics are *exactly*
  equivalent (32 parity tests + matched eval-loss at init); the ~0.05 spread is
  end-to-end training stochasticity (collator/packing + bf16 non-determinism),
  and n=3 variance is high at this scale.
- **k=1 costs ~0.05 em**, consistently on both backends (tiny k=1 stds):
  1-hop masking removes some multi-hop reachability signal.
- These graphs are token-tiny (packed L ≤ 34), fitting a *single* flex block —
  flex is actually ~1.5× slower here. The speed story needs large graphs ↓.

### Throughput scaling — where flex earns its keep

Few-step fwd+bwd+opt throughput + peak CUDA memory on synthetic HARD graphs,
bf16, `max-autotune-no-cudagraphs`, RCM ordering, tight per-size flex buckets,
batch 2:

| N | k | eager ms / GB | flex ms / GB | flex vs eager | block sp |
|---|---|---|---|---|---|
| 500 | 0 | 516 / 7.79 | **244 / 6.55** | **2.12×** | 0.33 |
| 500 | 1 | 535 / 7.79 | **275 / 6.55** | **1.95×** | 0.70 |
| 1000 | 0 | 2631 / 23.96 | **910 / 13.05** | **2.89×** | 0.00 |
| 1000 | 1 | 2644 / 23.96 | **766 / 13.05** | **3.45×** | 0.78 |
| 2000 | 0 | **OOM** | 3423 / 29.79 | eager OOM | 0.15 |
| 2000 | 1 | **OOM** | 2703 / 29.80 | eager OOM | 0.92 |
| 4000 | 0 | **OOM** | 13237 / 75.73 | eager OOM | 0.11 |
| 4000 | 1 | **OOM** | 13166 / 75.76 | eager OOM | 0.93 |

- **flex: 2.1–3.5× faster, ~½ the memory, and pushes the OOM wall from N≈1500
  to N≈5000** (eager dies at N≥2000; flex runs to N=4000).
- **k=1 helps flex in the N=1000–2000 band** (1.19–1.27×); it's a slight
  slowdown at N=500 (mask overhead > tiny-attention savings) and flat at
  N=4000. RCM lifts realised block sparsity to 0.78–0.93, but that only
  converts to wall-clock where attention is a meaningful share of the step.
- The win is ~3×, not the ~8× of the isolated-attention benchmark
  (`src/models/flex_attn/`), because these graphs are token-poor (≈1.4
  tokens/node, L≈N): the dense per-layer O(N²) graph-bias — paid identically by
  both backends — is co-dominant with attention, and Amdahl caps the speed-up.

---

### Large-graph results (N=500–2000)

Multi-seed scaling, the magnetic/k-hop ablation at scale, and step-time
profiling. Per-run records: `results/train_runs.jsonl` (standalone runs) and
`results/<sweep>/runs.jsonl` (sweeps).

#### Scaling: the plateau escape holds at scale; k=1 collapses (3 seeds, 20 epochs)

Earlier large-graph runs that looked stuck at the ~0.5 marginal were
**under-training**, not a capability wall: the loss sits on a plateau (longer
for bigger graphs) before a delayed-generalization escape. Trained long enough
(20 epochs, `magnetic_m=128`, 2 000 train / 200 eval, RCM):

| N | k | eval em (mean ± std) |
|---|---|---|
| 500  | 0 | **0.838 ± 0.098** |
| 500  | 1 | 0.530 ± 0.026 |
| 1000 | 0 | **0.757 ± 0.018** |
| 1000 | 1 | 0.555 ± 0.005 |

k=1 sits at the marginal at N≥500 — that's **over-masking, not scale**: a
single-seed k=3 probe at N=500 recovered to 0.825.

#### `big_test`: N=1000/2000 × k ∈ {0,3} × magnetic on/off (Jul 2026, seed 0)

```bash
python3 -m sweep src.experiments.expressiveness src/experiments/expressiveness/configs/big_test.jsonc
```

25 epochs (1 550 steps), spd always on, B300. s/it is wall-clock (total run ÷
steps), not `step_ms_mean`:

| N | k | magnetic | eval em | s/it | train time |
|---|---|---|---|---|---|
| 1000 | 0 | ✔ | **0.750** | 10.4 | 4h 29m |
| 1000 | 0 | ✘ | 0.730 | 5.9 | 2h 32m |
| 1000 | 3 | ✔ | 0.690 | 9.1 | 3h 55m |
| 1000 | 3 | ✘ | 0.730 | 4.5 | 1h 57m |
| 2000 | 0 | ✔ | **0.780** | 36.6 | 15h 47m |
| 2000 | 0 | ✘ | 0.765 | 18.8 | 8h 06m |
| 2000 | 3 | ✔ | 0.715 | 31.0 | 13h 22m |
| 2000 | 3 | ✘ | 0.650 | 13.2 | 5h 42m |

- **N=2000 now trains to convergence**; best config 0.780 (k=0 + magnetic).
- **Magnetic costs ~2× training and ~1.7–2× eval time** for ≤2 pp at k=0
  (within the ±3 pp single-seed / 200-eval noise). Its one clear win (+6.5 pp
  at N=2000 k=3) only recovers part of what masking lost.
- **k=3 is 15–30 % faster** (block sparsity ~0.75) **but costs accuracy at
  scale**: −11.5 pp at N=2000 spd-only, neutral at N=1000.
- Caveat: this task is direction-blind (`to_directed()` symmetrizes every
  edge), so the magnetic Laplacian's directional phase is identically zero
  here **and** spd alone nearly encodes the connectivity answer — don't read
  this as a general keep/drop verdict on magnetic.

#### Step-time profiling (single-axis ablations, 1 seed)

Configs: `exp_num_workers.jsonc`, `exp_bias_ablation.jsonc`. Two clocks
disagree on purpose: **wall-clock s/it** (tqdm, whole iteration — what gates
throughput, includes the between-step dataloader wait) vs **`step_ms_mean`**
(GPU compute inside the step only). Judge loader effects by wall-clock.

- **Dataloader workers: 2.4× wall-clock at N=1000** (26.7 → 11.0 s/it,
  `num_workers` 0 → 8; GPU ms/step identical). The synchronous O(N²)
  spd+magnetic feature build in `__getitem__` stalls the main thread; workers
  prefetch it behind the GPU step.
- **Magnetic bias: 1.66× step compute at N=500** (1.81 → 2.93 s/it, nw=8 on
  both): the per-layer O(N²·m) magnetic einsums are the dominant *added*
  compute; peak memory unchanged. (Single-seed accuracy columns are not
  evidence either way on magnetic — see `big_test` caveat.)

---

### Engineering decisions (and why)

These were tuned during this study; several are now defaults.

- **bf16 weights + autotune for flex.** Loading the (frozen) backbone in bf16
  unlocks tensor cores (≈5–6× over fp32 alone), and `max-autotune-no-cudagraphs`
  is what makes the flex kernel actually fast (≈4.7× fwd at k>0 + 64-wide blocks).
  Together they took flex-vs-eager from a misleading 1.07× (fp32 + default
  compile) to the 2.9× above. `build_model` defaults to bf16; the float graph
  features are cast to match (`causal_lm.py`, and `modeling_gtlm_llama_v0.py` for
  v0).
- **RCM node ordering (default).** Concentrates each node's k-hop neighbourhood
  into contiguous token blocks so flex can skip them — the reason block sparsity
  reaches 0.78–0.93 at k=1. Permutation-safe (accuracy unchanged).
- **Graph-bias gradient checkpointing on** (`checkpoint_graph_bias=True`,
  model-config default) — recomputes the per-layer bias in backward; large memory
  saving at big N, what keeps N=4000 runnable.
- **Only `spd` + magnetic computed.** `rrwp` (the dominant on-disk feature,
  `(N,N,16)`, ~84% of the dataset), `rwse`, and the plain Laplacian coordinates
  were dropped — `data_gen.prepare_dataset` now computes only shortest-path
  distance and the magnetic Laplacian, the two features this task uses.
- **fp32 / int16 feature storage.** Float features are stored fp32 (not the fp64
  that `list→Arrow` produced); SPD is int16 (its 32767 unreachable sentinel is
  int16-native). With `rrwp` gone these cut the on-disk datasets ~4.5×
  (`2k_hard…` 71 → 16 MB, `500_hard…` 18 → 3.9 MB).
- **Magnetic eigenvectors untruncated (`--magnetic-m 0`, full M=N) by default.**
  Truncating buys *no* runtime speed/memory (the output bias is `(B,H,N,N)`
  regardless) and only ~10% disk once `rrwp` is gone, so full spectral resolution
  is kept. `--magnetic-m M` truncates to the `M` lowest eigenpairs for large-graph
  datasets (used consistently by both the dataset generation and the collator),
  gated on an N>32 accuracy ablation.
- **Tight flex buckets.** `len_buckets`/`node_buckets` are passed to the collator;
  `bench` derives a single tight bucket per size from the actual packed lengths.
  Small graphs use `[128]`/`[32]` (one block); default would have padded to 512.

---

## Running experiments

The experiment is a **standalone argparse program** that runs **one**
configuration (`python3 -m src.experiments.expressiveness --impl v2-flex
--k-hop 0 --seed 0 --num-nodes 500 …`; `--help` lists every flag). Sweeps are
driven by the generic top-level [`sweep`](../../../sweep) runner — see
[`sweep/README.md`](../../../sweep/README.md) for the config-expansion
semantics, output layout, and Slurm/sbatch execution. Run everything from the
**repo root** (dataset paths and the config's `results_dir` are
repo-root-relative).

```bash
python3 -m src.experiments.expressiveness --init my_sweep    # template -> src/experiments/expressiveness/configs/my_sweep.jsonc
# ...edit src/experiments/expressiveness/configs/my_sweep.jsonc...
python3 -m sweep src.experiments.expressiveness src/experiments/expressiveness/configs/my_sweep.jsonc
python3 -m sweep.report src/experiments/expressiveness/results/my_sweep      # aggregate runs.jsonl
```

**Dataset preparation** is the experiment's `--mode data_prep` (builds that
config's `.gtds` and exits): run the sweep once with `"mode": "data_prep"` to
build every dataset it needs, then again with `"mode": "train"` (datasets also
build on demand, under an flock, if missing). Generation is host-RAM heavy at
large N — the 2400×2400 SPD feature needs ~80–100 GB, so prep the N=2000
datasets with `--mem=224G` before submitting training.

### Key parameters

| Key | Example | Meaning |
|---|---|---|
| `impl` | `["v2-flex"]` | `v0-eager`, `v2-eager`, `v2-flex` |
| `k_hop` | `[0, 1]` | k-hop radii (v2 only; v0 is dense) |
| `seed` | `[0, 1, 2]` | seeds |
| `num_nodes` | `500` | graph size; node range is `0.8N–1.2N` unless `min_nodes`/`max_nodes` pin it |
| `len_buckets` / `node_buckets` | `[640,768]` / `[512,640]` | flex L/N padding ladders (each L bucket a multiple of the block size) |
| `batch_size` / `accumulation_steps` | `4` / `8` | micro-batch + accumulation (set explicitly per size — usually in a bundle with `num_nodes`) |
| `magnetic_m` | `128` | magnetic-Laplacian eigenvectors kept (`0` = all `N`) |
| `bias_lr` / `lr` | `1e-3` / `1e-4` | graph-bias LR / LoRA(+base) LR |
| `num_epochs`, `eval_steps`, `lora`, `lora_r` | | training schedule + LoRA |
| `wandb_project` | `"GraphLLM"` | wandb project (`null` = no tracking) |
| `flex_compile_mode` | `"max-autotune-no-cudagraphs"` | `"default"` for quick iteration |

**Graph-bias features.** `spd` + `magnetic` are wired end-to-end. The schema also
accepts `laplacian` / `rwse` / `rrwp` (plus `max_spd` / `magnetic_dim` /
`magnetic_q`), but this experiment **rejects** the three unwired features with a
clear error — `data_gen` no longer produces their dataset features.

### Benchmarking (throughput/memory, separate entry)

Bench is a probe, not a sweep, so it keeps an argparse interface and runs one
fixed graph size per invocation:

```bash
for N in 500 1000 2000 4000; do
    python3 -m src.experiments.expressiveness.bench \
        --impls v2-eager,v2-flex --k-hops 0,1 --num-nodes "$N"
done   # appends to results/benchmarks.jsonl
```

Attention-map visualisation: `python3 -m src.utils.plot_attention` (set the
checkpoint path inside).
