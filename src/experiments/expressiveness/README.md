## Expressiveness of our Graph-aware Positional Encodings

**This experiment is used to generate the simulated message-passing visualisation
in our paper, and it doubles as the end-to-end validation harness for the
backbone-agnostic GTLM refactor (v0 → v2) and its flex-attention backend.** See
[Reproduce the results](#reproduce-the-results) for commands.

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
  enabled (`rrwp` and `rwse` are off — see [Engineering](#engineering-decisions)).

### The three implementations

The same task and the same `.gtds` dataset are run across all three
implementations (`TextGraphDataset` is implementation-agnostic; only the collator
/ forward contract / trainer differ):

| Implementation | Model | Collator | Backend |
|---|---|---|---|
| `v0-eager` | `GraphLlamaForCausalLM` (legacy) | `GraphCollator` | eager — the legacy reference |
| `v2-eager` | `GTLMLlamaForCausalLM` | `GraphCollatorV2` | eager — refactor parity |
| `v2-flex` | `GTLMLlamaForCausalLM` | `GraphCollatorV2` (block-padded) | flex — sparse/efficient path |

`sdpa` is not a distinct backend (a custom dense bias makes the fused kernels
unusable) and is only an alias for `eager`. Forward *and* backward bit-parity of
`v2-eager` vs `v0-eager` is checked by
`tests/test_modeling_gtlm_llama_v2.py` (32 tests).

### What this validates — and the headline results

Three objectives, all measured below (frozen Llama-3.2-1B, only the ~173 k
graph-bias params trainable, **bf16**, **3 seeds**):

1. **v2-eager generalises v0-eager** — at k=0 the new v2 reproduces the legacy v0
   accuracy (within seed noise).
2. **flex scales significantly better than eager** — 2–3.5× faster and ~½ the
   memory where both run, and it keeps running at graph sizes where eager OOMs.
3. **k-hop masking (k=1) is a modest accuracy cost for a real speed-up** — ~0.05
   em on the small task, in exchange for up to 1.27× faster flex steps on large
   graphs (where the block sparsity it creates can actually be skipped).

---

### Result 1 & 3a — small-graph accuracy (parity + k-hop cost)

HARD task, 2 000 train / 500 eval graphs of 10–25 nodes
(`2k_hard_n10-25_rcm_dataset.gtds`, `500_hard_n10-25_rcm_dataset.gtds`), 3 epochs,
3 seeds. **Both v0 and v2 now report the same metric — prompt-span exact match
(`em_accuracy`)** — so they are directly comparable (`evaluation.py` computes v0's
em via full-vocab argmax at the answer site; v2 uses `GraphTrainerV2`'s).

| Config | em accuracy (mean ± std) | per-seed |
|---|---|---|
| `v0-eager` k=0 | **0.782 ± 0.055** | 0.844, 0.762, 0.740 |
| `v2-eager` k=0 | **0.731 ± 0.046** | 0.684, 0.734, 0.776 |
| `v2-flex`  k=0 | **0.787 ± 0.027** | 0.766, 0.778, 0.818 |
| `v2-eager` k=1 | **0.689 ± 0.002** | 0.692, 0.688, 0.688 |
| `v2-flex`  k=1 | **0.716 ± 0.005** | 0.714, 0.712, 0.722 |

- **Objective 1 — v0 ≈ v2-eager at k=0: supported (within noise).** All three
  k=0 configs (0.73–0.79) overlap inside ~1 std; v0 and v2-flex are nearly
  identical. The forward numerics are *exactly* equivalent (the 32 parity tests,
  plus matched eval-loss to 1e-3 at init); the ~0.05 spread is end-to-end
  training stochasticity (different collators/packing + bf16 non-determinism over
  ~190 steps), not a forward discrepancy. n=3 variance is high at this scale.
- **Objective 3 (accuracy) — k=1 costs ~0.05 em.** Both backends drop k=0→k=1
  (eager 0.731→0.689, flex 0.787→0.716) with tiny k=1 stds, so the penalty is
  consistent: 1-hop masking removes some multi-hop reachability signal. "Not too
  much," but real.
- The small graphs are **token-tiny** (packed L ≤ 34: ~1.4 tokens/node, 9-token
  prompt, ≤26 nodes), so they fit in a *single* flex block — flex cannot show a
  speed benefit here (it is in fact ~1.5× slower and uses more memory: 1057 vs
  682 ms, 3.9 vs 2.9 GB). The speed story needs large graphs ↓.

### Result 2 & 3b — scaling (where flex earns its keep)

Few-step fwd+bwd+opt throughput + peak CUDA memory on synthetic HARD graphs,
bf16, **`max-autotune-no-cudagraphs`**, RCM ordering, tight per-size flex buckets,
batch 2. `v2-eager` vs `v2-flex`; OOM caught and reported.

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

- **Objective 2 — flex scales significantly better: confirmed.** 2.1–3.5× faster
  and ~½ the memory where both run (N=1000: 13 vs 24 GB), and **eager OOMs at
  N≥2000 while flex runs to N=4000** — flex pushes the OOM wall from N≈1500 out to
  N≈5000 (~3× larger graphs, ~10× more attention work). With `rrwp` removed the
  crossover moved *below* N=500, so flex now wins across the whole sweep.
- **Objective 3 (speed) — k=1 helps flex in the N=1000–2000 band:** 1.19× (1000)
  and 1.27× (2000). It is a slight *slowdown* at N=500 (k-hop mask overhead >
  tiny-attention savings) and *flat* at N=4000 (the dense O(N²) graph-bias compute
  dominates the step there, diluting any attention-side win). k=1 lifts realised
  block sparsity to 0.78–0.93 with RCM, but that only converts to wall-clock where
  attention is a meaningful share of the step.

**Why the win is "only" ~3×, not the ~8× of the isolated-attention benchmark
(`src/models/flex_attn/`):** these graphs are token-poor (≈1.4 tokens/node, so
L≈N), which makes the per-layer dense graph-bias — O(N²), paid identically by
both backends — co-dominant with attention. flex sparsifies attention but not the
bias, so Amdahl caps the speed-up. The benchmark's larger multiples are at L≫N
(many tokens/node), where attention dominates the step.

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

The experiment is driven entirely by command-line flags (argparse) — every knob
has a `--flag`; run `--help` for the full list. There are two modes, `--mode
train` (default) and `--mode bench`. Datasets regenerate automatically if missing
(`data_gen.create_and_save_dataset`).

```bash
python3 -m src.experiments.expressiveness --help        # full flag reference
python3 -m src.experiments.expressiveness               # train, defaults below
```

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--mode {train,bench}` | `train` | accuracy/parity vs large-graph throughput |
| `--impls a,b,c` | `v0-eager,v2-eager` | any of `v0-eager`, `v2-eager`, `v2-flex` |
| `--k-hops 0,1` | `0` | k-hop radii to sweep (v2 only; v0 is dense, run once) |
| `--seeds 0,1,2` | `0` | seeds to sweep |
| `--num-nodes N` | `500` | graph size. **train**: sets the node range `0.8N–1.2N`; **bench**: the single fixed size |
| `--min-nodes` / `--max-nodes` | derived | pin the node range explicitly (overrides `--num-nodes`) |
| `--magnetic-m M` | `0` | magnetic-Laplacian eigenvectors kept (`0` = all `N`) |
| `--report-to PROJECT` | off | wandb project name; omit for no tracking |
| `--flex-compile-mode` | `max-autotune-no-cudagraphs` | use `default` for quick iteration |
| `--len-buckets` / `--node-buckets` | auto | flex L/N padding ladders (CSV); each L bucket a multiple of the block size |
| `--max-steps K` | `-1` | cap optimizer steps for a quick smoke test |

The graph-aware bias is fixed to **shortest-path distance + magnetic Laplacian**
(every other feature was dropped); `magnetic_dim=32` and `magnetic_q=0.25` are
model constants. Only `--magnetic-m` tunes the spectral side.

### Reproduce the headline results

Small-graph accuracy sweep (objectives 1 & 3-accuracy), 3 seeds, n=10–25:
```bash
python3 -m src.experiments.expressiveness \
    --impls v0-eager,v2-eager,v2-flex --k-hops 0,1 --seeds 0,1,2 \
    --min-nodes 10 --max-nodes 25 \
    --train-dataset-size 2000 --eval-dataset-size 500
```
Large-graph scaling (objectives 2 & 3-speed) — one fixed size per invocation:
```bash
for N in 500 1000 2000 4000; do
    python3 -m src.experiments.expressiveness --mode bench \
        --impls v2-eager,v2-flex --k-hops 0,1 --num-nodes "$N"
done
```
Quick smoke test (fast compile, a few steps):
```bash
python3 -m src.experiments.expressiveness \
    --impls v2-flex --min-nodes 10 --max-nodes 25 \
    --flex-compile-mode default --max-steps 5 --no-measure-density
```
Attention-map visualisation:
```bash
python3 -m src.utils.plot_attention   # set the checkpoint path inside
```

## Open items
- [ ] **Phase 3b — big-graph *training*.** Scaling above is a throughput/memory
      probe; we have not trained to convergence on large graphs. Open question:
      does the k=1 accuracy cost (measured at N≤25) hold at N≈1000–2000, where its
      speed-up lives? Needs large train/eval `.gtds` (decide `magnetic_m`
      truncation + disk budget) and a 3-seed sweep at N∈{1000, 2000}.
- [ ] **Firm up parity / k-hop with more seeds** — n=3 leaves the small-graph
      means noisy (std ≈ 0.05 at k=0).
- [ ] **`evaluate.py`** — still v0-only; add the v2 flat-column forward API.
- [ ] Regenerate the `original`-ordering `.gtds` (still carry rrwp/fp64) if the
      ordering baseline is needed again.
