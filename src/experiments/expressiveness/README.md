## Expressiveness of our Graph-aware Positional Encodings

**This experiment is used to generate the simulated message passing visualisation in our paper! See [this section](#replate-the-results) for instructions on how to replicate the experimental results.**

In this experiment, we will try to train both the GraphLlamaModel and the LlamaModel on a simple task to show that the modified graph-aware positional encodings empower the LLM to see graphs, without having explicit information about edges in the token sequence.

### Problem setup
- The model is given a set of $N$ node labels $l_1, l_2,... l_N$ (possibly letters or numbers).
- These nodes will form a graph, with two distinct connected components $\{l_{i_1},...l_{i_{M}}\}$ and $\{l_{i_{M+1}},...l_{i_N}\}$, where $M<N$.
- The model is asked the following question: *"Are nodes $l_x$ and $l_y$ connected?"* $\rightarrow$ *Yes/No*
- We will use the Laplacian coordinates and Shortest Path Distance matrix to see if those are expressive enough to solve the task.
- **HARD version** - this version of the problem contains a variable number of components, where they are not fully connected and the graph contains directed edges 

### Hypothesis
1. The default LlamaModel will not have any information on the graph structure, as it is only presented the set of nodes without edge information. Therefore, the model cannot perform better than a random binary classifier (~$50\%$ accuracy)
2. The GraphLlamaModel will be able to learn the simple pattern and answer with high accuracy, as a result of the graph-aware positional embeddings.

### Findings
Success ✅

## Refactor validation (v0 vs v2)

Beyond its original role, this experiment now doubles as the **end-to-end
validation harness for the backbone-agnostic GTLM refactor**. The same HARD
connectivity task is run across all three implementations against a single
shared `.gtds` dataset (the `TextGraphDataset` is implementation-agnostic — only
the collator / forward contract / trainer differ):

| Implementation | Model | Collator | Backend |
|---|---|---|---|
| `v0-eager` | `GraphLlamaForCausalLM` (legacy) | `GraphCollator` | eager (accuracy anchor) |
| `v2-eager` | `GTLMLlamaForCausalLM` | `GraphCollatorV2` | eager (refactor parity) |
| `v2-flex` | `GTLMLlamaForCausalLM` | `GraphCollatorV2` (block-padded) | flex (efficient path) |

GTLM has two backends — `eager` (dense reference) and `flex` (the sparse
speedup); `sdpa` is not a distinct backend (a custom dense bias makes the fused
kernels unusable/unhelpful) and is accepted only as an alias for `eager`. So the
bit-parity control is `v2-eager` vs `v0-eager`, checked forward *and* backward by
`tests/test_modeling_gtlm_llama_v2.py::test_v2_eager_matches_v0_backward`.

This validates two claims:
1. **The math still works** — small graphs (10–25 nodes) trained to convergence
   should give v0-eager ≈ v2-eager accuracy (the eager backends are the
   bit-parity control; flex is numerically checked by the unit tests and its
   *speed* is a `bench`-mode concern, not a learning one). See
   [Results](#results--v0v2-equivalence--k-hop-effect-eager) below.
2. **It is faster / leaner at scale** — large graphs (~500, then 1000+) measured
   for step-time and peak CUDA memory; v2-flex is expected to pull ahead while
   v0-eager OOMs at the top of the sweep. (At the small *training* sizes here
   flex is actually slower — its block-sparse kernel only pays off once N is
   large; that is exactly what `bench` mode isolates.)

### Knobs
- **Arbitrary graph size** — node labels use a scalable spreadsheet-style scheme
  (` A`…` Z`, ` AA`, ` AB`, …) instead of the old 26-letter cap, so HARD
  datasets of any node count can be generated.
- **Structural sparsity** `k_hop` — defaults to the sweep `{0, 1}` (the generated
  graphs are relatively dense, so `k=2` is usually unnecessary), but it is a free
  parameter and larger values are supported.
- **Graph size sweep** — small (10–25) for learning/parity, ~500 and 1000+ for
  throughput/memory.

### Results — v0/v2 equivalence & k-hop effect (eager)

Eager-only study (flex excluded — see claim 2 above; flex is a throughput
concern for `bench` mode, not a learning one). HARD task, frozen Llama-3.2-1B
backbone with only the ~173 k graph-bias params trainable; 2 000 train / 500 eval
graphs of 10–25 nodes (`2k_hard_n10-25_dataset.gtds`, `500_hard_n10-25_dataset.gtds`),
3 epochs. Eval accuracy = best-checkpoint Yes/No correctness (v0 scores the
Yes/No token directly; v2 uses prompt-span exact match — both measure the same
answer).

A single run cannot separate a real effect from training noise, so each config
is run over **3 seeds** and reported as mean ± std. The harness re-seeds before
each model build (`set_seed(seed)` in `__main__.py`), so within a seed every
config shares its bias-param init and data ordering — a valid **paired**
comparison. (Seed 0 is the initial run, which shared data ordering but not init;
seeds 1–2 are fully paired. Reproduce all three with `SEEDS="0,1,2"`.)

| Config | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| `v0-eager` k=0 | 0.884 | 0.852 | 0.858 | **0.865 ± 0.017** |
| `v2-eager` k=0 | 0.894 | 0.854 | 0.836 | **0.861 ± 0.030** |
| `v2-eager` k=1 | 0.880 | 0.828 | 0.812 | **0.840 ± 0.036** |

Note the absolute accuracy swings ~0.03–0.04 across seeds (seed 0 happened to be
high for *every* config) — which is exactly why pairing matters: it cancels the
shared seed/data-ordering luck.

- **v0 ↔ v2 equivalence (k=0): confirmed.** Means differ by 0.004, far inside the
  ±0.02–0.03 seed spread, and the within-seed gap **flips sign** across seeds
  (+0.010, +0.002, −0.022). No systematic difference — the refactored v2
  reproduces v0. (This complements the exact forward/backward bit-parity unit
  test, which checks the numerics directly.)
- **k-hop effect: a small, consistent penalty.** In *absolute* terms 0.861 vs
  0.840 overlaps heavily (unpaired two-sample t ≈ 0.8, not significant). But the
  **paired** within-seed gap `k0 − k1` is **positive in all 3 seeds**
  (+0.014, +0.026, +0.024; mean **+0.021**) — k=0 beats k=1 every time. This is
  suggestive of a real but small effect, in the expected direction: restricting
  to 1-hop structural attention drops some multi-hop reachability signal the
  connectivity task can use, which the 16-layer depth only partly compensates
  for. With n=3 (df=2) it is not nailed down statistically; a few more paired
  seeds (the paired test has good power despite the noisy absolute accuracy)
  would confirm it.

## Replate the Results

To train the model on our synthetic task, run the following command:
```
python3 -m src.experiments.expressiveness
```
Make sure that the model is being trained on the "EASY" variant of the problem, for quicker convergence and clearer visualisation.

To visualise the attention map, run:
```
python3 -m src.utils.plot_attention
```
Again, make sure that the correct checkpoint paths are used in the plot_attention module.

## TODO — refactor-validation rework

- [x] **`data_gen.py`**: replace the 26-element `LETTERS` cap
      (`random.sample(LETTERS, len(G.nodes))`) with a scalable spreadsheet-style
      unique-label scheme so HARD datasets of arbitrary node count generate.
- [x] **`data_gen.py`**: verify a large HARD dataset (~500 and 1000+ nodes)
      generates end-to-end (features + tokenize + labels).
- [x] **`__main__.py`**: add a v0/v2 dispatch (model + collator + forward + trainer)
      so the run is selectable across `v0-eager`, `v2-eager`, `v2-flex`.
- [x] **`__main__.py`**: thread `k_hop` (default sweep `{0, 1}`), backend, and
      graph-size mode through as flags; make `report_to` a flag (default `"none"`).
- [x] **`__main__.py`**: port the custom eval — v0 keeps the
      `smuggle_prediction_step` / `PreprocessLogits` / `ComputeMetrics` path; v2
      uses `GraphTrainerV2`'s own prompt-span exact-match metrics (flat layout).
- [x] **`__main__.py`**: add the large-graph mode — few-step throughput + peak
      CUDA memory instrumentation (`torch.cuda.max_memory_allocated`), `MODE="bench"`.
- [x] **`__main__.py`**: multi-seed sweep with `set_seed` so parity / k-hop are
      read as mean ± std (paired within seed); `SEEDS` env-overridable.
- [ ] **`evaluate.py`**: support both v0 (`input_graph_batch=…`) and v2 (flat
      columns) forward APIs.  *(still v0-only)*
- [x] Smoke-run small graphs — confirmed v0-eager ≈ v2-eager and measured the
      k-hop effect (see [Results](#results--v0v2-equivalence--k-hop-effect-eager)).
- [ ] Large-graph timing/memory pass (`MODE="bench"`, sizes 500/1000) — harness
      ready, not yet run/recorded here.
- [x] Regenerate the HARD `.gtds` artifacts at the needed sizes — generated
      `2k_hard_n10-25_dataset.gtds` and `500_hard_n10-25_dataset.gtds` (self-describing
      node-range names; existing `.gtds` dirs untouched).
- [ ] **Firm up the k-hop effect** — add a few more paired seeds (the paired
      `k0 − k1` gap is consistent but n=3 is underpowered).