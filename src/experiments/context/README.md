# Context exhaustion — needle in a graph

Does GTLM degrade when the graph gets large and the text gets long? The reviewer concern is
that GTLM feeds node text through uncompressed, so attention should dilute as the number of
competing nodes grows. This experiment builds a synthetic retrieval task where the answer is
provably present, sweeps graph size (N nodes) × node length (T tokens per node), and measures
where retrieval breaks — against a flat-serialization LLM on byte-identical data.

**Headline (§3.3):** on a k-hop pointer chase, the graph arm holds at 0.985 mean accuracy
while flat text collapses to 0.641, and the gap *widens* with node count (+0.268 at N=16 →
+0.418 at N=128) and survives 4× length extrapolation. The gap is driven by N and by traversal
depth k, **not** by tokens per node — T has never moved anything in this experiment.

Design details are in [Appendix A](#appendix-a--design-reference); code comments cite it by
section.

---

## 1. What this measures, and what it does not

Two architectural properties bound the claim, and both belong in any writeup:

* **Per-node RoPE reset means position ids never exceed T.** `GraphCollatorV2._pack_one`
  restarts `position_ids` at every node boundary (`node_position_mode="reset"`), so a
  64,640-token cell is 126 segments each positioned 0..511 — not one 64,640-position
  sequence. Whatever the graph arm does here, **it is not RoPE length extrapolation.** The
  failure mode under test is softmax dilution over a growing key set, isolated from the
  positional failure mode that dominates ordinary long-context results. The flat arm, by
  contrast, runs real positions 0..65,301, so the comparison does confound representation
  with positional encoding — see the asymmetry list in §A.11.
* **On a star, the structural biases are nearly information-free.** SPD ∈ {0,1,2} for every
  pair and the magnetic-Laplacian spectrum is degenerate, so the graph channel can encode
  "centre vs leaf" and little else. The star grid (§3.1) therefore measures dilution under
  GTLM's packing and masking regime, **not** structural reasoning. Only the chain task
  (§3.2, §3.3) puts real information in the topology.

The task is **retrieval**, not comprehension: all graphs share one KV-sentence template, so
the model can learn "find the `access code` span whose id matches". That is the mechanism
under test — but say "retrieval" in the paper.

## 2. Layout and running it

Layout follows the `template` experiment (one `RunConfig`, one standalone program, one JSONL
record per run) with `kgqa`'s two-stage data-prep/train split. **Nothing in
`src/utils/text_graph_dataset.py`, the collator, or `src/models/` is modified** — every
deviation is absorbed inside this package.

| File | Responsibility |
|---|---|
| `config.py` | `RunConfig` — every knob once; `validate()`, `data_config_key()`, grid arithmetic (`cells`, `train_cells`, `cell_length`, flex bucket ladders) |
| `data.py` | Graph synthesis: filler corpus, fixed-token-length codes, needle splicing, chain/decoy construction, nested node subsets. Pure CPU, no torch |
| `process_dataset.py` | `.gtds` builds (train/dev mixture + one split per test cell), idempotent, asserts the §A.4 invariants per graph; sharded builds + `data_merge` |
| `model.py` | GTLM construction — fresh and from checkpoint — plus the collator |
| `train.py` | Training run + `CellGroupedSampler` (batches never mix cells; rank-aware for DDP) |
| `evaluate.py` | `ContextGraphTrainer` (windowed loss/eval) + `grid_eval` + `wilson_interval` |
| `flat.py` | The flat-serialization arm: `serialize_graph`, `FlatCollator`, train + grid modes |
| `grid.py` | Score a checkpoint over every cell → `grid.jsonl` |
| `calibrate.py` | Standalone cost calibration (§A.12) |
| `analysis/` | `plot.py` (heatmap + table), `audit_cells.py` (the structural gate, §A.10) |

Two mechanisms run things, and which one applies depends on whether the step needs a GPU:

* **Sweep configs** (`configs/NNN_*.jsonc`, run with `python3 -m sweep src.experiments.context
  <config>`) for anything that trains or scores. The sweep runner's sbatch mode always
  requests a GPU.
* **`sbatch_*.sh`** for the dataset builds. Those are CPU-bound (dominated by HF
  `datasets.map` tokenizer plumbing, *not* the eigendecomposition — the magnetic decomposition
  is 2.8 ms/graph at N=128), so they request no GPU on purpose, which the sweep runner cannot
  express. `sbatch_mainsweep_data.sh` additionally chains 16 shard jobs into a merge with
  `--dependency=afterok`, which the sweep runner has no way to express at all.

> **Never run a build on the login node**, and do not follow sbatch's hint to use the `amd`
> partition for GPU-less jobs — `amd` is the MI210/ROCm node and cannot start this CUDA
> container (dead in 27 s). Builds are idempotent: a split already on disk is skipped, so an
> interrupted build resumes.

### 2.1 Reproducing each result

| result | 1. build data | 2. train | 3. score |
|---|---|---|---|
| §3.1 star grid | `001_data_prep.jsonc` (config) | `003_train_16k.jsonc` | `004_grid.jsonc` |
| §3.1 flat arms | — (reuses the above) | `007_flat_train.jsonc` | `006_flat_zeroshot.jsonc` |
| §3.2 chain, Phase A | `sbatch_chain_data.sh` | `009_chain_graph` / `010_chain_flat` | in-job |
| §3.2 learnability control | `sbatch_chain_small.sh` | `011_small_graph` / `012_small_flat` | in-job |
| §3.2 hard cell at 8k | `sbatch_chain_hard8k.sh` | `015_hard8k_graph` / `016_hard8k_flat` | in-job |
| §A.10 decoy / fan_out | `sbatch_chain_decoy.sh` | `017_decoy_graph` / `018_decoy_flat` | in-job |
| **§3.3 main sweep** | `sbatch_mainsweep_data.sh` | `021_mainsweep_graph` / `022_mainsweep_flat` | `023_mainsweep_grid` |

The full headline path, in order:

```bash
# 1. build the k-mixture dataset: 16 shards + dev/test grid, then a dependent merge.
#    ~3.9 GB on disk; the shards exist because the build's RAM peak does not fit a node (§A.12).
./src/experiments/context/sbatch_mainsweep_data.sh

# 2. GATE — audit every built (N, T, k) cell before spending GPU time (§A.10).
#    Must print GATE PASSED; it fails on an unbuilt cell rather than reporting a clean bill.
python3 -m src.experiments.context.analysis.audit_cells

# 3. calibrate cost before committing to a schedule (§A.12) — never skip this.
python3 -m sweep src.experiments.context src/experiments/context/configs/024_mainsweep_calibrate.jsonc

# 4. train both arms (3 seeds each; graph on 2x H100 under DDP, flat on A100_80GB).
python3 -m sweep src.experiments.context src/experiments/context/configs/021_mainsweep_graph.jsonc
python3 -m sweep src.experiments.context src/experiments/context/configs/022_mainsweep_flat.jsonc

# 5. score the graph arm's 64 conditions (the flat arm scores in-job) and plot.
#    Fill checkpoint_path in 023 from the train sweep's runs.jsonl first.
python3 -m sweep src.experiments.context src/experiments/context/configs/023_mainsweep_grid.jsonc
python3 -m src.experiments.context.analysis.plot \
    src/experiments/context/results/mainsweep_grid/grid.jsonl
```

`plot.py` writes one heatmap per k (`fig_context_grid_k{1..4}.png`) plus a per-condition
table — never one figure pooled over k, which would average a condition the graph arm solves
perfectly with one it does not.

Tests: `pytest tests/experiments/test_context_flags.py tests/experiments/context`.

---

## 3. Results

### 3.1 Single-needle retrieval is at ceiling everywhere (2026-07-30)

The original star grid — QUESTION names the gold node, one KV sentence per node — is flat at
ceiling in all 25 cells, including 4× length extrapolation past the 16,384-token training cap.
Mean EM 0.9993 over 75 records (25 cells × 3 seeds × 200 items), min 0.9950.

Adding a flat arm made the null sharper rather than weaker. By packed length, `code_acc`:

| L | 245 | 4,792 | 8,829 | 16,882 | 33,023 | 65,283 |
|---|---|---|---|---|---|---|
| flat 0-shot | 0.995 | 0.905 | 0.840 | 0.835 | 0.780 | **0.560** |
| flat trained | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | **0.995** |
| graph trained | 1.000 | 0.998 | 1.000 | 0.997 | 0.998 | **1.000** |

* The **pretrained** model does decay with real context length (0.995 → 0.560). That is the
  only genuine length effect this experiment ever produced — and **one epoch of LoRA erases
  it completely**, at 4× the training cap. The decay measures "has not been trained on this
  task", not "cannot address this much context".
* Graph and flat trained are indistinguishable (0.9993 vs 0.9996): the graph bias buys nothing
  when one literal string match suffices.
* **`distractor_rate` is 0.0000 in all 175 records.** No model ever retrieves another node's
  code. Dilution is not being measured, because the QUESTION names the gold id and it occurs
  exactly once — no amount of surrounding text makes a unique exact match harder to find.

This is also where the metric artifact surfaced: Stage 0 first read **EM 0.000 in all 25
cells** with `code_no_eos_rate` 0.887 — the pretrained model retrieves the right code and
then predicts whitespace instead of EOS. Teacher-forced EM demands `code + EOS`, a convention
no untrained model has been taught, so **EM cannot compare a trained arm against an untrained
one**. Hence `code_acc` (§A.9).

**Conclusion: scaling N or T further is pointless.** The answer has to be reachable only by
traversal, which is what the k-hop chain generator (`data.realize_chain`) implements.

### 3.2 k-hop chains, and the floor rule (2026-07-30)

Phase A ran the chain at N=64, T=128, one seed, 2 epochs × 2,000 graphs. Both arms floored at
every k ≥ 2; the graph arm floored even at k=1 (0.040 EM, roughly the rate of copying a random
node's code) while flat solved k=1 after 125 steps. The obvious reading was architectural, and
two real measurements seemed to support it:

```
SPD(prompt -> answer) = 32767     prompt-row histogram = {32767: 62}   # PROMPT has no edges
position_ids: min=0 max=127       start-node and answer-node positions IDENTICAL
```

The PROMPT node is isolated in both generators, so SPD bias — indexed by *(query, key)* — is
constant at the read-out position; and per-node RoPE reset removes the "find the earlier
mention, copy what follows" induction mechanism flat relies on.

**That conclusion was wrong.** A learnability control at a smaller cell (N=16/32, T=64,
8,000 graphs) refuted it within 250 steps: at epoch 0.25 the graph arm reached **0.990** at
k=1 and led flat at every k. Both measurements above stand; they simply do not imply what was
inferred from them. The floor was a budget/difficulty result.

This produced the **floor rule**, which the main sweep was pre-registered against (§A.11):
every floored run in this experiment stagnated for 3–5 evals and then jumped to ceiling in one
interval (`decoy_graph` k=2: 0.065 → 0.91). **A floored cell is untested, not failed**, until
the budget has been extended and it is still floored.

Do not "fix" the isolated PROMPT node by adding a `prompt → start` edge: that makes the answer
the unique node at distance `hops+1` from the prompt, which is exactly the SPD leak already
removed from the question node. **Any bias that makes the answer identifiable from the prompt
IS the leak.** The graph arm can only win here by composition.

### 3.3 Main sweep: the graph arm holds where flat text collapses (2026-07-31)

One GTLM and one flat-LLM, each trained **once** on a mixture over difficulty
(k ∈ {1,2,3,4}) and size (N ∈ {16,32,64,128}, T ∈ {64,128,256,512}) at `fan_out=2`, then
scored on every (N, T, k) condition individually — 64 conditions per arm. Axes and exclusions
are in §A.10.

Metric is `code_acc` at n = 200 per condition, teacher-forced; `em == code_acc` in all 64
graph records. 95% Wilson half-width ±6.9 pp at p=0.5, ±4.2 pp at p=0.9. **Reported
checkpoints are the median seed of 3 per arm**, selected on in-training `eval_em_accuracy`
with `eval_loss` as tiebreak. Both arms: 16,000 graphs × 2 epochs, `max_train_len` 16,384,
LoRA r=64.

`*` marks the three cells over the training cap — evaluated, never trained on. Packed length
per cell as **graph / flat** (identical across k); the two differ because the graph pack is
padded to a flex block multiple while the flat serialization carries `## Article {i}` headers
and is unpadded.

| N \ T | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| **16** | 1,024 / 1,015 | 1,920 / 1,909 | 3,712 / 3,702 | 7,296 / 7,287 |
| **32** | 2,048 / 2,132 | 3,968 / 4,053 | 7,808 / 7,894 | 15,488 / 15,575 |
| **64** | 4,096 / 4,366 | 8,064 / 8,338 | 16,000 / 16,278 | 31,872 / 32,150\* |
| **128** | 8,192 / 8,841 | 16,256 / 16,906 | 32,384 / 33,043\* | 64,640 / 65,301\* |

Per-condition `code_acc`:

| cell (N×T) | g k=1 | g k=2 | g k=3 | g k=4 | f k=1 | f k=2 | f k=3 | f k=4 |
|---|---|---|---|---|---|---|---|---|
| 16×64 | 1.000 | 1.000 | 0.990 | 0.925 | 0.985 | 0.955 | 0.590 | 0.275 |
| 16×128 | 1.000 | 1.000 | 0.985 | 0.910 | 0.985 | 0.940 | 0.625 | 0.310 |
| 16×256 | 1.000 | 1.000 | 0.985 | 0.935 | 0.990 | 0.950 | 0.565 | 0.350 |
| 16×512 | 1.000 | 1.000 | 0.985 | 0.890 | 0.950 | 0.955 | 0.580 | 0.310 |
| 32×64 | 1.000 | 1.000 | 1.000 | 0.970 | 0.985 | 0.940 | 0.575 | 0.170 |
| 32×128 | 1.000 | 1.000 | 0.995 | 0.950 | 0.990 | 0.930 | 0.620 | 0.200 |
| 32×256 | 1.000 | 1.000 | 1.000 | 0.905 | 0.985 | 0.920 | 0.520 | 0.190 |
| 32×512 | 1.000 | 1.000 | 1.000 | 0.940 | 0.985 | 0.910 | 0.530 | 0.200 |
| 64×64 | 1.000 | 1.000 | 0.995 | 0.955 | 0.980 | 0.905 | 0.505 | 0.120 |
| 64×128 | 1.000 | 1.000 | 0.995 | 0.955 | 0.970 | 0.905 | 0.540 | 0.160 |
| 64×256 | 0.995 | 1.000 | 1.000 | 0.950 | 0.980 | 0.875 | 0.525 | 0.120 |
| 64×512\* | 1.000 | 1.000 | 0.995 | 0.940 | 0.955 | 0.870 | 0.350 | 0.115 |
| 128×64 | 1.000 | 1.000 | 1.000 | 0.980 | 0.945 | 0.880 | 0.490 | 0.085 |
| 128×128 | 1.000 | 1.000 | 0.995 | 0.965 | 0.965 | 0.830 | 0.420 | 0.105 |
| 128×256\* | 1.000 | 1.000 | 1.000 | 0.975 | 0.965 | 0.840 | 0.430 | 0.065 |
| 128×512\* | 1.000 | 1.000 | 1.000 | 0.980 | 0.945 | 0.780 | 0.375 | 0.085 |

Marginals:

| | graph | flat | gap | | graph | flat | gap |
|---|---|---|---|---|---|---|---|
| k=1 | 0.9997 | 0.9725 | +0.027 | N=16 | 0.9753 | 0.7072 | +0.268 |
| k=2 | 1.0000 | 0.8991 | +0.101 | N=32 | 0.9850 | 0.6656 | +0.319 |
| k=3 | 0.9950 | 0.5150 | +0.480 | N=64 | 0.9862 | 0.6172 | +0.369 |
| k=4 | 0.9453 | 0.1787 | +0.767 | N=128 | 0.9934 | 0.5753 | +0.418 |
| T=64 | 0.9884 | 0.6491 | +0.339 | T=256 | 0.9841 | 0.6419 | +0.342 |
| T=128 | 0.9844 | 0.6559 | +0.328 | T=512 | 0.9831 | 0.6184 | +0.365 |
| | | | | **overall** | **0.9850** | **0.6413** | **+0.344** |

* **The arms move in opposite directions along N.** Flat decays with node count
  (0.707 → 0.575); the graph arm *improves* (0.975 → 0.993). Against the pre-registered
  read-out (§A.11) this is the top row — the structural channel resists context dilution — but
  the axis is **N, not T**: the gap is flat in T (+0.339 / +0.328 / +0.342 / +0.365).
* **Length extrapolation is where the gap is widest.** On the three over-cap cells (up to
  65,301 tokens, 4× the cap): graph **0.991** vs flat **0.565**, against 0.984 / 0.659
  in-distribution. The graph arm loses nothing outside its training length distribution; the
  flat arm loses ~10 pp.
* **The graph arm's weakest row is N=16, not N=128** — the opposite of a dilution failure, and
  it matches the design audit (§A.10): at N=16/k=4 a decoy shortcuts the chain to below
  distance k in 73% of graphs, so the structural channel is noisiest exactly there. Worst
  single cell 0.890 at N=16/T=512/k=4.
* **`distractor_rate` is 0.000 in all 64 graph records** (mean malformed 0.015). Even at k=4
  the graph arm never retrieves another node's code — it either traverses or produces nothing.

**Caveats that travel with the headline.**

* **The flat arm is not converged.** All three flat seeds were still improving monotonically
  at epoch 2.0 (dev EM 0.530 / 0.645 / 0.725) while the graph median hit 1.000 at epoch 1.5.
  So +0.344 measures **sample efficiency at a fixed 2-epoch budget**, not a capability
  ceiling. By this experiment's own floor rule (§3.2) a still-rising arm is untested, not
  beaten. Whether flat closes the gap with more budget is not answered here.
* **1 of 3 graph seeds did not converge** — seed 0 ended at dev EM 0.285, still rising. Seeds
  1 and 2 both reached 1.000. The median rule reports a converged seed and hides that spread.
* **Known arm asymmetries** (§A.11): bidirectional prefix vs strictly causal sdpa, per-node
  RoPE reset, `## Article {i}` headers costing the flat arm tokens, 45.95M vs 45.09M
  trainable parameters. The edge set and the verbalized-reference set are bijective at every
  k, so this is architecture, not leakage.

**Provenance.**

| | job(s) | GPU | wall |
|---|---|---|---|
| graph train, seeds 0/1/2 | 119851–119853 | 2× H100 each (DDP) | 12h16m |
| flat train, seeds 0/1/2 | 119849, 119850, 119854 | A100_80GB | ~11h40m |
| graph grid, 64 conditions | 120030–120041 | 6× B200, 3× H100, 3× A100 | 25m |
| flat grid, 64 conditions | in-job (`flat.py`) | A100_80GB | 2h16m |

`results/` is **not tracked in git** — the run records are large, machine-local and
regenerated by §2.1, so the tables above are the record of this experiment, not a summary of
files you will find in the repo. On the machine that ran it the numbers come from
`results/mainsweep_grid/grid.jsonl` (graph, 64 rows) and `results/flat_trained_grid.jsonl`
(flat, `sweep_id: mainsweep_flat`, 3 seeds × 64 rows). The graph grid is reproduced by
`configs/023_mainsweep_grid.jsonc` as a single job (~2.7 h on one B200); it was *run* as 12
cell/k shards whose union is that config — cost goes as ~L^1.7 and the 128×512 cell alone is
half the grid, so sharding by cell and k (`--only-cells` / `--only-hops`, neither in the cache
key) cuts 2.7 h to 25 min.

---

# Appendix A — design reference

The decisions the code depends on. Code comments cite these section numbers.

### A.1 Topology and node accounting

*N* is the **total** node count:

```
N = 1 QUESTION node    (star centre, bidirectional prefix, edges to all content nodes)
  + 1 PROMPT node      (target, isolated, causal, the only node carrying labels)
  + (N−2) content nodes
L(N, T) = (N−2)·T + |QUESTION| + |PROMPT|
```

QUESTION lives in the bidirectional prefix, so every content token can attend to it — that
question-conditioned encoding is what makes retrieval learnable at all. PROMPT has **no
edges** in both generators; see §3.2 for why that is deliberate and must not be "fixed".

Two generators, selected by `hops`:

* `realize` (k=0) — a **star**: `QUESTION → every content node`, no content–content edges,
  diameter 2 (hence `max_spd 8` is ample; nothing on a star exceeds 2).
* `realize_chain` (k≥1) — pointer and decoy edges between content nodes, where SPD is
  load-bearing. The QUESTION states k literally ("follow 2 'Continue at' references, ignoring
  any decoy references"), so k can vary per graph.

The two must not be mixed in one training set: different SPD distributions, and the k=0 needle
is 20 tokens against 29 at `fan_out=2`, so cells would not be length-matched.

**Keep `k_hop=0`.** At `k_hop=0` the mask is dense over the prefix regardless of topology, so
edges feed only the SPD/magnetic features. `k_hop=1` would make the mask genuinely sparse
(~97% token sparsity, 10–15× cheaper) but **changes what is measured** — the model would no
longer dilute over all N·T keys. It is the escape hatch if compute forces it, and must then be
reported as a different experiment.

### A.2 Node text: filler, ids, codes

* **Filler** — `Salesforce/wikitext`, `wikitext-103-raw-v1`, snapshotted under `raw_data/` and
  read locally afterwards so rebuilds are reproducible offline. Articles are concatenated into
  one token stream and sliced per node; slices containing `access code` are discarded (a
  corpus-native false needle would silently poison the distractor set).
* **Node ids** — sampled without replacement from a pool of ≥ 4096 (`NODE-{0000..4095}`), not
  `range(N)`, so the gold id is not predictable from N and no id→position prior is learnable.
* **Codes** — must have **fixed token length** or EM is not comparable across graphs. The code
  vocabulary is built at data-prep time by tokenizing candidates and keeping only those that
  tokenize to exactly `code_len` tokens (3; pool asserted ≥ 4096). Llama-3 merges digit runs in
  groups of up to three, so `"482913"` and `"482 913"` differ in token count — the filter turns
  that into a build-time assertion instead of a silent metric bug.
* **KV sentence** — `"The access code for {node_id} is {code}."`, identical in gold and
  distractor nodes; every content node gets exactly one, so their count is not a cue.

Note the KV sentence's tokens *in context* differ from its tokens *in isolation* (BPE merges
the preceding space into the first token), so the build asserts **string** containment, not
token-subsequence containment.

### A.3 Needle placement (T axis) and node subsets (N axis)

"Truncate texts for smaller T" cannot be taken literally: a needle at a uniform offset in 512
tokens survives truncation to 32 with probability ~6%, and pinning it inside the first 32
makes it node-initial at every larger T — a within-node position confound that gets *easier*
exactly where degradation is expected.

Resolution: for each graph *g* and each T, with `rng = Random(hash(base_seed, g, T))`, sample
`offset ~ U{0, …, T − needle − SUFFIX_SLACK}` and splice the KV sentence in at `offset`.
Everything else — node subset, gold node, gold id, gold code, distractor ids/codes, filler
stream — is **identical across the T axis**, so cells stay paired for a within-graph
comparison; only the within-node offset is re-randomized. Strict byte-nesting across T is given
up on purpose.

Along N: one random permutation π of the content-node slots is drawn per graph at build time,
and cell N uses `{gold} ∪ {π[0] … π[N−4]}` — nested by construction, gold always present. The
gold node's *index* in the packed order is re-randomized per cell.

For the chain, `bp.slot_order[:hops + 1]` means **the same blueprint at k=1..4 yields nested
chains** — same start node, progressively deeper answer — so the k-curve is paired at zero
cost. Do not let a split refactor break that.

### A.4 Build invariants

`TextGraphDataset.tokenize()` is the only public path to `input_ids` and it tokenizes the
`text` column with truncation, so exact-T nodes are produced by a verify-and-adjust loop on
the public path rather than by writing `input_ids` directly (which would need a setter on the
shared class). `fit_node_text` is one-directional — shrink to ≤ T, then close the gap with
verified single-token pad words — because adjusting by the token shortfall oscillates.

Then hard-assert, at build time, on **every** graph:

1. every content node tokenizes to exactly T tokens;
2. the gold code appears exactly once in the gold node and in **no** other node (a distractor
   re-drawing the gold code makes the item unanswerable);
3. the gold id appears in the QUESTION node and in exactly one content node;
4. `len(labels != -100) == code_len + 1` (code + EOS).

These four are `tests/experiments/context/test_data.py`. They are what stands between this
experiment and a heatmap of a build bug.

### A.5 Cache key and feature storage

Features depend on the node subset, so every test cell is its own build:

```
processed_datasets/<data_config_key>/
    train.gtds  dev.gtds          # the mixture, capped at max_train_len
    test/n{N}_t{T}[_k{k}].gtds    # one split per evaluation condition
```

`data_config_key` includes `model_name`, the grid axes, `n_train`, `data_seed`, `code_len`,
`magnetic_q`, `magnetic_m`, `max_train_len` and a `DATA_FORMAT_VERSION` constant, so a semantic
change to the builder cannot silently reuse a stale cache. Knobs added later default to values
that leave the key unchanged, so existing builds are never orphaned.

**RRWP is deliberately off.** It is `(N,N,K)` fp32 = 1 MB/graph at K=16 — a 5× storage
multiplier — and on a star its walk profile is near-degenerate. This deviates from the paper's
SPD+RRWP+MagLap suite and must be stated; `039_rrwp_webqsp` independently established that
adding RRWP does not move results.

### A.6 Training mixture and the length cap

`sample_cell` draws N uniform, then T uniform among the T admissible at that N under the
length cap — keeping N balanced (the axis of interest) and letting T fill the remaining
budget. `n_train` is the total over that mixture. `sample_hops` mirrors it for k.

At `max_train_len = 16,384` the star grid admits 22 of 25 cells; the chain grid admits **13 of
16**, leaving (64,512), (128,256) and (128,512) over-cap. Over-cap cells are **evaluated but
never trained on**, so they measure length extrapolation — which means a drop there is
ambiguous between dilution and extrapolation unless a higher-cap run is available for
comparison (`005_train_32k_diag.jsonc` is that diagnostic for the star grid: identical except
`max_train_len = 32768`, which pulls two of the three cells back in-distribution).

### A.7 Cell-homogeneous batching and flex buckets

**Length-homogeneous batching is required, not optional.** Batch a graph at L=512 with one at
L=16,384 and the collator pads both to 16,384; the flex kernel then does 32× the necessary
work on the short one, and the windowed loss (§A.8) loses its window.

**HF `group_by_length` cannot do this** (verified on transformers 4.50.3):
`Trainer._get_train_sampler` only reads a `length` column when `train_dataset` is a
`datasets.Dataset`, and `TextGraphDataset` *wraps* one rather than being one — so
`LengthGroupedSampler` falls back to `[len(f["input_ids"]) for f in dataset]`, and our
`input_ids` is a list of per-node lists. That returns the **node count, not the token count**:
it would group by N and silently mix T.

Instead each graph stores its cell `(N, T)` in its graph attrs, and `CellGroupedSampler` emits
batches drawn from a single cell, wired via `_get_train_sampler`. `GraphTrainerV2` overrides
neither that nor `get_train_dataloader`, so there is no conflict.

`pad_to_block=True` is mandatory for flex (~14× cliff at unaligned lengths). The default bucket
ladder would give up to 33 distinct compiled shapes at ~320 s of autotune each; passing an
explicit `len_buckets` list of just the cell lengths the sampler can produce — each rounded up
with `flex_kernel.align_len(L, 128)` — cuts that to the number of cells at no extra padding,
because batches are already cell-homogeneous. The collator *raises* if a bucket is not a
`block_size` multiple (the 128×512 cell's L = 64,542 must be listed as 64,640). Use
`--compile-mode default` (~16 s/shape) for smoke runs.

Measured padding waste across the 13 chain train cells: a constant **93 tokens** per cell, 1.5%
cost-weighted. Not worth tuning.

### A.8 The windowed loss

The supervised span is ~4 tokens at the end of a sequence up to 65,536 long, which breaks the
ordinary path in two places:

* `causal_lm.forward` runs `lm_head` over all positions by default. At L=16,384 that is 4.2 GB
  of bf16 logits (16.5 GB at 64,512), spent on positions that are all `-100`. `logits_to_keep`
  accepts an int or index tensor, but `loss_function` is then handed the **full** labels
  against sliced logits and mis-shapes — so the slice cannot be passed through HF's Trainer.
* HF's evaluation loop calls the model with `labels` and no `logits_to_keep`, so dev eval
  reproduces the same blowup even with `preprocess_logits_for_metrics` set — the reduction runs
  *after* the full logits exist.

`ContextGraphTrainer(GraphTrainerV2)` fixes both inside this package. `compute_loss` finds
`start = min_i(first index where labels[i] != -100)` over the batch, calls the model with
`logits_to_keep = L_pad − start` and **no** labels, and computes the shifted CE itself. With
cell-homogeneous batching the window is `bucket padding + code_len + 1` — ~128 tokens instead
of 16,384, i.e. ~33 MB instead of 4.2 GB. `prediction_step` applies the same window and returns
**sliced logits and sliced labels together** — they must stay aligned or `make_compute_metrics`
compares different spans. With that, `Trainer.evaluate`, best-checkpoint selection and
`GraphTrainerV2._load_best_model` all work unmodified.

`tests/experiments/context/test_windowed_loss.py` pins the sliced path against the full-logits
path. `grid_eval` does not go through the Trainer at all.

### A.9 Metric

**Teacher-forced greedy EM.** One forward per graph on the flex path with the answer tokens
present in the PROMPT node: `EM(g) = 1` iff `argmax logits[t] == labels[t+1]` at every
supervised position (the `code_len` code tokens + EOS).

Free generation is not an option: `generative_eval` forces `graph_attn_impl="eager"` for the
whole generation loop, and eager materializes a token-level `(B, H, L, L)` bias — **274 TB for
one layer** at L = 65,536. The top of the grid is not slow on that path, it is unrepresentable.
Teacher-forced EM upper-bounds free-running greedy EM (no error propagation across ≤4 answer
tokens); name it "teacher-forced EM" in captions, not plain "EM".

**`code_acc`** — the first `code_len` tokens, EOS ignored — is the arm-independent measure, and
the only one valid across trained and untrained arms (§3.1). For a trained arm at ceiling
`em == code_acc`.

**Failure classification**, recorded per cell alongside accuracy: a **distractor hit** (the
prediction equals some *other* node's code) is a selection failure — attention landed
somewhere; **malformed** (neither gold nor any distractor) is a representation failure. The
ratio between them as a function of (N, T, k) is the mechanistic story behind the surface.

**Statistics.** n = 200 per cell gives a 95% Wilson half-width of ±6.9 pp at p = 0.5 and
±4.2 pp at p = 0.9. That resolves a 15–30 pp transition and does **not** resolve 5 pp
cell-to-cell differences — read the surface for the contour, never for a ranking between
adjacent cells. Because cells share base graphs (§A.3), *paired* differences along a row or
column are tighter than the marginal CIs suggest; use McNemar if a specific adjacent-cell
claim is needed.

### A.10 Main-sweep axes and exclusions

| axis | values | excluded | why |
|---|---|---|---|
| **k** | 1, 2, 3, 4 | 0 | different builder (§A.1); already run as the pure-retrieval baseline |
| **N** | 16, 32, 64, 128 | 8 | 6 content nodes, of which a k=4 chain occupies 5 — one distractor, no dynamic range |
| **T** | 64, 128, 256, 512 | 32 | the needle does not fit |
| **fan_out** | 2 | 1 | fan_out=1 has an SPD shortcut; kept as a reported ablation |

**Why T=32 is excluded.** T is tokens per content node, exact — so T is the *dilution* axis.
`max_needle_offset = T − needle − SUFFIX_SLACK` is the range offsets randomize over; at
fan_out=2 the needle is 29 tokens, so T=32 collapses that range to 0: three tokens of filler and
every needle pinned to position 0. That is not a low-dilution cell, it is "node = needle", and
it removes the position randomization every other T has. `validate()`'s guard is fan_out-aware
for this reason (the floor is `needle(fan_out) + SUFFIX_SLACK + 1`).

**Why fan_out=2.** At fan_out=1 every content node has exactly one successor, so the content
subgraph is *functional* and the answer is the unique node at SPD == k. The graph arm never
traverses anything — it locates the start by name and reads the answer off the distance bias,
which is O(1) in k. fan_out=2 gives each node one real pointer and one explicitly-labelled
decoy, both entering the DiGraph identically, so ~2^k nodes sit at distance k: topology can
**prune** the candidate set but cannot identify the answer. The disambiguating signal moves
into the text, where both arms can read it.

**Small N does not leak more than large N.** Measured against an oracle that sees every
distance and no text, playing the best distance-only strategy:

| cell | P(answer ∈ shell_k) | mean \|shell_k\| | SPD-only acc | chance | leak |
|---|---|---|---|---|---|
| N=16 k=2 | 92.5% | 3.13 | 31.0% | 7.1% | +23.9 |
| N=32 k=2 | 98.0% | 3.60 | 28.0% | 3.3% | +24.6 |
| N=16 k=3 | 59.0% | 3.20 | 19.0% | 7.1% | +11.8 |
| N=32 k=3 | 81.5% | 5.34 | 15.6% | 3.3% | +12.2 |
| N=16 k=4 | 27.0% | 1.79 | 16.2% | 7.1% | +9.1 |
| N=32 k=4 | 52.5% | 5.63 | 9.9% | 3.3% | +6.5 |

Excess over chance is comparable at both cells. What small N costs is *signal*: at N=16/k=4 a
decoy shortcuts the answer to below distance k in 73% of graphs, so the structural channel is
mostly noise there — a fact to report about the cell, not a reason to exclude it. **Bound worth
carrying into the paper:** at N=32/k=3 a distance-only oracle scores 15.6% while the graph arm
scored 0.990, so the win is not topology identifying the answer.

`analysis/audit_cells.py` computes all of the above per built cell — shell statistics, SPD-only
oracle accuracy, out-degree histogram, and the edge ↔ verbalized-reference bijection. **Run it
as a gate before spending GPU time**, and publish the table beside the results. The mislabeled
statistic that this catches (`len(shell_k) == 1` reported as "answer alone at distance k") once
rejected N=16 wrongly.

### A.11 Pre-registered read-out

Written before the main sweep ran, so the result could not be reinterpreted after the fact:

| outcome | reading |
|---|---|
| graph > flat, gap **grows** with length at fixed k | the structural channel resists context dilution — the paper's claim |
| graph > flat, gap **flat** in length | graph helps, but not because of context exhaustion; reframe |
| graph > flat only at small k | traversal depth, not length, is the binding constraint |
| graph ≈ flat everywhere | the decoy win was cell-specific; report as such |
| either arm at floor across a whole row | **budget, not architecture** — extend before theorising (the floor rule, §3.2) |

Report `code_acc` from the final `grid_eval`, not `eval_em_accuracy` (only the latter is logged
during training). Wilson intervals on every cell.

**Arm asymmetries to state plainly** — architecture, not leakage, since the edge set and the
verbalized-reference set are bijective at every k: the graph arm gets bidirectional prefix
attention while the flat arm is strictly causal sdpa; per-node RoPE reset; `## Article {i}`
headers cost the flat arm tokens but give it segmentation; 45.95M vs 45.09M trainable
parameters (1.9%). Both arms select checkpoints on `eval_em_accuracy`.

### A.12 Cost, calibration, and the ceiling check

**Do not plan a budget from a table — measure it.** `calibrate.py` runs the real model at the
real cell shapes with the knobs the run will use, on the GPU the run will request, and reports
wall-clock s/it and peak GB per cell. Everything downstream (`max_train_len`, `n_train`,
epochs, sbatch `time`/`mem`) depends on its output. The reasons this is not optional:

* Published flex tables predate bias-checkpointing and decoder gradient checkpointing, and are
  per-GPU — `flex_attn/README.md` is explicit that mixed-directory ratios are meaningless.
* **Cost is strongly superlinear in length.** Fitting the available anchors gives
  `cost ∝ L^1.4–2.1`. And `E[L^a] > (E[L])^a`, so costing a mixture from its mean length
  understates it by 30–55% — the N=128 cells are a quarter of the draws and cost ~2× the
  average graph.
* Compile mode confounds cross-run comparison: `max-autotune-no-cudagraphs` is 1.2–1.5× faster
  per step than `default`, so anchors that differ in `compile_mode` cannot be compared
  directly.
* tqdm's displayed rate is smoothed and reads low — derive s/it from wall clock.

**Memory:** the train split is materialised whole in RAM during `datasets.map`, and peak scales
with graphs × tokens. The N=64 cell peaked at 61.7 GB at n_train=2,000, and a 96 G request
OOM-killed the 8k builds. Large builds must be **sharded** (`--train-shards` / `--train-shard`,
then `--mode data_merge`); shard *i* must pass a distinct `id_offset` or every shard draws the
same graphs. The merge is a separate, unmeasured peak — measure it on 2 shards before assuming
an N-way merge fits. On-disk size is ~20.4 KB per 1k tokens per graph, so the final artifact is
small; this is purely a build-time transient.

**Check the largest cell first.** After training, score the biggest cell before running the
full grid: if it is at ceiling there is no contour, and the fix is a harder task rather than a
bigger grid (§3.1). `grid.py` and `flat.py` therefore score largest-first, so a ceiling shows up
in the first few records instead of after hours.
