# `bias_sharing` — how many layers should share one magnetic bias?

**Status:** complete. Implementation and verification 2026-08-02; all 90 training
runs landed 2026-08-03 (§4). D2 verdict: **`G=4`** (§4.4). A follow-up speed
benchmark at 1024–4096 nodes, the regime none of the three sweeps reaches, is in
§6.

This package answers **D2** of `src/generalist/PLAN.md` §3.1: the magnetic bias
sharing granularity, which is a permanent compute-vs-accuracy trade on a trunk run
that lasts months and cannot be changed mid-trunk.

---

## 0. What this package is

**A configs-and-results package, not an experiment module.** There is no
`train.py`, no `config.py`, no data pipeline here. Every sweep invokes an
*existing* experiment exactly the way that experiment is normally invoked, and
only points at a config that lives here:

```bash
python3 -m sweep src.experiments.graphqa src/experiments/bias_sharing/configs/001_graphqa_g_sweep.jsonc
python3 -m sweep src.experiments.kgqa    src/experiments/bias_sharing/configs/002_webqsp_g_sweep.jsonc
./src/experiments/bias_sharing/sbatch_context4k_data.sh          # once, first
python3 -m sweep src.experiments.context src/experiments/bias_sharing/configs/003_context4k_g_sweep.jsonc
```

Results land in `results/<sweep_name>/runs.jsonl`; aggregate with
`python3 -m sweep.report src/experiments/bias_sharing/results/<sweep_name>`, or
with `python3 -m src.experiments.bias_sharing.analyse` for the cross-arm tables
of §4 (which also recovers step time from the slurm logs).

The one piece of *code* here is `bench/` (§6), a speed-only benchmark on
synthetic graphs — it exists because all three sweeps sit below 512 nodes.

Each config is its source experiment's **headline recipe verbatim**, with one
axis added. Anything else would make the comparison a recipe comparison.

---

## 1. The knob

`magnetic_groups: G` (`src/models/bias.py`). The magnetic bias is instantiated
**G times instead of once per layer**; layer `l` is served by group
`l * G // num_layers`. Llama-3.2-1B has 16 layers, so `G ∈ {1,2,4,8,16}` are
exactly the divisors, and the endpoints coincide with the two flags that already
existed:

| G | equivalent to | bias instances |
|---:|---|---:|
| 16 | `magnetic` (per-layer) | 16 |
| 8, 4, 2 | *new* | 8, 4, 2 |
| 1 | `magnetic_shared` | 1 |

`magnetic_groups: 0` selects the **legacy per-layer code path** and is swept as
the baseline arm. It and `G=16` are numerically identical by construction; they
are both run so the campaign also measures old-vs-new step time on one card.

**`magnetic_groups` is a model-side knob only.** It does not enter feature
generation or any dataset cache key — verified, all six arms of every sweep
resolve to one `data_config_key()` — so a G sweep trains on byte-identical data.
`magnetic: true` stays set in every arm because that is what gates the collator's
eigenvector emission; `magnetic_groups` supersedes it when building the model.

### Why it is not just "instantiate G modules"

HF's non-reentrant gradient checkpointing requires a region's recompute to save
the same tensors its forward did. The obvious schedule — compute the bias at the
group's first layer, recompute it at its last during backward — violates that,
because forward and backward disagree about *which region owns the computation*.
It fails loudly (`CheckpointError: A different number of tensors was saved`)
under a `no_grad` recompute and differently (`Recomputed values ... have
different metadata`) under a grad-enabled one.

What works is an **owner/follower** rule:

* each group's **owner** is its lowest layer, and it computes the bias *with
  grad* — in the forward, and automatically again in its own region's recompute,
  so its saved-tensor frame matches by construction;
* every **follower** only ever reads a value, so no bias intermediates enter its
  frame in either direction;
* backward reaches followers first, so the first one rematerialises the value
  under `no_grad` (adding nothing to its frame) and passes it on as a **leaf that
  requires grad**, so downstream ops save exactly what they saved in the forward.

The leaf's `.grad` is discarded — gradient reaches the parameters through the
graph node the owner built in the forward, which all `k` consumers are attached
to. Each group's tensor is released once its last consumer takes it, so peak
residency is one `(B,H,N,N)` tensor rather than G.

Cost per group per step: **3 bias computes** (1 forward + 1 follower remat +
1 owner recompute) — dropping to **2** when a group holds a single layer, which
has no follower to rematerialize, so `G=L` costs `2L`, not `3L`.

The per-layer path costs **`3L`**, not the `2L` this section claimed until §6
measured it. `dispatch.compute_node_bias` wraps the bias in its own
`torch.utils.checkpoint.checkpoint` (`src/models/dispatch.py:241-245`) *inside*
the decoder layer's checkpoint, so with `gradient_checkpointing=True` each layer
evaluates the bias three times: once in the forward, once when the layer is
recomputed, and once more when the inner checkpoint is recomputed for its own
backward. §6.4 fits the per-compute cost on `G=0` vs `G=1` and predicts the
held-out `G∈{2,4,8}` arms to within 0.4% under `3L`, but misses by 6–15% under
`2L`.

---

## 2. Verification

`tests/models/test_bias_sharing.py` — 51 tests, all passing (47 on CPU, the 4
flex ones on an H100; job `120943`).

The core check compares against a **redundant-compute reference**: the same
parameters, recomputed inside every layer, which is plain autograd with nothing
shared and therefore nothing to get wrong.

| what | how |
|---|---|
| loss + **every** parameter gradient match the reference | float64, gradient checkpointing **on and off**, `(L,G)` ∈ {(4,1),(4,2),(4,4),(8,2),(8,4),(6,2),(6,3)} |
| same, on **flex** | float32 (Inductor cannot lower FlexAttention in float64), ckpt on and off, on GPU |
| every grouped-bias parameter gets a **non-zero, finite** gradient | all `G × 8` tensors — a silent zero would make the sweep measure nothing |
| gradients are **not double-counted** | per-group grad-norm ratio vs reference within 1e-9 |
| gradient reaches the **token embeddings** | end-to-end flow |
| accumulation over two backwards | release bookkeeping survives |
| `G = L` ≡ per-layer `magnetic`, `G = 1` ≡ `magnetic_shared` | weights synced, losses and grads compared leaf by leaf |
| group tensors are **really released** after the forward | weakref death |
| eval / generation parity | eval logits == train-mode logits; decode runs |

Two harness traps worth remembering, both of which produced *wrong greens or
wrong reds* before being fixed: module construction order shifts the RNG stream,
so two models built from the same seed with different bias configs have
**different backbones** (`_identical_pair` now asserts equality); and flex
autotunes to an **empty choice list** at head_dim 16 / block 16, so the flex
tests use the same shapes as `tests/models/test_flex_attention.py`.

Full repo suite: 710 passed, 23 skipped.

---

## 3. The sweeps

Three sub-experiments, chosen to span the regimes D2 has to hold across.

| config | experiment | recipe cloned from | GPU | runs | ~each |
|---|---|---|---|---:|---:|
| `001_graphqa_g_sweep` | `src.experiments.graphqa` | `006_question_node_ablation` | A100 | 54 | 10 min |
| `002_webqsp_g_sweep` | `src.experiments.kgqa` | `029_question_node_webqsp` (`isolated`) | B200 | 18 | 4 h |
| `003_context4k_g_sweep` | `src.experiments.context` | `021_mainsweep_graph` @ cap 4096 | H100 | 18 | 3–5 h |

6 arms (`G ∈ {0,1,2,4,8,16}`) × 3 seeds each.

**One GPU type *within* each sub-experiment, deliberately different *between*
them.** Step time is half of what D2 measures and is only comparable on identical
hardware — but that constraint binds per sub-experiment, and nothing is gained by
queueing all three behind the same 8-GPU node. B200 is also the hardware `029`
itself ran on, so WebQSP's step times stay comparable to that recipe's history.
**Do not compare `s/it` across sub-experiments.**

**GraphQA tasks** — one per difficulty tier of the published table
(`CLAUDE_CONTEXT.md` §3.1, GTLM standard): `node_degree` 99.7 (easy),
`shortest_path` 90.1 (medium), `edge_count` 56.9 (hard). `triangle_counting` and
`connected_nodes` are the tasks one would otherwise reach for and are **not
used**: both are PLAN §3.3 permanent hold-outs, and selecting an architecture
constant on them would spend the only clean read on generality. `edge_count`
scores below 56.9 here because `question_node: isolated` regresses it (a known
magnetic-shortcut trade, not a bug) — harmless when every arm shares the recipe,
and it leaves headroom.

**WebQSP** is the load-bearing one: real data, a published headline (74.07 F1 /
83.60 Hit, seed 2 of `029`), and the **flex** backend. `graph_construction:
triplet` scores higher but is excluded from GTLM claims by decision, so
`029/isolated` is the right reference.

**Context at 4k** is where the bias is largest relative to everything else — many
nodes carrying little text, PLAN §D5's bad corner. Two deliberate changes from
`021`: the cap (16384 → 4096, shrinking the training distribution from 13 cells
to 6 at 960–4032 packed tokens, the other 10 still scored as length
extrapolation) and single-GPU instead of 2×DDP. The effective batch and step
count are preserved exactly (`1 × 4 × 2 ranks` → `1 × 8 × 1 rank` = 8; 4000
steps); DDP is dropped because all-reduce sits inside the wall-clock this
experiment exists to measure.

### Reading the results

Accuracy is the primary axis, step time the secondary. Two sanity checks are
built into the design and should be verified before anything else is believed:

1. **`G=0` and `G=16` must agree** on accuracy within seed noise — they are the
   same computation via different code paths.
2. **`G=0` and `G=16` step times should differ only slightly** — the new path
   pays one extra bias compute per group and at `G=16` every group has one layer.

---

## 4. Results

**Complete.** All 90 runs landed (54 + 18 + 18) on 2026-08-03; every sweep has
exactly 6 arms × its seed count, no duplicate `sweep_run`, and `003`'s 18 runs
resolve to a **single `data_config_key`** — the byte-identical-data claim of §1 is
confirmed empirically, not just argued.

Regenerate every number below with:

```bash
python3 -m src.experiments.bias_sharing.analyse
```

Accuracy is read from `runs.jsonl`. Step time is **not recorded** for `002`/`003`
(only the GraphQA runner writes a timing column), so it is recovered from the
tqdm stamps in the slurm logs: one rate per evaluation-free window, median across
windows. See `analyse.step_time_from_log` for why consecutive samples cannot be
used (1-second stamp resolution quantizes a sub-second step to 0 or 1).

### 4.1 The two sanity checks

| sweep | metric | `G=0` | `G=16` | |
|---|---|---:|---:|---|
| 002 WebQSP | test F1 | 72.95 ± 0.60 | 73.37 ± 0.23 | ✅ p = 0.20 |
| 003 context4k | dev EM | 98.67 | 99.83 | ✅ (both saturated) |
| 001 node_degree | test acc | 99.20 ± 0.20 | 98.80 ± 0.20 | ✅ |
| 001 shortest_path | test acc | 97.73 ± 0.76 | 97.67 ± 0.23 | ✅ |
| 001 edge_count | test acc | 43.93 ± 1.10 | 46.27 ± 1.14 | ⚠️ see below |

**Check 1 passes everywhere except `edge_count`**, where `G=0` is the lowest of
all six arms with no seed overlap (43.4 / 43.2 / 45.2 against a pooled `G≥1` mean
of 46.64; permutation p = 0.0025). It is **test-only**: the same comparison on
`best_val_accuracy` — the metric that actually selected these checkpoints — is
46.13 vs 46.61, p = 0.32. Two code paths that computed different things would
diverge on both splits, so this is not evidence of a numerical discrepancy;
§2's equivalence tests pin `G=L ≡ magnetic` leaf by leaf. The likely cause is
§2's other harness trap: **construction order shifts the RNG stream, so `G=0` and
`G=16` at the same seed do not share a backbone**. Exact agreement was never
available — only agreement in distribution, and 3 seeds on the noisiest task is a
weak instrument. Treat it as passed-with-a-flag.

**Check 2 passes, in the opposite direction to what §1 originally predicted.**
The old arithmetic (3 computes per group vs `2L` for the legacy path) said `G=16`
should be *slower* — 48 computes against 32. It is consistently **faster**: by
1.3% on context4k, 1.8% on WebQSP, 8–14% on GraphQA.

§6.4 resolved this, and the correction runs the other way on both sides of the
comparison. The legacy path costs `3L = 48`, not 32, because of the bias
checkpoint nested inside the layer checkpoint; and `G=L` costs `2L = 32`, not 48,
because a one-layer group has no follower to rematerialize. So `G=16` really does
do a third less bias work than `G=0`, and being faster is the expected result.
Fitting the per-compute cost on the synthetic grid reproduces `G=16` to within
2% at N ≥ 1024 under 32 computes, and misses by 13–17% under 48.

GraphQA is a separate case: it is the one experiment with
`gradient_checkpointing=False` (`src/experiments/graphqa/config.py:214`), so the
outer checkpoint that makes the legacy path `3L` is absent there. Its much larger
8–14% margin is therefore *not* explained by this mechanism and remains
unattributed — it is also the noisiest of the three tasks (§4.1).

### 4.2 Step time

**Do not compare across rows** — three different GPUs, three different recipes.

| sweep | unit | G=0 | G=1 | G=2 | G=4 | G=8 | G=16 |
|---|---|---:|---:|---:|---:|---:|---:|
| 002 WebQSP (B200) | s/it | 1.583 | 1.369 | **1.346** | 1.398 | 1.459 | 1.554 |
| 003 context4k (H100) | s/it | 3.223 | **3.062** | 3.080 | 3.093 | 3.119 | 3.183 |
| 001 GraphQA (A100) | steps/s | 0.480 | **0.673** | 0.662 | 0.639 | 0.595 | 0.536 |

Best saving against the legacy baseline: **GraphQA 1.40×, WebQSP 1.18×,
context4k 1.05×** (GraphQA's 20-epoch wall-clock falls from 1291 s to 922 s).
Within the grouped path the ordering is monotone in `G` on every sweep, as
expected: fewer groups, fewer bias computes.

**The 4k corner behaves opposite to the design prediction, and the config
explains why.** §3 picked context-at-4k as the place "where the bias is largest
relative to everything else", quoting PLAN §D5's 7.9× at *2048 nodes × 2 tokens*.
But `003`'s cells are `n16-32-64-128` × `t64-128-256-512` — **at most 128 nodes,
carrying up to 512 tokens each**. The bias is `(B,H,N,N)` in **nodes**, so at
N ≤ 128 it is a rounding error next to a 4096-token attention and MLP stack.
`003` measures the *long-sequence, few-nodes* corner, which is the opposite of
D5's bad corner; it is a useful accuracy control but it does **not** price the
regime the trunk cares about. GraphQA, with the shortest sequences, shows the
largest saving for the same reason. The regime that D5 actually names is measured
separately in §6.

### 4.3 Accuracy

**WebQSP (the load-bearing sweep).**

| G | test F1 | Hits@1 | Hit* |
|---:|---:|---:|---:|
| 0 | 72.95 ± 0.60 | 77.85 ± 0.09 | 83.27 ± 0.09 |
| 1 | **71.52 ± 1.19** | 77.35 ± 0.64 | 82.27 ± 0.23 |
| 2 | 72.55 ± 0.23 | 77.21 ± 0.18 | 82.35 ± 0.57 |
| 4 | 72.85 ± 1.03 | 77.95 ± 0.80 | 82.39 ± 0.77 |
| 8 | 72.58 ± 0.99 | 78.05 ± 0.40 | 82.84 ± 0.36 |
| 16 | **73.37 ± 0.23** | 78.46 ± 0.34 | 83.46 ± 0.41 |

`G=1` (≡ `magnetic_shared`) is a genuine cliff: **−1.85 F1** against `G=16`, and
−1.32 against pooled `G≥2` (p = 0.03). Above it the surface is shallow but not
flat — `G=2` sits 0.82 F1 below `G=16` with all three seeds separated, and every
`G < 16` arm is below `G=16` on all three metrics. With 3 vs 3 seeds a
permutation test bottoms out at p = 0.05, so *perfect separation* is the
strongest available evidence and 0.05 should be read as that, not as marginal.

**GraphQA.** No effect on the easy or medium tier (`node_degree` 98.8–99.3,
`shortest_path` 96.9–98.1 across all six arms — every arm inside every other
arm's seed spread). On the hard tier `edge_count` spans 43.9–47.8 with no ordered
trend in `G`; the only structure is the `G=0`/`G≥1` test-split offset discussed in
§4.1. GraphQA does not discriminate between sharing granularities.

**Context at 4k.** Saturated: five of six arms are at 98.7–100.0 dev EM. The
exception is **one collapsed run**, `G=2` seed 1, which never escaped —
eval EM went 0.08 → 0.34 and then sat at 0.33–0.375 for both epochs
(eval_loss stuck at 0.61, `malformed_rate` 0.625), where healthy runs are at
0.975 by the fourth evaluation. The trajectory was already flat long before any
checkpoint reload, so this is an optimization failure on that seed, not a reload
artifact. It drags the `G=2` cell from ~99 to 78.5; read that cell as
{100.0, 37.5, 98.0}, **not** as a `G=2` effect. Nothing else in the sweep
supports a `G=2` pathology — WebQSP's `G=2` arm is its tightest (sd 0.23).

### 4.4 D2 verdict

**`G=4`.** It is statistically indistinguishable from full per-layer on WebQSP
(72.85 vs 73.37, p = 0.35), keeps 1.13× of the WebQSP step-time saving and 1.33×
of GraphQA's, and cuts the magnetic bias to 4 instances. `G=2` buys roughly 4
more points of speed for a small but seed-consistent 0.8 F1; `G=1` is ruled out —
it is the only setting with a clear accuracy cost, and it is not meaningfully
faster than `G=2`. On a trunk run that lasts months and cannot be revisited
mid-flight, spending 4% of step time rather than 0.8 F1 is the right side of that
trade.

---

## 5. Notes and caveats

* **Step-time noise.** `ixh` is shared, so co-tenancy perturbs `s/it`. Three
  seeds per arm; treat small differences as noise and prefer wall-clock `s/it`
  over `step_ms_mean`.
* **Step time is not in the run records for `002`/`003`,** and had to be scraped
  back out of the slurm logs (`analyse.step_time_from_log`). Only the GraphQA
  runner writes `train_steps_per_second`. The scrape is reproducible and its
  seed-to-seed spread is small (sd ≤ 0.05 s/it), but half of what D2 measures
  should not live in a log file — the kgqa and context runners should record a
  step-time column like GraphQA's.
* **`magnetic_content` cannot be grouped** and is rejected by config validation:
  its input is the live per-layer residual stream, so its bias is layer-dependent
  by construction.
* **The legacy flags keep their own code paths** rather than being rewritten as
  aliases. Folding them would have moved parameter names
  (`model.layers.{i}.self_attn.graph_bias.bias_modules.{j}.*` →
  `group_graph_bias.{g}.*`) and broken every existing checkpoint, for no gain the
  equivalence tests do not already provide.
* **A wiring bug caught mid-campaign, and the test that now guards it.**
  `--magnetic-groups` was added to all three argparse builders but never passed
  into the `RunConfig(...)` construction, so the flag parsed, validated, and was
  then dropped: every arm of the six-arm sweep ran the default legacy path and
  recorded `magnetic_groups: 0`. Nothing errored — the sweep simply measured one
  configuration six times, and it was only visible because a 3-seed arm came back
  with n=10. `tests/experiments/test_magnetic_groups_cli.py` now pins the whole
  chain (argv → `parse_args` → `RunConfig` → `bias_params()`) for all three
  experiments, and was verified to fail against the pre-fix code. The first two
  submissions were discarded; the campaign was restarted from scratch.
* **Run records carry `magnetic_groups` as a first-class column.** Before that
  was added, the arm was only recoverable by parsing `sweep_run`.
* **`sbatch_context4k_data.sh` fixes a job-id capture bug** inherited from
  `src/experiments/context/sbatch_mainsweep_data.sh`: the cluster MOTD's clock
  line (`18:07:24 up 47 days`) starts with digits, so `grep -oE '^[0-9]+'`
  captures `18` for every job and the merge's `--dependency` is malformed. The
  fix anchors both ends (`'^[0-9]+$'`).

---

## 6. Follow-up: step time at 1024–4096 nodes, and the plain-LLM floor

*(pending — benchmark built 2026-08-03, results below once the job lands)*

### 6.1 Why

§4.2 measured the `G` knob in three places that all sit **at or below 512 nodes**,
and the magnetic bias is `(B, H, N, N)` in *nodes* — so the sweeps priced the
sharing knob where the shared object is smallest, and `003`, chosen to be the
worst corner, turned out to be the corner with the fewest nodes of the three
(≤ 128). Two questions were therefore left open:

1. how the saving scales into the 1024–4096-node regime the trunk is aimed at;
2. what any of it costs against a plain LLM at the same sequence length.

`bench/` answers both. It is speed-only — no training, no accuracy, a few hundred
timed steps.

### 6.2 What it runs

```bash
./src/experiments/bias_sharing/bench/sbatch_speed.sh full     # one GPU, all sources
python3 -m src.experiments.bias_sharing.bench.report
```

| source | batches | arms |
|---|---|---|
| `synth` | synthetic, N ∈ {512, 1024, 2048, 4096} | `G ∈ {0,1,2,4,8,16}` + `llm` |
| `webqsp` / `graphqa` / `context` | the experiments' own cached splits | same |

**Recipes are replayed, not retyped.** Each source reads the *command line the
sweep actually ran* out of `results/<sweep>/jobs/*.sh` and pushes it back through
that experiment's own `build_parser` → `config_from_args`, overriding only
`--magnetic-groups`. This is the chain `tests/experiments/test_magnetic_groups_cli.py`
pins, so a benchmark arm cannot silently drift from the training arm it prices.

**The `llm` arm** is a stock `LlamaForCausalLM` — same backbone, same LoRA rank,
same dtype, same gradient checkpointing — handed the *identical* `input_ids`,
`attention_mask` and `labels` tensors and nothing else. `sdpa` where GTLM runs
flex, `eager` where GTLM runs eager (GraphQA). It is deliberately the most
favourable baseline available: plain causal attention is flash-eligible and
GTLM's bidirectional-prefix mask is not. Read it as a floor, not as an
equivalent model.

**Synthetic graphs carry WebQSP's token profile, not a stand-in.** Per-node token
counts are sampled i.i.d. from WebQSP's empirical histogram
(`bench/webqsp_token_stats.json`, extracted from the same cache `002` trained on:
mean 2.99, sd 2.28, median 3, tail to 127). A ±20 %-jittered constant would give
the right sequence length with the wrong between-node length variance. Topology is
a random attachment tree, which is free to be arbitrary **only because `k_hop=0`**:
the flex block mask is then causal + bidirectional-prefix + padding, which the
graph never enters, and the SPD bias is a lookup whose cost does not depend on the
values looked up. `bench/synth.py` states the full faithful/not-faithful split, and
`verify_against_webqsp` re-measures the generated batches and writes the drift
next to the timings.

### 6.3 Measurement protocol

* **Flex compile time is excluded by construction.** Every distinct `(L, N)`
  shape is compiled *and re-executed* during `--warmup-passes 2` full passes,
  whose wall time is recorded separately as `warmup_s` and never enters a
  statistic.
* **The exclusion is checked, not assumed.** Every cell reports
  `first_over_median`; a leaked compile or an allocator storm shows up there, and
  `report.py` reprints any cell above 1.5× as a warning.
* Steps are timed with CUDA events around a synchronized region. One step is one
  `forward + backward` **micro-batch** — a Trainer "it" in §4.2 is
  `accumulation_steps` of these plus an optimizer step (4× for WebQSP, 8× for
  context), so the two tables are not directly comparable.
* No `empty_cache()` between arms' timed regions: returning blocks to the driver
  turns the next allocation into a cudaMalloc storm, which is latency training
  would never pay.

### 6.4 Results

Measured on one A100_80GB, `002_webqsp_g_sweep`'s recipe, one fwd+bwd micro-step.
Reproduce with `python3 -m src.experiments.bias_sharing.bench.report`.

#### Node scaling

Token counts are sampled i.i.d. per seed, so each `N` lands in **two** length
buckets (N=512 → L∈{1536, 2048}, N=1024 → {3072, 4096}, N=4096 → {12288, 16384};
only N=2048 is single-shape). Everything below is therefore **per sequence
length** — a median pooled over two shapes lands wherever the middle step falls
and is not comparable across arms. `report.py` flags mixed-shape rows.

```
     N       L         g0         g1         g2         g4         g8        g16     nobias llm_causal
   512    1536      371.1      271.0      276.7      290.5      321.2      344.5      212.8      187.8
   512    2048      473.9      373.2      385.7      397.2      426.1      452.4      269.2      247.4
  1024    3072     1078.7      661.0      690.0      747.2      860.7      955.5      409.0      367.3
  1024    4096     1405.1      988.6     1019.1     1078.1     1190.4     1286.2      533.9      492.6
  2048    6144     3570.4     1936.4     2049.8     2268.7     2711.6     3066.6      889.9      749.5
  4096   12288    12945.2     6472.6     6910.4     7780.1     9497.2    10904.4     2147.4     1607.0
  4096   16384    16389.0     9868.5    10317.5    11182.0    12917.3    14328.0     2808.0     2222.8
```

`G=1` is **1.3–1.9× faster than `G=0`**, and the ordering
`g1 < g2 < g4 < g8 < g16 < g0` is strict in every row.

**A one-parameter cost model explains the entire grid.** Fitting a single
per-magnetic-compute cost `k` on the `g0`/`g1` pair — the only pair differing
*solely* in magnetic computes, since SPD stays per-layer at every `G` — predicts
the held-out arms to better than 1.5% everywhere:

| N | L | `k` ms | `g2` | `g4` | `g8` | `g16` |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1536 | 2.22 | +0.4% | +0.2% | −1.1% | −2.6% |
| 512 | 2048 | 2.24 | −1.5% | −1.0% | −1.4% | −3.2% |
| 1024 | 3072 | 9.28 | −0.2% | −0.3% | −0.6% | −2.6% |
| 1024 | 4096 | 9.26 | −0.3% | −0.6% | −0.6% | −2.3% |
| 2048 | 6144 | 36.31 | −0.2% | −0.2% | −0.5% | −2.5% |
| 4096 | 12288 | 143.84 | −0.1% | −0.2% | −0.0% | −2.4% |
| 4096 | 16384 | 144.90 | −0.1% | −0.1% | −0.0% | −1.8% |

Two independent checks that this is physics and not curve-fitting: `k` **quadruples
per doubling of N** (2.23 → 9.27 → 36.31 → 144.4), as O(N²·M·m) requires; and `k` is
**the same at both sequence lengths of a given N** (2.22 vs 2.24; 143.84 vs 144.90),
as it must be for a cost that depends on nodes and not tokens.

Two corrections to §1 fall out, in opposite directions:

* **`G=0` costs `3L = 48` computes, not `2L`.** Under 32 the same fit misses the
  held-out arms by 6–15%. The third evaluation is the bias checkpoint nested inside
  the layer checkpoint (§1).
* **`G=L` costs `2L = 32`, not 48** — a one-layer group has no follower to
  rematerialize. Under 48 the fit overshoots `g16` by ~14%; under 32 it lands within
  1.8–3.2%, the small residual being 16-group cache bookkeeping.

So `G=16` really does a third *less* bias work than `G=0`, and §4.1's "check 2 fails
in the opposite direction to the prediction" was a bookkeeping error, not a result.
`cost_model_table()` recomputes this from the jsonl, so the README cannot drift.

**Sharing does not save memory.** Peak is flat across `G` (36.3 GB at `g0`,
36.1 GB at `g16`, N=4096). Peak is a max over the batch list rather than a median,
so it is unaffected by the shape-pooling problem above. The legacy path already
frees each layer's bias before the next, so there was never `G`-fold residency to
remove: **`G` buys time, not space.**

#### Real batches, per sequence length

```
  source      L         g0         g1         g16     nobias        llm
  webqsp    512      290.0      214.4      269.1      192.9      175.4
  webqsp   2048      974.2      739.5      920.0      481.6      534.1
 graphqa     31      189.4      128.6      164.2       95.2       91.5
 context   2048      419.0      400.6      422.1      275.9      288.7
 context   4096      951.0      942.1      954.8      563.7      638.6
```

Pooled medians are **not** reported: real batches span several sequence lengths
(context 2048–4096, WebQSP 512/2048), and pooling once made a 48-compute arm look
faster than a 24-compute one. `report.py` compares shape by shape.

The three tasks sit in different regimes, and node count is what separates them:

* **WebQSP (512 nodes)** — the regime the synthetic study models. `G` buys 26%.
* **context (64 nodes)** — `g0` through `g16` are within **1.5%** of each other.
  The magnetic bias is `(B,H,64,64)`; there is nothing worth sharing. It still
  pays ~380 ms of bias cost (`nobias` 563.7 → `g1` 942.1), and that cost is
  `G`-invariant. The two candidates are the per-layer SPD term (still 48 computes
  at any `G`) and flex's per-score `node_bias` gather; **this benchmark does not
  separate them** — that would need an SPD-off arm. **`G` is the wrong knob for
  this task** either way.
* **GraphQA (21 nodes, L≈32)** — `G` buys 32%, which cannot be compute at that
  size. It is 16 per-layer module dispatches collapsing into one; the step is
  launch-latency-bound. This is also the one task with
  `gradient_checkpointing=False`, so the `3L` mechanism above does not apply.

#### Is the autotuning compile worth its wall time?

`max-autotune-no-cudagraphs` (the model default) against plain `torch.compile`,
per shape and averaged over the two shapes each `N` produces:

| N | extra compile | gain/step | breakeven | wall-clock |
|---:|---:|---:|---:|---:|
| 512 | 93 s | 1.1 ms (**0.4%**) | 81k steps | **7.3 h** |
| 1024 | 185 s | 17.5 ms (2.1%) | 10.6k steps | **2.4 h** |
| 2048 | 233 s | 30.7 ms (1.6%) | 7.6k steps | **4.1 h** |
| 4096 | 3676 s | 216.0 ms (2.6%) | 17.0k steps | **38.6 h** |

Autotuning buys **0.4–2.6%**, and the `llm` control moves 0.993–1.002× between the
two runs, confirming nothing but the compile mode changed. At N=512 the gain is
indistinguishable from zero (per-shape ratios 0.994–1.009×).

**Keep the default at N ≤ 2048** — breakeven is 2.4–7.3 h and every real sweep runs
longer than that. Context pinning `flex_compile_mode="default"` gives up ~2% for
nothing. **Turn it off at N=4096**: the bill jumps 16× to just over an hour, and
recouping it takes 38.6 h of continuous training.

An earlier version of this section reported a flat ~2 h breakeven at every `N` and a
1–3% gain. That came from pooled medians (see the node-scaling note above) and is
superseded by the per-shape numbers here; the compile *wall* column was always
sound, being measured wall time rather than a median.

The bill is **per process, not per arm** — `_FLEX_CACHE`
(`src/models/flex_kernel.py:83`) is keyed on `("flex", dynamic, mode)` and not on
`score_mod`, so whichever arm compiles first pays for all of them. The per-arm
`warmup_s` column reflects that ordering and must not be read as seven bills.

#### The plain-LLM floor: use `llm_causal`, not `llm`

There are two floors, and the difference between them is large enough to change
conclusions:

| | padded `llm` | `llm_causal` | ratio |
|---|---:|---:|---:|
| synth N=4096 (L=15360) | 4652.7 | **2209.7** | 2.11× |
| webqsp L=2048 | 534.1 | 469.5 | 1.14× |
| context L=3456 | 573.6 | 453.1 | 1.27× |
| graphqa L≈32 | 90.1 | 78.5 | 1.15× |

Both are the same model on the same backend with the same LoRA; the only
difference is that `llm_causal` is not given an `attention_mask`. Passing a mask
that contains zeros makes transformers materialize an explicit 4D float mask, and
**sdpa given an explicit mask cannot take the `is_causal` fast path** — it computes
the full square instead of the triangle. WebQSP batches are 61% padding, so this is
not a rounding error, and it grows with `L`: 2.11× at L=15360.

Against the padded floor, `nobias` came out *faster* than the plain LLM
(context 0.89×, webqsp@2048 0.90×) — GTLM's masked flex kernel apparently beating
unmasked attention, which is not physical. Against `llm_causal` every arm orders
correctly: **`llm_causal` < `nobias` < `g1` < … < `g0`**. Quote ratios against
`llm_causal`; ratios against `llm` understate GTLM's overhead by up to 2×.

#### Where the gap to a plain LLM actually comes from

At N=4096, L=16384, splitting `g1`'s 9868 ms step (`k` = 144.90 from the fit):

| component | ms | share |
|---|---:|---:|
| `llm_causal` floor | 2223 | 23% |
| graph mask + flex kernel (`llm_causal` → `nobias`) | +585 | 6% |
| non-magnetic bias (`nobias` → `g1`, minus magnetic) | +6626 | 67% |
| magnetic bias at `G=1` (3 computes × `k`) | +435 | **4.4%** |
| = `g1` | 9868 | |

**Read that 4.4% carefully — it is the *remaining headroom*, not `G`'s value.**
Going from `G=0` to `G=1` removes 45 of 48 magnetic computes, worth
45 × 144.90 = 6520 ms, or **40% of `G=0`'s step**: the knob is a large win over the
legacy path (`g0` 16389 → `g1` 9868, a 1.66× speedup). What the 4.4% says is that
**the knob is nearly exhausted at `G=1`** — every remaining choice of `G` is
competing for a 435 ms slice, while 67% of the step sits in work `bias_params()`
never routes through the group cache (§1) and no value of `G` can touch.

**The 67% is per-attention-score work, not SPD compute.** The two shapes per `N`
are a natural experiment that separates them: SPD *compute* is O(N²) and does not
depend on `L` at all, while everything `score_mod` does runs once per attention
score and is O(L²) at fixed `N`. Holding `N` fixed and varying `L`:

| N | L | non-magnetic bias | exponent in L |
|---:|---:|---:|---:|
| 512 | 1536 → 2048 | 51.5 → 97.3 | 2.21 |
| 1024 | 3072 → 4096 | 224.2 → 426.9 | 2.24 |
| 4096 | 12288 → 16384 | 3893.7 → 6625.8 | 1.85 |

Fitting `cost(L) = A + B·L²` per `N` gives **B = 23–28 ns per token-pair,
essentially constant across a 64× range of node counts** — exactly the signature of
a per-score cost — while the `L`-independent term `A` is consistent with zero at
N=512/1024 and only ~381 ms at N=4096.

| N=4096, L=16384 | ms | share of `g1` |
|---|---:|---:|
| per-score bias machinery (`B·L²`) | 6245 | **63%** |
| SPD compute (`A`) | 381 | 4% |
| magnetic at `G=1` | 435 | 4% |

So the ranking of *remaining* optimizations is **per-score bias machinery (63%) ≫
SPD sharing (4%) ≈ further magnetic sharing (4%)**. Group-sharing SPD the way
`magnetic` is shared — the obvious next move, and cheap, since the owner/follower
machinery already exists — would buy about 4%. The cost that matters is in how flex
*applies* the bias, not in how many times it is computed.

§4.4's verdict (`G=4`, buying accuracy with step time) is unaffected in direction:
`G=4` is still **1.47×** faster than the legacy path at N=4096 (11182 vs 16389).
But §4.4 priced the `G=4`-over-`G=1` premium at "4% of step time", measured where
WebQSP lives — graphs capped at 512 nodes. **That premium grows with graph size:**

| N, L | `g1` | `g4` | premium |
|---|---:|---:|---:|
| 512, 2048 | 373.2 | 397.2 | 6.4% |
| 2048, 6144 | 1936.4 | 2268.7 | 17.2% |
| 4096, 16384 | 9868.5 | 11182.0 | 13.3% |

So on a 512-node task the verdict's trade (≈6% step time for 0.8 F1) is the one
§4.4 describes; at a few thousand nodes the same choice costs 13–17%. If the
generalist trunk trains on graphs much larger than WebQSP's, `G=4` deserves
re-pricing against `G=2` (10317 at N=4096 — only 4.6% over `G=1`) before being locked in.

Caveat: `A` is fitted from two points per `N`, so it carries no error bars, and a
small negative `A` should be read as "consistent with zero". `B` is the load-bearing
number and it is overdetermined across three node counts. `gather_scaling_table()`
recomputes both.

##### Inside the 63%: the backward scatter, not the forward gather

`nobias` cannot split that term further, because `make_score_mod` returns **None**
when there is no bias (`src/models/flex_kernel.py:298`) — the arm removes the
`score_mod` entirely rather than making it cheap, so it bundles the forward gather,
the backward atomic scatter-add into `node_bias`, and the mere existence of a
non-trivial `score_mod`.

`src/models/flex_attn/bench_isolation.py --bias-mode {none,frozen,full}` does split
it (`frozen − none` = gather, `full − frozen` = scatter). At **k=0, our setting**,
its published single-layer decomposition at N=512 is:

| | fwd | bwd |
|---|---:|---:|
| `none` (bare masked kernel) | 19.6 ms | 38.1 ms |
| `frozen` (+ gather) | 85.4 ms (**+66**) | 68.0 ms (+30) |
| `full` (+ scatter) | 83.2 ms | 300.0 ms (**+232**) |

≈ **71% backward atomic scatter, 29% forward gather**, with the bias machinery 85%
of the layer. The scatter is not a bandwidth problem — it is **same-address
contention**: every token pair of a node pair adds into one address in the
`(B,H,N,N)` bias gradient, so conflicts scale as **tokens-per-node²** (2.99 on
WebQSP, so ~9 threads serialize per address). The root cause is that `score_mod`
does per-**token**-pair work for per-**node**-pair data.

`bench/bias_modes.py` re-runs this decomposition at `bias_sharing`'s node counts
(N up to 4096, where the bias table is 1.07 GB and spills L2) to check the split
survives out of the N=512 regime it was measured in. **Results: pending job 121687.**

> The first attempt (job 121672) is **invalid** and kept only as
> `results/bench/bias_modes_INVALID_inprocess.jsonl`. It drove all 16 cells in one
> process to share the flex compile cache; fragmentation accumulated across cells,
> giving `flex[none]` forwards of 0.71 ms at N=1024 against 90.6 ms at N=2048 (for
> 4× the work) and OOMs at N≥2048 where `speed.py` runs a full 16-layer model on the
> same card. `flex_attn/run_sweep.py:5` requires a fresh subprocess per cell for
> exactly this reason.

One thing that fell out of reading that package: it found `int32` `node_ids` to be
a free, lossless −1.4 to −20% (finding #10), but the **model path never adopted it**
— the collator emits `long` (`src/utils/text_graph_collator_v2.py:11`) and
`src/models/flex_kernel.py` has no cast. That is an unclaimed win sitting in front
of the largest cost in the model.
