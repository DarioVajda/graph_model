# Generalist GTLM — plan

**Status:** planning. Written 2026-08-01; revised the same day with scale, trainable
surface (D4) and the compute budget (§8); revised 2026-08-02 with D5–D7, the measurement
protocol (§3.4), the status table below and the RL postscript (§11). Nothing here is
implemented yet — see the table for where each decision stands.

**Goal:** transition from one-model-per-task GTLM to a single, continuously-trained,
graph-aware LLM that (a) serves every task we already solve, (b) learns *new* graph
tasks it has never seen, and (c) does not lose the base model's text-only ability.

**Scale:** the trunk targets a **7–12B instruction-tuned backbone**, not the 1B used
for every published result. That single choice propagates into D2 (§3.1), D4 (§3.1),
the memory arithmetic and the whole compute budget — read those together.

---

## 0. What this package is, and is not

**Is:** the machinery for one long-lived training run — a task registry and unified
example schema, mixture sampling, the trunk/fork lifecycle, forgetting control, and the
evaluation suites that decide whether new data is allowed in.

**Is not:**

* **Not architecture.** Every model change (edge-type biases, magnetic sharing
  granularity — §3.1) lands in `src/models/` and is validated through the *existing*
  experiment packages. This package consumes a frozen architecture.
* **Not a data pipeline.** Each source domain keeps its own package under
  `src/experiments/` (`graphqa`, `probes`, `kgqa`, `tag_benchmarks`, `relbench`, and the
  new molecule / CLRS ones). This package holds only a thin *adapter* per domain that
  converts an existing dataset into the unified schema.
* **Not another experiment.** An experiment answers one question and ends. The trunk has
  no end date, accumulates state across months, and its checkpoints have lineage. That
  is why it sits beside `src/experiments/`, not inside it.

---

## 1. Location and layout

`src/generalist/`, a sibling of `src/experiments/`. Like `experiments`, it is **not**
added to `[tool.setuptools] packages` in `pyproject.toml` — it is research
orchestration, not a library other projects import. Run from the repo root as
`python -m src.generalist …`, and sweep it with the existing runner
(`python -m sweep src.generalist configs/....jsonc`), so it inherits the sbatch path,
DDP handling and `runs.jsonl` reporting for free.

```
src/generalist/
├── PLAN.md                  # this document
├── README.md                # how to run it (written when Phase 0 lands)
├── __init__.py
├── __main__.py              # thin dispatcher: modes trunk | fork | admit | eval | data_prep
├── config.py                # TrunkConfig / ForkConfig dataclasses (one knob = one place)
├── schema.py                # THE CONTRACT: unified example format + validator
├── registry.py              # task name → adapter + metadata (domain, weight, splits, caps)
├── adapters/                # per-domain: existing dataset → schema. No new data logic.
│   ├── graphqa.py           ├── kgqa.py       ├── relbench.py   ├── clrs.py
│   ├── probes.py            ├── tag.py        ├── molecules.py  └── text.py
├── mixture.py               # temperature sampling, per-dataset epoch caps, stream builder
├── trunk.py                 # the long run: WSD stable phase, resume w/ optimizer state, lineage
├── fork.py                  # anneal forks (release) and admission forks (gating)
├── forgetting.py            # KL-to-base self-distillation on text-only batches
├── evaluate/
│   ├── suites.py            # in-mixture / held-out / text-only / flat-twin
│   ├── adaptation.py        # steps-to-target on a held-out task, trunk vs base
│   └── report.py
├── configs/                 # .jsonc for the sweep runner
├── results/                 # runs.jsonl, eval snapshots, lineage.json
└── analysis/
```

Two files carry most of the design risk and should be written first:

* **`schema.py`** — every task, in every domain, forever, has to fit this. Getting it
  wrong is a rewrite of all eight adapters.
* **`registry.py`** — the single place that answers "what is in the mixture right now,
  at what weight, from which dataset version." Everything reproducible depends on it.

`adapters/text.py` is the odd one: text-only replay examples are **single-node graphs**,
which by Property 2 make GTLM's forward pass exactly the base LLM's. No special path in
the model is needed — the replay stream is just another task in the registry.

---

## 2. Vocabulary

* **Trunk** — the single long-lived run. One LoRA adapter + one bias set, constant LR
  (WSD stable phase), consuming the growing mixture, no planned end. Checkpoints are
  ordered continuations with optimizer state carried through: `trunk@10k`, `trunk@40k`, …
* **Anneal fork** — copy a trunk checkpoint, run the short LR decay, evaluate. *That* is
  the releasable/reportable model. The trunk never decays and continues untouched.
* **Admission fork** — copy the trunk, train on trunk-mixture + one candidate dataset for
  a fixed budget, run the regression suites. On pass, the dataset joins the trunk's
  mixture going forward; the fork itself is discarded.

---

## Progress at a glance

**This table is the only status record in the document.** Everything below is detail, and
§10 is the definition of *done* for each row — not a second status list. Update rows here
as decisions land.

| | Decision / step | Status | What it gates |
|---|---|---|---|
| **D1** | Edge encoding — node-pair extrapolation vs Levi | ✅ Decision: **Levi** | schema, relbench, molecules |
| **D2** | Magnetic sharing granularity | ✅ Decision: **`G=4`** at 1B (`bias_experiments/bias_sharing` §4.4); re-profile on D4's backbone still owed | trunk cost for its whole life |
| **D3** | Remaining constants | **decided**, evidence in `CLAUDE_CONTEXT.md` | — |
| **D4** | Backbone + trainable surface | leaning Llama-3.1-8B-Instruct, arm B; arms A/B/C settled in Phase 1 | D2, §6, everything at scale |
| **D5** | Context budget + graph-size policy | not started | schema, mixture, §8 |
| **D6** | Bias-path numerics under bf16 | not started | every invariant claim at trunk scale |
| **D7** | Batch and loss accounting | (a) ✅ **two-level, per-example default**; (b)(c) open | whether Phase 1 is interpretable at all |
| §3.2 | Unified schema + `registry.py` + adapters | not started | Phase 1 |
| §3.3 | Held-out task set declared | not started | every generality claim |
| §3.4 | Measurement protocol frozen | not started | every longitudinal number |
| §4 | **Phase 1** feasibility at 1B | not started | go/no-go for 7–12B |
| §5 | Trunk chain harness proven on a throwaway run | not started | the trunk |
| §5 | Trunk + flat twin running | not started | — |
| §7 | relbench / molecules / CLRS admitted | relbench in progress in its own package | — |
| §11 | RL | explicitly deferred, no work before all of the above | — |

---

## 3. Phase 0 — blockers before any trunk training

**Sort by reversibility, not by importance.** WSD exists precisely so that mixture
weights, replay ratio, `bias_lr` and LR magnitude can all change mid-trunk after a
re-warm — those are *not* Phase 0 work, and treating them as such invites the
early-stopping thinking §5 rejects. What belongs here is the two kinds of decision that
cannot be taken back:

* **D1–D7 (§3.1)** — mechanically irreversible. They change parameter shapes or the input
  representation, so no trunk checkpoint survives them.
* **§3.3–§3.4** — the run survives, but the *claim* does not. Held-out sets and eval
  protocol cannot be honestly chosen after seeing results.

**Sequencing.** `D4 → D2` (depth sets the per-layer:shared ratio — 16× at 1B, 32× at 8B,
48× at 12B, so the 1B profile does not transfer) → `D1 → D5 → schema → adapters`. D7
before Phase 1, or Phase 1's interference result is uninterpretable. Held-out set and eval
protocol before *any* mixture run, Phase 1 included.

### 3.1 Decisions to lock (architecture and accounting)

Each of these changes per-step cost, the input format, or what the loss actually weights.
Changing any of them after the trunk is running invalidates the trunk. **D1–D6 are not
generalist-package changes — they land in `src/models/` and are measured through existing
experiment packages.**

---

#### ✅ D1 — Edge encoding: extrapolate node-pair biases to edges, instead of the Levi transform – **Dismissed**

**Motivation.** We currently encode typed/text-bearing edges by converting the graph to a
**Levi graph** — every edge becomes a node — as in KGQA. That preserves the edge's full
text (it becomes a node's text) but pays for it twice:

---

#### D2 — Magnetic bias sharing granularity: lock it before the trunk

*(Details to be worked out; recorded here at the fidelity decided on 2026-08-01.)*

**Why now.** This is a large, permanent compute-vs-accuracy trade on a run that lasts
months. Getting it wrong costs either performance or a lot of GPU-hours, and it cannot be
changed mid-trunk. What is already known:

| | accuracy | speed | memory |
|---|---|---|---|
| `magnetic` (per layer) | WebQSP 0.7066 F1 | ~1.17 s/it; 5.5–6.8 s/it on 100–400-node probes (4–6× baseline) | lower |
| `magnetic_shared` (one instance, all layers) | 0.6971 F1 (**−1 pp**) | ~0.73 s/it (**~38% faster**); SPD-level 1.4–2.0 s/it on probes | **highest** — the shared `(N,N)` tensor sits outside gradient checkpointing |

**What to measure.** Per-layer magnetic's true compute overhead and `magnetic_shared`'s
true accuracy drop, at *trunk* scale and on trunk-representative graph sizes — not at the
sizes those two rows were measured on.

**Middle-ground variants to try:**

1. **Layer-grouped sharing.** `G` groups of layers, one bias instance per group; `G=1` is
   exactly `magnetic_shared`, `G=L` is exactly `magnetic`. Sweep `G ∈ {1, 2, 4, 8, 16}`.
   The existing `BaseBias.shared` flag and `build_shared_bias_modules` already provide
   most of the plumbing — this is a generalisation of a mechanism that exists.
2. **GQA-analogue: fewer bias channels per layer**, broadcast across head groups (e.g.
   `H/4` channels instead of `H`).

**Does the GQA-analogue actually save compute? A prediction to test, not an assumption.**
Reading `MagneticBias.forward`, the per-layer cost decomposes as:

* `_folded_spectral` — four `bil,bjl,blk->bijk` einsums, `O(B·N²·M·m)`. **Head-independent.**
* `proj[1]` (`m→m`) — `O(N²·m²)`.
* `proj[2]` (`m→H`) — `O(N²·m·H)`. *This is the only head-dependent term.*
* token-level expansion to `(B,H,q,kv)` — `O(H·L²)` memory and bandwidth.

So the prediction is that head-grouping does **not** touch the dominant `N²·M·m` einsums,
and saves only `proj[2]` (roughly a fifth of the magnetic cost at `M=128`) — *but* it does
cut the `(B,H,N,N)` output and its `(B,H,L,L)` token-level expansion by 4×, which is the
dominant **memory/bandwidth** term at long `L`. Meanwhile layer-grouping is what actually
cuts the einsum FLOPs. If that holds, the two levers are **orthogonal and combinable**:
layer-grouping for compute, head-grouping for memory. Worth verifying by profiling before
building either.

**Depth makes this decision much bigger than the 1B numbers suggest.** Per-layer
`magnetic` instantiates the `O(N²·M·m)` einsums *once per layer*; `magnetic_shared`
computes them **once per forward regardless of depth**. So the gap between the two scales
with layer count:

| Backbone | layers | per-layer : shared einsum ratio |
|---|---:|---:|
| Llama-3.2-1B (all published results) | 16 | 16× |
| Llama-3.1-8B | 32 | **32×** |
| Gemma-3-12B | 48 | **48×** |

Two consequences. **(a) D2 must be profiled at the target scale, not at 1B** — the −1 pp /
+38% trade measured on WebQSP at 16 layers does not transfer to 32 or 48. **(b) Layer count
becomes a backbone selection criterion** (see D4): for GTLM specifically, a shallower,
wider model is cheaper than a deeper one of equal parameter count.

Note also that the GTLM-specific *memory* is roughly scale-invariant: the `(B,H,N,N)` and
`(B,H,L,L)` bias tensors depend on heads, nodes and sequence length, and Llama-3.2-1B and
Llama-3.1-8B both have 32 heads. Only the transformer activations grow (~4×, checkpointed).

**Decision output:** one `(sharing granularity, head grouping, magnetic_dim)` triple,
written into the trunk's frozen architecture config, with the measured cost and accuracy
delta recorded next to it — **measured on the chosen backbone, at trunk sequence lengths.**

---

#### D3 — The remaining constants (cheaper, but still locked here)

* `question_node: isolated` **everywhere**. Not optional for a generalist: a task-specific
  model can bake the task into its weights, but the trunk receives the task *only* through
  the question text, so question-blind graph encoding is architecturally wrong. Worth
  +5–6 pts on relational GraphQA tasks and +0.8–1.4 F1 on WebQSP (`CLAUDE_CONTEXT.md`
  §4.2, §4.3).
* `k_hop = 0`. Hard gating hurt nearly everywhere it was tested; the one exception
  (`local_hop`) is local by construction and won't generalise.
* Bias arm `spd+magnetic` (subject to D2's sharing decision). The probes show the channels
  are *not* redundant across tasks — magnetic is the only carrier of edge direction, SPD is
  non-trivial on node degree — and the trunk has to serve all of them.
* `lora_dropout = 0.15` (the only regularisation winner in the 32-run campaign); LoRA rank
  `64–128` rather than per-task `8–16`, to be bracketed once in Phase 1.
* Chat formatting in `schema.py`, non-negotiably paired with instruct weights — the
  measured +0.5 F1 / +1.0 Hits@1 required **both together**, neither alone.
* *(What is trainable is no longer a constant — see D4.)*

---

#### D4 — Backbone choice and trainable surface

**Backbone.** A 7–12B instruction-tuned model. Selection criteria, in order:

1. **Layer count, not just parameter count** (D2): per-layer bias cost scales with depth.
   Llama-3.1-8B (32 layers) is cheaper for GTLM than Gemma-3-12B (48 layers) by more than
   the parameter ratio suggests.
2. **Backbone adapter maturity.** `modeling_gtlm_llama.py` is by far the most tested path;
   `modeling_gtlm_gemma3.py` exists but Gemma-3's dual RoPE and sliding-window attention
   both interact with the mask/bias machinery and would need validating at 12B first.
3. Instruct weights + chat formatting (D3).

Current lean: **Llama-3.1-8B-Instruct**, with Gemma-3-12B-it as a considered alternative
only if the gemma3 adapter is validated independently.

**Trainable surface.** Three options, and the cheap one is the one our own hypothesis
predicts we need:

| Arm | Trainable | Optimizer state (AdamW, 16 B/param) |
|---|---:|---:|
| A. LoRA only (`r=64–128`) | ~200M | ~3 GB |
| B. **LoRA + unfrozen `W_q`/`W_k`** | ~670M (8%) | **~11 GB** |
| C. Full fine-tuning | 8B / 12B | **128 GB / 192 GB** |

**Arm B is the trunk default, pending Phase 1.** The standing "RoPE shock" hypothesis in
`src/models/TODO.md` is that per-node position resets feed the frozen backbone an
out-of-distribution positional signal and **LoRA lacks the rank to unlearn it** — with
unfreezing `W_q`/`W_k` as the named untested fix. Arm B *is* that fix. At ~11 GB of
optimizer state it fits on an 80 GB A100 alongside LoRA, so it costs almost nothing to
test, and it is the standing candidate explanation for the unexplained ~1.4 F1
flat-vs-graph gap. **Run A vs B vs C as an arm in Phase 1 at 1B**, where it is cheap; that
also settles the RoPE-shock question as a side effect, which is worth doing regardless.

**What escalating past arm A costs, beyond memory:**

* **Property 2 is voided.** Exact backward compatibility is a *published theoretical
  claim* and depends on the backbone being untouched. Arms B and C both break it.
* **The free teacher disappears.** §6's KL-to-base self-distillation works because
  adapters-off *is* the base model. Once any backbone weight moves, the teacher must be a
  separate frozen copy resident in memory (+16 GB at 8B) or precomputed logits.
* **Forgetting stops being bounded by construction** and becomes a real risk requiring
  genuine replay — which raises the token budget, not just the memory.

**Feasibility of arm C, if Phase 1 demands it** (Slurm `GPU_MEM`: B200 = 180 GB,
B300 = 288 GB, A100 = 80 GB; ~40–80 GB of activations on top):

| | A100 80 GB | B200 180 GB | B300 288 GB |
|---|---|---|---|
| 8B, AdamW | ❌ | edge (~178 GB) | ✅ ~110 GB spare |
| 8B, 8-bit Adam | ❌ | ✅ | ✅ |
| 12B, AdamW | ❌ | ❌ (192 > 180) | ✅ ~45 GB spare |

So **unsharded full fine-tuning across the whole 7–12B range requires B300 specifically** —
making B300 availability a hard dependency, not a preference. Expect **~1.3–1.6× slower per
token** than LoRA: ~1.5× on backbone FLOPs (LoRA still backprops full depth, since biases
sit in every layer — only `dW` is saved), diluted to ~1.3–1.4× by the bias einsums, plus
micro-batch collapse from the memory pressure, which is the term that actually bites and
the one to measure.

**Do not avoid sharding on efficiency grounds.** Unsharded DDP across 8 GPUs all-reduces
the full gradient every step (16 GB at 8B vs ~400 MB for LoRA), while **ZeRO-2 has
essentially the same communication volume** (reduce-scatter + all-gather ≡ all-reduce) and
cuts optimizer+gradient memory ~8×. The legitimate reason to stay unsharded is operational
— single-GPU jobs are trivially resumable across the 7-day walltime chain (§5) — not speed.

---

#### D5 — Context budget and graph-size policy

**Why it is irreversible.** The bias is `O(N²)` per layer and the token expansion is
`O(L²)`; a cap therefore has to exist. Once the trunk has run a few hundred thousand steps
under one cap it has learned the structural statistics of *that regime* — degree
distributions, diameters, how much text sits on a node. Raising the cap later is a
distribution shift, not a config change.

**Three numbers plus a policy:** `max_nodes`, `max_edges`, `max_tokens`, and what happens
to examples that exceed them. The second half is the harder half: **the subgraph sampler is
part of the task definition**, not a preprocessing detail. Relbench and CWQ do not fit
under any plausible cap without one, and swapping samplers mid-trunk changes the task.

**The cost surface is non-linear in the wrong direction for us.** Measured overhead scales
with *nodes per token*: 7.9× at 2048 nodes × 2 tokens vs 1.45× at 512 nodes × 32 tokens
(§9). The bad corner is many nodes carrying little text — which is exactly relbench and
exactly the molecule sets. So the cap is not one budget but a trade between `N` and text
density, and it should be chosen against measured s/it on the *largest* mixture component,
not against a round number.

**Decision output:** `(max_nodes, max_edges, max_tokens)` in the frozen architecture
config, plus a named, versioned sampler per task recorded in `registry.py`.

---

#### D6 — Numerics of the bias path under bf16

**Why now.** Every invariant this project rests on was verified in **float64** —
permutation equivariance at 2.77e-5 ± 2.87e-6, backward compatibility at 2.1e-5 ± 8.3e-6.
The trunk will not run in float64. The magnetic path is where this bites: an
eigendecomposition followed by four chained `bil,bjl,blk->bijk` einsums is exactly the
shape of computation where bf16 cancellation shows up, and nothing currently pins its dtype
independently of autocast.

**The decision:** which parts of the bias path are forced to fp32 (or float64) regardless
of the autocast context — eigendecomposition, `_folded_spectral`, `proj`, the final
`(B,H,N,N)` tensor — and which are allowed to follow the backbone. It is simultaneously a
memory decision: that tensor in fp32 is 2× its bf16 size, and by D2's note the shared
variant already holds the largest single allocation.

**Changing it mid-trunk is a silent discontinuity.** The biases shift slightly and the
model has already adapted to the old ones — no error, no crash, just a quiet regression
with no obvious cause six weeks later.

**What settles it:** re-run the two invariant tests under the trunk's actual autocast
config, at trunk graph sizes, against the float64 reference — measuring drift rather than
asserting a threshold that was calibrated in a different precision.

---

#### D7 — Batch and loss accounting

Three coupled choices that look like implementation detail and are not.

**(a) Loss normalization — DECIDED: two-level, per-example default.** CLRS-Text answers are
long execution traces; GraphQA answers are one word, and answer length is uncorrelated with
how much the model should learn the task. Under token-weighted loss a 5% *example* share can
be a far larger *gradient* share, and §5 specifies mixture weights in examples — so a global
token rule makes mixture weight and gradient weight different quantities and confounds every
mixture-tuning result.

The rule is therefore **two-level**: each task contributes gradient exactly proportional to
its mixture weight (non-negotiable, this is what makes the mixture interpretable), while how
that share is spread over the task's own examples is a **per-task field in `registry.py`**,
defaulting to per-example. The escape hatch matters for CLRS-Text, where the trace *is* the
supervision — per-example normalization gives each execution step ~500× less signal than a
one-word answer — and adding the field now is free where retrofitting it mid-trunk is a
re-warm. Note the interaction with (c): per-example is trivial under homogeneous batches and
needs cross-rank example counts under mixed ones, which is the same footgun class as the
known gradient-accumulation normalization bug and needs a test pinning it across
accumulation steps and ranks.

**(b) Effective batch size in tokens.** LR is coupled to it, so changing it mid-trunk is an
unlabelled LR change on a schedule chosen specifically to have no LR events. Fix it once,
in tokens rather than examples (examples vary in length by two orders of magnitude here).

**(c) Batch composition: task-homogeneous or mixed, and bucketing by `N`.** With `N`
ranging from single digits to the D5 cap, padding waste and gradient noise both ride on
this. Homogeneous batches make bucketing trivial but raise per-task gradient noise; mixed
batches do the reverse and pay in padding. It also interacts with D5 — bucketing is what
makes a large `max_nodes` affordable on average rather than always.

**Decision output:** normalization rule, tokens-per-step, and batching policy in
`TrunkConfig`, fixed for the trunk's life.

---

### 3.2 The unified schema

One example format for every task and every domain: prefix nodes (`SYSTEM`, `QUESTION`) →
context nodes (+ edge spans, post-D1) → prompt node. Answer formatting normalised across
tasks. Worth over-investing in: the single biggest lever found in the entire KGQA thread
was **data format v3 at +4.6 F1** — larger than any architectural knob tested.

**One hard constraint carried in.** The best-scoring WebQSP arm (74.49 F1) was
triplet-serialized and is excluded from GTLM claims because it defeats the architecture's
purpose. Whatever the unified schema does with edges post-D1, it must not reintroduce that
encoding under another name — otherwise the trunk's headline number is uncitable for
exactly the reason that one already is.

While designing `registry.py`, give each task an optional `verify(prediction, example)`
next to its metric, and a generator-vs-corpus flag. Both are needed anyway — the flag for
§5's pass caps, the verifier as the eval-suite scorer — and they happen to be what §11
would need if it ever happens. Adding two fields now is free; retrofitting them across
eight adapters is not.

### 3.3 Declare the held-out task set — before the first mixture run

Every number in `CLAUDE_CONTEXT.md` is held-out *examples* of a *seen* task. Under that
protocol, multi-task fitting and genuine generality are indistinguishable. This decision
costs nothing now and is impossible to make later.

**Held out from all training, permanently:**

* GraphQA: **triangle counting** (the known-hard task) and **connected nodes**
* Probes: **`direction`** (has a provable structural discriminator — symmetric features
  cannot solve it by construction, so transfer there is unambiguous)
* **Family Tree** (`our_tests/family`)
* TAG: **Pubmed**. Not `reddit`, though it is the structurally distinct one
  (`tag_benchmarks/config.py` splits `CITATION_DATASETS` from `REDDIT_DATASETS`): a
  held-out task is only informative where transfer is a fair expectation, and reddit is
  different enough that failure there would say nothing.
* Later: a scaffold-split molecule set, one held-out CLRS algorithm family

**Two metrics on them, not one:**

1. **Zero-shot transfer** — honest, but likely weak at 1B; a weak signal either way.
2. **Adaptation efficiency** (`evaluate/adaptation.py`) — examples/steps to reach a target
   from the trunk vs. from base Llama. This is what actually answers "can it learn new
   graph tasks." A model that isn't zero-shot but learns a new graph task 10× faster is
   still the result we want.

Also keep a **flat-serialization twin trunk** as a running control. Flat still beats graph
on real KGQA (74.9 vs 74.1 WebQSP, 58.6 vs 54.0 CWQ); without the twin, the thesis becomes
unfalsifiable at exactly the scale we care about most.

### 3.4 Measurement protocol to freeze — before Phase 1, not before the trunk

None of these break the run. All of them break the *claim*, and none can be chosen honestly
after seeing results.

* **Eval protocol, version-stamped as files.** Generation config (greedy, per-task
  `max_new_tokens`), answer extraction, metric implementation. Three protocol defects have
  already bitten this project (§9); a trunk produces a *longitudinal* series, so a protocol
  change halfway through doesn't corrupt one number, it corrupts the comparison.
* **Seeds per gate decision.** §5's admission test is stated against ±0.4–1.0 F1 noise bars
  but not against a seed count — without one it is not a test. Fix `n`, and fix the
  aggregation rule (mean ± sd, or worst-case) before the first fork.
* **The flat twin's matching rule** (§3.3): matched tokens, matched FLOPs, or matched
  wall-clock. These give different answers and the control is only valid under whichever is
  declared first.
* **Replay corpus identity.** §6 says "general instruction data"; name the dataset and
  version. It is baked into the trunk as surely as the mixture is.
* **`results/lineage.json` fields** fixed before the first admission, not after the first
  unexplained regression.

---

## 4. Phase 1 — multi-task feasibility (go/no-go)

One model, one adapter, one bias set, on data that **already exists**. No new data
pipelines. The mixture, held-outs of §3.3 removed:

| source | in the mixture | kind |
|---|---|---|
| `graphqa` | the 7 reported tasks minus the 2 held out; `disconnected_nodes` + `node_classification` **train-only** (no official val split) | corpus |
| `probes` | `substructure`, `local_hop`, `text_path` | generator |
| `kgqa` | `webqsp`, `cwq` — **Levi, never triplet** (§3.2) | corpus |
| `our_tests` | `kg_qa` (synthetic KG-QA via `kgqa_gen`, distinct from `kgqa`) | generator |
| `tag_benchmarks` | `cora`, `ogbn-arxiv`, `reddit` | corpus |
| `expressiveness` | HARD, large-N | generator |
| `context` | **eval only** (length extrapolation); saturated, so no training weight | generator |

`expressiveness` is in for one reason: at 1600–2400 nodes it is the *only* source of
large-graph training data in the repo, and D5 locks the trunk into the structural
statistics of whatever it trains on.

Two questions:

1. **Interference** — does it match the per-task models? (We have per-task numbers for all
   of these, so this is directly measurable.)
2. **Transfer** — does it move at all on the held-out set, zero-shot or in adaptation
   efficiency?
3. **Trainable surface** — D4 arms A vs B vs C, run here because 1B makes them cheap. This
   is also the long-owed test of the RoPE-shock hypothesis, so it has standalone value even
   if the generalist programme stalls.

This is the experiment that converts the current guess into a measurement, and it runs
before relbench / molecules / CLRS data work, so a negative result is cheap. **Nothing at
7–12B starts until Phase 1 passes** — months of trunk training on an architecture that is
then revised is the single worst outcome available here.

---

## 5. Phase 2 — the trunk

**Schedule: WSD (`warmup_stable_decay`, already in transformers 4.50.3).** Short fixed
warmup → constant LR trunk with no horizon → short decay applied only in anneal forks.
Cosine bakes `total_steps` into every step's LR, so extending a cosine run is unsound;
WSD's stable phase is horizon-free, which also means a dataset added at step 200k is seen
at full LR rather than only during a tail.

Specifics:

* Trunk config: `num_decay_steps: 0`, generous `num_stable_steps`. Note HF requires
  `warmup + stable + decay == num_training_steps` or the remainder silently runs at
  `min_lr`.
* **Re-warm a few hundred steps after every discontinuity** — mixture change, resume,
  hardware change.
* **Save and restore Adam moments.** Resuming with fresh optimizer state runs the first
  few hundred steps at an effectively huge LR; this is the most common way a long run dies.
* `GraphTrainerV2.create_optimizer`'s two LR groups keep their ratio through all three
  phases. The ~150× `bias_lr`/LoRA differential exists because the biases start random —
  on the trunk they don't. Re-bracket `bias_lr` once at trunk start (3e-2 is known to
  destabilise the bias MLP; 5e-3 is the current default) and consider decaying the *ratio*
  toward 1.

**Operational shape is dictated by the scheduler, not by preference.** `frida` caps
walltime at **7 days**, so the trunk *cannot* be one job — it must be a chained,
requeue-safe sequence with optimizer state persisted across every hop. Same pattern as the
existing `sbatch_chain_*.sh` launchers in `src/experiments/context/`. Two corollaries:

* WSD makes this nearly free — there is no schedule position to reconstruct on resume,
  only optimizer state (§5, and the Adam-moments warning above is doubly load-bearing here).
* **Prefer many short jobs to few long ones.** A trunk that runs in 12–24h chunks backfills
  into fragmented availability; a 4-day 8-GPU reservation sits pending. `squeue` currently
  shows multi-day multi-GPU B200/B300 requests queueing behind each other, so this is the
  live constraint, not a hypothetical.

**Mixture (`mixture.py`).** Temperature sampling (∝ size^0.5 as the default) so relbench
and CWQ don't swamp the probes, plus explicit **per-dataset pass caps**. Exploit the fact
that the probes, expressiveness, Family Tree, `our_tests/kg_qa` and CLRS-Text are
**generators, not corpora**: generate fresh and train single-pass, which starves the
overfitting that produces early peaks. Finite data gets 2–3 passes maximum — and that
includes **GraphQA**, which is a corpus *here*: it was generated programmatically upstream
but the generation code is not in this repo. Porting it is a real project, and worth
scheduling for a reason beyond single-pass data: fresh generation would supply held-out
eval sets for every GraphQA task without carving them out of train, which is the eval
hygiene §9 asks for and the only way `disconnected_nodes` / `node_classification` become
gate-eligible. Same for relbench, TAG, WebQSP/CWQ and molecules.

**Why this fixes the "peaks at 20%" pathology.** That pattern is a high-capacity adapter
overfitting a small fixed dataset under a global checkpoint-selection rule — not a law.
Fresh data removes the repetition; mixture dilution cuts effective epochs by ~20× at 5%
weight; and in a mixture, peaks are **per-task**, so the lever is mixture weight, not early
stopping. A task that peaks and decays is over-weighted or out of data.

**Admission test (`fork.py --mode admit`)**, run for every new dataset. Merge into the
trunk only if all four hold, measured against the recorded seed-noise bars (±0.4–1.0 F1 on
KGQA):

1. held-out task suite does not regress;
2. text-only suite does not regress;
3. in-mixture suite does not regress;
4. the new task actually improves.

Write the criterion into the config *before* the fork runs.

---

## 6. Forgetting control

**This section assumes D4 arm A (LoRA only). Arms B and C weaken it — see the caveat
below.** Under arm A, the frozen backbone plus Property 2 means text-domain damage is
bounded by what LoRA can express: this is **not** full-parameter continual pretraining and
the usual replay ratios don't transfer directly.

* **Primary mechanism: KL-to-base self-distillation on text-only batches**
  (`forgetting.py`). Adapter off = exact base model = a free teacher, same weights, no
  second checkpoint. This optimises "don't change text behaviour" directly instead of
  proxying it through a corpus we'd have to guess at, and it works with *any* text.
  **Caveat: this is a LoRA-only benefit.** Under arm B or C the backbone has moved, so the
  teacher must be a separate frozen copy resident in memory (+16 GB at 8B) or precomputed
  logits over a fixed text set. Budget for it if Phase 1 selects B or C.
* **Replay ratio: start ~15–25%**, weighted toward general instruction data rather than
  raw text — what we're protecting is instruction-*following* more than knowledge, since
  the trunk must answer arbitrary natural-language questions about graphs.
* **Measure, don't assume.** Fixed suite (small MMLU slice, IFEval, GSM8K slice, held-out
  perplexity) every N steps, plus an adapters-off run asserting the base is still
  bit-exact. Then tune the ratio *down* until the suite moves. Expect to go lower than the
  literature suggests, because of the frozen backbone — but that's a measurement.
  Under arm B or C the adapters-off assertion is meaningless (there is no recoverable base)
  and the ratio should be expected to go **up**, toward the continual-pretraining norms —
  forgetting changes from bounded-by-construction to a live risk this section must contain.

---

## 7. New domains

Each develops as an isolated experiment package, then enters through the admission gate.

* **relbench** — `src/experiments/relbench/PLAN.md` is already in progress. Typed foreign
  keys make it the natural first consumer of D1. **Admit it last, and score it on its own
  terms:** it asks for uncertain predictive estimates rather than graph retrieval or
  reasoning, which is not what GTLM is for. On rel-trial the graph arm is −7.7 pp (11.8σ)
  under flat with the bias channel inert; write into the admission criterion that this
  bounds relbench, not the architecture, or a later reader will take it as a general
  failure.
* **Molecules** — needs bond types (D1) and a representation decision (SMILES vs.
  atom-node text) that will matter more than most modelling choices. The `substructure`
  probe (`magnetic` 97.9 vs `none` 51.2 on ring membership) is the encouraging prior.
* **CLRS-Text** — the best OOD instrument we have, more than just another task: it
  generates at arbitrary problem size, so train at `n ≤ 16` and test at `n = 32/64`.
  Size generalisation is a far stronger claim than in-distribution accuracy and is the
  closest thing to a direct test of "learned the algorithm, not the distribution." It also
  forces a decision — does the graph enter as GTLM nodes or stay as text? Both arms are
  informative.

---

## 8. Compute budget

**Cluster inventory** (`sinfo`/`scontrol`, 2026-08-01): ~64 Blackwell GPUs — 32× **B200
(180 GB)** on `ixb1–4`, 32× **B300 (288 GB)** on `ixb5–8` — plus 8× H100 and ~20× A100
(8× 80 GB on `ana`). `frida` walltime 7 days. Aggregate capacity is not the constraint;
**sustained multi-GPU multi-day allocations are**, and they are visibly contended.

**Throughput anchor.** KGQA WebQSP at 1B (flex, bf16, `spd+magnetic`, `lora_r=64`, bs 2 ×
accum 4) runs at **~1.05 s/it** ≈ **15k tok/s per Blackwell GPU** at ~2k tokens/sequence.
Scaling to 8B: transformer FLOPs ×8, magnetic einsums ×2 (16 → 32 layers, head-independent).
At 1B the split is ~⅔ bias / ⅓ transformer (consistent with the published 3× overhead), so
blended ≈ 4× → **~2–4k tok/s per GPU**, i.e.

> **≈ 100–150 GPU-hours per 1B training tokens at 8B.**

| Phase | Estimate |
|---|---|
| D2 sharing sweep (1B, `G ∈ {1,2,4,8,16}` × seeds) + re-profile at target scale | ~50–150 GPU-h |
| D1 edge encoding: build + KGQA Levi head-to-head + invariant tests | ~200–400 GPU-h |
| Phase 1 multi-task feasibility **at 1B**, incl. D4 arms A/B/C | ~300–500 GPU-h |
| Trunk, 5B tokens @ 8B (arm A/B; ×1.3–1.6 for arm C) | ~600 GPU-h ≈ **3–4 days on one 8-GPU node** |
| Flat twin control (no bias, ~0.4×) | ~250 GPU-h |
| Anneal forks (~15% of steps-so-far × ~5 milestones) | ~400–700 GPU-h |
| Admission forks (3 domains × ~2 attempts) | ~500–800 GPU-h |
| Evaluation — generation-heavy; adaptation harness runs from trunk **and** base | ~800–1500 GPU-h |

**≈ 3,000–5,000 GPU-hours for a first full cycle**; 8,000–15,000 across two or three trunk
iterations. At one 8-GPU node-week (1,344 GPU-h) that is **3–6 months wall-clock** under
realistic contention.

**Where the money actually goes: not the trunk.** Forks and evaluation are ~2–3× the trunk
itself. The adaptation harness in particular ("steps-to-target on 6 held-out tasks × 2
starting points × 3 seeds") is a full sweep *every time it runs* — subsample it or schedule
it at milestones only, or it will quietly dominate usage.

**Why this is worth running as a standing background programme.** The trunk is the only
asset in this project that compounds — every run in `checkpoints/` today starts from base
Llama and ends. A resumable, chained, deadline-free job is the ideal consumer of fragmented
and off-peak capacity. The two preconditions are non-negotiable: Phase 0 locked and Phase 1
passed (§4), and lineage discipline (§9). It should not crowd out the two *owed* re-runs
(`our_tests` on the fixed reload path, `tag_benchmarks` without test-set selection) — those
are small, correct published numbers, and are time-sensitive in a way the trunk is not.

---

## 9. Risks

* **D1, D2, D4 or the backbone changing after the trunk starts** → trunk invalidated.
  Hence Phase 0, and hence Phase 1 running at 1B before anything at 7–12B.
* **Compute.** 3× overhead, `O(N²·m)` bias, overhead scaling as nodes-per-token (7.9× at
  2048 nodes × 2 tokens vs 1.45× at 512 × 32). Relbench subgraphs and long molecule sets
  sit in the bad corner. Budget the trunk against measured s/it on the *largest* mixture
  component, on the flex backend. Full budget in §8.
* **Hardware dependency if Phase 1 selects D4 arm C.** Unsharded full fine-tuning across
  7–12B requires **B300 specifically** (12B needs 192 GB of optimizer state; B200 has 180).
  That turns a contended resource into a hard dependency. Mitigations, in order: prefer arm
  B; use a reduced-state optimizer (8-bit Adam / Adafactor); accept ZeRO-2, which costs
  essentially nothing in communication and removes the dependency entirely.
* **Eval hygiene.** Three protocol defects have already bitten this project (test-set
  selection on TAG, loss-vs-metric selection, the bias-reload bug). A long trunk with a
  growing mixture multiplies the chances of a leak. Freeze eval sets as versioned files;
  never select on test; split by *task family*; check that generated data cannot reproduce
  held-out eval items.
* **Attribution drift.** Six weeks in, a regression must be traceable to a specific
  admission. Pin dataset versions per merge in `results/lineage.json`.

---

## 10. Milestones

Definition of *done* per row; current status lives in the table at the top, not here.

- [ ] **D1** edge-encoding design + invariant proofs (float64) + KGQA Levi head-to-head
- [ ] **D2** magnetic sharing profiled and locked (`G` sweep, head-grouping prediction
      tested) — **re-profiled on the chosen backbone, not only at 1B**
- [ ] **D3** remaining constants written into a frozen architecture config
- [ ] **D4** backbone selected (layer count + adapter maturity) and validated end-to-end
- [ ] **D5** `(max_nodes, max_edges, max_tokens)` chosen against measured s/it on the
      largest mixture component; a named, versioned sampler per oversized task
- [ ] **D6** invariant tests re-run under the trunk's autocast config at trunk graph sizes;
      dtype pinned per bias-path stage
- [ ] **D7** loss normalization, tokens-per-step and batching policy fixed in `TrunkConfig`
- [ ] `schema.py` + `registry.py` + adapters for the six existing domains
- [ ] Eval suites frozen; held-out task set declared and committed
- [ ] Measurement protocol frozen (§3.4): seeds per gate, flat-twin matching rule, replay
      corpus version, lineage fields
- [ ] Plumbing smoke run (1B, ~1 day, 3 maximally different-shaped tasks) asserting
      per-task gradient share == configured mixture weight, every adapter round-tripping
      the schema validator, and no task silently contributing zero — *before* Phase 1
      spends 300–500 GPU-h on a result the `--magnetic-groups` class of silent wiring bug
      would make uninterpretable
- [ ] **Phase 1** at 1B: multi-task feasibility **+ D4 arms A/B/C** → go/no-go
- [ ] Trunk chain harness (resumable ≤7-day jobs, optimizer state persisted) proven on a
      throwaway run *before* the real trunk starts
- [ ] Trunk started (WSD), flat twin started
- [ ] relbench / molecules / CLRS admitted through the gate

---

## 11. Postscript — RL, someday

**Not part of this plan. No work here happens before everything above is finished.** This
section exists so the option isn't accidentally closed off, and for no other reason.

RLVR would be an unusually good fit later, because the verifier problem is already solved
for half the mixture: GraphQA, the probes, Family Tree and CLRS-Text are *generators*, so
every example carries an exact answer by construction and prompts are unlimited. That is
the opposite of the corpora, where §5 needs pass caps precisely because data is finite.

Three things worth remembering, none of which cost anything today:

* **Ordering is forced.** RLVR sharpens what a model already does sometimes; at ~0 pass
  rate the gradient is ~0. So it would run on an *anneal-fork checkpoint*, never instead
  of the trunk.
* **The blocker is rollout throughput, not theory.** Generation works
  (`src/models/causal_lm.py`, `generate()` + `prepare_inputs_for_generation`), but only
  via HF `generate()` — vLLM/SGLang have no path for custom additive attention biases and
  per-node RoPE resets. RL is typically 5–15× SFT wall-clock per step, nearly all of it
  generation, which against §8 is comparable to the trunk itself. Before any commitment,
  the one cheap measurement is batched `generate()` throughput on the chosen backbone at
  realistic graph sizes.
* **It interacts with D4.** §6's guarantees assume arm A. If Phase 1 forces arm B or C,
  Property 2 is gone and forgetting stops being bounded by construction; RL with a KL
  penalty to a reference policy is the other known way to add capability while moving the
  policy far less than SFT does. That is a reason to *reconsider* this section later under
  arm B/C — not a reason to start now.

Keep the door open at zero cost: the held-out set of §3.3 stays unspent (RL that trains on
triangle counting destroys the only clean read on whether it generalises past its own
verifier), and if RL is ever budgeted it gets its own line, never folded into §8's trunk
estimate.
