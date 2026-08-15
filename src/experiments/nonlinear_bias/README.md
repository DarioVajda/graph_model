# nonlinear_bias — the non-linear pooled magnetic head

Phase 2 of `src/models/NON_LINEAR_BIAS.md`. The plan document is the authority on
the maths and the reasoning; this file is the operational record.

`mixed_bias` closed negatively and concluded that WebQSP's residual is **pairwise**
non-linearity, which no per-node feature reaches. This arm is also a per-node
feature — so by the letter of that conclusion it should also be null. It is worth
GPU-hours anyway for one reason: arms 3, 4 and v2 all spent the *same* per-node
feature, `Re K(i,i)`, and this one spends the strongest feature the family admits.

|  | previous arms | this arm |
|---|---|---|
| node feature | `Re K(i,i)` — the kernel's **diagonal** | a learned pool of the kernel's whole **row** and **column** |
| non-linearity applied to | a node-level scalar | a **pairwise** quantity, before the pool |
| can encode | local structural role | anything a neighbourhood profile can carry, paths included |

So the prior arms asked *is the diagonal enough?*; this asks *is **any** node-level
feature enough?* A null here closes the line properly rather than by extrapolation
from a weak feature. **The prior is against.** The value is in the strength of the
conclusion, and the design below is built so that a null is readable rather than
ambiguous.

## Status — COMPLETE, 2026-08-15. **Negative on WebQSP; do not build the kernel.**

72 training runs across `032`/`033`/`034`/`035`, all recorded, pool audit clean on
every one, plus `036` — a 9-run eval-only pass that measured *why*. The headline in
one line: **the arm reaches the dense ceiling on GraphQA's path task (101.0% of
headroom) and reaches 22.7% on WebQSP against arm 2's 87.8%.** See "Verdict" for
why that is a strong negative and what it revises in `mixed_bias`'s reasoning.

**The mechanism is not what this file first claimed.** The original diagnosis was
capacity — a 64-wide pool cannot summarize a 512-node row. `036` stratified the
per-example deficit by graph size and refuted it: the deficit is **flat in `N`**
(Pearson +0.09, wrong sign), and is 17.7 pp even on graphs of under 32 nodes where
the pool has twice the width it needs. Replacing the exact pair representation with
a pooled marginal costs a uniform ~25% relative, independent of scale. The
capacity paragraphs below are kept, struck through, because they were the
pre-registered reasoning this experiment then overturned.

### Phase 1 record

Phase 1 (implementation + the §7 gate) is green: 40 tests in
`tests/models/test_magnetic_nonlinear_bias.py` plus the full repo suite and the
preflight, on a GPU node via `sbatch_tests.sh` (`GATE_EXIT=0`, job `126280`).
An end-to-end 6-step WebQSP run on the sweep's exact rendered flags completed the
whole path — train, eval, generation, metrics, checkpoint save — before any array
was submitted.

Phase 2 was submitted once and **voided** — `030`/`031` hit an
uninitialized-parameter bug on the `from_pretrained` path (see "030 and 031 are
VOID" below) and were replaced by `032` (array `126399`, 12 runs) and `033`
(array `126411`, 36 runs), submitted 2026-08-14 after a re-run of the full gate
(`GATE_EXIT=0`, job `126352`). That bug is the fourth defect in the list below and
the only one that reached the queue.

Four defects were found and fixed, recorded because three of them would have
silently degraded what the suite could assert rather than failing loudly:

1. `_rms_norm` hard-coded `.float()`, which **down**casts an fp64 input and capped
   every fp64 correctness test at a ~1e-7 floor. Now `promote_types`, so bf16/fp16
   still upcast to fp32 while fp64 stays fp64.
2. The incoming pool's diagnostics overwrote the outgoing pool's, so
   `_pool_stats` reported whichever ran last. Now keyed by direction — the two
   pools are separately parameterized and can move by different amounts.
3. `tests/experiments/test_graphqa_flags.py` was **already red at `f1a7261`** for
   `magnetic_linear_v2` (no `_COMPANIONS` entry, so the arm was tested alongside
   graphqa's default `magnetic=True` and hit the exclusivity rule). Verified
   pre-existing against a worktree at HEAD rather than assumed. This arm added a
   second instance of the same defect; both are fixed. A permanently-red test in
   the gate is a gate that nobody reads.
4. **Parameters not surviving `from_pretrained`** — the one that cost a sweep.
   Full write-up under "030 and 031 are VOID".

## Arms

| Arm | Config | Diagonal | Isolates |
|---|---|---|---|
| 0 | *(no soft bias)* | — | the floor (headroom denominator) — **reused** |
| 1 | `magnetic` | masked | the ceiling — **reused**, context only |
| 2 | `magnetic_linear` | unmasked | **the primary comparator** — reused |
| N | `magnetic_nonlinear`, `magnetic_pool=attn` | unmasked | the proposal |
| N-u | `magnetic_nonlinear`, `magnetic_pool=uniform` | unmasked | is the *learned* pool doing the work? |

Arms 0–2 are read from `mixed_bias`'s `021`/`023` (WebQSP) and `013`/`017`/`019`
(GraphQA), not re-run: same recipe, same M=64, same build, same `bias_self_node`.

Arm N **replaces** the phase channel rather than adding to it, so it has to beat
arm 2 outright. A tandem (`[Q_phase ‖ q_pool]`) is a cheap follow-up if this lands
well; it is deliberately not in this grid, which keeps the comparison clean.

Arm 1 is **not** like-for-like: it runs with the intra-node diagonal masked, which
an inner product cannot express (`LINEAR_BIAS.md` §7.3). Its gap is the headroom,
not the target.

## Sweeps

| Config | Experiment | Arms | Runs | GPU | Budget |
|---|---|---|---:|---|---|
| ~~`030_webqsp_nonlinear`~~ | `kgqa` | — | — | — | **VOID**, see below |
| ~~`031_graphqa_nonlinear`~~ | `graphqa` | — | — | — | **VOID**, see below |
| `032_webqsp_nonlinear` | `kgqa` | N, N-u | 12 | H100/B200/B300 | 15 epochs, `12:00:00` |
| `033_graphqa_nonlinear` | `graphqa` | N, N-u | 36 | A100/H100/B200/B300 | 20 epochs, `00:40:00` |
| `034_graphqa_nonlinear_hot` | `graphqa` | N, N-u @2e-2 | 18 | A100/H100/B200/B300 | 20 epochs, `00:40:00` |
| `035_webqsp_nonlinear_hot` | `kgqa` | N, N-u @2e-2 | 6 | H100/B200/B300 | 15 epochs, `12:00:00` |
| `036_stratified_eval` | `kgqa` | N, N-u, **arm 2** | 9 | B200/B300 | **eval only**, ~14 min each |

### 030 and 031 are VOID — the uninitialized-parameter bug

Recorded in full because the failure was invisible to every test that existed at
the time, and because the shape of it is reusable.

`MagneticNonlinearBias` initialized its parameters with a separate
`nn.init.uniform_` **after** registration:

```python
self.W_attn_out = nn.Parameter(torch.empty(m, h_q))     # WRONG
nn.init.uniform_(self.W_attn_out, -bound, bound)
```

Training does not build the model with the constructor — it calls
`from_pretrained`, which materializes parameters absent from the checkpoint and
then runs `_init_weights`. That only recognizes `nn.Linear` / `nn.Embedding` /
`RMSNorm`, so a bare `nn.Parameter` on a custom module keeps whatever the
materialization left. Measured on the real path with Llama-1B:

| tensor | direct constructor | `from_pretrained` |
|---|---:|---:|
| `W_attn_out` | 3.20 | **0.00** |
| `W_attn_in` | 1.63 | **0.00** |
| `W_val_out` | 26.12 | **NaN** |
| `W_val_in` | 13.12 | **0.00** |
| `gamma_out` | 45.25 | 45.25 |
| `gamma_in` | 0.00 | 0.00 |

The two gammas survived only because they are built as `torch.ones` / `torch.zeros`
*inside* the `nn.Parameter(...)` call. `_build_magnitude_channel` uses that same
idiom (`nn.Parameter(torch.empty(...).uniform_(...))`), which is why the magnitude
arm was never affected — **verified**, so nothing in `mixed_bias` is in question.

**Why no test caught it.** Every unit test constructed the module directly, where
the deferred init works. The two full-model tests build with the constructor, not
`from_pretrained`. And the failure mode is not a crash: with `W_attn == 0` the pool
logits are all equal, so the learned pool **silently degenerates into the uniform
ablation** — the bias is still non-zero, logits still move, and the arm and its own
control become the same model.

**How it was caught.** The pre-registered pool audit, run before reading anything:
`||W_attn_out||_F` came back `0.0000` where init is `~3.2660`, and `inf` in one
run. The `031` table then read as expected under that: several attn/uniform pairs
byte-identical, most cells sitting exactly on the known no-bias floor
(`shortest_path` 0.4700 against arm 0's 0.469). `030` was cancelled ~1 h in on that
evidence, saving ~48 GPU-hours.

The fix initializes inside the `nn.Parameter(...)` call and is pinned by
`test_parameters_survive_from_pretrained`, which round-trips a tiny model through
`save_pretrained` / `from_pretrained` so every bias tensor is a missing key —
exactly the training case — and asserts each is finite and not identically zero.

Void records are kept under `results/030_webqsp_nonlinear_VOID` and
`results/031_graphqa_nonlinear_VOID` (their checkpoints, 20 GB, were deleted).
**No number from either is reportable.**

**Confirmed live on the resubmission**, at step 80 rather than after five hours —
the audit was re-run on `033`'s first checkpoints deliberately early, because the
whole point of the void was that the defect is invisible in a training curve:

| | `0008` (attn) | `0003` (uniform) |
|---|---|---|
| `‖W_attn_out‖_F` | **3.50 – 3.86** (init ~3.27) | absent, as designed |
| `\|gamma_in\|_inf` | 0.057 – 0.168 (0.0 at init) | 0.034 – 0.067 |
| `\|gamma_out\|_inf` | 1.09 – 1.20 (1.0 at init) | 1.03 – 1.05 |
| gain bound on `\|b\|` | 3.97 – 12.6 | 2.24 – 4.52 |

`W_attn` is now non-zero *and* off its initialization, `gamma_in` has left the
floor so the bias is live, and the gain bound is finite where the void runs
reported 287 and `inf`. The two arms are genuinely different models again.

`bias_lr` is `{1e-3, 5e-3}`, down from `mixed_bias`'s `{5e-3, 2e-2}`, because this
head is ~2.7 M parameters — ~4.6× `MagneticBias`'s own head. 5e-3 is retained
deliberately: it is where arms 1 and 2 were measured, so it is the cell that makes
the reused baselines comparable, and it is ordered first in both grids so that if
only half a grid runs, the half that runs is the comparable half.

```bash
./src/experiments/nonlinear_bias/sbatch_tests.sh          # 1. the §7 gate (must be green)
python3 -m src.experiments.nonlinear_bias.preflight       # 2. every run parses, validates, gets its head
python3 -m sweep src.experiments.kgqa    src/experiments/nonlinear_bias/configs/032_webqsp_nonlinear.jsonc
python3 -m sweep src.experiments.graphqa src/experiments/nonlinear_bias/configs/033_graphqa_nonlinear.jsonc
python3 -m sweep.report src/experiments/nonlinear_bias/results/<sweep>
```

Before reading any result:

```bash
python3 -m src.experiments.nonlinear_bias.pool_audit 'checkpoints/kgqa/032_webqsp_nonlinear_*'
```

`gamma_in` is zero-initialised and the bias is **identically zero** at zero, so a
run whose `gamma_in` never moved trained with no graph bias at all and its "null"
is worth nothing. This is `mixed_bias`'s arm-v2 discipline, applied to the
quantity that plays the same role here.

## The pre-registered reading

Written before the runs land, so the result is read against a fixed rule instead
of a rationalised one. Every number below is from `mixed_bias/README.md`.

**WebQSP.** Floor 0.4566, ceiling (arm 1 @5e-3) 0.7198, headroom 0.2632. Arm 2
@5e-3 is 0.6876 = 87.8%, so the residual this arm is aiming at is **3.22 pp**.
Seed sd on this recipe is 0.19–0.89 pp, and one byte-identical cell reproduced
1.65 pp apart at fixed seed (`022` vs `023`), so 3-seed medians **bound** an
effect rather than resolve one.

| arm N − arm 2 @ its better LR | reading |
|---|---|
| ≥ **+1.6 pp** (half the residual) | the residual is reachable by a node-level feature. `MIXED_BIAS.md` §1's constraint is wrong as stated, and the kernel (`NON_LINEAR_BIAS.md` §9) becomes worth building. |
| within ±1 pp | **null.** The strongest node-level feature the family admits does not move it either, and `mixed_bias`'s conclusion is confirmed on its own terms rather than by extrapolation. The line closes. |
| ≤ **−1 pp** | worse than the linear phase channel it replaces. The pool destroys pair-specific geometry faster than the non-linearity recovers it (`NON_LINEAR_BIAS.md` §3). Same closure, plus a reason. |

**GraphQA — the discriminating half.** `shortest_path` is the task to read first.
Arm 3 recovered **39%** of the selfnode-matched headroom there against arm 2's
**96%**, while matching arm 2 on `node_degree` and `edge_count` — the shape that
produced "a *local* feature detector; it fails exactly where the answer is a
path". A pooled row is not local, so:

* `shortest_path` ≈ arm 2 (≳90%) ⇒ the pooled feature is **not** locality-limited,
  and a WebQSP null then has to be explained by something else — which would be a
  genuinely new fact, since locality is the current explanation.
* `shortest_path` ≈ arm 3 (~40%) ⇒ it fails in the same place for the same reason,
  and the two halves agree.

**The ablation.** If N ≈ N-u on both datasets, the learned pool bought nothing
over any non-linear row summary, and `W_attn` and half of the deferred kernel are
unnecessary regardless of the headline. If N > N-u, the learned selection is
load-bearing and that is the part worth optimizing.

## Results

### GraphQA (`033`) — test accuracy, 3 seeds, COMPLETE 36/36

Reference arms 0–3 from `mixed_bias/README.md` (same recipe, `+selfnode`, 5e-3).
Headroom is arm 1 − arm 0 per task. Pool audit clean on all 36: `gamma_in` off
zero everywhere, `W_attn` off its initialization, every gain bound finite.

Mean±sd over 3 seeds; `%` is fraction of headroom, 0% = arm 0 and 100% = arm 1,
per task. Arms 0–4 are `mixed_bias`'s numbers on the identical recipe.

| arm | `bias_lr` | `edge_count` | `node_degree` | `shortest_path` |
|---|---:|---:|---:|---:|
| 0 no soft bias | 5e-3 | 0.026±0.004 (0%) | 0.082±0.007 (0%) | 0.469±0.001 (0%) |
| 1 `magnetic` (dense, masked) | 5e-3 | 0.425±0.022 (100%) | 0.971±0.015 (100%) | 0.946±0.012 (100%) |
| 2 `magnetic_linear` | 5e-3 | 0.433±0.042 (102%) | 0.963±0.014 (99%) | 0.925±0.007 (96%) |
| 3 `magnetic_magnitude` | 5e-3 | 0.418±0.010 (98%) | 0.972±0.013 (100%) | **0.655±0.006 (39%)** |
| 4 `magnetic_hybrid` | 5e-3 | 0.461±0.028 (109%) | 0.975±0.011 (100%) | 0.931±0.005 (97%) |
| **N pooled (attn)** | **5e-3** | **0.463±0.015 (110%)** | **0.986±0.003 (102%)** | **0.813±0.012 (72%)** |
| N-u pooled (uniform) | 5e-3 | 0.473±0.029 (112%) | 0.993±0.007 (102%) | 0.750±0.061 (59%) |
| N pooled (attn) | 1e-3 | 0.391±0.031 (91%) | 0.990±0.007 (102%) | 0.676±0.016 (43%) |
| N-u pooled (uniform) | 1e-3 | 0.365±0.015 (85%) | 0.987±0.003 (102%) | 0.677±0.007 (44%) |

**`shortest_path` is the only discriminating column.** Every arm saturates the two
local counting tasks — arm 3 included — so they carry no information about the
hypothesis, and arm N leading there (110%/102%) is not evidence for anything.

**Arm 4 is the competitive baseline, and arm N loses to it on paths** (72% vs
97%). Arm 4 gets there by KEEPING the phase channel and adding the magnitude one;
arm N replaces phase outright. On GraphQA the phase channel is therefore not yet
dispensable, which is the criterion §"Arms" set for this arm.

**Read `033` alone and `shortest_path` lands at 72%, between the two
pre-registered outcomes. `034` shows that reading was wrong** — 5e-3 is not this
arm's operating point. See "The LR trend" below; the 2e-2 row is the result.

On the two local counting tasks every arm matches or exceeds the dense ceiling,
arm 3 included. Those tasks do not discriminate and nothing should be read from
arm N leading there.

### `034` — at 2e-2 the arm reaches the dense ceiling on paths

| `shortest_path` | test acc | sd | % headroom |
|---|---:|---:|---:|
| arm 3 `magnetic_magnitude` @5e-3 | 0.655 | .006 | 39.0% |
| arm N (attn) @1e-3 | 0.676 | .016 | 43.4% |
| arm N (attn) @5e-3 | 0.813 | .012 | 72.2% |
| arm 2 `magnetic_linear` @5e-3 | 0.925 | .007 | 95.6% |
| arm 4 `magnetic_hybrid` @5e-3 | 0.931 | .005 | 96.9% |
| arm 1 `magnetic` (dense) @5e-3 | 0.946 | .012 | 100% |
| **arm N (attn) @2e-2** | **0.951** | **.008** | **101.0%** |
| arm N-u (uniform) @2e-2 | 0.924 | .007 | 95.4% |

**The 72% was an under-trained pool, not a ceiling.** At 2e-2 the arm matches the
dense `magnetic` ceiling on the path task and beats both factorized baselines,
with a tighter sd than either. The audit tracks the mechanism exactly:
`‖W_attn‖_F` is 3.3 at 1e-3, 4.2–9.8 at 5e-3 and 7–17 at 2e-2, against 3.27 at
init — the pool simply trains more, monotonically, across the whole range tested.

This is the first factorizable arm to reach the dense ceiling on `shortest_path`.
It also answers `mixed_bias`'s locality diagnosis in full: a pooled row is not
locality-limited at all, and the 39% floor of the diagonal-only feature was a
property of that feature rather than of node-level features as a class.

Ablation at 2e-2, paired by seed (attn − uniform, pp): `shortest_path`
**+2.67, 3/3 positive** (+3.40/+2.20/+2.40) — consistent and no longer resting on
one outlier as it did at 5e-3; `node_degree` +0.47 (2/3), `edge_count` +1.13
(2/3). The learned pool is load-bearing on paths and a wash on counting, at both
LRs — the same mechanistic split, measured twice.

The cost is a much wider bias range: gain bounds reach 100–213 at 2e-2 against
13–47 at 5e-3. No run diverged and no gate died, but this is the quantity to watch
if the arm is ever pushed hotter.

### WebQSP (`032`) — test F1, 3 seeds, COMPLETE 12/12. **The arm fails here.**

Floor and ceiling from `mixed_bias` (arm 0 = 0.4566, arm 1 @5e-3 = 0.7198,
headroom 0.2632). Pool audit clean on all 12: `‖W_attn‖_F` 6.8–10.5 at 5e-3
against 3.27 at init, no dead gates. **This is a real null, not an untrained
module** — which is the distinction the audit exists to make.

| arm | `bias_lr` | test F1 | sd | % headroom |
|---|---:|---:|---:|---:|
| 0 no soft bias | 5e-3 | 0.4566 | .0038 | 0% |
| 3 `magnetic_magnitude` | 5e-3 | 0.4614 | .0080 | 1.8% |
| **N pooled (attn)** | 5e-3 | **0.5101** | .0117 | **20.3%** |
| **N-u pooled (uniform)** | 5e-3 | **0.5164** | .0016 | **22.7%** |
| N pooled (attn) | 1e-3 | 0.4989 | .0037 | 16.1% |
| N-u pooled (uniform) | 1e-3 | 0.4913 | .0131 | 13.2% |
| N pooled (attn) | 2e-2 | 0.4820 | .0159 | 9.7% |
| N-u pooled (uniform) | 2e-2 | 0.5030 | .0037 | 17.6% |
| 2 `magnetic_linear` | 5e-3 | 0.6876 | .0019 | 87.8% |
| 4 `magnetic_hybrid` | 5e-3 | 0.6891 | .0089 | 88.3% |
| 1 `magnetic` (dense, masked) | 5e-3 | 0.7198 | .0174 | 100% |

**The LR grid brackets an interior optimum, so under-training is ruled out.**
`035` was run precisely because `034` showed 5e-3 was too cold on GraphQA. On
WebQSP it is not: 2e-2 is *worse* than 5e-3 on both arms (9.7%/17.6% against
20.3%/22.7%), while `‖W_attn‖_F` climbs to 9–20 against 3.27 at init. The pool
trains harder and the answer gets worse. Three LRs spanning 20× with the peak in
the middle is as much as a grid can say: **this is the arm's ceiling on WebQSP,
not its warm-up.**

The best cell of this arm is **−17.1 pp** against arm 2, **−20.3 pp** against
dense, and only **+6.0 pp** above having no graph bias at all. The ordering is
identical on Hits@1 (0.604 vs arm 2's ~0.69) and EM, so it is not an artifact of
one metric. The pre-registered rule's third branch applies, and by a margin far
past its −1 pp threshold.

The ablation is null-to-negative here: attn − uniform is **−0.64 pp** at 5e-3
(1/3 seeds) and **−2.10 pp** at 2e-2 (**0/3**). On WebQSP the learned pool buys
nothing over mean-pooling and actively hurts once it trains hard — the exact
opposite of GraphQA, where it was +2.67 pp and 3/3. Whatever the pool learns to
select on 512-node graphs is worse than not selecting at all.

### `036` — the deficit is FLAT in graph size. Capacity is refuted.

Eval-only: the 3 arm-N-uniform, 3 arm-N-attn and 3 arm-2 checkpoints at 5e-3,
re-scored per example on the same 1628-question test split, stratified by node
count. All 9 reproduce their recorded aggregate to **<0.06 pp** (max |Δ| 0.00052),
so the decomposition is exact and the reload is verified — see "Reading `036`".

WebQSP's built test split is not the "up to 512" the paragraph below assumed:

    min 3   p25 35   median 61.5   mean 118.5   p90 336   max 512
    fraction above the 64-wide pool: 47.7%

Seed-averaged per-example F1 by size, and arm N-uniform's deficit against arm 2:

| nodes | questions | arm 2 | arm N (uniform) | deficit | relative |
|---|---:|---:|---:|---:|---:|
| 0–31 | 315 | 0.8175 | 0.6408 | **−17.66 pp** | 21.6% |
| 32–63 | 516 | 0.7423 | 0.5376 | −20.47 pp | 27.6% |
| 64–127 | 364 | 0.6977 | 0.5023 | −19.53 pp | 28.0% |
| 128–255 | 206 | 0.5732 | 0.4715 | −10.17 pp | 17.7% |
| 256–512 | 227 | 0.4716 | 0.3596 | −11.19 pp | 23.7% |

```
Pearson(deficit, num_nodes) = +0.086      Spearman = +0.032
```

Capacity predicts a strongly **negative** correlation — bigger graph, bigger
deficit. It is zero, and the sign is the wrong way. Across a 16× range of graph
size the relative deficit sits in a flat 18–28% band with no trend. Decisively:
**on graphs under 32 nodes, where the 64-wide pool has twice the width it needs,
arm N is still 17.7 pp behind.** Width is not the binding constraint at any size.

The one degeneracy proxy already in the records is null too — relative deficit by
gold-set size is 25.8 / 26.3 / 17.9 / 32.0 / 25.8 %, `Spearman(deficit, n_gold) =
−0.017`. It is a weak proxy (it counts correct answers, not structurally
equivalent distractors), so it does not refute role degeneracy; it just fails to
find it.

**What `036` establishes** is narrower and firmer than either hypothesis:
replacing the exact pair representation with a pooled marginal costs a flat
**~20 pp / ~25% relative, invariant to graph size and to answer-set size**. A
uniform structural penalty, not a scaling one.

The ablation by size (attn − uniform: +0.43, −0.67, −1.07, −1.83, −0.33 pp) is
directionally consistent with softmax over-concentrating at scale in four of five
bins, but the last bin breaks it and every magnitude is under 2 pp — inside this
recipe's seed noise (0.19–0.89 pp sd; 1.65 pp at fixed seed). Suggestive only; do
not build on it.

### ~~The dissociation, and the most likely reason~~ — REFUTED by `036`

**Kept struck-through rather than deleted: this was the pre-registered reasoning,
and `036` was run to test it. It was wrong.** The error to learn from is that the
size premise ("up to 512") was taken from the `max_nodes` cap rather than from the
built split's distribution, whose median is 61.5.

> | | GraphQA `shortest_path` | WebQSP |
> |---|---:|---:|
> | nodes per graph | ~20 | up to **512** |
> | arm N, best cell | **101.0%** of headroom | **22.7%** |
> | learned pool vs uniform | +2.67 pp, 3/3 | −0.64 pp, 1/3 |
>
> Both are path-structured tasks and the arm reaches the dense ceiling on one while
> barely clearing the floor on the other. The variable that moves is graph size...
> Compressing a 20-entry row into `d_struct = 64` can be near-lossless; compressing
> a 512-entry row into the same 64 cannot... That the learned pool *helps* at N≈20
> and does *nothing* at N≈512 is the signature one would expect if the pool is
> running out of capacity, not out of training.
>
> **This is a hypothesis, not a measured result.** Graph size is confounded with
> dataset, task and sequence length across these two experiments; nothing here
> isolates it.

### The dissociation, restated

`036` sharpens the puzzle rather than dissolving it. Arm N reaches 101% of headroom
on GraphQA `shortest_path` at 7–21 nodes (mean 13.9), and is 21.6% relatively
behind arm 2 on WebQSP graphs of **under 32 nodes**. Both regimes are small, so
size cannot be what separates them.

What differs is the *question being asked*. GraphQA's answer **is** a structural
quantity — a distance, a degree, a count — which a structural role descriptor
computes directly. WebQSP's answer is a specific named entity, which requires
resolving *which* node, and a marginal over partners cannot express that at any
width. This is `NON_LINEAR_BIAS.md` §3's pre-registered risk, and it is the half
of that section that survived:

> the pool marginalizes over the partner index, so `z^out_i` cannot retain *which
> particular* node a relation was with. Directionality is a property of the
> aggregate and survives; **pair-specific resolution is what is at risk**.

§3 was right; the capacity gloss added afterwards was not. The failure is
form-level, not scale-level.

**This remains an interpretation.** `036` rules out size and gold-set size
directly; it does not positively measure identity-vs-role. Doing so would need a
degeneracy statistic computed from the graphs (nodes sharing an identical
relation-type profile, or automorphism-orbit sizes) correlated against the
per-example deficits, which are on disk. Not run — the verdict does not depend on
it.

### Reading `036`

```bash
python3 -m src.experiments.nonlinear_bias.stratified_eval submit   # smoke-gated array
python3 -m src.experiments.nonlinear_bias.stratified_eval report
```

Two guards, because a silently-failed bias reload puts every arm near the no-bias
floor — which reads as "the deficit is flat in `n`", i.e. it would manufacture
this experiment's conclusion out of a loading bug. This project has a bias-reload
incident on record, so both are hard errors:

1. **Reload assertion**, one second after load: parameters that are *exactly zero
   at init* (`gamma_in` for arm N, `proj.0.weight` for arm 2) must come back
   non-zero. A naive "some bias parameter is non-zero" check would pass on a failed
   reload, since `from_pretrained` leaves `W_val`/`W_attn` randomly initialised.
2. **Aggregate reproduction**: each job recomputes its own recorded
   `test_webqsp_f1` and refuses to write its JSONL if it misses by >2 pp.

Scoring primitives are imported from `kgqa/evaluate.py` rather than
reimplemented, and the loop mirrors `generative_eval` including its skip rule, so
the per-example mean *is* the sweep's aggregate. `021` and `032` differ only in
`magnetic_dim`, which is model-side and not in the dataset cache key — verified —
so all 9 runs score the identical split and per-example indices align, which is
what makes the paired deltas meaningful.

### The ablation — the learned pool is load-bearing only on paths

Paired by seed at 5e-3 (attn − uniform, pp):

| task | seed 42 | seed 43 | seed 44 | mean | sign |
|---|---:|---:|---:|---:|---:|
| `node_degree` | −0.20 | −0.20 | −1.60 | **−0.67** | 0/3 |
| `shortest_path` | +1.60 | +14.60 | +2.80 | **+6.33** | **3/3** |
| `edge_count` | −5.80 | +0.60 | +2.40 | −0.93 | 2/3 |

This is the cleanest mechanistic finding in the sweep, and it is what the ablation
was added for. A *learned selection over the row* buys nothing where the answer is
a count of local structure, and buys a lot where the answer is a path — which is
precisely what one would predict if the pool's job is to pick out which partners
matter. Note the magnitude is not resolved: the mean is dominated by seed 43,
where the uniform run underperforms its own siblings (0.680 against 0.786/0.784).
The *sign* is consistent 3/3; the size is not established by three seeds.

### `bias_lr` — 1e-3 is too cold; 2e-2 is the operating point on GraphQA (`034`)

The LR trend on the discriminating task was monotone over the original grid —
43% at 1e-3, 72% at 5e-3 — and the audit said why: the pool trains more at higher
LR. `mixed_bias` had direct precedent that this keeps paying on exactly this task
(arm 3: 0.655 @5e-3 → **0.749 @2e-2** on `shortest_path`). `034` therefore ran the
same two arms at 2e-2, 18 runs / ~12 GPU-h, to decide whether 72% was a ceiling or
an under-trained pool. **It was under-trained: 2e-2 gives 101.0%.**

So the operating point is **dataset-dependent**, which is why both were measured:

| | GraphQA `shortest_path` | WebQSP |
|---|---|---|
| best LR | **2e-2** (101.0% vs 72.2% at 5e-3) | **5e-3** (22.7%; 2e-2 gives 17.6%) |
| why | the pool keeps training — `‖W_attn‖_F` 3.3 → 4.2–9.8 → 7–17 across the grid | the pool trains harder (9–20) and the answer gets *worse* |

`035` ran the WebQSP 2e-2 cell that this section originally argued against
running. The argument for skipping it was that the reference arms themselves
degrade at 2e-2 (arm 1 0.7198 → 0.6297, arm 2 0.6876 → 0.6734), so the comparison
would be against weaker baselines — still true, and the WebQSP *comparison*
accordingly stays at 5e-3. But `034` made the cell worth its 30 GPU-h for a
different purpose: bracketing arm N's own optimum, which is what excludes
under-training as the explanation of the WebQSP null. Three LRs over a 20× span
with the peak in the middle is what a grid can say, and it says this is a ceiling.

Moving the grid down from `mixed_bias`'s `{5e-3, 2e-2}` was a hedge against the
larger parameter count, and on GraphQA it cost signal: at 1e-3 `shortest_path`
collapses to 43% of headroom for **both** arms and the ablation vanishes
(+0.13 pp). The audit explains it — at 1e-3 `‖W_attn‖_F` is 3.33–3.71 against
3.27 at init, i.e. the pool barely moved. **1e-3 is below the threshold at which
the pool trains at all**, and the hedge cost a full extra sweep to undo.

## Verdict — the arm does not replace the phase channel. Do not build the kernel.

`NON_LINEAR_BIAS.md` §9 licenses the fused kernel only if Phase 2 "closes at least
half of WebQSP's residual to dense `magnetic`". It closes **none** of it: the best
cell of the whole 18-run WebQSP grid reaches 22.7% of headroom against arm 2's
87.8%, a **−17.1 pp** deficit, and sits only +6.0 pp above having no graph bias at
all. **The kernel is not scheduled.**

Three properties make that a strong negative rather than a weak one:

* **The mechanism engaged.** `‖W_attn‖_F` 6.8–10.5 at 5e-3 and 9–20 at 2e-2
  against 3.27 at init, `gamma_in` off zero on every run. This is not arm v2's
  "did the gate move" question — it moved, everywhere.
* **The LR is bracketed.** Three LRs over a 20× span with the peak in the middle
  (16.1% → 22.7% → 17.6%). Under-training is excluded, which is exactly what
  `035` was run to establish after `034` showed 5e-3 was too cold on GraphQA.
* **The ablation is negative.** The learned pool loses to mean-pooling on WebQSP
  (−0.64 pp at 5e-3, −2.10 pp and 0/3 at 2e-2). Even the arm's own extra
  machinery does not pay for itself at that graph size.
* **The failure is not a budget** (`036`). The deficit is flat in graph size
  (Pearson +0.09) and already 17.7 pp where the pool has 2× the width it needs, so
  there is no `d_struct` at which this arm becomes competitive on WebQSP. This is
  what removes the last route by which the kernel could have been re-licensed
  later: a capacity failure would have been fixable, and this one is not.

`mixed_bias`'s conclusion therefore **survives, but its reasoning does not**. That
document held the residual to be *pairwise* non-linearity, unreachable by any
per-node feature. This arm is a per-node feature that reaches the dense ceiling on
GraphQA's path task (101.0%), beating both factorized baselines — so "no per-node
feature reaches it" is false as stated. What is true is narrower:

> A per-node feature obtained by pooling the pairwise kernel is a good *structural*
> summary and a poor *identity* resolver. Where the answer is a structural quantity
> it reaches the dense ceiling; where the answer is a specific node it loses a flat
> ~25% relative to the exact pair representation, **at every graph size**.

The size-scaling version of this claim — the one this file carried until `036` —
is **measured false**: the deficit is flat in `N` (Pearson +0.09) and is already
17.7 pp on graphs under 32 nodes. The obstacle is the *form* of the
representation, not its width, so it is not fixable by a budget.

That is a sharper closure than an impossibility argument, and a more useful one:
it says exactly which tasks this family can serve (structural read-outs, where it
is now the best factorizable arm on record) and which it cannot (entity
resolution). `linear_bias` §7 still closes negatively for this family on WebQSP; it
just closes for a different reason than `mixed_bias` recorded, and a different one
again from what this file first proposed.

### The one positive finding, with its scope condition

On GraphQA `shortest_path` at 2e-2 this is **the first factorizable arm to reach
the dense ceiling** — 0.951±0.008, 101.0% of headroom, against arm 2's 95.6% and
arm 4's 96.9%, with a tighter sd than either. That result stands on its own and is
the durable output of this experiment.

**The scope condition is about the TASK, not the graph size.** This file
originally scoped it as "a genuine result about small-graph regimes"; `036` shows
that framing is wrong, because the WebQSP deficit is just as large on WebQSP's own
small graphs (17.7 pp under 32 nodes). The correct statement:

> Valid where the answer is a **structural quantity** the graph determines — a
> distance, a degree, a count. **Not** valid where the answer is a specific named
> entity, at any graph size, and it must not be quoted as a WebQSP-scale or a
> knowledge-graph result.

It is also **not** a reason to keep the line open: item 1 below is struck, because
the sweep it would have licensed is now known to measure nothing.

## Still open

`036` closed the first three. Kept with their resolutions so they are not re-opened.

1. ~~**Is it capacity per node, or graph size per se?** A `d_struct` sweep on
   WebQSP (64 → 128 → 256).~~ **STRUCK — do not run.** `036` shows width is not the
   binding constraint: the deficit is flat in `N` and is 17.7 pp where the pool
   already has 2× the width it needs. A `d_struct` sweep would buy nothing. It was
   independently self-defeating on cost — at `d_struct = 256` the appended head
   width is 4× `head_dim` and the argument against `magnetic_linear`'s `2M = 128`
   disappears — so it lost the engineering *and* the science.
2. ~~**An `N`-sweep on one dataset**, to separate graph size from dataset, task and
   sequence length.~~ **ANSWERED without new runs.** Graph size varies 3 → 512
   *within* WebQSP at fixed dataset, task and sequence length; `036` reads that
   variation off the existing checkpoints. Size does not modulate the deficit.
3. ~~**Tandem** (`[Q_phase ‖ q_pool]`), deliberately excluded from this grid.~~
   **STRUCK.** The pre-registered go/no-go was whether arm N does well on the
   WebQSP graphs small enough for the pool to be lossless. It does not — 17.7 pp
   behind at `n < 32`, where conditions most favour it. Arm 4 remains the
   precedent that *adding* beats *replacing*, but arm 4 was itself +0.15 pp on
   WebQSP, so tandem has no target left.
4. **Context-4k** remains untested for this arm, as it has since `020`. Unaffected
   by `036`.
5. **A positive measurement of identity-vs-role**, if the mechanism is ever wanted
   for a paper rather than for a build decision: correlate a degeneracy statistic
   computed from the graphs (nodes sharing an identical relation-type profile, or
   automorphism-orbit sizes) against the per-example deficits in
   `results/036_stratified_eval/per_example/`. CPU-only, no new training. `036`
   rules out size and gold-set size but does not positively establish this.

## Deliberate differences from the reused baselines

Recorded up front so they are not discovered later in an analysis.

* **`magnetic_dim` is 64, not 128 (WebQSP) / 32 (GraphQA).** It is not the same
  quantity: for arms 1–4 it is a *per-layer* DeepSets width; here it is the width
  of the shared pair tensor `E`, computed once per forward and pooled by all 16
  layers. That amortization is what makes 64 affordable. It is **not** in either
  experiment's dataset cache key, so every build is byte-identical — pinned by
  `test_experiment_config_gates_accept_the_new_arm`.
* **`bias_lr` grid moved** — see Sweeps above; 5e-3 is retained as the bridge.
* **`bias_self_node=true` everywhere.** An inner product yields `<z_out_i, z_in_i>`
  and cannot be forced to zero, so unmasked is the only configuration this arm can
  run. `linear_bias` Phase 3 established this is not a free choice and that its
  sign is dataset-dependent, which is why arm 2 — also unmasked — is the
  comparator and arm 1 is context.

## Traps when reading these results

Both silently corrupt an arm-keyed analysis rather than erroring.

1. **The `arm` label in `013`/`017` `runs.jsonl` is misaligned with the flags.**
   Run `0004` — the no-bias cell — carries the label `no-spd+rrwp+magnetic`. Key
   off the booleans. Keying off the label swaps the GraphQA floor and ceiling and
   makes arm 0 look like it scores 0.97 without seeing the graph. This sweep's own
   labels are correct (`mag-nonlinear` / `mag-nonlinear-uniform`, pinned by
   `test_the_uniform_arm_is_labelled_distinctly`) — the reused ones are not.
2. **WebQSP's two headroom denominators.** `mixed_bias` quotes 12.2% against the
   *masked* arm 1 that `021` ran, while `linear_bias`'s header quotes 5.6% against
   an *unmasked* arm 1. They differ by roughly the value of the mask. Every
   percentage in this file uses the masked arm-1 denominator, matching
   `mixed_bias`.
