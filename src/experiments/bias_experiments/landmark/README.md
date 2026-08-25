# `landmark` — Phase 0: does the anchor coordinate carry the graph metric?

## Status

| | |
|---|---|
| **Question** | (a) Do the coordinates `LANDMARK_BIAS.md` §2 proposes separate nodes at all? (b) How much of the true graph metric do they carry, at short distances? (c) Which anchor rule? |
| **Answer** | (a) **Yes, under `degree` or `betweenness`** — `dead_node_frac = 0.000` at every $k$, on both WebQSP builds. (b) Well at $d \ge 2$, **poorly at $d = 1$**, and that is structural. (c) **`degree`** — it beats betweenness on 3 of 4 metrics and is $O(1)$ against $O(NM)$. |
| **Cost** | one CPU job, 8 min 54 s (126759). No GPU, no training. |
| **Verdict** | **Proceed**, with `degree` anchors, `d_max = 8`, and the $k$-sweep on WebQSP only. Two findings change the plan: GraphQA cannot price $k$, and the $d{=}1$ stratum is the weak point of the mechanism. |

Run: `./src/experiments/bias_experiments/landmark/sbatch_diagnostic.sh`
(`DIAG_ARGS="--only webqsp --max-graphs 500"` for a fast pass). Results in `results/`.

Data: the WebQSP cache the KGQA arms train on (`…nmax50_ver8_spd64_magq0.25m128_len1024_rcm1_seed42_dfv3`),
both the base build and the `_qnisolated` build that produced the best graph-native
run to date, plus three GraphQA tasks (`shortest_path`, `connected_nodes`, and
`edge_count` as the global-aggregation control a metric channel should *not* help).

---

## What is measured, and why here

The proposed pipeline is `D_out, D_in -> F/G lookups -> Q_pos, K_pos -> inner product`.
This stops at the first arrow. Everything after `D` is a design choice that can be
iterated; `D` is fixed by the data *and by the anchor rule*, and two nodes with the
same `D` row get the same `Q_pos` under any head at any width. A degeneracy found
here is permanent.

**This is deliberately not the Phase 0 `linear_bias` ran.** That one fit the
trained bias offline, and its own Conclusion 6 is that offline imitation R² did not
predict trained quality — it measured how well a head could *fit* a target. Two
things are measured instead:

- **Degeneracy** (`unreach_frac`, `dead_node_frac`, `collision_frac`) — a *kill
  criterion*, in the sense `factorized_rwpe` established with its 94.5% zero rows.
  No head rescues a constant feature.
- **Oracle fidelity** — how much of the true metric the coordinates carry, via
  `o(u,v) = min_j [D_out[u,j] + D_in[v,j]]`. This is a statement about the *input*,
  not about a head's ability to imitate one, which is what makes it a different
  quantity from the R² that misled `linear_bias`. It is still only a **screen**:
  the soft-min is the *initialisation*, and training is free to move `F` and `G`
  elsewhere. High fidelity is evidence the metric is present; low fidelity with low
  degeneracy is ambiguous, not a kill.

Because `o ≥ d` always (triangle inequality), the error is one-sided — "gap", not
"error". And `o = d` **iff some anchor lies on a shortest u→v path**, so
`exact_frac` is the anchor-on-shortest-path hit rate: the mechanism itself rather
than a proxy for it. That identity is also why **betweenness** is a candidate rule
at all, and why `exact_frac → 1` as `k → N`.

**Everything is stratified by `d(u,v)`.** The hypothesis is fine-grained *local*
structure, so the gap at `d ∈ {1,2,3}` decides it. An oracle exact at `d=6` and
useless at `d=1` is worthless for this bias, and an aggregate number would hide
exactly that.

## Anchor rules

All are component-stratified (`LANDMARK_BIAS.md` §2): anchors are apportioned
across weakly-connected components by size, then the rule runs *within* each.
Without that, FPS is **undefined** on a disconnected graph — every eccentricity is
∞, so the tie-break decides everything — and the centrality rules pool in the
largest component, leaving every node elsewhere with an all-`UNREACH` row.

| rule | note |
|---|---|
| `betweenness` | the rule the mechanism implies |
| `pagerank` | directed; hubs — high coverage, low variance expected |
| `fps` | greedy k-center on the undirected skeleton (selection need not be directed; the *coordinates* are) |
| `degree` | in+out, the cheap centrality proxy |
| `mixed` | half pagerank, half fps per component — the hedge |
| `random` | **not a candidate** — a function of the labelling, so it breaks Property 1. It is the null that says whether the rule matters at all. |

Weak components, not strong: inside a weakly-connected component `SPD(i,a_j)` can
still be ∞. `dir_unreach_within_comp` measures that residual, which the allocation
cannot fix.

## What this phase can and cannot decide

**Can:** kill the bias outright (degeneracy); eliminate degenerate rules; set
`d_max` from the observed alphabet (§1 fixes 16 and SR subgraphs are suspected to
be far shallower); tell us whether the rule matters at all (`random` vs the best
structural rule).

**Cannot:** rank the surviving rules. That is a small training sweep at fixed
`k=32`, affordable only because the rule stays a collate-time argument outside
`data_config_key`.

## Results (job 126759, 2026-08-15)

### 1. Topology — the two datasets are in opposite regimes

| dataset | multi-component | components (mean) | `dir_unreach_within_comp` | pairs directed-reachable |
|---|---:|---:|---:|---:|
| WebQSP train | 0.000 | 1.00 | **0.944** | **5.57%** |
| WebQSP test | 0.000 | 1.00 | 0.945 | — |
| WebQSP `_qnisolated` | **1.000** | 2.00 | 0.944 | — |
| GraphQA shortest_path | 0.237 | 1.81 | **0.076** | 83.3% |
| GraphQA connected_nodes | 0.256 | 1.97 | 0.064 | — |
| GraphQA edge_count | 0.000 | 1.00 | 0.161 | — |

**WebQSP is one weak component in which 94.4% of node pairs have no directed
path.** The Levi construction only permits forward traversal along triples, so
directed reachability is far thinner than connectivity suggests, and the channel
can speak about roughly **one pair in eighteen**. Identical on the base and
`_qnisolated` builds (0.9443 vs 0.9442), so the QUESTION node changes nothing
about the main component.

**GraphQA is the mirror image** — near-symmetric, but **24–26% of graphs are
disconnected** on the two relational tasks. Without the component stratification
FPS would have been undefined on a quarter of that data. This is the measured form
of the disconnection concern, not a hypothetical.

`_qnisolated` behaves exactly as designed: 2.00 components, `largest_component_frac`
0.980. The QUESTION node is a size-1 component that always receives one anchor —
itself — so its row is `[0, PAD, …]` rather than all-`UNREACH`. Well-defined, still
the same constant row on every graph. The allocation fixes definedness, not
blankness; nothing can fix blankness for a node with no edges.

### 2. The anchor rule matters enormously, and the ranking was not the predicted one

WebQSP train, $k=32$ (`ex@d` = `exact_frac` restricted to pairs at true distance $d$):

| rule | `unreach` | `dead` | `collision` | `exact` | `gap_mean` | `ex@1` | `ex@2` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`degree`** | 0.867 | **0.000** | **0.512** | **0.927** | **0.009** | **0.738** | 0.991 |
| `betweenness` | **0.821** | **0.000** | 0.582 | 0.921 | 0.005 | 0.668 | 0.984 |
| `pagerank` | 0.913 | 0.186 | 0.546 | 0.871 | 0.046 | 0.533 | 0.849 |
| `mixed` | 0.920 | 0.202 | 0.545 | 0.862 | 0.050 | 0.490 | 0.823 |
| `random` | 0.908 | 0.269 | 0.558 | 0.475 | 1.489 | 0.396 | 0.419 |
| `fps` | 0.931 | **0.452** | 0.513 | **0.249** | 0.162 | 0.293 | 0.206 |

Four readings, three of which contradict the pre-registered prediction:

* **`degree` and `betweenness` are the only live rules**, with `dead_node_frac`
  exactly 0.000 at every $k$ on both builds. The `factorized_rwpe` kill criterion is
  **not** triggered.
* **`fps` is catastrophic, not merely worse.** 45% of nodes get a fully dead row.
  Max-eccentricity seeding selects the *periphery*, and in a sparse directed graph
  the periphery neither reaches nor is reached by anything.
* **`pagerank`'s failure is diagnosable.** Directed PageRank mass flows to **sinks**
  (tail entities), and a sink anchor has $d(a_j, v) = \infty$ for every $v$, so half
  of its coordinate contribution is structurally dead — hence `dead = 0.186` where
  degree scores 0.000. `mixed` is dragged *below* plain `pagerank` by its FPS half:
  the hedge is a liability here.
* **`degree` beats `betweenness`** on exact (0.927 vs 0.921), `ex@1` (0.738 vs
  0.668) and collision (0.512 vs 0.582), losing only on `unreach`. The rule the
  mechanism implies does not win, and the trivial $O(1)$ rule does. Plausible cause:
  with 94% of pairs unreachable, betweenness is computed over very few paths and
  concentrates on relation nodes, while degree picks hub entities *and* relation
  nodes.

**`random` is far behind every structural rule** (exact 0.475, `gap_mean` 1.489 vs
degree's 0.009). The null is decisively rejected — the selection rule is
load-bearing, not a free parameter, so the equivariance machinery earns its keep.

### 2b. How it scales in $k$ — the two live rules

WebQSP train. The ranking is **constant across every $k$**, so the $k{=}32$ table
above is not a lucky slice: `degree` leads on `exact`, `ex@1` and `collision` at all
four values, and `betweenness` leads on `unreach` and `gap` at all four.

| rule | $k$ | `unreach` | `dead` | `collision` | `exact` | `gap_mean` | `oInf` | `ex@1` | `ex@2` | `ex@3` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `degree` | 8 | 0.766 | 0.000 | 0.819 | 0.894 | 0.025 | 0.100 | 0.551 | 0.965 | 0.998 |
| `degree` | 16 | 0.826 | 0.000 | 0.683 | 0.908 | 0.016 | 0.088 | 0.637 | 0.979 | 0.999 |
| `degree` | 32 | 0.867 | 0.000 | 0.512 | 0.927 | 0.009 | 0.070 | 0.738 | 0.991 | 1.000 |
| `degree` | 64 | 0.900 | 0.000 | 0.327 | 0.948 | 0.005 | 0.051 | 0.838 | 0.998 | 1.000 |
| `betweenness` | 8 | 0.697 | 0.000 | 0.866 | 0.887 | 0.025 | 0.106 | 0.498 | 0.950 | 0.996 |
| `betweenness` | 16 | 0.766 | 0.000 | 0.745 | 0.901 | 0.013 | 0.095 | 0.572 | 0.969 | 0.998 |
| `betweenness` | 32 | 0.821 | 0.000 | 0.582 | 0.921 | 0.005 | 0.078 | 0.668 | 0.984 | 0.999 |
| `betweenness` | 64 | 0.866 | 0.000 | 0.392 | 0.942 | 0.001 | 0.057 | 0.773 | 0.995 | 1.000 |

**The columns scale in three different directions, and that is the whole point.**

* **Aggregate fidelity saturates immediately.** `exact` moves 0.894 → 0.948 across an
  **8×** increase in $k$ — +5.4 pp for 8× the head width. Read alone, this says
  $k{=}8$ is enough and the $k$-sweep is pointless.
* **Local resolution does not.** `ex@1` moves 0.551 → 0.838, **+28.7 pp**, and is
  still climbing at $k{=}64$. Since $\hat o = d$ at $d{=}1$ requires an *endpoint* to
  be an anchor, this column is essentially "what fraction of adjacent pairs have an
  anchor at one end", and it can only be bought with more anchors.
* **Discriminability improves fastest of all.** `collision` falls 0.819 → 0.327, and
  the per-doubling drop *grows* (−0.14, −0.17, −0.19). At $k{=}8$ four nodes in five
  share their address with another node in the same graph; at $k{=}64$ it is one in
  three.

`unreach` **rises** with $k$ (0.766 → 0.900), which looks wrong and is not: later
anchors are lower-degree and therefore less reachable, so the average entry is more
often `UNREACH` even as the row as a whole gets more informative. It is the
diminishing-returns signal made visible — the marginal anchor is worth less than
the first one — and it is why `exact` flattens while `collision` does not.

**Consequence for the budget split.** If the aggregate oracle were the target,
$k{=}8$ (24 dims) would do. The two columns that speak to *this bias's* purpose —
local resolution and node discriminability — are still paying at $k{=}64$. That is
a real tension with a 3:1 magnetic-favouring split and is exactly what the training
$k$-sweep has to settle; Phase 0 can only say that the fidelity curve is the wrong
one to read it off.

### 3. The $d=1$ stratum is the weak point, and it is structural

For $d(u,v) = 1$ there is no intermediate node, so $\hat o = d$ **iff an anchor is
$u$ or $v$ itself**. Hence `ex@1` $= P(u \in A \lor v \in A)$, and it is the worst
stratum by a wide margin:

| rule, $k{=}32$ | `ex@1` | `ex@2` | `ex@3` |
|---|---:|---:|---:|
| `degree` | **0.738** | 0.991 | 1.000 |
| `betweenness` | **0.668** | 0.984 | 0.999 |

The oracle is *least* accurate at the shortest distance — exactly where the
"fine-grained local structure" motivation needs it most. Stated as the risk it is,
not as a kill: the soft-min is only the **initialisation**, and the head learns
$F$ and $G$, so this measures whether the *metric* is recoverable, not whether the
coordinates are *informative*. A $d{=}1$ pair with a distinctive coordinate
signature can still receive a distinctive bias.

`ex@1` is also the stratum that improves most with $k$ (degree: 0.551 → 0.637 →
0.738 → 0.838 for $k = 8,16,32,64$) while aggregate `exact` is nearly flat
(0.894 → 0.948). If local resolution is what the bias is for, $k$ buys more than
the aggregate curve suggests — a mild tension with a magnetic-heavy budget split,
to be settled by the training $k$-sweep rather than here.

### 4. GraphQA saturates and cannot price $k$

At $k \ge 32$ **every rule** scores `exact = 1.000`, `collision = 0.000`,
`dead = 0.000` on all three tasks; even $k{=}16$ is ≈0.99. The graphs are small
enough that $k \ge N$ and the anchors become the entire node set — at which point
the coordinates *are* the full SPD rows and the bias is a bilinear form on the
exact SPD matrix, saving nothing.

Consequences: **the $k$-sweep must run on WebQSP**, GraphQA cannot discriminate
anchor rules at all, and a GraphQA landmark arm is measuring a different object
than a WebQSP one. This was flagged as a confound before the run; it is now
measured.

### 5. `d_max = 16` is dead range — use 8

Distribution of finite coordinate distances (WebQSP, degree, $k{=}32$):

| $\le 4$ | $\le 6$ | $\le 8$ | $\le 12$ |
|---:|---:|---:|---:|
| 0.906 | 0.995 | **0.99999** | 0.99999 |

Modal distance is 3. `d_max = 8` gives $S = 11$ instead of 19 at a cost of 1e-5 of
the mass — and the repo's `max_spd` default of 32 is deader still.

## 6. Channel 3 (undirected) — is it worth the head width? (job 126781)

WebQSP train, `degree`. `visDir`/`visUnd` are the fraction of **all ordered pairs**
for which the oracle is finite — i.e. for which the bias is non-zero *at
initialisation*, since $F[\textsf{UNREACH}] = 0$.

| $k$ | `uExact` | `uGap` | `uEx@1` | `coll` 2k | `coll` 3k | `dead` 3k | `visDir` | `visUnd` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.990 | 0.024 | 0.551 | 0.819 | 0.814 | 0.000 | 0.050 | **1.000** |
| 16 | 0.992 | 0.019 | 0.637 | 0.683 | 0.678 | 0.000 | 0.051 | **1.000** |
| 32 | 0.994 | 0.013 | 0.737 | 0.512 | **0.510** | 0.000 | 0.052 | **1.000** |
| 64 | 0.996 | 0.009 | 0.837 | 0.327 | 0.326 | 0.000 | 0.053 | **1.000** |

**What it buys — coverage, by 19×.** The directed pair leaves the bias exactly zero
at init on **94.8%** of ordered pairs; with channel 3 it is non-zero on **100%**.
The undirected oracle is also far more faithful than the directed one (`uExact`
0.994 vs 0.927, `uGap` 0.013 vs 0.009 at $k{=}32$), because the graph is a single
weak component of small undirected diameter.

**What it does not buy — resolution.** `collision` moves **0.512 → 0.510**. Fifty
percent more head width reduces node-level collisions by 0.2 pp. The undirected
coordinates are near-redundant *as a node address*: where a directed distance
exists it usually already equals the undirected one, so the third block restates
what blocks 1–2 hold. `dead` was already 0.000, so there is no gain there either.
And `uEx@1` = 0.737 is identical to the directed `ex@1` = 0.738 — **channel 3 does
not fix the $d{=}1$ weakness**, because that is the endpoint-anchor constraint and
it binds on any metric.

**The budget-matched comparison is the one that decides it.** At a fixed 96 dims
the choice is not "3k vs 2k at the same $k$" — it is:

| 96 dims spent as | pair visibility | `collision` | `exact` |
|---|---:|---:|---:|
| 3 channels × $k{=}32$ | **1.000** | 0.510 | 0.927 |
| 2 channels × $k{=}48$ | ~0.052 | ≈0.42 (interp.) | ≈0.937 (interp.) |

So it is a **pure trade of coverage against resolution**, not a free win.
Directed-only with more anchors is better on every node-level metric and 19× worse
on coverage.

**On GraphQA channel 3 is nearly dead weight**: `visDir` is already 0.833 and
`collision` is 0.000 at $k{\ge}32$ with or without it. It is a WebQSP-shaped fix,
which argues for making the channel configurable rather than unconditional.

On `_qniso`, `visUnd` is 0.992 rather than 1.000 — the QUESTION node is its own
component, so its pairs stay invisible even undirected. As established in §1,
nothing fixes a node with no edges.

### Verdict

**Keep channel 3, and carry the budget-matched 2-channel form as a first-class
ablation arm.** The reasoning, in order of weight:

1. A bias that is *identically zero at init on 95% of pairs* barely functions as a
   bias on those pairs, and the entire design rests on starting as a distance
   oracle and learning from there. Channel 3 is what makes the initialisation
   meaningful on this dataset.
2. Its parameter cost is **negative** — 3 channels at $d_{max}{=}8$ is ~18k against
   2 channels at $d_{max}{=}16$'s ~21k. The cost is purely head width.
3. The collision evidence says channel 3 is redundant as an *address*, so if it
   wins it will win **through coverage alone**. That is a sharp, falsifiable
   prediction, and the 2k arm at matched dims is what tests it.

**The measurement that would sharpen this, not yet run:** SR retrieval is
near-shortest-path from topic entity to answer, so the pairs that matter most may
be exactly the directed-reachable 5%. If gold (topic, answer) pairs are
disproportionately inside `visDir`, the 19× coverage gain is over most pairs but
few *useful* ones, and the trade shifts toward directed-only. Phase 0 measured
coverage over all pairs uniformly; it cannot weight them by task relevance.

## 7. Implementation (locked 2026-08-15)

**Decisions carried from Phase 0**, both settled by measurement rather than
judgement: anchor rule **`degree`**, and the **undirected channel included**
($d_{pos} = 3k$). The second survived its own strongest counter-test — §6 flagged
that SR retrieval is near-shortest-path from topic to answer, so directed coverage
might concentrate on the pairs that matter. `gold_coverage.py` (job 126794)
measured the opposite: gold-answer pairs are *less* directed-visible than average
(`dir_to_gold` 0.034 vs `dir_all` 0.052 on train, 0.035 vs 0.046 on test,
`dir_from_gold` 0.023), because answer entities are tail entities, i.e. sinks. The
directed channels are blind to 96.6% of the pairs touching a gold answer.

| piece | where |
|---|---|
| feature | `src/utils/landmark.py` — k-source BFS, WL tie-break, round-robin emission |
| bias module | `src/models/bias.py :: LandmarkBias` (+ `BIAS_TYPES`, `require_landmark`) |
| config | `src/models/config.py`, `src/experiments/kgqa/config.py`, `__main__.py` |
| plumbing | `causal_lm.py`, `dispatch.py`, `text_graph_collator_v2.py`, `text_graph_dataset.py` |
| dataset augmentation | `add_landmark_column.py` — in place, no 3.5 GB rebuild |
| gate | `tests/models/test_landmark_bias.py` (20 tests), `sbatch_tests.sh` |

**Three things the gate caught that a training curve would not have:**

1. **`PAD` was inert only at initialisation.** `F[PAD]`/`G[PAD]` are trainable rows,
   so once training moved them a padded anchor slot injected a constant into every
   pair — and `k_val` already excludes those slots from the mean, so the result was
   not even normalized for it. Now masked structurally on the query side.
   `UNREACH` is deliberately *not* masked: "no path" is a real symbol.
2. **The per-head gain in the original spec was a dead saddle.** With $\gamma = 0$
   and $G = 0$, both $\partial b/\partial G \propto \gamma$ and
   $\partial b/\partial \gamma \propto G$ vanish and the module never leaves the
   origin — `NON_LINEAR_BIAS.md` §4.4's failure exactly. The gain is redundant with
   $F$ (already per-head) and was removed rather than re-initialised.
3. **Degree alone is too coarse a tie-break.** Every degree-1 leaf on a KG looks
   alike, so ties fell through to node index — i.e. to the labelling, which breaks
   Property 1. Selection now refines with directed Weisfeiler-Lehman colours;
   equivariance is asserted exactly on WL-distinct graphs, and the automorphism-orbit
   residual is recorded as a test rather than hidden.

The gate is green (job 126799, `GATE_EXIT=0`): 20 landmark tests plus the 73
pre-existing collator/linear-magnetic/flex tests the plumbing could have broken.

## 8. Sweep 040 — the dimension scaling law, and a defect it exposed

26 runs (24 + 2 floor), WebQSP, `question_node: isolated`, 15 epochs, 2 seeds,
`bias_self_node=True` on every biased arm. Median seed, best `bias_lr` per cell.

| type | dims | F1 | Hits@1 | best bias_lr | vs floor |
|---|---:|---:|---:|---:|---:|
| floor (`none`) | 0 | 0.4622 | 0.5378 | — | — |
| `magnetic_linear` | 24 | 0.6202 | 0.6935 | 5e-3 | **+0.158** |
| `magnetic_linear` | 48 | 0.6565 | 0.7224 | 5e-3 | **+0.194** |
| `magnetic_linear` | 96 | **0.6825** | 0.7365 | 5e-3 | **+0.220** |
| `landmark` | 24 | 0.3575 | 0.4358 | 5e-3 | **−0.105** |
| `landmark` | 48 | 0.3545 | 0.4367 | 5e-3 | −0.108 |
| `landmark` | 96 | 0.3658 | 0.4410 | 5e-3 | −0.096 |

`magnetic_linear` behaves: monotone in dims, +0.22 F1 over the floor at 96, and
its M=48 cell (0.6565) reproduces `linear_bias` 010's M=32 number (0.6565)
exactly — an unplanned cross-sweep regression check that passed.

**Landmark landed 10 pp BELOW the no-bias floor**, flat in dims, at both LRs, both
seeds (spread 0.002–0.013). That is not a null. The arm strictly contains the
floor — set the gain to 0 and it *is* the floor — so scoring below it means the
optimization was being actively harmed, and the tight spread says systematic.

### The cause: the bias was unbounded, and the spec said it did not need to be

`diagnose_scale.py` (job 126943) read the trained tables:

| bias_lr | max·F | max·G | ‖b‖max | ‖b‖mean | F1 |
|---|---:|---:|---:|---:|---:|
| 5e-3 | 2.0–2.6 | 1.5–2.0 | 9–15 | 0.53 | 0.357 |
| 2e-2 | 5–10 | 4–8 | 64–240 | 1.6–2.0 | 0.211 |

Attention logits are $q\cdot k/\sqrt{64}$, i.e. O(1–10). At ‖b‖max = 9–240 the
bias was not nudging attention, it was replacing it — and F1 tracked ‖b‖
inversely, which is why the higher LR was so much worse. `magnetic_linear`'s head
weight maxes at 0.9–1.1 against orthonormal features, so its bias stays O(1).

**This was an error in `LANDMARK_BIAS.md`'s own reasoning**, not in the data. The
doc argued no normalization was needed because the form is "degree-1 in each
side". True, and irrelevant: $b$ is the product of two *trainable* tables, so it
is **degree-2 in the learned parameters** with no bound on $|F||G|$.
`magnetic_linear` is not a counterexample — one side there is the raw orthonormal
eigenvector ($\lVert V\rVert \le 1$) with only $W$ learned, i.e. degree-1 overall.
This is `MIXED_BIAS.md` §5.7's divergence in a different costume, and §5.8's
remedy applies verbatim: **normalize the factors, put the magnitude in a per-head
gain.** Cauchy–Schwarz then gives $|b| \le 3\max|\gamma|$, a hard bound
independent of the tables.

Init moves with it: $F = G = e^{-d/\tau}$ and $\gamma = 0$, so $b = 0$ at step 0
and $\partial b/\partial\gamma \neq 0$. Zeroing $G$ instead — 040's init — would
make $\hat k = \mathrm{normalize}(0) = 0$ and kill $\partial b/\partial\gamma$
too: a dead saddle.

**What 040 still buys.** Its `magnetic_linear` curve is unaffected by any of this
(different code path) and is reused as the comparator; the floor is reused; and
the landmark cells are kept as the measured cost of the unnormalized form, which
is why `--no-landmark-norm` still exists. Sweep **042** re-runs only the landmark
arms on the identical grid, so 042-vs-040 isolates the normalization.

## 9. Sweep 042 — the bound holds, the arm still loses (job 126949, 12/12 COMPLETED)

Two things are settled mid-run and neither depends on the final numbers.

**The normalization is live and it works as specified.** `diagnose_norm.py` (job
127000) reads $\gamma$ out of the running checkpoints:

| `bias_lr` | $\max\lvert\gamma\rvert$ | $\mathrm{mean}\lvert\gamma\rvert$ | bound $3\max\lvert\gamma\rvert$ | 040's realised $\lvert b\rvert$ |
|---|---|---|---|---|
| 5e-3 | 1.7–2.0 | 0.22–0.23 | **5–6** | 9–15 |
| 2e-2 | 3.4–4.3 | 0.45–0.51 | **10–13** | 64–240 |

All 16 layers report a gain, and it is nowhere near 0 — so this is **not** a
silent no-op, the failure mode that would otherwise make a clean-looking negative.
The bias now sits in the same order as the $O(1\text{–}10)$ attention logits it is
added to, instead of an order above them.

**But $\gamma$ is not converging, it is ramping.** $\mathrm{mean}\lvert\gamma\rvert$
is $0.22$ at `bias_lr` 5e-3 and $0.47$ at 2e-2 — a $4.3\times$ ratio against a
$4\times$ LR ratio. That is $\lvert\gamma\rvert \sim \text{lr}\times\text{steps}$,
the AdamW signature of a persistently-signed gradient: no interior optimum is
being found inside the horizon, and **the learning rate, not the loss surface, is
setting the magnitude.**

### Final numbers: the bound helps exactly where it binds

Median-seed F1, 042 against 040 at matched cells:

| dims | `bias_lr` | 040 (unnorm) | 042 (norm) | $\Delta$ |
|---|---|---|---|---|
| 24 | 5e-3 | 0.3575 | 0.3885 | **+0.0310** |
| 48 | 5e-3 | 0.3545 | 0.3714 | +0.0169 |
| 96 | 5e-3 | 0.3658 | 0.3496 | −0.0162 |
| 24 | 2e-2 | 0.2108 | 0.2901 | **+0.0793** |
| 48 | 2e-2 | 0.2189 | 0.2981 | **+0.0792** |
| 96 | 2e-2 | 0.2415 | 0.2786 | +0.0371 |

The gain concentrates at `bias_lr` 2e-2 — the regime where $\lvert\gamma\rvert\sim
\text{lr}\times\text{steps}$ makes $\lvert b\rvert$ large and the Cauchy–Schwarz
bound actually binds. At 5e-3 the bound is slack and the two forms are within seed
noise. hits@1 at 2e-2 says it louder ($0.2012 \to 0.3381$ at 24 dims, $+0.137$).
**§5.8's remedy is confirmed on real runs** — keep this result; it is independent
of whether the arm wins.

It does not change the verdict. Every normalized cell is still **below the no-bias
floor** (best $-0.0737$ F1 at 24 dims) against a seed spread of 0.0022–0.014.

And it makes the anti-scaling legible: normalized landmark falls monotonically in
dimension ($0.3885 \to 0.3714 \to 0.3496$ over $24 \to 48 \to 96$), where the
unnormalized form looked flat ($0.3575/0.3545/0.3658$) only because runaway
magnitude masked the trend. `magnetic_linear` over the same axis rises
($0.6202 \to 0.6565 \to 0.6825$). **The two scaling curves have opposite signs**,
which decides the 128-dim budget split by measurement: all of it to
`magnetic_linear`.

### EM and F1 are not the same predictions — an earlier reading here was wrong

An earlier version of this section argued that EM up with hit\* down was a
*sharpening*, "since EM $\subseteq$ hit\*". **That subset relation does not hold,
and the inference from it was invalid.** The two metrics come from two different
evaluation procedures:

* `em_accuracy` (`src/train/eval.py:13`) is **teacher-forced**: argmax at each
  answer position, masked to `labels != -100`, compared to gold. No generation —
  at answer token $t$ the model is conditioned on the **gold** prefix. It is
  labelled as such at `kgqa/train.py:259` ("cheap teacher-forced EM (secondary)").
* `f1`, `hits1`, `hit_star` (`kgqa/evaluate.py:150`) come from a real
  autoregressive loop — `model.generate(do_sample=False, num_beams=1)`, decoded,
  `parse_answer_list`, GNN-RAG substring match. The model sees its own output.

There is no algebraic relation between them, so nothing stops them moving in
opposite directions. The data splits 3–1 along exactly that line: in **every**
landmark cell all three generative metrics fall together as `bias_lr` rises while
EM alone rises. For `magnetic_linear` they peak in the *same* cell — so the
divergence is **landmark-specific and diagnostic**, not a metric artifact.

### Why it fails: the query factor is constant where the answer is generated

The prompt node is isolated — no edges — so $d(\text{prompt}, a_j) = \infty$ for
every landmark, the row clamps to `d_max`, and $F[\text{prompt}]$ is uniform.
After L2 normalization it is a fixed unit direction, identical for every question.
The bias at every answer position collapses to

$$b(\text{prompt}, j) \;=\; \gamma \cdot \langle \text{uniform},\, G[j]\rangle$$

which varies over $j$ only through $j$'s own aggregate distance profile. It is a
**question-independent node-saliency prior** — a fixed ranking over candidate
nodes, the same for every question in the dataset. It cannot do retrieval, because
retrieval is query-dependent by definition.

This predicts the observations. Teacher-forced EM measures *continuation* with the
gold prefix already naming the entity, and a mild prior toward answer-like central
nodes helps → EM up. Free generation must *choose* the first entity, where the
fixed prior overrides content attention with the same globally-salient guess →
wrong entity. The damage should therefore concentrate in the **first** pick, and it
does: floor → landmark 24@2e-2 costs $-0.34$ hits@1 but only $-0.20$ hit\*, in
every landmark cell. Raising `bias_lr` raises the magnitude of a signal carrying no
query information, so every generative metric degrades monotonically in LR while
EM climbs.

This is **structural, not a tuning failure**: no `bias_lr` and no $k$ can fix a
query factor that is constant by construction at the position where the answer is
produced.

### 043 — the bracket was inherited, and it is mis-centred

`{5e-3, 2e-2}` came from `magnetic_linear`, where it is well centred because that
bias is degree-1 in its learned parameters and self-limiting. Landmark learns both
sides *and* a gain, so the same LR buys far more magnitude — and every point in
the bracket is on the too-much side. The bracket's own trend agrees: $4\times$ less
LR bought $+14$ pp F1 in 040 (final F1 $0.23 \to 0.37$).

Judging landmark on a bracket tuned for a different bias is not a fair comparison,
so **043** continues the line another $5\times$ and $17\times$ down —
`bias_lr` $\in \{3\times10^{-4}, 10^{-3}\}$, all three dims, 2 seeds. All three
dims, because the $k$-scaling curve is the input to the 128-dim budget split and a
curve measured at the wrong LR is worthless.

Two outcomes, both reportable:

* **F1 peaks above the floor at an interior LR** — landmark is real, and that is
  the curve the budget split reads off.
* **F1 rises monotonically toward the floor as $\text{lr}\to 0$** — landmark's
  optimum is *no bias*: a clean null, reported as one.

This is the prediction the constant-query-factor mechanism makes: shrinking
$\gamma$ shrinks a term that carries no query information, so F1 should approach
0.4622 **from below and not cross it**. If it does cross, the mechanism above is
wrong and this section must be rewritten. The EM gain is *not* evidence against the
null — it is teacher-forced, and the mechanism explains why it moves the other way.

Note 042 tags **every** arm LOW-EDGE, `magnetic_linear` included: 5e-3 is the
bottom of the sampled bracket for all of them, so all these numbers are lower
bounds and the incumbent's margin may be understated too. That is why **044** runs
`magnetic_linear` at the same $\{3\times10^{-4}, 10^{-3}\}$. Without it, 043 would
drop landmark into new LR territory while the comparator stayed fixed — pricing
optimization instead of math, the exact error `linear_bias` Conclusion 5 warns
about.

F1 stays the headline either way, because every prior number in this line of work
is F1 (`linear_bias`, `magnetic_content`, `bias_sharing`). The EM divergence is
reported next to it and never swapped in for it.

## 10. Sweep 045 — GraphQA, the opposite topological regime (job 127021)

WebQSP alone cannot separate "landmark carries real structure that free-generation
F1 does not reward" from "landmark carries little and the EM gain is incidental",
so the arm is run on GraphQA's three standard bias-experiment subtasks
(`node_degree`, `shortest_path`, `edge_count`) — the same ones 013/017/019/024 use,
at the same recipe, so their controls are reused rather than re-burned.

| | WebQSP | GraphQA |
|---|---|---|
| nodes / graph | ~500 | **12.9** (measured) |
| components | 1 weak | 1.81 mean, 23.7% multi |
| pairs with no directed path | **94.4%** | 16.7% |
| metric | free-generation F1 | exact-answer accuracy |

**What `k` means here is not what it means on WebQSP**, and this is the one thing
that must not be misread. With 12.9 nodes per graph, Phase 0 measured under
`degree`:

| k | anchors_mean | oracle `exact_frac` |
|---|---|---|
| 8 | 7.78 | 0.885–0.897 |
| 16 | 12.23 | 0.994–0.995 |
| 32 | 12.90 | 1.000 |

At k=16 nearly every node is an anchor, so the "landmark oracle" stops being an
approximation and becomes the exact all-pairs distance matrix in factorized form.
On `shortest_path` that is close to encoding the label — which is exactly why
every GraphQA bias config here runs `spd: false` and takes its floor from 013 D's
no-spd arm. So **k=4 and k=8 are the honest landmark cells** and any claim about
the method rests on them; **k=16 is a ceiling**, reported as one, and on
`shortest_path` it is never quoted as a win.

`bias_lr` keeps the standard `{5e-3, 2e-2}` bracket rather than 043's lower one,
and that is a substantive choice rather than inertia. Since |γ| ~ lr × steps, the
step count is part of the effective magnitude: GraphQA runs ~625 optimizer steps
against WebQSP's 4890, ~8× fewer, so 5e-3 here lands near where 6e-4 lands there —
already inside the range 043 is probing. Keeping the standard bracket therefore
costs nothing and keeps these numbers comparable to 013/017/019/024.

### Reading the arm labels: `no-spd+rrwp` is the MAGNETIC arm, not the floor

`arm()` names what is **off**, so `no-spd+rrwp` means spd and rrwp are off and
**magnetic is on**; the arm with no graph bias at all is `no-spd+rrwp+magnetic`.
Read the other way round, the strongest bias arm and the floor swap places and
every comparison in the sweep inverts. Verified against the stored boolean flags
rather than the labels:

| arm label | spd | rrwp | magnetic | magnetic_linear |
|---|---|---|---|---|
| `no-spd+rrwp+magnetic` | F | F | **F** | F | ← the floor |
| `no-spd+rrwp` | F | F | **T** | F |
| `mag-linear+no-spd+rrwp` | F | F | F | **T** |

### There is a lot of headroom here, and the incumbents use most of it

| arm | `node_degree` | `shortest_path` | `edge_count` |
|---|---|---|---|
| floor (no bias) | 0.086 | 0.470 | 0.026 |
| `magnetic` (MLP) | **0.984** | **0.970** | 0.470 |
| `magnetic` +selfnode | 0.976 | 0.942 | 0.412 |
| `magnetic_linear` | 0.974 | 0.944 | **0.496** |
| `magnetic_linear` +selfnode | 0.968 | 0.926 | 0.446 |
| **headroom over floor** | **0.898** | **0.500** | **0.470** |

A graph bias is worth 45–90 pp here, not the 1–2 pp WebQSP moves in. So "beats
the floor" is a low bar on GraphQA and is not the question; **how much of the
floor → best-incumbent gap landmark recovers** is. `analyse_graphqa.py` reports
that directly as `gap closed`, alongside per-arm deltas against `magnetic` and
`magnetic_linear` individually.

The like-for-like comparators are 017's `+selfnode` rows: landmark cannot mask its
diagonal (an inner product cannot be forced to zero at i=j), so it runs unmasked
and must be read against arms that also do. Masking is worth 0.6–5.0 pp to the
magnetic arms, so comparing unmasked landmark to a masked incumbent would charge
landmark for a handicap it did not choose.

### Results (job 127021, 54/54 COMPLETED, 2026-08-16)

Median over seeds 42/43/44, each cell at its better `bias_lr`. `vs magnetic` and
`vs mag_linear` are against 017's **unmasked** arms, the like-for-like ones.

| task | k | dims | acc | vs floor | vs `magnetic` | vs `mag_linear` | gap closed |
|---|---|---|---|---|---|---|---|
| `node_degree` | 4 | 12 | 0.552 | +0.466 | −0.424 | −0.416 | 0.52 |
| | 8 | 24 | 0.644 | +0.558 | −0.332 | −0.324 | 0.62 |
| | 16 | 48 | 0.886 | +0.800 | −0.090 | −0.082 | 0.89 |
| `shortest_path` | 4 | 12 | 0.796 | +0.326 | −0.146 | −0.130 | 0.65 |
| | 8 | 24 | 0.886 | +0.416 | −0.056 | −0.040 | 0.83 |
| | 16 | 48 | 0.972 | +0.502 | **+0.030** | **+0.046** | 1.00 |
| `edge_count` | 4 | 12 | 0.240 | +0.214 | −0.172 | −0.206 | 0.46 |
| | 8 | 24 | 0.424 | +0.398 | **+0.012** | −0.022 | 0.85 |
| | 16 | 48 | 0.584 | +0.558 | **+0.172** | **+0.138** | **1.19** |

**1. Landmark scales monotonically in $k$ on all three tasks** — 0.55→0.64→0.89,
0.80→0.89→0.97, 0.24→0.42→0.58. This is the sweep's cleanest result and it is the
opposite of WebQSP, where the arm was flat in dims (0.357 / 0.354 / 0.366). The
bias is reading structure here, and more anchors buy more of it.

**2. At the honest cells it loses to both incumbents.** At $k \in \{4, 8\}$ —
the range where the oracle is still an approximation — landmark is below both
`magnetic` and `magnetic_linear` on every task, by 4–42 pp. The single exception
is `edge_count` at $k=8$, +1.2 pp over unmasked `magnetic` and −2.2 pp under
`magnetic_linear`: a wash.

**3. The $k=16$ wins are label-adjacent on all three tasks, not just
`shortest_path`.** At $k=16$ nearly every node is an anchor, so the features
determine $d(u,v)$ exactly — and *every one of these tasks is a function of the
exact distance matrix*: `shortest_path` is $d(u,v)$ directly, `edge_count` is
$\#\{(u,v) : d = 1\}/2$, `node_degree` is $\#\{v : d(u,v) = 1\}$. So the $k=16$
column measures what a model can do when handed near-exact APSP through an
attention bias. That is worth knowing, and it is **not** a landmark result. The
earlier note flagging only `shortest_path` understated this.

Read that way, the $k=16$ row is also a mild indictment: given the label in
derivable form, landmark still only reaches 0.886 on `node_degree`, where
`magnetic` gets 0.984 without it.

### The bracket is at the wrong edge again — these numbers are a lower bound

**Eight of the nine cells pick `bias_lr` 2e-2, the top of the bracket**, and the
LR effect is large and mostly one-directional:

| task | $k$=4 | $k$=8 | $k$=16 |
|---|---|---|---|
| `shortest_path` | +0.096 | +0.154 | **+0.304** |
| `edge_count` | −0.008 | +0.074 | **+0.126** |
| `node_degree` | +0.034 | +0.014 | −0.016 |

045's header predicted the standard bracket would land *inside* the useful range
because GraphQA runs ~8× fewer optimizer steps and $|\gamma| \sim \text{lr}\times
\text{steps}$. The direction was right and the size was not: the optimum is still
**above** 2e-2, so this sweep under-reads landmark on GraphQA exactly as 040/042
under-read it on WebQSP — with the sign flipped. The mechanism is the same one
and it has now mis-centred the bracket twice, in both directions, which is itself
the strongest evidence that `bias_lr` is this bias's magnitude knob rather than an
optimizer detail.

A follow-up at `bias_lr` $\in \{5\times10^{-2}, 0.1\}$ would settle it. Until
then, every number in the table above is a floor on what the arm can do.

**Wiring.** GraphQA had no landmark support at all (separate config, parser and
cache layout). Added, with three traps closed on the way:

* `arm()` labelled a landmark-only run `no-spd+no-magnetic+no-rrwp` — the label of
  the arm with *no* graph bias, which is its own comparator. It now reports
  `landmark`.
* `preflight.py` was hardwired to KGQA's parser and cache layout; it is now a
  two-row table (parser, splits, cache resolver, matched-dimension grid, expected
  cache count) routed by whether the config carries `task` or `dataset`. GraphQA's
  grid is `{12,24,48}`, not WebQSP's `{24,48,96}` — 96 anchor dims on a 13-node
  graph is not a matched cell, it is a category error.
* `verify_live.py` asserted that `G` receives gradient at init. Under the
  normalized form that is **false by construction**: with γ = 0 the query factor is
  0, so the tables cannot move until the gain opens. It now asserts the two-step
  unroll the spec actually claims — gain moves at step 0 with the tables frozen,
  tables move at step 1 — and both datasets pass it (GraphQA `PAD frac 0.411`,
  which is also the only place PAD inertness is exercised at real padding density).

## 11. Sweep 043 — the null closes (job 126995, 12/12 COMPLETED)

043 dropped `bias_lr` to $\{3\times10^{-4}, 10^{-3}\}$, $5\times$ and $17\times$
below the inherited bracket, on all three dims at 2 seeds. Median-seed F1:

| dims | `bias_lr` | F1 (both seeds) | median | vs floor |
|---|---|---|---|---|
| 24 | 3e-4 | 0.4548, 0.4578 | 0.4563 | −0.0059 |
| 24 | 1e-3 | 0.4347, 0.4560 | 0.4453 | −0.0169 |
| 48 | 3e-4 | 0.4494, 0.4551 | 0.4523 | −0.0099 |
| 96 | 3e-4 | 0.4594, 0.4660 | **0.4627** | **+0.0005** |
| 96 | 1e-3 | 0.4482, 0.4502 | 0.4492 | −0.0130 |
| **floor (041)** | — | 0.4610, 0.4633 | 0.4622 | — |

**Outcome 2 of the two predicted in §9.** F1 rises monotonically as
$\text{lr}\to 0$ and stops exactly at the floor. The best of twelve runs across
three dimensions and two learning rates is $+0.0005$ against a floor seed spread of
$0.0023$ — indistinguishable from having no bias at all. The same 96 dims spent on
`magnetic_linear` buy $+0.2203$.

A single seed of 96\@3e-4 briefly read $+0.0038$ over the floor; the second seed
came in below it and the median collapsed to parity. Recorded because it was
reported mid-flight as a possible falsification of §9's mechanism, and it was not
one.

**The EM artifact is confirmed as an artifact.** Teacher-forced EM peaked at 0.2933
(96 dims, `bias_lr` 2e-2) and is 0.2070–0.2116 here, against the floor's 0.2073. It
scaled with $\lvert\gamma\rvert$ and vanished with it. A capability would not.
`analyse.py`'s `dEM` column for landmark is now $-0.0064$ to $+0.0043$, i.e. zero.

**The residue is in the first pick.** Median hits@1 is *below* the floor in all
five populated cells (0.5273–0.5347 vs 0.5378), including the cell whose F1 reached
parity. As $\gamma\to 0$ the aggregate recovers before the retrieval decision does
— which is where §9's constant-query-factor mechanism says the damage lives. (One
individual seed, 96\@1e-3 seed 0, reads 0.5418; the effect is a consistent median
shift of $\sim\!-0.005$, not a clean per-run separation.)

**The dimension curve inverts its own diagnosis, consistently.** At high LR
landmark *anti-scales* (042: $0.3885 \to 0.3714 \to 0.3496$); at 3e-4 it is flat and
non-monotone ($0.4563 / 0.4523 / 0.4627$). Both are signatures of a channel with no
usable signal: when the bias is large, more dims mean more damage; when it is
negligible, its width cannot matter. Neither is a scaling law.

### The LOW-EDGE tag means the opposite thing here — do not read it the usual way

`analyse.py` tags all three landmark cells LOW-EDGE at 3e-4, and the standard
reading ("the optimum is outside the bracket, so the cell **under-reads** the arm")
is **wrong in this case**. The direction of improvement is toward *smaller
magnitude*, and $\lvert\gamma\rvert \sim \text{lr}\times\text{steps}$, so the limit
of extending the bracket downward is $\gamma = 0$ — which is the floor, already
measured. Extending it cannot produce a win; it can only reproduce 041 more
expensively.

The tag is still correct and still load-bearing for `magnetic_linear`, whose
improvement direction points at a bias that does something. **Read the tag together
with the sign of the trend**: LOW-EDGE toward a stronger bias is unfinished
measurement; LOW-EDGE toward a weaker one is a null converging.

### Verdict on WebQSP

Landmark is a **null**, and 042 + 043 separate the two candidate explanations
cleanly: 042 rules out unbounded magnitude (the normalization fixed that, and the
arm still lost), 043 rules out the mis-centred bracket (correcting it recovers the
floor and no more). What remains is §9's mechanism, unrefuted by its own
falsification test. **No tandem arm**: its premise was that landmark contributes
something `magnetic_linear` does not, and there is no metric on which that holds.
The 128-dim structural budget goes entirely to `magnetic_linear`.

## 12. Sweep 044 — the fairness control, and it changes the reading of 040 (8/8 COMPLETED)

043 extended landmark's LR search $17\times$ downward. Comparing the result against
a `magnetic_linear` measured only at $\{5\times10^{-3}, 2\times10^{-2}\}$ would give
one arm a 17× wider search than the other — `linear_bias` Conclusion 5's error, with
the asymmetry pointing at the incumbent. 044 runs `magnetic_linear` at 043's LRs.

| dims | `bias_lr` | F1 (both seeds) | median | vs 5e-3 |
|---|---|---|---|---|
| 24 | 1e-3 | 0.6048, 0.6138 | 0.6093 | −0.0109 |
| 48 | 1e-3 | 0.6326, 0.6394 | 0.6360 | −0.0205 |
| 96 | 1e-3 | 0.6408, 0.6549 | 0.6479 | −0.0346 |
| 96 | 3e-4 | 0.5725, 0.5761 | 0.5743 | **−0.1082** |

**`magnetic_linear` gets monotonically worse as the LR falls, and worse faster at
larger dims.** With 040's 2e-2 cells this brackets the 96-dim arm on both sides —
$0.5743 \to 0.6479 \to \mathbf{0.6825} \to 0.6610$ across $3\times10^{-4} \to
10^{-3} \to 5\times10^{-3} \to 2\times10^{-2}$ — so **5e-3 is an interior optimum**,
and `analyse.py` now drops the LOW-EDGE tag from all three `magnetic_linear` rows.

This retires a caveat carried since §8: every arm was edge-tagged, so the
incumbent's $+0.2203$ was itself reported as a lower bound. It is not a lower
bound; it is the measurement. **Add 044 to `analyse.py`'s `_DEFAULT_SWEEPS` or the
LR search stays asymmetric** — it is in the default list for that reason.

### The two arms respond to $\lvert\gamma\rvert$ with opposite signs

At `bias_lr` 3e-4, on 96 dims, in the same week on the same cluster:

| arm | @ 5e-3 | @ 3e-4 | $\Delta$ |
|---|---|---|---|
| `magnetic_linear` | 0.6825 | 0.5743 | **−0.1082** |
| landmark | 0.3496 | 0.4627 | **+0.1131** |

Shrinking the bias costs one arm 0.11 F1 and *gains* the other 0.11. This is the
campaign's single cleanest statement: one arm has signal to lose, the other has
damage to shed. It is not a tuning coincidence — it is §9's mechanism measured from
the other side, and it is why "landmark just needed a lower LR" is not an available
reading of §11.

## Conclusions

1. **The bias is not killed.** Under `degree`, no node has a dead coordinate row on
   either WebQSP build, at any $k$. This is the criterion that killed
   `factorized_rwpe` and it is not met here.
2. **`degree` is the anchor rule.** Free to compute, and it beats the
   mechanism-implied `betweenness` on 3 of 4 metrics. Carry `betweenness` as the
   documented near-tie if a training arm is cheap.
3. **`fps` and `mixed` are eliminated**, and `pagerank` with them. The failure has a
   mechanism (periphery / sinks), so it is not expected to reverse on another
   directed KG.
4. **The rule is load-bearing** — `random` loses by 45 pp of `exact_frac`.
5. **Coverage is the ceiling on WebQSP**: 5.57% of pairs are directed-reachable, so
   the channel is silent on the other 94%. This is the single biggest structural
   limit found, and it is a property of the Levi construction, not of the rule.
6. **`d_max = 8`, and the $k$-sweep runs on WebQSP only.**

### Training phase (040–045) — the verdict

7. **Landmark is a null on WebQSP, and the null is fully diagnosed.** Best of 12
   runs at the corrected LR is $+0.0005$ F1 over the no-bias floor (§11). The three
   candidate excuses are each eliminated by their own sweep: unbounded magnitude by
   042, a mis-centred bracket by 043, a favourably-tuned comparator by 044.
8. **It is worse than a null wherever it is actually active.** At any `bias_lr`
   large enough to give the bias real magnitude it lands 0.07–0.11 **below** the
   floor and anti-scales in dimension. Parity is reached only in the limit where
   $\gamma \to 0$. There is no operating point that beats switching it off, so it is
   not available as a cheap extra channel either.
9. **The mechanism is structural, not fixable by tuning.** The prompt node is
   isolated, so all landmark distances from it clamp to `d_max` and the query factor
   is *constant* exactly where the answer is generated — leaving a
   question-independent node-saliency prior that cannot retrieve (§9). Consistent
   with Conclusion 5's 5.6% directed reachability: the channel is silent on 94% of
   pairs and blank on the row that matters.
10. **The EM/F1 "divergence" was an evaluation artifact, not a capability.**
    `em_accuracy` is teacher-forced, `f1`/`hits1`/`hit_star` are generative (§9);
    they are not two views of one prediction. Landmark's EM advantage scaled with
    $\lvert\gamma\rvert$ and vanished with it (0.2933 → 0.2070 ≈ floor 0.2073).
11. **No tandem arm.** Its premise was that landmark contributes something
    `magnetic_linear` does not; `magnetic_linear` wins on F1, EM, hits@1 and hit\*
    simultaneously. The 128-dim structural budget goes entirely to
    `magnetic_linear`, whose curve is still climbing at $M=48$ and which
    `linear_bias` §3 puts on a plateau at $M=64$ ($=128$ head dims).
12. **The normalized form works and is worth keeping independently of this arm.**
    L2-normalized factors plus a per-head gain bound $\lvert b\rvert$ by
    Cauchy–Schwarz; 042-vs-040 measures $+0.079$ F1 where the bound binds
    (`bias_lr` 2e-2) and nothing where it is slack. This is a general remedy for
    factorized biases (MIXED_BIAS.md §5.8), not a landmark-specific fix, and it
    belongs with whatever family runs into runaway magnitude next.
13. **GraphQA is a different regime and the arm is *not* discarded there** (§10).
    Landmark beats the floor by large margins and scales monotonically in $k$
    (`node_degree` 0.086 → 0.886). It still loses to both incumbents at the honest
    cells ($k=4,8$), and the $k=16$ wins are label-adjacent. So the WebQSP null is
    about WebQSP's topology, not about the oracle being broken.

### Open — for the plan, not for this phase

**An undirected channel is now the obvious extension.** WebQSP is a single weak
component, so *undirected* reachability is 100% against directed reachability's
5.6%. A third channel on undirected anchor distances would cost $+k$ head dims and
raise coverage by ~18×, without displacing the two directed channels that carry
direction. `factorized_rwpe`'s Phase 0 reached the same place independently — *"the
one component with incremental value over degree is the undirected block, which is
not in the doc."* Two Phase 0s, two bias families, one conclusion about this
dataset. Not adopted here: it is a spec change, and the directed-only arm is the
one the current doc prices.
